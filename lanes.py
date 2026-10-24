"""Step 6: the two lanes as STREAMS (importable module, used by 06_stream_demo.py and 07_racetrack_app.py).
Same loop as 05_benchmark.run_lane, but instead of returning everything at the end it yields each token
the moment it exists. The caller sees the first token after one prefill (perceived latency), even though
the full answer still takes the same total time. Stopping the caller's loop stops generation (cancel).
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
FILLER = (
    "The history of human communication is a story of innovation. From cave paintings and spoken "
    "language to writing systems, people have always built new ways to share ideas. The printing press "
    "spread books at a scale never seen before, and the telegraph, radio and television changed how we "
    "connect. The internet then let billions of people share messages, media and experiences in real time. "
)

# --- Load once at import: every script that imports lanes shares one model on the GPU ---
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()


# --- Prompt of an EXACT token length (same trick as step 5) ---
def make_prompt(n_tokens):
    ids = tok(FILLER * (n_tokens // 50 + 1), return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(DEVICE)


# --- One lane as a generator: each `yield` hands back (new_text, latency_ms) for one token.
# Text is decoded from ALL generated ids and only the new tail is yielded, because one token can be
# half of a multi-byte character (Chinese, emoji): decoding it alone gives U+FFFD, so we hold it back.
# Cancel point: if the consumer stops iterating, Python closes the generator at the `yield` and the
# loop (and its KV cache) is simply dropped. A server's abort() plugs in exactly here ---
@torch.no_grad()
def stream_lane(prompt_ids, max_new_tokens, use_cache):
    all_ids, feed, cache = prompt_ids, prompt_ids, None
    gen_ids, shown = [], ""
    for _ in range(max_new_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=feed, past_key_values=cache, use_cache=use_cache, logits_to_keep=1)
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        all_ids = torch.cat([all_ids, next_id], dim=-1)
        if use_cache:
            cache, feed = out.past_key_values, next_id
        else:
            feed = all_ids

        gen_ids.append(next_id.item())
        text = tok.decode(gen_ids, skip_special_tokens=True)
        delta = "" if text.endswith("\ufffd") else text[len(shown):]
        shown += delta
        yield delta, latency_ms
        if gen_ids[-1] == tok.eos_token_id:
            return


# --- The race: one GPU, two lanes taking turns. Each lane has its own clock (sum of its step
# latencies); the lane that has used LESS GPU time always goes next. So a lane's progress at any moment
# is what it would show if it had the GPU to itself: the cached lane pulls ahead at its true speed ---
def race(prompt_ids, max_new_tokens):
    lanes = {
        "no-cache": stream_lane(prompt_ids, max_new_tokens, use_cache=False),
        "cache": stream_lane(prompt_ids, max_new_tokens, use_cache=True),
    }
    clock_ms = {name: 0.0 for name in lanes}
    while lanes:
        name = min(lanes, key=clock_ms.get)
        try:
            delta, latency_ms = next(lanes[name])
        except StopIteration:
            del lanes[name]
            continue
        clock_ms[name] += latency_ms
        yield name, delta, latency_ms, clock_ms[name]


# --- Warmup: run a few tokens per lane at each prompt length, so cuBLAS kernel selection and memory
# allocation happen here instead of inside the first measured token ---
def warmup(prompt_lens=(256, 1024, 2048), n=3):
    for L in prompt_lens:
        ids = make_prompt(L)
        for use_cache in (False, True):
            for _ in stream_lane(ids, n, use_cache):
                pass
