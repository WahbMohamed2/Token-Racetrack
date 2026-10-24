"""Step 4: the FAST lane. Same loop, with a KV cache.
Causal masking means a token's keys/values never change once computed, so we keep them.
Step 0 (prefill) runs the whole prompt and fills the cache; every later step (decode) feeds ONLY the
new token, whose query attends to all cached K/V. Then we prove both lanes emit identical greedy tokens.
"""
import statistics
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
N_NEW = 100
sys.stdout.reconfigure(encoding="utf-8")

# --- Setup: same model + 7-token prompt as step 3 ---
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()
prompt_ids = tok("The KV cache speeds up generation because", return_tensors="pt").input_ids.to(DEVICE)


# --- Reference lane: step 3's loop as a function (timing stripped), only used to check correctness ---
@torch.no_grad()
def generate_nocache(input_ids, n):
    for _ in range(n):
        next_id = model(input_ids=input_ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
    return input_ids


# --- Cached lane: `feed` is what enters the model (whole prompt, then 1 token), `cache` carries K/V
# between steps, `all_ids` is just the transcript. Cache shape per layer: (batch, kv_heads=2, seq, 64) ---
@torch.no_grad()
def generate_cached(input_ids, n):
    all_ids, step_times = input_ids, []
    feed, cache = input_ids, None  # step 0 feeds the whole prompt (prefill), later steps feed 1 token
    for step in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=feed, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if step == 0:
            k, v = cache.layers[0].keys, cache.layers[0].values
            print(f"cache: {type(cache).__name__}, {len(cache)} layers, layer0 K {tuple(k.shape)} V {tuple(v.shape)} {k.dtype}")
        feed = next_id
        all_ids = torch.cat([all_ids, next_id], dim=-1)
    print(f"cache after {n} steps: seq_len {cache.get_seq_length()}, layer0 K {tuple(cache.layers[0].keys.shape)}")
    return all_ids, step_times


# --- Proof: the cache is only an optimization if it changes NOTHING about the output.
# If the lanes ever split, show where (usually a bf16 near-tie, see step 5) ---
cached_ids, times = generate_cached(prompt_ids, N_NEW)
nocache_ids = generate_nocache(prompt_ids, N_NEW)
if torch.equal(cached_ids, nocache_ids):
    print(f"IDENTICAL: all {N_NEW} greedy tokens match")
else:
    first = (cached_ids != nocache_ids).nonzero()[0, 1].item()
    print(f"DIVERGED at position {first} (new token #{first - prompt_ids.shape[1]})")
    print("  cached:  ", repr(tok.decode(cached_ids[0, first:first + 10])))
    print("  no-cache:", repr(tok.decode(nocache_ids[0, first:first + 10])))

# --- Report: prefill (step 0) vs decode (steps 1+) are different kinds of work, so report them apart ---
print(tok.decode(cached_ids[0]))
print(f"step 0 (prefill): {times[0] * 1000:.1f} ms")
print(f"steps 1-{N_NEW - 1} (decode) median: {statistics.median(times[1:]) * 1000:.1f} ms")
print(f"total: {sum(times):.2f} s for {N_NEW} tokens")
