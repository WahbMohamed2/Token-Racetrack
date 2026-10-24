"""Step 5: the fair race. Both lanes, 4 prompt lengths, warmup, 3 repeats, medians.
Prefill (step 0) is compute-bound and costs the same in both lanes. Decode is where they differ:
no-cache re-runs prefill every step (cost grows with length), cache stays flat (~18 ms of overhead).
Writes bench_results.json for the README in step 7.
"""
import json
import statistics
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
N_NEW = 100
PROMPT_LENS = [32, 256, 1024, 2048]
N_REPEATS = 3
sys.stdout.reconfigure(encoding="utf-8")

PARAGRAPH = (
    "The history of human communication is a story of innovation. From cave paintings and spoken "
    "language to writing systems, people have always built new ways to share ideas. The printing press "
    "spread books at a scale never seen before, and the telegraph, radio and television changed how we "
    "connect. The internet then let billions of people share messages, media and experiences in real time. "
)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()


# --- Prompt of an EXACT token length: repeat filler text, then cut the ids ---
def make_prompt(n_tokens):
    ids = tok(PARAGRAPH * (n_tokens // 50 + 1), return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(DEVICE)


# --- Both lanes in one function, so the ONLY difference is the if/else at the bottom (fair race).
# logits_to_keep=1: compute vocab scores for the last position only. Without it, no-cache built a
# ~650 MB logits tensor per step at 2k tokens, fragmented the allocator, and hit OOM ---
@torch.no_grad()
def run_lane(prompt_ids, n, use_cache):
    all_ids, feed, cache, step_times = prompt_ids, prompt_ids, None, []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=feed, past_key_values=cache, use_cache=use_cache, logits_to_keep=1)
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        all_ids = torch.cat([all_ids, next_id], dim=-1)
        if use_cache:
            cache, feed = out.past_key_values, next_id
        else:
            feed = all_ids
    return all_ids, step_times


# --- Per-run metrics: prefill = time to first token; decode median = cost of each extra token;
# first-10 vs last-10 decode shows whether the lane slows down as it goes ---
def summarize(step_times):
    decode = step_times[1:]
    return {
        "prefill_ms": step_times[0] * 1000,
        "decode_median_ms": statistics.median(decode) * 1000,
        "decode_first10_ms": statistics.median(decode[:10]) * 1000,
        "decode_last10_ms": statistics.median(decode[-10:]) * 1000,
        "total_s": sum(step_times),
    }


# --- The sweep: for each prompt length, warm up (absorbs one-time kernel selection + allocation),
# then 3 measured runs per lane, tracking peak VRAM ---
results = []
for L in PROMPT_LENS:
    prompt_ids = make_prompt(L)
    for use_cache in (False, True):
        run_lane(prompt_ids, 5, use_cache)  # warmup: CUDA context, cuBLAS kernel picks for these shapes

    runs = {False: [], True: []}
    for use_cache in (False, True):
        torch.cuda.reset_peak_memory_stats()
        for r in range(N_REPEATS):
            ids, times = run_lane(prompt_ids, N_NEW, use_cache)
            stats = summarize(times)
            stats["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            runs[use_cache].append((ids, stats))

    # --- Correctness + aggregation: did both lanes emit the same tokens (else: at which new token)?
    # Median across repeats is the headline number; spread (max - min) says how much to trust it ---
    a, b = runs[False][0][0], runs[True][0][0]
    diverge = None if torch.equal(a, b) else (a != b).nonzero()[0, 1].item() - L
    for use_cache in (False, True):
        stats = [s for _, s in runs[use_cache]]
        row = {k: statistics.median(s[k] for s in stats) for k in stats[0]}
        row["total_spread_s"] = max(s["total_s"] for s in stats) - min(s["total_s"] for s in stats)
        row.update(prompt_len=L, lane="cache" if use_cache else "no-cache", diverged_at=diverge)
        results.append(row)

# --- Output: table to the console, raw rows to bench_results.json ---
print(f"{'prompt':>6} {'lane':>8} {'prefill':>9} {'dec med':>8} {'dec 1-10':>9} {'dec 90-99':>10} "
      f"{'total':>8} {'spread':>7} {'peak':>8}  tokens")
for r in results:
    same = "identical" if r["diverged_at"] is None else f"split @ new token {r['diverged_at']}"
    print(f"{r['prompt_len']:>6} {r['lane']:>8} {r['prefill_ms']:>7.1f}ms {r['decode_median_ms']:>6.1f}ms "
          f"{r['decode_first10_ms']:>7.1f}ms {r['decode_last10_ms']:>8.1f}ms {r['total_s']:>7.2f}s "
          f"{r['total_spread_s']:>6.2f}s {r['peak_mem_gib']:>5.2f}GiB  {same}")

with open("bench_results.json", "w") as f:
    json.dump(results, f, indent=2)
