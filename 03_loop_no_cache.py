"""Step 3: the SLOW lane. Autoregressive generation without a KV cache.
Each step feeds the ENTIRE sequence so far back through all 24 layers, recomputing keys/values the
model already computed last step. Work per step grows with length; at short lengths (< ~256 tokens)
the ~19 ms fixed per-step overhead hides that growth, which is why step 5 uses long prompts.
"""
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
N_NEW = 100
sys.stdout.reconfigure(encoding="utf-8")

# --- Setup: model + a short 7-token prompt ---
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()
input_ids = tok("The KV cache speeds up generation because", return_tensors="pt").input_ids.to(DEVICE)

# --- The loop: forward(full sequence) -> pick next token -> append -> repeat.
# synchronize() before BOTH clock reads: CUDA is async, without it you time kernel launches, not compute.
# Each entry in step_times is THIS step's duration, not a running total ---
step_times = []
with torch.no_grad():
    for step in range(N_NEW):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=False)
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        input_ids = torch.cat([input_ids, next_id], dim=-1)

# --- Report: step 0 is slow from cold start (no warmup yet); the rest sit flat at ~19 ms here ---
print(tok.decode(input_ids[0]))
for step in [0, 1, 2, *range(10, N_NEW, 10), N_NEW - 1]:
    print(f"step {step:>3}  seq_len {input_ids.shape[1] - N_NEW + step:>3}  {step_times[step] * 1000:6.1f} ms")
print(f"total: {sum(step_times):.2f} s for {N_NEW} tokens")
