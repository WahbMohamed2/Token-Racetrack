"""Step 2: one forward pass = one next-token prediction.
The model scores every vocab entry at every position. Position i predicts token i+1, so only the
LAST position matters for "what comes after the prompt". A generation loop is this, repeated.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
sys.stdout.reconfigure(encoding="utf-8")

# --- Setup: same model + prompt as step 1 ---
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()
enc = tok("The KV cache speeds up generation because", return_tensors="pt").to(DEVICE)

# --- Forward pass: logits shape (batch=1, seq=7, vocab=151936), one score per vocab entry per position ---
with torch.no_grad():
    out = model(**enc)
print("logits:", tuple(out.logits.shape), out.logits.dtype)

# --- Keep only the last position (the prediction for token #8); fp32 so softmax keeps small probs ---
last = out.logits[0, -1].float()

# --- Greedy: take the highest score. Deterministic, which is what lets step 4 compare lanes exactly ---
greedy_id = last.argmax().item()
print(f"greedy: {greedy_id} -> {tok.decode(greedy_id)!r}")

# --- Softmax turns scores into probabilities; the top 5 show how unsure the model is ---
probs = torch.softmax(last, dim=-1)
top = torch.topk(probs, 5)
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {p:6.2%}  {i:>7} -> {tok.decode(i)!r}")


# --- Sampling: temperature sharpens (<1) or flattens (>1) the distribution; top-k cuts off the junk tail.
# torch.topk returns positions 0..k-1 inside the shortlist, so map back to real vocab ids with idx[pick] ---
def sample(logits, temperature=1.0, top_k=50):
    vals, idx = torch.topk(logits / temperature, top_k)
    pick = torch.multinomial(torch.softmax(vals, dim=-1), 1)
    return idx[pick].item()


for T in (0.2, 1.0, 2.0):
    print(f"T={T}:", [tok.decode(sample(last, T)) for _ in range(8)])
