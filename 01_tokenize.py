"""Step 1: text -> token ids -> text.
The model never sees text, only integer ids that index rows of its embedding table.
Everything later (logits, KV cache, latency) is measured per token, so this is the unit of the whole race.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252, which can't print 'Ġ'

# --- Load: tokenizer (text <-> ids) and model weights in bf16 on the GPU (~0.93 GiB) ---
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()

# --- Tokenize: prompt becomes input_ids of shape (batch=1, seq=7) plus an all-ones attention mask ---
prompt = "The KV cache speeds up generation because"
enc = tok(prompt, return_tensors="pt").to(DEVICE)

print("input_ids:     ", enc["input_ids"], tuple(enc["input_ids"].shape))
print("attention_mask:", enc["attention_mask"])

# --- Inspect: one id per word here; 'Ġ' marks "this token starts with a space" (byte-level BPE) ---
ids = enc["input_ids"][0]
for i, piece in zip(ids.tolist(), tok.convert_ids_to_tokens(ids)):
    print(f"  {i:>7} -> {piece!r}")

# --- Round trip: decode(encode(text)) must give back the exact text, or the tokenizer is lossy ---
decoded = tok.decode(ids)
assert decoded == prompt, f"round-trip mismatch: {decoded!r}"
print("decoded:", decoded)

# --- Sizes: tokenizer knows 151,665 tokens, embedding has 151,936 rows (padded for GPU-friendly shapes),
# so logits later will have 151,936 columns ---
print("vocab: tokenizer", len(tok), "| model embedding rows", model.config.vocab_size)
p = next(model.parameters())
print("weights:", p.dtype, p.device, f"| VRAM {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
