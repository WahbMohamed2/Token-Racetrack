"""Step 0: is the pit lane ready?
Confirms the venv has a CUDA build of torch, the 4060 is visible, and the GPU can actually do math.
If this fails, nothing else in the project will run on the GPU.
"""
import sys
import torch
import transformers
import gradio

# --- Versions: what's installed in this venv (pinned in requirements.txt) ---
print(f"python:       {sys.executable}")
print(f"torch:        {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"gradio:       {gradio.__version__}")

# --- CUDA: torch must be a CUDA build (cu130) AND see a GPU; a CPU-only wheel prints None / False ---
print(f"cuda (build): {torch.version.cuda}")
print(f"cuda avail:   {torch.cuda.is_available()}")

props = torch.cuda.get_device_properties(0)
print(f"gpu:          {props.name}, {props.total_memory / 1024**3:.2f} GiB")

# --- Smoke test: one real matmul on the GPU. synchronize() waits for it to finish,
# because CUDA launches are async and an error would otherwise surface later, somewhere confusing ---
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = a @ b
torch.cuda.synchronize()
print(f"matmul:       {c.device}, {c.dtype}, {tuple(c.shape)}")
