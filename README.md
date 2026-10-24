# Token Racetrack

A side-by-side visualization of autoregressive text generation with and without a KV cache.

The same prompt is streamed through Qwen/Qwen2.5-0.5B in two lanes:

- **No-cache lane:** re-feeds the entire sequence through the model at every step.
- **Cache lane:** runs the prompt once (prefill), then feeds only the newest token together with the
  cached keys and values.

Tokens stream into each lane as they are generated. A live chart plots per-token latency for both
lanes. With long prompts, the no-cache lane falls behind while the cache lane holds a flat per-token
cost.

Built as a study project for Chapter 2 of *Hands-On LLM Serving and Optimization* (O'Reilly).
It reproduces the chapter's KV cache experiment with a controlled benchmark.

## Results

Measured on an NVIDIA GeForce RTX 4060 (8 GB): bf16 weights, greedy decoding, 100 new tokens.
Each configuration was warmed up first and then run 3 times; the table shows medians. Every step is
timed between `torch.cuda.synchronize()` calls.

| Prompt length | No cache, total | KV cache, total | No cache, per decode step | KV cache, per decode step | Speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 1.88 s | 1.81 s | 18.4 ms | 18.0 ms | 1.0x |
| 256 | 1.88 s | 1.86 s | 18.8 ms | 18.0 ms | 1.0x |
| 1024 | 4.24 s | 1.88 s | 42.4 ms | 18.1 ms | 2.3x |
| 2048 | 8.19 s | 1.89 s | 81.8 ms | 18.1 ms | 4.3x |

Across the 3 repeats, the total time of any configuration varied by at most 0.14 s. Peak VRAM was
about 1.0 GiB in every configuration. Raw data is in `bench_results.json`.

### Comparison with the book

| Setup | No cache | KV cache | Ratio |
|---|---:|---:|---:|
| Book: Colab T4, fp32, logits for all positions, 225-token prompt, sampling | 9.12 s | 3.14 s | 2.9x |
| This repo, book-like config: RTX 4060, fp32, logits for all positions, 225-token prompt, greedy | 5.28 s | 1.90 s | 2.8x |
| This repo, optimized: bf16, last-position logits only, 256-token prompt | 1.88 s | 1.86 s | 1.0x |
| This repo, optimized: bf16, last-position logits only, 2048-token prompt | 8.19 s | 1.89 s | 4.3x |

With the book's configuration, the RTX 4060 reproduces the book's ratio. With bf16 weights and
logits computed only for the last position, a no-cache step at a few hundred tokens costs no more
than the fixed per-step overhead. In that setup, the cache's advantage only appears with longer prompts.

## Key findings

- **Prefill and decode are different workloads.** Prefill processes the whole prompt in large matrix
  multiplications. It is compute-bound and scales with prompt length (about 77 ms at 2048 tokens).
  Decode processes one token per step. At batch size 1, it is limited by kernel-launch overhead and
  memory traffic, not by arithmetic.
- **The cached decode step costs about 18 ms at every prompt length.** That cost is fixed per-step
  overhead: hundreds of small kernel launches across 24 layers, plus Python. Streaming the roughly
  1 GB of weights once at the GPU's ~272 GB/s would take about 4 ms. CUDA graphs and batching are the
  standard ways to reduce this overhead.
- **Without a cache, every decode step repeats the prefill.** Once the sequence is longer than about
  256 tokens, per-step cost grows with sequence length: 42 ms at 1k tokens, 82 ms at 2k tokens.
- **The KV cache is small at this scale.** Qwen2.5-0.5B uses grouped-query attention, with 2 key/value
  heads shared by 14 query heads. The cache therefore costs 12 KiB per token
  (24 layers x 2 tensors x 2 heads x 64 dims x 2 bytes), about 25 MiB at 2,148 tokens.
- **The cache is exact up to floating-point rounding.** Greedy outputs from both lanes matched
  token-for-token in every configuration but one. On the 32-token prompt, the lanes diverged at new
  token 12, where two candidates were exactly tied in bf16 (both 19.125). The cached path sums in a
  different order and broke the tie the other way. This is expected numerical behavior, not a
  correctness issue.
- **Streaming improves perceived latency, not total latency.** Both lanes show their first token after
  one prefill (about 40 ms with a 1024-token prompt). Total time differs: 2.59 s against 1.24 s for
  60 tokens.

## Benchmarking pitfalls addressed

| Pitfall | Effect | Fix |
|---|---|---|
| No GPU synchronization | CUDA kernels launch asynchronously, so the timer measures launch time, not compute | `torch.cuda.synchronize()` before both clock reads |
| No warmup | The first step paid one-time setup costs (335 to 578 ms) | Throwaway runs at each prompt length before measuring |
| Logits for all positions in the no-cache lane | About 650 MB of logits per step at 2k tokens, a different size every step. This fragmented the CUDA caching allocator and caused OOM retries and a 9 s spread between runs | `logits_to_keep=1` in both lanes |
| Single run | No way to tell signal from noise | 3 repeats, medians, max-min spread reported |
| I/O inside the timed region | `print` and `decode` time counted as model time | Only the forward pass and token selection are timed |

## How the race is scheduled

Both lanes share one GPU and take turns. Each lane keeps its own clock: the sum of its measured step
latencies. The lane that has used less GPU time always runs next. As a result, each lane's position on
screen matches what it would reach with the GPU to itself, and the lanes never compete for the GPU at
the same time. With a 2048-token prompt, the cache lane finishes 100 tokens in about 2.0 s, by which
point the no-cache lane has produced about 26.

## Project structure

The numbered scripts build on one another and can be read in order.

| File | Purpose |
|---|---|
| `00_check_gpu.py` | Verifies the CUDA build of PyTorch, GPU visibility, and a GPU matmul |
| `01_tokenize.py` | Tokenization, the attention mask, round-trip decoding, vocabulary vs embedding size |
| `02_next_token.py` | A single forward pass, logits shape, greedy decoding, temperature and top-k sampling |
| `03_loop_no_cache.py` | Autoregressive loop that re-feeds the full sequence every step |
| `04_loop_kv_cache.py` | Cached loop (prefill, then single-token decode), cache inspection, token-identity check |
| `05_benchmark.py` | Controlled benchmark across prompt lengths; writes `bench_results.json` |
| `lanes.py` | Both lanes as streaming generators, the turn-based race scheduler, and warmup |
| `06_stream_demo.py` | Terminal demo: time to first token vs total time, cancellation, and the race |
| `07_racetrack_app.py` | The Gradio application |

## Setup

Requirements: an NVIDIA GPU with a recent driver (CUDA 13.0 support), and Python 3.10.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 13.0 build of PyTorch through the PyTorch package index. The model
weights (about 1 GB) download from the Hugging Face Hub on first run.

## Usage

Run the application:

```
python 07_racetrack_app.py
```

The app warms up its GPU kernels, then opens `http://127.0.0.1:7860`. Choose a prompt length and the
number of new tokens per lane, then start the race. The default 2048-token prompt makes the gap clearly
visible. At 256 tokens or fewer, both lanes run at nearly the same speed, for the reason given in
Key findings.

Reproduce the benchmark:

```
python 05_benchmark.py
```

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB |
| Python | 3.10 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.20.2 |
| Gradio | 6.30.0 |
| Model | Qwen/Qwen2.5-0.5B (base model, bf16) |

## Reference

- *Hands-On LLM Serving and Optimization*, O'Reilly, Chapter 2.
- Companion code: https://github.com/orca3/llm-model-inference/tree/main/ch02
