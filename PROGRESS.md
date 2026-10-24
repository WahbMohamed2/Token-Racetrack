# Progress

## Current position
- **Status:** ALL BLOCKS DONE (2026-10-24). App: `python 07_racetrack_app.py`.
- **Possible next steps (user's call):** CUDA graphs / torch.compile to break the ~18 ms decode floor;
  batching several prompts per lane; plotting KV-cache memory vs length; StaticCache vs DynamicCache.
- **Working mode (from 2026-10-24, user's request):** Claude writes the code itself, then walks
  through it line by line explaining what each line does. No TODO skeletons, no quiz rounds.
  Weak spots below stay logged but aren't re-quizzed unless the user asks.
- Verified API (transformers 5.20.2): out.past_key_values is a DynamicCache, len 24, .get_seq_length(),
  .layers[i].keys / .values with shape (batch, kv_heads=2, seq, head_dim=64), bf16. Cache is MUTATED
  IN PLACE (the returned object is the same one passed in). GQA: 14 query heads, 2 KV heads.
  => 24 * 2 (K,V) * 2 heads * 64 * 2 bytes = 12 KiB per token.

## Calibration (2026-10-23)
- Q1 logits shape / which position: correct ([1, 10, V], last position).
- Q2 what the KV cache stores and why reuse is safe: half. Got "K and V for each past
  token"; missed the *why* (causal mask) and why Q isn't cached. Re-test in Block 4.

## Book repo cross-reference (orca3/llm-model-inference, ch02), read 2026-10-24
- ch2_Workthrough_LLM_execution.ipynb: cell 6 = no-cache loop (our Block 3), cell 9 = cached loop
  (Block 4), cell 11 = per-token bar charts (Block 7), cell 14 = sliding context window (not in plan).
- Book setup: fp32 (no dtype), Colab T4, multinomial sampling at T=1 (no greedy, so lanes can't be
  compared token-for-token), prompt = 188 words / 225 tokens, 100 new tokens.
- Book timing: per-step (start_time reset each iteration), NOT cumulative as CLAUDE.md's tutor note
  claims, at least in this repo version. Real flaws: no cuda.synchronize in the no-cache loop;
  print+decode inside the timed region; no warmup. Cell 9 loads the model twice and passes
  generate()-only kwargs (max_new_tokens/min_new_tokens) to forward().
- Streaming/Batching/vLLM notebooks use vLLM (Linux; on Windows needs WSL2). Our Block 6 does
  streaming by hand with generators instead.

## Completed blocks
- **Block 0** (2026-10-24): venv on py3.10, torch 2.14.1+cu130, transformers 5.20.2,
  gradio 6.30.0, pinned in requirements.txt. 00_check_gpu.py confirms CUDA 13.0 build, RTX 4060 visible,
  GPU matmul works. Note: Qwen2.5-0.5B not in the HF cache yet, downloads (~1 GB) on first load.
  In transformers 5.x use `dtype=`, not `torch_dtype=` (deprecated).
- **Block 1** (2026-10-24): user asked for the solution; written + explained. Result: prompt
  "The KV cache speeds up generation because" -> 7 tokens, shape (1, 7), mask all 1s, round-trip OK.
  Tokenizer len 151665 vs embedding rows 151936 (padded). bf16 weights = 0.93 GiB VRAM.
  Windows gotcha: need sys.stdout.reconfigure(encoding="utf-8") to print 'Ġ'.
- **Block 2** (2026-10-24): user asked for the solution; written + explained. logits (1, 7, 151936)
  bf16. Greedy next token ' it' (44.4%), then ' the' 18.5%, ' of' 5.3%. Sampling T=0.2 mostly ' it',
  T=2.0 diverse. sample() = topk on logits/T -> softmax -> multinomial -> map back via indices.
- **Block 3** (2026-10-24): user asked for the solution; written + explained. Timing per step with
  synchronize() on both sides of the clock, list of per-step deltas (no cumulative bug).
- **Block 4** (2026-10-24): Claude wrote it (user switched to "write it and explain" mode). Cached loop
  matches no-cache loop on all 100 greedy tokens (torch.equal). Cache after 100 steps: seq_len 106
  (7 prompt + 99 fed tokens; the 100th token is produced but never fed).
- **Block 5** (2026-10-24): Claude wrote `05_benchmark.py` (sweep 32/256/1024/2048-token prompts, warmup,
  3 repeats, medians, prefill vs decode split, peak memory, saves bench_results.json). First run hit OOM:
  no-cache forward materialized logits for ALL positions (~650 MB at 2148 tokens, a new size every step ->
  caching-allocator fragmentation -> OOM retry + 9 s spread). Fixed with logits_to_keep=1 in both lanes.
- **Block 6** (2026-10-24): `lanes.py` (stream_lane generator yielding (delta_text, latency_ms) with
  incremental detokenization that holds back '\ufffd'; race() schedules lanes by their own GPU clocks;
  warmup()) + `06_stream_demo.py`. 1024-token prompt, 60 tokens: TTFT 39 vs 40 ms, total 2.59 vs 1.24 s.
  Cancel demo: GEN_SUSPENDED after break, GEN_CLOSED after close(). Race: cache done at 1.21 s, no-cache at 29/60.
- **Block 7** (2026-10-24): `07_racetrack_app.py` (Gradio 6.30, generator handler, Stop cancels the event,
  LinePlot per-token latency) + README.md. Verified headless via gradio_client: 2048 prompt, 100 tokens,
  no-cache 8.03 s vs cache 2.02 s (4.0x), no-cache at token 26 when cache finished. Browser rendering
  not eyeballed by Claude. pandas pinned in requirements.txt.

## Gate-check results
(none yet)

## Weak spots
- **Why the KV cache is safe to reuse**
  - *Missed:* 2026-10-23 — didn't explain why past K/V never change (causal masking), or why Q isn't cached
  - *Missed:* 2026-10-23 (session 2 re-quiz) — Q part roughly right; K/V part circular
    ("already computed, so no reason to be wrong"), causal mask still not named.
    Re-explained via layer-by-layer dependency induction; follow-up asked (bidirectional model).
  - *Hit:* 2026-10-23 — bidirectional follow-up: correctly said the cache goes stale because
    every position sees token 50. Precision miss: said "all layers" — layer 1 K/V survive
    (pure projections of each token's own embedding, no mixing yet). Re-check that detail in Block 4.
  - *Correct streak:* 1

- **Model size as an experimental design choice (too big AND too small both fail)**
  - *Missed:* 2026-10-23 — Block 0 gate Q1: gave only convenience reasons (download, iteration
    speed); missed VRAM headroom vs ~6.9 GiB free and that a too-tiny model is overhead-bound
    so the no-cache slowdown wouldn't be visible.
  - *Hit:* 2026-10-23 — re-test (10M model curves): said both lanes flat because overhead
    dominates. Correct, though right after the re-explanation; re-check cold next session.
  - *Correct streak:* 1

## Measured numbers
- Block 3, no cache, 7-token prompt, 100 tokens: total 2.67 s. Step 0 = 578 ms (cold start:
  CUDA/cuBLAS init, no warmup). Steps 1-99 FLAT at ~19-20 ms, with spikes to 37 ms (noise).
- Single no-cache forward vs seq len (warm, median of 5): 8: 25 ms, 64: 24, 128: 26, 256: 29,
  512: 42, 1024: 81, 2048: 163, 4096: 272. Flat below ~256 tokens (launch/overhead-bound),
  roughly linear above. => Block 5 needs a prompt of 1000+ tokens to show the no-cache slowdown.
- Block 4, cached, same 7-token prompt: prefill (cold) 335 ms, decode median 18.5 ms, total 2.26 s.
  No visible win over no-cache (~19 ms/step) at seq <= 106: both lanes sit on the ~19 ms overhead floor.
- Block 5 (logits_to_keep=1, warm, median of 3, 100 new tokens):
  | prompt | no-cache total | cache total | no-cache decode/step | cache decode/step | prefill |
  | 32     | 1.88 s | 1.81 s | 18.4 ms | 18.0 ms | ~19 ms |
  | 256    | 1.88 s | 1.86 s | 18.8 ms | 18.0 ms | ~20 ms |
  | 1024   | 4.24 s | 1.88 s | 42.4 ms | 18.1 ms | ~38 ms |
  | 2048   | 8.19 s | 1.89 s | 81.8 ms | 18.1 ms | ~77 ms |
  Spread across repeats <= 0.14 s. Peak VRAM ~0.94-1.03 GiB. Cache lane flat at ~18 ms regardless of length.
  Tokens identical except 32-token prompt: split at new token 12 because of an EXACT bf16 tie
  (' revolution' vs ' made', both 19.125 in the no-cache lane; bf16 spacing at 16-32 is 0.125);
  max logit diff between lanes 0.156. Numerics, not a bug.
- Book-like config on the 4060 (fp32, all-position logits, 225-token prompt, 100 tokens, warm):
  no-cache 5.28 s vs cache 1.90 s (2.8x), matching the book's 9.12 vs 3.14 (2.9x) ratio.
- Greedy output of the base model falls into a repetition loop (Chinese quiz text). Expected for a
  base, non-instruct model under greedy decoding.
