"""Step 6 demo: streaming in the terminal.
1) Each lane streams its tokens live: time-to-first-token (what the user FEELS) vs total time (what it COSTS).
2) Cancel: stop reading a stream after a few tokens and the generator is shut down.
3) The race on virtual clocks: where was the no-cache lane when the cached lane crossed the line?
"""
import inspect
import sys
import time
from lanes import make_prompt, race, stream_lane, warmup

PROMPT_LEN = 1024
N_NEW = 60
sys.stdout.reconfigure(encoding="utf-8")

print("warming up...")
warmup(prompt_lens=(PROMPT_LEN,))
prompt_ids = make_prompt(PROMPT_LEN)

# --- 1) Stream each lane. TTFT = wall time until the first token is on screen (prefill dominates it,
# so both lanes feel equally responsive at first). Total = until the last token ---
for use_cache in (False, True):
    name = "cache" if use_cache else "no-cache"
    print(f"\n=== {name} lane, {PROMPT_LEN}-token prompt ===")
    t_start = time.perf_counter()
    ttft = None
    for delta, latency_ms in stream_lane(prompt_ids, N_NEW, use_cache):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        print(delta, end="", flush=True)
    total = time.perf_counter() - t_start
    print(f"\n[{name}] time to first token {ttft * 1000:.0f} ms | total {total:.2f} s for {N_NEW} tokens")

# --- 2) Cancel: break out after 5 tokens, then close() the generator. Its state becomes CLOSED and the
# rest of the tokens are never computed. This is the hook a server uses when a client disconnects ---
print("\n=== cancel after 5 tokens ===")
stream = stream_lane(prompt_ids, N_NEW, use_cache=False)
for i, (delta, _) in enumerate(stream):
    print(delta, end="", flush=True)
    if i == 4:
        break
print(f"\nstate after break: {inspect.getgeneratorstate(stream)}")
stream.close()
print(f"state after close: {inspect.getgeneratorstate(stream)}")

# --- 3) The race: count tokens per lane on their own clocks; freeze the score when the cache lane finishes ---
print("\n=== race ===")
count = {"no-cache": 0, "cache": 0}
for lane, _, _, clock_ms in race(prompt_ids, N_NEW):
    count[lane] += 1
    if lane == "cache" and count["cache"] == N_NEW:
        print(f"cache lane finished {N_NEW} tokens at {clock_ms / 1000:.2f} s of GPU time; "
              f"no-cache lane was at token {count['no-cache']}")
print(f"final: {count}")
