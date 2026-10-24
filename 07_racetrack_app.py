"""Step 7: the racetrack. Gradio app, two lanes side by side, tokens streaming live, per-token latency chart.
Same prompt, same greedy decoding, same model; the ONLY difference between lanes is the KV cache.
Lanes take turns on one GPU, scheduled by their own clocks (see lanes.race), so what you see is each
lane's true speed. Run:  python 07_racetrack_app.py  -> opens http://127.0.0.1:7860
"""
import statistics
import pandas as pd
import gradio as gr
from lanes import make_prompt, race, warmup

LANES = ("no-cache", "cache")
COLORS = {"no-cache": "#d1495b", "cache": "#2a9d8f"}


# --- Per-lane scoreboard: token 0 is the prefill (whole prompt), the rest are decode steps ---
def scoreboard(s):
    if not s["lat"]:
        return "waiting..."
    decode = s["lat"][1:]
    dec = f"{statistics.median(decode):.1f} ms" if decode else "-"
    return (f"**{len(s['lat'])}** tokens | prefill **{s['lat'][0]:.0f} ms** | "
            f"decode median **{dec}** | GPU time **{s['clock'] / 1000:.2f} s**")


# --- The race handler is a generator: every `yield` pushes fresh text, scoreboards and chart to the
# browser, so the UI streams instead of waiting for the end. Stop button cancels this generator ---
def run_race(prompt_tokens, new_tokens):
    prompt_ids = make_prompt(int(prompt_tokens))
    state = {lane: {"text": "", "lat": [], "clock": 0.0} for lane in LANES}
    rows, status = [], f"Racing: {int(prompt_tokens)}-token prompt, {int(new_tokens)} new tokens per lane."

    for lane, delta, latency_ms, clock_ms in race(prompt_ids, int(new_tokens)):
        s = state[lane]
        s["text"] += delta
        s["lat"].append(latency_ms)
        s["clock"] = clock_ms
        rows.append({"token": len(s["lat"]) - 1, "latency_ms": latency_ms, "lane": lane})

        if lane == "cache" and len(s["lat"]) == int(new_tokens):
            behind = len(state["no-cache"]["lat"])
            status = (f"Cache lane crossed the line at **{clock_ms / 1000:.2f} s**. "
                      f"No-cache lane was at token **{behind}/{int(new_tokens)}**.")

        yield (status,
               scoreboard(state["no-cache"]), state["no-cache"]["text"],
               scoreboard(state["cache"]), state["cache"]["text"],
               pd.DataFrame(rows))

    nc, c = state["no-cache"]["clock"], state["cache"]["clock"]
    status += f" Final: no-cache {nc / 1000:.2f} s vs cache {c / 1000:.2f} s ({nc / c:.1f}x)."
    yield (status,
           scoreboard(state["no-cache"]), state["no-cache"]["text"],
           scoreboard(state["cache"]), state["cache"]["text"],
           pd.DataFrame(rows))


# --- Layout: controls on top, two lanes side by side, shared latency chart below ---
with gr.Blocks(title="Token Racetrack") as demo:
    gr.Markdown("# Token Racetrack\nQwen2.5-0.5B, greedy, bf16. Same prompt in both lanes. "
                "Left re-reads the whole sequence every token; right reuses the KV cache.")
    with gr.Row():
        prompt_tokens = gr.Slider(32, 2048, value=2048, step=32, label="Prompt length (tokens)")
        new_tokens = gr.Slider(10, 200, value=100, step=10, label="New tokens per lane")
    with gr.Row():
        start = gr.Button("Start race", variant="primary")
        stop = gr.Button("Stop")
    status = gr.Markdown("Long prompts make the gap visible: below ~256 tokens both lanes sit on the "
                         "same ~18 ms per-step overhead floor.")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Lane 1: no KV cache")
            nc_score = gr.Markdown("waiting...")
            nc_text = gr.Textbox(lines=10, max_lines=10, interactive=False, autoscroll=True, show_label=False)
        with gr.Column():
            gr.Markdown("### Lane 2: KV cache")
            c_score = gr.Markdown("waiting...")
            c_text = gr.Textbox(lines=10, max_lines=10, interactive=False, autoscroll=True, show_label=False)
    chart = gr.LinePlot(x="token", y="latency_ms", color="lane", color_map=COLORS,
                        title="Per-token latency (token 0 = prefill)", x_title="token #",
                        y_title="ms", height=320)

    race_event = start.click(run_race, [prompt_tokens, new_tokens],
                             [status, nc_score, nc_text, c_score, c_text, chart])
    stop.click(None, cancels=[race_event])

if __name__ == "__main__":
    print("warming up kernels for 256/1024/2048-token prompts...")
    warmup()
    demo.queue().launch(inbrowser=True)
