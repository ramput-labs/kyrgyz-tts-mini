#!/usr/bin/env python
"""Tiny web UI: type Kyrgyz text, pick a voice, listen and download the WAV.

make demo            # or: python scripts/gradio_demo.py --share
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np

from kyrgyz_tts import config
from kyrgyz_tts.engine import get_tts
from kyrgyz_tts.text import dropped_characters

EXAMPLES = Path(__file__).resolve().parents[1] / "samples" / "texts.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    args = ap.parse_args()

    tts = get_tts(args.device)
    voices = list(config.VOICES)
    for voice in voices:  # load everything up front so the first click is fast
        tts.warm_up(voice)

    def run(text, voice, rate, temperature, steps, denoise):
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if not lines:
            raise gr.Error("Текст жазыңыз · Enter some Kyrgyz text.")
        try:
            speeches = [
                tts.synthesize(
                    line,
                    voice,
                    temperature=float(temperature),
                    speaking_rate=float(rate),
                    steps=int(steps),
                    denoiser_strength=float(denoise),
                )
                for line in lines
            ]
        except ValueError as e:
            raise gr.Error(str(e)) from None
        sr = speeches[0].sample_rate
        pause = np.zeros(int(0.3 * sr), dtype=np.float32)
        audio = np.concatenate([part for s in speeches for part in (s.audio, pause)][:-1])
        elapsed = sum(s.seconds for s in speeches)
        info = f"{len(audio) / sr:.1f}s of speech in {elapsed:.2f}s on {tts.device}"
        if skipped := dropped_characters(text).strip():
            info += f"\nskipped (no pronunciation): {skipped}"
        return (sr, audio), info

    examples = []
    if EXAMPLES.exists():
        examples = [[line.strip()] for line in EXAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]

    with gr.Blocks(title="kyrgyz-tts") as ui:
        gr.Markdown("## kyrgyz-tts · Кыргызча текстти үнгө айландыруу")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text (each line is spoken with a short pause)",
                    lines=6,
                    value="Саламатсызбы! Бүгүн аба ырайы абдан жакшы.",
                )
                voice = gr.Radio(voices, value="woman", label="Voice")
                with gr.Accordion("Settings", open=False):
                    rate = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Rate (length scale, higher is slower)")
                    temperature = gr.Slider(0.0, 1.5, value=0.667, step=0.01, label="Temperature (0 = deterministic)")
                    steps = gr.Slider(2, 50, value=10, step=1, label="ODE steps")
                    denoise = gr.Slider(0.0, 0.01, value=0.00025, step=0.00005, label="Denoiser strength")
                btn = gr.Button("Speak", variant="primary")
            with gr.Column():
                out = gr.Audio(label="Speech", type="numpy", format="wav", autoplay=True)
                info = gr.Textbox(label="Info", lines=2)
        if examples:
            gr.Examples(examples, inputs=[text])
        inputs = [text, voice, rate, temperature, steps, denoise]
        btn.click(run, inputs, [out, info])
        text.submit(run, inputs, [out, info])

    ui.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
