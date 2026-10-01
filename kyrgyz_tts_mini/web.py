import gradio as gr

from kyrgyz_tts_mini import config
from kyrgyz_tts_mini.config import Settings
from kyrgyz_tts_mini.engine import get_tts
from kyrgyz_tts_mini.text import dropped_characters


def launch(args) -> None:
    tts = get_tts(args.device)
    for voice in config.VOICES:
        tts.warm_up(voice)

    def speak(text, voice, rate, temperature, steps, denoise):
        try:
            settings = Settings(temperature=temperature, rate=rate, steps=int(steps), denoise=denoise)
            speech = tts.synthesize_lines((text or "").splitlines(), voice, settings)
        except ValueError as e:
            raise gr.Error(str(e)) from None
        info = f"{speech.duration:.1f}s of speech in {speech.elapsed:.2f}s on {tts.device}"
        if skipped := dropped_characters(text).strip():
            info += f"\nskipped (no pronunciation): {skipped}"
        return (speech.sample_rate, speech.audio), info

    examples = []
    if config.SAMPLES.exists():
        examples = [[line] for line in config.SAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]

    d = Settings()
    with gr.Blocks(title="kyrgyz-tts-mini") as ui:
        gr.Markdown("## kyrgyz-tts-mini · Кыргызча текстти үнгө айландыруу")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text (each line is spoken with a short pause)",
                    lines=6,
                    value="Саламатсызбы! Бүгүн аба ырайы абдан жакшы.",
                )
                voice = gr.Radio(list(config.VOICES), value=config.DEFAULT_VOICE, label="Voice")
                with gr.Accordion("Settings", open=False):
                    rate = gr.Slider(0.5, 2.0, value=d.rate, step=0.05, label="Rate (length scale, higher is slower)")
                    temperature = gr.Slider(
                        0.0, 1.5, value=d.temperature, step=0.01, label="Temperature (0 = deterministic)"
                    )
                    steps = gr.Slider(2, 50, value=d.steps, step=1, label="ODE steps")
                    denoise = gr.Slider(0.0, 0.01, value=d.denoise, step=0.00005, label="Denoiser strength")
                button = gr.Button("Speak", variant="primary")
            with gr.Column():
                audio = gr.Audio(label="Speech", type="numpy", format="wav", autoplay=True)
                info = gr.Textbox(label="Info", lines=2)
        if examples:
            gr.Examples(examples, inputs=[text])
        inputs, outputs = [text, voice, rate, temperature, steps, denoise], [audio, info]
        button.click(speak, inputs, outputs)
        text.submit(speak, inputs, outputs)

    ui.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.open)
