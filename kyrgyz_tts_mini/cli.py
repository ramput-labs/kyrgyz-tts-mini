"""Kyrgyz text-to-speech.

python -m kyrgyz_tts_mini speak "Саламатсызбы!" --voice woman --play   text → outputs/<time>-woman.wav
python -m kyrgyz_tts_mini speak -f samples/texts.txt                   every line of a file, joined into one WAV
python -m kyrgyz_tts_mini speak                                        interactive: type a line, hear it
python -m kyrgyz_tts_mini web                                          web UI at http://127.0.0.1:7860
python -m kyrgyz_tts_mini doctor                                       environment, models and a test synthesis
python -m kyrgyz_tts_mini download                                     fetch missing models from Hugging Face
python -m kyrgyz_tts_mini upload                                       push local models to Hugging Face
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

from kyrgyz_tts_mini import audio, config

DIM, CYAN, GREEN, RED, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[31m", "\033[0m"


def status(message: str) -> None:
    print(message, file=sys.stderr)


def synthesize(args, text: str):
    from kyrgyz_tts_mini.engine import get_tts
    from kyrgyz_tts_mini.text import dropped_characters

    speech = get_tts(args.device).synthesize(
        text,
        args.voice,
        temperature=args.temperature,
        speaking_rate=args.rate,
        steps=args.steps,
        denoiser_strength=args.denoise,
    )
    if skipped := dropped_characters(text).strip():
        status(f"{DIM}skipped (no pronunciation): {skipped}{RESET}")
    return speech


def report(speech, path: Path) -> None:
    rtf = speech.seconds / speech.audio_seconds if speech.audio_seconds else 0
    print(path)
    status(f"{DIM}{speech.audio_seconds:.1f}s of speech in {speech.seconds:.2f}s (RTF {rtf:.2f}){RESET}")


def warm_up(args) -> None:
    from kyrgyz_tts_mini.engine import get_tts

    tts = get_tts(args.device)
    status(f"Loading the {args.voice} voice on {tts.device}…")
    tts.warm_up(args.voice)


def cmd_speak(args) -> None:
    from kyrgyz_tts_mini.engine import Speech

    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        texts = [line.strip() for line in lines if line.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        return interactive(args)

    warm_up(args)
    speeches = [synthesize(args, text) for text in texts]
    sr = speeches[0].sample_rate
    pause = np.zeros(int(0.3 * sr), dtype=np.float32)
    joined = np.concatenate([part for s in speeches for part in (s.audio, pause)][:-1])
    speech = Speech(joined, sr, sum(s.seconds for s in speeches))
    path = audio.save(speech.audio, sr, args.output or audio.timestamped(config.OUTPUTS_DIR, f"-{args.voice}"))
    report(speech, path)
    if args.play:
        audio.play(speech.audio, sr)


def interactive(args) -> None:
    warm_up(args)
    status(f"Type Kyrgyz text and press Enter to hear it ({args.voice} voice). Ctrl-D quits.")
    try:
        while True:
            text = input(f"\n{CYAN}{args.voice}›{RESET} ").strip()
            if not text:
                continue
            try:
                speech = synthesize(args, text)
            except ValueError as e:
                status(str(e))
                continue
            report(speech, audio.save(speech.audio, speech.sample_rate, audio.timestamped(config.OUTPUTS_DIR)))
            audio.play(speech.audio, speech.sample_rate)
    except (KeyboardInterrupt, EOFError):
        status("")


def venv_status() -> tuple[bool, str]:
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if sys.prefix == sys.base_prefix or not cfg.exists():
        return True, "not in a virtual environment"
    fields = dict(line.split(" = ", 1) for line in cfg.read_text().splitlines() if " = " in line)
    created = fields.get("version", fields.get("version_info", "?"))
    if not created.startswith(f"{sys.version_info.major}.{sys.version_info.minor}"):
        return (
            False,
            f"created by Python {created} but running {platform.python_version()} → make clean-all && make setup",
        )
    return True, sys.prefix


def cmd_doctor(args) -> None:
    import torch

    from kyrgyz_tts_mini.models import MODELS, problem

    ok = True

    def row(label: str, value: str, good: bool = True) -> None:
        nonlocal ok
        ok &= good
        mark = f"{GREEN}✓{RESET}" if good else f"{RED}✗{RESET}"
        print(f"{mark} {label:<14} {value}")

    row("python", f"{platform.python_version()} ({sys.executable})", sys.version_info >= (3, 11))
    venv_ok, venv_note = venv_status()
    row("venv", venv_note, venv_ok)
    row("platform", f"{platform.system()} {platform.machine()}")
    row("torch", torch.__version__)
    row("device", args.device or audio.pick_device())
    for model in MODELS:
        issue = problem(model)
        row(f"model {model.name}", "ok" if issue is None else f"{issue} → run `make download`", issue is None)
    try:
        import sounddevice as sd

        row("audio output", sd.query_devices(kind="output")["name"])
    except Exception as e:  # no sound device (CI, SSH): only playback is affected
        print(f"{DIM}- audio output   unavailable ({e}); --play will not work{RESET}")

    if ok:
        from kyrgyz_tts_mini.engine import get_tts

        start = time.perf_counter()
        try:
            speech = get_tts(args.device).synthesize("Саламатсызбы!", next(iter(config.VOICES)))
            row("test synthesis", f"{speech.audio_seconds:.1f}s of speech in {time.perf_counter() - start:.1f}s")
        except Exception as e:
            row("test synthesis", f"failed: {e}", False)

    print(f"\n{GREEN}Ready.{RESET} Try: make run" if ok else f"\n{RED}Not ready{RESET}: fix the ✗ items above.")
    if not ok:
        sys.exit(1)


def cmd_web(args) -> None:
    try:
        from kyrgyz_tts_mini import web
    except ImportError:
        sys.exit("error: the web UI needs Gradio: pip install -r requirements.txt")
    web.launch(args)


def cmd_upload(args) -> None:
    from kyrgyz_tts_mini.models import DownloadError, upload

    try:
        upload(public=args.public)
    except DownloadError as e:
        sys.exit(f"error: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kyrgyz_tts_mini", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("speak", help="text → speech (no text: interactive)")
    p.add_argument("text", nargs="*", help="text to speak")
    p.add_argument("-f", "--file", help="speak every non-empty line of a UTF-8 text file")
    p.add_argument("-v", "--voice", choices=list(config.VOICES), default="woman", help="voice (default: woman)")
    p.add_argument("-o", "--output", help=f"WAV path (default: {config.OUTPUTS_DIR.name}/<time>-<voice>.wav)")
    p.add_argument("-p", "--play", action="store_true", help="play the result")
    p.add_argument("--rate", type=float, default=1.0, help="length scale, higher is slower (default: 1.0)")
    p.add_argument("--temperature", type=float, default=0.667, help="variation; 0 = deterministic (default: 0.667)")
    p.add_argument("--steps", type=int, default=10, help="ODE solver steps (default: 10)")
    p.add_argument("--denoise", type=float, default=0.00025, help="vocoder denoiser strength, 0 = off")
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("doctor", help="check the environment and models, run a test synthesis")
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("web", help="web UI in the browser")
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.set_defaults(func=cmd_web)

    sub.add_parser("download", help="download / verify the models (see: download --help)", add_help=False)

    p = sub.add_parser("upload", help=f"push local models to huggingface.co/{config.HF_REPO}")
    p.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    p.set_defaults(func=cmd_upload)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["download"]:
        from kyrgyz_tts_mini import models

        return models.main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
