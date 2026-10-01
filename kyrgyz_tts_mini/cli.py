"""Kyrgyz text-to-speech.

python -m kyrgyz_tts_mini speak "Саламатсызбы!" --voice woman --play   text → outputs/<time>-woman.wav
python -m kyrgyz_tts_mini speak -f samples/texts.txt                   every line of a file, joined into one WAV
python -m kyrgyz_tts_mini speak                                        interactive: type a line, hear it
python -m kyrgyz_tts_mini web                                          web UI at http://127.0.0.1:7860
python -m kyrgyz_tts_mini doctor                                       environment, models and a test synthesis
python -m kyrgyz_tts_mini download [--check]                           fetch (or verify) the models from Hugging Face
"""

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from kyrgyz_tts_mini import config, download
from kyrgyz_tts_mini.config import Settings
from kyrgyz_tts_mini.text import dropped_characters

DIM, CYAN, GREEN, RED, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[31m", "\033[0m"


def status(message: str) -> None:
    print(message, file=sys.stderr)


def output_path(args) -> Path:
    return (
        Path(args.output) if args.output else config.OUTPUTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{args.voice}.wav"
    )


def settings(args) -> Settings:
    return Settings(temperature=args.temperature, rate=args.rate, steps=args.steps, denoise=args.denoise)


def load(args):
    from kyrgyz_tts_mini.engine import get_tts

    tts = get_tts(args.device)
    status(f"Loading the {args.voice} voice on {tts.device}…")
    tts.warm_up(args.voice)
    return tts


def report(speech, path: Path, text: str) -> None:
    print(path)
    if skipped := dropped_characters(text).strip():
        status(f"{DIM}skipped (no pronunciation): {skipped}{RESET}")
    rtf = speech.elapsed / speech.duration if speech.duration else 0
    status(f"{DIM}{speech.duration:.1f}s of speech in {speech.elapsed:.2f}s (RTF {rtf:.2f}){RESET}")


def cmd_speak(args) -> None:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = " ".join(args.text)
    else:
        return interactive(args)

    speech = load(args).synthesize_lines(text.splitlines(), args.voice, settings(args))
    report(speech, speech.save(output_path(args)), text)
    if args.play:
        speech.play()


def interactive(args) -> None:
    tts = load(args)
    status(f"Type Kyrgyz text and press Enter to hear it ({args.voice} voice). Ctrl-D quits.")
    try:
        while True:
            if not (text := input(f"\n{CYAN}{args.voice}›{RESET} ").strip()):
                continue
            try:
                speech = tts.synthesize(text, args.voice, settings(args))
            except ValueError as e:
                status(str(e))
                continue
            report(speech, speech.save(output_path(args)), text)
            speech.play()
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

    from kyrgyz_tts_mini.engine import get_tts, pick_device

    ok = True

    def row(label: str, value: str, good: bool = True) -> None:
        nonlocal ok
        ok &= good
        print(f"{f'{GREEN}✓' if good else f'{RED}✗'}{RESET} {label:<14} {value}")

    row("python", f"{platform.python_version()} ({sys.executable})", sys.version_info >= (3, 11))
    venv_ok, venv_note = venv_status()
    row("venv", venv_note, venv_ok)
    row("platform", f"{platform.system()} {platform.machine()}")
    row("torch", torch.__version__)
    row("device", args.device or pick_device())
    for model in download.MODELS:
        issue = download.problem(model)
        row(f"model {model.name}", "ok" if issue is None else f"{issue} → run `make download`", issue is None)
    try:
        import sounddevice as sd

        row("audio output", sd.query_devices(kind="output")["name"])
    except Exception as e:  # no sound device (CI, SSH): only --play is affected
        print(f"{DIM}- audio output   unavailable ({e}); --play will not work{RESET}")

    if ok:
        start = time.perf_counter()
        try:
            speech = get_tts(args.device).synthesize("Саламатсызбы!")
            row("test synthesis", f"{speech.duration:.1f}s of speech in {time.perf_counter() - start:.1f}s")
        except Exception as e:
            row("test synthesis", f"failed: {e}", False)

    if not ok:
        sys.exit(f"\n{RED}Not ready{RESET}: fix the ✗ items above.")
    print(f"\n{GREEN}Ready.{RESET} Try: make run")


def cmd_web(args) -> None:
    try:
        from kyrgyz_tts_mini import web
    except ImportError:
        sys.exit("error: the web UI needs Gradio: pip install -r requirements.txt")
    web.launch(args)


def cmd_download(args) -> None:
    if args.check:
        sys.exit(0 if download.check() else 1)
    try:
        download.download(args.models, args.force)
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: run the same command again to resume")


def model_name(name: str) -> str:
    names = [m.name for m in download.MODELS]
    if name not in names:
        raise argparse.ArgumentTypeError(f"unknown model {name!r}; choose from {', '.join(names)}")
    return name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kyrgyz_tts_mini", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    device = argparse.ArgumentParser(add_help=False)
    device.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    d = Settings()

    p = sub.add_parser("speak", parents=[device], help="text → speech (no text: interactive)")
    p.add_argument("text", nargs="*", help="text to speak")
    p.add_argument("-f", "--file", help="speak every non-empty line of a UTF-8 text file")
    p.add_argument(
        "-v", "--voice", choices=list(config.VOICES), default=config.DEFAULT_VOICE, help="(default: %(default)s)"
    )
    p.add_argument("-o", "--output", help=f"WAV path (default: {config.OUTPUTS_DIR.name}/<time>-<voice>.wav)")
    p.add_argument("-p", "--play", action="store_true", help="play the result")
    p.add_argument("--rate", type=float, default=d.rate, help="length scale, higher is slower (default: %(default)s)")
    p.add_argument(
        "--temperature", type=float, default=d.temperature, help="variation; 0 = same every time (default: %(default)s)"
    )
    p.add_argument("--steps", type=int, default=d.steps, help="ODE solver steps (default: %(default)s)")
    p.add_argument(
        "--denoise", type=float, default=d.denoise, help="vocoder denoiser strength; 0 = off (default: %(default)s)"
    )
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("doctor", parents=[device], help="check the environment and models, run a test synthesis")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("web", parents=[device], help="web UI in the browser")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--open", action="store_true", help="open the UI in the default browser")
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("download", help="download or verify the models")
    p.add_argument("models", nargs="*", type=model_name, metavar="MODEL", help="woman, man, vocoder (default: all)")
    p.add_argument("--force", action="store_true", help="download again even if installed")
    p.add_argument("--check", action="store_true", help="verify the installed models against their SHA-256 and exit")
    p.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, download.DownloadError) as e:
        sys.exit(f"error: {e}")
