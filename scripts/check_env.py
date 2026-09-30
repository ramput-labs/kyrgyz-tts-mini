"""Check this machine before `make setup` creates the environment. Standard library only.

Usage: python scripts/check_env.py MIN_FREE_GB
"""

import importlib.util
import platform
import shutil
import sys

MIN_PYTHON = (3, 11)


def main() -> int:
    need_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    system = platform.system()
    problems = []

    if sys.version_info < MIN_PYTHON:
        hint = "brew install python@3.12" if system == "Darwin" else "sudo apt install python3.12 python3.12-venv"
        problems.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, found {platform.python_version()} "
            f"({sys.executable}).\n    Install one ({hint}) or point to it: make setup PYTHON=/path/to/python3.12"
        )
    if importlib.util.find_spec("venv") is None or importlib.util.find_spec("ensurepip") is None:
        problems.append("the Python venv module is missing.\n    On Debian/Ubuntu: sudo apt install python3-venv")
    if system not in ("Darwin", "Linux"):
        problems.append(f"{system} is not supported; use macOS or Linux (on Windows: WSL2).")

    free_gb = shutil.disk_usage(".").free / 1e9
    if free_gb < need_gb:
        problems.append(f"not enough disk space: {free_gb:.1f} GB free, about {need_gb:.0f} GB needed.")

    print(f"Python {platform.python_version()} · {system} {platform.machine()} · {free_gb:.0f} GB free")
    for p in problems:
        print(f"✗ {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
