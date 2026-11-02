"""Entry point: ``python main.py`` (or ``uv run asr-dashboard``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":  # required by vLLM (spawn) and PyInstaller
    import multiprocessing

    multiprocessing.freeze_support()
    from asr_dashboard.app import main

    sys.exit(main())
