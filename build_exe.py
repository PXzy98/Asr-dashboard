"""Build a standalone Windows/Linux bundle with PyInstaller.

    uv pip install pyinstaller
    uv run python build_exe.py

The bundle supports CPU or CUDA PyTorch (whatever is installed in the build environment).
It does NOT include the DiariZen environment (install it from Settings after unpacking) and
does not include model weights (downloaded on first use). ROCm and vLLM are not supported in the bundle.
"""

import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "PyInstaller", "--noconfirm", "asr_dashboard.spec"]))
