# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ASR Dashboard (CPU or CUDA build; no DiariZen environment, no model weights).
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("qfluentwidgets", "qwen_asr", "pyannote", "silero_vad", "transformers", "librosa", "jsonpath_ng",
            "lightning_fabric", "pytorch_lightning", "asteroid_filterbanks", "speechbrain"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass
datas += [("src/asr_dashboard/resources", "asr_dashboard/resources"),
          ("src/asr_dashboard/core/diarization/diarizen_worker.py", "asr_dashboard/core/diarization"),
          ("config.example.yaml", ".")]
hiddenimports += ["keyring.backends.Windows", "keyring.backends.SecretService", "keyring.backends.macOS",
                  "PyQt6.QtMultimedia", "websockets.asyncio.client"]

a = Analysis(["main.py"], pathex=["src"], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["vllm", "flash_attn", "tkinter"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ASR-Dashboard", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="ASR-Dashboard")
