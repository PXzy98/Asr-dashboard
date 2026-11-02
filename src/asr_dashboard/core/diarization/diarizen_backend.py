"""DiariZen run through a subprocess in an isolated virtual environment.

DiariZen ships a modified pyannote-audio and pins torch 2.1.1 / Python 3.10 (README of BUTSpeechFIT/DiariZen),
so it cannot share the main environment with pyannote.audio 4.x. Model weights are CC BY-NC 4.0
(non-commercial only) – the UI asks for explicit confirmation before first use.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from asr_dashboard.core.diarization.base import (
    BaseDiarizer,
    DiarizationOutput,
    DiarizeRequest,
    ProgressCb,
    SpeakerTurn,
)
from asr_dashboard.models import CancelledError, CancelToken

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/BUTSpeechFIT/DiariZen.git"
DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
LICENSE_TEXT = (
    "DiariZen 的代码采用 MIT 许可，但预训练模型权重采用 CC BY-NC 4.0 许可，仅限非商业用途（研究和学术）。\n"
    "启用即表示你确认只在非商业场景中使用，并自行承担相应的法律责任。"
)
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def default_env_dir() -> Path:
    from asr_dashboard.paths import envs_dir

    return envs_dir() / "diarizen"


def env_python(env_dir: str | Path | None) -> Path:
    d = Path(env_dir) if env_dir else default_env_dir()
    return d / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def worker_script() -> Path:
    return Path(__file__).with_name("diarizen_worker.py")


def availability(env_dir: str | None, license_accepted: bool) -> tuple[bool, str]:
    if not env_python(env_dir).exists():
        return False, "未安装 DiariZen 隔离环境（设置 → 说话人分离 → 安装 DiariZen 环境）"
    if not license_accepted:
        return False, "尚未确认 DiariZen 模型权重的非商业许可"
    return True, ""


class DiarizenDiarizer(BaseDiarizer):
    name = "diarizen"
    display_name = "DiariZen（隔离环境）"

    def __init__(self, env_dir: str | None, model_id: str | None, device: str = "cpu",
                 python: str | None = None, script: str | None = None) -> None:
        self.python = python or str(env_python(env_dir))
        self.script = script or str(worker_script())
        self.model_id = model_id or DEFAULT_MODEL
        self.device = device

    def load(self) -> None:
        if not Path(self.python).exists():
            raise RuntimeError(f"DiariZen 环境不存在：{self.python}")

    def diarize(self, req: DiarizeRequest, progress: ProgressCb, cancel: CancelToken) -> DiarizationOutput:
        self.load()
        payload = {"wav": req.wav_path, "num_speakers": req.num_speakers, "min_speakers": req.min_speakers,
                   "max_speakers": req.max_speakers, "device": req.device or self.device, "model_id": self.model_id}
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("PYTHONPATH", None)
        proc = subprocess.Popen([self.python, "-u", self.script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, creationflags=_NO_WINDOW)
        cancel.on_cancel(proc.kill)
        stderr_lines: list[str] = []

        def pump_err() -> None:
            for line in proc.stderr:  # type: ignore[union-attr]
                stderr_lines.append(line.decode("utf-8", "replace"))
                del stderr_lines[:-200]

        t = threading.Thread(target=pump_err, daemon=True)
        t.start()
        result = None
        error = None
        try:
            proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    progress(float(msg.get("value", 0)), msg.get("message", "DiariZen"))
                elif kind == "result":
                    result = msg
                elif kind == "error":
                    error = msg.get("message", "unknown error")
            proc.wait()
        finally:
            cancel.remove_callback(proc.kill)
        t.join(timeout=2)
        if cancel.cancelled:
            raise CancelledError("任务已取消")
        if error or result is None:
            tail = "".join(stderr_lines[-15:])
            raise RuntimeError(f"DiariZen 子进程失败：{error or '没有返回结果'}\n{tail}".strip())
        turns = [SpeakerTurn(float(x["start"]), float(x["end"]), str(x["speaker"])) for x in result.get("turns", [])]
        # DiariZen output can overlap; derive an exclusive version by keeping the dominant speaker per instant
        return DiarizationOutput(turns=turns, exclusive_turns=make_exclusive(turns), backend=self.name)


def make_exclusive(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Resolve overlaps: at each instant keep the speaker whose current turn is the longest."""
    if not turns:
        return []
    bounds = sorted({t.start for t in turns} | {t.end for t in turns})
    out: list[SpeakerTurn] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        active = [t for t in turns if t.start <= a and t.end >= b]
        if not active:
            continue
        spk = max(active, key=lambda t: t.duration).speaker
        if out and out[-1].speaker == spk and abs(out[-1].end - a) < 1e-6:
            out[-1].end = b
        else:
            out.append(SpeakerTurn(a, b, spk))
    return out


# ------------------------------------------------------------------------------ installer
class DiarizenInstaller:
    """Create ``~/.asr-dashboard/envs/diarizen`` with uv and install DiariZen into it."""

    def __init__(self, env_dir: str | None = None, cuda: bool = True,
                 log_cb: Callable[[str], None] | None = None) -> None:
        self.env_dir = Path(env_dir) if env_dir else default_env_dir()
        self.cuda = cuda
        self.log = log_cb or (lambda s: log.info(s))
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def find_uv() -> str | None:
        exe = shutil.which("uv")
        if exe:
            return exe
        cand = Path(sys.executable).with_name("uv.exe" if sys.platform == "win32" else "uv")
        return str(cand) if cand.exists() else None

    def commands(self) -> list[list[str]]:
        uv = self.find_uv() or "uv"
        git = shutil.which("git") or "git"
        py = str(env_python(self.env_dir))
        src = str(self.env_dir / "src")
        index = "https://download.pytorch.org/whl/cu121" if self.cuda else "https://download.pytorch.org/whl/cpu"
        return [
            [uv, "venv", "--python", "3.10", str(self.env_dir)],
            [uv, "pip", "install", "--python", py, "torch==2.1.1", "torchvision==0.16.1", "torchaudio==2.1.1",
             "--index-url", index],
            [git, "clone", "--depth", "1", "--recurse-submodules", REPO_URL, src],
            [uv, "pip", "install", "--python", py, "-r", str(Path(src) / "requirements.txt")],
            [uv, "pip", "install", "--python", py, "-e", src],
            [uv, "pip", "install", "--python", py, "-e", str(Path(src) / "pyannote-audio"),
             "-c", str(Path(src) / "constraints.txt")],
        ]

    def install(self, cancel: CancelToken | None = None, progress: ProgressCb | None = None) -> None:
        if not self.find_uv():
            raise RuntimeError("未找到 uv。请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/")
        if not shutil.which("git"):
            raise RuntimeError("未找到 git，安装 DiariZen 需要 git")
        cmds = self.commands()
        for i, cmd in enumerate(cmds):
            if cancel and cancel.cancelled:
                raise CancelledError("安装已取消")
            if cmd[1] == "clone" and Path(cmd[-1]).exists():
                self.log(f"[跳过] 源码已存在：{cmd[-1]}")
                continue
            if cmd[-1].endswith("constraints.txt") and not Path(cmd[-1]).exists():
                cmd = cmd[:-2]
            self.log("$ " + " ".join(cmd))
            if progress:
                progress(i / len(cmds), f"步骤 {i + 1}/{len(cmds)}")
            self._run(cmd, cancel)
        if progress:
            progress(1.0, "安装完成")

    def _run(self, cmd: list[str], cancel: CancelToken | None) -> None:
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=_NO_WINDOW)
        if cancel:
            cancel.on_cancel(self._proc.kill)
        for raw in self._proc.stdout:  # type: ignore[union-attr]
            self.log(raw.decode("utf-8", "replace").rstrip())
        rc = self._proc.wait()
        if cancel:
            cancel.remove_callback(self._proc.kill)
        if cancel and cancel.cancelled:
            raise CancelledError("安装已取消")
        if rc != 0:
            raise RuntimeError(f"命令失败（退出码 {rc}）：{' '.join(cmd)}")

    def uninstall(self) -> None:
        shutil.rmtree(self.env_dir, ignore_errors=True)
