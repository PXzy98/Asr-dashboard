"""FFmpeg-based audio extraction and helpers. Everything is normalised to 16 kHz mono PCM16 wav."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asr_dashboard.models import CancelledError, CancelToken

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

FFMPEG_HELP = (
    "未检测到 FFmpeg。请安装后重启程序：\n"
    "  Windows：winget install Gyan.FFmpeg  （或从 https://www.gyan.dev/ffmpeg/builds/ 下载并加入 PATH）\n"
    "  Ubuntu/Debian：sudo apt install ffmpeg\n"
    "  macOS：brew install ffmpeg"
)

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


class AudioError(RuntimeError):
    pass


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def ffmpeg_version() -> str | None:
    exe = ffmpeg_path()
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10,
                             encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
        return out.stdout.splitlines()[0] if out.stdout else "unknown"
    except Exception:  # noqa: BLE001
        return None


def is_media_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTS


@dataclass
class MediaInfo:
    path: str
    duration: float
    size_bytes: int
    format: str
    has_audio: bool = True

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024


def probe(path: str | Path) -> MediaInfo:
    p = Path(path)
    size = p.stat().st_size if p.exists() else 0
    exe = ffprobe_path()
    duration = 0.0
    fmt = p.suffix.lower().lstrip(".")
    has_audio = True
    if exe:
        try:
            out = subprocess.run(
                [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)],
                # ffprobe writes UTF-8 (tags, paths); the locale default on Windows is cp1252/gbk
                capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
                creationflags=_NO_WINDOW,
            )
            if out.returncode != 0:
                raise AudioError(out.stderr.strip() or "ffprobe 失败")
            info = json.loads(out.stdout or "{}")
            duration = float(info.get("format", {}).get("duration") or 0.0)
            fmt = info.get("format", {}).get("format_name", fmt)
            has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        except AudioError:
            raise
        except Exception as e:  # noqa: BLE001
            log.debug("ffprobe failed for %s: %s", p, e)
    if duration <= 0 and p.suffix.lower() == ".wav":
        try:
            import soundfile as sf

            duration = sf.info(str(p)).duration
        except Exception:  # noqa: BLE001
            pass
    return MediaInfo(str(p), duration, size, fmt, has_audio)


def temp_root(temp_dir: str | None = None) -> Path:
    base = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "asr-dashboard"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _run_ffmpeg(args: list[str], cancel: CancelToken | None) -> None:
    exe = ffmpeg_path()
    if not exe:
        raise AudioError(FFMPEG_HELP)
    proc = subprocess.Popen([exe, "-hide_banner", "-loglevel", "error", "-y", *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    kill = proc.kill
    if cancel:
        cancel.on_cancel(kill)
    try:
        _, err = proc.communicate()
    finally:
        if cancel:
            cancel.remove_callback(kill)
    if cancel and cancel.cancelled:
        raise CancelledError("任务已取消")
    if proc.returncode != 0:
        msg = (err or b"").decode("utf-8", "replace").strip()
        raise AudioError(f"FFmpeg 处理失败（文件可能已损坏或不含音轨）：{msg[-800:]}")


def to_wav16k(src: str | Path, temp_dir: str | None = None, cancel: CancelToken | None = None,
              start: float | None = None, end: float | None = None) -> Path:
    """Convert any media file to a 16 kHz mono PCM16 wav in the temp dir."""
    src = Path(src)
    if not src.exists():
        raise AudioError(f"文件不存在：{src}")
    out = temp_root(temp_dir) / f"{src.stem[:40]}_{uuid.uuid4().hex[:8]}.wav"
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(src)]
    if end is not None:
        args += ["-t", f"{max(0.0, end - (start or 0.0)):.3f}"]
    args += ["-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)]
    _run_ffmpeg(args, cancel)
    if not out.exists() or out.stat().st_size <= 44:
        raise AudioError(f"未能从 {src.name} 提取到音频")
    return out


def encode(src_wav: Path, fmt: str, temp_dir: str | None = None, bitrate: str = "64k",
           cancel: CancelToken | None = None) -> Path:
    """Re-encode a wav to mp3/flac (16 kHz mono) to shrink uploads."""
    out = temp_root(temp_dir) / f"{src_wav.stem}_{uuid.uuid4().hex[:6]}.{fmt}"
    args = ["-i", str(src_wav), "-ac", "1", "-ar", str(SAMPLE_RATE)]
    if fmt == "mp3":
        args += ["-c:a", "libmp3lame", "-b:a", bitrate]
    elif fmt == "flac":
        args += ["-c:a", "flac"]
    args.append(str(out))
    _run_ffmpeg(args, cancel)
    return out


def load_wav(path: str | Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        data = resample(data, sr, SAMPLE_RATE)
    return data.astype(np.float32, copy=False)


def save_wav(path: str | Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def slice_samples(samples: np.ndarray, start: float, end: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(samples), int(round(end * sr)))
    return samples[a:b]


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * np.log10(max(r, 1e-6))


def safe_unlink(*paths: Path | str | None) -> None:
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except OSError:
                pass


def generate_test_tone(seconds: float = 3.0) -> np.ndarray:
    """Synthetic 'speech-like' signal used when no bundled test clip exists."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    sig = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((180, 360, 720, 1440)))
    return (0.2 * env * sig).astype(np.float32)


def bundled_test_clip(seconds: float = 3.0) -> np.ndarray:
    """Built-in test audio for 'test connection' buttons (resources/test_speech.wav if present)."""
    from asr_dashboard.paths import resources_dir

    p = resources_dir() / "test_speech.wav"
    if p.exists():
        x = load_wav(p)
        return x[: int(seconds * SAMPLE_RATE)] if seconds else x
    return generate_test_tone(seconds)
