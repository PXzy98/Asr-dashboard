"""Strictly sequential folder batch processing with a crash-safe manifest and resume support."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from asr_dashboard import paths
from asr_dashboard.core.audio import MEDIA_EXTS, probe
from asr_dashboard.models import CancelledError, CancelToken

log = logging.getLogger(__name__)

MANIFEST = "batch_manifest.json"
PENDING, RUNNING, DONE, FAILED, SKIPPED, CANCELLED = "pending", "running", "done", "failed", "skipped", "cancelled"
STATUS_LABEL = {PENDING: "等待", RUNNING: "处理中", DONE: "完成", FAILED: "失败", SKIPPED: "跳过", CANCELLED: "已取消"}


# ------------------------------------------------------------------------------ scanning / sorting
def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def path_sort_key(rel: Path) -> tuple:
    return (natural_key(rel.parent.as_posix()), natural_key(rel.name))


def _hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def scan_folder(folder: str | Path, recursive: bool = True, extensions: list[str] | None = None,
                skip_hidden: bool = True, skip_empty: bool = True, sort: str = "natural") -> list[Path]:
    root = Path(folder)
    exts = {("." + e.lower().lstrip(".")) for e in extensions} if extensions else MEDIA_EXTS
    it = root.rglob("*") if recursive else root.glob("*")
    files = []
    for p in it:
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        if skip_hidden and (_hidden(rel) or _win_hidden(p)):
            continue
        if skip_empty and p.stat().st_size == 0:
            continue
        files.append(p)
    if sort == "mtime":
        files.sort(key=lambda p: (p.stat().st_mtime, natural_key(p.name)))
    elif sort == "size":
        files.sort(key=lambda p: (p.stat().st_size, natural_key(p.name)))
    else:
        files.sort(key=lambda p: path_sort_key(p.relative_to(root)))
    return files


def _win_hidden(p: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(os.stat(p).st_file_attributes & 2)  # FILE_ATTRIBUTE_HIDDEN
    except (OSError, AttributeError):
        return False


# ------------------------------------------------------------------------------ manifest
@dataclass
class BatchItem:
    path: str
    rel: str
    size: int
    mtime: float
    status: str = PENDING
    outputs: dict[str, str] = field(default_factory=dict)
    elapsed: float = 0.0
    error: str = ""
    duration: float = 0.0
    speakers: list[str] = field(default_factory=list)
    warnings: int = 0
    result_path: str = ""

    @property
    def rtf(self) -> float | None:
        return self.elapsed / self.duration if self.duration > 0 and self.elapsed > 0 else None


@dataclass
class BatchManifest:
    batch_id: str
    created_at: float
    folder: str
    output_dir: str
    settings: dict
    items: list[BatchItem]
    finished_at: float | None = None
    path: str = ""

    @classmethod
    def create(cls, folder: str | Path, files: list[Path], output_dir: str | Path, settings: dict) -> BatchManifest:
        root = Path(folder)
        items = []
        for f in files:
            st = f.stat()
            try:
                rel = f.relative_to(root).as_posix()
            except ValueError:  # appended from outside the folder
                rel = f.name
            items.append(BatchItem(str(f), rel, st.st_size, st.st_mtime))
        m = cls(uuid.uuid4().hex[:10], time.time(), str(root), str(output_dir), settings, items)
        m.path = str(Path(output_dir) / MANIFEST)
        return m

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        d["files"] = d.pop("items")
        return d

    @classmethod
    def from_dict(cls, d: dict, path: str = "") -> BatchManifest:
        items = [BatchItem(**{k: v for k, v in it.items() if k in BatchItem.__dataclass_fields__})
                 for it in d.get("files", [])]
        return cls(d["batch_id"], d["created_at"], d["folder"], d["output_dir"], d.get("settings", {}), items,
                   d.get("finished_at"), path)

    def save(self) -> None:
        """Atomic write: temp file + os.replace (never leaves a half-written manifest)."""
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | Path) -> BatchManifest:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh), str(path))

    @property
    def unfinished(self) -> bool:
        return self.finished_at is None and any(i.status in (PENDING, RUNNING) for i in self.items)

    def next_index(self) -> int:
        return next((k for k, i in enumerate(self.items) if i.status in (PENDING, RUNNING)), len(self.items))

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for i in self.items:
            c[i.status] = c.get(i.status, 0) + 1
        return c


def is_completed(item: BatchItem) -> bool:
    """Done in the manifest, source unchanged, and every output still exists."""
    if item.status != DONE:
        return False
    try:
        st = os.stat(item.path)
    except OSError:
        return False
    if st.st_size != item.size or abs(st.st_mtime - item.mtime) > 1e-3:
        return False
    return bool(item.outputs) and all(Path(p).exists() for p in item.outputs.values())


# ------------------------------------------------------------------------------ registry of batches
def _registry_file() -> Path:
    return paths.app_home() / "batches.json"


def register_manifest(path: str) -> None:
    f = _registry_file()
    try:
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    except Exception:  # noqa: BLE001
        data = []
    if path not in data:
        data.append(path)
    f.write_text(json.dumps(data[-50:], ensure_ascii=False), encoding="utf-8")


def find_unfinished() -> list[BatchManifest]:
    f = _registry_file()
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for p in data:
        try:
            m = BatchManifest.load(p)
            if m.unfinished:
                out.append(m)
        except Exception:  # noqa: BLE001
            continue
    return out


def abandon(manifest: BatchManifest) -> None:
    for i in manifest.items:
        if i.status in (PENDING, RUNNING):
            i.status = CANCELLED
    manifest.finished_at = time.time()
    manifest.save()


# ------------------------------------------------------------------------------ sleep prevention
class SleepInhibitor:
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
            except Exception as e:  # noqa: BLE001
                log.warning("无法阻止系统休眠：%s", e)
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except Exception:  # noqa: BLE001
                pass
        return False


# ------------------------------------------------------------------------------ runner
@dataclass
class BatchCallbacks:
    on_item_start: Callable[[int, BatchItem], None] = lambda i, it: None
    on_item_done: Callable[[int, BatchItem], None] = lambda i, it: None
    on_progress: Callable[[int, float, str, float, float | None], None] = lambda i, f, m, o, eta: None
    on_info: Callable[[str], None] = lambda m: None
    on_paused: Callable[[], None] = lambda: None


ProcessFn = Callable[[BatchItem, Path, str, Callable[[float, str], None], CancelToken], "ProcessOutcome"]


@dataclass
class ProcessOutcome:
    outputs: dict[str, str]
    duration: float
    speakers: list[str] = field(default_factory=list)
    warnings: int = 0
    result_path: str = ""


class BatchRunner:
    """Runs a manifest strictly in order with one worker. ``process`` does the per-file work."""

    def __init__(self, manifest: BatchManifest, process: ProcessFn, on_existing: str = "skip",
                 prevent_sleep: bool = True, oom_retry: bool = True, reduce_batch: Callable[[], bool] | None = None,
                 callbacks: BatchCallbacks | None = None) -> None:
        self.m = manifest
        self.process = process
        self.on_existing = on_existing
        self.prevent_sleep = prevent_sleep
        self.oom_retry = oom_retry
        self.reduce_batch = reduce_batch or (lambda: False)
        self.cb = callbacks or BatchCallbacks()
        self.cancel = CancelToken()
        self._pause = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self.started_at = 0.0
        self.processed_audio = 0.0
        self.processing_time = 0.0

    # controls (thread-safe) --------------------------------------------------------
    def pause(self) -> None:
        self._pause.set()
        self._resume.clear()

    def resume(self) -> None:
        self._pause.clear()
        self._resume.set()

    def stop(self) -> None:
        self.cancel.cancel()
        self._resume.set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    # helpers -------------------------------------------------------------------------
    def out_dir_for(self, item: BatchItem) -> Path:
        return Path(self.m.output_dir) / Path(item.rel).parent

    def _existing_outputs(self, item: BatchItem) -> dict[str, str]:
        d = self.out_dir_for(item)
        stem = Path(item.rel).stem
        fmts = self.m.settings.get("formats", ["srt", "txt"])
        found = {f: str(d / f"{stem}.{f}") for f in fmts if (d / f"{stem}.{f}").exists()}
        return found if len(found) == len(fmts) else {}

    def total_audio(self) -> float:
        return sum(max(i.duration, 0.0) or i.size / 32000 for i in self.m.items)

    def _eta(self, current_audio_done: float) -> float | None:
        if self.processed_audio + current_audio_done <= 0 or self.processing_time <= 0:
            return None
        speed = (self.processed_audio) / self.processing_time if self.processed_audio else None
        if not speed:
            return None
        remaining = sum((i.duration or i.size / 32000) for i in self.m.items if i.status in (PENDING, RUNNING))
        return max(0.0, (remaining - current_audio_done) / speed)

    def _overall(self, idx: int, frac: float) -> float:
        tot = self.total_audio() or 1.0
        done = sum((i.duration or i.size / 32000) for i in self.m.items[:idx])
        cur = (self.m.items[idx].duration or self.m.items[idx].size / 32000) * frac if idx < len(self.m.items) else 0
        return min(1.0, (done + cur) / tot)

    # main loop -------------------------------------------------------------------------
    def run(self) -> BatchManifest:
        self.started_at = time.time()
        for it in self.m.items:  # durations up front for weighted progress (cheap ffprobe)
            if not it.duration:
                try:
                    it.duration = probe(it.path).duration
                except Exception:  # noqa: BLE001
                    it.duration = 0.0
        register_manifest(self.m.path)
        self.m.save()
        with SleepInhibitor() if self.prevent_sleep else _Null():
            for idx, item in enumerate(self.m.items):
                if self.cancel.cancelled:
                    break
                if self._pause.is_set():
                    self.cb.on_paused()
                    self._resume.wait()
                    if self.cancel.cancelled:
                        break
                if item.status in (DONE, SKIPPED):
                    if is_completed(item) or item.status == SKIPPED:
                        continue
                    item.status = PENDING  # source changed or outputs missing -> redo
                if item.status in (FAILED, CANCELLED):
                    continue  # only "retry failed" re-queues these
                if self.on_existing == "skip" and (existing := self._existing_outputs(item)):
                    item.status, item.outputs, item.error = SKIPPED, existing, "输出已存在，按设置跳过"
                    self.m.save()
                    self.cb.on_item_done(idx, item)
                    continue
                self._run_item(idx, item)
        if not self.cancel.cancelled and not any(i.status in (PENDING, RUNNING) for i in self.m.items):
            self.m.finished_at = time.time()
        self.m.save()
        write_reports(self.m)
        return self.m

    def _run_item(self, idx: int, item: BatchItem) -> None:
        item.status, item.error = RUNNING, ""
        self.m.save()
        self.cb.on_item_start(idx, item)
        t0 = time.time()
        attempts = 2 if self.oom_retry else 1
        for attempt in range(attempts):
            try:
                def prog(frac: float, msg: str) -> None:
                    cur_audio = (item.duration or 0) * frac
                    self.cb.on_progress(idx, frac, msg, self._overall(idx, frac), self._eta(cur_audio))

                on_ex = "overwrite" if self.on_existing == "overwrite" else "rename"
                out = self.process(item, self.out_dir_for(item), on_ex, prog, self.cancel)
                item.outputs, item.duration = out.outputs, out.duration or item.duration
                item.speakers, item.warnings, item.result_path = out.speakers, out.warnings, out.result_path
                item.status = DONE
                break
            except CancelledError:
                item.status, item.error = CANCELLED, "已取消"
                break
            except Exception as e:  # noqa: BLE001 - one bad file never stops the batch
                from asr_dashboard.core.devices import empty_cache, is_oom_error

                empty_cache()
                if is_oom_error(e) and attempt + 1 < attempts and self.reduce_batch():
                    self.cb.on_info(f"{item.rel}：显存不足，批大小减半后重试")
                    continue
                log.exception("批量处理失败：%s", item.path)
                item.status, item.error = FAILED, f"{type(e).__name__}: {e}"
                break
        item.elapsed = time.time() - t0
        if item.status == DONE:
            self.processed_audio += item.duration
            self.processing_time += item.elapsed
        try:
            from asr_dashboard.core.devices import empty_cache, memory_summary

            empty_cache()
            log.info("批量：%s 完成（%s），%s", item.rel, item.status, memory_summary("cuda:0"))
        except Exception:  # noqa: BLE001
            pass
        self.m.save()
        self.cb.on_item_done(idx, item)

    def retry_failed(self) -> None:
        for i in self.m.items:
            if i.status in (FAILED, CANCELLED):
                i.status, i.error = PENDING, ""
        self.m.finished_at = None
        self.cancel = CancelToken()
        self.m.save()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def write_reports(m: BatchManifest) -> tuple[Path, Path]:
    out = Path(m.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in m.items:
        rows.append({
            "file": i.rel, "duration_s": round(i.duration, 2), "status": STATUS_LABEL.get(i.status, i.status),
            "elapsed_s": round(i.elapsed, 2), "rtf": round(i.rtf, 3) if i.rtf else "",
            "speakers": "、".join(i.speakers), "warnings": i.warnings, "error": i.error,
        })
    csv_p, json_p = out / "batch_report.csv", out / "batch_report.json"
    with open(csv_p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    c = m.counts()
    summary = {"batch_id": m.batch_id, "folder": m.folder, "done": c.get(DONE, 0), "failed": c.get(FAILED, 0),
               "skipped": c.get(SKIPPED, 0), "cancelled": c.get(CANCELLED, 0),
               "total_elapsed_s": round(sum(i.elapsed for i in m.items), 2), "files": rows}
    json_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_p, json_p
