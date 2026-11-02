"""Task history persisted to ``~/.asr-dashboard/history.json``; full results stored as JSON files."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from asr_dashboard import paths
from asr_dashboard.models import TranscriptionResult


@dataclass
class HistoryEntry:
    id: str
    file: str
    engine: str
    created_at: float
    status: str  # done / failed / cancelled
    elapsed: float = 0.0
    batch_id: str | None = None
    result_path: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""
    kind: str = "file"  # file / realtime


class HistoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.history_file()
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []

    def _write(self, data: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def add(self, file: str, engine: str, status: str, elapsed: float = 0.0, result: TranscriptionResult | None = None,
            outputs: dict | None = None, batch_id: str | None = None, error: str = "", kind: str = "file") -> HistoryEntry:
        eid = uuid.uuid4().hex[:12]
        rp = ""
        if result is not None:
            p = paths.results_dir() / f"{eid}.json"
            p.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
            rp = str(p)
        e = HistoryEntry(eid, file, engine, time.time(), status, elapsed, batch_id, rp,
                         {k: str(v) for k, v in (outputs or {}).items()}, error, kind)
        with self._lock:
            data = self._read()
            data.insert(0, asdict(e))
            self._write(data[:2000])
        return e

    def list(self) -> list[HistoryEntry]:
        with self._lock:
            return [HistoryEntry(**{k: v for k, v in d.items() if k in HistoryEntry.__dataclass_fields__})
                    for d in self._read()]

    def load_result(self, entry: HistoryEntry) -> TranscriptionResult | None:
        if not entry.result_path or not Path(entry.result_path).exists():
            return None
        return TranscriptionResult.from_dict(json.loads(Path(entry.result_path).read_text(encoding="utf-8")))

    def update_result(self, entry_id: str, result: TranscriptionResult) -> None:
        for e in self.list():
            if e.id == entry_id and e.result_path:
                Path(e.result_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")

    def delete(self, entry_id: str) -> None:
        with self._lock:
            data = self._read()
            keep = []
            for d in data:
                if d["id"] == entry_id:
                    if d.get("result_path"):
                        Path(d["result_path"]).unlink(missing_ok=True)
                else:
                    keep.append(d)
            self._write(keep)
