"""Write results while recognition is still running, so a crash never loses finished chunks.

TXT and SRT are appended directly; JSON is written as JSON Lines to a temp file and merged at the end.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

from asr_dashboard.export.writers import ExportOptions, fmt_clock, srt_cue, unique_path
from asr_dashboard.models import Segment, TranscriptionResult


class StreamingWriter:
    def __init__(self, out_dir: str | Path, stem: str, formats: list[str], opts: ExportOptions,
                 overwrite: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.opts = opts
        self.formats = [f for f in formats if f in ("txt", "srt", "json")]
        self._lock = threading.Lock()
        self._cue = 0
        self.paths: dict[str, Path] = {}
        for f in self.formats:
            p = self.out_dir / f"{stem}.{f}"
            self.paths[f] = p if overwrite else unique_path(p)
        self.jsonl = self.paths["json"].with_suffix(".jsonl.part") if "json" in self.paths else None
        for f in ("txt", "srt"):
            if f in self.paths:
                self.paths[f].write_text("", encoding="utf-8")
        if self.jsonl:
            self.jsonl.write_text("", encoding="utf-8")

    def _append(self, path: Path, text: str) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    def write_segments(self, segments: list[Segment], names: dict[str, str] | None = None) -> None:
        names = names or {}
        with self._lock:
            for s in segments:
                if not s.text.strip():
                    continue
                name = names.get(s.speaker or "", s.speaker) if self.opts.include_speakers else None
                if "txt" in self.paths:
                    prefix = f"[{fmt_clock(s.start)}] " if self.opts.txt_timestamps else ""
                    if name:
                        prefix += f"{name}: "
                    self._append(self.paths["txt"], prefix + s.text.strip() + "\n")
                if "srt" in self.paths:
                    self._cue += 1
                    self._append(self.paths["srt"], srt_cue(self._cue, s, name))
                if self.jsonl:
                    self._append(self.jsonl, json.dumps(asdict(s), ensure_ascii=False) + "\n")

    def finalize(self, result: TranscriptionResult | None = None) -> dict[str, Path]:
        """Merge JSONL into the final JSON (full result if given)."""
        with self._lock:
            if self.jsonl and self.jsonl.exists():
                if result is not None:
                    data = result.to_dict()
                else:
                    segs = [json.loads(ln) for ln in self.jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    data = {"segments": segs, "partial": True}
                tmp = self.paths["json"].with_suffix(".json.part")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(self.paths["json"])
                self.jsonl.unlink(missing_ok=True)
        return dict(self.paths)

    def rewrite_final(self, result: TranscriptionResult) -> None:
        """After post-processing / diarization, replace the streamed drafts with the final content."""
        from asr_dashboard.export.writers import render

        with self._lock:
            for f in ("txt", "srt"):
                if f in self.paths:
                    tmp = self.paths[f].with_suffix(f".{f}.part")
                    tmp.write_text(render(result, f, self.opts) or "", encoding="utf-8")
                    tmp.replace(self.paths[f])
