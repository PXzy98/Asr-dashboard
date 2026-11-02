"""Voiceprint library: SQLite metadata + wav sample files under ``~/.asr-dashboard/voiceprints``.

Voiceprints never leave this machine (no API ever receives them).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asr_dashboard import paths
from asr_dashboard.core.audio import SAMPLE_RATE, save_wav

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, note TEXT DEFAULT '', created_at REAL, last_matched_at REAL
);
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    file TEXT NOT NULL, duration REAL, created_at REAL, source TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS embeddings (
    sample_id TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE, model_id TEXT NOT NULL,
    dim INTEGER, vector BLOB, PRIMARY KEY (sample_id, model_id)
);
"""


@dataclass
class Sample:
    id: str
    person_id: str
    file: Path
    duration: float
    created_at: float
    source: str = ""
    models: list[str] = field(default_factory=list)


@dataclass
class Person:
    id: str
    name: str
    note: str = ""
    created_at: float = 0.0
    last_matched_at: float | None = None
    samples: list[Sample] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.samples)


class VoiceprintStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else paths.voiceprint_dir()
        self.audio_dir = self.root / "samples"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "voiceprints.db"
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.execute("PRAGMA foreign_keys = ON")
        return c

    # ---- people -----------------------------------------------------------------
    def add_person(self, name: str, note: str = "") -> Person:
        name = name.strip()
        if not name:
            raise ValueError("名字不能为空")
        p = Person(uuid.uuid4().hex[:12], name, note, time.time())
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO people (id, name, note, created_at) VALUES (?,?,?,?)",
                      (p.id, p.name, p.note, p.created_at))
        return p

    def update_person(self, person_id: str, name: str | None = None, note: str | None = None) -> None:
        with self._lock, self._conn() as c:
            if name is not None:
                if not name.strip():
                    raise ValueError("名字不能为空")
                c.execute("UPDATE people SET name=? WHERE id=?", (name.strip(), person_id))
            if note is not None:
                c.execute("UPDATE people SET note=? WHERE id=?", (note, person_id))

    def delete_person(self, person_id: str) -> None:
        p = self.get_person(person_id)
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM people WHERE id=?", (person_id,))
        if p:
            for s in p.samples:
                s.file.unlink(missing_ok=True)

    def touch_matched(self, person_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE people SET last_matched_at=? WHERE id=?", (time.time(), person_id))

    def list_people(self) -> list[Person]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT id, name, note, created_at, last_matched_at FROM people ORDER BY name").fetchall()
            srows = c.execute("SELECT id, person_id, file, duration, created_at, source FROM samples "
                              "ORDER BY created_at").fetchall()
            erows = c.execute("SELECT sample_id, model_id FROM embeddings").fetchall()
        models: dict[str, list[str]] = {}
        for sid, mid in erows:
            models.setdefault(sid, []).append(mid)
        by_person: dict[str, list[Sample]] = {}
        for sid, pid, f, d, ca, src in srows:
            by_person.setdefault(pid, []).append(Sample(sid, pid, self.audio_dir / f, d or 0.0, ca or 0.0, src or "",
                                                        models.get(sid, [])))
        return [Person(i, n, nt or "", ca or 0.0, lm, by_person.get(i, [])) for i, n, nt, ca, lm in rows]

    def get_person(self, person_id: str) -> Person | None:
        return next((p for p in self.list_people() if p.id == person_id), None)

    def find_by_name(self, name: str) -> Person | None:
        return next((p for p in self.list_people() if p.name == name.strip()), None)

    # ---- samples ----------------------------------------------------------------
    def add_sample(self, person_id: str, samples: np.ndarray, source: str = "",
                   embedding: np.ndarray | None = None, model_id: str | None = None) -> Sample:
        sid = uuid.uuid4().hex[:12]
        fname = f"{person_id}_{sid}.wav"
        save_wav(self.audio_dir / fname, samples)
        dur = len(samples) / SAMPLE_RATE
        s = Sample(sid, person_id, self.audio_dir / fname, dur, time.time(), source)
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO samples (id, person_id, file, duration, created_at, source) VALUES (?,?,?,?,?,?)",
                      (sid, person_id, fname, dur, s.created_at, source))
        if embedding is not None and model_id:
            self.set_embedding(sid, model_id, embedding)
        return s

    def delete_sample(self, sample_id: str) -> None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT file FROM samples WHERE id=?", (sample_id,)).fetchone()
            c.execute("DELETE FROM samples WHERE id=?", (sample_id,))
        if row:
            (self.audio_dir / row[0]).unlink(missing_ok=True)

    def set_embedding(self, sample_id: str, model_id: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO embeddings (sample_id, model_id, dim, vector) VALUES (?,?,?,?)",
                      (sample_id, model_id, len(v), v.tobytes()))

    def sample_embeddings(self, model_id: str) -> dict[str, list[np.ndarray]]:
        """person_id -> list of sample vectors for ``model_id``."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT s.person_id, e.vector FROM embeddings e JOIN samples s ON s.id = e.sample_id "
                "WHERE e.model_id=?", (model_id,)).fetchall()
        out: dict[str, list[np.ndarray]] = {}
        for pid, blob in rows:
            out.setdefault(pid, []).append(np.frombuffer(blob, dtype=np.float32).copy())
        return out

    def centroids(self, model_id: str) -> list[tuple[str, str, np.ndarray]]:
        names = {p.id: p.name for p in self.list_people()}
        out = []
        for pid, vecs in self.sample_embeddings(model_id).items():
            if pid not in names or not vecs:
                continue
            m = np.mean([v / max(np.linalg.norm(v), 1e-9) for v in vecs], axis=0)
            out.append((pid, names[pid], (m / max(np.linalg.norm(m), 1e-9)).astype(np.float32)))
        return out

    def missing_embeddings(self, model_id: str) -> list[Sample]:
        return [s for p in self.list_people() for s in p.samples if model_id not in s.models]

    def recompute(self, model_id: str, embed_fn, progress=None, only_missing: bool = False) -> int:
        """(Re)compute vectors for all samples with ``embed_fn(np.ndarray) -> vector``."""
        from asr_dashboard.core.audio import load_wav

        todo = self.missing_embeddings(model_id) if only_missing else [s for p in self.list_people() for s in p.samples]
        for i, s in enumerate(todo):
            self.set_embedding(s.id, model_id, embed_fn(load_wav(s.file)))
            if progress:
                progress((i + 1) / len(todo), f"重新计算声纹 {i + 1}/{len(todo)}")
        return len(todo)

    # ---- import / export ----------------------------------------------------------
    def export_zip(self, dest: str | Path) -> Path:
        dest = Path(dest)
        people = self.list_people()
        with self._lock, self._conn() as c:
            emb_rows = c.execute("SELECT sample_id, model_id, dim, vector FROM embeddings").fetchall()
        manifest = {
            "format": "asr-dashboard-voiceprints", "version": 1, "exported_at": time.time(),
            "people": [{"id": p.id, "name": p.name, "note": p.note, "created_at": p.created_at,
                        "samples": [{"id": s.id, "file": s.file.name, "duration": s.duration,
                                     "created_at": s.created_at, "source": s.source} for s in p.samples]}
                       for p in people],
        }
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            z.writestr("embeddings.json", json.dumps(
                [{"sample_id": s, "model_id": m, "vector": np.frombuffer(v, np.float32).tolist()}
                 for s, m, _d, v in emb_rows]))
            for p in people:
                for s in p.samples:
                    if s.file.exists():
                        z.write(s.file, f"samples/{s.file.name}")
        return dest

    def import_zip(self, src: str | Path, merge_same_name: bool = True) -> int:
        """Import people/samples from a zip produced by ``export_zip``. Returns number of people imported."""
        with zipfile.ZipFile(src) as z:
            manifest = json.loads(z.read("manifest.json"))
            if manifest.get("format") != "asr-dashboard-voiceprints":
                raise ValueError("不是声纹库导出文件")
            embs = json.loads(z.read("embeddings.json")) if "embeddings.json" in z.namelist() else []
            by_sample: dict[str, list[tuple[str, list[float]]]] = {}
            for e in embs:
                by_sample.setdefault(e["sample_id"], []).append((e["model_id"], e["vector"]))
            count = 0
            for pdata in manifest.get("people", []):
                person = self.find_by_name(pdata["name"]) if merge_same_name else None
                if person is None:
                    person = self.add_person(pdata["name"], pdata.get("note", ""))
                    count += 1
                for sdata in pdata.get("samples", []):
                    member = f"samples/{sdata['file']}"
                    if member not in z.namelist():
                        continue
                    sid = uuid.uuid4().hex[:12]
                    fname = f"{person.id}_{sid}.wav"
                    with z.open(member) as fin, open(self.audio_dir / fname, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    with self._lock, self._conn() as c:
                        c.execute("INSERT INTO samples (id, person_id, file, duration, created_at, source) "
                                  "VALUES (?,?,?,?,?,?)", (sid, person.id, fname, sdata.get("duration", 0.0),
                                                           sdata.get("created_at", time.time()), "import"))
                    for model_id, vec in by_sample.get(sdata["id"], []):
                        self.set_embedding(sid, model_id, np.asarray(vec, np.float32))
        return count
