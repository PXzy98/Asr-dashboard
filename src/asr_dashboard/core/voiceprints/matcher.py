"""One-to-one speaker <-> person matching (Hungarian algorithm) and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    label: str
    person_id: str
    name: str
    score: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def similarity_matrix(speakers: dict[str, np.ndarray], people: list[tuple[str, str, np.ndarray]]):
    labels = list(speakers)
    sim = np.zeros((len(labels), len(people)))
    for i, lab in enumerate(labels):
        for j, (_pid, _name, cen) in enumerate(people):
            sim[i, j] = cosine(speakers[lab], cen) if len(speakers[lab]) == len(cen) else -1.0
    return labels, sim


def match_speakers(speakers: dict[str, np.ndarray], people: list[tuple[str, str, np.ndarray]],
                   threshold: float) -> dict[str, Match]:
    """Assign each diarized speaker at most one person (and vice versa) maximising total similarity."""
    if not speakers or not people:
        return {}
    from scipy.optimize import linear_sum_assignment

    labels, sim = similarity_matrix(speakers, people)
    rows, cols = linear_sum_assignment(-sim)
    out: dict[str, Match] = {}
    for r, c in zip(rows, cols, strict=True):
        score = float(sim[r, c])
        if score >= threshold:
            pid, name, _ = people[c]
            out[labels[r]] = Match(labels[r], pid, name, score)
    return out


def best_scores(speakers: dict[str, np.ndarray], people) -> dict[str, float]:
    """Highest similarity per speaker (shown in UI even when below threshold)."""
    if not speakers or not people:
        return {}
    labels, sim = similarity_matrix(speakers, people)
    return {lab: float(sim[i].max()) for i, lab in enumerate(labels)}


@dataclass
class Calibration:
    threshold: float | None
    intra_mean: float | None
    inter_max: float | None
    message: str


def calibrate(per_person: dict[str, list[np.ndarray]]) -> Calibration:
    """Recommend a threshold from same-person vs different-person similarity distributions."""
    intra: list[float] = []
    inter: list[float] = []
    cents = {}
    for pid, vecs in per_person.items():
        vecs = [v / max(np.linalg.norm(v), 1e-9) for v in vecs]
        if len(vecs) >= 2:
            for i in range(len(vecs)):
                rest = np.mean([v for j, v in enumerate(vecs) if j != i], axis=0)
                intra.append(cosine(vecs[i], rest))
        if vecs:
            cents[pid] = (np.mean(vecs, axis=0), vecs)
    pids = list(cents)
    for i in range(len(pids)):
        for j in range(len(pids)):
            if i != j:
                for v in cents[pids[i]][1]:
                    inter.append(cosine(v, cents[pids[j]][0]))
    if not inter:
        return Calibration(None, None, None, "至少需要 2 个人的样本才能校准阈值")
    inter_max = float(np.percentile(inter, 95))
    if not intra:
        thr = float(np.clip(inter_max + 0.1, 0.3, 0.95))
        return Calibration(thr, None, inter_max,
                           f"每人只有 1 个样本，按不同人之间相似度（95 分位 {inter_max:.2f}）上浮 0.1 推荐 {thr:.2f}")
    intra_low = float(np.percentile(intra, 10))
    thr = (inter_max + intra_low) / 2 if intra_low > inter_max else inter_max + 0.02
    thr = float(np.clip(thr, 0.3, 0.95))
    return Calibration(thr, float(np.mean(intra)), inter_max,
                       f"同一人相似度 10 分位 {intra_low:.2f}，不同人 95 分位 {inter_max:.2f}，推荐阈值 {thr:.2f}")
