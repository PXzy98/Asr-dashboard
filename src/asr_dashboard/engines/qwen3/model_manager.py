"""Download / inspect / delete Qwen3 model weights (HuggingFace or ModelScope caches)."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from asr_dashboard.engines.qwen3.common import ALIGNER_REPO, REPO_IDS

log = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]  # (0..1, message)


@dataclass
class CachedModel:
    repo_id: str
    path: str
    size_bytes: int
    source: str

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1024**3


def repos_for(variant: str, backend: str, with_aligner: bool) -> list[str]:
    native = backend == "transformers"
    repos = [REPO_IDS[(variant, native)]]
    if with_aligner:
        repos.append(ALIGNER_REPO[native])
    return repos


def _tqdm_factory(cb: ProgressCb, label: str):
    from tqdm.auto import tqdm as _base

    class _Tqdm(_base):  # type: ignore[misc]
        def update(self, n=1):
            r = super().update(n)
            if self.total:
                cb(min(1.0, self.n / self.total), f"{label}：{self.n / 2**20:.0f}/{self.total / 2**20:.0f} MB")
            return r

    return _Tqdm


def hf_download(repo_id: str, cache_dir: str | None, cb: ProgressCb | None = None, token: str | None = None,
                local_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    kw: dict = {"repo_id": repo_id, "cache_dir": cache_dir or None, "token": token or None}
    if local_dir:
        kw["local_dir"] = local_dir
    if cb:
        try:
            kw["tqdm_class"] = _tqdm_factory(cb, repo_id)
        except Exception:  # noqa: BLE001
            pass
    return snapshot_download(**kw)


def modelscope_download(repo_id: str, cache_dir: str | None, cb: ProgressCb | None = None) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError("模型来源为 ModelScope，需要安装 modelscope：uv pip install modelscope") from e
    if cb:
        cb(0.0, f"ModelScope 下载 {repo_id}…")
    path = snapshot_download(repo_id, cache_dir=cache_dir or None)
    if cb:
        cb(1.0, f"{repo_id} 已就绪")
    return path


def download(variant: str, backend: str, source: str, with_aligner: bool, cache_dir: str | None,
             cb: ProgressCb | None = None) -> list[str]:
    out = []
    for repo in repos_for(variant, backend, with_aligner):
        if source == "modelscope":
            out.append(modelscope_download(repo, cache_dir, cb))
        else:
            out.append(hf_download(repo, cache_dir, cb))
    return out


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def list_cached(cache_dir: str | None = None) -> list[CachedModel]:
    found: list[CachedModel] = []
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(cache_dir or None)
        for repo in info.repos:
            if repo.repo_id.startswith(("Qwen/Qwen3-ASR", "Qwen/Qwen3-ForcedAligner", "pyannote/", "microsoft/wavlm")):
                found.append(CachedModel(repo.repo_id, str(repo.repo_path), repo.size_on_disk, "huggingface"))
    except Exception as e:  # noqa: BLE001
        log.debug("scan_cache_dir: %s", e)
    try:
        import os

        ms_root = Path(cache_dir or os.environ.get("MODELSCOPE_CACHE") or Path.home() / ".cache" / "modelscope" / "hub")
        for d in list(ms_root.glob("Qwen/Qwen3-*")) + list(ms_root.glob("models/Qwen/Qwen3-*")):
            if d.is_dir():
                found.append(CachedModel(f"Qwen/{d.name}", str(d), _dir_size(d), "modelscope"))
    except Exception:  # noqa: BLE001
        pass
    return found


def delete_cached(model: CachedModel) -> None:
    if model.source == "huggingface":
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(str(Path(model.path).parent))
        hashes = [rev.commit_hash for repo in info.repos if repo.repo_id == model.repo_id for rev in repo.revisions]
        if hashes:
            info.delete_revisions(*hashes).execute()
            return
    shutil.rmtree(model.path, ignore_errors=True)
