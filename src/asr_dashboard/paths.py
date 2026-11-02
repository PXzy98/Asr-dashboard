"""Well-known locations. ``ASR_DASHBOARD_HOME`` overrides the per-user data dir (used by tests)."""

from __future__ import annotations

import os
from pathlib import Path


def app_home() -> Path:
    env = os.environ.get("ASR_DASHBOARD_HOME")
    p = Path(env) if env else Path.home() / ".asr-dashboard"
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    env = os.environ.get("ASR_DASHBOARD_CONFIG")
    if env:
        return Path(env)
    return project_root() / "config.yaml"


def secrets_file() -> Path:
    return app_home() / "secrets.yaml"


def history_file() -> Path:
    return app_home() / "history.json"


def voiceprint_dir() -> Path:
    p = app_home() / "voiceprints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def envs_dir() -> Path:
    p = app_home() / "envs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = app_home() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resources_dir() -> Path:
    return Path(__file__).resolve().parent / "resources"
