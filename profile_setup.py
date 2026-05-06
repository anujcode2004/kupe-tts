from __future__ import annotations

import json
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from omni_infer import PROFILES_DIR


def list_profiles() -> List[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted([p.name for p in PROFILES_DIR.iterdir() if p.is_dir()])


def sanitize_profile_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _write_placeholder_ref_wav(path: Path, duration_sec: float = 1.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = max(1, int(sr * duration_sec))
    silence = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(silence.tobytes())


def ensure_profile(profile_name: str) -> tuple[str, bool]:
    """
    Ensure profiles/<name>/ exists with meta.json and ref_audio.wav.
    Returns (canonical directory name, created_new_profile).
    """
    raw = (profile_name or "").strip()
    if not raw:
        existing = list_profiles()
        if existing:
            return existing[0], False
        raw = "default"

    name = sanitize_profile_name(raw)
    if not name:
        name = "default"

    profile_dir = PROFILES_DIR / name
    created_new = not profile_dir.exists()
    profile_dir.mkdir(parents=True, exist_ok=True)

    meta_path = profile_dir / "meta.json"
    if not meta_path.exists():
        meta = {
            "name": name,
            "ref_text": "Hello, this is an auto-generated profile for voice cloning.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        created_new = True

    ref_path = profile_dir / "ref_audio.wav"
    if not ref_path.exists() or ref_path.stat().st_size < 44:
        _write_placeholder_ref_wav(ref_path)
        created_new = True

    return name, created_new


def resolve_profile_name(profile_name: str | None) -> str:
    """
    Resolve dropdown / stale input to a valid profile directory, creating it if needed.
    """
    profiles = list_profiles()
    if profile_name and str(profile_name).strip():
        requested = str(profile_name).strip()
        if requested in profiles:
            return requested
        name, _ = ensure_profile(requested)
        return name

    if profiles:
        return profiles[0]
    name, _ = ensure_profile("default")
    return name
