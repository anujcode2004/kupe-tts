"""Voice profile system — JSON descriptor + cached numpy embedding.

A *voice profile* is a small JSON file in ``voice_reference/`` that points to:
  - a reference audio clip (used once to build the voice-clone prompt)
  - a reference transcript
  - (auto-populated) a numpy ``.npz`` embedding cache

On server startup the profile is resolved exactly once.  If the cache exists,
the worker process skips the audio-tokeniser step entirely and reconstructs
the :class:`omnivoice.models.omnivoice.VoiceClonePrompt` from numpy arrays.
This shaves ~200-500 ms off cold start and removes the audio-encoder load.

Profile JSON schema
───────────────────
::

    {
      "name":          "soham",
      "ref_text":      "Hello, my name is Soham how are you?",
      "ref_audio":     "voice_reference/soham_ref.wav",  // path or URL
      "language":      "en",                              // optional default lang
      "embedding_path": "voice_reference/soham_embedding.npz",  // auto-set
      "embedding_meta": {                                 // auto-set
        "model_id":       "k2-fsa/OmniVoice",
        "sampling_rate":  24000,
        "num_codebooks":  9,
        "num_tokens":     438,
        "ref_rms":        0.124,
        "built_at":       "2026-05-07T14:30:00+00:00"
      },
      "created_at":    "2026-05-04T10:39:45+00:00"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("omnivoice.profiles")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR    = PROJECT_ROOT / "voice_reference"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VoiceEmbedding:
    """In-memory representation of a cached voice-clone embedding."""

    ref_audio_tokens: np.ndarray   # (num_codebooks, num_tokens)  int / long
    ref_text:         str
    ref_rms:          float
    sampling_rate:    int
    model_id:         str
    num_codebooks:    int
    num_tokens:       int

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            ref_audio_tokens=self.ref_audio_tokens.astype(np.int64),
            ref_text=np.array(self.ref_text, dtype=object),
            ref_rms=np.float32(self.ref_rms),
            sampling_rate=np.int32(self.sampling_rate),
            model_id=np.array(self.model_id, dtype=object),
            num_codebooks=np.int32(self.num_codebooks),
            num_tokens=np.int32(self.num_tokens),
        )

    @classmethod
    def from_npz(cls, path: Path) -> "VoiceEmbedding":
        with np.load(str(path), allow_pickle=True) as data:
            return cls(
                ref_audio_tokens=data["ref_audio_tokens"].astype(np.int64),
                ref_text=str(data["ref_text"].item()),
                ref_rms=float(data["ref_rms"]),
                sampling_rate=int(data["sampling_rate"]),
                model_id=str(data["model_id"].item()),
                num_codebooks=int(data["num_codebooks"]),
                num_tokens=int(data["num_tokens"]),
            )


@dataclass
class VoiceProfile:
    """Profile descriptor backed by a JSON file."""

    name:           str
    ref_text:       str
    ref_audio:      str
    json_path:      Path
    language:       str = "en"
    embedding_path: Optional[str] = None
    embedding_meta: dict[str, Any] = field(default_factory=dict)
    created_at:     Optional[str]  = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolve_ref_audio(self) -> Path:
        """Return the absolute path to the reference audio clip.

        Tolerates several path styles:
          1. Absolute path
          2. Path relative to the JSON file
          3. Path relative to the project root (handles paths that include
             a leading ``OmniVoice/`` prefix)
          4. Bare filename → assumed to live next to the JSON file
        """
        candidate = Path(self.ref_audio)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Relative to JSON's directory
        rel_to_json = (self.json_path.parent / candidate).resolve()
        if rel_to_json.exists():
            return rel_to_json

        # Relative to project root
        rel_to_root = (PROJECT_ROOT.parent / candidate).resolve()
        if rel_to_root.exists():
            return rel_to_root

        rel_to_root2 = (PROJECT_ROOT / candidate).resolve()
        if rel_to_root2.exists():
            return rel_to_root2

        # Strip leading "OmniVoice/" if present and try project root
        if candidate.parts and candidate.parts[0] == "OmniVoice":
            stripped = Path(*candidate.parts[1:])
            for base in (PROJECT_ROOT, PROJECT_ROOT.parent):
                p = (base / stripped).resolve()
                if p.exists():
                    return p

        # Bare filename next to json
        bare = self.json_path.parent / candidate.name
        if bare.exists():
            return bare

        raise FileNotFoundError(
            f"Cannot resolve ref_audio '{self.ref_audio}' for profile '{self.name}'. "
            f"Tried absolute, JSON-relative, project-root, and 'OmniVoice/' prefixes."
        )

    def resolve_embedding_path(self) -> Path:
        """Path where the npz embedding is (or will be) stored."""
        if self.embedding_path:
            p = Path(self.embedding_path)
            if not p.is_absolute():
                p = (self.json_path.parent / p.name).resolve()
            return p
        return (self.json_path.parent / f"{self.name}_embedding.npz").resolve()

    def has_cached_embedding(self) -> bool:
        return self.resolve_embedding_path().exists()

    def load_cached_embedding(self) -> VoiceEmbedding:
        return VoiceEmbedding.from_npz(self.resolve_embedding_path())

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> "VoiceProfile":
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            name=data["name"],
            ref_text=data["ref_text"].strip(),
            ref_audio=data["ref_audio"],
            json_path=path,
            language=data.get("language", "en"),
            embedding_path=data.get("embedding_path"),
            embedding_meta=data.get("embedding_meta", {}),
            created_at=data.get("created_at"),
        )

    def save_json(self) -> None:
        """Persist (potentially-mutated) profile data back to disk."""
        payload: dict[str, Any] = {
            "name":           self.name,
            "ref_text":       self.ref_text,
            "ref_audio":      self.ref_audio,
            "language":       self.language,
            "embedding_path": self.embedding_path,
            "embedding_meta": self.embedding_meta,
            "created_at":     self.created_at or datetime.now(timezone.utc).isoformat(),
        }
        with self.json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    def update_embedding_metadata(self, embedding: VoiceEmbedding) -> None:
        """Update the JSON descriptor to point at the cached npz file."""
        self.embedding_path = self.resolve_embedding_path().name
        self.embedding_meta = {
            "model_id":       embedding.model_id,
            "sampling_rate":  embedding.sampling_rate,
            "num_codebooks":  embedding.num_codebooks,
            "num_tokens":     embedding.num_tokens,
            "ref_rms":        round(float(embedding.ref_rms), 6),
            "built_at":       datetime.now(timezone.utc).isoformat(),
        }
        self.save_json()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_profile_json(name: str, search_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate a profile JSON by name.  Tries ``<name>_ref.json`` and ``<name>.json``."""
    base = search_dir or VOICE_DIR
    for fname in (f"{name}_ref.json", f"{name}.json"):
        p = base / fname
        if p.exists():
            return p
    return None


def list_profiles(search_dir: Optional[Path] = None) -> list[str]:
    """Return all profile names available in the voice_reference directory."""
    base = search_dir or VOICE_DIR
    if not base.exists():
        return []
    names: set[str] = set()
    for f in base.glob("*.json"):
        try:
            with f.open("r", encoding="utf-8") as fh:
                names.add(json.load(fh).get("name", f.stem))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(names)


def load_profile_by_name(
    name: str,
    search_dir: Optional[Path] = None,
) -> VoiceProfile:
    """Load a profile by name from the voice_reference directory."""
    json_path = find_profile_json(name, search_dir)
    if json_path is None:
        available = list_profiles(search_dir)
        raise FileNotFoundError(
            f"Voice profile '{name}' not found in {search_dir or VOICE_DIR}. "
            f"Available profiles: {available or '(none)'}"
        )
    profile = VoiceProfile.from_json(json_path)
    logger.info(
        "Voice profile loaded: name=%s  ref_audio=%s  cached=%s",
        profile.name,
        profile.resolve_ref_audio(),
        profile.has_cached_embedding(),
    )
    return profile
