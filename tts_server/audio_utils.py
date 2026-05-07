"""Audio processing utilities — crossfade, WAV encoding, numpy helpers."""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# WAV ↔ numpy
# ---------------------------------------------------------------------------

def wav_bytes_to_np(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode in-memory WAV bytes → (float32 1-D array, sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    return audio, sr


def np_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode float32 waveform → in-memory WAV bytes (PCM-16)."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def b64_encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Crossfade — linear overlap-add between consecutive TTS chunks
# ---------------------------------------------------------------------------

class Crossfader:
    """Server-side overlap-add crossfade between successive TTS chunks.

    Algorithm per chunk
    ───────────────────
    1. **First chunk** → linear ramp-up on the first ``n`` samples.
    2. **Middle chunk** → blend held tail (fade-out) with new head (fade-in).
    3. **Last chunk**  → blend + linear ramp-down on the tail.
    4. ``n = sr * crossfade_ms / 1000``, capped at 1/3 of chunk length.

    If ``crossfade_ms == 0`` the crossfader is a transparent pass-through.
    """

    __slots__ = ("sr", "ms", "_n", "_tail")

    def __init__(self, sr: int, crossfade_ms: int) -> None:
        self.sr = sr
        self.ms = crossfade_ms
        self._n: int = max(1, int(sr * crossfade_ms / 1000)) if crossfade_ms > 0 else 0
        self._tail: Optional[np.ndarray] = None

    def process(
        self,
        audio: np.ndarray,
        *,
        is_first: bool,
        is_last: bool,
    ) -> np.ndarray:
        """Process one raw synthesis chunk.  Returns audio ready to stream."""
        if self._n == 0 or len(audio) == 0:
            return audio

        n = min(self._n, max(1, len(audio) // 3))
        parts: list[np.ndarray] = []

        if self._tail is not None:
            tn = min(n, len(self._tail))
            ro = np.linspace(1.0, 0.0, tn, dtype=np.float32)
            ri = np.linspace(0.0, 1.0, tn, dtype=np.float32)
            parts.append(self._tail[-tn:] * ro + audio[:tn] * ri)
            body = audio[tn:]
        else:
            head = audio[:n].copy()
            head *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            parts.append(head)
            body = audio[n:]

        if is_last:
            if len(body) > n:
                parts.append(body[:-n])
                fade_end = body[-n:].copy()
                fade_end *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                parts.append(fade_end)
            else:
                out = body.copy()
                out *= np.linspace(1.0, 0.0, max(1, len(body)), dtype=np.float32)
                parts.append(out)
            self._tail = None
        else:
            if len(body) > n:
                parts.append(body[:-n])
                self._tail = body[-n:].copy()
            else:
                parts.append(body)
                self._tail = None

        return np.concatenate(parts) if parts else audio

    def flush(self) -> Optional[np.ndarray]:
        """Return remaining held tail with fade-out.  Call once after the last chunk."""
        if self._tail is None or self._n == 0:
            return None
        tail = self._tail.copy()
        tail *= np.linspace(1.0, 0.0, len(tail), dtype=np.float32)
        self._tail = None
        return tail


def crossfade_stitch(chunks: list[np.ndarray], sr: int, crossfade_ms: int) -> np.ndarray:
    """Stitch a list of audio chunks with crossfade overlap into one waveform."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    xf = Crossfader(sr, crossfade_ms)
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        out = xf.process(
            chunk,
            is_first=(i == 0),
            is_last=(i == len(chunks) - 1),
        )
        parts.append(out)

    tail = xf.flush()
    if tail is not None:
        parts.append(tail)

    return np.concatenate(parts)
