"""WS /ws/tts — Streaming TTS with chunked diffusion + pipelined first chunk.

Each WebSocket message is a JSON object::

    {
      "type":     "tts.request",
      "text":     "<your text>",
      "language": "en",        // optional
      "voice":    "ajay"       // optional — must be preloaded
    }

``language`` is optional (defaults to the server's ``OMNIVOICE_LANGUAGE``
setting).  ``voice`` is optional (defaults to ``OMNIVOICE_DEFAULT_VOICE``)
and must be one of the profiles preloaded via ``OMNIVOICE_VOICE_PROFILES``.

Latency strategy (see top-level docs)
─────────────────────────────────────
1. **Aggressive first-chunk text split** (~25 chars).
2. **Few diffusion steps for the first chunk** (FIRST_CHUNK_STEPS, default 4).
3. **Bypass the batch timeout** for chunk 0 via ``submit_immediate``.
4. **Pipelined generation**: while chunk N streams, chunk N+1 is generating.
5. **Crossfade overlap** hides any audible mismatch between the fast first
   chunk and the higher-quality chunks that follow.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..audio_utils import Crossfader, b64_encode, np_to_wav_bytes, wav_bytes_to_np
from ..config import (
    CROSSFADE_MS,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEED,
    FIRST_CHUNK_CFG,
    LAST_CHUNK_CFG,
    MID_CHUNK_CFG,
    SPEED_MAX,
    SPEED_MIN,
)
from ..lang_utils import resolve_language
from ..text_utils import split_first_chunk_early, split_to_chunks


def _coerce_text(raw) -> str:
    """Robustly convert any ``msg["text"]`` payload into a single string.

    Accepts plain strings, lists/tuples (joined with spaces), or coerces other
    types via ``str(...)``.  Returns the trimmed string, or ``""`` on None.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (list, tuple)):
        return " ".join(str(part).strip() for part in raw if part is not None).strip()
    return str(raw).strip()


def _coerce_speed(raw) -> tuple[Optional[float], Optional[str]]:
    """Validate a user-supplied speed value.

    Returns ``(speed, error_message)``.  ``speed`` is ``None`` when the user
    didn't provide a value (server falls back to DEFAULT_SPEED).
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "default", "none", "auto"):
            return None, None
        try:
            raw = float(s)
        except ValueError:
            return None, f"speed must be a number, got {raw!r}"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None, f"speed must be a number, got {raw!r}"
    if not (SPEED_MIN <= v <= SPEED_MAX):
        return None, f"speed {v} is out of range [{SPEED_MIN}, {SPEED_MAX}]"
    return v, None

logger = logging.getLogger("omnivoice.streaming")

router = APIRouter()


@router.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS connected: %s", websocket.client)
    batcher = websocket.app.state.batcher
    sample_rate: int = getattr(websocket.app.state, "sample_rate", 24_000)
    available_voices: dict = getattr(websocket.app.state, "voice_profiles", {})
    default_voice:    str  = getattr(websocket.app.state, "default_voice", "")

    try:
        while True:
            # ──────────────────────────────────────────────────────────
            # 1. Receive request
            # ──────────────────────────────────────────────────────────
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break

            mt = msg.get("type", "")
            if mt == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if mt not in ("tts.request", ""):
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown type '{mt}'. Send tts.request.",
                })
                continue

            text = _coerce_text(msg.get("text"))
            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text."})
                continue

            # Language: client sends ISO code, name, or 'auto' (None = auto).
            raw_lang = msg.get("language")
            if raw_lang is None or (isinstance(raw_lang, str) and not raw_lang.strip()):
                raw_lang = DEFAULT_LANGUAGE
            language = resolve_language(raw_lang)

            raw_voice = msg.get("voice")
            voice = (str(raw_voice).strip() if raw_voice else "") or default_voice
            if available_voices and voice not in available_voices:
                await websocket.send_json({
                    "type": "error",
                    "message": (
                        f"Voice '{voice}' not loaded. Available: "
                        f"{sorted(available_voices.keys())}"
                    ),
                })
                continue

            speed, speed_err = _coerce_speed(msg.get("speed"))
            if speed_err:
                await websocket.send_json({"type": "error", "message": speed_err})
                continue
            if speed is None:
                speed = DEFAULT_SPEED
            logger.info(
                "WS TTS request  voice=%s  lang=%s  speed=%s  text=%.80s",
                voice, language or "auto",
                f"{speed:.2f}" if speed is not None else "default",
                text,
            )

            # ──────────────────────────────────────────────────────────
            # 2. Split: aggressive short first chunk + remainder
            # ──────────────────────────────────────────────────────────
            first_text, remainder = split_first_chunk_early(text)
            rest_chunks = split_to_chunks(remainder) if remainder else []
            all_chunks  = [first_text] + rest_chunks
            n           = len(all_chunks)
            logger.info(
                "Split %d chunk(s)  first=%d chars  rest=%d chunks",
                n, len(first_text), len(rest_chunks),
            )

            t_req = time.perf_counter()
            first_latency_ms: Optional[float] = None
            total_samples = 0
            xfader = Crossfader(sample_rate, CROSSFADE_MS)

            # ──────────────────────────────────────────────────────────
            # 3. PIPELINE: generate first chunk IMMEDIATELY (bypass batcher)
            #    while pre-submitting chunk 1 to the batcher
            # ──────────────────────────────────────────────────────────
            prefetch_task: Optional[asyncio.Task] = None

            if n > 1:
                cfg_1 = LAST_CHUNK_CFG if n == 2 else MID_CHUNK_CFG
                prefetch_task = asyncio.create_task(
                    batcher.submit(
                        all_chunks[1], cfg_1,
                        language=language, voice=voice, speed=speed,
                    )
                )

            try:
                first_wav, first_gen_ms = await batcher.submit_immediate(
                    first_text, FIRST_CHUNK_CFG,
                    language=language, voice=voice, speed=speed,
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                if prefetch_task:
                    prefetch_task.cancel()
                continue

            first_audio, _ = wav_bytes_to_np(first_wav)
            first_audio = xfader.process(first_audio, is_first=True, is_last=(n == 1))
            total_samples += len(first_audio)
            first_latency_ms = (time.perf_counter() - t_req) * 1000.0

            logger.info(
                "FIRST CHUNK  latency=%.1fms  gen=%.1fms  audio=%.0fms  text='%s'",
                first_latency_ms, first_gen_ms,
                len(first_audio) / sample_rate * 1000, first_text,
            )

            wav_out = np_to_wav_bytes(first_audio, sample_rate)
            await websocket.send_json({
                "type":                  "response.audio.delta",
                "delta":                 b64_encode(wav_out),
                "encoding":              "wav/pcm16",
                "sample_rate":           sample_rate,
                "chunk_index":           0,
                "chunk_text":            first_text,
                "chunk_audio_ms":        round(len(first_audio) / sample_rate * 1000),
                "chunk_gen_ms":          round(first_gen_ms, 1),
                "language":              language or "auto",
                "voice":                 voice,
                "speed":                 speed,
                "first_chunk_latency_ms": round(first_latency_ms, 1),
            })

            # ──────────────────────────────────────────────────────────
            # 4. Remaining chunks — pipelined
            # ──────────────────────────────────────────────────────────
            for i in range(1, n):
                is_last = (i == n - 1)

                if prefetch_task is not None:
                    try:
                        wav_bytes = await prefetch_task
                    except Exception as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        break
                else:
                    cfg_i = LAST_CHUNK_CFG if is_last else MID_CHUNK_CFG
                    wav_bytes = await batcher.submit(
                        all_chunks[i], cfg_i,
                        language=language, voice=voice, speed=speed,
                    )

                prefetch_task = None
                if i + 1 < n:
                    cfg_next = LAST_CHUNK_CFG if (i + 1 == n - 1) else MID_CHUNK_CFG
                    prefetch_task = asyncio.create_task(
                        batcher.submit(
                            all_chunks[i + 1], cfg_next,
                            language=language, voice=voice, speed=speed,
                        )
                    )

                t_chunk = time.perf_counter()
                audio, sr = wav_bytes_to_np(wav_bytes)
                audio = xfader.process(audio, is_first=False, is_last=is_last)
                total_samples += len(audio)
                chunk_gen_ms = (time.perf_counter() - t_chunk) * 1000.0

                wav_out = np_to_wav_bytes(audio, sr or sample_rate)
                await websocket.send_json({
                    "type":           "response.audio.delta",
                    "delta":          b64_encode(wav_out),
                    "encoding":       "wav/pcm16",
                    "sample_rate":    sr or sample_rate,
                    "chunk_index":    i,
                    "chunk_text":     all_chunks[i],
                    "chunk_audio_ms": round(len(audio) / (sr or sample_rate) * 1000),
                    "chunk_gen_ms":   round(chunk_gen_ms, 1),
                    "language":       language or "auto",
                    "voice":          voice,
                    "speed":          speed,
                })

            tail = xfader.flush()
            if tail is not None:
                total_samples += len(tail)

            # ──────────────────────────────────────────────────────────
            # 5. Done
            # ──────────────────────────────────────────────────────────
            total_audio_ms = round(total_samples / sample_rate * 1000)
            total_wall_ms  = round((time.perf_counter() - t_req) * 1000, 1)

            await websocket.send_json({
                "type":                   "response.audio.done",
                "total_chunks":           n,
                "total_audio_ms":         total_audio_ms,
                "total_gen_ms":           total_wall_ms,
                "first_chunk_latency_ms": round(first_latency_ms or 0, 1),
                "language":               language or "auto",
                "voice":                  voice,
                "speed":                  speed,
            })
            logger.info(
                "Done.  chunks=%d  audio=%dms  wall=%.0fms  first_chunk=%.0fms  "
                "voice=%s  lang=%s  speed=%s",
                n, total_audio_ms, total_wall_ms, first_latency_ms or 0,
                voice, language or "auto",
                f"{speed:.2f}" if speed is not None else "default",
            )

    except WebSocketDisconnect:
        logger.info("WS disconnected: %s", websocket.client)
    except Exception as exc:
        logger.exception("WS error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logger.info("WS closed: %s", websocket.client)
