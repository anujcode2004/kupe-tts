"""POST /api/tts/batch — synchronous batch generation endpoint."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from ..audio_utils import b64_encode, wav_bytes_to_np
from ..config import DEFAULT_LANGUAGE, DEFAULT_SPEED, LAST_CHUNK_CFG, MID_CHUNK_CFG
from ..lang_utils import resolve_language
from ..schemas import BatchTTSItem, BatchTTSRequest, BatchTTSResponse

router = APIRouter()


@router.post("/api/tts/batch", response_model=BatchTTSResponse)
async def batch_tts(req: BatchTTSRequest, request: Request):
    """Generate audio for a list of texts.

    All texts compete for the same DynamicBatcher window so they naturally
    pack into as few ``model.generate()`` calls as possible.

    Optional fields in the JSON body:
      - ``language``: ISO 639 code for the synthesised speech.
      - ``voice``:    Voice profile name (must be preloaded at startup).
    """
    if not req.texts:
        raise HTTPException(400, "'texts' list cannot be empty.")

    batcher = getattr(request.app.state, "batcher", None)
    if batcher is None:
        raise HTTPException(503, "Server not ready.")

    # Resolve + validate voice
    available_voices: dict = getattr(request.app.state, "voice_profiles", {})
    default_voice:    str  = getattr(request.app.state, "default_voice", "")
    voice = (req.voice or default_voice).strip() or default_voice
    if available_voices and voice not in available_voices:
        raise HTTPException(
            400,
            f"Voice '{voice}' is not loaded. Available voices: "
            f"{sorted(available_voices.keys())}",
        )

    cfg = LAST_CHUNK_CFG if req.use_high_quality else MID_CHUNK_CFG
    language = resolve_language(req.language or DEFAULT_LANGUAGE)
    speed    = req.speed if req.speed is not None else DEFAULT_SPEED
    batches_before = batcher.total_batches
    t0 = time.perf_counter()

    tasks = [
        asyncio.create_task(
            batcher.submit(text, cfg, language=language, voice=voice, speed=speed)
        )
        for text in req.texts
    ]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    total_ms = (time.perf_counter() - t0) * 1000.0

    items: list[BatchTTSItem] = []
    for i, result in enumerate(results_raw):
        if isinstance(result, Exception):
            raise HTTPException(500, f"Generation failed for item {i}: {result}")
        audio, sr = wav_bytes_to_np(result)
        items.append(BatchTTSItem(
            id=i,
            audio_base64=b64_encode(result),
            audio_ms=round(len(audio) / sr * 1000),
            sample_rate=sr,
        ))

    return BatchTTSResponse(
        results=items,
        total_gen_ms=round(total_ms, 1),
        batch_size=len(req.texts),
        server_batches_formed=batcher.total_batches - batches_before,
        language=language or "auto",
        voice=voice,
        speed=speed,
    )
