from __future__ import annotations

import asyncio
import base64
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from omni_infer import OmniVoiceEngine
from profile_setup import resolve_profile_name
from text_chunking import split_into_sentence_chunks

ENGINE: OmniVoiceEngine | None = None


def get_engine() -> OmniVoiceEngine:
    if ENGINE is None:
        raise RuntimeError("Engine is not initialized.")
    return ENGINE


def audio_float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    safe = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.clip(safe, -1.0, 1.0)
    pcm = (safe * 32767.0).astype(np.int16)
    return pcm.tobytes()


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    profile_name: Optional[str] = Field(
        default=None,
        description="Optional profile name; if set, profile-based voice cloning is used.",
    )
    chunk_by_sentence: bool = Field(
        default=False,
        description="If true, split by sentence and synthesize chunk-by-chunk before concatenation.",
    )
    generation_config: Optional[dict] = Field(
        default=None,
        description=(
            "Optional OmniVoiceGenerationConfig overrides. "
            "Example: {\"num_step\":24,\"guidance_scale\":2.0,\"postprocess_output\":true}."
        ),
    )


class GenerateResponse(BaseModel):
    sample_rate: int
    profile_name: Optional[str]
    text: str
    chunk_count: int
    total_seconds: float
    audio_pcm16_base64: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ENGINE
    ENGINE = OmniVoiceEngine()
    ENGINE.preload()
    yield


app = FastAPI(
    title="OmniVoice Caching API",
    version="1.0.0",
    description="HTTP and WebSocket TTS generation with optional profile-based voice cloning.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    engine = get_engine()
    profile_name = resolve_profile_name(req.profile_name) if req.profile_name else None
    chunks = split_into_sentence_chunks(req.text) if req.chunk_by_sentence else [req.text.strip()]

    start = time.perf_counter()
    all_audio = []
    sample_rate = None
    for chunk in chunks:
        result = engine.generate(
            chunk,
            profile_name=profile_name,
            generation_params=req.generation_config,
        )
        sample_rate = result.sample_rate
        all_audio.append(result.audio)

    if all_audio:
        merged = np.concatenate(all_audio)
    else:
        merged = np.zeros(160, dtype=np.float32)
        sample_rate = sample_rate or 16000

    elapsed = time.perf_counter() - start
    payload = base64.b64encode(audio_float_to_pcm16_bytes(merged)).decode("ascii")

    return GenerateResponse(
        sample_rate=int(sample_rate or 16000),
        profile_name=profile_name,
        text=req.text,
        chunk_count=len(chunks),
        total_seconds=elapsed,
        audio_pcm16_base64=payload,
    )


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()
    engine = get_engine()
    try:
        while True:
            msg = await websocket.receive_json()
            text = str(msg.get("text", "")).strip()
            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue

            requested_profile = msg.get("profile_name")
            profile_name = resolve_profile_name(requested_profile) if requested_profile else None
            generation_config = msg.get("generation_config")
            chunks = split_into_sentence_chunks(text)
            request_start = time.perf_counter()

            await websocket.send_json(
                {
                    "type": "tts.start",
                    "chunk_count": len(chunks),
                    "profile_name": profile_name,
                }
            )

            for idx, chunk in enumerate(chunks):
                result = engine.generate(
                    chunk,
                    profile_name=profile_name,
                    generation_params=generation_config,
                )
                audio_bytes = audio_float_to_pcm16_bytes(result.audio)
                await websocket.send_json(
                    {
                        "type": "audio.chunk",
                        "index": idx,
                        "text": chunk,
                        "sample_rate": result.sample_rate,
                        "ttft_seconds": result.ttft_seconds,
                        "total_seconds": result.total_seconds,
                        "is_last": idx == len(chunks) - 1,
                        "audio_pcm16_base64": base64.b64encode(audio_bytes).decode("ascii"),
                    }
                )
                await asyncio.sleep(0)

            await websocket.send_json(
                {
                    "type": "tts.end",
                    "elapsed_seconds": time.perf_counter() - request_start,
                }
            )
    except WebSocketDisconnect:
        return
