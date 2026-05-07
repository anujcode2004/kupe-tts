"""GET /health — server status, config snapshot, batcher metrics, voice profile."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..config import (
    ATTN_IMPL,
    BATCH_TIMEOUT_MS,
    CROSSFADE_MS,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    FC_BATCH_TIMEOUT_MS,
    FIRST_CHUNK_GUIDANCE,
    FIRST_CHUNK_STEPS,
    MAX_BATCH_SIZE,
    MAX_CONCURRENT,
    MAX_REST_BATCH,
    MAX_WORKERS,
    MODEL_ID,
    REST_CHUNK_STEPS,
    SORT_BATCH,
    SPEED_MAX,
    SPEED_MIN,
    USE_CUDNN_BENCH,
    USE_SAGE_ATTN,
    USE_TF32,
    USE_TORCH_COMPILE,
    VOICE_PROFILES,
    VOICE_PROFILES_AUTO,
    WEIGHT_DTYPE,
)

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    state    = request.app.state
    batcher  = getattr(state, "batcher", None)
    profiles = getattr(state, "voice_profiles", {})
    default  = getattr(state, "default_voice", DEFAULT_VOICE)

    voices_info: list[dict] = []
    for name, profile in profiles.items():
        voices_info.append({
            "name":           profile.name,
            "language":       profile.language,
            "ref_audio":      str(profile.resolve_ref_audio()),
            "embedding_path": str(profile.resolve_embedding_path()),
            "cached":         profile.has_cached_embedding(),
            "embedding_meta": profile.embedding_meta,
            "is_default":     name == default,
        })

    return {
        "status":               "ok",
        "model":                MODEL_ID,
        "device":               getattr(state, "device", "unknown"),
        "weight_dtype":         WEIGHT_DTYPE,
        "default_language":     DEFAULT_LANGUAGE,
        "default_speed":        DEFAULT_SPEED,
        "speed_range":          [SPEED_MIN, SPEED_MAX],
        "sample_rate":          getattr(state, "sample_rate", 24_000),
        "default_voice":        default,
        "available_voices":     sorted(profiles.keys()),
        "voice_profiles":       voices_info,
        "configured_profiles":  VOICE_PROFILES if not VOICE_PROFILES_AUTO else "auto",
        "voice_auto_discovery": VOICE_PROFILES_AUTO,
        "workers":              MAX_WORKERS,
        "max_batch_size":       MAX_BATCH_SIZE,
        "batch_timeout_ms":     BATCH_TIMEOUT_MS,
        "fc_batch_timeout_ms":  FC_BATCH_TIMEOUT_MS,
        "max_rest_batch":       MAX_REST_BATCH,
        "rest_chunk_steps":     REST_CHUNK_STEPS,
        "max_concurrent":       MAX_CONCURRENT,
        "attn_impl":            ATTN_IMPL,
        "sage_attn":            USE_SAGE_ATTN,
        "torch_compile":        USE_TORCH_COMPILE,
        "tf32":                 USE_TF32,
        "cudnn_benchmark":      USE_CUDNN_BENCH,
        "sort_batch":           SORT_BATCH,
        "crossfade_ms":         CROSSFADE_MS,
        "first_chunk_steps":    FIRST_CHUNK_STEPS,
        "first_chunk_guidance": FIRST_CHUNK_GUIDANCE,
        "batcher": {
            "total_requests": batcher.total_requests if batcher else 0,
            "total_batches":  batcher.total_batches  if batcher else 0,
            "avg_batch_size": round(batcher.avg_batch_size, 2) if batcher else 0.0,
            "total_gen_ms":   round(batcher.total_gen_ms, 1) if batcher else 0.0,
        },
    }
