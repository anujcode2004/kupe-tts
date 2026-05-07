"""FastAPI application factory — lifespan, CORS, router wiring.

Lifespan responsibilities (the "static memory cache")
────────────────────────────────────────────────────
1. Resolve every voice profile listed in ``OMNIVOICE_VOICE_PROFILES``.
2. For each profile:
    - if the profile already has a cached numpy embedding → load it into RAM
      and ship it to the worker.
    - otherwise → load + trim the reference audio in the main process and
      ship the raw float32 bytes to the worker, which will encode and
      persist a fresh ``.npz`` cache on first run.
3. Spawn the ProcessPoolExecutor (workers load the model + apply
   ``torch.compile`` + run a 3-shape warm-up + load every voice prompt).
4. Start the DynamicBatcher.
5. Detect any newly-saved embeddings and update each profile's JSON so the
   next start uses the fast path.
6. On shutdown: cancel the batcher, shut the executor down, free CUDA.

Per-request voice selection
───────────────────────────
HTTP and WebSocket clients can specify ``voice: "<name>"`` to select any
profile loaded at startup; otherwise the default voice is used.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omnivoice.utils.audio import load_audio

from .batcher import DynamicBatcher
from .config import (
    ATTN_IMPL,
    BATCH_TIMEOUT_MS,
    CROSSFADE_MS,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    DEVICE,
    FIRST_CHUNK_GUIDANCE,
    FIRST_CHUNK_STEPS,
    MAX_BATCH_SIZE,
    MAX_CONCURRENT,
    MAX_WORKERS,
    MODEL_ID,
    REF_PROMPT_MAX_SEC,
    SORT_BATCH,
    USE_CUDNN_BENCH,
    USE_SAGE_ATTN,
    USE_TF32,
    USE_TORCH_COMPILE,
    VOICE_PROFILES,
    VOICE_PROFILES_AUTO,
    WEIGHT_DTYPE,
)
from .routes import batch_router, health_router, streaming_router
from .voice_profiles import (
    VoiceEmbedding,
    VoiceProfile,
    list_profiles,
    load_profile_by_name,
)
from .worker import worker_init, worker_probe

# ---------------------------------------------------------------------------
# Multiprocessing spawn context (CUDA-safe)
# ---------------------------------------------------------------------------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

logger = logging.getLogger("omnivoice.server")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _trim_ref_audio(
    path: Path, sr: int, max_sec: float, ref_text: Optional[str],
) -> tuple[np.ndarray, Optional[str]]:
    """Load + trim a reference audio file to ``max_sec`` and proportionally
    shorten the transcript so they stay aligned."""
    wav   = load_audio(str(path), sr)
    total = wav.shape[-1]
    max_s = max(1, int(max_sec * sr))
    if total <= max_s:
        return wav, ref_text

    trimmed = wav[..., :max_s]
    frac    = trimmed.shape[-1] / total
    short_text = ref_text
    if ref_text and frac < 1.0:
        n   = max(32, int(len(ref_text) * frac))
        cut = ref_text[:n]
        sp  = cut.rfind(" ")
        if sp > n // 2:
            cut = cut[:sp]
        short_text = cut.strip()

    logger.warning(
        "Reference audio trimmed %.1fs → %.1fs (max=%.1fs).",
        total / sr, trimmed.shape[-1] / sr, max_sec,
    )
    return trimmed, short_text


def _serialise_embedding(embedding: VoiceEmbedding) -> dict:
    """Return a dict that crosses the spawn boundary cleanly."""
    return {
        "ref_audio_tokens": embedding.ref_audio_tokens.astype(np.int64),
        "ref_text":         embedding.ref_text,
        "ref_rms":          float(embedding.ref_rms),
        "sampling_rate":    int(embedding.sampling_rate),
        "model_id":         embedding.model_id,
        "num_codebooks":    int(embedding.num_codebooks),
        "num_tokens":       int(embedding.num_tokens),
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

def _build_voice_init_spec(
    profile: VoiceProfile,
    sr:      int,
) -> tuple[dict, bool]:
    """Prepare the spawn-safe init spec for one voice profile.

    Returns ``(spec, used_cache)`` where ``used_cache=True`` means the
    pre-built numpy embedding was used (fast path).
    """
    spec: dict = {
        "cached_embedding": None,
        "raw_ref_bytes":    None,
        "raw_ref_sr":       None,
        "raw_ref_text":     None,
        "cache_save_path":  None,
    }

    if profile.has_cached_embedding():
        try:
            embedding = profile.load_cached_embedding()
            spec["cached_embedding"] = _serialise_embedding(embedding)
            logger.info(
                "  [%s] cached embedding %s  tokens=(%d, %d)",
                profile.name,
                profile.resolve_embedding_path(),
                embedding.num_codebooks, embedding.num_tokens,
            )
            return spec, True
        except Exception as exc:
            logger.warning(
                "  [%s] could not load cached embedding (%s) — rebuilding.",
                profile.name, exc,
            )

    ref_path = profile.resolve_ref_audio()
    ref_clip, ref_text_short = _trim_ref_audio(
        ref_path, sr, REF_PROMPT_MAX_SEC, profile.ref_text,
    )
    spec["raw_ref_bytes"]   = ref_clip.flatten().astype(np.float32).tobytes()
    spec["raw_ref_sr"]      = sr
    spec["raw_ref_text"]    = ref_text_short
    spec["cache_save_path"] = str(profile.resolve_embedding_path())
    logger.info(
        "  [%s] cold-path ref clip %.2fs  (%s)",
        profile.name, ref_clip.shape[-1] / sr, ref_path,
    )
    return spec, False


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = DEVICE or _auto_device()
    logger.info("╔═══ OmniVoice TTS Server starting ═══╗")
    logger.info("  model               : %s", MODEL_ID)
    logger.info("  device              : %s", device)
    logger.info("  weight_dtype        : %s", WEIGHT_DTYPE)
    logger.info("  default_language    : %s", DEFAULT_LANGUAGE)
    logger.info("  default_voice       : %s", DEFAULT_VOICE)
    logger.info(
        "  voice_profiles      : %s",
        "AUTO (scan voice_reference/)" if VOICE_PROFILES_AUTO
        else ",".join(VOICE_PROFILES),
    )
    logger.info("  workers             : %d", MAX_WORKERS)
    logger.info("  max_batch_size      : %d", MAX_BATCH_SIZE)
    logger.info("  batch_timeout       : %.0f ms", BATCH_TIMEOUT_MS)
    logger.info("  max_concurrent      : %d", MAX_CONCURRENT)
    logger.info("  attn_impl           : %s", ATTN_IMPL)
    logger.info("  sage_attn           : %s", USE_SAGE_ATTN)
    logger.info("  torch.compile       : %s", USE_TORCH_COMPILE)
    logger.info("  tf32                : %s", USE_TF32)
    logger.info("  cudnn_benchmark     : %s", USE_CUDNN_BENCH)
    logger.info("  sort_batch          : %s", SORT_BATCH)
    logger.info("  crossfade_ms        : %d", CROSSFADE_MS)
    logger.info("  first_chunk_steps   : %d", FIRST_CHUNK_STEPS)
    logger.info("  first_chunk_guidance: %.1f", FIRST_CHUNK_GUIDANCE)

    # ------------------------------------------------------------------
    # Resolve every voice profile to preload
    #   - Auto mode : scan voice_reference/*.json and load every valid one.
    #   - Manual    : honour the comma list in OMNIVOICE_VOICE_PROFILES.
    # In both cases each profile's embedding cache is verified, and any
    # missing cache will be built by the worker BEFORE the lifespan yields
    # (see the probe-await block below).
    # ------------------------------------------------------------------
    if VOICE_PROFILES_AUTO:
        discovered = list_profiles()
        if not discovered:
            raise RuntimeError(
                "Auto-discovery enabled but no voice profiles found in "
                "voice_reference/. Drop a <name>_ref.json (with ref_text + "
                "ref_audio) into the folder, or set OMNIVOICE_VOICE_PROFILES."
            )
        names_to_load = discovered
        logger.info(
            "  Auto-discovered     : %d profile(s) → %s",
            len(discovered), discovered,
        )
    else:
        names_to_load = list(VOICE_PROFILES)
        logger.info("  Profiles requested  : %s", names_to_load)

    profiles:    dict[str, VoiceProfile] = {}
    voices_init: dict[str, dict]         = {}
    cold_voices: dict[str, str]          = {}    # name → cache_save_path
    tmp_sr = 24_000

    cached_count   = 0
    cold_count     = 0
    for vname in names_to_load:
        try:
            profile = load_profile_by_name(vname)
        except Exception as exc:
            logger.error("  [%s] failed to load profile JSON: %s", vname, exc)
            continue

        # Sanity-check: ref_audio must be reachable on disk.
        try:
            ref_path = profile.resolve_ref_audio()
        except FileNotFoundError as exc:
            logger.error("  [%s] %s — skipping.", vname, exc)
            continue

        cached_now = profile.has_cached_embedding()
        # Use the first cached embedding's SR; otherwise stay on default
        if cached_now:
            try:
                tmp_sr = profile.load_cached_embedding().sampling_rate
            except Exception:
                pass

        spec, used_cache = _build_voice_init_spec(profile, tmp_sr)
        profiles[profile.name]    = profile
        voices_init[profile.name] = spec
        if used_cache:
            cached_count += 1
        else:
            cold_count += 1
            if spec.get("cache_save_path"):
                cold_voices[profile.name] = spec["cache_save_path"]

    if not profiles:
        raise RuntimeError(
            "No valid voice profiles could be loaded. "
            "Check voice_reference/ for missing audio or malformed JSON files."
        )

    # Resolve effective default voice
    default_voice = DEFAULT_VOICE if DEFAULT_VOICE in profiles else next(iter(profiles))
    if default_voice != DEFAULT_VOICE:
        logger.warning(
            "OMNIVOICE_DEFAULT_VOICE='%s' not in loaded profiles; using '%s' instead.",
            DEFAULT_VOICE, default_voice,
        )

    logger.info(
        "  Voice load plan     : %d cached  +  %d cold-build  =  %d total",
        cached_count, cold_count, len(profiles),
    )
    if cold_count:
        logger.info(
            "  Cold-build voices   : %s  (embeddings will be created in worker before port opens)",
            sorted(cold_voices.keys()),
        )
    logger.info(
        "  Loaded voices       : %s  (default=%s)",
        list(profiles.keys()), default_voice,
    )

    # ------------------------------------------------------------------
    # Spawn worker processes
    # ------------------------------------------------------------------
    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=worker_init,
        initargs=(
            MODEL_ID,
            device,
            ATTN_IMPL,
            USE_SAGE_ATTN,
            WEIGHT_DTYPE,
            USE_TORCH_COMPILE,
            voices_init,
            default_voice,
            DEFAULT_LANGUAGE,
        ),
    )
    logger.info("  ProcessPoolExecutor : %d worker(s) created (workers will spawn now)", MAX_WORKERS)

    # ------------------------------------------------------------------
    # FORCE every worker to spawn + run worker_init NOW, before we yield.
    # ProcessPoolExecutor spawns lazily by default, so without this the
    # very first request would pay the full model-load + compile cost.
    # By awaiting MAX_WORKERS probe tasks, we guarantee the model is loaded,
    # torch.compile has been applied, and pre-warm has finished BEFORE
    # Uvicorn binds port 8000.
    # ------------------------------------------------------------------
    logger.info(
        "  Loading model + warming up in %d worker(s) — port %s will open AFTER this completes …",
        MAX_WORKERS, "8000",
    )
    t_warm = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        probe_results = await asyncio.gather(*[
            loop.run_in_executor(executor, worker_probe)
            for _ in range(MAX_WORKERS)
        ])
    except Exception as exc:
        logger.exception("Worker initialisation FAILED — aborting startup.")
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f"Worker initialisation failed: {exc}") from exc

    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    for r in probe_results:
        logger.info(
            "  Worker[pid=%-6s]   : ready=%s  voices=%s  sr=%d",
            r.get("pid"), r.get("ready"),
            r.get("voices"), r.get("sample_rate", 0),
        )
        if not r.get("ready"):
            executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(
                f"Worker pid={r.get('pid')} reported NOT ready — aborting startup."
            )
    # All workers reported ready
    if probe_results:
        tmp_sr = int(probe_results[0].get("sample_rate", tmp_sr))
    logger.info("  Model + warm-up     : DONE in %.0f ms", warm_ms)

    # ------------------------------------------------------------------
    # Start dynamic batcher (only after workers are warm)
    # ------------------------------------------------------------------
    batcher = DynamicBatcher(
        executor=executor,
        max_batch=MAX_BATCH_SIZE,
        timeout_ms=BATCH_TIMEOUT_MS,
    )
    batcher.start()
    logger.info("  DynamicBatcher      : started")
    logger.info("╚══════════════════════════════════════╝")
    logger.info("Endpoints: WS /ws/tts  |  POST /api/tts/batch  |  GET /health")
    logger.info("Server is READY — Uvicorn will now bind the listening port.")

    # Attach to app state
    app.state.executor        = executor
    app.state.batcher         = batcher
    app.state.sample_rate     = tmp_sr
    app.state.device          = device
    app.state.voice_profiles  = profiles            # name -> VoiceProfile
    app.state.default_voice   = default_voice
    # Backwards compat alias used by /health
    app.state.voice_profile   = profiles[default_voice]

    # ------------------------------------------------------------------
    # If we built any fresh embedding cache, update its profile JSON.
    # ------------------------------------------------------------------
    if cold_voices:
        async def _persist_cache_metadata() -> None:
            pending = dict(cold_voices)
            for _ in range(60):                    # poll for up to 60s
                done = []
                for name, path in pending.items():
                    if Path(path).exists():
                        try:
                            embedding = VoiceEmbedding.from_npz(Path(path))
                            profiles[name].update_embedding_metadata(embedding)
                            logger.info(
                                "[%s] profile JSON updated → embedding_path=%s",
                                name, profiles[name].embedding_path,
                            )
                            done.append(name)
                        except Exception as exc:
                            logger.warning(
                                "[%s] could not update profile JSON: %s", name, exc,
                            )
                            done.append(name)
                for n in done:
                    pending.pop(n, None)
                if not pending:
                    return
                await asyncio.sleep(1.0)
            for name in pending:
                logger.warning(
                    "[%s] embedding cache did not appear within 60s; "
                    "profile JSON not updated.", name,
                )

        asyncio.create_task(_persist_cache_metadata())

    try:
        yield
    finally:
        logger.info("Shutting down …")
        batcher.stop()
        executor.shutdown(wait=False, cancel_futures=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
    )

    app = FastAPI(
        title="OmniVoice Production TTS Server",
        version="2.2.0",
        description=(
            "Modular OmniVoice TTS — ProcessPoolExecutor, DynamicBatcher, "
            "SageAttention, chunked-diffusion pipelined streaming, "
            "JSON-based voice profiles with cached numpy embeddings, "
            "INT4 / INT8 / BF16 / FP16 / FP32 weight quantization."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(batch_router)
    app.include_router(streaming_router)
    return app
