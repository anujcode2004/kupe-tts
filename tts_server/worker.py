"""GPU worker process — model loading, quantization, SageAttention, batch generation.

Each ProcessPoolExecutor worker calls :func:`worker_init` once via
``initializer``, then serves :func:`worker_generate` calls for the lifetime
of the process.

Static memory cache strategy
────────────────────────────
1. The model weights are loaded ONCE here (not per request).
2. The voice-clone prompt is either:
   - reconstructed from a pre-built numpy embedding (fast path), OR
   - built once from raw audio + saved to disk for next time (cold path).
3. ``torch.compile`` is applied when ``OMNIVOICE_COMPILE=1`` (default).
4. The model is warmed up at batch sizes 1, 2, 4 so cuDNN can cache its
   best kernels for each shape we'll see in production.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)

from .config import (
    MID_CHUNK_CFG,
    TORCH_COMPILE_MODE,
    USE_CUDNN_BENCH,
    USE_TF32,
)
from .voice_profiles import VoiceEmbedding

log = logging.getLogger("omnivoice.worker")


# ---------------------------------------------------------------------------
# Language sanitisation
# ---------------------------------------------------------------------------
# OmniVoice does NOT have an "auto" language code. To ask for language-agnostic
# generation you must pass ``language=None``.  This helper guarantees we never
# leak our friendly aliases ("auto", "none", "", "null", …) to the model.
_LANG_AUTO_TOKENS = frozenset({"", "auto", "none", "null", "nil", "any", "*"})


def _clean_language(lang) -> Optional[str]:
    """Return ``None`` for any auto/empty token, otherwise the trimmed string."""
    if lang is None:
        return None
    s = str(lang).strip()
    if not s or s.lower() in _LANG_AUTO_TOKENS:
        return None
    return s

# ---------------------------------------------------------------------------
# Per-process globals (set once during worker_init)
# ---------------------------------------------------------------------------
_model:          Optional[OmniVoice]            = None
_prompts:        dict[str, VoiceClonePrompt]    = {}    # name → prompt
_default_voice:  str                            = ""    # fallback voice name
_sr:             int                            = 24_000


# ---------------------------------------------------------------------------
# SageAttention smart wrapper
# ---------------------------------------------------------------------------

def patch_sage_attention(enable: bool) -> None:
    """Monkey-patch ``F.scaled_dot_product_attention`` with a SageAttention wrapper.

    The wrapper routes mask-free calls to ``sageattn`` (int8 quantised fast path)
    and forwards everything else to the original SDPA so OmniVoice's block-padding
    attention masks keep working.
    """
    if not enable:
        log.info("SageAttention disabled.")
        return
    try:
        from sageattention import sageattn  # type: ignore[import]
    except ImportError:
        log.warning("SageAttention requested but not installed — using SDPA.")
        return

    import torch.nn.functional as F

    _orig_sdpa = F.scaled_dot_product_attention

    def _smart_sdpa(
        query, key, value,
        attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=None, enable_gqa=False,
    ):
        if (
            attn_mask is None
            and dropout_p == 0.0
            and query.dim() == 4
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.is_cuda
        ):
            try:
                return sageattn(
                    query, key, value,
                    tensor_layout="HND",
                    is_causal=is_causal,
                    sm_scale=scale,
                )
            except Exception:
                pass
        return _orig_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
        )

    F.scaled_dot_product_attention = _smart_sdpa  # type: ignore[assignment]
    torch.nn.functional.scaled_dot_product_attention = _smart_sdpa
    log.info("SageAttention: smart SDPA wrapper installed.")


def apply_torch_perf_flags() -> None:
    """Free GPU speedups: TF32, cuDNN benchmark, matmul precision."""
    if not torch.cuda.is_available():
        return
    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if USE_CUDNN_BENCH:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model loading with quantization
# ---------------------------------------------------------------------------

def _build_load_kwargs(
    device:        str,
    weight_dtype:  str,
    attn_impl:     str,
) -> dict:
    """Translate ``WEIGHT_DTYPE`` into ``OmniVoice.from_pretrained`` kwargs."""
    load_kw: dict[str, Any] = {"device_map": device}

    if device == "cpu":
        # bitsandbytes & fp16/bf16 require CUDA
        load_kw["dtype"] = torch.float32
        if weight_dtype != "fp32":
            log.warning(
                "device=cpu does not support %s; falling back to fp32.",
                weight_dtype,
            )
    elif weight_dtype == "fp32":
        load_kw["dtype"] = torch.float32
    elif weight_dtype == "fp16":
        load_kw["dtype"] = torch.float16
    elif weight_dtype == "bf16":
        load_kw["dtype"] = torch.bfloat16
    elif weight_dtype in ("int8", "int4"):
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                f"OMNIVOICE_WEIGHT_DTYPE={weight_dtype!r} requires "
                f"`transformers` and `bitsandbytes`. "
                f"Install: pip install bitsandbytes"
            ) from exc

        if weight_dtype == "int8":
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
        else:  # int4
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        # bitsandbytes overrides dtype, but compute_dtype is fp16
        load_kw["dtype"] = torch.float16
    else:
        load_kw["dtype"] = torch.float16

    if attn_impl and attn_impl not in ("auto", ""):
        load_kw["attn_implementation"] = attn_impl

    return load_kw


def _load_model(model_id: str, load_kw: dict) -> OmniVoice:
    """Load OmniVoice, gracefully retrying without ``attn_implementation``
    if the checkpoint metadata doesn't accept it."""
    try:
        return OmniVoice.from_pretrained(model_id, **load_kw)
    except TypeError:
        load_kw.pop("attn_implementation", None)
        log.warning("attn_implementation kwarg not accepted; retrying without it.")
        return OmniVoice.from_pretrained(model_id, **load_kw)


# ---------------------------------------------------------------------------
# Voice prompt building
# ---------------------------------------------------------------------------

def _build_prompt_from_embedding(
    model: OmniVoice, embedding: VoiceEmbedding,
) -> VoiceClonePrompt:
    """Reconstruct a :class:`VoiceClonePrompt` from a cached numpy embedding."""
    tokens = torch.as_tensor(
        embedding.ref_audio_tokens,
        dtype=torch.long,
        device=model.audio_tokenizer.device,
    )
    return VoiceClonePrompt(
        ref_audio_tokens=tokens,
        ref_text=embedding.ref_text,
        ref_rms=float(embedding.ref_rms),
    )


def _build_prompt_from_audio(
    model: OmniVoice, ref_bytes: bytes, ref_sr: int, ref_text: Optional[str],
) -> tuple[VoiceClonePrompt, VoiceEmbedding]:
    """Run ``create_voice_clone_prompt`` and snapshot the result for caching."""
    ref_np     = np.frombuffer(ref_bytes, dtype=np.float32).copy()
    ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)  # (1, T)

    prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_tensor, ref_sr),
        ref_text=ref_text,
        preprocess_prompt=True,
    )

    tokens_np = prompt.ref_audio_tokens.detach().cpu().numpy().astype(np.int64)
    embedding = VoiceEmbedding(
        ref_audio_tokens=tokens_np,
        ref_text=prompt.ref_text,
        ref_rms=float(prompt.ref_rms),
        sampling_rate=int(model.sampling_rate),
        model_id="",   # filled by caller
        num_codebooks=int(tokens_np.shape[0]),
        num_tokens=int(tokens_np.shape[1]),
    )
    return prompt, embedding


# ---------------------------------------------------------------------------
# Worker init / generate  (called via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _resolve_voice_prompt(
    model:    OmniVoice,
    name:     str,
    spec:     dict,
    model_id: str,
) -> VoiceClonePrompt:
    """Build/restore one voice prompt from its init spec.

    ``spec`` is a dict with one of two shapes:

    A. Cached path::

           { "cached_embedding": {<serialised VoiceEmbedding>} }

    B. Cold path::

           {
             "raw_ref_bytes":   bytes,    # float32 audio @ raw_ref_sr
             "raw_ref_sr":      int,
             "raw_ref_text":    str | None,
             "cache_save_path": str | None,
           }
    """
    cached = spec.get("cached_embedding")
    if cached is not None:
        log.info("[%s] Restoring voice prompt from cached embedding.", name)
        embedding = VoiceEmbedding(
            ref_audio_tokens=np.asarray(cached["ref_audio_tokens"], dtype=np.int64),
            ref_text=str(cached["ref_text"]),
            ref_rms=float(cached["ref_rms"]),
            sampling_rate=int(cached["sampling_rate"]),
            model_id=str(cached["model_id"]),
            num_codebooks=int(cached["num_codebooks"]),
            num_tokens=int(cached["num_tokens"]),
        )
        return _build_prompt_from_embedding(model, embedding)

    raw_bytes = spec.get("raw_ref_bytes")
    raw_sr    = spec.get("raw_ref_sr")
    raw_text  = spec.get("raw_ref_text")
    if raw_bytes is None or raw_sr is None:
        raise RuntimeError(
            f"Voice '{name}' has neither a cached embedding nor raw reference audio."
        )
    log.info("[%s] Building voice prompt from raw audio (cold path) …", name)
    prompt, embedding = _build_prompt_from_audio(model, raw_bytes, int(raw_sr), raw_text)
    embedding.model_id = model_id

    cache_save_path = spec.get("cache_save_path")
    if cache_save_path:
        try:
            embedding.to_npz(Path(cache_save_path))
            log.info("[%s] Saved voice embedding cache → %s", name, cache_save_path)
        except Exception as exc:
            log.warning("[%s] Failed to save embedding cache: %s", name, exc)
    return prompt


def worker_init(
    model_id:           str,
    device:             str,
    attn_impl:          str,
    use_sage:           bool,
    weight_dtype:       str,
    use_compile:        bool,
    voices_init:        dict[str, dict],   # name → init spec (see _resolve_voice_prompt)
    default_voice:      str,
    default_language:   str,
) -> None:
    """One-time per-process initialisation.

    All voices listed in ``voices_init`` are loaded into worker memory once
    and addressable by name during generation.
    """
    global _model, _prompts, _default_voice, _sr

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [Worker-%(process)d] %(message)s",
        level=logging.INFO, force=True,
    )

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    apply_torch_perf_flags()
    patch_sage_attention(use_sage)

    load_kw = _build_load_kwargs(device, weight_dtype, attn_impl)
    log.info(
        "Loading OmniVoice  model=%s  device=%s  weight_dtype=%s  attn=%s  sage=%s",
        model_id, device, weight_dtype, attn_impl, use_sage,
    )
    _model = _load_model(model_id, load_kw)
    _sr = _model.sampling_rate

    # ---- Resolve every voice prompt ---------------------------------
    if not voices_init:
        raise RuntimeError("Worker spawned with no voice profiles.")

    _prompts = {}
    for name, spec in voices_init.items():
        try:
            _prompts[name] = _resolve_voice_prompt(_model, name, spec, model_id)
            p = _prompts[name]
            log.info(
                "[%s] Voice ready  ref_text=%.60s…  tokens=(%d, %d)  rms=%.4f",
                name, p.ref_text,
                p.ref_audio_tokens.shape[0], p.ref_audio_tokens.shape[1],
                p.ref_rms,
            )
        except Exception as exc:
            log.error("[%s] Failed to load voice: %s", name, exc)

    if not _prompts:
        raise RuntimeError("No voice profiles could be loaded in the worker.")

    if default_voice not in _prompts:
        # Fall back to first available voice
        default_voice = next(iter(_prompts.keys()))
        log.warning("default_voice not loaded; falling back to '%s'.", default_voice)
    _default_voice = default_voice

    log.info("Worker has %d voice(s) loaded: %s  (default=%s)",
             len(_prompts), sorted(_prompts.keys()), _default_voice)

    # ---- Optional torch.compile --------------------------------------
    if use_compile and weight_dtype not in ("int4", "int8"):
        try:
            log.info("Compiling model.forward (mode=%s) …", TORCH_COMPILE_MODE)
            _model.forward = torch.compile(  # type: ignore[assignment]
                _model.forward,
                mode=TORCH_COMPILE_MODE,
                fullgraph=False,
                dynamic=True,
            )
        except Exception as exc:
            log.warning("torch.compile failed: %s — continuing eager.", exc)
    elif use_compile:
        log.info("torch.compile skipped (incompatible with %s).", weight_dtype)

    # ---- Pre-warm at multiple batch sizes ----------------------------
    warm_lang = _clean_language(default_language)
    log.info(
        "Pre-warming model (batch sizes 1, 2, 4)  default_lang=%s …",
        warm_lang or "auto (None)",
    )
    cfg = OmniVoiceGenerationConfig(**MID_CHUNK_CFG)
    warm_prompt = _prompts[_default_voice]
    for bs in (1, 2, 4):
        try:
            _model.generate(
                text=["Hello there."] * bs,
                language=warm_lang,
                voice_clone_prompt=warm_prompt,
                generation_config=cfg,
            )
        except Exception as exc:
            log.warning("Warm-up at batch %d failed: %s", bs, exc)
            break
    log.info("Worker ready.")


def worker_probe() -> dict:
    """No-op task used to force ``worker_init`` to complete during lifespan
    startup.  Submitting ``MAX_WORKERS`` of these in parallel guarantees every
    worker process has loaded the model, applied ``torch.compile``, and finished
    its pre-warm BEFORE the FastAPI app yields and Uvicorn opens port 8000.
    """
    import os
    if _model is None or not _prompts:
        return {
            "pid":    os.getpid(),
            "ready":  False,
            "voices": [],
        }
    return {
        "pid":           os.getpid(),
        "ready":         True,
        "voices":        sorted(_prompts.keys()),
        "default_voice": _default_voice,
        "sample_rate":   int(_sr),
    }


def worker_generate(
    texts:        list[str],
    cfg_dict:     dict,
    languages:    Optional[list[Optional[str]]] = None,
    voice_names:  Optional[list[Optional[str]]] = None,
    speeds:       Optional[list[Optional[float]]] = None,
) -> tuple[list[bytes], float]:
    """Batch generation — one ``model.generate(text=[…])`` call.

    Args:
        texts: List of texts to synthesise (one per request).
        cfg_dict: Generation config kwargs.
        languages: Optional per-text language code/name.  ``None`` entries
            fall back to the model's auto language detection.
        voice_names: Optional per-text voice profile name.  ``None`` entries
            fall back to the worker's default voice.
        speeds: Optional per-text speaking-speed multiplier (0.25–3.0).
            ``None`` entries use the model's natural duration estimation.

    Returns:
        ``(wav_bytes_list, generation_ms)``
    """
    global _model, _prompts, _default_voice, _sr

    if _model is None or not _prompts:
        raise RuntimeError("Worker is not initialised. Call worker_init first.")

    cfg = OmniVoiceGenerationConfig(**cfg_dict)

    # ---- Resolve voice prompts per request --------------------------
    if voice_names is None:
        prompts = [_prompts[_default_voice]] * len(texts)
    else:
        prompts = []
        for v in voice_names:
            name = v or _default_voice
            if name not in _prompts:
                raise KeyError(
                    f"Voice profile '{name}' not loaded in this worker. "
                    f"Available: {sorted(_prompts.keys())}"
                )
            prompts.append(_prompts[name])

    # ---- Resolve languages -----------------------------------------
    # Defence-in-depth: routes already strip "auto" → None, but if any caller
    # forgets, we still never forward a fake "auto" code to OmniVoice.
    cleaned_langs: Optional[list[Optional[str]]]
    if languages is None:
        cleaned_langs = None
    else:
        cleaned_langs = [_clean_language(l) for l in languages]

    lang_arg: Any
    if cleaned_langs is None or all(l is None for l in cleaned_langs):
        lang_arg = None
    elif len(set(cleaned_langs)) == 1:
        lang_arg = cleaned_langs[0]
    else:
        lang_arg = list(cleaned_langs)

    # ---- Resolve speed ---------------------------------------------
    speed_arg: Any
    if speeds is None or all(s is None for s in speeds):
        speed_arg = None
    elif len(set(speeds)) == 1:
        # All-equal → pass single float for fast path
        s0 = speeds[0]
        speed_arg = float(s0) if s0 is not None else None
    else:
        speed_arg = [float(s) if s is not None else None for s in speeds]

    gen_kwargs: dict[str, Any] = dict(
        text=texts,
        language=lang_arg,
        voice_clone_prompt=prompts,
        generation_config=cfg,
    )
    if speed_arg is not None:
        gen_kwargs["speed"] = speed_arg

    t0 = time.perf_counter()
    audios = _model.generate(**gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000.0

    wav_list: list[bytes] = []
    for audio in audios:
        buf = io.BytesIO()
        sf.write(buf, audio, _sr, format="WAV", subtype="PCM_16")
        wav_list.append(buf.getvalue())

    return wav_list, gen_ms
