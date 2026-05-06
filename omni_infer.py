from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import wave

import numpy as np
import torch

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceGenerationConfig,
        VoiceClonePrompt,
    )
except Exception:  # pragma: no cover - allows local lint/test without omnivoice
    OmniVoice = None  # type: ignore
    OmniVoiceGenerationConfig = None  # type: ignore
    VoiceClonePrompt = None  # type: ignore


ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "profiles"


@dataclass
class GenerationResult:
    text: str
    generated_text: str
    ttft_seconds: float
    total_seconds: float
    profile_name: Optional[str]
    used_cached_voice: bool
    audio: np.ndarray
    sample_rate: int


class OmniVoiceEngine:
    """
    OmniVoice inference engine with profile-driven voice prompt caching.
    """

    def __init__(
        self,
        model: str = "k2-fsa/OmniVoice",
        dtype: str = "float16",
    ) -> None:
        self.model_name = model
        self.dtype = dtype
        self._model = None
        self._voice_prompts: Dict[str, Any] = {}
        self._default_generation_params: Dict[str, Any] = {
            "num_step": 16,
            "guidance_scale": 1.5,
            "t_shift": 0.1,
            "denoise": True,
            "postprocess_output": True,
            "layer_penalty_factor": 5.0,
            "position_temperature": 5.0,
            "class_temperature": 0.0,
        }

    def _torch_dtype(self):
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _get_model(self):
        if self._model is None:
            if OmniVoice is None:
                raise RuntimeError(
                    "omnivoice is not importable. Install OmniVoice dependencies first."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=device,
                dtype=self._torch_dtype(),
            )
        return self._model

    def _build_generation_config(self, generation_params: Optional[Dict[str, Any]] = None):
        if OmniVoiceGenerationConfig is None:
            raise RuntimeError("OmniVoiceGenerationConfig is unavailable.")
        merged = dict(self._default_generation_params)
        if generation_params:
            merged.update(generation_params)

        sig = inspect.signature(OmniVoiceGenerationConfig)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in merged.items() if k in allowed and v is not None}
        invalid = [k for k in merged.keys() if k not in allowed]
        if invalid:
            raise ValueError(
                "Unsupported generation params: "
                + ", ".join(sorted(invalid))
                + ". Check OmniVoiceGenerationConfig supported fields."
            )
        return OmniVoiceGenerationConfig(**filtered)

    def preload(self) -> None:
        """
        Explicitly load model weights before serving UI requests.
        """
        self._get_model()

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        profile_dir = PROFILES_DIR / profile_name
        meta_path = profile_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Profile metadata not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _profile_cache_dir(self, profile_name: str) -> Path:
        cache_dir = PROFILES_DIR / profile_name / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _voice_cache_path(self, profile_name: str) -> Path:
        return self._profile_cache_dir(profile_name) / "voice_embedding.npy"

    def _voice_meta_path(self, profile_name: str) -> Path:
        return self._profile_cache_dir(profile_name) / "voice_prompt_meta.json"

    def _profile_default_audio_path(self, profile_name: str) -> Path:
        return PROFILES_DIR / profile_name / "ref_audio.wav"

    @staticmethod
    def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio.astype(np.float32)
        src_len = len(audio)
        dst_len = max(1, int(src_len * (dst_sr / src_sr)))
        src_x = np.linspace(0.0, 1.0, src_len, dtype=np.float64)
        dst_x = np.linspace(0.0, 1.0, dst_len, dtype=np.float64)
        return np.interp(dst_x, src_x, audio).astype(np.float32)

    @staticmethod
    def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Ensure waveform is finite and bounded for Gradio playback conversion.
        """
        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)
        safe = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(safe)))
        if peak <= 1e-12:
            return np.zeros(max(1, safe.shape[0]), dtype=np.float32)
        safe = np.clip(safe / peak, -1.0, 1.0)
        return safe.astype(np.float32)

    def _load_audio(self, path: Path, target_sr: int) -> np.ndarray:
        if sf is not None:
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return self._resample_linear(audio, int(sr), target_sr)

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise RuntimeError("Only 16-bit PCM WAV is supported when soundfile is unavailable.")
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            return self._resample_linear(audio, int(sr), target_sr)

    def _build_voice_embedding(self, audio_bytes: bytes) -> np.ndarray:
        # Placeholder fast embedding for cache demo.
        # Replace with OmniVoice speaker encoder output if available in your stack.
        digest = hashlib.sha256(audio_bytes).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(256, dtype=np.float32)

    def clone_voice(self, profile_name: str, reference_audio_path: str) -> Path:
        profile_dir = PROFILES_DIR / profile_name
        if not profile_dir.exists():
            raise FileNotFoundError(f"Profile directory not found: {profile_dir}")
        audio_path = Path(reference_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        audio_bytes = audio_path.read_bytes()
        embedding = self._build_voice_embedding(audio_bytes)
        cache_path = self._voice_cache_path(profile_name)
        np.save(cache_path, embedding)

        meta = self.load_profile(profile_name)
        meta_path = self._voice_meta_path(profile_name)
        meta_path.write_text(
            json.dumps(
                {
                    "reference_audio_path": str(audio_path.resolve()),
                    "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                    "ref_text": meta.get("ref_text", ""),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Build and hold voice prompt in memory so first generation is faster.
        # If omnivoice deps are unavailable in the current environment, keep disk cache only.
        try:
            model = self._get_model()
            model_sr = int(model.sampling_rate)
            ref_audio = self._load_audio(audio_path, model_sr)
            prompt = model.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(ref_audio.astype(np.float32)), model_sr),
                ref_text=meta.get("ref_text", ""),
                preprocess_prompt=True,
            )
            self._voice_prompts[profile_name] = prompt
        except Exception:
            pass
        return cache_path

    def _load_cached_voice(self, profile_name: str) -> Optional[np.ndarray]:
        cache_path = self._voice_cache_path(profile_name)
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def generate(
        self,
        text: str,
        profile_name: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        if not text.strip():
            raise ValueError("Input text is empty.")

        start = time.perf_counter()
        model = self._get_model()
        used_cached_voice = False
        voice_prompt = None

        if profile_name:
            meta = self.load_profile(profile_name)
            voice_emb = self._load_cached_voice(profile_name)
            used_cached_voice = voice_emb is not None

            voice_prompt = self._voice_prompts.get(profile_name)
            if voice_prompt is None:
                # Rehydrate voice prompt lazily from cached metadata.
                meta_path = self._voice_meta_path(profile_name)
                if not meta_path.exists():
                    raise RuntimeError(
                        f"No voice cache metadata found for profile '{profile_name}'. "
                        "Create voice clone cache first."
                    )
                cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model_sr = int(model.sampling_rate)
                ref_audio_path = Path(cache_meta["reference_audio_path"])

                try:
                    if not ref_audio_path.exists():
                        raise FileNotFoundError(str(ref_audio_path))
                    ref_audio = self._load_audio(ref_audio_path, model_sr)
                except Exception:
                    # Handle stale/invalid cache metadata (e.g., test tmp paths or bad files).
                    fallback_audio = self._profile_default_audio_path(profile_name)
                    if not fallback_audio.exists():
                        raise RuntimeError(
                            f"Cached reference audio is unusable: {ref_audio_path}. "
                            "Recreate voice clone cache with a valid audio file."
                        )
                    ref_audio = self._load_audio(fallback_audio, model_sr)
                    cache_meta["reference_audio_path"] = str(fallback_audio.resolve())
                    meta_path.write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")

                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=(torch.from_numpy(ref_audio.astype(np.float32)), model_sr),
                    ref_text=meta.get("ref_text", ""),
                    preprocess_prompt=True,
                )
                self._voice_prompts[profile_name] = voice_prompt

        gen_config = self._build_generation_config(generation_params)
        if voice_prompt is not None:
            audios = model.generate(
                text=text,
                voice_clone_prompt=voice_prompt,
                generation_config=gen_config,
            )
        else:
            audios = model.generate(text=text, generation_config=gen_config)
        first_token_ts = time.perf_counter()
        audio = audios[0] if audios else np.array([], dtype=np.float32)
        audio = self._sanitize_audio(audio)
        end = time.perf_counter()

        return GenerationResult(
            text=text,
            generated_text=f"[audio_samples={len(audio)}]",
            ttft_seconds=first_token_ts - start,
            total_seconds=end - start,
            profile_name=profile_name,
            used_cached_voice=used_cached_voice,
            audio=audio.astype(np.float32),
            sample_rate=int(model.sampling_rate),
        )
