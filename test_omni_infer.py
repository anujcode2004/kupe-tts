from pathlib import Path
import wave

import numpy as np

from omni_infer import OmniVoiceEngine, PROFILES_DIR


def test_load_profile_soham():
    engine = OmniVoiceEngine()
    meta = engine.load_profile("soham")
    assert meta["name"] == "soham"


def test_clone_voice_creates_cache(tmp_path: Path):
    engine = OmniVoiceEngine()
    profile_name = "pytest_profile"
    profile_dir = PROFILES_DIR / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "meta.json").write_text(
        '{"name":"pytest_profile","ref_text":"hello test","created_at":"2026-05-05T00:00:00+00:00"}',
        encoding="utf-8",
    )

    ref_audio = tmp_path / "ref_audio.wav"
    with wave.open(str(ref_audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        silence = np.zeros(1600, dtype=np.int16)  # 100ms
        wf.writeframes(silence.tobytes())

    cache_path = engine.clone_voice(profile_name, str(ref_audio))
    assert cache_path.exists()
    emb = np.load(cache_path)
    assert emb.shape == (256,)
