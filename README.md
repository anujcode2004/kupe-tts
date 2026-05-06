# OmniVoice Caching + Benchmark

This folder provides:
- vLLM-Omni package setup plus OmniVoice inference backend
- Profile-driven generation from `caching/profiles/<name>/meta.json` (missing profiles are auto-created with `meta.json` and a placeholder `ref_audio.wav`)
- Voice clone embedding cache saved per profile for faster repeated runs
- Gradio benchmark UI to measure generation speed (default metric: TTFT)

## Install

```bash
cd caching
pip install -r requirements.txt
```

Install/upgrade core runtime packages:

```bash
pip install --upgrade transformers vllm-omni
```

## Run Gradio benchmark app

```bash
cd caching
python gradio_app.py
```

Startup behavior:
- The script preloads OmniVoice model weights first.
- Gradio launches only after preload succeeds (fail-fast on model init issues).

## Gradio parallel generation controls

The benchmark UI supports optional **parallel sentence generation**:
- Toggle **Parallel sentence generation** to split the prompt by sentence boundaries and synthesize chunks concurrently.
- Use **Max concurrent workers (1–30)** to cap parallelism.
- Toggle **Assemble stitched audio in strict sentence index order**:
  - enabled: audio is concatenated in original sentence order (recommended for playback)
  - disabled: audio is concatenated in completion order (useful for measuring scheduling effects)

## How caching works

1. Model caching:
   - `OmniVoiceEngine` preloads one model instance and reuses it for benchmark runs.
   - This avoids model init overhead in each generation.

2. Voice clone cache:
   - Click **Create/refresh voice clone cache** in the app.
   - It stores:
     - `caching/profiles/<profile>/cache/voice_embedding.npy`
     - `caching/profiles/<profile>/cache/voice_prompt_meta.json`
   - Future generations rebuild and reuse the cached voice prompt path quickly.
   - If cached metadata points to a stale/invalid audio path, runtime falls back to
     `caching/profiles/<profile>/ref_audio.wav` when available.

## Notes

- The current voice embedding path uses a deterministic placeholder embedding so the cache workflow is ready immediately.
- Replace `_build_voice_embedding()` in `omni_infer.py` with your OmniVoice speaker embedding extractor for production voice cloning quality.
# kupe-tts
