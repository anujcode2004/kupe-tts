# FastAPI + WebSocket TTS API

This server provides:
- HTTP generation endpoint with optional profile voice cloning
- WebSocket streaming endpoint that splits text by sentence boundary and sends audio chunks
- Auto docs via FastAPI (`/docs`)

## Run

```bash
cd caching
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

API docs:
- `http://127.0.0.1:8000/docs`

## HTTP Request

Endpoint: `POST /generate`

Body:
```json
{
  "text": "Hello world. This is a streaming test.",
  "profile_name": "soham",
  "chunk_by_sentence": true,
  "generation_config": {
    "num_step": 24,
    "guidance_scale": 2.2,
    "t_shift": 0.1,
    "denoise": true,
    "postprocess_output": true,
    "layer_penalty_factor": 5.0,
    "position_temperature": 5.0,
    "class_temperature": 0.0
  }
}
```

Notes:
- `profile_name` is optional. If omitted, generation runs without voice-clone profile prompt.
- Unknown profile names are auto-created.

Example curl:
```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text":"Hello from HTTP generation.",
    "profile_name":"soham",
    "chunk_by_sentence": true
  }'
```

Response fields:
- `sample_rate`: output sample rate
- `audio_pcm16_base64`: raw mono PCM16 audio bytes in base64
- `chunk_count`: number of chunks synthesized
- `total_seconds`: request generation time

### OmniVoice generation parameters

Pass any supported `OmniVoiceGenerationConfig` field via `generation_config`.
Common high-impact knobs:
- `num_step` (quality/speed tradeoff; higher = slower, often better quality)
- `guidance_scale` (conditioning strength)
- `t_shift`
- `denoise`
- `postprocess_output`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`

`api_server.py` forwards `generation_config` directly to the engine. Unknown fields return a validation-style error from the server.

## WebSocket Streaming

Endpoint: `ws://127.0.0.1:8000/ws/generate`

Send message:
```json
{
  "text": "Hello there. Send this in sentence chunks.",
  "profile_name": "soham",
  "generation_config": {
    "num_step": 20,
    "guidance_scale": 1.8,
    "postprocess_output": false
  }
}
```

Server events:
- `tts.start`:
  - `chunk_count`
  - `profile_name`
- `audio.chunk` for each sentence:
  - `index`, `text`, `sample_rate`, `ttft_seconds`, `total_seconds`, `is_last`
  - `audio_pcm16_base64`
- `tts.end`:
  - `elapsed_seconds`

If request payload text is empty, server sends:
```json
{"type":"error","message":"Empty text"}
```
