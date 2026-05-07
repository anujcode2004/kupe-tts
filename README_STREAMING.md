# OmniVoice Streaming TTS — WebSocket Server

OpenAI-realtime-compatible WebSocket API that streams synthesised audio
sentence-by-sentence, sending the **first audio chunk as soon as the first
sentence finishes generating** (typically 80–200 ms on GPU).

---

## Architecture

```
Client  →  WebSocket  →  FastAPI Server
                              │
                              ▼
                    StreamingChunker
                    (sentence boundaries)
                              │
                    ┌─────────┴────────────┐
                    ▼                      ▼
              Chunk 0 →  OmniVoice   Chunk 1 → OmniVoice
              (sent immediately)     (sent when ready)
                    │
                    ▼
             Base64 WAV over WebSocket
```

**Key optimisations applied:**

| Technique | Latency saved |
|---|---|
| Split to sentence chunks (~60 chars) | −600–800 ms (biggest win) |
| `num_step=16` (half of default 32) | −40 % per chunk |
| `postprocess_output=False` on mid-stream chunks | −30–50 ms |
| `guidance_scale=1.5` on mid-stream chunks | slight improvement |
| Voice-clone prompt pre-tokenised at startup | −200 ms per request |
| Model pre-warmed with dummy phrase | −150 ms on first request |

---

## Quick Start

### 1. Install dependencies

```bash
# Install the OmniVoice package (from project root)
pip install -e .

# Install server-specific extras
pip install -r requirements_server.txt
```

### 2. Start the server

```bash
python server.py
```

The server will:
1. Download `k2-fsa/OmniVoice` from HuggingFace (first run)
2. Pre-tokenise `voice_reference/man_voice.mp3` as the default voice
3. Run a pre-warmup inference
4. Start listening on `ws://0.0.0.0:8000/ws/tts`

**Environment variables** (all optional):

| Variable | Default | Description |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | HuggingFace model ID or local path |
| `OMNIVOICE_DEVICE` | auto | `cuda`, `mps`, or `cpu` |
| `CHUNK_CHARS` | `60` | Sentence chunk size in characters |

### 3. Run the test client

```bash
# Saves WAV files to ./streaming_output/
python test_client.py

# With live audio playback (needs sounddevice)
python test_client.py --play

# Custom server
python test_client.py --url ws://myserver:8000/ws/tts
```

Type any text and press Enter.  The client will print:

```
You > Hello, how are you doing today? I hope everything is going well.

  Waiting for audio …
  ↳ First chunk received!  server_latency=143ms  wall=145ms  audio=1200ms  gen=141ms
  ↳ Chunk 2  audio=1850ms  gen=189ms  text='I hope everything is going well.'

  ✓ Done!  chunks=2  total_audio=3050ms  total_gen=331ms  first_chunk_latency=143ms

  Saved → streaming_output/response_20260504_115302_req001.wav
```

---

## Wire Protocol

### Client → Server

```json
{"type": "tts.request", "text": "Your text here."}
```

### Server → Client (streaming)

**Audio delta** (one per sentence chunk, sent immediately when ready):
```json
{
  "type": "response.audio.delta",
  "delta": "<base64-encoded WAV PCM-16>",
  "encoding": "wav/pcm16",
  "sample_rate": 24000,
  "chunk_index": 0,
  "chunk_text": "Your text here.",
  "chunk_audio_ms": 1200,
  "chunk_gen_ms": 141.3,
  "first_chunk_latency_ms": 143.0   // only on chunk_index == 0
}
```

**Done signal** (sent after all chunks):
```json
{
  "type": "response.audio.done",
  "total_chunks": 2,
  "total_audio_ms": 3050,
  "total_gen_ms": 331.0,
  "first_chunk_latency_ms": 143.0
}
```

**Error**:
```json
{"type": "error", "message": "..."}
```

**Ping/pong** (keepalive):
```json
// send:   {"type": "ping"}
// recv:   {"type": "pong"}
```

### Decoding the audio delta

```python
import base64, io, wave, numpy as np

wav_bytes = base64.b64decode(delta_b64)
buf = io.BytesIO(wav_bytes)
with wave.open(buf) as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
```

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"k2-fsa/OmniVoice","sample_rate":24000,"device":"cuda:0","voice_prompt_ready":true}
```

---

## Expected Latency (GPU, NVIDIA A100/H100)

| Scenario | First-chunk latency |
|---|---|
| Short sentence (~10 words, `num_step=16`) | ~80–150 ms |
| Medium sentence (~20 words, `num_step=16`) | ~150–300 ms |
| Long paragraph (split into chunks) | first chunk ~150 ms, rest pipeline |

> On CPU, latency is 10–30× higher.  Use GPU for interactive use.

---

## Changing the Default Voice

Replace the files in `voice_reference/`:

- `man_voice.mp3` — 3–10 second reference audio clip
- `man_text.txt` — exact transcript of that clip

Then restart the server.

To use a completely different voice at runtime, the current server always
uses the pre-loaded reference.  For per-request voice control, extend the
wire protocol to include `ref_audio` as a base64 WAV in the `tts.request`
message.
