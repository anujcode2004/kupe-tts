# OmniVoice — Concurrent Stream Tester (Web UI)

A single-page HTML tool for stress-testing the OmniVoice streaming server.
It fires **N concurrent WebSocket streams** against `/ws/tts`, broadcasts the
**same text** to all of them, shows per-stream metrics in a live table, and
plays the audio of one chosen stream so you can hear that everything works.

## Files

| File                      | What it is                                                        |
| ------------------------- | ----------------------------------------------------------------- |
| `concurrent_streams.html` | The full UI (vanilla JS, no build step)                           |
| `sample_text.txt`         | The multilingual stress text (Punjabi · Hindi · Gujarati · Bengali) |

## How to run

1. Start the OmniVoice server on `localhost:8000`:
   ```bash
   python server.py
   ```
2. Serve this folder over HTTP (so the UI can fetch `sample_text.txt`):
   ```bash
   cd web_demo
   python -m http.server 5500
   ```
3. Open <http://localhost:5500/concurrent_streams.html> in your browser.

Opening the file directly via `file://` also works — the multilingual sample
text is embedded as a fallback if `fetch('./sample_text.txt')` fails.

## What the UI shows

- **Configuration panel** — Server URL, voice, language, speed (0.25–3.0),
  concurrent streams, **stream start delay (s)**, audio playback target,
  high-quality toggle, text.
  - `Stream start delay = 0` → all streams fire simultaneously.
  - `delay = 2s, N = 5` → stream 1 fires at t=0, stream 2 at t=2s, …,
    stream 5 at t=8s.  Useful for measuring how the server behaves under
    a slow ramp-up.
- **Live metrics table** — columns are `Stream 1, 2, 3, …`, rows are:
  - `Status`
  - `FCL — client (ms)` — first-chunk latency observed by the browser
  - `FCL — server (ms)` — first-chunk latency reported by the server
  - `First-chunk gen (ms)` — time spent inside `model.generate()` for chunk 0
  - `First-chunk audio (ms)` — duration of the first audio chunk
  - `Chunks received`
  - `Audio received so far (ms)`
  - `Total audio (ms)`
  - `Server wall (ms)`
  - `Client wall (ms)`
  - `RTF (server_wall / audio)`
- **Audio playback** — chunks of the chosen stream are decoded and queued
  into a single `AudioContext` so playback is gapless.
- **Live log** — every chunk from every stream, color-coded per stream.

## Notes

- **Chunking happens server-side.** The browser sends the full text in one
  `tts.request`. The server runs `split_first_chunk_early` +
  `split_to_chunks` and streams the audio chunks back over the same
  WebSocket — the UI does not pre-split anything.
- **Fully-parallel chunk generation.** As of this change, the server fires
  every chunk into the DynamicBatcher at `t=0`. Awaiting the futures in
  order on the route side preserves chunk ordering on the wire, but
  generation never blocks on a previous chunk being sent or played. This
  means later chunks are usually already done by the time the client
  reaches them in playback.
- The UI passes `voice`, `language`, `speed`, and `use_high_quality` through
  to the server's WebSocket protocol. They map directly onto the params
  documented in `tts_server/routes/streaming.py`.
- `language=auto` is sent as no `language` field at all, matching the
  server's behaviour (the model auto-detects from text + voice profile).
