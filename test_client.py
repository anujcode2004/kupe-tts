#!/usr/bin/env python3
"""OmniVoice Streaming TTS — Interactive WebSocket Test Client

Connects to the OmniVoice streaming server and lets you type text
continuously.  Each response streams back audio chunks as they are
synthesised.  The client:

  - Prints first-chunk latency for every request
  - Plays streamed audio automatically in order (requires sounddevice)
  - Decodes incoming chunks on a parallel receiver task while playback drains a
    bounded queue — overlaps websocket I/O with speaker output
  - Optionally saves each response as a WAV (--save)

Usage
-----
  # Default: play only, no files written
  python test_client.py

  # Also save WAVs under streaming_output/
  python test_client.py --save

  # No playback (e.g. headless)
  python test_client.py --no-play

  # Custom server URL
  python test_client.py --url ws://192.168.1.10:8000/ws/tts

  # Long built-in paragraph (stress multi-chunk streaming + buffered playback)
  python test_client.py --demo

  # Demo script from file (UTF-8); --demo optional if you pass --demo-file
  python test_client.py --demo-file speech.txt

Commands during session
-----------------------
  <any text>  →  synthesise and stream
  /quit or Ctrl-C  →  exit
  /ping          →  check connection
  /save off      →  stop saving WAV files
  /save on       →  resume saving WAV files
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    sys.exit("websockets not installed.  Run: pip install websockets")

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

OUTPUT_DIR = Path("streaming_output")

# Back-pressure queue: receiver fills while playback drains — overlaps network/decode with speaker I/O.
STREAM_QUEUE_MAX = 48

# Default crossfade overlap window between consecutive audio chunks.  0 = disabled.
CROSSFADE_MS = 80

DEMO_TEXT = """Welcome to this extended OmniVoice streaming demonstration designed to exercise long-form synthesis.

We are walking through multiple sentences so the server splits the passage into several timed chunks. The goal is to hear continuous speech without awkward gaps while the model keeps working ahead.

Latency sensitive assistants care about time-to-first-audio. After that, smooth pacing matters just as much: listeners tolerate brief waits once playback starts if later sentences arrive early and wait safely in a client-side buffer.

This paragraph stacks coordinating clauses on purpose. Each comma invites another clause; each semicolon marks a deliberate pause in writing even though text-to-speech may glide straight through.

Imagining a voice assistant summarizing your inbox: it streams tokens from an LLM, batches clauses into speech-ready fragments, and hands those fragments to a realtime synthesizer running beside your websocket stack.

Hardware differs wildly across deployments. Some GPUs chew through diffusion steps quickly; others serialize batches conservatively to avoid VRAM spikes. End-to-end tuning blends chunk sizing, scheduler concurrency, and playback buffering here at the edge client.

Developers sometimes chase naive parallelism until kernels collide on shared tensors; bounded concurrency tends to be safer while still overlapping chunk two against chunk one whenever memory permits.

Finally we land enough prose that your ears—not only your eyes—can judge continuity across stitch boundaries. If chunk seams bother you, adjust punctuation splitting upstream or shorten clauses until phonetics blend cleanly."""

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _wav_bytes_to_numpy(wav_bytes: bytes):
    """Decode WAV bytes to a float32 numpy array + sample rate."""
    if not _HAS_NUMPY:
        return None, 24000
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width, np.int16)
    pcm = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sample_width == 2:
        pcm /= 32768.0
    elif sample_width == 4:
        pcm /= 2147483648.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    return pcm, sr


def _play_audio(audio: "np.ndarray", sr: int) -> None:
    if not _HAS_SOUNDDEVICE or not _HAS_NUMPY:
        return
    try:
        sd.play(audio, samplerate=sr)
        sd.wait()
    except Exception as exc:
        print(f"  [playback error] {exc}", flush=True)


def _save_wav(chunks: list, sr: int, tag: str) -> "Path | None":
    """Concatenate chunks and save as WAV.  Returns saved path or None."""
    if not _HAS_NUMPY or not _HAS_SOUNDFILE:
        return None
    combined = np.concatenate(chunks, axis=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"response_{ts}_{tag}.wav"
    sf.write(str(out_path), combined, sr)
    return out_path


# ---------------------------------------------------------------------------
# Crossfade pipeline
# ---------------------------------------------------------------------------

class ChunkCrossfader:
    """Overlap-add crossfade between successive streaming TTS chunks.

    Algorithm per chunk
    -------------------
    1. First chunk  → linear fade-in on the first ``n`` samples (silence → full).
    2. Every chunk  → the last ``n`` raw samples are *held* (not played yet) as
                      ``_tail``.  This adds ``crossfade_ms`` latency but prevents
                      hard audible cuts.
    3. Next chunk   → ``_tail`` is fade-out blended with the new chunk's first ``n``
                      samples (fade-in).  The blended region is prepended and played.
    4. After ``done`` call ``flush()`` → returns the final tail with a fade-out so
                      speech ends cleanly rather than cutting abruptly.

    If ``crossfade_ms == 0`` the crossfader is a transparent pass-through.
    """

    def __init__(self, sr: int, crossfade_ms: int = CROSSFADE_MS) -> None:
        self.sr = sr
        self.crossfade_ms = crossfade_ms
        self._n: int = max(1, int(sr * crossfade_ms / 1000)) if crossfade_ms > 0 else 0
        self._tail: "np.ndarray | None" = None

    def feed(self, audio: "np.ndarray", *, is_first: bool) -> "np.ndarray":
        """Process one raw chunk.  Returns audio ready to send to the speaker."""
        if not _HAS_NUMPY or len(audio) == 0:
            return audio
        if self._n == 0:
            return audio

        # Never let overlap exceed 1/3 of this chunk (short last-sentence guard).
        n = min(self._n, len(audio) // 3 or 1)
        parts: "list[np.ndarray]" = []

        if self._tail is not None:
            # Blend previous tail (fade-out) with this chunk's head (fade-in).
            tn = min(n, len(self._tail))
            ramp_out = np.linspace(1.0, 0.0, tn, dtype=np.float32)
            ramp_in  = np.linspace(0.0, 1.0, tn, dtype=np.float32)
            blend = self._tail[-tn:] * ramp_out + audio[:tn] * ramp_in
            parts.append(blend)
            body = audio[tn:]
        else:
            # First chunk: ramp up from silence to avoid click at utterance start.
            head = audio[:n].copy()
            head *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            parts.append(head)
            body = audio[n:]

        # Carve out the tail for the next crossfade (held, not played yet).
        if len(body) > n:
            parts.append(body[:-n])
            self._tail = body[-n:].copy()
        else:
            parts.append(body)
            self._tail = None  # chunk too short to hold a separate tail

        return np.concatenate(parts) if parts else audio

    def flush(self) -> "np.ndarray | None":
        """Return any held tail with a fade-out applied.  Call once after ``done``."""
        if self._tail is None or not _HAS_NUMPY or self._n == 0:
            return None
        tail = self._tail.copy()
        tail *= np.linspace(1.0, 0.0, len(tail), dtype=np.float32)
        self._tail = None
        return tail


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on Windows)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# Streaming: WS receiver ↔ playback queue (parallel decode vs play)
# ---------------------------------------------------------------------------


async def _tts_ws_receiver(
    ws,
    queue: asyncio.Queue,
    *,
    recv_timeout: float,
) -> None:
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
            msg = json.loads(raw)
            mt = msg.get("type", "")
            if mt == "response.audio.delta":
                wav_bytes = base64.b64decode(msg["delta"])
                audio, sr = _wav_bytes_to_numpy(wav_bytes)
                await queue.put(("delta", msg, audio, sr))
            elif mt == "response.audio.done":
                await queue.put(("done", msg))
                return
            elif mt == "error":
                await queue.put(("error", msg))
                return
            elif mt == "pong":
                continue
            else:
                await queue.put(("other", msg))
    except asyncio.TimeoutError:
        await queue.put(("error", {"message": "timeout waiting for server response"}))
    except ConnectionClosed as exc:
        await queue.put(("error", {"message": f"connection closed: {exc}"}))
    except Exception as exc:
        await queue.put(("error", {"message": str(exc)}))


async def _tts_playback_consumer(
    queue: asyncio.Queue,
    *,
    play: bool,
    save_enabled: bool,
    request_count: int,
    t_send: float,
) -> None:
    audio_chunks: list = []
    sample_rate = 24000
    chunk_count = 0
    first_latency = None
    loop = asyncio.get_running_loop()

    while True:
        item = await queue.get()
        kind = item[0]

        if kind == "delta":
            msg, audio, sr_decode = item[1], item[2], item[3]
            chunk_index = msg.get("chunk_index", chunk_count)
            chunk_text = msg.get("chunk_text", "")
            chunk_audio_ms = msg.get("chunk_audio_ms", 0)
            chunk_gen_ms = msg.get("chunk_gen_ms", 0)
            sample_rate = msg.get("sample_rate", sr_decode if sr_decode else sample_rate)

            if audio is not None:
                audio_chunks.append(audio)

            if chunk_index == 0:
                first_latency = msg.get("first_chunk_latency_ms")
                wall_ms = (time.perf_counter() - t_send) * 1000
                print(
                    _c(_GREEN, f"  ↳ First chunk received!")
                    + f"  server_latency={first_latency:.0f}ms"
                      f"  wall={wall_ms:.0f}ms"
                      f"  audio={chunk_audio_ms}ms"
                      f"  gen={chunk_gen_ms}ms",
                    flush=True,
                )
            else:
                print(
                    _c(_DIM, f"  ↳ Chunk {chunk_index + 1}")
                    + f"  audio={chunk_audio_ms}ms  gen={chunk_gen_ms}ms"
                    + (f"  text='{chunk_text[:40]}'" if chunk_text else ""),
                    flush=True,
                )

            if play and _HAS_SOUNDDEVICE and audio is not None:
                await loop.run_in_executor(None, _play_audio, audio, sample_rate)

            chunk_count += 1

        elif kind == "done":
            msg = item[1]
            total_chunks = msg.get("total_chunks", chunk_count)
            total_audio_ms = msg.get("total_audio_ms", 0)
            total_gen_ms = msg.get("total_gen_ms", 0)
            first_latency = msg.get("first_chunk_latency_ms", first_latency or 0)

            print(
                _c(_BOLD, "\n  ✓ Done!")
                + f"  chunks={total_chunks}"
                + f"  total_audio={total_audio_ms}ms"
                + f"  total_gen={total_gen_ms:.0f}ms"
                + _c(_GREEN, f"  first_chunk_latency={first_latency:.0f}ms\n"),
                flush=True,
            )
            break

        elif kind == "error":
            msg = item[1]
            print(_c(_RED, f"  [server error] {msg.get('message')}"), flush=True)
            break

        elif kind == "other":
            msg = item[1]
            print(_c(_DIM, f"  [unknown msg] {msg.get('type')}"), flush=True)

    if save_enabled and audio_chunks and _HAS_NUMPY and _HAS_SOUNDFILE:
        tag = f"req{request_count:03d}"
        path = _save_wav(audio_chunks, sample_rate, tag)
        if path:
            print(_c(_DIM, f"  Saved → {path}"), flush=True)


async def run_single_tts_exchange(
    ws,
    text: str,
    *,
    play: bool,
    save_enabled: bool,
    request_count: int,
    recv_timeout: float = 60.0,
) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
    t_send = time.perf_counter()
    await ws.send(json.dumps({"type": "tts.request", "text": text}))
    print(_c(_DIM, "  Waiting for audio …"), flush=True)

    recv_task = asyncio.create_task(
        _tts_ws_receiver(ws, queue, recv_timeout=recv_timeout)
    )
    try:
        await _tts_playback_consumer(
            queue,
            play=play,
            save_enabled=save_enabled,
            request_count=request_count,
            t_send=t_send,
        )
    finally:
        await recv_task


# ---------------------------------------------------------------------------
# Main async client
# ---------------------------------------------------------------------------

async def run_client(url: str, play: bool, save: bool, demo_text=None) -> None:
    print(_c(_BOLD, f"\nConnecting to {url} …"), flush=True)

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=30,
            max_size=50 * 1024 * 1024,  # 50 MB (large audio chunks)
        ) as ws:
            print(_c(_GREEN, "Connected!"), flush=True)

            if demo_text is not None:
                print(
                    _c(_DIM, "Demo mode: long multi-chunk script (receiver and playback run in parallel).\n"),
                    flush=True,
                )
                await run_single_tts_exchange(
                    ws,
                    demo_text.strip(),
                    play=play,
                    save_enabled=save,
                    request_count=1,
                )
                print(_c(_DIM, "Demo finished."), flush=True)
                return

            print(_c(_DIM, "Type text and press Enter to synthesise.  /quit to exit.\n"), flush=True)

            save_enabled = save
            request_count = 0

            while True:
                # ---- Get user input ----------------------------------------
                try:
                    line = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: input(_c(_CYAN, "You > "))
                    )
                except EOFError:
                    break

                line = line.strip()
                if not line:
                    continue

                # ---- Commands -----------------------------------------------
                if line in ("/quit", "/exit", "/q"):
                    print(_c(_DIM, "Bye!"))
                    break

                if line == "/ping":
                    await ws.send('{"type":"ping"}')
                    resp = await ws.recv()
                    print(_c(_DIM, f"  Server: {resp}"))
                    continue

                if line == "/save off":
                    save_enabled = False
                    print(_c(_DIM, "  WAV saving disabled."))
                    continue

                if line == "/save on":
                    save_enabled = True
                    print(_c(_DIM, "  WAV saving enabled."))
                    continue

                if line.startswith("/"):
                    print(_c(_YELLOW, f"  Unknown command: {line}"))
                    continue

                # ---- Send TTS request (receiver task + buffered playback) ---
                request_count += 1
                await run_single_tts_exchange(
                    ws,
                    line,
                    play=play,
                    save_enabled=save_enabled,
                    request_count=request_count,
                )

    except ConnectionClosed as exc:
        print(_c(_RED, f"\nConnection closed: {exc}"), flush=True)
    except OSError as exc:
        print(_c(_RED, f"\nCannot connect to {url}: {exc}"), flush=True)
        print(_c(_YELLOW, "Make sure the server is running:  python server.py"), flush=True)
    except KeyboardInterrupt:
        print(_c(_DIM, "\nInterrupted."))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice streaming TTS interactive test client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/tts",
        help="WebSocket server URL.",
    )
    parser.add_argument(
        "--no-play",
        dest="play",
        action="store_false",
        default=True,
        help="Disable live playback via sounddevice.",
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=False,
        help="Save each response as a WAV under streaming_output/.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Send built-in long DEMO_TEXT once and exit (many server chunks).",
    )
    parser.add_argument(
        "--demo-file",
        type=Path,
        default=None,
        help="With --demo (or alone): UTF-8 script file instead of DEMO_TEXT.",
    )
    args = parser.parse_args()

    demo_body = None
    if args.demo or args.demo_file:
        if args.demo_file is not None:
            demo_body = args.demo_file.read_text(encoding="utf-8")
        else:
            demo_body = DEMO_TEXT

    if args.play and not _HAS_SOUNDDEVICE:
        print(
            _c(_YELLOW, "[warning] sounddevice not installed — live playback disabled.\n"
               "  Install with: pip install sounddevice"),
            flush=True,
        )

    asyncio.run(
        run_client(
            url=args.url,
            play=args.play,
            save=args.save,
            demo_text=demo_body,
        )
    )


if __name__ == "__main__":
    main()
