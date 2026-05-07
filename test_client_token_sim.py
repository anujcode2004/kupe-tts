#!/usr/bin/env python3
"""Simulated token-stream → OmniVoice TTS (realtime overlay demo).

The production server accepts full ``tts.request`` messages only.  This client
*simulates* an LLM by printing words or characters with delays, then pushes
each completed sentence to a queue.  A separate task sends sentences to the
WebSocket and plays audio — so **text keeps appearing while earlier speech is
still playing**.

Default script matches ``python test_client.py --demo`` (long multi-sentence passage).
Colour legend and per-sentence banners make the LLM column vs TTS lane easy to read.

Usage::

    python test_client_token_sim.py
    python test_client_token_sim.py --file my_script.txt
    python test_client_token_sim.py --chars --speed 0.7
    python test_client_token_sim.py --save --url ws://192.168.1.10:8000/ws/tts
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import re
import sys
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

# Same long-form script as `python test_client.py --demo` — many sentences so LLM↔TTS overlap is obvious.
DEFAULT_SCRIPT = """Welcome to this extended OmniVoice streaming demonstration designed to exercise long-form synthesis.

We are walking through multiple sentences so the server splits the passage into several timed chunks. The goal is to hear continuous speech without awkward gaps while the model keeps working ahead.

Latency sensitive assistants care about time-to-first-audio. After that, smooth pacing matters just as much: listeners tolerate brief waits once playback starts if later sentences arrive early and wait safely in a client-side buffer.

This paragraph stacks coordinating clauses on purpose. Each comma invites another clause; each semicolon marks a deliberate pause in writing even though text-to-speech may glide straight through.

Imagining a voice assistant summarizing your inbox: it streams tokens from an LLM, batches clauses into speech-ready fragments, and hands those fragments to a realtime synthesizer running beside your websocket stack.

Hardware differs wildly across deployments. Some GPUs chew through diffusion steps quickly; others serialize batches conservatively to avoid VRAM spikes. End-to-end tuning blends chunk sizing, scheduler concurrency, and playback buffering here at the edge client.

Developers sometimes chase naive parallelism until kernels collide on shared tensors; bounded concurrency tends to be safer while still overlapping chunk two against chunk one whenever memory permits.

Finally we land enough prose that your ears—not only your eyes—can judge continuity across stitch boundaries. If chunk seams bother you, adjust punctuation splitting upstream or shorten clauses until phonetics blend cleanly."""

# ---------------------------------------------------------------------------
# Audio helpers (aligned with test_client.py)
# ---------------------------------------------------------------------------


def _wav_bytes_to_numpy(wav_bytes: bytes):
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


def _save_wav(chunks: list, sr: int, tag: str) -> Path | None:
    if not _HAS_NUMPY or not _HAS_SOUNDFILE:
        return None
    combined = np.concatenate(chunks, axis=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"token_sim_{ts}_{tag}.wav"
    sf.write(str(out_path), combined, sr)
    return out_path


# ---------------------------------------------------------------------------
# Text → sentences (flush boundaries for server requests)
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_MAGENTA = "\033[35m"

_BAR = "═" * 76


def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def _print_visual_legend(n_sentences: int) -> None:
    print(_c(_BLUE, _BAR), flush=True)
    print(
        _c(_BOLD, "  OmniVoice token-stream simulation")
        + _c(_DIM, f"   ·   {n_sentences} sentences after split   ·   read top-to-bottom as time flows"),
        flush=True,
    )
    print(
        _c(_MAGENTA, "  LLM  ")
        + _c(_DIM, "simulated token stream (left column)"),
        flush=True,
    )
    print(
        _c(_YELLOW, "  TTS  ")
        + _c(_DIM, "requests + synthesis (right-hand messages may interleave with LLM lines)"),
        flush=True,
    )
    print(
        _c(_GREEN, "  PLAY ")
        + _c(_DIM, "speaker plays each response when chunks arrive"),
        flush=True,
    )
    print(_c(_BLUE, _BAR + "\n"), flush=True)


async def simulate_llm_stream(
    full_text: str,
    sentence_queue: "asyncio.Queue[str | None]",
    *,
    char_level: bool,
    speed: float,
) -> None:
    """Print tokens with jitter; enqueue each full sentence for TTS."""
    sentences = split_sentences(full_text)
    n_total = len(sentences)
    if not sentences:
        await sentence_queue.put(None)
        return

    _print_visual_legend(n_total)

    print(
        _c(_MAGENTA + _BOLD, "▶ LLM stream begins")
        + _c(_DIM, "  (watch later sentences appear while audio from earlier lines may still be playing)\n"),
        flush=True,
    )

    for si, sentence in enumerate(sentences):
        sub = _c(_MAGENTA, "═" * 72)
        print(f"\n{sub}", flush=True)
        print(
            _c(_BOLD + _MAGENTA, f"  SIM · sentence {si + 1}/{n_total} ")
            + _c(_DIM, "· tokens:"),
            flush=True,
        )
        print(_c(_MAGENTA, sub + "\n"), flush=True)

        lead = _c(_CYAN, "  │ ")
        print(lead, end="", flush=True)
        if char_level:
            tokens = list(sentence)
            delim = ""
        else:
            tokens = sentence.split()
            delim = " "

        for i, tok in enumerate(tokens):
            piece = (delim if i > 0 else "") + tok if not char_level else tok
            print(piece, end="", flush=True)
            base = 0.012 if char_level else 0.04
            jitter = random.uniform(0.55, 1.45)
            await asyncio.sleep(base * jitter / max(speed, 0.05))

        print(flush=True)
        print(
            _c(_DIM, f"  └─ sentence {si + 1}/{n_total} complete → queued for WebSocket TTS\n"),
            flush=True,
        )
        await sentence_queue.put(sentence)

    print(
        _c(_GREEN + _BOLD, "\n✓ LLM stream finished")
        + _c(_DIM, f"  — all {n_total} sentences were queued for synthesis.\n"),
        flush=True,
    )
    await sentence_queue.put(None)


async def consume_one_tts_response(
    ws,
    *,
    play: bool,
    sentence_idx: int,
) -> tuple[list, int]:
    """Receive deltas until done/error; optionally play sequentially."""
    audio_chunks: list = []
    sample_rate = 24000
    response_done = False

    while not response_done:
        raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
        msg = json.loads(raw)
        msg_type = msg.get("type", "")

        if msg_type == "response.audio.delta":
            wav_bytes = base64.b64decode(msg["delta"])
            audio, sr = _wav_bytes_to_numpy(wav_bytes)
            if audio is not None:
                audio_chunks.append(audio)
                sample_rate = sr

            idx = msg.get("chunk_index", 0)
            ctext = (msg.get("chunk_text") or "")[:52]
            if idx == 0:
                lat = msg.get("first_chunk_latency_ms", 0)
                print(
                    _c(_GREEN, f"  ◆ TTS #{sentence_idx} · audio chunk 1")
                    + f"  latency={lat:.0f}ms"
                    + (f"  «{ctext}…»" if ctext else ""),
                    flush=True,
                )
            else:
                print(
                    _c(_DIM, f"  ◆ TTS #{sentence_idx} · chunk {idx + 1}")
                    + (f"  «{ctext}…»" if ctext else ""),
                    flush=True,
                )

            if play and _HAS_SOUNDDEVICE and audio is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _play_audio, audio, sample_rate)

        elif msg_type == "response.audio.done":
            tot = msg.get("total_audio_ms", 0)
            nchunks = msg.get("total_chunks", 1)
            print(
                _c(_BOLD, f"  ✓ TTS #{sentence_idx} done")
                + _c(_GREEN, f"  audio={tot}ms")
                + _c(_DIM, f"  server_chunks={nchunks}\n"),
                flush=True,
            )
            response_done = True

        elif msg_type == "error":
            print(_c(_RED, f"  [server error] {msg.get('message')}"), flush=True)
            response_done = True

        elif msg_type == "pong":
            pass
        else:
            print(_c(_DIM, f"  [unknown] {msg_type}"), flush=True)

    return audio_chunks, sample_rate


async def tts_worker(
    ws,
    sentence_queue: "asyncio.Queue[str | None]",
    *,
    play: bool,
    save_enabled: bool,
    total_sentences: int,
) -> None:
    req_id = 0
    while True:
        sentence = await sentence_queue.get()
        if sentence is None:
            break

        req_id += 1
        payload = {"type": "tts.request", "text": sentence}
        await ws.send(json.dumps(payload))
        preview = sentence.replace("\n", " ").strip()[:64]
        print(
            _c(_YELLOW + _BOLD, f"\n  ▶ TTS request {req_id}/{total_sentences}")
            + _c(_DIM, f"  · {len(sentence)} chars"),
            flush=True,
        )
        print(_c(_DIM, f"     «{preview}{'…' if len(sentence) > 64 else ''}»\n"), flush=True)

        chunks, sr = await consume_one_tts_response(
            ws, play=play, sentence_idx=req_id
        )

        if save_enabled and chunks and _HAS_NUMPY and _HAS_SOUNDFILE:
            path = _save_wav(chunks, sr, f"s{req_id:03d}")
            if path:
                print(_c(_DIM, f"  Saved → {path}"), flush=True)


async def run_demo(
    url: str,
    script_text: str,
    *,
    play: bool,
    save: bool,
    char_level: bool,
    speed: float,
) -> None:
    print(_c(_BOLD, f"\nConnecting to {url} …"), flush=True)

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=30,
            max_size=50 * 1024 * 1024,
        ) as ws:
            print(_c(_GREEN, "Connected!"), flush=True)
            n_sents = len(split_sentences(script_text))
            print(
                _c(_DIM, f"Script splits into {n_sents} sentences — LLM column and TTS lane interleave.\n"),
                flush=True,
            )

            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            sim_task = asyncio.create_task(
                simulate_llm_stream(
                    script_text,
                    sentence_queue,
                    char_level=char_level,
                    speed=speed,
                )
            )
            try:
                await tts_worker(
                    ws,
                    sentence_queue,
                    play=play,
                    save_enabled=save,
                    total_sentences=n_sents,
                )
            finally:
                await sim_task

    except ConnectionClosed as exc:
        print(_c(_RED, f"\nConnection closed: {exc}"), flush=True)
    except OSError as exc:
        print(_c(_RED, f"\nCannot connect to {url}: {exc}"), flush=True)
        print(_c(_YELLOW, "Start the server:  python server.py"), flush=True)
    except KeyboardInterrupt:
        print(_c(_DIM, "\nInterrupted."))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate streaming LLM tokens → sentence-wise TTS playback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/tts",
        help="WebSocket server URL.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Read script from file (UTF-8).  Default: built-in demo paragraph.",
    )
    parser.add_argument(
        "--chars",
        action="store_true",
        help="Stream character-by-character (finer “token” simulation).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Token delay multiplier (>1 = faster typing simulation).",
    )
    parser.add_argument(
        "--no-play",
        dest="play",
        action="store_false",
        default=True,
        help="Disable speaker playback.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save each sentence response as WAV under streaming_output/.",
    )
    args = parser.parse_args()

    if args.file:
        script_text = args.file.read_text(encoding="utf-8")
    else:
        script_text = DEFAULT_SCRIPT

    if args.play and not _HAS_SOUNDDEVICE:
        print(
            _c(_YELLOW, "[warning] sounddevice not installed — playback disabled.\n"
               "  pip install sounddevice"),
            flush=True,
        )

    asyncio.run(
        run_demo(
            args.url,
            script_text,
            play=args.play,
            save=args.save,
            char_level=args.chars,
            speed=args.speed,
        )
    )


if __name__ == "__main__":
    main()
