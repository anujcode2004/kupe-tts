#!/usr/bin/env python3
"""OmniVoice Streaming Latency Test — measures first-chunk latency + saves 3 audio files.

For every request this script produces THREE audio files:
  1. first_chunk_*.wav   — only the very first chunk (what the user hears first)
  2. rest_chunks_*.wav   — all remaining chunks crossfaded together
  3. full_audio_*.wav    — first + rest stitched with crossfade

It also fires N concurrent WS connections and reports per-request latency
with percentile breakdowns.

Usage
─────
  # Single request — saves 3 WAV files, prints first-chunk latency
  python test_streaming_latency.py

  # 4 concurrent WS connections, 12 total requests
  python test_streaming_latency.py --concurrent-requests 4 --total-requests 12

  # Custom text
  python test_streaming_latency.py --text "Hello world, this is a streaming test."

  # No audio save (latency measurement only)
  python test_streaming_latency.py --no-save

  # Custom server
  python test_streaming_latency.py --url ws://192.168.1.10:8000/ws/tts
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    sys.exit("websockets not installed.  pip install websockets")

try:
    import numpy as np
    import soundfile as sf
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False

OUTPUT_DIR = Path("streaming_output")

DEFAULT_TEXTS = [
    "Hello! Welcome to the OmniVoice low-latency streaming demonstration. "
    "This text is long enough to produce multiple chunks so we can measure "
    "the pipelined first-chunk delivery versus full generation time.",

    "Artificial intelligence is transforming how we build voice assistants. "
    "Real-time synthesis with sub-200 millisecond first-chunk latency is "
    "the new gold standard for conversational applications.",

    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Pack my box with five dozen liquor jugs. How vexingly quick daft "
    "zebras jump! The five boxing wizards jump quickly.",

    "Dynamic batching groups concurrent requests into a single GPU call. "
    "This amortises the fixed overhead of diffusion steps across multiple "
    "users while SageAttention accelerates each attention kernel.",

    "Voice cloning requires only a short reference audio clip to capture "
    "speaker identity. The OmniVoice model encodes timbre, pitch, and "
    "speaking style into a reusable voice-clone prompt object.",
]

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_RESET = "\033[0m"; _BOLD = "\033[1m"; _DIM = "\033[2m"
_GREEN = "\033[32m"; _CYAN = "\033[36m"; _YELLOW = "\033[33m"; _RED = "\033[31m"
def _c(code, s): return f"{code}{s}{_RESET}" if sys.stdout.isatty() else s


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class StreamResult:
    req_id:             int
    text_preview:       str
    first_chunk_ms:     float = 0.0
    first_chunk_gen_ms: float = 0.0
    first_chunk_audio_ms: int = 0
    total_wall_ms:      float = 0.0
    total_audio_ms:     int   = 0
    total_chunks:       int   = 0
    error:              Optional[str] = None
    first_chunk_np:     Optional["np.ndarray"] = None
    rest_chunks_np:     list  = field(default_factory=list)
    sample_rate:        int   = 24000

    @property
    def ok(self) -> bool: return self.error is None


# ---------------------------------------------------------------------------
# Single WS exchange
# ---------------------------------------------------------------------------
async def _ws_exchange(
    url: str, text: str, req_id: int, save: bool,
    language: Optional[str] = None,
    voice:    Optional[str] = None,
    speed:    Optional[float] = None,
) -> StreamResult:
    result = StreamResult(req_id=req_id, text_preview=text[:60])

    try:
        async with websockets.connect(
            url, ping_interval=20, ping_timeout=30, max_size=50 * 1024 * 1024,
        ) as ws:
            t0 = time.perf_counter()
            request_msg: dict = {"type": "tts.request", "text": text}
            if language:
                request_msg["language"] = language
            if voice:
                request_msg["voice"] = voice
            if speed is not None:
                request_msg["speed"] = speed
            await ws.send(json.dumps(request_msg))

            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                msg = json.loads(raw)
                mt = msg.get("type", "")

                if mt == "response.audio.delta":
                    wav_bytes = base64.b64decode(msg["delta"])
                    sr = msg.get("sample_rate", 24000)
                    result.sample_rate = sr
                    chunk_idx = msg.get("chunk_index", 0)

                    audio_np = None
                    if _HAS_AUDIO:
                        buf = io.BytesIO(wav_bytes)
                        audio_np, _ = sf.read(buf, dtype="float32")

                    if chunk_idx == 0:
                        result.first_chunk_ms = msg.get(
                            "first_chunk_latency_ms",
                            (time.perf_counter() - t0) * 1000,
                        )
                        result.first_chunk_gen_ms = msg.get("chunk_gen_ms", 0)
                        result.first_chunk_audio_ms = msg.get("chunk_audio_ms", 0)
                        if audio_np is not None:
                            result.first_chunk_np = audio_np
                    else:
                        if audio_np is not None:
                            result.rest_chunks_np.append(audio_np)

                elif mt == "response.audio.done":
                    result.total_wall_ms = (time.perf_counter() - t0) * 1000
                    result.total_audio_ms = msg.get("total_audio_ms", 0)
                    result.total_chunks = msg.get("total_chunks", 0)
                    break

                elif mt == "error":
                    result.error = msg.get("message", "unknown error")
                    result.total_wall_ms = (time.perf_counter() - t0) * 1000
                    return result

    except Exception as exc:
        result.error = str(exc)
        return result

    # Save 3 audio files
    if save and _HAS_AUDIO and result.first_chunk_np is not None:
        _save_three_audios(result)

    return result


def _save_three_audios(r: StreamResult) -> None:
    """Save first_chunk, rest_chunks, and full_audio as separate WAV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"req{r.req_id:03d}_{ts}"
    sr = r.sample_rate

    # 1. First chunk only
    p1 = OUTPUT_DIR / f"first_chunk_{tag}.wav"
    sf.write(str(p1), r.first_chunk_np, sr)
    print(_c(_DIM, f"    Saved first_chunk  → {p1}  ({len(r.first_chunk_np)/sr*1000:.0f}ms)"))

    # 2. Remaining chunks combined
    if r.rest_chunks_np:
        combined_rest = np.concatenate(r.rest_chunks_np)
        p2 = OUTPUT_DIR / f"rest_chunks_{tag}.wav"
        sf.write(str(p2), combined_rest, sr)
        print(_c(_DIM, f"    Saved rest_chunks  → {p2}  ({len(combined_rest)/sr*1000:.0f}ms)"))
    else:
        combined_rest = np.array([], dtype=np.float32)

    # 3. Full audio = first + rest (simple concatenation; server already crossfaded)
    all_parts = [r.first_chunk_np]
    if len(combined_rest) > 0:
        all_parts.append(combined_rest)
    full = np.concatenate(all_parts)
    p3 = OUTPUT_DIR / f"full_audio_{tag}.wav"
    sf.write(str(p3), full, sr)
    print(_c(_DIM, f"    Saved full_audio   → {p3}  ({len(full)/sr*1000:.0f}ms)"))


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------
def _pct(vals, p):
    if not vals: return 0.0
    s = sorted(vals)
    return s[max(0, int(len(s) * p / 100) - 1)]


# ---------------------------------------------------------------------------
# Load test runner
# ---------------------------------------------------------------------------
async def run_test(
    url: str, concurrent: int, total: int, texts: list[str], save: bool,
    language: Optional[str] = None,
    voice:    Optional[str] = None,
    speed:    Optional[float] = None,
) -> list[StreamResult]:
    sem = asyncio.Semaphore(concurrent)
    results: list[StreamResult] = []
    lock = asyncio.Lock()

    async def _one(i: int):
        text = texts[i % len(texts)]
        async with sem:
            r = await _ws_exchange(url, text, i, save, language, voice, speed)
        async with lock:
            results.append(r)
            _print_one(r, len(results), total)

    tasks = [asyncio.create_task(_one(i)) for i in range(total)]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


def _print_one(r: StreamResult, done: int, total: int):
    if r.ok:
        print(
            f"  {_c(_GREEN, '✓')} [{done:>3}/{total}] "
            f"req={r.req_id:<3}  "
            f"first_chunk={_c(_CYAN, f'{r.first_chunk_ms:>6.0f}ms')}  "
            f"first_gen={r.first_chunk_gen_ms:>5.0f}ms  "
            f"first_audio={r.first_chunk_audio_ms:>5}ms  "
            f"total_wall={r.total_wall_ms:>7.0f}ms  "
            f"total_audio={r.total_audio_ms:>6}ms  "
            f"chunks={r.total_chunks}",
            flush=True,
        )
    else:
        print(
            f"  {_c(_RED, '✗')} [{done:>3}/{total}] req={r.req_id}  "
            + _c(_RED, f"ERROR: {r.error}"),
            flush=True,
        )


def _print_summary(results: list[StreamResult], elapsed: float, concurrent: int):
    ok = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]

    fc = [r.first_chunk_ms for r in ok]
    fg = [r.first_chunk_gen_ms for r in ok]
    tw = [r.total_wall_ms for r in ok]
    ta = [r.total_audio_ms for r in ok]

    print()
    print(_c(_BOLD, "─" * 70))
    print(_c(_BOLD, " STREAMING LATENCY SUMMARY"))
    print(_c(_BOLD, "─" * 70))
    print(f"  Total requests     : {len(results)}  (ok={len(ok)}  err={len(err)})")
    print(f"  Concurrency        : {concurrent}")
    print(f"  Wall time          : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  Throughput         : {len(ok)/elapsed:.2f} req/s")

    if ok:
        hdr = f"  {'Metric':<24}  {'P50':>8}  {'P75':>8}  {'P95':>8}  {'P99':>8}  {'avg':>8}"
        print(_c(_BOLD, hdr))
        print("  " + "─" * 66)

        def row(label, vals, unit="ms"):
            if not vals: return
            avg = sum(vals) / len(vals)
            print(
                f"  {label:<24}  "
                f"{_pct(vals,50):>7.0f}{unit}  "
                f"{_pct(vals,75):>7.0f}{unit}  "
                f"{_pct(vals,95):>7.0f}{unit}  "
                f"{_pct(vals,99):>7.0f}{unit}  "
                f"{avg:>7.0f}{unit}"
            )

        row("First chunk latency",  fc)
        row("First chunk gen",      fg)
        row("Total wall time",      tw)
        row("Total audio",          ta)
        print()
        print(f"  Total audio synthesised: {sum(ta)/1000:.1f}s")

    if err:
        print(_c(_RED, f"\n  {len(err)} request(s) failed:"))
        for r in err:
            print(_c(_RED, f"    req={r.req_id}  {r.error}"))

    print(_c(_BOLD, "─" * 70))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="OmniVoice streaming latency test — 3 audio outputs per request",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default="ws://localhost:8000/ws/tts")
    p.add_argument("--concurrent-requests", "-c", type=int, default=1, metavar="N")
    p.add_argument("--total-requests", "-n", type=int, default=1, metavar="M")
    p.add_argument("--text", type=str, default=None,
                   help="Custom text (overrides built-in corpus).")
    p.add_argument("--language", "-l", type=str, default="auto", metavar="CODE",
                   help=(
                       "Language for synthesis. Default 'auto' lets OmniVoice "
                       "detect from text + reference voice. Accepts ISO-639-3 "
                       "codes ('en', 'hi', 'gu', 'pa', 'bn', 'ta', 'te', "
                       "'mr', 'zh', 'ja', 'ko', …) or English names "
                       "('English', 'Hindi', 'Gujarati', 'Panjabi', …)."
                   ))
    p.add_argument("--voice", "-v", type=str, default=None, metavar="NAME",
                   help=(
                       "Voice profile name (e.g. 'ajay', 'soham'). The voice must "
                       "be preloaded server-side. Omit to use the server's default."
                   ))
    p.add_argument("--speed", "-s", type=float, default=None, metavar="X",
                   help=(
                       "Speaking-speed multiplier (0.25–3.0). 1.0=normal, "
                       "<1.0 slower, >1.0 faster. Omit for server default."
                   ))
    p.add_argument("--no-save", dest="save", action="store_false", default=True)
    p.add_argument("--no-header", action="store_true")
    args = p.parse_args()

    texts    = [args.text] if args.text else DEFAULT_TEXTS
    language = None if args.language.lower() in ("auto", "none", "") else args.language
    voice    = args.voice if args.voice else None
    speed    = args.speed
    if speed is not None and not (0.25 <= speed <= 3.0):
        sys.exit(f"--speed must be between 0.25 and 3.0 (got {speed}).")

    if not args.no_header:
        print(_c(_BOLD, "\n╔══ OmniVoice Streaming Latency Test ══╗"))
        print(f"  server          : {args.url}")
        print(f"  concurrent      : {args.concurrent_requests}")
        print(f"  total requests  : {args.total_requests}")
        print(f"  language        : {language or 'auto (server default)'}")
        print(f"  voice           : {voice    or '(server default)'}")
        print(f"  speed           : {speed if speed is not None else '(server default)'}")
        print(f"  save 3 audios   : {args.save}")
        print(_c(_BOLD, "╚══════════════════════════════════════╝\n"))

    t0 = time.perf_counter()
    results = asyncio.run(run_test(
        url=args.url,
        concurrent=args.concurrent_requests,
        total=args.total_requests,
        texts=texts,
        save=args.save,
        language=language,
        voice=voice,
        speed=speed,
    ))
    elapsed = time.perf_counter() - t0

    _print_summary(results, elapsed, args.concurrent_requests)

    if args.save and _HAS_AUDIO:
        saved = sum(1 for r in results if r.ok and r.first_chunk_np is not None)
        if saved:
            print(f"\n  Saved 3 audio files per request → {OUTPUT_DIR}/")
            print(f"    first_chunk_*.wav  — what the user hears FIRST")
            print(f"    rest_chunks_*.wav  — remaining chunks combined")
            print(f"    full_audio_*.wav   — full stitched audio\n")

    sys.exit(0 if all(r.ok for r in results) else 1)


if __name__ == "__main__":
    main()
