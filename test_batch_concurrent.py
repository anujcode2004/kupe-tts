#!/usr/bin/env python3
"""OmniVoice Concurrent Batch Inference Test Client

Fires N concurrent requests to the server's batch inference endpoint and
measures per-request latency, audio duration, RTF, and aggregate percentiles.

This stresses both the server's DynamicBatcher (groups concurrent requests into
single model.generate() calls) and the ProcessPoolExecutor worker throughput.

Usage examples
──────────────
  # 4 concurrent requests, 12 total, 1 text each
  python test_batch_concurrent.py --concurrent-requests 4 --total-requests 12

  # 3 concurrent, 9 total, 3 texts per request (batch size=3 each time)
  python test_batch_concurrent.py --concurrent-requests 3 --total-requests 9 --texts-per-request 3

  # Hit WebSocket endpoint instead of HTTP batch
  python test_batch_concurrent.py --concurrent-requests 4 --mode ws

  # Save output WAVs
  python test_batch_concurrent.py --concurrent-requests 4 --save

  # Custom server URL
  python test_batch_concurrent.py --url http://192.168.1.10:8000 --concurrent-requests 5

Metrics reported
────────────────
  Wall latency   : total round-trip per request (client clock)
  Server gen ms  : server-reported generation time (inside model.generate())
  Audio ms       : duration of returned audio
  RTF            : server_gen_ms / audio_ms  (lower = faster)

Percentiles: P50, P75, P95, P99 across all completed requests.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

try:
    import numpy as np
    import soundfile as sf
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False

# ---------------------------------------------------------------------------
# Default texts to synthesise (varied length to stress the batcher)
# ---------------------------------------------------------------------------
DEFAULT_TEXTS = [
    "Hello!",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we interact with technology.",
    "Streaming text-to-speech enables real-time voice assistants at scale.",
    "Batch inference maximises GPU utilisation across concurrent requests.",
    "Dynamic batching groups multiple requests into a single model call.",
    "SageAttention accelerates transformer attention with optimised CUDA kernels.",
    "ProcessPoolExecutor isolates GPU workers from the async event loop.",
    "Voice cloning requires only a short reference audio clip.",
    "OmniVoice supports multilingual synthesis with a single unified model.",
    "Low first-chunk latency is critical for responsive voice applications.",
    "The crossfade blending hides stitch boundaries between audio chunks.",
    "Server-side audio processing reduces client complexity significantly.",
    "FastAPI and WebSockets together provide a clean real-time API surface.",
    "Uvicorn's async workers handle thousands of concurrent connections.",
    "Sentence-boundary chunking preserves natural prosody across splits.",
    "Reference text transcripts improve voice cloning accuracy and naturalness.",
    "Production TTS pipelines balance latency, quality, and throughput.",
]

OUTPUT_DIR = Path("batch_output")

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"

def _c(code: str, s: str) -> str:
    return f"{code}{s}{_RESET}" if sys.stdout.isatty() else s


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    req_id:       int
    texts:        list[str]
    t_start:      float
    t_end:        float        = 0.0
    wall_ms:      float        = 0.0
    server_gen_ms: float       = 0.0
    audio_ms:     int          = 0
    batch_size:   int          = 0
    error:        Optional[str] = None
    saved_paths:  list[str]    = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.server_gen_ms / self.audio_ms if self.audio_ms > 0 else float("inf")

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# HTTP batch mode
# ---------------------------------------------------------------------------
async def _http_request(
    session: "aiohttp.ClientSession",
    url: str,
    texts: list[str],
    req_id: int,
    save: bool,
    use_high_quality: bool,
    language: Optional[str] = None,
    voice:    Optional[str] = None,
    speed:    Optional[float] = None,
) -> RequestResult:
    result = RequestResult(req_id=req_id, texts=texts, t_start=time.perf_counter())
    payload: dict = {"texts": texts, "use_high_quality": use_high_quality}
    if language:
        payload["language"] = language
    if voice:
        payload["voice"] = voice
    if speed is not None:
        payload["speed"] = speed

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                result.t_end  = time.perf_counter()
                result.wall_ms = (result.t_end - result.t_start) * 1000
                return result

            data = await resp.json()

        result.t_end      = time.perf_counter()
        result.wall_ms    = (result.t_end - result.t_start) * 1000
        result.server_gen_ms = data.get("total_gen_ms", 0.0)
        result.batch_size    = data.get("batch_size", len(texts))
        result.audio_ms      = sum(
            item.get("audio_ms", 0) for item in data.get("results", [])
        )

        if save and _HAS_AUDIO:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            for item in data.get("results", []):
                b64 = item.get("audio_base64", "")
                if not b64:
                    continue
                wav_bytes = base64.b64decode(b64)
                sr = item.get("sample_rate", 24000)
                buf = io.BytesIO(wav_bytes)
                audio, _ = sf.read(buf, dtype="float32")
                path = OUTPUT_DIR / f"req{req_id:04d}_item{item['id']:02d}_{ts}.wav"
                sf.write(str(path), audio, sr)
                result.saved_paths.append(str(path))

    except Exception as exc:
        result.error   = str(exc)
        result.t_end   = time.perf_counter()
        result.wall_ms = (result.t_end - result.t_start) * 1000

    return result


# ---------------------------------------------------------------------------
# WebSocket single-request mode
# ---------------------------------------------------------------------------
async def _ws_request(
    url: str,
    text: str,
    req_id: int,
    save: bool,
    language: Optional[str] = None,
    voice:    Optional[str] = None,
    speed:    Optional[float] = None,
) -> RequestResult:
    result = RequestResult(req_id=req_id, texts=[text], t_start=time.perf_counter())
    audio_chunks: list = []
    sr = 24000

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=30,
            max_size=50 * 1024 * 1024,
        ) as ws:
            request_msg: dict = {"type": "tts.request", "text": text}
            if language:
                request_msg["language"] = language
            if voice:
                request_msg["voice"] = voice
            if speed is not None:
                request_msg["speed"] = speed
            await ws.send(json.dumps(request_msg))

            first_latency = None
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                msg = json.loads(raw)
                mt  = msg.get("type", "")

                if mt == "response.audio.delta":
                    wav_bytes = base64.b64decode(msg["delta"])
                    if _HAS_AUDIO:
                        buf = io.BytesIO(wav_bytes)
                        chunk, sr = sf.read(buf, dtype="float32")
                        audio_chunks.append(chunk)
                    if msg.get("chunk_index") == 0:
                        first_latency = msg.get("first_chunk_latency_ms", 0)

                elif mt == "response.audio.done":
                    result.t_end      = time.perf_counter()
                    result.wall_ms    = (result.t_end - result.t_start) * 1000
                    result.server_gen_ms = msg.get("total_gen_ms", 0.0)
                    result.audio_ms   = msg.get("total_audio_ms", 0)
                    result.batch_size = 1
                    break

                elif mt == "error":
                    result.error  = msg.get("message", "unknown server error")
                    result.t_end  = time.perf_counter()
                    result.wall_ms = (result.t_end - result.t_start) * 1000
                    return result

    except Exception as exc:
        result.error   = str(exc)
        result.t_end   = time.perf_counter()
        result.wall_ms = (result.t_end - result.t_start) * 1000
        return result

    if save and _HAS_AUDIO and audio_chunks:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"req{req_id:04d}_ws_{ts}.wav"
        import numpy as np
        sf.write(str(path), np.concatenate(audio_chunks), sr)
        result.saved_paths.append(str(path))

    return result


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return sorted_v[idx]


# ---------------------------------------------------------------------------
# Run the load test
# ---------------------------------------------------------------------------
async def run_load_test(
    base_url:          str,
    mode:              str,              # "http" or "ws"
    concurrent:        int,
    total_requests:    int,
    texts_per_request: int,
    use_high_quality:  bool,
    save:              bool,
    language:          Optional[str] = None,
    voice:             Optional[str] = None,
    speed:             Optional[float] = None,
) -> list[RequestResult]:
    """Fire `total_requests` requests with at most `concurrent` in-flight at once."""

    sem     = asyncio.Semaphore(concurrent)
    results: list[RequestResult] = []
    lock    = asyncio.Lock()

    async def _run_one(req_id: int) -> None:
        texts = [
            DEFAULT_TEXTS[(req_id * texts_per_request + i) % len(DEFAULT_TEXTS)]
            for i in range(texts_per_request)
        ]

        async with sem:
            if mode == "ws":
                if not _HAS_WS:
                    raise RuntimeError("websockets package not installed.")
                ws_url  = base_url.replace("http://", "ws://").replace("https://", "wss://")
                ws_url  = ws_url.rstrip("/") + "/ws/tts"
                result  = await _ws_request(
                    ws_url, texts[0], req_id, save, language, voice, speed,
                )
            else:
                if not _HAS_AIOHTTP:
                    raise RuntimeError("aiohttp package not installed. pip install aiohttp")
                http_url = base_url.rstrip("/") + "/api/tts/batch"
                async with aiohttp.ClientSession() as session:
                    result = await _http_request(
                        session, http_url, texts, req_id, save,
                        use_high_quality, language, voice, speed,
                    )

        async with lock:
            results.append(result)
            _print_result(result, len(results), total_requests)

    tasks = [asyncio.create_task(_run_one(i)) for i in range(total_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


# ---------------------------------------------------------------------------
# Per-result print
# ---------------------------------------------------------------------------
def _print_result(r: RequestResult, done: int, total: int) -> None:
    status = _c(_GREEN, "✓") if r.succeeded else _c(_RED, "✗")
    text_preview = r.texts[0][:55] + ("…" if len(r.texts[0]) > 55 else "")
    if r.succeeded:
        print(
            f"  {status} [{done:>3}/{total}] "
            f"req={r.req_id:<4}  "
            f"wall={_c(_CYAN,   f'{r.wall_ms:>7.0f}ms')}  "
            f"gen={_c(_YELLOW,  f'{r.server_gen_ms:>7.0f}ms')}  "
            f"audio={r.audio_ms:>6}ms  "
            f"RTF={r.rtf:.3f}  "
            f"texts={r.batch_size}  "
            f"'{text_preview}'",
            flush=True,
        )
    else:
        print(
            f"  {status} [{done:>3}/{total}] req={r.req_id:<4}  "
            + _c(_RED, f"ERROR: {r.error}"),
            flush=True,
        )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _print_summary(results: list[RequestResult], elapsed: float, concurrent: int) -> None:
    ok  = [r for r in results if r.succeeded]
    err = [r for r in results if not r.succeeded]

    wall_vals      = [r.wall_ms       for r in ok]
    gen_vals       = [r.server_gen_ms for r in ok]
    audio_vals     = [r.audio_ms      for r in ok]
    rtf_vals       = [r.rtf           for r in ok if r.audio_ms > 0]
    total_audio_ms = sum(audio_vals)
    total_gen_ms   = sum(gen_vals)

    print()
    print(_c(_BOLD, "─" * 62))
    print(_c(_BOLD, " LATENCY SUMMARY"))
    print(_c(_BOLD, "─" * 62))
    print(f"  Total requests   : {len(results)}  (ok={len(ok)}  err={len(err)})")
    print(f"  Concurrency      : {concurrent}")
    print(f"  Wall time        : {elapsed:.2f}s")
    print(f"  Throughput       : {len(ok)/elapsed:.2f} req/s")
    print()

    if ok:
        header = f"  {'Metric':<22}  {'P50':>8}  {'P75':>8}  {'P95':>8}  {'P99':>8}  {'avg':>8}"
        print(_c(_BOLD, header))
        print("  " + "─" * 60)

        def row(label: str, vals: list[float], unit: str = "ms") -> None:
            if not vals:
                return
            avg = sum(vals) / len(vals)
            print(
                f"  {label:<22}  "
                f"{_pct(vals,50):>7.0f}{unit}  "
                f"{_pct(vals,75):>7.0f}{unit}  "
                f"{_pct(vals,95):>7.0f}{unit}  "
                f"{_pct(vals,99):>7.0f}{unit}  "
                f"{avg:>7.0f}{unit}"
            )

        row("Wall latency",   wall_vals)
        row("Server gen",     gen_vals)
        row("Audio duration", audio_vals)
        if rtf_vals:
            avg_rtf = sum(rtf_vals) / len(rtf_vals)
            p50_rtf = _pct(rtf_vals, 50)
            p95_rtf = _pct(rtf_vals, 95)
            print(
                f"  {'RTF (gen/audio)':<22}  "
                f"{p50_rtf:>8.3f}  "
                f"{'':>8}  "
                f"{p95_rtf:>8.3f}  "
                f"{'':>8}  "
                f"{avg_rtf:>8.3f}"
            )

        print()
        print(
            f"  Total audio synthesised : {total_audio_ms/1000:.1f}s  "
            f"({total_audio_ms}ms)"
        )
        print(
            f"  Total server gen time   : {total_gen_ms/1000:.1f}s  "
            f"({total_gen_ms:.0f}ms)"
        )
        if total_audio_ms > 0:
            overall_rtf = total_gen_ms / total_audio_ms
            print(f"  Overall RTF             : {overall_rtf:.4f}")

    if err:
        print()
        print(_c(_RED, f"  {len(err)} request(s) failed:"))
        for r in err:
            print(_c(_RED, f"    req={r.req_id}  {r.error}"))

    print(_c(_BOLD, "─" * 62))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OmniVoice concurrent batch inference stress test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--concurrent-requests", "-c",
        type=int,
        default=4,
        metavar="N",
        help="Number of requests in flight simultaneously.",
    )
    p.add_argument(
        "--total-requests", "-n",
        type=int,
        default=12,
        metavar="M",
        help="Total number of requests to send.",
    )
    p.add_argument(
        "--texts-per-request", "-t",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Texts per HTTP batch request (ignored in ws mode). "
            "K texts → one POST /api/tts/batch with K items."
        ),
    )
    p.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server base URL.",
    )
    p.add_argument(
        "--mode",
        choices=["http", "ws"],
        default="http",
        help="http → POST /api/tts/batch  |  ws → WS /ws/tts (one text per request).",
    )
    p.add_argument(
        "--high-quality",
        action="store_true",
        default=False,
        help="Use high-quality config (guidance_scale=2.0, postprocess=True).",
    )
    p.add_argument(
        "--language", "-l",
        type=str,
        default="auto",
        metavar="CODE",
        help=(
            "Language for synthesis. Default 'auto' lets OmniVoice detect "
            "from text + reference voice. Accepts ISO-639-3 codes "
            "('en', 'hi', 'gu', 'pa', 'bn', 'ta', 'te', 'mr', 'zh', 'ja', "
            "'ko', …) or English names ('English', 'Hindi', 'Gujarati', "
            "'Panjabi', 'Chinese', …)."
        ),
    )
    p.add_argument(
        "--voice", "-v",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Voice profile name to use (e.g. 'ajay', 'soham'). The voice must "
            "be preloaded on the server (see OMNIVOICE_VOICE_PROFILES). "
            "Omit this flag to use the server's default voice."
        ),
    )
    p.add_argument(
        "--speed", "-s",
        type=float,
        default=None,
        metavar="X",
        help=(
            "Speaking-speed multiplier (0.25–3.0). 1.0 = normal, "
            "<1.0 slower, >1.0 faster. Omit to let the server pick "
            "(OMNIVOICE_DEFAULT_SPEED, normally 1.0/model default)."
        ),
    )
    p.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save returned WAV files under batch_output/.",
    )
    p.add_argument(
        "--no-header",
        action="store_true",
        help="Skip the run configuration header.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "http" and not _HAS_AIOHTTP:
        sys.exit(
            "aiohttp is required for HTTP mode.\n"
            "  Install: pip install aiohttp"
        )
    if args.mode == "ws" and not _HAS_WS:
        sys.exit(
            "websockets is required for WS mode.\n"
            "  Install: pip install websockets"
        )

    language = None if args.language.lower() in ("auto", "none", "") else args.language
    voice    = args.voice if args.voice else None
    speed    = args.speed
    if speed is not None and not (0.25 <= speed <= 3.0):
        sys.exit(f"--speed must be between 0.25 and 3.0 (got {speed}).")

    if not args.no_header:
        print(_c(_BOLD, "\n╔══ OmniVoice Concurrent Batch Test ══╗"))
        print(f"  server          : {args.url}")
        print(f"  mode            : {args.mode}")
        print(f"  concurrent      : {args.concurrent_requests}")
        print(f"  total requests  : {args.total_requests}")
        if args.mode == "http":
            print(f"  texts/request   : {args.texts_per_request}")
        print(f"  language        : {language or 'auto (server default)'}")
        print(f"  voice           : {voice    or '(server default)'}")
        print(f"  speed           : {speed if speed is not None else '(server default)'}")
        print(f"  high quality    : {args.high_quality}")
        print(f"  save wavs       : {args.save}")
        print(_c(_BOLD, "╚═════════════════════════════════════╝\n"))

    t0 = time.perf_counter()
    results = asyncio.run(
        run_load_test(
            base_url=args.url,
            mode=args.mode,
            concurrent=args.concurrent_requests,
            total_requests=args.total_requests,
            texts_per_request=args.texts_per_request,
            use_high_quality=args.high_quality,
            save=args.save,
            language=language,
            voice=voice,
            speed=speed,
        )
    )
    elapsed = time.perf_counter() - t0

    _print_summary(results, elapsed, args.concurrent_requests)

    if args.save:
        saved = [p for r in results for p in r.saved_paths]
        if saved:
            print(f"\n  Saved {len(saved)} WAV file(s) → {OUTPUT_DIR}/")

    # Exit with non-zero if any request failed
    errors = sum(1 for r in results if not r.succeeded)
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
