#!/usr/bin/env python3
"""OmniVoice realtime WebSocket player.

Sends text to ``/ws/tts`` and starts playback immediately when the first audio
chunk arrives.  Later chunks are queued into one continuous ``sounddevice``
output stream, so playback keeps running while the server continues generating.

Usage
-----
  python test_ws_realtime_player.py --text "Hello, this is realtime playback."
  python test_ws_realtime_player.py --text-file speech.txt --save
  python test_ws_realtime_player.py --url ws://192.168.1.10:8000/ws/tts

Requirements
------------
  pip install websockets numpy soundfile sounddevice
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not installed. Run: pip install numpy")

try:
    import soundfile as sf
except ImportError:
    sys.exit("soundfile not installed. Run: pip install soundfile")

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    sys.exit("websockets not installed. Run: pip install websockets")

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False


OUTPUT_DIR = Path("streaming_output")

DEFAULT_TEXT = (
    "नमस्ते! यह है स्वाद और सेहत से भरपूर सीताफल, "
    "જેનો મીઠો અને ક્રીમી સ્વાદ દરેકને ખૂબ ગમે છે અને જે કુદરતી ઊર્જાનો ઉત્તમ સ્ત્રોત છે। "
    "The first chunk should start playing as soon as it arrives, while the "
    "remaining chunks continue to stream and play continuously."
)


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text


def _decode_wav_b64(delta: str) -> tuple[np.ndarray, int]:
    """Decode base64 WAV payload from the server into mono float32 audio."""
    wav_bytes = base64.b64decode(delta)
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), sr


class RealtimeAudioPlayer:
    """Continuous low-latency audio player backed by a sounddevice callback.

    Chunks are added with ``enqueue`` as they arrive from the WebSocket.  The
    callback drains the queue without restarting the device, preventing the gaps
    you get from repeated ``sd.play(...); sd.wait()`` calls.
    """

    def __init__(self, *, queue_max: int = 64) -> None:
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_max)
        self._stream: Optional["sd.OutputStream"] = None
        self._current: Optional[np.ndarray] = None
        self._pos = 0
        self._started = threading.Event()
        self._closed = False

    def start(self, sample_rate: int) -> None:
        if not _HAS_SOUNDDEVICE:
            return
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
            latency="low",
        )
        self._stream.start()
        self._started.set()

    def enqueue(self, audio: np.ndarray) -> None:
        if not _HAS_SOUNDDEVICE or self._closed:
            return
        self._queue.put(audio)

    def finish(self) -> None:
        """Signal end-of-stream and wait until queued audio has drained."""
        if not _HAS_SOUNDDEVICE or self._stream is None:
            return
        self._queue.put(None)
        while not self._closed:
            time.sleep(0.02)

    def close(self) -> None:
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def _callback(self, outdata, frames, _time_info, _status) -> None:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0

        while filled < frames:
            if self._current is None or self._pos >= len(self._current):
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._closed = True
                    break
                self._current = nxt
                self._pos = 0

            remaining_out = frames - filled
            remaining_chunk = len(self._current) - self._pos
            n = min(remaining_out, remaining_chunk)
            out[filled : filled + n] = self._current[self._pos : self._pos + n]
            self._pos += n
            filled += n

        outdata[:, 0] = out


@dataclass
class PlaybackStats:
    first_chunk_latency_ms: float = 0.0
    first_chunk_gen_ms: float = 0.0
    first_chunk_audio_ms: int = 0
    total_wall_ms: float = 0.0
    total_audio_ms: int = 0
    total_chunks: int = 0
    sample_rate: int = 24000
    chunks: list[np.ndarray] = field(default_factory=list)


async def stream_and_play(
    *,
    url: str,
    text: str,
    play: bool,
    save: bool,
    output_dir: Path,
    recv_timeout: float,
    language: Optional[str] = None,
    voice:    Optional[str] = None,
    speed:    Optional[float] = None,
) -> PlaybackStats:
    stats = PlaybackStats()
    player = RealtimeAudioPlayer()
    t_send = time.perf_counter()
    first_received = False

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=30,
            max_size=50 * 1024 * 1024,
        ) as ws:
            payload: dict = {"type": "tts.request", "text": text}
            if language:
                payload["language"] = language
            if voice:
                payload["voice"] = voice
            if speed is not None:
                payload["speed"] = speed
            await ws.send(json.dumps(payload))
            print(_c(_DIM, "Sent text. Waiting for first audio chunk..."), flush=True)

            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "response.audio.delta":
                    audio, sr = _decode_wav_b64(msg["delta"])
                    stats.sample_rate = sr
                    stats.chunks.append(audio)

                    chunk_index = int(msg.get("chunk_index", len(stats.chunks) - 1))
                    chunk_audio_ms = int(msg.get("chunk_audio_ms", round(len(audio) / sr * 1000)))
                    chunk_gen_ms = float(msg.get("chunk_gen_ms", 0.0))

                    if play:
                        player.start(sr)
                        player.enqueue(audio)

                    if chunk_index == 0 and not first_received:
                        first_received = True
                        stats.first_chunk_latency_ms = float(
                            msg.get(
                                "first_chunk_latency_ms",
                                (time.perf_counter() - t_send) * 1000,
                            )
                        )
                        stats.first_chunk_gen_ms = chunk_gen_ms
                        stats.first_chunk_audio_ms = chunk_audio_ms
                        wall_ms = (time.perf_counter() - t_send) * 1000
                        print(
                            _c(_GREEN, "\nFIRST CHUNK PLAYING NOW")
                            + f"  server_latency={stats.first_chunk_latency_ms:.1f}ms"
                            + f"  client_wall={wall_ms:.1f}ms"
                            + f"  gen={chunk_gen_ms:.1f}ms"
                            + f"  audio={chunk_audio_ms}ms\n",
                            flush=True,
                        )
                    else:
                        print(
                            _c(_CYAN, f"chunk {chunk_index + 1}")
                            + f"  gen={chunk_gen_ms:.1f}ms"
                            + f"  audio={chunk_audio_ms}ms"
                            + f"  queued_for_playback={play}",
                            flush=True,
                        )

                elif msg_type == "response.audio.done":
                    stats.total_wall_ms = (time.perf_counter() - t_send) * 1000
                    stats.total_audio_ms = int(msg.get("total_audio_ms", 0))
                    stats.total_chunks = int(msg.get("total_chunks", len(stats.chunks)))
                    print(
                        _c(_BOLD, "\nDONE")
                        + f"  chunks={stats.total_chunks}"
                        + f"  total_audio={stats.total_audio_ms}ms"
                        + f"  total_wall={stats.total_wall_ms:.1f}ms"
                        + f"  first_chunk={stats.first_chunk_latency_ms:.1f}ms",
                        flush=True,
                    )
                    break

                elif msg_type == "error":
                    raise RuntimeError(msg.get("message", "unknown server error"))

                elif msg_type == "pong":
                    continue

                else:
                    print(_c(_YELLOW, f"Unknown server message: {msg_type}"), flush=True)

    except ConnectionClosed as exc:
        raise RuntimeError(f"WebSocket closed: {exc}") from exc
    finally:
        if play:
            print(_c(_DIM, "Draining playback queue..."), flush=True)
            player.finish()
            player.close()

    if save and stats.chunks:
        save_outputs(stats, output_dir)

    return stats


def save_outputs(stats: PlaybackStats, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sr = stats.sample_rate

    first = stats.chunks[0]
    rest = np.concatenate(stats.chunks[1:]) if len(stats.chunks) > 1 else np.array([], dtype=np.float32)
    full = np.concatenate(stats.chunks)

    p_first = output_dir / f"realtime_first_chunk_{ts}.wav"
    p_rest = output_dir / f"realtime_rest_chunks_{ts}.wav"
    p_full = output_dir / f"realtime_full_audio_{ts}.wav"

    sf.write(str(p_first), first, sr)
    sf.write(str(p_rest), rest, sr)
    sf.write(str(p_full), full, sr)

    print("\nSaved audio files:", flush=True)
    print(f"  first chunk : {p_first}", flush=True)
    print(f"  rest chunks : {p_rest}", flush=True)
    print(f"  full audio  : {p_full}", flush=True)


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return args.text.strip() if args.text else DEFAULT_TEXT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime OmniVoice WebSocket player.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts")
    parser.add_argument("--text", default=None, help="Text to synthesize.")
    parser.add_argument("--text-file", default=None, help="UTF-8 text file to synthesize.")
    parser.add_argument(
        "--language", "-l", default="auto", metavar="CODE",
        help=(
            "Language for synthesis. Default 'auto' lets OmniVoice detect "
            "from text + reference voice. Accepts ISO-639-3 codes "
            "('en', 'hi', 'gu', 'pa', 'bn', 'ta', 'te', 'mr', 'zh', 'ja', "
            "'ko', …) or English names ('English', 'Hindi', 'Gujarati', "
            "'Panjabi', 'Chinese', …)."
        ),
    )
    parser.add_argument(
        "--voice", "-v", default=None, metavar="NAME",
        help=(
            "Voice profile name (e.g. 'ajay', 'soham'). The voice must "
            "be preloaded server-side. Omit to use the server's default."
        ),
    )
    parser.add_argument(
        "--speed", "-s", type=float, default=None, metavar="X",
        help=(
            "Speaking-speed multiplier (0.25–3.0). 1.0 = normal pace, "
            "<1.0 slower, >1.0 faster. Omit to use the server's default."
        ),
    )
    parser.add_argument("--no-play", dest="play", action="store_false", default=True)
    parser.add_argument("--save", action="store_true", help="Save first/rest/full WAV files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--recv-timeout", type=float, default=120.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.play and not _HAS_SOUNDDEVICE:
        print(
            _c(_YELLOW, "sounddevice not installed, disabling playback. Install: pip install sounddevice"),
            flush=True,
        )
        args.play = False

    text = load_text(args)
    if not text:
        sys.exit("Empty text.")

    language = None if args.language.lower() in ("auto", "none", "") else args.language
    voice    = args.voice if args.voice else None
    speed    = args.speed
    if speed is not None and not (0.25 <= speed <= 3.0):
        sys.exit(f"--speed must be between 0.25 and 3.0 (got {speed}).")

    print(_c(_BOLD, "\nOmniVoice Realtime WebSocket Player"))
    print(f"  url       : {args.url}")
    print(f"  language  : {language or 'auto (server default)'}")
    print(f"  voice     : {voice    or '(server default)'}")
    print(f"  speed     : {speed if speed is not None else '(server default)'}")
    print(f"  playback  : {args.play}")
    print(f"  save wavs : {args.save}")
    print(f"  text chars: {len(text)}\n")

    try:
        asyncio.run(
            stream_and_play(
                url=args.url,
                text=text,
                play=args.play,
                save=args.save,
                output_dir=args.output_dir,
                recv_timeout=args.recv_timeout,
                language=language,
                voice=voice,
                speed=speed,
            )
        )
    except KeyboardInterrupt:
        print(_c(_DIM, "\nInterrupted."))
    except Exception as exc:
        sys.exit(_c(_RED, f"ERROR: {exc}"))


if __name__ == "__main__":
    main()
