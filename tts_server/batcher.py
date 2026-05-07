"""DynamicBatcher — groups concurrent synthesis requests into GPU batch calls.

Batching logic
──────────────
1. Block until the first request arrives (no busy-waiting).
2. Wait up to ``timeout_ms`` for more requests (up to ``max_batch``).
3. Dispatch the collected batch to a ProcessPoolExecutor worker.
4. Resolve each request's Future with its corresponding WAV bytes.

``submit_immediate`` bypasses the collection window for latency-critical
first-chunk generation.

Each request can carry its own ISO language code AND voice profile name;
``worker_generate`` handles mixed-language / mixed-voice batches by passing
both arguments as lists.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .config import SORT_BATCH
from .worker import worker_generate

logger = logging.getLogger("omnivoice.batcher")


@dataclass
class _SynthReq:
    text:     str
    cfg:      dict
    language: Optional[str]
    voice:    Optional[str]
    speed:    Optional[float]
    future:   asyncio.Future
    t_submit: float = field(default_factory=time.perf_counter)


class DynamicBatcher:
    """Accumulates concurrent requests and dispatches as single GPU batches."""

    def __init__(
        self,
        executor:   ProcessPoolExecutor,
        max_batch:  int,
        timeout_ms: float,
    ) -> None:
        self._executor  = executor
        self._max_batch = max_batch
        self._timeout   = timeout_ms / 1000.0
        self._queue: asyncio.Queue[_SynthReq] = asyncio.Queue()
        self._task:  Optional[asyncio.Task]   = None

        # Public stats
        self.total_requests: int   = 0
        self.total_batches:  int   = 0
        self.total_gen_ms:   float = 0.0

    @property
    def avg_batch_size(self) -> float:
        return (self.total_requests / self.total_batches) if self.total_batches else 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="dyn-batcher")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Enqueue a request into the batching window.  Resolves when audio is ready."""
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        await self._queue.put(
            _SynthReq(
                text=text, cfg=cfg, language=language, voice=voice,
                speed=speed, future=fut,
            )
        )
        return await fut

    async def submit_immediate(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, float]:
        """Bypass the batching window — generate a single text ASAP.

        Used for latency-critical first-chunk generation where the batch
        collection window would add unwanted delay.

        Returns ``(wav_bytes, gen_ms)``.
        """
        loop = asyncio.get_running_loop()
        wav_list, gen_ms = await loop.run_in_executor(
            self._executor,
            worker_generate,
            [text], cfg, [language], [voice], [speed],
        )
        return wav_list[0], gen_ms

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            first = await self._queue.get()
            batch: list[_SynthReq] = [first]
            deadline = time.perf_counter() + self._timeout

            while len(batch) < self._max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=max(0.0, remaining)
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            asyncio.create_task(self._dispatch(batch))

    async def _dispatch(self, batch: list[_SynthReq]) -> None:
        if SORT_BATCH and len(batch) > 1:
            order = sorted(range(len(batch)), key=lambda i: -len(batch[i].text))
            ordered = [batch[i] for i in order]
        else:
            ordered = batch

        texts     = [r.text for r in ordered]
        languages = [r.language for r in ordered]
        voices    = [r.voice for r in ordered]
        speeds    = [r.speed for r in ordered]
        cfg       = ordered[0].cfg
        avg_q     = sum((time.perf_counter() - r.t_submit) * 1000 for r in ordered) / len(ordered)
        unique_langs = sorted(
            {l for l in languages if l is not None},
        ) or ["auto"]
        unique_voices = sorted(
            {v for v in voices if v is not None},
        ) or ["default"]
        unique_speeds = sorted({f"{s:.2f}" for s in speeds if s is not None}) or ["1.00"]

        logger.info(
            "Batch dispatch  size=%d  avg_queue=%.1fms  chars=%d..%d  "
            "langs=%s  voices=%s  speeds=%s",
            len(ordered), avg_q,
            min(len(t) for t in texts), max(len(t) for t in texts),
            ",".join(unique_langs),
            ",".join(unique_voices),
            ",".join(unique_speeds),
        )

        loop = asyncio.get_running_loop()
        try:
            wav_list, gen_ms = await loop.run_in_executor(
                self._executor, worker_generate,
                texts, cfg, languages, voices, speeds,
            )
            self.total_requests += len(ordered)
            self.total_batches  += 1
            self.total_gen_ms   += gen_ms

            logger.info(
                "Batch done  size=%d  gen=%.1fms  (%.1fms/text)",
                len(ordered), gen_ms, gen_ms / len(ordered),
            )
            for req, wav_bytes in zip(ordered, wav_list):
                if not req.future.done():
                    req.future.set_result(wav_bytes)
        except Exception as exc:
            logger.exception("Batch dispatch error: %s", exc)
            for req in ordered:
                if not req.future.done():
                    req.future.set_exception(exc)
