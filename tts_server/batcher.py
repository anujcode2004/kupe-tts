"""DynamicBatcher — unified priority scheduler for GPU synthesis requests.

Architecture
────────────
A single ``_scheduler`` loop drives ALL GPU submissions through an
``asyncio.PriorityQueue``:

  *First-chunk* requests (priority 0) always dequeue before *rest-chunk*
  requests (priority 1).  Each dispatch is **awaited** — the scheduler
  blocks until the GPU call returns, then immediately re-checks the queue.
  This guarantees that first-chunk items are dispatched as soon as the GPU
  becomes available, regardless of how many rest-chunk items are queued.

Maximum first-chunk latency
───────────────────────────
  ``remaining time of the currently-running GPU call + FC generation time``

To keep this low, rest-chunks are capped at ``MAX_REST_BATCH`` items per
dispatch (default 1) and use fewer diffusion steps (``REST_CHUNK_STEPS``,
default 8 vs 16), so each GPU call finishes quickly and the scheduler can
re-check for incoming FC items.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .config import FC_BATCH_TIMEOUT_MS, MAX_REST_BATCH, SORT_BATCH
from .worker import worker_generate

logger = logging.getLogger("omnivoice.batcher")

_PRIO_FC = 0
_PRIO_REST = 1


@dataclass
class _SynthReq:
    text:     str
    cfg:      dict
    language: Optional[str]
    voice:    Optional[str]
    speed:    Optional[float]
    future:   asyncio.Future
    is_fc:    bool  = False
    t_submit: float = field(default_factory=time.perf_counter)


@dataclass(order=True)
class _PrioItem:
    priority: int
    seq:      int
    req:      _SynthReq = field(compare=False)


class DynamicBatcher:
    """Priority-aware GPU scheduler for concurrent TTS streams."""

    def __init__(
        self,
        executor:   ProcessPoolExecutor,
        max_batch:  int,
        timeout_ms: float,
    ) -> None:
        self._executor       = executor
        self._max_batch      = max_batch
        self._timeout        = timeout_ms / 1000.0
        self._fc_timeout     = FC_BATCH_TIMEOUT_MS / 1000.0
        self._max_rest_batch = MAX_REST_BATCH

        self._pq: asyncio.PriorityQueue[_PrioItem] = asyncio.PriorityQueue()
        self._seq  = 0
        self._task: Optional[asyncio.Task] = None

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
        self._task = asyncio.create_task(self._scheduler(), name="gpu-scheduler")

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
        """Enqueue a rest-chunk request (normal priority)."""
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._seq += 1
        await self._pq.put(_PrioItem(
            _PRIO_REST, self._seq,
            _SynthReq(text=text, cfg=cfg, language=language, voice=voice,
                      speed=speed, future=fut, is_fc=False),
        ))
        return await fut

    async def submit_first_chunk(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, float]:
        """Submit a first-chunk request with highest priority.

        Concurrent FC requests are collected within a short window
        (``FC_BATCH_TIMEOUT_MS``, default 30 ms) and dispatched as one
        GPU batch.

        Returns ``(wav_bytes, gen_ms_per_item)``.
        """
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._seq += 1
        await self._pq.put(_PrioItem(
            _PRIO_FC, self._seq,
            _SynthReq(text=text, cfg=cfg, language=language, voice=voice,
                      speed=speed, future=fut, is_fc=True),
        ))
        return await fut

    async def submit_immediate(
        self,
        text: str,
        cfg: dict,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, float]:
        """Convenience wrapper — routes through the FC priority path."""
        return await self.submit_first_chunk(
            text, cfg, language=language, voice=voice, speed=speed,
        )

    # ------------------------------------------------------------------
    # Unified priority scheduler
    # ------------------------------------------------------------------

    async def _scheduler(self) -> None:
        """Single loop: dequeue by priority, dispatch, await, repeat.

        By awaiting each dispatch (never fire-and-forget), we guarantee
        that after every GPU call the queue is re-checked and FC items
        jump ahead of any queued REST items.
        """
        while True:
            first = await self._pq.get()

            if first.priority == _PRIO_FC:
                batch = await self._collect_fc(first)
                await self._dispatch_fc(batch)
            else:
                batch = await self._collect_rest(first)
                await self._dispatch_rest(batch)

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    async def _collect_fc(self, first: _PrioItem) -> list[_SynthReq]:
        """Collect FC items within a short window (default 30 ms)."""
        batch = [first.req]
        deadline = time.perf_counter() + self._fc_timeout

        while len(batch) < self._max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._pq.get(), timeout=max(0.001, remaining),
                )
                if item.priority == _PRIO_FC:
                    batch.append(item.req)
                else:
                    await self._pq.put(item)
                    break
            except asyncio.TimeoutError:
                break

        return batch

    async def _collect_rest(self, first: _PrioItem) -> list[_SynthReq]:
        """Collect REST items, stopping immediately if an FC item appears."""
        batch = [first.req]
        deadline = time.perf_counter() + self._timeout

        while len(batch) < self._max_rest_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    self._pq.get(), timeout=max(0.001, remaining),
                )
                if item.priority == _PRIO_FC:
                    await self._pq.put(item)
                    break
                batch.append(item.req)
            except asyncio.TimeoutError:
                break

        return batch

    # ------------------------------------------------------------------
    # GPU dispatch (no lock — single scheduler serialises all access)
    # ------------------------------------------------------------------

    async def _dispatch_fc(self, batch: list[_SynthReq]) -> None:
        texts     = [r.text for r in batch]
        languages = [r.language for r in batch]
        voices    = [r.voice for r in batch]
        speeds    = [r.speed for r in batch]
        cfg       = batch[0].cfg

        logger.info(
            "FC dispatch  size=%d  chars=%d..%d",
            len(batch),
            min(len(t) for t in texts), max(len(t) for t in texts),
        )

        loop = asyncio.get_running_loop()
        try:
            wav_list, gen_ms = await loop.run_in_executor(
                self._executor, worker_generate,
                texts, cfg, languages, voices, speeds,
            )
            self.total_requests += len(batch)
            self.total_batches  += 1
            self.total_gen_ms   += gen_ms

            per_gen = gen_ms / len(batch)
            logger.info(
                "FC done  size=%d  gen=%.1fms  (%.1fms/item)",
                len(batch), gen_ms, per_gen,
            )
            for req, wav_bytes in zip(batch, wav_list):
                if not req.future.done():
                    req.future.set_result((wav_bytes, per_gen))
        except Exception as exc:
            logger.exception("FC dispatch error: %s", exc)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    async def _dispatch_rest(self, batch: list[_SynthReq]) -> None:
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
        avg_q     = sum(
            (time.perf_counter() - r.t_submit) * 1000 for r in ordered
        ) / len(ordered)

        logger.info(
            "REST dispatch  size=%d  avg_queue=%.1fms  chars=%d..%d",
            len(ordered), avg_q,
            min(len(t) for t in texts), max(len(t) for t in texts),
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
                "REST done  size=%d  gen=%.1fms  (%.1fms/text)",
                len(ordered), gen_ms, gen_ms / len(ordered),
            )
            for req, wav_bytes in zip(ordered, wav_list):
                if not req.future.done():
                    req.future.set_result(wav_bytes)
        except Exception as exc:
            logger.exception("REST dispatch error: %s", exc)
            for req in ordered:
                if not req.future.done():
                    req.future.set_exception(exc)
