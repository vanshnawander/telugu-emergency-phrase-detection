"""
Batch scheduler for ASR inference — inspired by mini-sglang's overlap scheduler.

Collects incoming audio requests into an async queue, groups them into micro-batches,
and dispatches them to the GPU for inference. This avoids the serial one-request-at-a-time
bottleneck of the original pipeline.

Key ideas borrowed from mini-sglang:
  - Async request queue with uid tracking
  - Batch formation with configurable max size + timeout
  - Background loop that continuously drains the queue
  - Event-based result delivery back to the API layer
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import numpy as np

logger = logging.getLogger("asr.scheduler")


@dataclass
class ASRRequest:
    uid: int
    audio: torch.Tensor          # (1, num_samples) float32, 16 kHz
    lang: str                    # "en" | "hi" | "te"
    stream: bool = False         # whether this is a streaming chunk
    timestamp: float = field(default_factory=time.time)


@dataclass
class ASRResult:
    uid: int
    text: str
    latency_ms: float
    finished: bool = True


class BatchScheduler:
    """
    Async batch scheduler that collects ASR requests and runs them through
    the models in micro-batches for maximum GPU utilization.
    """

    def __init__(self, engine: "ASREngine", max_batch_size: int = 8, batch_timeout_ms: float = 50.0):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_ms / 1000.0

        # Incoming request queue
        self._queue: asyncio.Queue[ASRRequest] = asyncio.Queue()

        # Result delivery: uid -> (result, event)
        self._results: Dict[int, ASRResult] = {}
        self._events: Dict[int, asyncio.Event] = {}
        self._uid_counter: int = 0

        self._running = False
        self._task: Optional[asyncio.Task] = None

    def new_uid(self) -> int:
        uid = self._uid_counter
        self._uid_counter += 1
        self._events[uid] = asyncio.Event()
        return uid

    async def submit(self, request: ASRRequest) -> ASRResult:
        """Submit a request and wait for the result."""
        self._events[request.uid] = asyncio.Event()
        await self._queue.put(request)
        await self._events[request.uid].wait()
        result = self._results.pop(request.uid)
        del self._events[request.uid]
        return result

    def submit_nowait(self, request: ASRRequest) -> None:
        """Submit without waiting — for streaming use cases."""
        if request.uid not in self._events:
            self._events[request.uid] = asyncio.Event()
        self._queue.put_nowait(request)

    async def wait_result(self, uid: int) -> ASRResult:
        """Wait for a specific uid's result."""
        event = self._events.get(uid)
        if event is None:
            raise ValueError(f"Unknown uid: {uid}")
        await event.wait()
        result = self._results.pop(uid, None)
        self._events.pop(uid, None)
        if result is None:
            raise RuntimeError(f"No result for uid {uid}")
        return result

    def _deliver(self, result: ASRResult) -> None:
        """Deliver a result and wake up the waiting coroutine."""
        self._results[result.uid] = result
        event = self._events.get(result.uid)
        if event:
            event.set()

    async def _collect_batch(self) -> list[ASRRequest]:
        """
        Collect up to max_batch_size requests, waiting at most batch_timeout_s.
        Always waits for at least one request (blocking).
        """
        batch: list[ASRRequest] = []

        # Block until at least one request arrives
        first = await self._queue.get()
        batch.append(first)

        # Then greedily collect more within the timeout
        deadline = time.monotonic() + self.batch_timeout_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch

    async def _run_loop(self) -> None:
        """Main scheduler loop — runs forever, processing batches."""
        logger.info("Batch scheduler started (max_batch=%d, timeout=%.1fms)",
                     self.max_batch_size, self.batch_timeout_s * 1000)
        while self._running:
            batch: list[ASRRequest] = []
            try:
                batch = await self._collect_batch()
                if not batch:
                    continue

                # Group by language for efficient batched inference
                lang_groups: Dict[str, list[ASRRequest]] = {}
                for req in batch:
                    lang_groups.setdefault(req.lang, []).append(req)

                # Process each language group
                loop = asyncio.get_running_loop()
                for lang, requests in lang_groups.items():
                    results = await loop.run_in_executor(
                        None, self.engine.process_batch, requests, lang
                    )
                    for result in results:
                        self._deliver(result)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Scheduler loop error: %s", e, exc_info=True)
                # Deliver error results for any pending requests in this batch
                for req in batch:
                    self._deliver(ASRResult(
                        uid=req.uid, text="", latency_ms=0.0, finished=True
                    ))

    def start(self) -> None:
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run_loop())
        logger.info("Scheduler task created")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
