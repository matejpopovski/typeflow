"""A process pool that can kill a stuck task.

``multiprocessing.Pool`` and ``ProcessPoolExecutor`` offer a timeout on waiting for a result but no
way to reclaim the worker, so one font that wedges FreeType costs a worker for the rest of the run.
Here each worker runs one task at a time over its own pipe, and the parent replaces any worker
that overruns or dies.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any

TIMED_OUT = "timed out"
CRASHED = "worker crashed"


@dataclass
class Failure:
    reason: str


def _worker(fn: Callable[[Any], Any], conn: Connection) -> None:
    while True:
        task = conn.recv()
        if task is None:
            return
        try:
            conn.send(fn(task))
        except Exception as exc:
            conn.send(Failure(f"{type(exc).__name__}: {exc}"))


@dataclass
class _Slot:
    process: mp.Process
    conn: Connection
    index: int = -1
    started: float = 0.0


def _spawn(ctx: Any, fn: Callable[[Any], Any]) -> _Slot:
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_worker, args=(fn, child), daemon=True)
    process.start()
    child.close()
    return _Slot(process, parent)


def _discard(slot: _Slot) -> None:
    slot.process.kill()
    slot.process.join(timeout=5)
    slot.conn.close()


def map_with_timeout(
    fn: Callable[[Any], Any],
    tasks: Iterable[Any],
    workers: int = 4,
    timeout: float = 60.0,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(task_index, result)`` as tasks finish. A failed task yields a ``Failure``."""
    ctx = mp.get_context("spawn")
    queue = list(enumerate(tasks))[::-1]
    slots = [_spawn(ctx, fn) for _ in range(min(workers, max(len(queue), 1)))]
    try:
        while True:
            for slot in slots:
                if slot.index < 0 and queue:
                    slot.index, task = queue.pop()
                    slot.started = time.monotonic()
                    slot.conn.send(task)
            busy = [s for s in slots if s.index >= 0]
            if not busy:
                return
            for conn in wait([s.conn for s in busy], timeout=0.2):
                slot = next(s for s in busy if s.conn is conn)
                index, slot.index = slot.index, -1
                try:
                    yield index, slot.conn.recv()
                except (EOFError, OSError):
                    yield index, Failure(CRASHED)
                    slots[slots.index(slot)] = _replace(ctx, fn, slot)
            now = time.monotonic()
            for i, slot in enumerate(slots):
                if slot.index >= 0 and now - slot.started > timeout:
                    yield slot.index, Failure(TIMED_OUT)
                    slots[i] = _replace(ctx, fn, slot)
    finally:
        for slot in slots:
            _discard(slot)


def _replace(ctx: Any, fn: Callable[[Any], Any], slot: _Slot) -> _Slot:
    _discard(slot)
    return _spawn(ctx, fn)
