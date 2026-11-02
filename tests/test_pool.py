import os
import time

import pytest

from typeflow.pool import CRASHED, TIMED_OUT, Failure, map_with_timeout


def job(task: tuple[str, int]) -> int:
    kind, value = task
    if kind == "hang":
        time.sleep(600)
    if kind == "crash":
        os._exit(9)
    if kind == "raise":
        raise ValueError("bad font")
    return value * 2


pytestmark = pytest.mark.slow


def test_all_results_arrive_with_their_indices() -> None:
    results = dict(map_with_timeout(job, [("ok", i) for i in range(6)], workers=2, timeout=30))
    assert results == {i: i * 2 for i in range(6)}


def test_exception_becomes_a_failure_and_the_worker_survives() -> None:
    results = dict(map_with_timeout(job, [("raise", 0), ("ok", 4)], workers=1, timeout=30))
    assert isinstance(results[0], Failure) and "bad font" in results[0].reason
    assert results[1] == 8


def test_hung_task_is_killed_and_the_rest_still_finish() -> None:
    tasks = [("hang", 0), ("ok", 1), ("ok", 2), ("ok", 3)]
    started = time.monotonic()
    # Long enough for a worker to start on a slow runner, since the clock starts at dispatch.
    results = dict(map_with_timeout(job, tasks, workers=2, timeout=8))
    assert time.monotonic() - started < 60
    assert results[0] == Failure(TIMED_OUT)
    assert [results[i] for i in (1, 2, 3)] == [2, 4, 6]


def test_crashed_worker_is_replaced() -> None:
    results = dict(map_with_timeout(job, [("crash", 0), ("ok", 5)], workers=1, timeout=30))
    assert results[0] == Failure(CRASHED)
    assert results[1] == 10
