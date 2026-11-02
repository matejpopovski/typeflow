"""What keeps a long unattended run alive: atomic checkpoints and an external watchdog.

Saving on MPS has been seen to deadlock, and a stuck native call can hold the GIL, in which case a
watchdog thread inside the process would never get to run. So the watchdog is a separate parent
process that judges the trainer by a heartbeat file and restarts it from the last checkpoint. The
trainer only has to keep that file fresh and to make a checkpoint that is either whole or absent.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HEARTBEAT = "heartbeat"
EXIT_OK = 0


def to_cpu(value: Any) -> Any:
    """Copy tensors out of device memory before anything is serialised."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def atomic_save(state: dict[str, Any], path: Path) -> None:
    """Write through a temporary file, so a crash at any point leaves the previous file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def beat(run_dir: Path) -> None:
    (run_dir / HEARTBEAT).touch()


@dataclass
class SuperviseResult:
    code: int
    restarts: int
    stalls: int


def supervise(
    command: Sequence[str],
    run_dir: Path,
    stall_seconds: float = 180.0,
    max_restarts: int = 20,
    poll_seconds: float = 2.0,
    grace_seconds: float = 3.0,
    log: Path | None = None,
) -> SuperviseResult:
    """Run ``command`` until it exits cleanly, killing and restarting it whenever it goes quiet."""
    run_dir.mkdir(parents=True, exist_ok=True)
    restarts = stalls = 0
    while True:
        beat(run_dir)
        sink = log.open("ab") if log else None
        child = subprocess.Popen(command, stdout=sink, stderr=sink)
        code = None
        while code is None:
            time.sleep(poll_seconds)
            code = child.poll()
            if code is not None:
                break
            quiet = time.time() - (run_dir / HEARTBEAT).stat().st_mtime
            if quiet > stall_seconds:
                stalls += 1
                print(f"stall: no heartbeat for {quiet:.0f}s, killing {child.pid}", flush=True)
                # Registered with faulthandler in the trainer, so this works with the GIL held.
                child.send_signal(signal.SIGUSR1)
                time.sleep(grace_seconds)
                child.kill()
                child.wait()
                code = -signal.SIGKILL
        if sink:
            sink.close()
        if code == EXIT_OK:
            return SuperviseResult(code, restarts, stalls)
        if restarts >= max_restarts:
            return SuperviseResult(code, restarts, stalls)
        restarts += 1
        print(f"child exited with {code}, restart {restarts}/{max_restarts}", flush=True)


def trainer_command(arguments: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "typeflow", *arguments]
