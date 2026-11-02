"""Freeze a real training process and check that the watchdog recovers from its checkpoint.

SIGSTOP stands in for an MPS deadlock: the process stays alive and silent, which is the failure an
exit-code check would never notice. Only the trainer started here is signalled.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

RUN = Path("runs/fault")
STALL_SECONDS = 20
CHECKPOINT_SECONDS = 20
MINUTES = 2.0


def log_lines() -> list[dict[str, float | str]]:
    path = RUN / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def resumed_from() -> int | None:
    events = [line["step"] for line in log_lines() if line.get("event") == "resumed"]
    return int(events[0]) if events else None


def main() -> None:
    shutil.rmtree(RUN, ignore_errors=True)
    supervisor = subprocess.Popen(
        [
            sys.executable, "-m", "typeflow", "supervise",
            "--run", str(RUN),
            "--minutes", str(MINUTES),
            "--stall-seconds", str(STALL_SECONDS),
            "--checkpoint-seconds", str(CHECKPOINT_SECONDS),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )  # fmt: skip

    while not (RUN / "latest.pt").exists():
        time.sleep(1)
    time.sleep(CHECKPOINT_SECONDS / 2)

    children = subprocess.run(
        ["pgrep", "-P", str(supervisor.pid)], capture_output=True, text=True, check=True
    ).stdout.split()
    if len(children) != 1:
        supervisor.kill()
        raise RuntimeError(f"expected one trainer, found {children}")
    trainer = int(children[0])
    last_logged_step = int(log_lines()[-1]["step"])
    frozen_at = time.monotonic()
    os.kill(trainer, signal.SIGSTOP)

    while resumed_from() is None and supervisor.poll() is None:
        time.sleep(1)
    recovered_seconds = time.monotonic() - frozen_at

    output, _ = supervisor.communicate()
    report = {
        "last_logged_step_before_freeze": last_logged_step,
        "resumed_from_step": resumed_from(),
        "seconds_from_freeze_to_resume": round(recovered_seconds, 1),
        "stall_threshold_seconds": STALL_SECONDS,
        "checkpoint_interval_seconds": CHECKPOINT_SECONDS,
        "final_step": int(log_lines()[-1]["step"]),
        "supervisor": json.loads(output.strip().splitlines()[-1]),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/fault_injection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
