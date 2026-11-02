import sys
import textwrap
from pathlib import Path

import pytest
import torch

from typeflow.runtime import atomic_save, beat, supervise, to_cpu


def test_to_cpu_copies_nested_tensors() -> None:
    tensor = torch.ones(2)
    state = to_cpu({"a": [tensor], "b": (tensor,), "c": 1})
    assert torch.equal(state["a"][0], tensor) and state["c"] == 1
    tensor.zero_()
    assert state["a"][0].sum() == 2


def test_atomic_save_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "state.pt"
    atomic_save({"x": torch.tensor([1.0])}, path)
    assert torch.load(path, weights_only=True)["x"].item() == 1.0
    assert list(path.parent.iterdir()) == [path]


def test_failed_save_keeps_the_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.pt"
    atomic_save({"step": 1}, path)

    def partial_write(state: object, handle: object) -> None:
        handle.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", partial_write)
    with pytest.raises(OSError, match="disk full"):
        atomic_save({"step": 2}, path)
    monkeypatch.undo()
    assert torch.load(path, weights_only=True)["step"] == 1
    assert list(tmp_path.iterdir()) == [path]


def write_child(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script), str(tmp_path)]


@pytest.mark.slow
def test_supervisor_kills_a_silent_child_and_restarts_it(tmp_path: Path) -> None:
    command = write_child(
        tmp_path,
        """
        import sys, time
        from pathlib import Path
        marker = Path(sys.argv[1]) / "ran"
        if marker.exists():
            sys.exit(0)
        marker.touch()
        time.sleep(600)
        """,
    )
    result = supervise(command, tmp_path, stall_seconds=1.0, poll_seconds=0.2, grace_seconds=0.1)
    assert (result.code, result.stalls, result.restarts) == (0, 1, 1)


@pytest.mark.slow
def test_supervisor_leaves_a_child_with_a_fresh_heartbeat_alone(tmp_path: Path) -> None:
    command = write_child(
        tmp_path,
        """
        import sys, time
        from pathlib import Path
        for _ in range(12):
            (Path(sys.argv[1]) / "heartbeat").touch()
            time.sleep(0.2)
        """,
    )
    result = supervise(command, tmp_path, stall_seconds=1.0, poll_seconds=0.1)
    assert (result.code, result.stalls, result.restarts) == (0, 0, 0)


@pytest.mark.slow
def test_supervisor_restarts_after_a_crash_and_gives_up_eventually(tmp_path: Path) -> None:
    command = write_child(tmp_path, "import sys; sys.exit(3)")
    result = supervise(command, tmp_path, max_restarts=2, poll_seconds=0.05)
    assert (result.code, result.restarts) == (3, 2)


def test_beat_creates_the_heartbeat(tmp_path: Path) -> None:
    beat(tmp_path)
    assert (tmp_path / "heartbeat").exists()
