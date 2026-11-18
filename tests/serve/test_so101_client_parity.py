"""The so101 client on Positronic's stack against the client it replaced, on one so101 server.

The server runs the stand-in networks of ``test_serve_so101`` under uvicorn. A
robot replays recorded readings: the follower reaches the row target that it
was given, as in ``FakeRobot``. Both clients run on a clock that moves only
when they sleep, so they read the same readings at the same rows.

* Chunked: the previous client (the hand-written msgpack wire of
  ``tests/serve/data/so101_client_v1.py``) and the new client send the same frame
  stacks, and the server answers the same rows, bit for bit.
* ``periodic_chunked``: the new client gets the rows of the chunked call, bit
  for bit.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import sys
import threading
import time
import types
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("positronic.policy.remote")
pytest.importorskip("uvicorn")

import uvicorn

from praxis.serve.server import InferenceServer

ROOT = Path(__file__).resolve().parents[2]

import test_serve_so101 as serve  # noqa: E402

CHUNKS = 4
#: The rows of the replay: the first chunk of a context, then 24 rows per chunk.
ROWS = 21 + 24 * (CHUNKS - 1) + 4


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def new() -> types.ModuleType:
    return _load("so101_client", ROOT / "demo" / "so101" / "so101_client.py")


@pytest.fixture(scope="module")
def old() -> types.ModuleType:
    return _load(
        "so101_client_v1", Path(__file__).parent / "data" / "so101_client_v1.py"
    )


class _Clock:
    """A ``time`` whose clock moves only in ``sleep`` (and in the reads of a
    :class:`_SlowRobot`)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    perf_counter = monotonic  # the capture times of a camera


class _Replay:
    """The recorded frames, row after row, and a follower that is at the last row target.

    ``reads[k]`` is the time of read ``k``. The frame of a read is the frame of
    its row: ``row = round((t - t0) * 15)``.
    """

    def __init__(
        self, clock: _Clock, joints: tuple[str, ...], views: list[str]
    ) -> None:
        self.clock, self.joints, self.views = clock, joints, views
        self.t0 = clock.t
        self.q = serve._state(0.0).astype(np.float64)
        self.sent: list[np.ndarray] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def row(self) -> int:
        return round((self.clock.t - self.t0) * 15.0)

    def get_observation(self) -> dict:
        obs: dict = {
            f"{j}.pos": float(v) for j, v in zip(self.joints, self.q, strict=True)
        }
        for v in self.views:
            obs[v] = serve._frame(float(self.row()), 240, 320)
        return obs

    def send_action(self, action: dict) -> dict:
        self.q = np.array([action[f"{j}.pos"] for j in self.joints], np.float64)
        self.sent.append(self.q.copy())
        return action


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[str, list[dict]]]:
    """The so101 server and the stacks of its policy calls."""
    executor = ThreadPoolExecutor(max_workers=1)
    factory = serve._rtc_factory()
    seen: list[dict] = []
    make = factory.__call__

    def recording(overrides: dict) -> object:
        policy = make(overrides)
        step = policy.session.step

        def record(obs: dict, executed_rows: int | None = None) -> object:
            seen.append(
                {
                    k: np.array(v)
                    for k, v in obs.items()
                    if k in ("state", "image.exterior")
                }
            )
            return step(obs, executed_rows=executed_rows)

        policy.session.step = record  # ty: ignore[invalid-assignment]
        return policy

    app = InferenceServer(
        {"default": recording}, executor
    )  # ty: ignore[invalid-argument-type]
    port = serve._free_port()
    srv = uvicorn.Server(
        uvicorn.Config(app.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    yield f"127.0.0.1:{port}", seen
    srv.should_exit = True
    thread.join(timeout=10)
    executor.shutdown()


def _drive(
    client: types.ModuleType,
    addr: str,
    monkeypatch: pytest.MonkeyPatch,
    schedule: str,
    log: Path,
    robot: _Replay | None = None,
) -> tuple[_Replay, list[dict]]:
    """``client.main`` on a replay robot (``robot``, by default a new one) for
    ``CHUNKS`` chunks; return the robot and the log records."""
    if robot is None:
        robot = _Replay(_Clock(), client.JOINTS, ["exterior"])
    clock = robot.clock
    monkeypatch.setattr(client, "time", clock)
    monkeypatch.setattr(client, "make_robot", lambda a, cams: robot)
    if hasattr(client, "PolicyRun"):
        # The answer of a call is read in the step that made it, as the previous
        # client waited for it: the clock stands still while a call runs.
        monkeypatch.setattr(client, "_monotonic_ns", lambda: round(clock.t * 1e9))
        run = functools.partial(client.PolicyRun, virtual=True)
        monkeypatch.setattr(client, "PolicyRun", run)
    session = json.dumps({"schedule": schedule, "sample_seed": 0})
    client.main(
        ["--url", f"ws://{addr}/api/v1/session", "--prompt", "Pick up the cube.",
         "--no-rest-move", "--no-rest-check", "--max-chunks", str(CHUNKS),
         "--control-hz", "15", "--session", session, "--log-jsonl", str(log)]
    )  # fmt: skip
    return robot, [json.loads(line) for line in log.read_text().splitlines()]


def test_the_new_chunked_client_sends_the_stacks_of_the_old_one(
    old: types.ModuleType,
    new: types.ModuleType,
    server: tuple[str, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    addr, seen = server
    seen.clear()
    _, before = _drive(old, addr, monkeypatch, "chunked", tmp_path / "old.jsonl")
    stacks_old = list(seen)
    seen.clear()
    _, after = _drive(new, addr, monkeypatch, "chunked", tmp_path / "new.jsonl")
    stacks_new = list(seen)
    assert [len(s["state"]) for s in stacks_new] == [
        len(s["state"]) for s in stacks_old
    ]
    for a, b in zip(stacks_old, stacks_new, strict=True):
        np.testing.assert_array_equal(a["state"], b["state"])
        np.testing.assert_array_equal(a["image.exterior"], b["image.exterior"])
    assert len(before) == len(after) == CHUNKS
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(
            np.float32(a["targets"]), np.float32(b["targets"])
        )


def test_the_periodic_chunked_client_gets_the_rows_of_the_chunked_call(
    new: types.ModuleType,
    server: tuple[str, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    addr, seen = server
    seen.clear()
    _, chunked = _drive(new, addr, monkeypatch, "chunked", tmp_path / "c.jsonl")
    seen.clear()
    _, periodic = _drive(
        new, addr, monkeypatch, "periodic_chunked", tmp_path / "p.jsonl"
    )
    assert [len(r["targets"]) for r in periodic] == [len(r["targets"]) for r in chunked]
    for a, b in zip(chunked, periodic, strict=True):
        np.testing.assert_array_equal(
            np.float32(a["targets"]), np.float32(b["targets"])
        )


@pytest.mark.parametrize("schedule", ["chunked", "periodic_chunked", "rtc"])
def test_max_chunks_ends_the_session_in_every_schedule(
    new: types.ModuleType,
    server: tuple[str, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    schedule: str,
) -> None:
    """``--max-chunks`` ends the session after that many answers, also in a
    real-time session, which answers again every stride while a chunk plays:
    those answers must not restart the rows of the last counted one."""
    _, records = _drive(new, server[0], monkeypatch, schedule, tmp_path / "s.jsonl")
    assert len(records) == CHUNKS


@pytest.mark.parametrize("schedule", ["chunked", "periodic_chunked", "rtc"])
def test_only_an_so101_server_without_half_frame_offsets_is_refused(
    new: types.ModuleType, monkeypatch: pytest.MonkeyPatch, schedule: str
) -> None:
    """The handshake of the so101 server passes the offsets check in each
    schedule. With the offsets of the earlier server (on the row grid) the
    chunked schedule fails it; the other schedules declare no stack."""
    import praxis.serve.policy as policy

    meta = serve._rtc_factory()({"schedule": schedule}).meta
    assert new.on_grid_offsets(meta) == []
    monkeypatch.setitem(policy.RESTRICT_SQUARE, "so101", (True, False))
    meta = serve._rtc_factory()({"schedule": schedule}).meta
    assert len(new.on_grid_offsets(meta)) == (24 if schedule == "chunked" else 0)


class _Camera:
    """A LeRobot camera on the virtual clock: its reader thread has frame ``k``
    (the frame of its row) at ``t0 + k * period_s``. ``read_latest`` gives the
    newest frame at once; ``async_read`` waits (the clock moves) for a frame
    newer than the last one it gave, as LeRobot's OpenCV camera waits for
    ``new_frame_event`` and clears it."""

    def __init__(self, robot: _SlowRobot, period_s: float) -> None:
        self.robot, self.period_s = robot, period_s
        self.taken = -1  # the last frame async_read gave

    def _latest(self) -> int:
        return math.floor((self.robot.clock.t - self.robot.t0) / self.period_s + 1e-9)

    def _frame(self, k: int) -> np.ndarray:
        return serve._frame(float(round(k * self.period_s * 15.0)), 240, 320)

    @property
    def latest_timestamp(self) -> float:
        return self.robot.t0 + self._latest() * self.period_s

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        age_ms = (self.robot.clock.t - self.latest_timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"latest frame is too old: {age_ms:.1f} ms")
        return self._frame(self._latest())

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        k = max(self._latest(), self.taken + 1)
        self.robot.clock.sleep(self.robot.t0 + k * self.period_s - self.robot.clock.t)
        self.taken = k
        return self._frame(k)


class _SlowRobot(_Replay):
    """A replay robot whose reading takes ``read_s`` of the virtual clock (the
    motor bus and the frame together). With ``camera_s`` it has a LeRobot
    camera at that period and a motor bus, and ``get_observation`` is the one
    of LeRobot 0.4.3 and earlier: ``async_read``, which waits for a new frame."""

    def __init__(
        self,
        clock: _Clock,
        joints: tuple[str, ...],
        read_s: float = 0.0,
        camera_s: float = 0.0,
    ) -> None:
        super().__init__(clock, joints, ["exterior"])
        self.read_s = read_s
        if camera_s:
            self.bus = self
            self.cameras = {"exterior": _Camera(self, camera_s)}

    def sync_read(self, name: str, **_: object) -> dict[str, float]:
        assert name == "Present_Position"
        return dict(zip(self.joints, map(float, self.q), strict=True))

    def get_observation(self) -> dict:
        self.clock.sleep(self.read_s)
        if not hasattr(self, "cameras"):
            return super().get_observation()
        obs: dict = {
            f"{j}.pos": v for j, v in self.sync_read("Present_Position").items()
        }
        return {**obs, "exterior": self.cameras["exterior"].async_read()}


def _drive_slow(
    client: types.ModuleType,
    addr: str,
    monkeypatch: pytest.MonkeyPatch,
    log: Path,
    **slow: float,
) -> tuple[int, int]:
    """``client.main`` on a :class:`_SlowRobot` (``slow``), ``periodic_chunked``;
    return the rows played and the rows answered."""
    robot = _SlowRobot(_Clock(), client.JOINTS, **slow)
    _, records = _drive(client, addr, monkeypatch, "periodic_chunked", log, robot)
    # the last command is the hold of the reading after the session
    return len(robot.sent) - 1, sum(len(r["targets"]) for r in records)


@pytest.mark.parametrize("camera_s", [0.12, 1 / 7.5])
def test_a_slow_camera_does_not_skip_rows(
    new: types.ModuleType,
    server: tuple[str, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    camera_s: float,
) -> None:
    """A camera that gives a frame every 0.12 s (or at 7.5 Hz) on the 15 Hz
    rows: the entry takes the latest frame (repeated on the next row) and the
    joints at once, so the rows answered are played. ``async_read`` waited for
    each new frame: 63 rows of 93 went out (59 at 7.5 Hz)."""
    played, answered = _drive_slow(
        new, server[0], monkeypatch, tmp_path / "s.jsonl", camera_s=camera_s
    )
    assert played == answered, (played, answered)


def test_a_paced_read_keeps_the_row_grid(
    new: types.ModuleType,
    server: tuple[str, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A reading of 20 ms on each row stays inside the row period: the next
    deadline is on the grid, not 20 ms after it. A deadline that slipped by
    the reading made each row 87 ms, and the player skipped 15 rows of 93."""
    played, answered = _drive_slow(
        new, server[0], monkeypatch, tmp_path / "s.jsonl", read_s=0.02
    )
    assert played == answered, (played, answered)
