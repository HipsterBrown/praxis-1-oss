"""The end-effector frame that the server states (praxis.serve.ee_frames).

A server that reads poses or sends pose commands states the frame of each:
a ``change_ee_frame`` entry first in its client stack and ``ee_frames`` in
the handshake meta (Positronic-Robotics/positronic#1073). The rows and the
actions do not change: the entry only tells the client where the poses are.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_serve_rtc import (
    DELAYS,
    _current,
    _drive,
    _droid_factory,
    _droid_sample,
    _factory,
    _obs,
)
from test_serve_so101 import _rtc_factory as _so101_factory

from praxis.serve import ee_frames, params, rtc

SCHEDULES = ("chunked", "periodic_chunked", "rtc")
FACTORIES = {
    "single_arm": _droid_factory,
    "bimanual": lambda: _factory(),
}


@pytest.mark.parametrize("rig", ["single_arm", "bimanual"])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_pose_rig_states_its_frame_first(rig: str, schedule: str) -> None:
    """In every schedule the frame entry is the first entry of the client
    stack, ahead of stop_on_fault, and the meta repeats it."""
    policy = FACTORIES[rig]()({"schedule": schedule, "sample_seed": 0})
    seq = policy.local_stack["seq"]
    assert seq[0] == ee_frames.entry_for(rig)
    assert seq[1]["name"] == "stop_on_fault"
    assert [e["name"] for e in seq].count("change_ee_frame") == 1
    assert policy.meta["ee_frames"] == ee_frames.meta_frames(seq[0])


@pytest.mark.parametrize("rig", ["single_arm", "bimanual"])
def test_the_meta_frames_equal_positronic_reading_of_the_entry(rig: str) -> None:
    """A v3 client compares the meta's ee_frames with its stack's: they must be equal."""
    positronic_spec = pytest.importorskip("positronic.policy.spec")
    entry = ee_frames.entry_for(rig)
    codec = positronic_spec.from_spec(entry)
    assert codec.meta["ee_frames"] == ee_frames.meta_frames(entry)


def test_the_frames_of_each_rig() -> None:
    """The single-arm rig states the FR3 Franka Hand point on its pose and its
    command; the bimanual rig states the YAM grasp site on both arms."""
    single = ee_frames.entry_for("single_arm")
    assert single is not None
    assert single["args"]["site"] == "gripper"
    assert single["args"]["keys"] == [
        "robot_state.ee_pose",
        "robot_command.pose",
        "robot_command",
    ]
    assert single["args"]["transform"] == list(ee_frames.FRANKA_HAND_TRANSFORM)
    both = ee_frames.entry_for("bimanual")
    assert both is not None
    assert both["args"]["transform"] == list(ee_frames.YAM_GRASP_SITE_TRANSFORM)
    for side in ("left", "right"):
        for key in (
            f"robot_state.{side}.ee_pose",
            f"robot_command.{side}.pose",
            f"robot_command.{side}",
        ):
            assert key in both["args"]["keys"]
    assert ee_frames.entry_for("so101") is None


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_so101_rig_states_no_frame(schedule: str) -> None:
    """The so101 rig sends joint targets: no entry and no ee_frames."""
    policy = _so101_factory()({"schedule": schedule})
    names = [e["name"] for e in policy.local_stack["seq"]]
    assert "change_ee_frame" not in names and "ee_frames" not in policy.meta


@pytest.mark.parametrize("rig", ["single_arm", "bimanual"])
@pytest.mark.parametrize("off", ["off", False, 0])
def test_declare_ee_frame_off_states_none(rig: str, off: object) -> None:
    """A client that changes the frame itself (or is older than Positronic
    0.4.1) sets declare_ee_frame=off; the stack is then the stack without it."""
    with_frame = FACTORIES[rig]()({"schedule": "chunked"})
    without = FACTORIES[rig]()({"schedule": "chunked", "declare_ee_frame": off})
    assert "ee_frames" not in without.meta
    assert without.local_stack["seq"] == with_frame.local_stack["seq"][1:]
    assert "declare_ee_frame" in params.SESSION_PARAMS


def test_the_frame_does_not_change_the_rows_of_a_real_time_session() -> None:
    """The rows of a session that states the frame equal those of one that
    does not, bit for bit: the server reads and sends the same values."""
    runs = []
    for declare in ("on", "off"):
        policy = _factory()(
            {
                "schedule": "rtc",
                "sample_seed": 0,
                "precompute": "off",
                "declare_ee_frame": declare,
            }
        )
        answers, at, _ = _drive(policy, DELAYS[:5], 6.0)
        runs.append((answers, at))
    (a, at_a), (b, at_b) = runs
    assert at_a == at_b and len(a) == 5
    chans = rtc.channels(("left", "right"))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(
            rtc.rows_to_array(x, chans), rtc.rows_to_array(y, chans)
        )


def test_the_frame_does_not_change_the_rows_of_a_chunked_call() -> None:
    """The same for a single-arm chunked call."""
    rows = []
    for declare in ("on", "off"):
        policy = _droid_factory()(
            {"schedule": "chunked", "sample_seed": 0, "declare_ee_frame": declare}
        )
        rows.append(policy.select_action(_droid_sample(0.0)))
    assert rows[0] is not None and rows[1] is not None
    chans = rtc.channels(())
    np.testing.assert_array_equal(
        rtc.rows_to_array(rows[0], chans), rtc.rows_to_array(rows[1], chans)
    )


@pytest.mark.parametrize(
    ("rig", "model", "keys"),
    [
        ("single_arm", "bundled_franka_model", ("robot_state.ee_pose",)),
        (
            "bimanual",
            "bundled_yam_model",
            ("robot_state.left.ee_pose", "robot_state.right.ee_pose"),
        ),
    ],
)
def test_positronic_builds_the_entry_and_it_gives_the_legacy_point(
    rig: str, model: str, keys: tuple[str, ...]
) -> None:
    """Positronic builds the entry, and the poses it gives equal the poses of
    the legacy-point fallback that a 0.4.x client applies to a server that
    states no frame (the rig's ``gripper`` site to its ``end_effector``),
    to float tolerance (the URDF's link chain composes the same transform)."""
    positronic_spec = pytest.importorskip("positronic.policy.spec")
    from positronic.drivers.roboarm import models
    from positronic.drivers.roboarm.ik import frame_transform
    from positronic.drivers.roboarm.models import EEFrame
    from positronic.policy.codecs.geometry import ChangeEEFrame

    declared = positronic_spec.from_spec(ee_frames.entry_for(rig))
    assert isinstance(declared, ChangeEEFrame)
    urdf = getattr(models, model)()["urdf"]
    legacy = ChangeEEFrame(
        EEFrame(
            site="gripper", offset=frame_transform(urdf, "gripper", "end_effector")
        ),
        keys,
    )
    rng = np.random.default_rng(0)
    for _ in range(100):
        obs = {}
        for k in keys:
            q = rng.normal(size=4)
            obs[k] = np.r_[rng.uniform(-0.6, 0.6, 3), q / np.linalg.norm(q)].astype(
                np.float32
            )
        got, want = declared.encode(dict(obs)), legacy.encode(dict(obs))
        for k in keys:
            np.testing.assert_allclose(got[k], want[k], rtol=0, atol=1e-7)


def test_bimanual_obs_helper_is_the_current_frame() -> None:
    """The bimanual pose keys the entry names are the keys the server reads."""
    obs = _current(_obs([0.0]))
    entry = ee_frames.entry_for("bimanual")
    assert entry is not None
    for side in ("left", "right"):
        assert f"robot_state.{side}.ee_pose" in obs
        assert f"robot_state.{side}.ee_pose" in entry["args"]["keys"]


def test_the_robolab_client_builds_the_stack_of_a_server_without_a_frame() -> None:
    """With Positronic installed: the client stack that ``RemotePolicy`` builds
    from the RoboLab session's meta is the same with and without the server's
    frame (no ``change_ee_frame`` codec), so RoboLab's rows don't change."""
    pytest.importorskip("positronic.policy.remote")
    import importlib.util
    import sys
    from pathlib import Path

    from positronic.policy.remote import declared_stack
    from positronic_model_server.protocol import ProtocolVersion

    cfg_dir = Path(__file__).resolve().parents[2] / "eval/robolab/positronic_cfg"
    sys.path.insert(0, str(cfg_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "ee_frame_policy_cfg", cfg_dir / "ee_frame_policy_cfg.py"
        )
        assert spec is not None and spec.loader is not None
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
    finally:
        sys.path.remove(str(cfg_dir))
    for declare in ("on", "off"):
        policy = _droid_factory()({"schedule": "chunked", "declare_ee_frame": declare})
        session = cfg._CallLoggingSession(
            type("S", (), {"metadata": policy.meta, "protocol_version": 1})(), []
        )
        stack = declared_stack(session.metadata, ProtocolVersion.V1)
        names = [type(p).__name__ for p in getattr(stack, "_processors", [stack])]
        assert "ChangeEEFrame" not in names and "ee_frames" not in session.metadata
        if declare == "on":
            on_seq = session.metadata["local_stack"]["seq"]
        else:
            assert session.metadata["local_stack"]["seq"] == on_seq


@pytest.mark.parametrize(
    ("value", "declared"),
    [
        (None, True),
        ("on", True),
        (True, True),
        ("off", False),
        (False, False),
        (0, False),
    ],
)
def test_declare_stop_on_fault_reads_off(value: object, declared: bool) -> None:
    """``?declare_stop_on_fault=off`` arrives as the string ``"off"`` (not valid
    JSON); it turns stop_on_fault off, as ``false`` and ``0`` do."""
    query = {"schedule": "chunked"} | (
        {} if value is None else {"declare_stop_on_fault": value}
    )
    policy = _droid_factory()(query)
    names = [e["name"] for e in policy.local_stack["seq"]]
    assert ("stop_on_fault" in names) is declared
