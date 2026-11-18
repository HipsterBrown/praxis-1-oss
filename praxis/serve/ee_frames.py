"""The end-effector frame that the server states to a Positronic client.

A Positronic client reports each pose at the ``gripper`` site of its arm.
Positronic 0.4.x gives a server that states no frame each pose at the legacy
point of the rig instead (``end_effector`` of the robot model), and a newer
Positronic drops that fallback: a server that reads poses or sends pose
commands then has to state the frame of each of them
([Positronic-Robotics/positronic#1073](https://github.com/Positronic-Robotics/positronic/pull/1073)).

The server states it with a ``change_ee_frame`` v2 entry, outermost in its
client stack, and repeats it as ``ee_frames`` in the handshake meta: a
protocol v1 or v2 client reads the frames from the meta only. Each entry is
Positronic's ``ChangeEEFrame(frame).to_spec()`` for the frame of a config of
``positronic.cfg.codecs`` (0.4.4), with the exact quaternion of its turn.
That frame is the legacy point of the rig, so a 0.4.x client gets the poses it
gets today, with the fallback or with the entry.

The single-arm rig states ``franka_hand_frame``, the FR3 Franka Hand point;
the bimanual rig states ``yam_grasp_site_frame`` on the keys of both arms; the
so101 rig sends joint targets and states none.
"""

from __future__ import annotations

import copy

# Positronic's ``franka_hand_frame``: 51.6 mm back toward the flange and -45 deg
# about z from the FR3 ``gripper`` site, the FR3 Franka Hand point.
FRANKA_HAND_TRANSFORM: tuple[float, ...] = (
    0.0,
    0.0,
    -0.0516,
    0.9238795325112867,
    0.0,
    0.0,
    -0.3826834323650898,
)
# Positronic's ``yam_grasp_site_frame``: 40.8 mm along y, 3.3 mm back along the
# wrist axis and -90 deg about it, the i2rt grasp site of the YAM.
YAM_GRASP_SITE_TRANSFORM: tuple[float, ...] = (
    0.0,
    0.0408,
    -0.0033,
    0.7071067811865476,
    0.0,
    0.0,
    -0.7071067811865476,
)


def _pose_keys(side: str | None) -> list[str]:
    """The observed pose, the commanded pose and the command channel of one arm."""
    if side is None:
        return ["robot_state.ee_pose", "robot_command.pose", "robot_command"]
    return [
        f"robot_state.{side}.ee_pose",
        f"robot_command.{side}.pose",
        f"robot_command.{side}",
    ]


def _entry(transform: tuple[float, ...], keys: list[str]) -> dict:
    return {
        "name": "change_ee_frame",
        "version": 2,
        "args": {"site": "gripper", "transform": list(transform), "keys": keys},
    }


#: The ``change_ee_frame`` entry of each rig that reads or commands poses.
RIG_ENTRIES: dict[str, dict] = {
    "single_arm": _entry(FRANKA_HAND_TRANSFORM, _pose_keys(None)),
    "bimanual": _entry(
        YAM_GRASP_SITE_TRANSFORM, _pose_keys("left") + _pose_keys("right")
    ),
}


def entry_for(rig: str) -> dict | None:
    """Return a copy of the ``change_ee_frame`` entry of ``rig``, None for a
    rig that states no frame (the so101 rig speaks joints)."""
    entry = RIG_ENTRIES.get(rig)
    return None if entry is None else copy.deepcopy(entry)


def meta_frames(entry: dict | None) -> dict[str, dict[str, object]] | None:
    """Return the handshake meta's ``ee_frames`` for ``entry``, None for none:
    Positronic's ``ee_frames_entry`` of the frame, keyed by pose signal."""
    if entry is None:
        return None
    args = entry["args"]
    return {
        key: {"site": args["site"], "offset": list(args["transform"])}
        for key in args["keys"]
    }


def prepend(stack: dict, entry: dict | None) -> dict:
    """Return ``stack`` with ``entry`` first in its ``seq``, unchanged for None.

    The entry must come first: inside ``temporal_stack`` it would get the
    stacked ``(T, 7)`` poses."""
    if entry is None:
        return stack
    return {**stack, "seq": [copy.deepcopy(entry), *stack["seq"]]}
