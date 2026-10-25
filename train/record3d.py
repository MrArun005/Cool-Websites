"""Record brains driving Bolt as 3D data for the browser viewer (the Gait Lab), no video.

    python record3d.py --brain posture=../models/bolt-dog-run2-1M \
                       --brain walk=../models/bolt-dog-walk-500k --gait-generator \
                       --script stand:2:posture sit:4:posture stand:2:posture walk:8:walk -o routine.json

Each script step is command:seconds:brain[:turn], turn in rad/s (+ left) for brains that can turn.
The brains share one input layout (the turn command is an extra last input), so one can
hand over to another mid-clip, the way the robot will swap skills. For every control step
(50 per second) it saves where each body part is and which parts touch the floor or each
other, so a web page can redraw the dog from any angle and mark the moments a knee, shin
or thigh is on the ground.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bolt_env import BoltEnv
from evaluate import Policy

LEGS = ["FL", "FR", "RL", "RR"]
PARTS = ["foot", "shin", "thigh"]


class NoBrain:
    """--brain name=none: zero output, so only the built-in trot moves the legs."""

    def act(self, obs, deterministic=True):
        return np.zeros(12)


def describe(m):
    """Every visible dog geom: shape, size, and what part of which leg it is."""
    geoms = []
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name in ("floor", "hurdle", "ice"):  # the world, not the dog
            continue
        kind = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}[int(m.geom_type[g])]
        leg = body[:2] if body[:2] in LEGS else ""
        if name.endswith("_foot"):
            part = "foot"
        elif kind == "capsule":
            part = "thigh" if body.endswith("thigh") else "shin"  # local +z points up the leg
        elif kind == "sphere":
            part = "hip joint" if body.endswith("hip") else "knee joint"
        else:
            part = name  # torso, head, snout, eyes
        geoms.append(dict(id=g, kind=kind, size=[round(float(v), 4) for v in m.geom_size[g]],
                          rgba=[round(float(v), 3) for v in m.geom_rgba[g]], leg=leg, part=part))
    return geoms


def capture(env, geoms, index, **extra):
    """One frame: where every part is, what touches the floor, and the body's motion."""
    m, d = env.model, env.data
    floor = m.geom("floor").id
    quat = np.zeros(4)
    pose = []
    for g in geoms:
        mujoco.mju_mat2Quat(quat, d.geom_xmat[g["id"]])
        pose += [round(float(v), 4) for v in d.geom_xpos[g["id"]]]
        pose += [round(float(v), 4) for v in quat]
    on_floor, scrape, rubbing = set(), set(), set()
    for c in d.contact[:d.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            on_floor.add(index[other])
            g = geoms[index[other]]
            # The paw sinks a few mm into the soft floor, so the shin's bottom tip (just above
            # the paw) touches too. Only contact more than 3 cm from the paw is a real scrape.
            if g["leg"] and g["part"] != "foot":
                paw = d.geom_xpos[m.geom(f"{g['leg']}_foot").id]
                if np.linalg.norm(c.pos - paw) > 0.03:
                    scrape.add(index[other])
        elif c.geom1 in index and c.geom2 in index:
            rubbing.add(tuple(sorted((index[c.geom1], index[c.geom2]))))
    R = d.xmat[1].reshape(3, 3)
    return dict(extra, yawr=round(float(d.qvel[5]), 3), p=pose, floor=sorted(on_floor), scrape=sorted(scrape),
                rub=[list(r) for r in sorted(rubbing)],
                vx=round(float(d.qvel[:3] @ R[:, 0]), 3), h=round(float(d.qpos[2]), 3),
                pitch=round(float(np.degrees(np.arcsin(R[2, 0]))), 1),  # nose up is positive
                yaw=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 1))


def record(brains, script, seed, gait_generator):
    turning = any(getattr(b, "turning", False) for b in brains.values())
    speed = any(getattr(b, "speed_control", False) for b in brains.values())
    sharp = any(getattr(b, "sharp_turns", False) for b in brains.values())
    env = BoltEnv(randomize=False, max_steps=100_000, gait_generator=gait_generator, turning=turning, speed_control=speed,
                  sharp_turns=sharp)
    env.forced_command = script[0][0]
    obs, _ = env.reset(seed=seed)
    geoms = describe(env.model)
    index = {g["id"]: i for i, g in enumerate(geoms)}
    frames = []
    for command, seconds, brain, turn, v in script:
        env.forced_command = command
        env.set_command(command)
        if turning:
            env.set_turn(turn)
        if speed:
            env.set_speed(v)
        for _ in range(int(seconds * 50)):
            obs, _, fell, _, _ = env.step(brains[brain].act(obs))
            frames.append(capture(env, geoms, index, cmd=command, brain=brain, turn=turn, speed=v, fell=bool(fell)))
            if fell:
                env.close()
                return geoms, frames
    env.close()
    return geoms, frames


def summary(geoms, frames, settle=50):
    """Per command: average height, pitch, speed and paws down, skipping the first second of
    each step (the change-over), plus how often a knee, shin or thigh is on the ground
    (contact away from the paw)."""
    rows, start = [], 0
    for k in range(1, len(frames) + 1):
        if k < len(frames) and frames[k]["cmd"] == frames[start]["cmd"]:
            continue
        window = frames[start + min(settle, (k - start) // 2):k]
        paws = {i for i, g in enumerate(geoms) if g["part"] == "foot"}
        rows.append(dict(command=frames[start]["cmd"], start=start, end=k,
                         height=round(float(np.mean([f["h"] for f in window])), 3),
                         pitch=round(float(np.mean([f["pitch"] for f in window])), 1),
                         speed=round(float(np.mean([f["vx"] for f in window])), 3),
                         paws=round(float(np.mean([len(paws & set(f["floor"])) for f in window])), 2),
                         scrape=round(float(np.mean([bool(f["scrape"]) for f in window])), 3)))
        start = k
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", action="append", required=True,
                    help="name=folder with policy.zip + vecnormalize.pkl (packaged model or run folder), or name=none")
    ap.add_argument("--script", nargs="+", required=True, help="command:seconds:brain steps, played in order")
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--gait-generator", nargs="?", const="crouch", default=None, choices=["crouch", "tall", "dog"], help="the walk runs on the built-in trot, around this stance (no value = crouch, as in runs 5-6)")
    ap.add_argument("--label", default="")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    brains = {name: (NoBrain() if path == "none" else Policy(path)) for name, path in (b.split("=", 1) for b in args.brain)}
    script = [(f[0], float(f[1]), f[2], float(f[3]) if len(f) > 3 else 0.0, float(f[4]) if len(f) > 4 else 0.4)
              for f in (step.split(":") for step in args.script)]
    geoms, frames = record(brains, script, args.seed, args.gait_generator)
    rows = summary(geoms, frames)
    stand_env = BoltEnv(gait_generator=args.gait_generator)  # the dog stance stands taller than the default
    targets = {c: {k: v for k, v in t.items() if k != "pose"} for c, t in stand_env.targets.items()}
    targets["walk"]["height"] = stand_env.walk_height
    Path(args.out).write_text(json.dumps(dict(label=args.label, fps=50, brains=dict(b.split("=", 1) for b in args.brain),
                                              targets=targets, geoms=geoms, frames=frames, summary=rows),
                                         separators=(",", ":")))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
