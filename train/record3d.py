"""Record a brain driving Bolt as 3D data for the browser viewer (no video).

    python record3d.py ../models/bolt-dog-walk-500k --command walk --gait-generator -o walk.json

For every control step (50 per second) it saves where each body part is and which parts
touch the floor or each other, so a web page can redraw the dog from any angle and mark
the moments a knee, shin or thigh scrapes the ground.
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


def describe(m):
    """Every visible dog geom: shape, size, and what part of which leg it is."""
    geoms = []
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name == "floor":
            continue
        kind = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}[int(m.geom_type[g])]
        leg = body[:2] if body[:2] in LEGS else ""
        if name.endswith("_foot"):
            part = "foot"
        elif kind == "capsule":
            part = "thigh" if body.endswith("thigh") else "shin"
        elif kind == "sphere":
            part = "hip joint" if body.endswith("hip") else "knee joint"
        else:
            part = name  # torso, head, snout, eyes
        geoms.append(dict(id=g, kind=kind, size=[round(float(v), 4) for v in m.geom_size[g]],
                          rgba=[round(float(v), 3) for v in m.geom_rgba[g]], leg=leg, part=part))
    return geoms


def record(policy, command, seconds, seed, gait_generator):
    env = BoltEnv(randomize=False, max_steps=100_000, gait_generator=gait_generator)
    env.forced_command = command
    obs, _ = env.reset(seed=seed)
    m, d = env.model, env.data
    floor = m.geom("floor").id
    geoms = describe(m)
    index = {g["id"]: i for i, g in enumerate(geoms)}
    frames = []
    quat = np.zeros(4)
    for _ in range(int(seconds * 50)):
        obs, _, fell, _, _ = env.step(policy.act(obs))
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
        frames.append(dict(p=pose, floor=sorted(on_floor), scrape=sorted(scrape), rub=[list(r) for r in sorted(rubbing)],
                           vx=round(float(d.qvel[:3] @ R[:, 0]), 3), h=round(float(d.qpos[2]), 3),
                           pitch=round(float(np.degrees(np.arcsin(-R[2, 0]))), 1),
                           yaw=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 1), fell=bool(fell)))
        if fell:
            break
    env.close()
    return geoms, frames


def summary(geoms, frames, skip=50):
    """Share of the time each leg part touches the floor (after the first second).

    "scrape" counts only contact away from the paw: a knee, shin or thigh really on the ground.
    """
    rows = {}
    window = frames[skip:] or frames
    for leg in LEGS:
        for part in PARTS:
            ids = {i for i, g in enumerate(geoms) if g["leg"] == leg and g["part"] == part}
            rows[f"{leg} {part}"] = round(np.mean([bool(ids & set(f["floor"])) for f in window]), 3)
        ids = {i for i, g in enumerate(geoms) if g["leg"] == leg}
        rows[f"{leg} scrape"] = round(np.mean([bool(ids & set(f["scrape"])) for f in window]), 3)
    body = {i for i, g in enumerate(geoms) if not g["leg"]}
    rows["body (torso/head)"] = round(np.mean([bool(body & set(f["floor"])) for f in window]), 3)
    rows["speed m/s"] = round(float(np.mean([f["vx"] for f in window])), 3)
    rows["height m"] = round(float(np.mean([f["h"] for f in window])), 3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="packaged model folder or run folder with policy.zip + vecnormalize.pkl")
    ap.add_argument("--command", default="walk")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--gait-generator", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    geoms, frames = record(Policy(args.model), args.command, args.seconds, args.seed, args.gait_generator)
    stats = summary(geoms, frames)
    Path(args.out).write_text(json.dumps(dict(label=args.label or Path(args.model).name, command=args.command,
                                              fps=50, geoms=geoms, frames=frames, summary=stats),
                                         separators=(",", ":")))
    for k, v in stats.items():
        print(f"{k:>18}: {v}")


if __name__ == "__main__":
    main()
