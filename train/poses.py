"""Find out which poses Bolt can actually hold, before asking a policy to learn them.

    python poses.py            # prints settled height, pitch, tilt and foot contacts per pose
    python poses.py --render poses.png

Each candidate pose is held by the motors for 3 seconds from slightly random
starting angles. A pose is usable as a training target only if it settles to
the same place every time, stays upright, and touches the ground the way the
trick needs (e.g. all four feet for "stand").
"""
import argparse

import mujoco
import numpy as np

from bolt_env import LIE_POSE, MODEL, SIT_POSE, STAND_POSE, pose


# Candidates (radians). The ones used for training live in bolt_env.py.
CANDIDATES = {
    "stand": STAND_POSE,
    "sit": SIT_POSE,
    "sit_upright": pose(0.35, 0.7, 1.9, 2.7),
    "lie": LIE_POSE,
    "lie_low": pose(1.5, 2.75, 1.5, 2.75),
}
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def settle(m, target, seed, seconds=3.0):
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    d.qpos[2] = 0.36
    d.qpos[7:] = STAND_POSE + rng.normal(0, 0.05, 12)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    steps = int(seconds / m.opt.timestep)
    for k in range(steps):
        blend = min(1.0, k / (steps * 0.4))  # move from stand to the pose over the first 40%
        d.ctrl[:] = np.clip(STAND_POSE + blend * (target - STAND_POSE), lo, hi)
        mujoco.mj_step(m, d)
    R = d.xmat[m.body("torso").id].reshape(3, 3)
    feet = [d.geom_xpos[m.geom(f).id][2] < 0.03 for f in FEET]
    touching = {m.geom(c.geom1).name for c in d.contact[:d.ncon]} | {m.geom(c.geom2).name for c in d.contact[:d.ncon]}
    body_down = bool({"torso", "head", "snout"} & touching)
    return dict(height=d.qpos[2], pitch=np.arcsin(np.clip(R[2, 0], -1, 1)), roll=np.arcsin(np.clip(R[2, 1], -1, 1)),
                up=R[2, 2], feet=feet, body_down=body_down, torque=np.abs(d.actuator_force).max(), data=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--render", help="save a picture of each pose")
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    shots = []
    for name, target in CANDIDATES.items():
        runs = [settle(m, target, s) for s in range(args.seeds)]
        h = np.array([r["height"] for r in runs]); p = np.array([r["pitch"] for r in runs])
        feet = np.mean([r["feet"] for r in runs], axis=0)
        print(f"{name:12} height {h.mean():.3f}±{h.std():.3f}  pitch {p.mean():+.2f}±{p.std():.2f}  "
              f"upright {min(r['up'] for r in runs):.2f}  feet FL/FR/RL/RR "
              + "/".join(f"{f:.0%}" for f in feet)
              + f"  body on floor {np.mean([r['body_down'] for r in runs]):.0%}  peak torque {max(r['torque'] for r in runs):.1f} N·m")
        if args.render:
            shots.append((name, runs[0]["data"]))
    if args.render:
        from PIL import Image, ImageDraw
        r = mujoco.Renderer(m, 300, 400)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        tiles = []
        for name, d in shots:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.15]; cam.distance, cam.elevation, cam.azimuth = 1.3, -12, 120
            r.update_scene(d, cam); img = Image.fromarray(r.render())
            ImageDraw.Draw(img).text((10, 8), name, fill=(20, 20, 20)); tiles.append(np.asarray(img))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(args.render)


if __name__ == "__main__":
    main()
