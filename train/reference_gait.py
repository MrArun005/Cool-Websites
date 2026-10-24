"""A reference trot for Bolt to imitate.

Learning to walk from nothing takes a huge amount of practice. Disney's BDX droid and Open Duck
Mini instead reward the robot for following a reference walk, so the brain only has to learn
balance and corrections. This file makes that reference: a simple trot where diagonal leg pairs
(front-left + rear-right, front-right + rear-left) swing together.

    python reference_gait.py      # plays the trot on the motors with no brain, reports what happens

For each leg, the hip swings back and forth around the standing angle; while the foot swings
forward, the knee bends extra to lift it. Increasing a hip angle moves that foot backward on
every leg, so a foot pushes the body forward while its hip angle rises (the stance half).
"""
import argparse

import mujoco
import numpy as np

from bolt_env import MODEL, STAND_POSE

FREQ = 2.0                      # steps per second per leg pair, same as the env's gait clock
PAIR = np.array([0, 1, 1, 0])   # FL, FR, RL, RR: which half of the cycle each leg is on
KNEE_SIGN = np.array([-1, -1, 1, 1])  # bending more means more negative on front knees


def trot(phase, hip_swing=0.25, knee_lift=0.45, base=None):
    """Joint targets and planned foot contacts at a gait phase (radians, 2π per cycle).

    base: the stance the legs swing around (default: the standing pose).
    """
    q = (STAND_POSE if base is None else base).copy()
    stance = np.zeros(4)
    for leg in range(4):
        p = phase + np.pi * PAIR[leg]
        q[3 * leg + 1] += hip_swing * np.sin(p)
        swing = max(0.0, -np.cos(p))               # 0 in stance, up to 1 mid-swing
        q[3 * leg + 2] += KNEE_SIGN[leg] * knee_lift * swing
        stance[leg] = float(np.cos(p) > 0)
    return q, stance


def play(seconds=6.0, **gait):
    """Run the trot open-loop (no feedback) and report forward travel and uprightness."""
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.33
    d.qpos[7:] = STAND_POSE
    mujoco.mj_forward(m, d)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    steps, x0, min_up = int(seconds / m.opt.timestep), d.qpos[0], 1.0
    for k in range(steps):
        t = k * m.opt.timestep
        ramp = min(1.0, t / 1.0)                    # ease in over the first second
        q, _ = trot(2 * np.pi * FREQ * t, **{key: v * ramp for key, v in gait.items()} if gait else {})
        if not gait:
            q_full, _ = trot(2 * np.pi * FREQ * t)
            q = STAND_POSE + ramp * (q_full - STAND_POSE)
        d.ctrl[:] = np.clip(q, lo, hi)
        mujoco.mj_step(m, d)
        min_up = min(min_up, d.xmat[1].reshape(3, 3)[2, 2])
    R = d.xmat[1].reshape(3, 3)
    return dict(forward=float(d.qpos[0] - x0), sideways=float(d.qpos[1]), speed=float((d.qpos[0] - x0) / seconds),
                height=float(d.qpos[2]), min_up=float(min_up), heading=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", type=float, default=0.25)
    ap.add_argument("--knee", type=float, default=0.45)
    args = ap.parse_args()
    r = play(hip_swing=args.hip, knee_lift=args.knee)
    print(f"open-loop trot (hip ±{args.hip}, knee lift {args.knee}): moved {r['forward']:+.2f} m forward "
          f"({r['speed']:+.2f} m/s), {r['sideways']:+.2f} m sideways, heading {r['heading']:+.0f}°, "
          f"height {r['height']:.2f} m, lowest uprightness {r['min_up']:.2f}")


if __name__ == "__main__":
    main()
