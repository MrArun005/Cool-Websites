"""Fall recovery on Unitree's own Go2 model (PLAN.md phase 2 for the recovery chain): from a fall, the self-right brain
rolls the dog belly-down, then the motion brain's getup clip stands it up -- each brain fed only what a real Go2 can
measure (IMU, joints, kinematic height; the self-right brain's velocity input from leg odometry, or 0 with --sr-vel 0).

    UNITREE_MUJOCO=... python deploy/lockstep_recovery.py [--tries 6] [--damping 0.1] [--dt 0.005] [--sr-vel odo|zero|true]

Needs SB3 + torch (the self-right brain is an SB3 policy). Falls: on the side, on the back, half-tipped; a random
roll/pitch/yaw, dropped from 0.35 m. Success: upright (> 0.9) with the body above 0.22 m at the end of the getup clip
+ 1 s, never tipped back over after standing.
"""
import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from controller import DT, Go2Controller  # noqa: E402
from estimator import LegEstimator  # noqa: E402
from lockstep_vendor import flatten  # noqa: E402


def euler_quat(roll, pitch, yaw):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfright", default=str(HERE.parent.parent / "models" / "bolt-go2-selfright-6.9M"))
    ap.add_argument("--brain", default=str(HERE.parent.parent / "models" / "bolt-go2-blind-592M" / "params.pkl"))
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--damping", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--sr-vel", choices=["odo", "zero", "true"], default="odo")
    a = ap.parse_args()
    from getup_env import GetUpPolicy
    sr = GetUpPolicy(a.selfright, body="go2")
    ctl = Go2Controller(a.brain)
    m = mujoco.MjModel.from_xml_path(str(Path(os.environ["UNITREE_MUJOCO"]) / "unitree_robots" / "go2" / "scene.xml"))
    m.opt.timestep = a.dt
    flatten(m)
    m.dof_damping[6:] = a.damping
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    act = np.array([ctl.body.joints.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]))
                    for i in range(m.nu)])
    jlo, jhi = ctl.lo, ctl.hi
    stance = ctl.stance
    sr_scale = np.maximum(jhi - stance, stance - jlo)   # getup_env: selfright acts over the full motor range
    n_sub = int(round(DT / m.opt.timestep))
    kp, kd = ctl.body.entry["kp"], ctl.body.entry["kv"]
    rng = np.random.default_rng(1)
    falls = {"side": lambda: (rng.choice([-1, 1]) * rng.uniform(1.2, 1.9), rng.uniform(-0.4, 0.4)),
             "back": lambda: (rng.choice([-1, 1]) * rng.uniform(2.4, np.pi), rng.uniform(-0.4, 0.4)),
             "tipped": lambda: (rng.choice([-1, 1]) * rng.uniform(0.5, 1.2), rng.uniform(-0.6, 0.6))}
    out = {}
    for kind, draw in falls.items():
        ok = 0
        for k in range(a.tries):
            d = mujoco.MjData(m)
            roll, pitch = draw()
            d.qpos[2], d.qpos[3:7] = 0.35, euler_quat(roll, pitch, rng.uniform(-np.pi, np.pi))
            d.qpos[7:19] = stance + rng.normal(0, 0.1, 12)
            target = d.qpos[7:19].copy()

            def run(n):
                for _ in range(n):
                    for _ in range(n_sub):
                        q, v = d.qpos[7:19], d.qvel[6:18]
                        d.ctrl[:] = np.clip(kp * (target[act] - q[act]) - kd * v[act], lo, hi)
                        mujoco.mj_step(m, d)

            run(int(0.6 / DT))   # let it land and settle limp-ish
            est, last_a = LegEstimator(), np.zeros(12)
            righted = None
            for t in range(int(4.0 / DT)):   # self-right, at most 4 s
                R = d.xmat[1].reshape(3, 3)
                v_odo, h = est.update(d.qpos[3:7].copy(), d.qpos[7:19].copy())
                v = {"odo": v_odo, "zero": np.zeros(3), "true": R.T @ d.qvel[0:3]}[a.sr_vel]
                obs = np.concatenate([R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, v, [h],
                                      d.qpos[7:19] - stance, d.qvel[6:18] * 0.1, last_a]).astype(np.float32)
                last_a = np.clip(np.asarray(sr.act(obs), float), -1, 1)
                target = np.clip(stance + sr_scale * last_a, jlo, jhi)
                run(1)
                if R[2, 2] > 0.9 and np.linalg.norm(d.qvel[0:3]) < 0.2:
                    righted = t if righted is None else righted
                    if t > righted + 15:
                        break
            if righted is None:
                continue
            ctl.last_action[:] = 0
            ctl.start_at("getup", 0.0)
            est, fell = LegEstimator(), False
            for _ in range(int((ctl.clips["getup"].length + 1.0) / DT)):
                R = d.xmat[1].reshape(3, 3)
                v, h = est.update(d.qpos[3:7].copy(), d.qpos[7:19].copy())
                target = ctl.step(d.qpos[3:7].copy(), d.qvel[3:6].copy(), d.qpos[7:19].copy(), d.qvel[6:18].copy(), v, h)
                run(1)
                if d.xmat[1].reshape(3, 3)[2, 2] < 0.5:
                    fell = True
                    break
            ok += (not fell) and d.xmat[1].reshape(3, 3)[2, 2] > 0.9 and d.qpos[2] > 0.22
        out[kind] = f"{ok}/{a.tries}"
    print(f"Unitree go2.xml, damping {a.damping}, dt {a.dt}, self-right velocity input {a.sr_vel}: fall -> roll -> stand "
          f"{out}")


if __name__ == "__main__":
    main()
