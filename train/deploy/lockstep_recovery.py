"""Fall recovery on Unitree's own Go2 model, through the robot program's own hand-over code (recovery.Recovery):
a fall -> the self-right brain (numpy, deploy/selfright.py) rolls the dog belly-down -> the motion brain's getup clip
stands it up -> back to trotting. Every brain is fed only what a real Go2 can measure (IMU, joints, the estimator's
kinematic height and leg-odometry velocity). No torch needed.

    UNITREE_MUJOCO=... python deploy/lockstep_recovery.py [--tries 10] [--damping 0.1] [--dt 0.005] [--shove 2.5]

Tries: dropped from 0.35 m on the side, on the back, half-tipped (random roll/pitch/yaw); and "shove": trotting for 2 s,
then pushed sideways (--shove m/s plus an 8 rad/s roll kick). Success: back to the trot clip by the hand-over, then
3 s of trot, upright (> 0.9) with the body above 0.22 m at the end, Recovery never giving up. --tries <= 2 prints
each try's events.
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
from recovery import Recovery  # noqa: E402


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
    ap.add_argument("--shove", type=float, default=2.5, help="sideways push (m/s) plus a roll kick, for the shove tries")
    a = ap.parse_args()
    ctl = Go2Controller(a.brain)
    m = mujoco.MjModel.from_xml_path(str(Path(os.environ["UNITREE_MUJOCO"]) / "unitree_robots" / "go2" / "scene.xml"))
    m.opt.timestep = a.dt
    flatten(m)
    m.dof_damping[6:] = a.damping
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    act = np.array([ctl.body.joints.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]))
                    for i in range(m.nu)])
    stance = ctl.stance
    n_sub = int(round(DT / m.opt.timestep))
    kp, kd = ctl.body.entry["kp"], ctl.body.entry["kv"]
    rng = np.random.default_rng(1)
    falls = {"side": lambda: (rng.choice([-1, 1]) * rng.uniform(1.2, 1.9), rng.uniform(-0.4, 0.4)),
             "back": lambda: (rng.choice([-1, 1]) * rng.uniform(2.4, np.pi), rng.uniform(-0.4, 0.4)),
             "tipped": lambda: (rng.choice([-1, 1]) * rng.uniform(0.5, 1.2), rng.uniform(-0.6, 0.6)),
             "shove": None}
    out, times = {}, []
    for kind, draw in falls.items():
        ok = 0
        for k in range(a.tries):
            d = mujoco.MjData(m)
            if kind == "shove":   # trotting, then pushed over sideways
                d.qpos[2], d.qpos[3:7], d.qpos[7:19] = 0.30, euler_quat(0, 0, rng.uniform(-np.pi, np.pi)), stance
                ctl.start_at("trot", 0.0)
            else:   # dropped from 0.35 m in a fallen pose
                roll, pitch = draw()
                d.qpos[2], d.qpos[3:7] = 0.35, euler_quat(roll, pitch, rng.uniform(-np.pi, np.pi))
                d.qpos[7:19] = stance + rng.normal(0, 0.1, 12)
                ctl.start_at("lie", 0.0)
            ctl.last_action[:] = 0
            mujoco.mj_forward(m, d)
            target = d.qpos[7:19].copy()
            rec, est = Recovery(ctl, a.selfright), LegEstimator()
            resume, pushed, t_fall, back_at = "trot", False, None, None
            for t in range(int(14.0 / DT)):
                if kind == "shove" and not pushed and t * DT >= 2.0:   # the push: sideways speed and a roll kick
                    side = rng.choice([-1, 1])
                    R = d.xmat[1].reshape(3, 3)
                    d.qvel[0:3] += R @ np.array([0, side * a.shove, 0])
                    d.qvel[3:6] += np.array([side * -8.0, 0, 0])
                    pushed = True
                quat, gyro, q, dq = d.qpos[3:7].copy(), d.qvel[3:6].copy(), d.qpos[7:19].copy(), d.qvel[6:18].copy()
                v, h = est.update(quat, q, dq, gyro)
                if kind != "shove" and t < int(0.6 / DT):
                    tgt = target   # landing
                else:
                    tgt = rec.step(quat, gyro, q, dq, v, h, resume)
                    if rec.failed:
                        break
                    if rec.mode == "selfright" and t_fall is None:
                        t_fall = t
                    if tgt is None:
                        if kind != "shove" and rec.recovered == 0:   # dropped tries: start in the self-right
                            rec.falls, rec.mode, rec.t, rec.still = 1, "selfright", 0, 0
                            rec.sr.last_action[:] = 0
                            t_fall = t
                            tgt = rec.sr.step(quat, gyro, q, dq, v, h)
                        else:
                            if rec.recovered and back_at is None:
                                back_at = t
                            tgt = ctl.step(quat, gyro, q, dq, v, h)
                target = np.clip(tgt, ctl.lo, ctl.hi)
                for _ in range(n_sub):
                    qq, vv = d.qpos[7:19], d.qvel[6:18]
                    d.ctrl[:] = np.clip(kp * (target[act] - qq[act]) - kd * vv[act], lo, hi)
                    mujoco.mj_step(m, d)
                if back_at is not None and t > back_at + int(3.0 / DT):   # 3 s of trot after getting back up
                    break
            if a.tries <= 2:
                print(kind, k, rec.failed, rec.events, rec.mode, t * DT, round(float(d.xmat[1][8]), 2), round(float(d.qpos[2]), 2), ctl.clip.name)
            fallen = kind == "shove" and t_fall is None
            good = (not rec.failed and back_at is not None and d.xmat[1].reshape(3, 3)[2, 2] > 0.9
                    and d.qpos[2] > 0.22 and ctl.clip.name == "trot")
            ok += good
            if good and t_fall is not None:
                times.append((back_at - t_fall) * DT)
            if fallen:
                print(f"  shove try {k}: the push did not tip it over (counted as a fail of the test, not the robot)")
        out[kind] = f"{ok}/{a.tries}"
    print(f"Unitree go2.xml, damping {a.damping}, dt {a.dt}, robot-only inputs, numpy self-right + Recovery hand-over: "
          f"fall -> roll -> stand -> trot again {out}; fall to trotting again median {np.median(times):.1f} s")


if __name__ == "__main__":
    main()
