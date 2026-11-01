"""Replay a run_go2.py log in Unitree's Go2 model and measure where the simulator and the run disagree (PLAN.md
phase 5: the real robot's logs are what tell us how to fix the simulator).

    UNITREE_MUJOCO=... python deploy/replay_log.py run.npz [--window 0.5] [--damping 0.1]

The log holds, every 20 ms: time, IMU quaternion, gyro, joint angles and speeds (our order), the joint targets sent,
the velocity/height used, the clip. Replay: the simulator starts every --window seconds from the logged state (joint
angles and speeds, body tilt; body position at the logged height) and is driven by the same targets for one window,
with the robot's joint PD (kp 40, kd 1). Reported per joint type (hip, thigh, calf): how far the simulator's joints end
up from the logged ones after one window (rad), and the tilt difference. On a log from Unitree's simulator this
measures the replay itself (it should be small); on a real-robot log it is the sim-to-real gap, joint by joint.
"""
import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from controller import DT  # noqa: E402
from lockstep_vendor import flatten  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", type=float, default=0.5, help="s of open-loop replay from each logged state")
    ap.add_argument("--damping", type=float, default=None)
    ap.add_argument("--kp", type=float, default=40.0)
    ap.add_argument("--kd", type=float, default=1.0)
    a = ap.parse_args()
    f = np.load(a.log)["frames"]
    t, quat, gyro, q, dq, tgt = f[:, 0], f[:, 1:5], f[:, 5:8], f[:, 8:20], f[:, 20:32], f[:, 32:44]
    height = f[:, 47]
    m = mujoco.MjModel.from_xml_path(str(Path(os.environ["UNITREE_MUJOCO"]) / "unitree_robots" / "go2" / "scene.xml"))
    m.opt.timestep = 0.002
    flatten(m)
    if a.damping is not None:
        m.dof_damping[6:] = a.damping
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    names = ["FL", "FR", "RL", "RR"]
    ours = [f"{leg}_{j}_joint" for leg in names for j in ("hip", "thigh", "calf")]
    act = np.array([ours.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])) for i in range(m.nu)])
    n_win, n_sub = int(round(a.window / DT)), int(round(DT / m.opt.timestep))
    errs, tilts = [], []
    for s in range(0, len(f) - n_win, n_win):
        d = mujoco.MjData(m)
        d.qpos[2], d.qpos[3:7], d.qpos[7:19] = height[s], quat[s], q[s]
        d.qvel[3:6], d.qvel[6:18] = gyro[s], dq[s]
        for k in range(s, s + n_win):
            for _ in range(n_sub):
                qq, vv = d.qpos[7:19], d.qvel[6:18]
                d.ctrl[:] = np.clip(a.kp * (tgt[k][act] - qq[act]) - a.kd * vv[act], lo, hi)
                mujoco.mj_step(m, d)
        e = d.qpos[7:19] - q[s + n_win]
        errs.append(np.abs(e))
        up_sim = d.xmat[1].reshape(3, 3)[2, 2]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat[s + n_win])
        tilts.append(abs(up_sim - R[8]))
    E = np.array(errs)
    by = {j: E[:, i::3].mean() for i, j in enumerate(("hip", "thigh", "calf"))}
    print(f"{len(E)} windows of {a.window} s from {Path(a.log).name} ({t[-1] - t[0]:.0f} s): joint gap after a window "
          f"(mean |rad|) hip {by['hip']:.3f}, thigh {by['thigh']:.3f}, calf {by['calf']:.3f}; worst joint "
          f"{ours[int(E.mean(0).argmax())]} {E.mean(0).max():.3f}; uprightness gap {np.mean(tilts):.3f}")


if __name__ == "__main__":
    main()
