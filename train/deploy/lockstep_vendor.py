"""Sim-to-sim gap finder (PLAN.md phase 2): our Go2Controller on Unitree's own Go2 model (unitree_mujoco's go2.xml:
torque motors, its own joint damping, armature, friction loss, contacts), stepped in lockstep with no DDS or wall-clock
timing in the way, so physics differences can be switched on one at a time.

    UNITREE_MUJOCO=/path/to/unitree_mujoco python deploy/lockstep_vendor.py --plan "trot:10,sit:5,lie:6" \
        [--dt 0.002] [--pd-every 1] [--damping 2.0] [--tries 5]

The motors are the robot's joint PD, computed every --pd-every physics steps (1: continuous, as the real motor
drivers; the vendor simulator recomputes it only when a command message arrives), torque clamped to the model's
ctrlrange. Velocity and height are the true ones (phase 3 replaces them).
"""
import argparse
import os
import sys
from pathlib import Path

import mujoco


def flatten(m):
    """Unitree's go2 scene has two 8 cm bumps and a staircase 1.2-3.4 m ahead; our brains are trained on flat floor:
    switch every world geom but the floor plane off (no contact, invisible)."""
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rgba[g][3] = 0.0
    return m
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from controller import DT, Go2Controller  # noqa: E402

FALL_UP = np.cos(np.radians(60))


def run(ctl, m, plan, pd_every, kp, kd, seed):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else None
    rng = np.random.default_rng(seed)
    d.qpos[2] = 0.33
    d.qpos[7:19] = ctl.stance + rng.normal(0, 0.02, 12)
    # vendor model joint order: FL, FR, RL, RR (as ours): checked by name below
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    act_joint = [m.actuator_trnid[i, 0] for i in range(m.nu)]   # actuator i drives joint act_joint[i]
    qadr = np.array([m.jnt_qposadr[j] for j in act_joint])
    vadr = np.array([m.jnt_dofadr[j] for j in act_joint])
    ours = [ctl.body.joints.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in act_joint]
    n_sub = int(round(DT / m.opt.timestep))
    target = ctl.stance.copy()
    ctl.last_action[:] = 0
    for k in range(int(1.0 / DT)):   # settle 1 s on the stance
        for s in range(n_sub):
            if s % pd_every == 0:
                q, dq = d.qpos[qadr], d.qvel[vadr]
                d.ctrl[:] = np.clip(kp * (target[ours] - q) - kd * dq, lo, hi)
            mujoco.mj_step(m, d)
    t_brain = 0.0
    for name, secs in plan:
        ctl.command(name, steer=0.0, q=d.qpos[7:19])
        for _ in range(int(secs / DT)):
            R = d.xmat[1].reshape(3, 3)
            if R[2, 2] < FALL_UP or (d.qpos[2] < 0.15 and name not in ("sit", "lie", "getup")):   # tipped, or belly-down in a gait
                return t_brain, name, float(d.qpos[2])
            q_ours, dq_ours = np.zeros(12), np.zeros(12)
            q_ours[ours], dq_ours[ours] = d.qpos[qadr], d.qvel[vadr]
            target = ctl.step(d.qpos[3:7].copy(), d.qvel[3:6].copy(), q_ours, dq_ours, R.T @ d.qvel[0:3], float(d.qpos[2]))
            for s in range(n_sub):
                if s % pd_every == 0:
                    q, dq = d.qpos[qadr], d.qvel[vadr]
                    d.ctrl[:] = np.clip(kp * (target[ours] - q) - kd * dq, lo, hi)
                mujoco.mj_step(m, d)
            t_brain += DT
    return t_brain, None, float(d.qpos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=str(HERE.parent.parent / "models" / "bolt-go2-gpu-489M" / "params.pkl"))
    ap.add_argument("--plan", default="trot:10,sit:5,lie:6")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--pd-every", type=int, default=1)
    ap.add_argument("--damping", type=float, default=None)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--stairs", action="store_true", help="keep the scene's bumps and staircase (default: flat floor)")
    a = ap.parse_args()
    xml = Path(os.environ["UNITREE_MUJOCO"]) / "unitree_robots" / "go2" / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    m.opt.timestep = a.dt
    if not a.stairs:
        flatten(m)
    if a.damping is not None:
        m.dof_damping[6:] = a.damping
    ctl = Go2Controller(a.brain)
    kp, kd = ctl.body.entry["kp"], ctl.body.entry["kv"]
    plan = [(c.split(":")[0], float(c.split(":")[1])) for c in a.plan.split(",")]
    total = sum(s for _, s in plan)
    res = [run(ctl, m, plan, a.pd_every, kp, kd, seed) for seed in range(a.tries)]
    falls = [r for r in res if r[1] is not None]
    print(f"dt {a.dt} pd every {a.pd_every} damping {m.dof_damping[6]:g}: {len(falls)}/{a.tries} fell; "
          f"time under the brain {np.mean([r[0] for r in res]):.1f}/{total:g} s; "
          f"falls in {[r[1] for r in falls]}; end heights {[round(r[2], 2) for r in res]}")


if __name__ == "__main__":
    main()
