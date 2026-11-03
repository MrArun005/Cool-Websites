"""Sim-to-sim gap finder (PLAN.md phase 2): our Go2Controller on Unitree's own Go2 model (unitree_mujoco's go2.xml:
torque motors, its own joint damping, armature, friction loss, contacts), stepped in lockstep with no DDS or wall-clock
timing in the way, so physics differences can be switched on one at a time.

    UNITREE_MUJOCO=/path/to/unitree_mujoco python deploy/lockstep_vendor.py --plan "trot:10,sit:5,lie:6" \
        [--dt 0.002] [--pd-every 1] [--damping 2.0] [--tries 5]
    LITE3_MJCF=<DEEP Robotics URDF_model>/Lite3/mjcf/Lite3.xml python deploy/lockstep_vendor.py --body lite3 \
        --brain ../models/bolt-lite3-dog-20.9M/params.pkl --dt 0.001 ...

Lite3: DEEP Robotics' own MJCF (mesh collisions, torque motors +-30 N·m, no joint damping), their RL deploy's joint
PD (kp 30, kd 1), DEEP's joint angles = minus ours (robot_bodies "lite3" sdk_sign), rear legs HL/HR = our RL/RR.

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


RATED = np.tile([30.1, 30.1, 15.7], 4)   # Go2 joint speed limits (rad/s): hip, thigh, calf


def run(ctl, m, plan, pd_every, kp, kd, seed, delay_ms=(0.0, 0.0), skip_p=0.0, est=None, errs=None, speeds=None, sign=1.0, rated=RATED):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else None
    rng = np.random.default_rng(seed)
    d.qpos[2] = ctl.body.stand_height + 0.03
    d.qpos[7:19] = sign * (ctl.stance + rng.normal(0, 0.02, 12))
    # vendor model joint order: FL, FR, RL, RR (as ours): checked by name below
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    act_joint = [m.actuator_trnid[i, 0] for i in range(m.nu)]   # actuator i drives joint act_joint[i]
    qadr = np.array([m.jnt_qposadr[j] for j in act_joint])
    vadr = np.array([m.jnt_dofadr[j] for j in act_joint])
    ren = lambda n: "R" + n[1:] if n[:3] in ("HL_", "HR_") else n   # noqa: E731  (DEEP's rear legs HL/HR = our RL/RR)
    ours = [ctl.body.joints.index(ren(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))) for j in act_joint]
    n_sub = int(round(DT / m.opt.timestep))
    target = ctl.stance.copy()
    ctl.last_action[:] = 0
    for k in range(int(1.0 / DT)):   # settle 1 s on the stance
        for s in range(n_sub):
            if s % pd_every == 0:
                q, dq = d.qpos[qadr], d.qvel[vadr]
                d.ctrl[:] = np.clip(kp * (sign * target[ours] - q) - kd * dq, lo, hi)
            mujoco.mj_step(m, d)
    t_brain = 0.0
    for name, secs in plan:
        q_now = np.zeros(12)
        q_now[ours] = sign * d.qpos[qadr]
        ctl.command(name, steer=0.0, q=q_now)
        for _ in range(int(secs / DT)):
            R = d.xmat[1].reshape(3, 3)
            if R[2, 2] < FALL_UP or (d.qpos[2] < 0.15 and name not in ("sit", "lie", "getup")):   # tipped, or belly-down in a gait
                return t_brain, name, float(d.qpos[2])
            q_ours, dq_ours = np.zeros(12), np.zeros(12)
            q_ours[ours], dq_ours[ours] = sign * d.qpos[qadr], sign * d.qvel[vadr]
            v_true, h_true = R.T @ d.qvel[0:3], float(d.qpos[2])
            if speeds is not None:   # joint speed / rated, as run_go2.py's speed stop sees it every 20 ms
                speeds.append(float(np.max(np.abs(dq_ours) / rated)))
            if est == "ref":      # velocity = what the clip asks for (no measurement); height measured by kinematics
                v_use, h_use = ctl.ref("v_body", 0), h_true
                errs.append(np.concatenate([v_use - v_true, [0.0]]))
            elif est is not None:   # phase 3: what a real Go2 can know, from the IMU rotation and the joints
                v_use, h_use = est.update(d.qpos[3:7].copy(), q_ours)
                errs.append(np.concatenate([v_use - v_true, [h_use - h_true]]))
            else:
                v_use, h_use = v_true, h_true
            new = ctl.step(d.qpos[3:7].copy(), d.qvel[3:6].copy(), q_ours, dq_ours, v_use, h_use)
            if rng.random() < skip_p:   # a missed brain step: the motors keep the old targets for another 20 ms
                new = target
            lag = int(round(rng.uniform(*delay_ms) / 1000 / m.opt.timestep))   # the new targets reach the motors this late
            for s in range(n_sub):
                if s == min(lag, n_sub - 1) or lag == 0:
                    target = new
                if s % pd_every == 0:
                    q, dq = d.qpos[qadr], d.qvel[vadr]
                    d.ctrl[:] = np.clip(kp * (sign * target[ours] - q) - kd * dq, lo, hi)
                mujoco.mj_step(m, d)
            t_brain += DT
    return t_brain, None, float(d.qpos[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="go2", choices=["go2", "lite3"])
    ap.add_argument("--brain", default=None, help="default: the body's current brain")
    ap.add_argument("--plan", default="trot:10,sit:5,lie:6")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--pd-every", type=int, default=1)
    ap.add_argument("--damping", type=float, default=None)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--walk-down", type=float, default=0.0, help="s of walk between trot/pace/canter and sit/lie (0: off)")
    ap.add_argument("--stairs", action="store_true", help="keep the scene's bumps and staircase (default: flat floor)")
    ap.add_argument("--delay-ms", type=float, nargs=2, default=[0.0, 0.0], help="per step, the targets arrive U(lo, hi) ms late (< 20)")
    ap.add_argument("--skip-p", type=float, default=0.0, help="chance per step that the brain's step is missed")
    ap.add_argument("--estimate", action="store_true", help="velocity and height from estimator.LegEstimator, not the truth")
    ap.add_argument("--vel-ref", action="store_true", help="velocity input = the clip's reference velocity (no measurement)")
    a = ap.parse_args()
    if a.body == "lite3":
        xml = Path(os.environ["LITE3_MJCF"])
        a.brain = a.brain or str(HERE.parent.parent / "models" / "bolt-lite3-dog-61.8M" / "params.pkl")
    else:
        xml = Path(os.environ["UNITREE_MUJOCO"]) / "unitree_robots" / "go2" / "scene.xml"
        a.brain = a.brain or str(HERE.parent.parent / "models" / "bolt-go2-damp-798M" / "params.pkl")
    m = mujoco.MjModel.from_xml_path(str(xml))
    m.opt.timestep = a.dt
    if not a.stairs:
        flatten(m)
    if a.damping is not None:
        m.dof_damping[6:] = a.damping
    ctl = Go2Controller(a.brain, body=a.body, walk_down=a.walk_down)
    sign = float(ctl.body.entry.get("sdk_sign", 1))
    rated = np.tile(ctl.body.entry["speed"], 4)
    kp, kd = ctl.body.entry["kp"], ctl.body.entry["kv"]
    plan = [(c.split(":")[0], float(c.split(":")[1])) for c in a.plan.split(",")]
    total = sum(s for _, s in plan)
    errs, speeds = [], []
    if a.estimate:
        from estimator import LegEstimator
    res = [run(ctl, m, plan, a.pd_every, kp, kd, seed, a.delay_ms, a.skip_p,
               "ref" if a.vel_ref else (LegEstimator(body=a.body) if a.estimate else None), errs, speeds, sign, rated) for seed in range(a.tries)]
    if errs:
        e = np.abs(np.array(errs))
        print(f"estimator error (mean abs): velocity x/y/z {e[:, 0].mean():.3f} / {e[:, 1].mean():.3f} / {e[:, 2].mean():.3f} m/s, "
              f"height {e[:, 3].mean() * 100:.1f} cm")
    falls = [r for r in res if r[1] is not None]
    print(f"dt {a.dt} pd every {a.pd_every} damping {m.dof_damping[6]:g} delay {a.delay_ms} ms skip {a.skip_p}{' estimated v/h' if a.estimate else ''}{' v = reference' if a.vel_ref else ''}: {len(falls)}/{a.tries} fell; "
          f"time under the brain {np.mean([r[0] for r in res]):.1f}/{total:g} s; "
          f"falls in {[r[1] for r in falls]}; end heights {[round(r[2], 2) for r in res]}")
    sp = np.array(speeds)
    print(f"joint speed / rated: worst {sp.max():.2f}; steps over 1.2x (run_go2.py would stop) {int((sp > 1.2).sum())} of "
          f"{len(sp)}{' -- FAILS the speed check' if (sp > 1.2).any() else ''}")


if __name__ == "__main__":
    main()
