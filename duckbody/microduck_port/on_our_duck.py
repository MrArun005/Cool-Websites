"""Microduck's trained brain on OUR duck's body (model/duck_mid.xml): does it transfer?

    python microduck_port/on_our_duck.py [--policy velstand]

An adapter translates between the two bodies, joint by joint (same names on both ducks):
  their joint angle = their home + s x (our angle - our home)      (into the brain: obs joint_pos / vel)
  our servo target  = our home + s x (their target - their home)   (out of the brain)
s per joint: +1 where both ducks turn the same way. The right knee and ankle are mirrored on Microduck
(home L -0.0049 / R +0.0049, L +0.453 / R -0.453) but not on ours (both knees +1.37), so s = -1 there;
the hip roll / yaw direction is not known from the poses, so both signs are tried. Microduck's passive
neck joints and head roll are fed as at home (our duck has no servo there).
obs (61) = gyro(3), gravity in the trunk frame(3), joint_pos - home(14), joint_vel(14), last action(14),
command(13, zero = stand still), exactly the layout of their harness.py. 50 Hz, our position servos.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from harness import DEFAULT_POSE, JOINT_NAMES, Policy  # noqa: E402

OUR_XML = HERE.parent / "model" / "duck_mid.xml"


def run(policy, start, s_rollyaw, seed=0, seconds=5.0):
    m = mujoco.MjModel.from_xml_path(str(OUR_XML))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    our_home_q = {n: d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in JOINT_NAMES}
    sign = {n: 1.0 for n in JOINT_NAMES}
    for n in ("right_knee", "right_ankle"):
        sign[n] = -1.0
    for n in ("left_hip_roll", "right_hip_roll", "left_hip_yaw", "right_hip_yaw"):
        sign[n] = s_rollyaw
    act = {n: m.actuator(n).id for n in JOINT_NAMES if _has_act(m, n)}
    if start != "standing":
        ang = np.pi / 2 if start == "face_down" else -np.pi / 2
        yaw = rng.uniform(-np.pi, np.pi)
        qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        qp = np.array([np.cos(ang / 2), 0, np.sin(ang / 2), 0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qy, qp)
        d.qpos[3:7] = q
        d.qpos[2] = 0.12
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=int(0.8 / m.opt.timestep))
    last = np.zeros(14, np.float32)
    trunk = m.body("trunk_assembly").id
    n_sub = int(round(0.02 / m.opt.timestep))
    ups = []
    for t in range(int(seconds * 50)):
        R = d.xmat[trunk].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.qvel[3:6].copy()                             # free joint: angular velocity in the body frame
        jp = np.zeros(14, np.float32)
        jv = np.zeros(14, np.float32)
        for i, n in enumerate(JOINT_NAMES):
            j = m.joint(n)
            jp[i] = sign[n] * (d.qpos[m.jnt_qposadr[j.id]] - our_home_q[n]) if n in act else 0.0
            jv[i] = sign[n] * d.qvel[m.jnt_dofadr[j.id]] if n in act else 0.0
        obs = np.concatenate([gyro, grav, jp, jv, last, np.zeros(13)]).astype(np.float32)
        a = policy(obs)
        last = a.astype(np.float32)
        tgt = DEFAULT_POSE + a * policy.action_scale
        for i, n in enumerate(JOINT_NAMES):
            if n in act:
                d.ctrl[act[n]] = our_home_q[n] + sign[n] * (tgt[i] - DEFAULT_POSE[i])
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        ups.append(float(d.xmat[trunk][8]))
    return round(ups[-1], 2), round(float(d.qpos[2]), 3), round(min(ups), 2)


def _has_act(m, n):
    try:
        m.actuator(n)
        return True
    except KeyError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="velstand")
    a = ap.parse_args()
    pol = Policy(a.policy)
    for s in (1.0, -1.0):
        for start in ("standing", "face_down", "face_up"):
            res = [run(pol, start, s, seed=k) for k in range(3)]
            print(f"{a.policy} hip roll/yaw sign {s:+.0f} | {start:9s}: (end up, height, min up) {res}", flush=True)


if __name__ == "__main__":
    main()
