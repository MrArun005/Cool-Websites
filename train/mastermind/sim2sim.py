"""Sim-to-sim check: the MasterMind student on the same Bolt body in a second physics engine (PyBullet).

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/sim2sim.py --engine bullet --brain mastermind/export_v4/student.pt \
        --teachers v4 --n 30 --only gaits rest shove --exam --out mastermind/results/sim2sim/v4_bullet.json
    ... --engine mujoco      the same tests in MuJoCo (the reference)

How: the Master env and the test sequences (test_skills.SEQUENCES through drive.Driver) run unchanged. Only the physics
call is swapped (env._REAL_PHYSICS): each control step the joint targets go to a PyBullet process (bullet_server.py),
which simulates the 10 physics steps of 2 ms with bolt.xml's motor model done in torque control, and returns the
robot's state; that state is written back into MuJoCo's data and mj_forward'ed, so the student's observation, the fall
detector, the tracking scores and every bookkeeping step are the same code as in MuJoCo. A state change made on the
MuJoCo side (a start, a shove) is pushed to PyBullet before the next step.

The body is bolt.xml converted to URDF (write_urdf: links, masses, full inertias, joint frames/axes/limits, collision
primitives). Both engines run the nominal body (randomize=False: no mass/friction/gain randomization and no sensor
noise), so the only difference is the engine: contact model (PyBullet's LCP-style solver vs MuJoCo's soft convex
contacts), integrator, and two things PyBullet cannot express: joint armature (bolt.xml 0.01 kg m^2 rotor inertia
on each joint; --armature-approx adds it to each leg link's inertia instead) and MuJoCo's friction-loss constraint
(here a smooth tanh friction of the same 0.02 N·m). No hurdle in PyBullet, so no jump tests.
PyBullet runs in its own Python (BULLET_PYTHON / --bullet-python; it has no wheel for Python 3.13):
    python3.11 -m venv venv311 && venv311/bin/pip install pybullet numpy
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

import bolt_env  # noqa: E402
import env as E  # noqa: E402
from baseline import run, show  # noqa: E402
from drive import Driver  # noqa: E402
from student import Student  # noqa: E402
from test_master import clip_exam  # noqa: E402

URDF = TRAIN / "robots" / "bolt.urdf"
BULLET_PY = os.environ.get("BULLET_PYTHON", "python3.11")


def _rpy(R):
    """URDF fixed-axis roll-pitch-yaw of a rotation matrix (R = Rz(y) Ry(p) Rx(r))."""
    return np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])


def _origin(pos, quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    r = _rpy(R.reshape(3, 3))
    return f'<origin xyz="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" rpy="{r[0]:.6g} {r[1]:.6g} {r[2]:.6g}"/>'


def write_urdf(path=URDF, xml=TRAIN / "bolt.xml", armature_approx=False):
    """bolt.xml -> URDF. Returns (path, base inertial offset ipos, joint names). armature_approx: each leg link's
    inertia gets the joint's armature added on its diagonal (PyBullet has no rotor inertia)."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    torso = m.body("torso").id
    robot = [b for b in range(m.nbody) if b and m.body_rootid[b] == torso]
    out = ['<?xml version="1.0"?>', '<!-- generated from bolt.xml by train/mastermind/sim2sim.py write_urdf(); do not edit -->',
           '<robot name="bolt">']
    joints = []
    for b in robot:
        name = m.body(b).name
        Rq = np.zeros(9)
        mujoco.mju_quat2Mat(Rq, m.body_iquat[b])
        Rq = Rq.reshape(3, 3)
        inertia = Rq @ np.diag(m.body_inertia[b]) @ Rq.T
        if armature_approx and b != torso:
            j = m.body_jntadr[b]
            inertia = inertia + np.eye(3) * m.dof_armature[m.jnt_dofadr[j]]
        ip = m.body_ipos[b] if b != torso else m.body_ipos[b]
        out.append(f'  <link name="{name}">')
        out.append(f'    <inertial><origin xyz="{ip[0]:.6g} {ip[1]:.6g} {ip[2]:.6g}" rpy="0 0 0"/><mass value="{m.body_mass[b]:.6g}"/>'
                   f'<inertia ixx="{inertia[0, 0]:.6g}" ixy="{inertia[0, 1]:.6g}" ixz="{inertia[0, 2]:.6g}" '
                   f'iyy="{inertia[1, 1]:.6g}" iyz="{inertia[1, 2]:.6g}" izz="{inertia[2, 2]:.6g}"/></inertial>')
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            t, s = m.geom_type[g], m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                shape = f'<box size="{2 * s[0]:.6g} {2 * s[1]:.6g} {2 * s[2]:.6g}"/>'
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                shape = f'<sphere radius="{s[0]:.6g}"/>'
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                shape = f'<capsule radius="{s[0]:.6g}" length="{2 * s[1]:.6g}"/>'
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                shape = f'<cylinder radius="{s[0]:.6g}" length="{2 * s[1]:.6g}"/>'
            else:
                raise ValueError(f"geom type {t} of {m.geom(g).name}")
            out.append(f'    <collision name="{m.geom(g).name or f"g{g}"}">{_origin(m.geom_pos[g], m.geom_quat[g])}<geometry>{shape}</geometry></collision>')
        out.append('  </link>')
        if b != torso:
            j = m.body_jntadr[b]
            assert m.body_jntnum[b] == 1 and np.allclose(m.jnt_pos[j], 0), name
            a, lo_hi = m.jnt_axis[j], m.jnt_range[j]
            jn = m.joint(j).name
            joints.append(jn)
            out.append(f'  <joint name="{jn}" type="revolute"><parent link="{m.body(m.body_parentid[b]).name}"/><child link="{name}"/>'
                       f'{_origin(m.body_pos[b], m.body_quat[b])}<axis xyz="{a[0]:g} {a[1]:g} {a[2]:g}"/>'
                       f'<limit lower="{lo_hi[0]:.6g}" upper="{lo_hi[1]:.6g}" effort="{bolt_env.MOTOR["torque"]:g}" velocity="{bolt_env.MOTOR["cutoff"]:g}"/></joint>')
    out.append('</robot>')
    Path(path).write_text("\n".join(out) + "\n")
    names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert sorted(names) == sorted(joints)
    dof = m.jnt_dofadr[[m.joint(n).id for n in names]]
    return path, m.body_ipos[torso].copy(), names, float(m.dof_damping[dof[0]]), float(m.dof_frictionloss[dof[0]])


class Bridge:
    """Swaps the Master env's physics for PyBullet (see the module doc)."""

    def __init__(self, python, armature_approx=False, iterations=50, friction=1.0):
        path, ipos, names, damping, frictionloss = write_urdf(armature_approx=armature_approx)
        self.proc = subprocess.Popen([python, str(HERE / "bullet_server.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        motor = {k: bolt_env.MOTOR[k] for k in ("torque", "power", "speed", "cutoff")}
        self.call(op="init", urdf=str(path), dt=0.002, friction=friction, motor=motor, damping=damping,
                  frictionloss=frictionloss, joints=names, ipos=ipos.tolist(), iterations=iterations)
        self.last = None
        self.steps = 0

    def call(self, **msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("bullet_server.py died")
        return json.loads(line)

    def physics(self, model, data, n):
        cur = np.concatenate([data.qpos[:19], data.qvel[:18]])
        if self.last is None or not np.array_equal(cur, self.last):   # a start or a shove on the MuJoCo side
            self.call(op="set", qpos=data.qpos[:19].tolist(), qvel=data.qvel[:18].tolist())
        kp = model.actuator_gainprm[:12, 0]
        kv = -model.actuator_biasprm[:12, 2]
        out = self.call(op="step", n=n, ctrl=data.ctrl[:12].tolist(), kp=kp.tolist(), kv=kv.tolist())
        data.qpos[:19], data.qvel[:18] = out["qpos"], out["qvel"]
        data.time += n * model.opt.timestep
        mujoco.mj_forward(model, data)
        self.last = np.concatenate([data.qpos[:19], data.qvel[:18]])
        self.steps += n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bullet", "mujoco"], default="bullet")
    ap.add_argument("--brain", default="mastermind/export_v4/student.pt")
    ap.add_argument("--teachers", default="v4")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*", default=["gaits", "rest", "shove"])
    ap.add_argument("--exam", action="store_true", help="also the per-clip tracking exam (4 starts per clip)")
    ap.add_argument("--randomize", action="store_true", help="randomized physics + sensor noise (PyBullet: not synced, keep off)")
    ap.add_argument("--armature-approx", action="store_true")
    ap.add_argument("--iterations", type=int, default=50, help="PyBullet solver iterations")
    ap.add_argument("--bullet-python", default=BULLET_PY)
    ap.add_argument("--out")
    a = ap.parse_args()
    m = E.Master(seed=a.seed, teachers=a.teachers, randomize=a.randomize)
    if a.engine == "bullet":
        bridge = Bridge(a.bullet_python, armature_approx=a.armature_approx, iterations=a.iterations)
        E._REAL_PHYSICS = bridge.physics
    brain = Student(a.brain, m)
    t0 = time.time()
    res = dict(engine=a.engine, brain=a.brain, teachers=a.teachers, n=a.n, seed=a.seed, randomize=a.randomize,
               armature_approx=a.armature_approx, iterations=a.iterations)
    if a.exam:
        res["exam"] = clip_exam(m, brain, clips=("walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie"))
        print(res["exam"], flush=True)
    rows, secs = run(Driver(m, brain), a.n, a.seed, a.only)
    show(rows)
    res.update(seconds=round(time.time() - t0), switches=rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
