"""PyBullet side of the sim-to-sim check (sim2sim.py). Runs in its own Python (pybullet has no wheel for Python 3.13
here, so sim2sim.py starts this file with a Python 3.11 venv that has only pybullet and numpy).

One JSON message per line on stdin, one reply per line on stdout:
    {"op": "init", "urdf": path, "dt": 0.002, "friction": 1.0, "motor": {...}, "damping": .., "frictionloss": .., "joints": [names]}
    {"op": "set", "qpos": [19], "qvel": [18]}             MuJoCo convention (see below); reply {}
    {"op": "step", "n": 10, "ctrl": [12], "kp": [12], "kv": [12]}   n physics steps; reply {"qpos": [19], "qvel": [18]}

MuJoCo convention: qpos = torso origin xyz, quaternion wxyz, 12 joint angles; qvel = torso origin velocity (world),
angular velocity (body frame), 12 joint speeds. PyBullet reports the base at its centre of mass, so the converter
uses the base link's inertial offset `ipos` (the URDF base inertial frame has no rotation).

The motors are bolt.xml's, done by hand in torque control at every physics step: a position servo
kp (ctrl - q) - kv qd clipped to bolt_env.motor_limits (33.5 N·m, 420 W, drive fading from 21 to 0 at 25 rad/s),
plus the joint damping and friction loss of bolt.xml. PyBullet's own joint motors and link damping are switched off.
"""
import json
import sys

import numpy as np
import pybullet as p


def quat_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def rot(q_xyzw):
    return np.array(p.getMatrixFromQuaternion(q_xyzw)).reshape(3, 3)


class Sim:
    def init(self, msg):
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(msg["dt"])
        p.setPhysicsEngineParameter(numSolverIterations=int(msg.get("iterations", 50)))
        plane = p.createCollisionShape(p.GEOM_PLANE)
        self.plane = p.createMultiBody(0, plane)
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.bot = p.loadURDF(msg["urdf"], [0, 0, 1], flags=flags)
        names = {p.getJointInfo(self.bot, j)[1].decode(): j for j in range(p.getNumJoints(self.bot))}
        self.j = [names[n] for n in msg["joints"]]
        mu = msg["friction"]
        p.changeDynamics(self.plane, -1, lateralFriction=mu, restitution=0.0)
        for link in [-1] + list(range(p.getNumJoints(self.bot))):
            p.changeDynamics(self.bot, link, lateralFriction=mu, spinningFriction=0.0, rollingFriction=0.0, restitution=0.0,
                             linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        p.setJointMotorControlArray(self.bot, self.j, p.VELOCITY_CONTROL, forces=[0.0] * 12)
        self.ipos = np.array(msg["ipos"])
        self.motor = msg["motor"]
        self.damping, self.frictionloss = msg["damping"], msg["frictionloss"]
        return {}

    def limits(self, qd):
        m = self.motor
        w = np.abs(qd)
        drive = np.minimum(m["torque"], m["power"] / np.maximum(w, 1e-6))
        drive *= np.clip((m["cutoff"] - w) / (m["cutoff"] - m["speed"]), 0.0, 1.0)
        return np.where(qd < 0, -drive, -m["torque"]), np.where(qd >= 0, drive, m["torque"])

    def set(self, msg):
        qpos, qvel = np.array(msg["qpos"]), np.array(msg["qvel"])
        q = [qpos[4], qpos[5], qpos[6], qpos[3]]
        R = rot(q)
        com = qpos[0:3] + R @ self.ipos
        w_world = R @ qvel[3:6]
        v_com = qvel[0:3] + np.cross(w_world, R @ self.ipos)
        p.resetBasePositionAndOrientation(self.bot, com.tolist(), q)
        p.resetBaseVelocity(self.bot, v_com.tolist(), w_world.tolist())
        for k, j in enumerate(self.j):
            p.resetJointState(self.bot, j, float(qpos[7 + k]), float(qvel[6 + k]))
        return {}

    def state(self):
        com, q = p.getBasePositionAndOrientation(self.bot)
        v_com, w_world = p.getBaseVelocity(self.bot)
        R = rot(q)
        origin = np.array(com) - R @ self.ipos
        v = np.array(v_com) - np.cross(np.array(w_world), R @ self.ipos)
        js = p.getJointStates(self.bot, self.j)
        qj = [s[0] for s in js]
        qdj = [s[1] for s in js]
        return dict(qpos=origin.tolist() + quat_wxyz(q).tolist() + qj, qvel=v.tolist() + (R.T @ np.array(w_world)).tolist() + qdj)

    def step(self, msg):
        ctrl, kp, kv = np.array(msg["ctrl"]), np.array(msg["kp"]), np.array(msg["kv"])
        for _ in range(int(msg["n"])):
            js = p.getJointStates(self.bot, self.j)
            q = np.array([s[0] for s in js])
            qd = np.array([s[1] for s in js])
            lo, hi = self.limits(qd)
            tau = np.clip(kp * (ctrl - q) - kv * qd, lo, hi) - self.damping * qd - self.frictionloss * np.tanh(qd / 0.01)
            p.setJointMotorControlArray(self.bot, self.j, p.TORQUE_CONTROL, forces=tau.tolist())
            p.stepSimulation()
        return self.state()


def main():
    sim = Sim()
    for line in sys.stdin:
        msg = json.loads(line)
        out = getattr(sim, msg["op"])(msg)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
