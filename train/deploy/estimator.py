"""Body height and velocity from what a real Go2 measures (PLAN.md phase 3, route A): the IMU's rotation and the joint
angles, by leg kinematics. No sensor on the robot gives either directly.

Height: the paws in contact are the lowest ones (within --contact-band of the lowest, in a gravity-aligned frame);
height = how far the body centre sits above them, plus the paw radius. Flat floor assumed (as the brain is trained).
Velocity (leg odometry): a paw on the ground does not move, so the body moves opposite to how those paws move
relative to it: v_body = -(J(q) dq + w x r) for each paw in contact (joint speeds and the gyro, exact kinematics, no
differencing), averaged over the paws in contact and smoothed (EMA). With dq/gyro None it falls back to differencing
paw positions between steps (noisier).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import robot_bodies  # noqa: E402
from controller import DT, quat_to_mat  # noqa: E402


class LegEstimator:
    def __init__(self, body="go2", contact_band=0.02, alpha=0.4):
        b = robot_bodies.get(body)
        self.m = b.model()
        self.d = mujoco.MjData(self.m)
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(self.m.ngeom)]
        self.paws = [names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot") if n in names] or \
                    [g for g in range(self.m.ngeom) if self.m.geom_priority[g] == 1]
        self.r = float(self.m.geom_size[self.paws[0]][0])
        self.band, self.alpha = contact_band, alpha
        self.prev, self.prev_contact, self.v = None, None, np.zeros(3)

    def paws_rel(self, R, q):
        """Paw centres relative to the body centre, in a frame aligned with the world (rotation R applied)."""
        d = self.d
        d.qpos[:3] = 0.0
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:19] = q
        mujoco.mj_kinematics(self.m, d)
        return np.array([R @ d.geom_xpos[g] for g in self.paws])

    def paw_vel_body(self, q, dq, gyro):
        """Paw centre velocities relative to the body, in the body frame: J(q) dq + w x r (mujoco, body at rest)."""
        d = self.d
        d.qpos[:3] = 0.0
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:19] = q
        d.qvel[:3] = 0.0
        d.qvel[3:6] = gyro
        d.qvel[6:18] = dq
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_comVel(self.m, d)
        out = np.zeros((len(self.paws), 3))
        v6 = np.zeros(6)
        for k, g in enumerate(self.paws):
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_GEOM, g, v6, 0)   # world (= body) frame
            out[k] = v6[3:]
        return out

    def update(self, quat, q, dq=None, gyro=None):
        """(v_body (3), height) after one 20 ms step."""
        R = quat_to_mat(quat)
        p = self.paws_rel(R, q)
        z = p[:, 2]
        contact = z < z.min() + self.band
        height = float(-z[contact].mean() + self.r)
        if dq is not None and gyro is not None:
            v_body = -self.paw_vel_body(q, dq, gyro)[contact].mean(0)
            self.v = (1 - self.alpha) * self.v + self.alpha * (R @ v_body)
            self.prev, self.prev_contact = p, contact
            return R.T @ self.v, height
        if self.prev is not None:
            both = contact & self.prev_contact
            if both.any():
                v_world = -(p[both] - self.prev[both]).mean(0) / DT
                self.v = (1 - self.alpha) * self.v + self.alpha * v_world
        self.prev, self.prev_contact = p, contact
        return R.T @ self.v, height
