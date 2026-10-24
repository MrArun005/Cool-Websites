"""Roll from the back onto the belly (a trained brain; open-loop kicks only worked from the start they were
searched on: back_roll.py, 1/20 on new back starts). Once face down, the runtime's "get up" stands (20/20).

    python train.py --skills --env rollover_env:RollOverEnv --model model/duck_mid.xml --task get_up \\
        --name mid_rollover --log-std -1.0

Start: lying on the back (nose up 90 deg about a random heading, partial roll +-60 deg about the long axis,
random leg angles), settled. 4 s episodes, no cut-off.
Reward per step: dense along the whole roll, from the IMU gravity in the trunk frame (g, upright = (0, 0, -1)):
    roll    0.5 x (1 + g_x)              (0 on the back with the nose up, 1 nose down)
          + 1.0 if face down (g_x > 0.5 and tilted > 55 deg)            the goal the "get up" plan starts from
          + 0.5 x exp(-|trunk angular velocity|^2 / 4) when face down   (come to rest there)
    - 0.002 x |vertical trunk acceleration|                             (no slamming)
Success: face down and nearly still at the end.
"""
import mujoco
import numpy as np

from backkick import _qmul, _quat
from skills_env import SkillEnv


class RollOverEnv(SkillEnv):
    def __init__(self, task=None, **kw):
        kw.setdefault("max_seconds", 4)
        super().__init__(task=task, **kw)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        m, d, r = self.model, self.data, self.rng
        q = self.home_q.copy()
        q[self.qadr] = np.clip(q[self.qadr] + r.uniform(-0.6, 0.6, m.nu), self.lo, self.hi)
        rot = _qmul(_quat([0, 1, 0], -np.pi / 2), _quat([1, 0, 0], r.uniform(-1.05, 1.05)))
        q[3:7] = _qmul(_quat([0, 0, 1], r.uniform(-np.pi, np.pi)), rot)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.qpos[2] = 0.12 * self.L / 0.69
        d.ctrl[:] = q[self.qadr]
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=int(0.8 / m.opt.timestep))
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.pending = [d.ctrl.copy() for _ in self.pending]
        return self._obs(), info

    def _gravity(self):
        R = self.data.xmat[self.trunk].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])           # gravity in the trunk frame (upright: (0, 0, -1))

    def _face_down(self):
        g = self._gravity()
        return g[0] > 0.5 and -g[2] < 0.6

    def _reward(self, action):
        p = self.posture()
        g = self._gravity()
        down = self._face_down()
        w = self.data.qvel[3:6]
        reward = 0.5 * (1 + g[0]) + (1.0 if down else 0.0)
        if down:
            reward += 0.5 * float(np.exp(-float(w @ w) / 4.0))
        reward -= 0.002 * abs(float(self.data.qacc[2]))
        return float(reward), dict(h_ref=self.stand_height, up_ref=1.0, gx=float(g[0]), **p)

    def success(self, info):
        w = self.data.qvel[3:6]
        return bool(self._face_down() and float(w @ w) < 1.0)
