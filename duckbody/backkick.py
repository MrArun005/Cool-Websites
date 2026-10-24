"""Get up off the back (owner's idea): lying on its back, kick a leg to roll over, then stand.

    python train.py --skills --env backkick:BackKickEnv --model model/duck_mid.xml --task get_up --name mid_get_up \\
        --resume runs/mid_sit_to_stand_v2/checkpoints/policy_500000_steps.zip

The first sit-to-stand brains (v1/v2) rocked onto their back and kicked; that move is the starting brain.
  * start: lying on the back (nose up 80-115 deg) with the legs near a tuck, settled 0.5 s (70%), or
    DuckEnv's random drop (30%: side, front, back)
  * no cut-off: rolling over is the whole point
  * reward = max(up, 0)^2 x (height progress lying -> standing)^2 + 0.1 x (up + 1) / 2 (a small pull to turn over),
    + 2 per step when upright (up > 0.9) at squat height or above; small landing-shock penalty
  * success: upright (up > 0.95) at 0.9 x standing height
"""
import mujoco
import numpy as np

from skills_env import SHOCK_W, SkillEnv

UP_BONUS = 2.0
LIE_H = 0.03


class BackKickEnv(SkillEnv):
    def __init__(self, task=None, **kw):
        kw.setdefault("max_seconds", 6)
        super().__init__(task=task, **kw)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.task != "get_up" or self.rng.random() < 0.3:
            return obs, info
        m, d, r = self.model, self.data, self.rng
        q = d.qpos.copy()
        q[self.qadr] = np.clip(self.crouch_q[self.qadr] + r.uniform(-0.4, 0.4, m.nu), self.lo, self.hi)
        a = -r.uniform(np.radians(80), np.radians(115))          # about y, negative: nose up (on the back)
        yaw = r.uniform(-np.pi, np.pi)
        qy = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        qp = [np.cos(a / 2), 0, np.sin(a / 2), 0]
        w1, x1, y1, z1 = qy
        w2, x2, y2, z2 = qp
        q[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
        q[2] = 0.12 * self.L
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.ctrl[:] = q[self.qadr]
        mujoco.mj_step(m, d, nstep=int(0.5 / m.opt.timestep))
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.pending = [d.ctrl.copy()] * len(self.pending) if hasattr(self, "pending") else None
        return self._obs(), info

    def _reward(self, action):
        if self.task != "get_up":
            return super()._reward(action)
        p = self.posture()
        prog = float(np.clip((p["height"] - LIE_H) / (self.stand_height - LIE_H), 0, 1))
        up = max(p["up"], 0.0)
        reward = up ** 2 * prog ** 2 + 0.1 * (p["up"] + 1) / 2     # small pull towards turning upright
        if p["up"] > 0.9 and p["height"] > self.crouch_height - 0.01 * self.L:
            reward += UP_BONUS
        reward -= SHOCK_W * abs(self.data.qacc[2])
        return float(reward), dict(h_ref=self.stand_height, up_ref=1.0, **p)

    def success(self, info):
        if self.task == "get_up":
            return bool(info["up"] > 0.95 and info["height"] > 0.9 * self.stand_height)
        return super().success(info)
