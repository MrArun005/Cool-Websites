"""Get up from anywhere, including the back: Microduck's stand-up recipe on this body.

    python train.py --skills --env backkick:BackKickEnv --model model/duck_mid.xml --task get_up --name mid_get_up \\
        --resume ../models/mid-stand/policy.zip

v1 (lie back, kick, stand; 70% back starts, reward only once upright) learned nothing in 1M steps: on the
back the reward was flat, exactly the failure pollen-robotics/microduck_rl documents for its stand-up task
(microduck_standup_env_cfg.py: face-up recovery "froze into do nothing"). This follows their fix:
  * start mix ramps easy -> hard over training (standing, sitting in the perch, face-down, then face-up last):
        steps      0: 40/40/20/0   1M: 25/30/35/10   2.5M: 20/25/30/25   4M: 15/20/30/35   (stand/sit/front/back)
  * face-up starts get +-90 deg of partial roll about the body's long axis (a built-in reverse curriculum)
  * reward, no attempt taxes while the moves are being discovered:
        1.5  x up (cos tilt: pays from any orientation, also upside down)
      + 0.75 x upward CoM speed (m/s, while below 0.125 x L / 0.69 m) - makes any rise attempt positive
      + 1.5  x exp(-(tilt / 0.3)^2) x smoothstep(height: sit -> stand)
      + 3.75 x exp(-((h - stand) / 0.04)^2) x exp(-(tilt / 0.4)^2) x exp(-(leg RMS from home / 0.4)^2)
      - 0.005 x |vertical trunk acceleration|
  * no cut-off; success: upright (up > 0.95) at 0.9 x standing height.
"""
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from skills_env import SkillEnv

HERE = Path(__file__).resolve().parent
STAGES = [(0, (0.40, 0.40, 0.20, 0.00)), (1_000_000, (0.25, 0.30, 0.35, 0.10)),
          (2_500_000, (0.20, 0.25, 0.30, 0.25)), (4_000_000, (0.15, 0.20, 0.30, 0.35))]
N_ENVS = 8                                 # train.py --envs: STAGES count total steps across the envs
SIT_Z = 0.060
ROLL_W = 0.6


def _quat(axis, ang):
    return np.array([np.cos(ang / 2), *(np.sin(ang / 2) * np.asarray(axis, float))])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class BackKickEnv(SkillEnv):
    local_steps = 0
    # get_up training steps done before this process started: set GETUP_STEP_OFFSET when resuming so
    # the start mix does not fall back to the easy stage (local_steps restarts at 0 in a new process).
    step_offset = int(os.environ.get("GETUP_STEP_OFFSET", "0"))

    def __init__(self, task=None, **kw):
        kw.setdefault("max_seconds", 6)
        super().__init__(task=task, **kw)
        poses = json.loads((HERE / "model" / "poses_mid.json").read_text())
        self.perch_q = np.array(poses["perch"]["qpos"]) if "perch" in poses else None
        self.start_kind = "stand"

    def _mix(self):
        steps = self.step_offset + self.local_steps * N_ENVS
        mix = STAGES[0][1]
        for s, m in STAGES:
            if steps >= s:
                mix = m
        return mix

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.task != "get_up":
            return obs, info
        m, d, r = self.model, self.data, self.rng
        mix = self._mix() if self.practice else (0.0, 0.0, 0.5, 0.5)       # exams: face-down or face-up
        kind = ("stand", "sit", "front", "back")[int(r.choice(4, p=np.array(mix) / sum(mix)))]
        kind = (options or {}).get("kind") or kind                          # getup_eval.py forces a start
        self.start_kind = kind
        q = self.home_q.copy()
        yaw = _quat([0, 0, 1], r.uniform(-np.pi, np.pi))
        if kind == "stand":
            q[self.qadr] = np.clip(q[self.qadr] + r.normal(0, 0.03, m.nu), self.lo, self.hi)
            q[3:7] = yaw
            z_lift = 0.002
        elif kind == "sit" and self.perch_q is not None:
            q = self.perch_q.copy()
            q[self.qadr] = np.clip(q[self.qadr] + r.normal(0, 0.12, m.nu), self.lo, self.hi)
            tilt = _qmul(_quat([1, 0, 0], r.uniform(-0.17, 0.17)), _quat([0, 1, 0], r.uniform(-0.17, 0.17)))
            q[3:7] = _qmul(yaw, tilt)
            z_lift = 0.002
        else:
            q[self.qadr] = np.clip(q[self.qadr] + r.uniform(-0.5, 0.5, m.nu), self.lo, self.hi)
            pitch = np.pi / 2 if kind == "front" else -np.pi / 2              # + nose down, - on the back
            rot = _quat([0, 1, 0], pitch)
            if kind == "back":
                rot = _qmul(rot, _quat([1, 0, 0], r.uniform(-np.pi / 2, np.pi / 2)))   # partial roll
            q[3:7] = _qmul(yaw, rot)
            z_lift = r.uniform(0.0, 0.03)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        # Exact lowest point (mesh vertices; v2 used bounding spheres for non-sole geoms, which dropped the
        # standing and perch starts from ~4 cm up: stand starts began at 0.149 m vs 0.111 m standing).
        low = min(self._geom_bottom(g)
                  for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g] + m.geom_conaffinity[g] > 0)
        d.qpos[2] -= low - z_lift
        d.ctrl[:] = q[self.qadr]
        mujoco.mj_forward(m, d)
        if kind in ("front", "back"):
            mujoco.mj_step(m, d, nstep=int(0.4 / m.opt.timestep))           # land and settle
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
        self.pending = [d.ctrl.copy() for _ in self.pending]
        mujoco.mj_subtreeVel(m, d)
        return self._obs(), info

    def _reward(self, action):
        if self.task != "get_up":
            return super()._reward(action)
        self.local_steps += 1
        m, d = self.model, self.data
        p = self.posture()
        up = float(np.clip(p["up"], -1, 1))
        tilt = float(np.arccos(up))
        h, L = p["height"], self.L
        mujoco.mj_subtreeVel(m, d)
        vz = float(d.subtree_linvel[1][2])
        rise = max(0.0, vz) if h < 0.125 * L / 0.69 else 0.0
        x = float(np.clip((h - SIT_Z) / (self.stand_height - SIT_Z), 0, 1))
        smooth = x * x * (3 - 2 * x)
        legs = self.qadr[self.legs]
        rms = float(np.sqrt(np.mean((d.qpos[legs] - self.home_q[legs]) ** 2)))
        composite = np.exp(-((h - self.stand_height) / 0.04) ** 2) * np.exp(-(tilt / 0.4) ** 2) * np.exp(-(rms / 0.4) ** 2)
        # Roll-over term (v3): cos(tilt) is ~0 on the back, on the side and on the belly alike (the flat
        # supine -> prone landscape Microduck documents). Pay for facing the floor while lying: trunk forward
        # axis pointing down (x_z = -1, belly down) = 1, on the side 0.5, on the back 0; fades out as it rises.
        x_z = float(d.xmat[self.trunk].reshape(3, 3)[2, 0])
        roll_over = 0.5 * (1.0 - x_z) * (1.0 - max(up, 0.0))
        reward = (1.5 * up + ROLL_W * roll_over + 0.75 * rise + 1.5 * np.exp(-(tilt / 0.3) ** 2) * smooth + 3.75 * composite
                  - 0.005 * abs(float(d.qacc[2])))
        return float(reward) * self.dt, dict(h_ref=self.stand_height, up_ref=1.0, start=self.start_kind, **p)

    def success(self, info):
        if self.task == "get_up":
            return bool(info["up"] > 0.95 and info["height"] > 0.9 * self.stand_height)
        return super().success(info)
