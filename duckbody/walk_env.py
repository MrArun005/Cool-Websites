"""Walk and run for the mid duck: SkillEnv's walk reward times imitation of a reference gait.

    python train.py --skills --env walk_env:WalkEnv --model model/duck_mid.xml --task walk --name mid_walk
    python train.py --skills --env walk_env:WalkEnv --model model/duck_mid.xml --task run --name mid_run --resume ...

Same observation / action layout as SkillEnv (the robot swaps brains); only the reward and the
practice commands differ. The Bolt dog's lessons:
  * a reference gait (walk_gait.py, verified open-loop first) gives the "warmer, warmer" signal that
    plain velocity tracking lacks: joint angles and foot contacts are matched to it at the gait phase
    the brain sees (same clock: 1.8 Hz walk, 2.6 Hz run);
  * the speed term is wide (standing still still scores a little) plus a linear progress term;
  * speed curriculum: practice commands start at a fraction of the full range and widen over the
    first CURRICULUM steps (exams always use the full range).

reward = upright × height × (0.6·track + 0.4·progress) × (0.3 + 0.7·contact match) × (0.3 + 0.7·joint match)
Standing still under a walk command scores a fraction of walking (checked in __main__).
"""
import os

import numpy as np

from skills_env import GAIT_HZ, SkillEnv
from walk_gait import Gait

# Gait verified open-loop with walk_gait.py (see its __main__).
GAIT = {"walk": dict(swing=0.35, lift=0.015, sway=0.015, lead=0.2, x_off=-0.012),
        "run": dict(swing=0.4, lift=0.015, sway=0.012, lead=0.2, x_off=-0.012)}
CURRICULUM = int(os.environ.get("WALK_CURRICULUM", 4_000_000))   # total training steps to full commands
CURRICULUM_START = int(os.environ.get("WALK_CURRICULUM_START", 0))  # steps already done (when resuming)
N_ENVS = int(os.environ.get("WALK_N_ENVS", 8))
JOINT_W = 0.08     # rad² mean over leg joints: halfway to the reference (~0.2 rad off) earns ~0.6


class WalkEnv(SkillEnv):
    _gaits = {}

    def __init__(self, task=None, **kw):
        super().__init__(task=task, **kw)
        self.local_steps = 0
        mp = kw.get("model_path")
        self.ref = {}
        for t in ("walk", "run"):
            key = (str(mp), t)
            if key not in WalkEnv._gaits:
                WalkEnv._gaits[key] = Gait(model_path=mp, **GAIT[t])
            self.ref[t] = WalkEnv._gaits[key]
        self.leg_q = self.qadr[self.legs]

    def curriculum(self):
        done = CURRICULUM_START + self.local_steps * N_ENVS
        return float(min(1.0, 0.4 + 0.6 * done / max(CURRICULUM, 1)))

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.command in GAIT_HZ:
            if self.practice and self.task == "walk":
                self.vel_cmd = self.vel_cmd * self.curriculum()
            elif self.practice and self.task == "run":
                # run: from a fast walk up to the full run range
                lo = self.vel_cmd[0] * self.curriculum()
                self.vel_cmd = np.array([max(lo, 0.25 * self.L), 0.0, self.vel_cmd[2]])
            self.table = self.ref[self.command].fast_table(self.vel_cmd, GAIT_HZ[self.command])
            obs = self._obs()
        return obs, info

    def reference(self):
        ctrl, stance = self.table
        n = len(ctrl)
        x = (GAIT_HZ[self.command] * self.t * self.dt % 1.0) * n
        k, f = int(x) % n, x - int(x)
        q = (1 - f) * ctrl[k] + f * ctrl[(k + 1) % n]
        return q, stance[int(round(x)) % n]

    def step(self, action):
        self.local_steps += 1
        return super().step(action)

    def _reward(self, action):
        reward, info = super()._reward(action)
        if self.command not in GAIT_HZ:
            return reward, info
        q_ref, stance = self.reference()
        err = np.mean((self.data.qpos[self.leg_q] - q_ref[self.legs]) ** 2)
        joint = float(np.exp(-err / JOINT_W))
        contact = float(np.mean(self.feet_on_ground() == stance))
        info.update(joint_match=joint, contact_match=contact)
        pen = 0.0
        if self.smooth_weight:            # SkillEnv subtracted it from the product; re-apply after scaling
            torque = self.data.actuator_force
            pen = self.smooth_weight * (0.5 * np.mean((action - self.last_action) ** 2)
                                        + 0.2 * np.mean((torque / self.nominal["force"][:, 1]) ** 2)
                                        + 0.001 * np.mean(self.data.qvel[self.dadr] ** 2))
        core = reward + pen
        return float(core * (0.3 + 0.7 * contact) * (0.3 + 0.7 * joint) - pen), info


def check(task="walk", seconds=4.0, seeds=(0, 1, 2)):
    """What the reward pays per step under the same command for:
      still      zero action (servos hold the home pose)
      open_loop  the reference played on the servos, no feedback (it lags and sags, so pays less)
      ideal      the reference played kinematically (base carried at the commanded speed, feet on the
                 floor): what a brain that walks like the reference at the command would earn.
    """
    import mujoco
    from walk_gait import HERE
    out = {}
    cmd = np.array([0.12, 0.0, 0.0]) if task == "walk" else np.array([0.3, 0.0, 0.0])
    for mode in ("still", "open_loop", "ideal"):
        per_step = []
        for s in seeds:
            env = WalkEnv(task=task, randomize=False, seed=s, model_path=str(HERE / "model" / "duck_mid.xml"),
                          max_seconds=seconds, practice=False)
            env.reset(seed=s)
            env.vel_cmd = cmd.copy()
            env.table = env.ref[task].fast_table(env.vel_cmd, GAIT_HZ[task])
            m, d = env.model, env.data
            done, total, n = False, 0.0, 0
            while not done:
                a = np.zeros(env.action_space.shape)
                if mode == "open_loop":
                    q, _ = env.reference()
                    ramp = min(1.0, env.t * env.dt / 0.5)
                    a = np.clip(ramp * (q[env.pol] - env.home[env.pol]) / env.action_scale, -1, 1)
                if mode == "ideal":
                    env.t += 1
                    q, _ = env.reference()
                    d.qpos[env.qadr[env.legs]] = q[env.legs]
                    d.qpos[3:7] = [1, 0, 0, 0]
                    d.qpos[0] += cmd[0] * env.dt
                    d.qvel[:] = 0
                    d.qvel[0] = cmd[0]
                    mujoco.mj_kinematics(m, d)
                    d.qpos[2] -= min(env._sole_bottom(g) for g in env.sole_ids) + 0.0005
                    mujoco.mj_forward(m, d)
                    r, _ = env._reward(a)
                    term, trunc = False, env.t >= env.max_steps
                else:
                    _, r, term, trunc, _ = env.step(a)
                total += r
                n += 1
                done = term or trunc
            per_step.append(round(total / n, 3))
        out[mode] = per_step
    return out


if __name__ == "__main__":
    import sys
    print(check(sys.argv[1] if len(sys.argv) > 1 else "walk"))
