"""Walk and run for the mid duck: velocity tracking times imitation of a reference gait.

    python walk_bc.py --out runs/mid_walk_bc          # brain that already steps like the reference
    python train.py --skills --env walk_env:WalkEnv --model model/duck_mid.xml --task walk --name mid_walk \
        --resume runs/mid_walk_bc/policy.zip
    python train.py --skills --env walk_env:WalkEnv --model model/duck_mid.xml --task run --name mid_run --resume ...

Same observation / action layout as SkillEnv (the robot swaps brains); only the reward, the exam and
the practice commands differ.
  * a reference gait (walk_gait.py) gives the "warmer, warmer" signal that plain velocity tracking
    lacks: joint angles and foot contacts are matched to it at the gait phase the brain sees (same
    clock: 1.8 Hz walk, 2.6 Hz run);
  * the first run (runs/mid_walk_v1, from a blank brain) settled on standing and absorbing shoves:
    every early attempt to step fell. The brain now starts as a behaviour-cloned copy of the
    reference (walk_bc.py), so it steps from the first update;
  * velocities are tracked averaged over about a gait cycle: within each step a walking body sways
    and twists, and instantaneous tracking paid standing still more than walking;
  * speed tracking is half wide (standing still keeps a little), half narrow (the exam's precision),
    plus a linear progress term; turning has its own factor.

reward = upright × height × (0.6·track + 0.4·progress) × (0.2 + 0.8·turn)
         × (0.3 + 0.7·contact match) × (0.3 + 0.7·joint match)
Per step (python walk_env.py): standing still under a walk command scores 0.09-0.18, the open-loop
reference 0.04-0.31 (it falls within a few seconds), the reference played kinematically ~0.88.
"""
import os

import numpy as np

from skills_env import GAIT_HZ, SkillEnv
from walk_gait import Gait

# Gait verified open-loop with walk_gait.py (see its __main__).
GAIT = {"walk": dict(swing=0.35, lift=0.015, sway=0.008, lead=0.2, x_off=-0.006),
        "run": dict(swing=0.4, lift=0.015, sway=0.012, lead=0.2, x_off=-0.012)}
CURRICULUM = int(os.environ.get("WALK_CURRICULUM", 4_000_000))   # total training steps to full commands
CURRICULUM_START = int(os.environ.get("WALK_CURRICULUM_START", 0))  # steps already done (when resuming)
N_ENVS = int(os.environ.get("WALK_N_ENVS", 8))
# Fraction of the command range at the start. The brain now starts from a behaviour-cloned walker
# (walk_bc.py) that already covers the full range, so the default is no curriculum.
CURRICULUM_FROM = float(os.environ.get("WALK_CURRICULUM_FROM", 1.0))
JOINT_W = 0.08     # rad² mean over leg joints: halfway to the reference (~0.2 rad off) earns ~0.6
TRACK_NARROW = 0.04   # m/s
TURN_W = 0.3          # rad/s
AVG_ALPHA = 0.05      # velocity average for the tracking terms: time constant 0.4 s (gait cycle 0.56 s)


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
        return float(min(1.0, CURRICULUM_FROM + (1 - CURRICULUM_FROM) * done / max(CURRICULUM, 1)))

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
        self.v_avg = np.zeros(3)
        self.avg_err = 1.0
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
        """Walk / run reward (replaces SkillEnv's for these commands, same spirit: a product of terms).

        core = upright × height × (0.6·track + 0.4·progress) × (0.2 + 0.8·turn)
          track     half wide (standing still keeps a little), half narrow (σ 4 cm/s, the exam's
                    precision): the wide-only term left no pull towards the exact speed
          progress  linear along the command up to the commanded speed
          turn      yaw rate within ~0.4 rad/s: under SkillEnv's 0.6 rad/s width, not turning at all
                    still kept 70% of the reward and the exam failed on yaw
        reward = core × (0.3 + 0.7·contact match) × (0.3 + 0.7·joint match) − smoothness
        """
        if self.command not in GAIT_HZ:
            return super()._reward(action)
        p = self.posture()
        L, v = self.L, self.vel_cmd
        # Body velocity averaged over about a gait cycle: a walking body sways and twists within each
        # step (yaw rate swings ±1 rad/s), so instantaneous tracking paid standing still more than
        # walking at the command.
        self.v_avg += AVG_ALPHA * (np.array([p["vx"], p["vy"], p["yaw_rate"]]) - self.v_avg)
        vx, vy, wz = self.v_avg
        self.avg_err = 0.98 * self.avg_err + 0.02 * (abs(vx - v[0]) + abs(vy - v[1]) + 0.2 * L * abs(wz - v[2]))
        up = 1.0 if p["up"] >= 0.85 else float(np.exp(-((p["up"] - 0.85) / 0.15) ** 2))
        sh = 0.03 * L
        r_h = np.exp(-((p["height"] - self.stand_height) / (2 * sh if self.command == "walk" else 3 * sh)) ** 2)
        speed = np.hypot(v[0], v[1])
        tol = 0.12 * L + 0.5 * speed
        e2 = (vx - v[0]) ** 2 + (vy - v[1]) ** 2
        track = 0.5 * np.exp(-e2 / tol ** 2) + 0.5 * np.exp(-e2 / TRACK_NARROW ** 2)
        progress = float(np.clip((vx * v[0] + vy * v[1]) / speed ** 2, 0, 1)) if speed > 0.01 * L else track
        turn = np.exp(-((wz - v[2]) / TURN_W) ** 2)
        core = up * r_h * (0.6 * track + 0.4 * progress) * (0.2 + 0.8 * turn)
        q_ref, stance = self.reference()
        err = np.mean((self.data.qpos[self.leg_q] - q_ref[self.legs]) ** 2)
        joint = float(np.exp(-err / JOINT_W))
        contact = float(np.mean(self.feet_on_ground() == stance))
        reward = core * (0.3 + 0.7 * contact) * (0.3 + 0.7 * joint)
        if self.smooth_weight:
            torque = self.data.actuator_force
            reward -= self.smooth_weight * (0.5 * np.mean((action - self.last_action) ** 2)
                                            + 0.2 * np.mean((torque / self.nominal["force"][:, 1]) ** 2)
                                            + 0.001 * np.mean(self.data.qvel[self.dadr] ** 2))
        return float(reward), dict(h_ref=self.stand_height, up_ref=1.0, joint_match=joint, contact_match=contact,
                                   track=float(track), turn=float(turn), **p)

    def success(self, info):
        """Exam: upright and going where it was told, judged on the gait-cycle-averaged velocity
        (same error sum and threshold as SkillEnv's, which judges the instantaneous velocity and so
        also fails a walk for its in-step sway and twist; that verdict is kept as info["success_inst"])."""
        if self.command not in GAIT_HZ:
            return super().success(info)
        info["success_inst"] = super().success(info)
        return bool(info["up"] > 0.9 and self.avg_err < (0.06 if self.command == "walk" else 0.1) * self.L)


def check(task="walk", seconds=6.0, seeds=(0, 1, 2), cmds=None):
    """What the reward pays per step, per command, for:
      still      zero action (servos hold the home pose)
      expert     the closed-loop reference walker (walk_expert.py): the brain's starting point
      ideal      the reference played kinematically (base carried at the commanded speed, feet on the
                 floor): what a brain that walks like the reference at the command would earn.
    Falling ends the episode, so per-step values are averaged over the full episode length.
    """
    import mujoco
    from walk_gait import HERE
    from walk_expert import GAINS, Expert
    cmds = cmds or ([(0.12, 0, 0), (0.06, 0, 0), (0.0, 0, 0.5), (0.05, 0.05, 0), (0.0, 0, 0)] if task == "walk"
                    else [(0.3, 0, 0), (0.22, 0, 0.2)])
    out = {}
    for cmd in cmds:
        cmd = np.array(cmd, dtype=float)
        row = {}
        for mode in ("still", "expert", "ideal"):
            per_step = []
            for s in seeds:
                env = WalkEnv(task=task, randomize=False, seed=s, model_path=str(HERE / "model" / "duck_mid.xml"),
                              max_seconds=seconds, practice=False)
                env.reset(seed=s)
                env.vel_cmd = cmd.copy()
                env.table = env.ref[task].fast_table(env.vel_cmd, GAIT_HZ[task])
                ex = Expert(env, **GAINS[task])
                m, d = env.model, env.data
                done, total = False, 0.0
                while not done:
                    a = np.zeros(env.action_space.shape)
                    if mode == "expert":
                        a = ex.action() * min(1.0, env.t * env.dt / 0.3)
                    if mode == "ideal":
                        env.t += 1
                        q, _ = env.reference()
                        d.qpos[env.qadr[env.legs]] = q[env.legs]
                        d.qpos[3:7] = [np.cos(cmd[2] * env.t * env.dt / 2), 0, 0, np.sin(cmd[2] * env.t * env.dt / 2)]
                        d.qpos[:2] += cmd[:2] * env.dt
                        d.qvel[:] = 0
                        d.qvel[:2] = cmd[:2]
                        d.qvel[5] = cmd[2]
                        mujoco.mj_kinematics(m, d)
                        d.qpos[2] -= min(env._sole_bottom(g) for g in env.sole_ids) + 0.0005
                        mujoco.mj_forward(m, d)
                        r, _ = env._reward(a)
                        term, trunc = False, env.t >= env.max_steps
                    else:
                        _, r, term, trunc, _ = env.step(a)
                    total += r
                    done = term or trunc
                per_step.append(round(total / env.max_steps, 3))
            row[mode] = float(np.mean(per_step))
        out[str(cmd.tolist())] = row
    return out


if __name__ == "__main__":
    import sys
    for k, v in check(sys.argv[1] if len(sys.argv) > 1 else "walk").items():
        print(k, v)
