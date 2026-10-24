"""Every skill for the v3 / mid duck, on one shared input layout (LESSONS §3), one brain per skill.

    obs = DuckEnv's layout, with the command one-hot widened to all seven commands:
          [stand, walk, run, sit, lie, jump, roll]; 60 values for duck_mid.xml.

Tasks (start → command):
  stand          standing → stand      idle: shoves, head looking around, talking
  walk           standing → walk       any direction and turn rate (10%: walk in place)
  run            standing → run        fast forward, 2.6 Hz gait clock
  sit            standing → sit        fold down to the deepest crouch and stay still (sleep pose)
  sit_to_stand   crouched → stand      wake up and stand
  lie            standing → lie        lower the body onto the floor on its belly, gently
  get_up         lying anywhere → stand
  jump           standing → jump       hop: both feet off the floor, land on them, again
  roll           crouched → roll       forward roll on the ground, back onto its feet

Rewards keep DuckEnv's rules: products of terms, references that move at a fixed rate (no jackpot),
nothing for being low or fallen unless being low is the task, and contact/orientation checks for
the tricks. An aerial flip is not a task: it needs ~10x the leg power of these servos.
"""
import mujoco
import numpy as np

import env as base
from env import CTRL_HZ, DuckEnv

SKILL_COMMANDS = ["stand", "walk", "run", "sit", "lie", "jump", "roll"]
SKILL_TASKS = ["stand", "walk", "run", "sit", "sit_to_stand", "lie", "get_up", "jump", "roll"]
TASK_COMMAND = {"stand": "stand", "walk": "walk", "run": "run", "sit": "sit", "sit_to_stand": "stand",
                "lie": "lie", "get_up": "stand", "jump": "jump", "roll": "roll"}
START = {"stand": "stand", "walk": "stand", "run": "stand", "sit": "stand", "sit_to_stand": "sit_to_stand",
         "lie": "stand", "get_up": "get_up", "jump": "stand", "roll": "sit_to_stand"}
SECONDS = {"stand": 10, "walk": 10, "run": 8, "sit": 6, "sit_to_stand": 5, "lie": 6, "get_up": 6,
           "jump": 6, "roll": 4}
GAIT_HZ = {"walk": 1.8, "run": 2.6}
QUIET = ("lie", "jump", "roll")          # no head turning or talking during these


class SkillEnv(DuckEnv):
    def __init__(self, task=None, **kw):
        kw.pop("task_weights", None)
        super().__init__(task=None, task_weights=(1, 1, 1, 1), **kw)
        self.forced_skill = task
        m, d = self.model, self.data
        n = len(SKILL_COMMANDS) - len(base.COMMANDS)
        self.obs_size += n
        self.observation_space = base.gym.spaces.Box(-np.inf, np.inf, (self.obs_size,), np.float32)
        # Lying height: base height when the lowest trunk shell touches the floor, upright.
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.home_q
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        self.trunk_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.trunk and m.geom_contype[g] + m.geom_conaffinity[g] > 0]
        low = min(self._geom_bottom(g) for g in self.trunk_geoms)
        self.lie_height = 1.0 - low + 0.002
        self.sit_height = self.crouch_height

    def _geom_bottom(self, g):
        m, d = self.model, self.data
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            return d.geom_xpos[g][2] - m.geom_rbound[g]
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()

    # ---------- episode ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        task = (options or {}).get("task") or self.forced_skill or SKILL_TASKS[r.integers(len(SKILL_TASKS))]
        super().reset(options={"task": START[task]})
        self.task, self.command = task, TASK_COMMAND[task]
        self.vel_cmd = np.zeros(3)
        if task == "walk" and r.random() > 0.1:
            self.vel_cmd = np.array([r.uniform(-0.1, 0.25) * self.L, r.uniform(-0.1, 0.1) * self.L, r.uniform(-0.6, 0.6)])
        elif task == "run":
            self.vel_cmd = np.array([r.uniform(0.3, 0.55) * self.L, 0.0, r.uniform(-0.3, 0.3)])
        self.max_steps = int((self.max_seconds or SECONDS[task]) * CTRL_HZ)
        self.start_height = float(self.data.qpos[2])
        self.start_up = self.posture()["up"]
        self.roll_progress, self.best_hop, self.vx_err = 0.0, 0.0, 1.0
        return self._obs(), {"task": task}

    def _personality(self, target):
        if self.task in QUIET and self.ext_ids:
            for k, i in self.ext_ids.items():
                target[i] = self.home[i]
            return
        super()._personality(target)

    def _obs(self):
        cmd_now = self.command
        self.command = self._obs_base_command()       # DuckEnv only knows stand / sit / walk
        o = super()._obs()
        self.command = cmd_now
        # Swap DuckEnv's 3-command one-hot for the 7-command one.
        n = len(self.model.actuator_ctrlrange) * 2 + len(self.pol) + 9
        cmd = np.zeros(len(SKILL_COMMANDS))
        cmd[SKILL_COMMANDS.index(self.command)] = 1
        moving = self.command in GAIT_HZ
        phase = 2 * np.pi * GAIT_HZ.get(self.command, 0) * self.t * self.dt
        rest = np.concatenate([self.vel_cmd if moving else np.zeros(3),
                               [np.sin(phase), np.cos(phase)] if moving else [0.0, 0.0], self.feet_on_ground()])
        return np.concatenate([o[:n], cmd, rest]).astype(np.float32)

    def _obs_base_command(self):
        return "walk" if self.command in GAIT_HZ else ("sit" if self.command in ("sit", "lie") else "stand")

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        target = self.home.copy()
        target[self.pol] += self.action_scale * action
        self._personality(target)
        target = np.clip(target, self.lo, self.hi)
        self.pending.append(target)
        d.ctrl[:] = self.pending[-1 - self.latency]
        self.pending = self.pending[-2:]
        if self.randomize and self.task in ("stand", "walk", "run") and self.t == self.next_push:
            d.xfrc_applied[self.trunk, :2] = self.rng.normal(0, 6.0 * self.M, 2)
            self.next_push += int(self.rng.integers(150, 300))
        mujoco.mj_step(m, d, nstep=self.frame_skip)
        d.xfrc_applied[self.trunk] = 0
        self.t += 1
        self.roll_progress += d.qvel[4] * self.dt          # body-frame pitch rate: + = nose goes down

        reward, info = self._reward(action)
        self.last_action = action
        if self.command in GAIT_HZ:
            v = self.vel_cmd
            err = abs(info["vx"] - v[0]) + abs(info["vy"] - v[1]) + 0.2 * self.L * abs(info["yaw_rate"] - v[2])
            self.vx_err = 0.98 * self.vx_err + 0.02 * err
        terminated = bool(info["up"] < 0.3 and self.task in ("stand", "walk", "run", "sit", "jump"))
        if terminated:
            reward = 0.0
        truncated = self.t >= self.max_steps
        info["task"] = self.task
        info["success"] = self.success(info)
        return self._obs(), float(reward), terminated, bool(truncated), info

    # ---------- rewards ----------
    def _trunk_on_floor(self):
        d = self.data
        for c in d.contact[:d.ncon]:
            if self.floor in (c.geom[0], c.geom[1]) and (c.geom[0] in self.trunk_geoms or c.geom[1] in self.trunk_geoms):
                return True
        return False

    def _reward(self, action):
        cmd = self.command
        if cmd == "stand":
            return super()._reward(action)
        p = self.posture()
        L, s = self.L, self.t * self.dt
        sh = 0.03 * L
        still = np.exp(-(p["vx"] ** 2 + p["vy"] ** 2) / (0.1 * L) ** 2) * np.exp(-(p["yaw_rate"] / 0.6) ** 2)
        up_at = lambda lim: 1.0 if p["up"] >= lim else float(np.exp(-((p["up"] - lim) / 0.15) ** 2))
        if cmd in ("walk", "run"):
            v = self.vel_cmd
            # Wide tracking term (a narrow one made standing still score ~0, so nothing paid for
            # starting to move: lesson from the Bolt dog), plus a linear progress term along the command.
            speed = np.hypot(v[0], v[1])
            tol = 0.12 * L + 0.5 * speed
            r_track = np.exp(-((p["vx"] - v[0]) ** 2 + (p["vy"] - v[1]) ** 2) / tol ** 2) * np.exp(-((p["yaw_rate"] - v[2]) / 0.6) ** 2)
            progress = float(np.clip((p["vx"] * v[0] + p["vy"] * v[1]) / speed ** 2, 0, 1)) if speed > 0.01 * L else r_track
            h_ref = self.stand_height
            r_h = np.exp(-((p["height"] - h_ref) / (2 * sh if cmd == "walk" else 3 * sh)) ** 2)
            reward = up_at(0.85) * r_h * (0.6 * r_track + 0.4 * progress)
        elif cmd == "sit":
            h_ref = max(self.sit_height, self.start_height - 0.10 * L * s)
            r_h = np.exp(-((p["height"] - h_ref) / sh) ** 2)
            reward = up_at(0.9) * r_h * still
        elif cmd == "lie":
            h_ref = max(self.lie_height, self.start_height - 0.08 * L * s)
            r_h = np.exp(-((p["height"] - h_ref) / sh) ** 2)
            gentle = np.exp(-(max(0.0, -self.data.qvel[2]) / (0.3 * L)) ** 2)        # no flopping down
            on_floor = 1.0 if (h_ref > self.lie_height + 0.01 * L or self._trunk_on_floor()) else 0.5
            reward = up_at(0.7) * r_h * still * gentle * on_floor
        elif cmd == "jump":
            airborne = self.feet_on_ground().sum() == 0
            hop = p["height"] - self.stand_height
            if airborne:
                self.best_hop = max(self.best_hop, hop)
                lift = 0.4 + 0.6 * float(np.clip(hop / (0.03 * L), 0, 1))
            else:
                lift = 0.4 * float(np.exp(-((p["height"] - min(p["height"], self.stand_height)) / sh) ** 2)
                                   if p["height"] > self.crouch_height - sh else 0.1)
            r_spin = np.exp(-(p["yaw_rate"] / 0.8) ** 2) * np.exp(-(p["vx"] ** 2 + p["vy"] ** 2) / (0.15 * L) ** 2)
            reward = up_at(0.85) * r_spin * lift
            h_ref = self.stand_height
        elif cmd == "roll":
            ref = float(np.clip((s - 0.3) * 2 * np.pi / 1.4, 0, 2 * np.pi))
            r_roll = np.exp(-((self.roll_progress - ref) / 1.0) ** 2)
            w = self.data.qvel[3:6]
            r_axis = np.exp(-(w[0] ** 2 + w[2] ** 2) / 3.0 ** 2)                  # roll straight over the nose
            r_end = 1.0
            if ref >= 2 * np.pi:                                                   # back on its feet
                r_end = up_at(0.85) * np.exp(-((p["height"] - self.crouch_height) / (2 * sh)) ** 2)
            reward = r_roll * r_axis * r_end
            h_ref = self.crouch_height
        else:
            return super()._reward(action)
        if self.smooth_weight:
            torque = self.data.actuator_force
            pen = (0.5 * np.mean((action - self.last_action) ** 2)
                   + 0.2 * np.mean((torque / self.nominal["force"][:, 1]) ** 2)
                   + 0.001 * np.mean(self.data.qvel[self.dadr] ** 2))
            reward -= self.smooth_weight * pen
        return float(reward), dict(h_ref=h_ref, up_ref=1.0, **p)

    def success(self, info):
        L, cmd = self.L, self.command
        if cmd in ("stand",):
            return super().success(info)
        if cmd in GAIT_HZ:
            return bool(info["up"] > 0.9 and self.vx_err < (0.06 if cmd == "walk" else 0.1) * L)
        if cmd == "sit":
            return bool(info["up"] > 0.9 and abs(info["height"] - self.sit_height) < 0.02 * L)
        if cmd == "lie":
            return bool(info["up"] > 0.6 and self._trunk_on_floor() and abs(info["vx"]) < 0.05 * L)
        if cmd == "jump":
            return bool(info["up"] > 0.9 and self.best_hop > 0.015 * L)
        if cmd == "roll":
            return bool(self.roll_progress > 2 * np.pi - 0.4 and info["up"] > 0.85)
        return False
