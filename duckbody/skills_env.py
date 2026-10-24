"""Every skill for the v3 / mid duck, on one shared input layout (LESSONS §3), one brain per skill.

    obs = DuckEnv's layout, with the command one-hot widened to all seven commands:
          [stand, walk, run, sit, lie, jump, roll]; 60 values for duck_mid.xml.

Tasks (start → command):
  stand          standing → stand      idle: shoves, head looking around, talking
  walk           standing → walk       any direction and turn rate (10%: walk in place)
  run            standing → run        fast forward, 2.6 Hz gait clock
  squat          standing → sit        fold into the deepest crouch on its feet and hold it
  sit            squatting → sit       really sit: settle back onto belly and thighs and rest there
                                       (chained after squat on the robot)
                                       (sleep pose); measured by poses.py, stored in model/poses_<size>.json
  sit_to_stand   sitting → stand       wake up and stand
  lie            standing → lie        lower the body onto the floor on its belly, gently
  get_up         lying anywhere → stand
  jump           standing → jump       hop: both feet off the floor, land on them, again
  roll           crouched → roll       forward roll on the ground, back onto its feet

Rewards keep DuckEnv's rules: products of terms, references that move at a fixed rate (no jackpot),
nothing for being low or fallen unless being low is the task, and contact/orientation checks for
the tricks. An aerial flip is not a task: it needs ~10x the leg power of these servos.
"""
import json

import mujoco
import numpy as np

import env as base
from env import CTRL_HZ, DuckEnv

SKILL_COMMANDS = ["stand", "walk", "run", "sit", "lie", "jump", "roll"]
SKILL_TASKS = ["stand", "walk", "run", "squat", "sit", "sit_to_stand", "lie", "get_up", "jump", "roll"]
TASK_COMMAND = {"stand": "stand", "walk": "walk", "run": "run", "squat": "sit", "sit": "sit", "sit_to_stand": "stand",
                "lie": "lie", "get_up": "stand", "jump": "jump", "roll": "roll"}
START = {"stand": "stand", "walk": "stand", "run": "stand", "squat": "stand", "sit": "stand",
         "sit_to_stand": "seated", "lie": "stand", "get_up": "get_up", "jump": "stand", "roll": "crouched"}
SECONDS = {"stand": 10, "walk": 10, "run": 8, "squat": 6, "sit": 6, "sit_to_stand": 5, "lie": 6, "get_up": 6,
           "jump": 6, "roll": 4}
GAIT_HZ = {"walk": 1.8, "run": 2.6}
# Gentle landing, Microduck's sitstand recipe (scaled to our ~1/step reward): a per-step cap on
# how fast the body may go down, and a penalty on vertical shocks, both always on.
GENTLE = ("sit", "sit_to_stand", "lie", "get_up")
MAX_DESCENT = 0.05          # m/s
DESCENT_W, SHOCK_W = 4.0, 0.007   # descent 1.5 -> 4 after 2.5M: touch-downs were still 26-41 cm/s
IMPACT_W, IMPACT_FREE = 0.3, 2.0  # belly contact force beyond 2x body weight, per extra body weight
QUIET = ("squat", "sit", "lie", "jump", "roll")   # no head turning or talking during these (calm head)


class SkillEnv(DuckEnv):
    def __init__(self, task=None, practice=True, **kw):
        kw.pop("task_weights", None)
        self.practice = practice          # practice starts in training; exams always start from standing
        super().__init__(task=None, task_weights=(1, 1, 1, 1), **kw)
        self.forced_skill = task
        m, d = self.model, self.data
        if task in GENTLE:
            # Seated/lying contacts need a real contact solve (Microduck sitstand: iterations 30/50).
            # With 1 iteration the belly contact spiked to 140x body weight.
            m.opt.iterations, m.opt.ls_iterations = 30, 50
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
        self.belly_geoms = self.trunk_geoms
        self.weight = float(m.body_subtreemass[1]) * 9.81
        self.thigh_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("knee_and_ankle_assembly", "knee_and_ankle_assembly_3")
                            and m.geom_contype[g] + m.geom_conaffinity[g] > 0]
        self.sit_q = None
        poses = base.Path(str(kw.get("model_path") or base.MODEL)).with_name(
            "poses_" + base.Path(str(kw.get("model_path") or base.MODEL)).stem.replace("duck_", "") + ".json")
        if poses.exists() and "sit" in json.loads(poses.read_text()):
            # Sitting means resting on belly and thighs, not holding a crouch (measured pose, poses.py).
            sit = json.loads(poses.read_text())["sit"]
            self.sit_q = np.array(sit["qpos"])
            sw = json.loads(poses.read_text()).get("swag")
            # Swag is this body's natural stable sit (the leg out stops it rolling back); the robot
            # sits with the swag sit brain, so standing up starts from there.
            self.seat_q = np.array(sw["qpos"]) if sw else self.sit_q
            self.seat_ctrl = np.array(sw["ctrl"]) if sw else None
            self.sit_height, self.sit_up = float(self.sit_q[2]), float(sit["up"])   # already settled when measured
            # Joint targets that produce it (settled angles sag under load; commanding those tips it over).
            self.sit_cmd = self.sit_q.copy()
            if "ctrl_legs" in sit:
                from poses import leg_q
                c = sit["ctrl_legs"]
                leg_q(m, self.sit_cmd, c["hip"], c["knee"], c["ankle"])

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
        super().reset(options={"task": {"seated": "sit_to_stand", "crouched": "sit_to_stand"}.get(START[task], START[task])})
        if START[task] == "seated" and self.sit_q is not None:
            # Start sitting exactly as measured, keeping the seat's lean-back tilt (a random heading
            # only); DuckEnv would rest it on the soles.
            m, d = self.model, self.data
            q = self.seat_q.copy()
            if self.practice and r.random() < 0.3:
                # Practise the hard middle: start part-way from the seat to the upright squat.
                f = r.uniform(0.3, 0.8)
                q[7:] = q[7:] + (self.crouch_q[7:] - q[7:]) * f
                q[2] = q[2] + (self.crouch_height - q[2]) * f
                q[3:7] = base._quat_mul([1, 0, 0, 0], q[3:7]) * (1 - f) + np.array([1.0, 0, 0, 0]) * f
                q[3:7] /= np.linalg.norm(q[3:7])
            q[self.qadr] = np.clip(q[self.qadr] + r.normal(0, 0.02, m.nu), self.lo, self.hi)
            yaw = r.uniform(-np.pi, np.pi)
            q[3:7] = base._quat_mul([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], q[3:7])
            mujoco.mj_resetData(m, d)
            d.qpos[:] = q
            d.ctrl[:] = self.seat_ctrl if self.seat_ctrl is not None else q[self.qadr]
            mujoco.mj_forward(m, d)
        if task == "sit" and self.sit_q is not None:
            # Sit is chained after squat (the robot runs the squat brain, then this one): episodes
            # start squatting; in training 30% start already seated, to learn to stay put without
            # rolling onto its back. (From standing, standing still was the safe choice it never left.)
            u = r.random() if self.practice else 0.0
            start = self.crouch_q if u < 0.7 else self.sit_q
            u = 0.0 if u < 0.7 else 0.5
            if start is not None:
                m, d = self.model, self.data
                q = start.copy()
                q[self.qadr] = np.clip(q[self.qadr] + r.normal(0, 0.02, m.nu), self.lo, self.hi)
                if u < 0.3:
                    q[2] = self.crouch_height
                yaw = r.uniform(-np.pi, np.pi)
                q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)] if u < 0.3 else q[3:7]
                mujoco.mj_resetData(m, d)
                d.qpos[:] = q
                d.ctrl[:] = (self.sit_cmd if u >= 0.3 else q)[self.qadr]
                mujoco.mj_forward(m, d)
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
        # Sitting back rocks the body far backwards on the way down (measured: up dips to ~0.3), so a
        # sit episode only ends when the duck is actually on its back.
        terminated = bool((info["up"] < 0.3 and self.task in ("stand", "walk", "run", "squat", "jump"))
                          or (info["up"] < 0.0 and self.task == "sit")
                          # rising from the seat: tipping past ~65° is a fall (the first try flipped over)
                          or (info["up"] < 0.4 and self.task == "sit_to_stand" and self.sit_q is not None))
        if terminated:
            reward = 0.0
        truncated = self.t >= self.max_steps
        info["task"] = self.task
        info["success"] = self.success(info)
        return self._obs(), float(reward), terminated, bool(truncated), info

    # ---------- rewards ----------
    def _touching(self, geoms):
        d = self.data
        if d.ncon == 0:
            return False
        g = d.contact.geom[:d.ncon]
        return bool(((g == self.floor).any(1) & np.isin(g, geoms).any(1)).any())

    def _trunk_on_floor(self):
        return self._touching(self.trunk_geoms)

    def _rise_reward(self, action):
        """Sit to stand, first link of the chain: from the swag seat, rock forward into an upright
        squat (the stand brain takes it from there). From seat to full standing in one go it never
        left the seat (0/5 at 1M, max height 6.5 cm of 11.1)."""
        p = self.posture()
        L, s, sh = self.L, self.t * self.dt, 0.03 * self.L
        h_ref = min(self.crouch_height, self.start_height + 0.05 * L * s)
        r_h = np.exp(-((p["height"] - h_ref) / (2 * sh)) ** 2)     # above the squat is as wrong as below
        up_ref = min(1.0, self.start_up + 0.3 * s)
        r_up = np.exp(-((p["up"] - up_ref) / 0.12) ** 2) if p["up"] < up_ref else 1.0
        legs = self.qadr[self.legs]
        r_pose = np.exp(-np.sum((self.data.qpos[legs] - self.crouch_q[legs]) ** 2) / 1.0)
        still = np.exp(-(p["vx"] ** 2 + p["vy"] ** 2) / (0.15 * L) ** 2)
        reward = r_h * r_up * still * (0.4 + 0.6 * r_pose)
        # Push harder: every bit of rise above the seat pays while staying fairly upright.
        rise = float(np.clip((p["height"] - self.seat_q[2]) / (self.crouch_height - self.seat_q[2]), 0, 1))
        reward += 0.5 * rise * (1.0 if p["up"] > 0.75 else 0.0)
        vz, az = self.data.qvel[2], self.data.qacc[2]
        reward -= DESCENT_W * max(0.0, -vz - MAX_DESCENT) + SHOCK_W * abs(az)
        reward -= IMPACT_W * max(0.0, self._belly_force() / self.weight - IMPACT_FREE)
        return float(reward), dict(h_ref=h_ref, up_ref=up_ref, **p)

    def _belly_force(self):
        d, f6, total = self.data, np.zeros(6), 0.0
        for i in range(d.ncon):
            g = d.contact.geom[i]
            if self.floor in g and (g[0] in self.trunk_geoms or g[1] in self.trunk_geoms):
                mujoco.mj_contactForce(self.model, d, i, f6)
                total += abs(f6[0])
        return total

    def _seated(self):
        return self._trunk_on_floor() and self._thighs_down()

    def _thighs_down(self):
        return self._touching(self.thigh_geoms)

    def _reward(self, action):
        cmd = self.command
        if self.task == "sit_to_stand" and self.sit_q is not None:
            return self._rise_reward(action)
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
        elif cmd == "sit" and (self.task == "squat" or self.sit_q is None):
            # Squat: the core stays straight (body upright within ~10°), only hips/knees/ankles fold.
            h_ref = max(self.crouch_height, self.start_height - 0.10 * L * s)
            r_h = np.exp(-((p["height"] - h_ref) / sh) ** 2)
            # Once the reference is at the bottom, depth matters: a half-squat 1.7 cm too high used to
            # keep half the reward. (Tight tracking during the whole descent left no signal at all.)
            if h_ref <= self.crouch_height + 1e-6:
                r_h *= 0.3 + 0.7 * np.exp(-((p["height"] - self.crouch_height) / (0.4 * sh)) ** 2)
            r_up = np.exp(-((1.0 - p["up"]) / 0.02) ** 2)
            legs = self.qadr[self.legs]
            r_pose = np.exp(-np.sum((self.data.qpos[legs] - self.crouch_q[legs]) ** 2) / 0.5)
            reward = r_up * r_h * still * (0.5 + 0.5 * r_pose)
        elif cmd == "sit":
            # Settle back onto belly and thighs at a pace it can follow (~2 s), then rest. The body may
            # lean back only as far as it has come down: tilt is judged against the height reached.
            h_ref = max(self.sit_height, self.start_height - 0.035 * L * s)
            r_h = np.exp(-((p["height"] - h_ref) / (2 * sh)) ** 2)
            frac = float(np.clip((self.stand_height - p["height"]) / (self.stand_height - self.sit_height), 0, 1))
            up_ref = 1.0 - (1.0 - self.sit_up) * frac
            r_up = np.exp(-((p["up"] - up_ref) / 0.15) ** 2)
            legs = self.qadr[self.legs]
            # Both legs folded like the measured sit (the brain found 'swag': one leg out, pose error ~5).
            r_pose = np.exp(-np.sum((self.data.qpos[legs] - self.sit_cmd[legs]) ** 2) / 1.0)
            gentle = np.exp(-(max(0.0, -self.data.qvel[2]) / (0.3 * L)) ** 2)
            seated = 1.0 if h_ref > self.sit_height + 0.005 else (1.0 if self._seated() else 0.4)
            reward = r_up * r_h * still * gentle * (0.2 + 0.8 * r_pose) * seated
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
        if self.task in GENTLE:
            vz = self.data.qvel[2]
            az = self.data.qacc[2]
            reward -= DESCENT_W * max(0.0, -vz - MAX_DESCENT) + SHOCK_W * abs(az)
            reward -= IMPACT_W * max(0.0, self._belly_force() / self.weight - IMPACT_FREE)
        if self.smooth_weight:
            torque = self.data.actuator_force
            pen = (0.5 * np.mean((action - self.last_action) ** 2)
                   + 0.2 * np.mean((torque / self.nominal["force"][:, 1]) ** 2)
                   + 0.001 * np.mean(self.data.qvel[self.dadr] ** 2))
            reward -= self.smooth_weight * pen
        return float(reward), dict(h_ref=h_ref, up_ref=1.0, **p)

    def success(self, info):
        L, cmd = self.L, self.command
        if self.task == "sit_to_stand" and self.sit_q is not None:
            return bool(info["up"] > 0.95 and abs(info["height"] - self.crouch_height) < 0.015 * L / 0.69
                        and not self._trunk_on_floor())
        if cmd in ("stand",):
            return super().success(info)
        if cmd in GAIT_HZ:
            return bool(info["up"] > 0.9 and self.vx_err < (0.06 if cmd == "walk" else 0.1) * L)
        if cmd == "sit" and (self.task == "squat" or self.sit_q is None):
            return bool(info["up"] > 0.985 and abs(info["height"] - self.crouch_height) < 0.02 * L)
        if cmd == "sit":
            legs = self.qadr[self.legs]
            folded = np.sum((self.data.qpos[legs] - self.sit_cmd[legs]) ** 2) < 0.6   # both legs, not swag
            return bool(self._seated() and folded and abs(info["up"] - self.sit_up) < 0.1 and abs(info["vx"]) < 0.05 * L)
        if cmd == "lie":
            return bool(info["up"] > 0.6 and self._trunk_on_floor() and abs(info["vx"]) < 0.05 * L)
        if cmd == "jump":
            return bool(info["up"] > 0.9 and self.best_hop > 0.015 * L)
        if cmd == "roll":
            return bool(self.roll_progress > 2 * np.pi - 0.4 and info["up"] > 0.85)
        return False
