"""Training ground for the duck body (model/duck.xml), following docs/LESSONS.md.

One observation and action layout for every skill (LESSONS §3), so the robot can swap brains:

  obs (61) = gravity in IMU frame (3) | gyro rad/s ×0.25 (3) | accelerometer /9.81 (3)
           | joint angles − home (14) | joint velocities ×0.05 (14) | last action (14)
           | command one-hot [stand, sit, walk] (3) | walk command vx, vy, yaw rate (3)
           | gait clock sin, cos (2; zeros unless walking) | foot switches L, R (2)
  action (14) = joint target offsets from the home pose, × ACTION_SCALE, clipped to the servo range.
  Joint order = actuator order in duck.xml: left leg (5), neck/head (4), right leg (5).

Tasks (training scenarios; the command is what the robot is told):
  stand          start standing (small errors, random pushes), command "stand"
  sit_to_stand   start in the deepest crouch the knees allow, command "stand"
  get_up         start lying in a random pose after a drop, command "stand"
  walk           start standing, command "walk" with a velocity (10% of episodes: zero velocity)

Rewards are products of tracking terms (LESSONS §4: a single failure costs everything), track a
reference that moves at a fixed rate instead of paying a jackpot at the goal, and never pay for
being low or fallen. Smoothness penalties are scaled by `smooth_weight`, which training raises only
after the skill exists. Domain randomization is drawn fresh at every reset (never accumulates).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model" / "duck.xml"

TASKS = ["stand", "sit_to_stand", "get_up", "walk"]
COMMANDS = ["stand", "sit", "walk"]
TASK_COMMAND = {"stand": "stand", "sit_to_stand": "stand", "get_up": "stand", "walk": "walk"}
OBS_SIZE = 61
CTRL_HZ = 50
GAIT_HZ = 1.8
# radians per unit action: hip yaw, roll, pitch, knee, ankle | neck pitch, head pitch, yaw, roll | right leg
_LEG = [0.4, 0.4, 1.0, 1.0, 1.0]
ACTION_SCALE = np.array(_LEG + [0.6, 0.6, 1.0, 0.4] + _LEG)
FORCE_LIMITS = (1.91, 3.23)  # N·m: STS3215 7.4 V stall … upstream (BAM) limit
SOLES = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


class DuckEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, task=None, task_weights=(0.25, 0.2, 0.25, 0.3), randomize=True, smooth_weight=0.0,
                 max_seconds=None, model_path=None, render_mode=None, seed=None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path or MODEL))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.frame_skip = int(round(1 / CTRL_HZ / m.opt.timestep))
        self.dt = m.opt.timestep * self.frame_skip
        self.forced_task, self.task_weights = task, np.asarray(task_weights) / np.sum(task_weights)
        self.randomize, self.smooth_weight = randomize, smooth_weight
        self.max_seconds = max_seconds
        self.render_mode, self.renderer = render_mode, None
        self.rng = np.random.default_rng(seed)

        self.trunk = m.body("trunk_assembly").id
        self.imu = m.site("imu").id
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.dadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.home_q = m.key("home").qpos.copy()
        self.home = m.key("home").ctrl.copy()
        self.sole_ids = [m.geom(g).id for g in SOLES]
        self.sens = {n: (m.sensor_adr[m.sensor(n).id], m.sensor_dim[m.sensor(n).id])
                     for n in ("gyro", "accelerometer", "upvector")}
        self.nominal = dict(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), ipos=m.body_ipos.copy(),
                            kp=m.actuator_gainprm[:, 0].copy(), force=m.actuator_forcerange.copy(),
                            damping=m.dof_damping.copy(), frictionloss=m.dof_frictionloss.copy(),
                            friction=m.geom_friction.copy())
        self.floor = m.geom("floor").id

        # Targets measured from the simulated robot, never guessed (LESSONS §1).
        self.stand_height = self._settle(self.home_q)[2]
        self.crouch_q = self._crouch_pose()
        self.crouch_height = self._settle(self.crouch_q)[2]

        self.action_space = gym.spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)

    # ---------- poses ----------
    def _settle(self, q, seconds=1.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.ctrl[:] = q[self.qadr]
        mujoco.mj_forward(m, d)
        d.qpos[2] -= min(self._sole_bottom(g) for g in self.sole_ids)
        mujoco.mj_step(m, d, nstep=int(seconds / m.opt.timestep))
        return d.qpos.copy()

    def _sole_bottom(self, gid):
        m, d = self.model, self.data
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return (v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid])[:, 2].min()

    def _crouch_pose(self):
        """Feet-flat crouch at the deepest knee angle the servo range allows (minus 3°)."""
        m = self.model
        q = self.home_q.copy()
        knee = min(m.actuator_ctrlrange[m.actuator("left_knee").id, 1] - np.radians(3), 2.1)
        hip = 0.46 * knee
        for side, sign in (("left", 1), ("right", -1)):
            q[m.jnt_qposadr[m.joint(f"{side}_knee").id]] = knee
            q[m.jnt_qposadr[m.joint(f"{side}_hip_pitch").id]] = -sign * hip
            q[m.jnt_qposadr[m.joint(f"{side}_ankle").id]] = -(knee - hip)
        return q

    # ---------- randomization (fresh every episode) ----------
    def _randomize(self):
        m, n, r = self.model, self.nominal, self.rng
        m.body_mass[:] = n["mass"] * r.uniform(0.9, 1.1, m.nbody)
        m.body_inertia[:] = n["inertia"] * (m.body_mass / np.maximum(n["mass"], 1e-9))[:, None]
        m.body_ipos[:] = n["ipos"]
        m.body_ipos[self.trunk] += r.uniform(-0.01, 0.01, 3)              # CoM error ±1 cm
        kp = n["kp"] * r.uniform(0.85, 1.15)
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
        f = r.uniform(*FORCE_LIMITS)                                       # 7.4 V stall … upstream
        m.actuator_forcerange[:] = np.array([-f, f])
        m.dof_damping[:] = n["damping"] * r.uniform(0.8, 1.2)
        m.dof_frictionloss[:] = n["frictionloss"] * r.uniform(0.7, 1.3)
        m.geom_friction[:] = n["friction"]
        m.geom_friction[self.floor, 0] = r.uniform(0.4, 1.0)              # floor wins (priority)
        self.latency = int(r.integers(0, 2))                               # 0–20 ms command delay
        self.imu_tilt = r.normal(0, np.radians(1.0), 2)                    # mounting error

    # ---------- observation ----------
    def _sensor(self, name):
        a, n = self.sens[name]
        return self.data.sensordata[a: a + n]

    def gravity(self):
        R = self.data.site_xmat[self.imu].reshape(3, 3)
        g = R.T @ np.array([0, 0, -1.0])
        if self.randomize:
            g[0] += self.imu_tilt[0]
            g[1] += self.imu_tilt[1]
        return g

    def feet_on_ground(self):
        d = self.data
        on = np.zeros(2)
        for c in d.contact[:d.ncon]:
            for i, s in enumerate(self.sole_ids):
                if s in (c.geom[0], c.geom[1]) and self.floor in (c.geom[0], c.geom[1]):
                    on[i] = 1
        return on

    def _obs(self):
        d = self.data
        cmd = np.zeros(3)
        cmd[COMMANDS.index(self.command)] = 1
        walking = self.command == "walk"
        phase = 2 * np.pi * GAIT_HZ * self.t * self.dt
        obs = np.concatenate([
            self.gravity(), self._sensor("gyro") * 0.25, self._sensor("accelerometer") / 9.81,
            d.qpos[self.qadr] - self.home, d.qvel[self.dadr] * 0.05, self.last_action,
            cmd, self.vel_cmd if walking else np.zeros(3),
            [np.sin(phase), np.cos(phase)] if walking else [0.0, 0.0], self.feet_on_ground(),
        ])
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)
        return obs.astype(np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d, r = self.model, self.data, self.rng
        self.task = (options or {}).get("task") or self.forced_task or TASKS[r.choice(len(TASKS), p=self.task_weights)]
        self.command = TASK_COMMAND[self.task]
        if self.randomize:
            self._randomize()
        else:
            self.latency, self.imu_tilt = 0, np.zeros(2)

        if self.task == "sit_to_stand":
            q = self.crouch_q.copy()
        elif self.task == "get_up":
            q = self._lying_start()
        else:
            q = self.home_q.copy()
        if self.task != "get_up":
            q[self.qadr] = np.clip(q[self.qadr] + r.normal(0, 0.03, m.nu), self.lo, self.hi)
            yaw = r.uniform(-np.pi, np.pi)
            q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_resetData(m, d)
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            d.qpos[2] -= min(self._sole_bottom(g) for g in self.sole_ids)
            d.ctrl[:] = q[self.qadr]
            mujoco.mj_forward(m, d)

        self.vel_cmd = np.zeros(3)
        if self.task == "walk" and r.random() > 0.1:   # 10%: explicit "walk in place / stand still"
            self.vel_cmd = np.array([r.uniform(-0.1, 0.25), r.uniform(-0.1, 0.1), r.uniform(-0.6, 0.6)])
        default_s = {"get_up": 6.0, "sit_to_stand": 5.0}.get(self.task, 10.0)
        self.max_steps = int((self.max_seconds or default_s) * CTRL_HZ)
        self.t = 0
        self.start_height = float(d.qpos[2])
        self.start_up = self.posture()["up"]
        self.last_action = np.zeros(m.nu)
        self.pending = [self.home.copy()] * 2
        self.next_push = int(r.integers(100, 300))
        self.vx_err = 1.0
        return self._obs(), {"task": self.task}

    def _lying_start(self):
        """Drop from 25 cm in a random orientation with random joints, let it come to rest."""
        m, d, r = self.model, self.data, self.rng
        q = self.home_q.copy()
        axis = r.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = r.uniform(np.pi / 2, np.pi)                 # on its side, front or back
        q[3:7] = _quat_mul([np.cos(ang / 2), *(np.sin(ang / 2) * axis)], [1, 0, 0, 0])
        q[2] = 0.25
        q[self.qadr] = np.clip(self.home + r.uniform(-0.6, 0.6, m.nu), self.lo, self.hi)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.ctrl[:] = q[self.qadr]
        mujoco.mj_step(m, d, nstep=int(1.0 / m.opt.timestep))
        d.qvel[:] = 0
        return d.qpos.copy()

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        target = np.clip(self.home + ACTION_SCALE * action, self.lo, self.hi)
        self.pending.append(target)
        d.ctrl[:] = self.pending[-1 - self.latency]
        self.pending = self.pending[-2:]
        if self.randomize and self.task in ("stand", "walk") and self.t == self.next_push:
            d.xfrc_applied[self.trunk, :2] = self.rng.normal(0, 6.0, 2)   # short shove
            self.next_push += int(self.rng.integers(150, 300))
        mujoco.mj_step(m, d, nstep=self.frame_skip)
        d.xfrc_applied[self.trunk] = 0
        self.t += 1

        reward, info = self._reward(action)
        self.last_action = action
        if self.command == "walk":   # running average of the velocity tracking error, for the exam
            v = self.vel_cmd
            err = abs(info["vx"] - v[0]) + abs(info["vy"] - v[1]) + 0.2 * abs(info["yaw_rate"] - v[2])
            self.vx_err = 0.98 * self.vx_err + 0.02 * err
        fallen = info["up"] < 0.3
        terminated = bool(fallen and self.task in ("stand", "walk"))
        if terminated:
            reward = 0.0                                  # no reward, no extra penalty needed: lost future
        truncated = self.t >= self.max_steps
        info["task"] = self.task
        info["success"] = self.success(info)
        return self._obs(), float(reward), terminated, bool(truncated), info

    # ---------- reward ----------
    def posture(self):
        d = self.data
        R = d.xmat[self.trunk].reshape(3, 3)
        v_body = R.T @ d.qvel[0:3]
        return dict(height=float(d.qpos[2]), up=float(R[2, 2]), vx=float(v_body[0]), vy=float(v_body[1]),
                    yaw_rate=float(d.qvel[5]))

    def _reference(self):
        """Height and uprightness the robot should have now: rises at a fixed rate (no jackpot)."""
        if self.task in ("stand", "walk"):
            return self.stand_height, 1.0
        rate_h, rate_up = 0.10, 0.8           # m/s, uprightness per second
        s = self.t * self.dt
        if self.task == "get_up":
            s = max(0.0, s - 0.5)             # half a second to organise itself
        h = min(self.stand_height, self.start_height + rate_h * s)
        up = min(1.0, self.start_up + rate_up * s)
        return h, up

    def _reward(self, action):
        p = self.posture()
        h_ref, up_ref = self._reference()
        r_up = np.exp(-((p["up"] - up_ref) / 0.15) ** 2) if p["up"] < up_ref else 1.0
        r_h = np.exp(-((p["height"] - h_ref) / 0.03) ** 2) if p["height"] < h_ref else np.exp(-((p["height"] - h_ref) / 0.06) ** 2)
        if self.command == "walk":
            v = self.vel_cmd
            r_move = np.exp(-((p["vx"] - v[0]) ** 2 + (p["vy"] - v[1]) ** 2) / 0.1 ** 2) * np.exp(-((p["yaw_rate"] - v[2]) / 0.4) ** 2)
        else:
            r_move = np.exp(-(p["vx"] ** 2 + p["vy"] ** 2) / 0.15 ** 2) * np.exp(-(p["yaw_rate"] / 0.6) ** 2)
        # Gentle pull towards the home pose once standing, so "stand" means the trained stance.
        pose_err = np.sum((self.data.qpos[self.qadr] - self.home)[:5] ** 2 + (self.data.qpos[self.qadr] - self.home)[9:] ** 2)
        r_pose = np.exp(-pose_err / 0.5) if self.command == "stand" else 1.0
        reward = r_up * r_h * r_move * (0.7 + 0.3 * r_pose)
        if self.smooth_weight:
            torque = self.data.actuator_force
            pen = (0.5 * np.mean((action - self.last_action) ** 2) + 0.02 * np.mean(torque ** 2)
                   + 0.001 * np.mean(self.data.qvel[self.dadr] ** 2))
            reward -= self.smooth_weight * pen
        return float(reward), dict(h_ref=h_ref, up_ref=up_ref, **p)

    def success(self, info):
        """Exam check: upright and at standing height (stand tasks) or moving as asked (walk)."""
        ok = info["up"] > 0.9 and info["height"] > self.stand_height - 0.02
        if self.command == "walk":
            ok = ok and self.vx_err < 0.06
        return bool(ok)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.trunk
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 1.0, -15, 140
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
