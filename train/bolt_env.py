"""Training ground: teach Bolt to obey commands, the way you'd train a puppy.

Every 1/50 s the policy (a small neural network) sees what the dog feels
(tilt, spin, joint angles, which command it was given) and picks 12 motor
targets. It gets a reward for doing what was asked: the right height and
posture for "sit" or "lie", moving forward for "walk"; and a penalty for
jerky, wasteful movement or falling over. Reinforcement learning (PPO)
nudges the network toward whatever earned more reward. Millions of tries
later, it has learned the tricks.

To make sure it learns the trick and not this one exact simulation, every
episode randomizes the body mass, floor friction and motor strength.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

MODEL = Path(__file__).with_name("bolt.xml")

COMMANDS = ["stand", "sit", "lie", "walk"]
# What each command should look like: body-centre height (m), nose-up pitch (rad), forward speed (m/s).
TARGETS = {
    "stand": dict(height=0.31, pitch=0.0, speed=0.0),
    "sit": dict(height=0.22, pitch=0.5, speed=0.0),
    "lie": dict(height=0.14, pitch=0.0, speed=0.0),
    "walk": dict(height=0.29, pitch=0.0, speed=0.5),
}

_A = 0.8128  # hip angle for the standing pose: feet straight under the hips
STAND_POSE = np.array([0, _A, -2 * _A, 0, _A, -2 * _A, 0, -_A, 2 * _A, 0, -_A, 2 * _A])
_L = 1.357
LIE_POSE = np.array([0, _L, -2 * _L, 0, _L, -2 * _L, 0, -_L, 2 * _L, 0, -_L, 2 * _L])
ACTION_SCALE = np.tile([0.4, 1.0, 1.2], 4)  # radians per unit action: abd, hip, knee
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
TROT = np.array([1, 0, 0, 1])  # FL+RR step together, then FR+RL


class BoltEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, randomize=True, command_period=200, max_steps=600, render_mode=None, seed=None):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10  # 0.002 s physics step × 10 = 50 Hz control
        self.dt = self.model.opt.timestep * self.frame_skip
        self.randomize = randomize
        self.command_period = command_period
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.renderer = None
        self.forced_command = None
        self.rng = np.random.default_rng(seed)

        m = self.model
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.nominal_kp = m.actuator_gainprm[:, 0].copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_friction = m.geom_friction[:, 0].copy()
        self.torso = m.body("torso").id
        self.foot_ids = [m.geom(f).id for f in FEET]

        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (52,), np.float32)

    # ---------- helpers ----------
    def set_command(self, name):
        """Give the dog a command, e.g. env.set_command("sit")."""
        self.command = COMMANDS.index(name)

    def _rotation(self):
        return self.data.xmat[self.torso].reshape(3, 3)

    def posture(self):
        """Height, nose-up pitch, roll, uprightness, body-frame velocity, yaw rate."""
        R = self._rotation()
        pitch = np.arcsin(np.clip(R[2, 0], -1, 1))   # nose-up is positive
        roll = np.arcsin(np.clip(R[2, 1], -1, 1))
        v_body = R.T @ self.data.qvel[0:3]
        return dict(height=self.data.qpos[2], pitch=pitch, roll=roll, up=R[2, 2],
                    vx=v_body[0], vy=v_body[1], yaw_rate=self.data.qvel[5])

    def feet_on_ground(self):
        return np.array([self.data.geom_xpos[g][2] < 0.03 for g in self.foot_ids], dtype=float)

    def _obs(self):
        d, R = self.data, self._rotation()
        gravity = R.T @ np.array([0, 0, -1.0])
        v_body = R.T @ d.qvel[0:3]
        clock = 2 * np.pi * 2.0 * self.t * self.dt  # 2 Hz gait clock
        cmd = np.zeros(len(COMMANDS)); cmd[self.command] = 1
        obs = np.concatenate([
            gravity, d.qvel[3:6] * 0.25, v_body, [d.qpos[2]],
            d.qpos[7:] - STAND_POSE, d.qvel[6:] * 0.1, self.last_action,
            cmd, [np.sin(clock), np.cos(clock)],
        ])
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)  # sensors are noisy
        return obs.astype(np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)

        start_lying = self.randomize and self.rng.random() < 0.25
        pose = LIE_POSE if start_lying else STAND_POSE
        d.qpos[2] = 0.15 if start_lying else 0.33
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:] = np.clip(pose + self.rng.normal(0, 0.05, 12), self.lo, self.hi)
        d.ctrl[:] = d.qpos[7:]
        mujoco.mj_forward(m, d)

        self.t = 0
        self.last_action = np.zeros(12)
        self.command = COMMANDS.index(self.forced_command) if self.forced_command else self.rng.integers(len(COMMANDS))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(STAND_POSE + ACTION_SCALE * action, self.lo, self.hi)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1

        reward, info = self._reward(action)
        self.last_action = action
        fell = info["up"] < 0.2
        if fell:
            reward -= 20.0
        if self.forced_command is None and self.t % self.command_period == 0:
            self.command = self.rng.integers(len(COMMANDS))  # new command, like a real owner would give
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), bool(fell), bool(truncated), info

    def _reward(self, action):
        name = COMMANDS[self.command]
        goal, p = TARGETS[name], self.posture()
        r_height = np.exp(-((p["height"] - goal["height"]) / 0.05) ** 2)
        r_pitch = np.exp(-((p["pitch"] - goal["pitch"]) / 0.2) ** 2)
        r_roll = np.exp(-(p["roll"] / 0.2) ** 2)
        r_speed = np.exp(-((p["vx"] - goal["speed"]) / 0.2) ** 2 - (p["vy"] / 0.2) ** 2)
        r_turn = np.exp(-(p["yaw_rate"] / 0.6) ** 2)
        reward = 1.0 * r_height + 1.0 * r_pitch + 0.5 * r_roll + (1.5 if name == "walk" else 0.5) * r_speed + 0.2 * r_turn

        if name == "walk":
            # Trot rhythm: diagonal pairs on the ground in turn, following the 2 Hz clock.
            stance_a = np.sin(2 * np.pi * 2.0 * self.t * self.dt) > 0
            want = TROT if stance_a else 1 - TROT
            reward += 0.1 * np.sum(self.feet_on_ground() == want)

        torque = self.data.actuator_force
        reward -= 0.02 * np.sum((action - self.last_action) ** 2)  # smooth, not twitchy
        reward -= 2e-4 * np.sum(torque ** 2)                        # don't waste energy
        reward -= 1e-4 * np.sum(self.data.qvel[6:] ** 2)
        return reward * 0.5, dict(command=name, **p)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 540, 960)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.torso
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 1.8, -18, 120
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
