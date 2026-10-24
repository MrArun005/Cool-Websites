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


def pose(front_hip, front_knee, rear_hip, rear_knee):
    """Joint targets for all 12 motors (abd, hip, knee per leg). Front legs bend opposite to rear legs."""
    front, rear = [0.0, front_hip, -front_knee], [0.0, -rear_hip, rear_knee]
    return np.array(front * 2 + rear * 2)


_A = 0.8128  # hip angle for the standing pose: feet straight under the hips
STAND_POSE = pose(_A, 2 * _A, _A, 2 * _A)
SIT_POSE = pose(0.55, 1.1, 1.75, 2.7)    # rear legs folded, rump down, front legs braced
LIE_POSE = pose(1.35, 2.7, 1.35, 2.7)    # all four legs folded under the body

# What each command should look like. Heights and pitches were MEASURED by holding each pose
# in simulation (poses.py), not guessed: body-centre height (m), nose-up pitch (rad), forward
# speed (m/s), and the joint pose to aim for.
TARGETS = {
    "stand": dict(height=0.315, pitch=0.0, speed=0.0, pose=STAND_POSE.round(4).tolist()),
    "sit": dict(height=0.208, pitch=0.54, speed=0.0, pose=SIT_POSE.round(4).tolist()),
    "lie": dict(height=0.131, pitch=0.0, speed=0.0, pose=LIE_POSE.round(4).tolist()),
    "walk": dict(height=0.29, pitch=0.0, speed=0.5, pose=None),
}
ACTION_SCALE = np.tile([0.4, 1.0, 1.2], 4)  # radians per unit action: abd, hip, knee


def trot(phase):
    from reference_gait import trot as reference_trot  # imported late: reference_gait imports this module
    return reference_trot(phase, hip_swing=0.15, knee_lift=0.3)
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class BoltEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, randomize=True, command_period=200, max_steps=600, render_mode=None, seed=None, scale=1.0,
                 skills=None):
        # skills: which commands this brain practises, e.g. ["walk"] for a walk-only brain. The
        # observation layout always keeps all commands, so every brain can be swapped in instantly.
        self.skills = [COMMANDS.index(c) for c in (skills or COMMANDS)]
        # scale < 1 builds a smaller, dynamically similar Bolt (see scale_robot.py). Everything the
        # policy observes and is scored on is converted back to full-size units, so the same brain
        # can drive any size.
        self.s, self.ts = scale, np.sqrt(scale)  # length and time scale
        if scale == 1.0:
            self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        else:
            from scale_robot import scaled_xml
            self.model = mujoco.MjModel.from_xml_string(scaled_xml(scale))
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
        self.total_steps = 0  # this env's steps across all episodes (drives the walk curriculum)

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
    walk_ramp_steps = 2_000_000  # total env steps over which the walk target rises 0.2 -> 0.5 m/s

    def walk_speed(self):
        """Walk target speed: starts easy and rises as training goes on (a curriculum)."""
        done = min(1.0, self.total_steps / self.walk_ramp_steps)
        return 0.2 + (TARGETS["walk"]["speed"] - 0.2) * done

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
        return dict(height=self.data.qpos[2] / self.s, pitch=pitch, roll=roll, up=R[2, 2],
                    vx=v_body[0] / self.ts, vy=v_body[1] / self.ts, yaw_rate=self.data.qvel[5] * self.ts)

    def feet_on_ground(self):
        return np.array([self.data.geom_xpos[g][2] < 0.03 * self.s for g in self.foot_ids], dtype=float)

    def _obs(self):
        d, R = self.data, self._rotation()
        gravity = R.T @ np.array([0, 0, -1.0])
        v_body = R.T @ d.qvel[0:3] / self.ts
        clock = 2 * np.pi * 2.0 * self.t * self.dt / self.ts  # 2 Hz gait clock (full-size time)
        cmd = np.zeros(len(COMMANDS)); cmd[self.command] = 1
        obs = np.concatenate([
            gravity, d.qvel[3:6] * self.ts * 0.25, v_body, [d.qpos[2] / self.s],
            d.qpos[7:] - STAND_POSE, d.qvel[6:] * self.ts * 0.1, self.last_action,
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
        d.qpos[2] = (0.14 if start_lying else 0.33) * self.s
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:] = np.clip(pose + self.rng.normal(0, 0.05, 12), self.lo, self.hi)
        d.ctrl[:] = d.qpos[7:]
        mujoco.mj_forward(m, d)

        self.t = 0
        self.last_action = np.zeros(12)
        self.command = COMMANDS.index(self.forced_command) if self.forced_command else self.rng.choice(self.skills)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(STAND_POSE + ACTION_SCALE * action, self.lo, self.hi)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1
        self.total_steps += 8  # 8 parallel envs share the curriculum clock (train.py default)

        reward, info = self._reward(action)
        self.last_action = action
        fell = info["up"] < 0.2
        if fell:
            reward -= 5.0  # falling ends the attempt; a huge penalty made "never try anything" win
        if self.forced_command is None and self.t % self.command_period == 0:
            self.command = self.rng.choice(self.skills)  # new command, like a real owner would give
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), bool(fell), bool(truncated), info

    def _reward(self, action):
        """Pay for doing what was asked, multiplied together so one bad part spoils the score.

        Additive rewards let the dog settle for a compromise (stand on three legs, half-sit);
        a product of terms doesn't. A small additive share keeps a learning signal when the
        dog is still far from the goal.
        """
        name = COMMANDS[self.command]
        goal, p = TARGETS[name], self.posture()
        feet = self.feet_on_ground()
        upright = np.exp(-(p["roll"] / 0.15) ** 2)

        if name == "walk":
            # Imitate the reference trot (reference_gait.py): match its joint angles and its
            # foot contacts at this moment of the stride. Following an example is far faster to
            # learn than discovering walking from scratch (Disney BDX, Open Duck Mini).
            phase = 2 * np.pi * 2.0 * self.t * self.dt / self.ts  # 2 Hz gait clock, same as the obs
            q_ref, stance_ref = trot(phase)
            gait = np.mean(feet == stance_ref)
            # Width matters: at 0.05 a dog halfway to the trot scored ~0, so there was no
            # "warmer, warmer" signal. At 0.3, halfway earns about a third.
            imitation = np.exp(-np.mean((self.data.qpos[7:] - q_ref) ** 2) / 0.3)
            # Wide enough that standing still still scores a little: a too-strict speed term makes
            # the whole product zero, and then nothing rewards starting to move.
            target = self.walk_speed()
            terms = [
                np.exp(-((p["vx"] - target) / 0.35) ** 2) * np.exp(-(p["vy"] / 0.25) ** 2),
                np.exp(-((p["height"] - goal["height"]) / 0.06) ** 2),
                np.exp(-(p["pitch"] / 0.25) ** 2) * upright,
                np.exp(-(p["yaw_rate"] / 0.6) ** 2),
                0.3 + 0.7 * gait,
                0.2 + 0.8 * imitation,
            ]
            progress = np.clip(p["vx"] / target, -1.0, 1.0)  # every bit of forward speed pays
            progress += imitation  # following the example pays on its own, even before it moves
        else:
            pose_err = np.mean((self.data.qpos[7:] - np.asarray(goal["pose"])) ** 2)
            speed = np.hypot(p["vx"], p["vy"])
            terms = [
                np.exp(-((p["height"] - goal["height"]) / 0.06) ** 2),
                np.exp(-((p["pitch"] - goal["pitch"]) / 0.25) ** 2) * upright,
                np.exp(-pose_err / 0.4),
                np.exp(-(speed / 0.15) ** 2),
            ]
            if name == "stand":
                terms.append((feet.sum() / 4) ** 4)  # three feet down scores 0.32, four scores 1
            progress = 0.0
        reward = 2.0 * np.prod(terms) + 0.3 * np.mean(terms) + 0.5 * progress

        torque = self.data.actuator_force / self.s ** 4  # full-size units
        reward -= 0.02 * np.sum((action - self.last_action) ** 2)  # smooth, not twitchy
        reward -= 2e-4 * np.sum(torque ** 2)                        # don't waste energy
        reward -= 1e-4 * np.sum((self.data.qvel[6:] * self.ts) ** 2)
        return reward * 0.5, dict(command=name, feet=feet, **p)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 540, 960)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.torso
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 1.8 * self.s, -18, 120
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
