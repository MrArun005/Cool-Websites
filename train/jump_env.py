"""Teach Bolt to really jump: clear a bar too high to step over, from standing, and land on its feet.

Bolt's belly is ~36 cm off the ground, so it steps over bars up to ~25 cm without leaving the ground
(bolt_env HURDLE_XML). A 40 cm bar can only be cleared by jumping: the whole body has to go up and
over. A hand-made leap (run 14) reared up and tipped over backwards in all 64 versions tried, so this
brain learns the jump from scratch, with the full motor range (not small corrections to a plan).

    python train_jump.py --name jump1 --steps 6000000

Each try: Bolt stands 45-65 cm in front of a bar, gets 2 s, and is paid for moving its hindmost paw
past the bar, a big bonus once all four are past (less if it touched the bar), then for landing and
standing tall. Every moment still in front of the bar costs a little; touching the bar costs; a fall
ends the try. The bar height adapts, like a coach's: it rises 1 cm while clean jumps are common (over
half), and drops when they are rare, from 12 cm up to 48 cm.

Run 15b (paid for standing upright before the jump, bars from 20 cm) learned to stand still and hop in
place: staying put paid steadily, trying risked touch and fall penalties. Run 15c's first 1M steps
learned to launch (all paws up to 22 cm, 0.4 s in the air) but dived over and crashed every time: the
clear bonus beat the fall penalty. So most of the bonus now waits for a landing: still upright 0.5 s
after clearing. By 2.5M it cleared 25 cm bars and landed, but touched the bar every time (sliding over
its top), so touching now costs 5 at once and 1 per step, and only clean jumps earn the big bonuses.
"""
import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import ACTION_SCALE, STANCES, BoltEnv

DOG = STANCES["dog"]


class JumpEnv(BoltEnv):
    OBS = 49

    def __init__(self, randomize=True, seed=None, max_steps=100):
        super().__init__(randomize=randomize, max_steps=max_steps, seed=seed, gait_generator="dog", hurdles=True,
                         skills=["stand"])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.exam_bar = None  # (distance, height): exams set the bar; training draws it
        # adaptive bar height, like a coach: it goes up after clean jumps and down after failed ones
        self.bar_level, self.clean_rate = 0.12, 0.0

    def _bar_height(self):
        return float(np.clip(self.rng.uniform(self.bar_level - 0.05, self.bar_level + 0.03), 0.06, 0.50))

    def _grade(self, clean):
        """After each training try: raise the bar when clean jumps are common, lower it when they are rare."""
        self.clean_rate = 0.9 * self.clean_rate + 0.1 * float(clean)
        if self.clean_rate > 0.5:
            self.bar_level = min(0.48, self.bar_level + 0.01)
        elif self.clean_rate < 0.15:
            self.bar_level = max(0.10, self.bar_level - 0.005)

    def paws_past(self):
        """How far each paw is past the bar, along the jump (negative: still in front of it)."""
        return np.array([np.dot(self.data.geom_xpos[f][:2] - self.hurdle_at, self.hurdle_dir) for f in self.foot_ids])

    def _obs(self):
        d, R = self.data, self._rotation()
        obs = np.concatenate([
            R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3], [d.qpos[2]],
            d.qpos[7:] - DOG, d.qvel[6:] * 0.1, self.last_action,
            [self.hurdle_ahead(), self.hurdle_h, self.t / self.max_steps],
        ])
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:  # the same physics randomization as the other brains
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[2] = 0.45
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:] = d.ctrl[:] = np.clip(DOG + self.rng.normal(0, 0.02, 12), self.lo, self.hi)
        mujoco.mj_forward(m, d)
        for _ in range(25):  # settle into the dog stance on the motors
            mujoco.mj_step(m, d, nstep=self.frame_skip)
        dist, h = self.exam_bar or (self.rng.uniform(0.45, 0.65), self._bar_height())
        self.place_hurdle(dist, h)
        self.t, self.last_action = 0, np.zeros(12)
        self.touched = self.cleared = self.landed = False
        self.clear_t = None
        self.best_past = self.paws_past().min()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(DOG + ACTION_SCALE * a, self.lo, self.hi)  # the full motor range
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1
        self.env_steps += 1
        d, R = self.data, self._rotation()
        p = self.posture()
        hits = self.hurdle_hits()
        first_touch = hits > 0 and not self.touched
        self.touched |= hits > 0
        past = self.paws_past()
        upright = np.exp(-(p["roll"] / 0.4) ** 2) * np.exp(-(p["pitch"] / 0.6) ** 2)
        reward = -1.0 * (hits > 0) - 5.0 * first_touch   # touching the bar is a knocked bar
        just_cleared = False
        if not self.cleared:
            reward -= 0.1  # every moment still in front of the bar costs a little
            reward += 15.0 * max(0.0, past.min() - self.best_past)  # the hindmost paw moving past the bar
            self.best_past = max(self.best_past, past.min())
            reward += 0.5 * np.clip(p["vx"], -0.5, 2.0) * upright
            # tuck: a paw passing over the bar (within 12 cm of it) should be above it, with 3 cm to spare.
            # At 4-5M it cleared and landed 11 times in 12 at 10-12 cm, but the trailing rear legs clipped
            # the bar, so clean jumps stayed rare and the practice bar stuck at 10-12 cm.
            over = np.abs(past) < 0.12
            if over.any():
                z = np.array([self.data.geom_xpos[f][2] for f in self.foot_ids])
                reward += 3.0 * np.sum(np.clip(z[over] - (self.hurdle_h + 0.03), -0.1, 0.1))
            if past.min() > 0.08:  # all four paws past the bar: a little now, the rest for landing
                self.cleared = just_cleared = True
                self.clear_t = self.t
                reward += 5.0 if not self.touched else 1.0
        else:  # landed: stand tall and still
            if not self.landed and self.t - self.clear_t >= 25 and R[2, 2] > 0.85 and p["height"] > 0.3:
                self.landed = True   # still on its feet half a second after clearing: the real prize
                reward += 12.0 if not self.touched else 1.0
            speed = np.hypot(p["vx"], p["vy"])
            reward += 1.0 * np.exp(-((p["height"] - 0.43) / 0.08) ** 2) * upright * np.exp(-(speed / 0.5) ** 2)
        reward -= 0.01 * np.sum((a - self.last_action) ** 2) + 5e-5 * np.sum(d.actuator_force ** 2)
        self.last_action = a
        fell = R[2, 2] < 0.2
        if fell:
            reward -= 10.0 if self.cleared else 5.0   # diving over and crashing must not pay
        if (fell or self.t >= self.max_steps) and self.exam_bar is None:
            self._grade(self.landed and not self.touched)
        info = dict(p, cleared=self.cleared, touched=self.touched, just_cleared=just_cleared, hits=hits,
                    bar=self.hurdle_h, past=float(past.min()), level=self.bar_level, landed=self.landed)
        return self._obs(), float(reward), bool(fell), self.t >= self.max_steps, info


class JumpPolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl"):
        from pathlib import Path
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: JumpEnv()]))
        self.norm.training = False

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, episodes=10, seed=5000, heights=(0.30, 0.40)):
    """Jump the bar from 55 cm away: cleared without touching it, landed, standing at the end, no fall."""
    out = {}
    env = JumpEnv(randomize=True)
    for h in heights:
        wins = 0
        for k in range(episodes):
            env.exam_bar = (0.55, h)
            obs, _ = env.reset(seed=seed + k)
            info, fell = {}, False
            for _ in range(env.max_steps):
                obs, _, fell, trunc, info = env.step(policy.act(obs))
                if fell or trunc:
                    break
            wins += bool(info["cleared"] and not info["touched"] and not fell and info["up"] > 0.9
                         and info["height"] > 0.35)
        out[f"jump_{round(h * 100)}"] = wins / episodes
    return out
