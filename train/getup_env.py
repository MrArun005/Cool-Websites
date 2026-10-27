"""Teach Bolt to get up after a fall: from lying on its side, on its back or half-tipped over, roll over
and stand in the dog stance, by itself.

In the games a fall used to end with Bolt teleported back onto its feet. A real robot dog has to get up
on its own, so this brain learns it from scratch, with the full motor range (like the jump brain).

    python train_getup.py --name getup1 --steps 15000000

Each try: Bolt is dropped from 0.5 m in a random fallen pose (any roll, legs anywhere) and lands; then
it gets 4 s. It is paid for being upright, then for standing at its height in the dog stance, and for
standing still once up; jerky or wasteful motor use costs a little. A tenth of the tries start already
standing, slightly tipped, so it also learns not to fall over while standing.

Run 17 learned to lie still: a charge on joint speed (0.001 x speed^2, all 12 joints) cost 1-5 per step
while flailing to roll over, far more than being upright paid. A search over motor sequences (no
learning) found a way to roll over from the side with this body, so it is possible. Run 17b: no
joint-speed charge, a fifth of the effort charge, a bonus for each new best uprightness (progress), and
more exploration at the start.

Run 17c (goal="selfright"), the way real robot dogs recover (ANYmal: Lee, Hwangbo & Hutter 2019, a
self-righting skill, then a standing skill, then walking, picked by a selector): this brain only rolls
Bolt onto its belly with its legs tucked and lies still. Bolt's posture brain already stands up from
there (20/20 from the lying pose, 17/20 with the legs 0.6 rad off it). Nothing pays for height.
By 6M it rolled from its back past its side (up to 0.53) but always settled back at a 0.16 tilt. Two
limits: the brain could only command the sideways hip joints to +-0.4 rad (ACTION_SCALE, made for
walking), and the joints stopped at +-0.5 rad (real robot dogs: Unitree Go2 about +-0.84). Run 17d:
Bolt's sideways hip range is +-0.9 rad (the other brains never command more than +-0.4 there), and the
self-righting brain's actions cover each joint's whole range around the dog stance.
"""
import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import ACTION_SCALE, LIE_POSE, STANCES, BoltEnv

DOG = STANCES["dog"]
STAND_HEIGHT = 0.43


class GetUpEnv(BoltEnv):
    OBS = 46

    def __init__(self, randomize=True, seed=None, max_steps=200, goal="stand"):
        super().__init__(randomize=randomize, max_steps=max_steps, seed=seed, gait_generator="dog", skills=["stand"])
        self.goal = goal   # "stand": get all the way up; "selfright": belly down, legs tucked, still
        # self-righting reaches each joint's whole range around the dog stance (walking's ACTION_SCALE
        # only reached +-0.4 rad sideways)
        self.scale = np.maximum(self.hi - DOG, DOG - self.lo) if goal == "selfright" else ACTION_SCALE
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.exam_start = None   # "side", "back", "tipped" or "standing": exams set it; training draws it

    def _obs(self):
        d, R = self.data, self._rotation()
        obs = np.concatenate([
            R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3], [d.qpos[2]],
            d.qpos[7:19] - DOG, d.qvel[6:18] * 0.1, self.last_action,
        ])
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)
        return obs.astype(np.float32)

    def _fallen_pose(self, kind):
        """Roll and pitch (radians) for a start: on its side, on its back, half-tipped, or standing."""
        r = self.rng
        if kind == "side":
            return r.choice([-1, 1]) * r.uniform(1.2, 1.9), r.uniform(-0.4, 0.4)
        if kind == "back":
            return r.choice([-1, 1]) * r.uniform(2.4, np.pi), r.uniform(-0.4, 0.4)
        if kind == "tipped":
            return r.choice([-1, 1]) * r.uniform(0.5, 1.2), r.uniform(-0.6, 0.6)
        return r.uniform(-0.15, 0.15), r.uniform(-0.15, 0.15)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:   # the same physics randomization as the other brains
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)
        kind = self.exam_start or self.rng.choice(["side", "back", "tipped", "standing"], p=[0.35, 0.3, 0.25, 0.1])
        roll, pitch = self._fallen_pose(kind)
        yaw = self.rng.uniform(-np.pi, np.pi)
        # roll about the body's x, then pitch about y, then yaw about the vertical
        qr = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
        qp = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
        qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qp, qr)
        mujoco.mju_mulQuat(q, qy, q.copy())
        d.qpos[3:7] = q
        if kind == "standing":
            legs = DOG + self.rng.normal(0, 0.03, 12)
            d.qpos[2] = 0.45
        else:   # legs anywhere: halfway between the dog stance and a random pose within the joint limits
            legs = 0.5 * DOG + 0.5 * self.rng.uniform(self.lo, self.hi)
            d.qpos[2] = 0.5
        d.qpos[7:19] = d.ctrl[:] = np.clip(legs, self.lo, self.hi)
        mujoco.mj_forward(m, d)
        for _ in range(50):   # 1 s to drop and come to rest, the motors holding that pose
            mujoco.mj_step(m, d, nstep=self.frame_skip)
        self.kind = kind
        self.t, self.last_action = 0, np.zeros(12)
        self.stood_at = None
        self.best_up = self._rotation()[2, 2]
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(DOG + self.scale * a, self.lo, self.hi)   # the full motor range
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1
        self.env_steps += 1
        d, R = self.data, self._rotation()
        p = self.posture()
        up = R[2, 2]                                           # 1 upright, -1 on its back
        upright = max(up, 0.0)
        tall = np.exp(-((p["height"] - STAND_HEIGHT) / 0.08) ** 2)
        stance = np.exp(-np.sum((d.qpos[7:19] - DOG) ** 2) / 1.0)
        speed = np.hypot(p["vx"], p["vy"])
        reward = 1.0 * (up + 1) / 2                           # turning over
        reward += 5.0 * max(0.0, up - self.best_up)            # progress: each new best uprightness
        self.best_up = max(self.best_up, up)
        if self.goal == "selfright":   # belly down, legs tucked under, still: ready for the posture brain
            fold = np.exp(-np.sum((d.qpos[7:19] - LIE_POSE) ** 2) / 2.0)
            reward += 1.0 * fold * upright ** 2
            still = np.exp(-(np.linalg.norm(d.qvel[0:3]) / 0.2) ** 2) * np.exp(-(np.linalg.norm(d.qvel[3:6]) / 0.5) ** 2)
            reward += 0.5 * still * upright ** 2
            standing = up > 0.9 and np.linalg.norm(d.qvel[0:3]) < 0.2   # "righted"
        else:
            reward += 1.0 * tall * upright                         # standing up to its height
            reward += 0.5 * stance * upright ** 2                  # in the dog stance
            standing = up > 0.9 and p["height"] > 0.38
        if standing and self.goal != "selfright":
            reward += 0.5 * np.exp(-(speed / 0.3) ** 2) * np.exp(-(p["yaw_rate"] / 0.5) ** 2)   # and still
        if standing and self.stood_at is None:
            self.stood_at = self.t
        reward -= 0.01 * np.sum((a - self.last_action) ** 2) + 1e-5 * np.sum(d.actuator_force ** 2)
        self.last_action = a
        done = self.t >= self.max_steps
        info = dict(p, up=up, standing=standing, stood_at=self.stood_at, kind=self.kind)
        return self._obs(), float(reward), False, done, info


class GetUpPolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl"):
        from pathlib import Path
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: GetUpEnv()]))
        self.norm.training = False

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, episodes=10, seed=7000, kinds=("side", "back", "tipped", "standing"), goal="stand"):
    """From each kind of fall: standing (upright, at height) at the end of the 4 s, and the median time it
    took to first stand up."""
    out = {}
    env = GetUpEnv(randomize=True, goal=goal)
    for kind in kinds:
        wins, times = 0, []
        for k in range(episodes):
            env.exam_start = kind
            obs, _ = env.reset(seed=seed + k)
            info = {}
            for _ in range(env.max_steps):
                obs, _, _, done, info = env.step(policy.act(obs))
                if done:
                    break
            ok = bool(info["standing"])
            wins += ok
            if ok and info["stood_at"] is not None:
                times.append(info["stood_at"] * env.dt)
        out[f"up_{kind}"] = wins / episodes
        out[f"secs_{kind}"] = round(float(np.median(times)), 2) if times else None
    return out
