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

from bolt_env import ACTION_SCALE, LIE_POSE, STANCES, BoltEnv, physics

DOG = STANCES["dog"]
STAND_HEIGHT = 0.43


EXAM_LATENCY_MS = 0.0   # fixed control delay (ms) in exams (fair_exam.py --latency-ms)


class GetUpEnv(BoltEnv):
    OBS = 46

    def __init__(self, randomize=True, seed=None, max_steps=200, goal="stand", hard=False, motor_brake=None,
                 latency_ms=(0.0, 0.0)):
        super().__init__(randomize=randomize, max_steps=max_steps, seed=seed, gait_generator="dog", skills=["stand"],
                         motor_brake=motor_brake)
        # hard=True (run "recover", from 17e): tougher falls and physics, see HARD_KINDS and _hard_physics
        self.hard = hard
        self.shove_at, self.ice = None, False
        self.goal = goal   # "stand": get all the way up; "selfright": belly down, legs tucked, still
        # self-righting reaches each joint's whole range around the dog stance (walking's ACTION_SCALE
        # only reached +-0.4 rad sideways)
        self.scale = np.maximum(self.hi - DOG, DOG - self.lo) if goal == "selfright" else ACTION_SCALE
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.exam_start = None   # "side", "back", "tipped" or "standing": exams set it; training draws it
        # latency_ms=(lo, hi): the motors get the targets sent that long ago, drawn per try (as imitate_env and
        # jump_env; 29 Oct: at 20 ms MasterMind's recovery stood up 16/30). Exams: EXAM_LATENCY_MS (fair_exam --latency-ms)
        self.latency_ms = tuple(latency_ms)

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

    def _hard_physics(self, kind):
        """Run "recover": wider physics than the other brains (motor stiffness 75-125%, every body's mass
        90-110% and the torso's 80-130%, friction 40-140%), and a fifth of the tries on ice (all friction
        12-30% of normal: MuJoCo takes the larger friction of a contact pair, so the paws must be slippery too)."""
        m, r = self.model, self.rng
        kp = self.nominal_kp * r.uniform(0.75, 1.25)
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
        m.body_mass[:] = self.nominal_mass * r.uniform(0.9, 1.1, self.nominal_mass.shape)
        m.body_mass[self.torso] = self.nominal_mass[self.torso] * r.uniform(0.8, 1.3)
        self.ice = kind == "ice" or (self.exam_start is None and r.random() < 0.2)
        m.geom_friction[:, 0] = self.nominal_friction * (r.uniform(0.12, 0.3) if self.ice else r.uniform(0.4, 1.4))

    def _shove(self, speed, spin):
        """A kick: body velocity sideways (and a little forward/back) and a roll (and some pitch/yaw) rate."""
        r, d = self.rng, self.data
        R = self._rotation()
        s = r.choice([-1, 1])
        d.qvel[0:3] += R @ np.array([r.uniform(-0.4, 0.4) * speed, s * speed, r.uniform(0.0, 0.3) * speed])
        d.qvel[3:6] += np.array([-s * spin, r.uniform(-0.3, 0.3) * spin, r.uniform(-0.3, 0.3) * spin])

    def _hard_start(self, kind):
        """shoved / ice: standing in the dog stance, kicked (1.5-4.5 m/s, rolling 3-12 rad/s), the motors
        holding the stance until it has fallen (body tipped past 60 degrees, at most 1.5 s), as when MasterMind's
        fall detector hands over. dropped: a bad landing, from 0.55-1.0 m in any roll, pitched up to 45
        degrees, tumbling and drifting, under control from the start. Lying starts slide and spin half the time."""
        m, d, r = self.model, self.data, self.rng
        if kind in ("shoved", "ice"):
            d.qpos[2], d.qpos[7:19] = 0.45, np.clip(DOG + r.normal(0, 0.03, 12), self.lo, self.hi)
            d.ctrl[:] = d.qpos[7:19]
            mujoco.mj_forward(m, d)
            for _ in range(25):
                physics(m, d, self.frame_skip)
            self._shove(r.uniform(1.5, 4.5), r.uniform(3.0, 12.0))
            for _ in range(75):
                physics(m, d, self.frame_skip)
                if self._rotation()[2, 2] < 0.5:
                    break
            return
        if kind == "dropped":
            d.qpos[2] = r.uniform(0.55, 1.0)
            mujoco.mj_forward(m, d)
            d.qvel[0:2] = r.uniform(-1.5, 1.5, 2)
            d.qvel[3:6] = r.normal(0, 3.0, 3)
            return
        for _ in range(50):   # lying kinds: as before, 1 s to drop and come to rest
            physics(m, d, self.frame_skip)
        if r.random() < 0.5:   # still sliding and rolling when the brain takes over
            d.qvel[0:2] += r.uniform(-1.5, 1.5, 2)
            d.qvel[3:6] += r.normal(0, 2.0, 3)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.hard:
            kind = self.exam_start or self.rng.choice(HARD_KINDS, p=HARD_P)
            self._hard_physics(kind)
            self._place(kind)
            self._hard_start(kind)
            # a second kick while getting up, in a third of the training tries (never in exams)
            self.shove_at = int(self.rng.integers(25, 150)) if self.exam_start is None and self.rng.random() < 0.35 else None
            return self._begin(kind)
        if self.randomize:   # the same physics randomization as the other brains
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)
        kind = self.exam_start or self.rng.choice(["side", "back", "tipped", "standing"], p=[0.35, 0.3, 0.25, 0.1])
        self._place(kind)
        for _ in range(50):   # 1 s to drop and come to rest, the motors holding that pose
            physics(m, d, self.frame_skip)
        return self._begin(kind)

    def _place(self, kind):
        """Bolt in the start pose of this kind of fall (not yet simulated)."""
        m, d = self.model, self.data
        if kind in ("shoved", "ice"):
            roll, pitch = 0.0, 0.0
        elif kind == "dropped":
            roll, pitch = self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-0.8, 0.8)
        else:
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

    def _begin(self, kind):
        self.kind = kind
        self.t, self.last_action = 0, np.zeros(12)
        lat = EXAM_LATENCY_MS if self.exam_start is not None else (
            self.rng.uniform(*self.latency_ms) if self.latency_ms[1] > 0 else 0.0)
        self.start_delay(lat)
        self.stood_at = None
        self.best_up = self._rotation()[2, 2]
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(DOG + self.scale * a, self.lo, self.hi)   # the full motor range
        if self.shove_at is not None and self.t == self.shove_at:   # knocked again while getting up
            self._shove(self.rng.uniform(0.5, 2.0), self.rng.uniform(1.0, 5.0))
        self._physics_step()
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


# run "recover": the kinds of fall in training (hard=True) and their shares; "ice" is a shove on ice
HARD_KINDS = ["side", "back", "tipped", "standing", "shoved", "dropped", "ice"]
HARD_P = [0.15, 0.15, 0.12, 0.05, 0.23, 0.2, 0.1]
EXAM_KINDS = ("side", "back", "tipped", "shoved", "dropped", "ice")


class GetUpPolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl"):
        from pathlib import Path
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: GetUpEnv()]))
        self.norm.training = False

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, episodes=10, seed=7000, kinds=("side", "back", "tipped", "standing"), goal="stand", hard=False,
         motor_brake=None):
    """From each kind of fall: standing (upright, at height) at the end of the 4 s, and the median time it
    took to first stand up."""
    out = {}
    env = GetUpEnv(randomize=True, goal=goal, hard=hard, motor_brake=motor_brake)   # motor_brake None: process default
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


_POSTURE = None


def full_recovery(env, righter, obs):
    """The whole recovery MasterMind and the games use (learnwatch.recovery, skills.Bolt.recover): self-right
    until belly-down and still for 0.3 s (at most 4 s), tuck with the posture brain's "lie" (until every joint
    is within 0.6 rad of the lying pose, at most 2 s), then its "stand" for 3 s. Returns (stood at the end:
    upright and body above 38 cm, seconds from the hand-over to first standing like that or None)."""
    global _POSTURE
    from pathlib import Path
    from bolt_env import LIE_POSE
    from evaluate import Policy
    if _POSTURE is None:
        _POSTURE = Policy(Path(__file__).resolve().parent.parent / "models" / "bolt-dog-posture-getup-3M")
    env.max_steps = 10 ** 6
    t, first, calm = 0, None, 0

    def check():
        nonlocal first
        p = env.posture()
        if first is None and p["up"] > 0.9 and p["height"] > 0.38:
            first = t * env.dt
    for _ in range(200):
        obs, _, _, _, info = env.step(righter.act(obs))
        t += 1
        calm = calm + 1 if info["standing"] else 0
        if calm >= 15:
            break
    for cmd, steps in (("lie", 100), ("stand", 150)):
        env.forced_command = cmd   # forced, as skills.Bolt does: BoltEnv.step never draws a new command
        env.set_command(cmd)
        env.last_action = np.zeros(12)
        for _ in range(steps):
            BoltEnv.step(env, _POSTURE.act(BoltEnv._obs(env)))
            t += 1
            if cmd == "stand":
                check()
            if cmd == "lie" and np.abs(env.data.qpos[7:19] - LIE_POSE).max() < 0.6:
                break
    p = env.posture()
    ok = bool(p["up"] > 0.9 and p["height"] > 0.38)
    return ok, (first if ok else None)


def exam_full(righter, episodes=10, seed=7000, kinds=EXAM_KINDS, hard=True, motor_brake=None):
    """Fair full-recovery exam: from each kind of fall (hard=True: run "recover"'s tougher falls and physics,
    the same fixed seeds for every brain, no second kick), stood_<kind>: share of tries standing at the end;
    tsec_<kind>: median seconds from the hand-over to first standing (self-right + tuck + stand)."""
    out = {}
    env = GetUpEnv(randomize=True, goal="selfright", hard=hard, motor_brake=motor_brake)   # motor_brake None: process default
    for kind in kinds:
        wins, times = 0, []
        for k in range(episodes):
            env.exam_start = kind
            env.max_steps = 200
            obs, _ = env.reset(seed=seed + k)
            ok, sec = full_recovery(env, righter, obs)
            wins += ok
            if sec is not None:
                times.append(sec)
        out[f"stood_{kind}"] = wins / episodes
        out[f"tsec_{kind}"] = round(float(np.median(times)), 2) if times else None
    return out
