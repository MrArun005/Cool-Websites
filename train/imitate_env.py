"""Teach Bolt to move like a real dog by imitating real dog motion capture.

The reference clips come from a real dog in a motion-capture suit (dogmotion/SOURCES.md), retargeted onto
Bolt's body (dogmotion/retarget.py, 50 Hz .npz files in dogmotion/clips/). This is the approach of Peng et
al. 2020 ("Learning Agile Robotic Locomotion Skills by Imitating Animals") and DeepMimic (Peng et al.
2018): every 1/50 s Bolt is paid for being in the pose the dog was in at that moment of the clip.

    python train_imitate.py --name imitate18 --steps 30000000

Each try: a clip is drawn (walk, trot, pace, turn left/right, sit down, lie down), and Bolt starts IN the
reference pose at a random moment of it, moving as the dog was ("reference state initialization": every
part of the motion gets practice from the start, not only the beginning). Looping gaits run for up to
5 s; sit and lie play once and then hold the final pose for 1 s. The try ends early when Bolt has
clearly lost the motion (body more than 15 cm off the reference height, or tilted 0.8 rad away from it)
or its body touches the floor while the dog's did not.

Reward (DeepMimic / Peng 2020 weights), every term in 0..1, so every step alive pays something and a fall
(which ends the try) is never a way out:
    0.50 joint pose        exp(-5   * sum (q_ref - q)^2)              12 joint angles
    0.05 joint velocity    exp(-0.1 * sum (qd_ref - qd)^2)
    0.20 paws              exp(-40  * sum |p_ref - p|^2)              paw centres relative to the body,
                                                                       in its heading frame (yaw removed)
    0.15 body pose         exp(-20 * (h_ref - h)^2 - 10 * tilt^2)     height, and the angle between the
                                                                       reference and actual "down" vectors
    0.10 body velocity     exp(-2 * |v_ref - v|^2 - 0.2 * (w_ref - w)^2)   body-frame velocity, spin rates
Everything is tracked relative to the body (not absolute world position), so a small lag in a long
walk does not pile up into an error that can never be recovered.

Motors: ctrl = clip(q_ref(next step) + ACTION_SCALE * action). The reference pose is played as a feed-
forward plan and the brain adds corrections, with the full ACTION_SCALE range (as in the jump and
get-up brains), so it can still do whatever balance needs. residual=False gives the plain full-range
control of jump_env/getup_env, centred on the dog stance, instead.

Observation (OBS = 75 floats; 74 for brains from before the getup clip), in this order:
     0  gravity in the body frame            3
     3  angular velocity x 0.25 (body)       3
     6  linear velocity (body frame)         3
     9  body height (m)                      1
    10  joint angles - dog stance           12
    22  joint velocities x 0.1              12
    34  previous action                     12
    46  clip phase sin, cos (2 pi phase)     2   loops: position in the stride; sit/lie: progress 0..1
                                                 (held at 1 once the clip has finished)
    48  clip one-hot                         9   ALL_CLIPS order (fixed, so brains stay swappable; 8 before getup)
    57  reference joints next step - stance 12
    69  reference body height next step      1
    70  reference body velocity (body frame) 3   next step
    73  reference spin rate (yaw)            1   next step
    74  time since clip end (s, 0 while it plays; sit/lie/getup hold)  1
Physics randomization (motor strength, body mass, floor friction) and sensor noise as in the other envs.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch  # noqa: F401  (imported by the policies below; keeps the import order of the other envs)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import ACTION_SCALE, STANCES, BoltEnv, physics

DOG = STANCES["dog"]
CLIP_DIR = Path(__file__).with_name("dogmotion") / "clips"
ALL_CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"]   # getup: lie reversed (run 18d)
DEFAULT_CLIPS = ["walk", "trot", "pace", "turn_left", "turn_right", "sit", "lie"]   # canter (3.4 m/s) later
WEIGHTS = dict(pose=0.5, vel=0.05, paw=0.2, root=0.15, root_vel=0.1)
HOLD = 1.0          # s: sit and lie hold their final pose this long
LOOP_SECONDS = 5.0  # s: longest try on a looping gait


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


class Clip:
    """A retargeted clip, as body-relative quantities per frame (what the reward compares)."""

    def __init__(self, name):
        z = np.load(CLIP_DIR / f"{name}.npz")
        self.name, self.loop, self.fps = name, bool(z["loop"]), float(z["fps"])
        self.n = len(z["joints"])
        self.length = self.n / self.fps
        self.joints, self.joint_vel = z["joints"], z["joint_vel"]
        self.height = z["base_pos"][:, 2]
        self.quat = z["base_quat"]
        R = np.array([_mat(q) for q in self.quat])
        self.gravity = np.einsum("tji,j->ti", R, [0, 0, -1.0])            # R^T (0,0,-1)
        self.v_body = np.einsum("tji,tj->ti", R, z["base_linvel"])
        self.w_body = np.einsum("tji,tj->ti", R, z["base_angvel"])
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        rel = z["foot_pos"] - z["base_pos"][:, None]
        self.paws = np.stack([rel[t] @ _rotz(yaw[t]) for t in range(self.n)])   # Rz(-yaw) applied to rows
        self.tilt = np.stack([_rotz(-yaw[t]) @ R[t] for t in range(self.n)])     # body rotation without yaw
        self.lowest = self.height.min()

    def frame(self, tau):
        """(index, next index, blend) for time tau (s); loops wrap, others hold the last frame."""
        u = tau * self.fps
        if self.loop:
            u %= self.n
            i = int(u)
            return i, (i + 1) % self.n, u - i
        u = min(max(u, 0.0), self.n - 1)
        i = int(u)
        return i, min(i + 1, self.n - 1), u - i

    def at(self, tau, key):
        i, j, b = self.frame(tau)
        a = getattr(self, key)
        return (1 - b) * a[i] + b * a[j]

    def phase(self, tau):
        return (tau / self.length) % 1.0 if self.loop else min(tau / self.length, 1.0)


class ImitateEnv(BoltEnv):
    OBS = 66 + len(ALL_CLIPS)

    def __init__(self, randomize=True, seed=None, clips=None, residual=True, weights=None, slots=len(ALL_CLIPS)):
        super().__init__(randomize=randomize, max_steps=int(LOOP_SECONDS * 50), seed=seed, gait_generator="dog",
                         skills=["stand"])
        self.slots = slots   # clip one-hot size: 8 for brains from before the getup clip (74 inputs)
        self.OBS = 66 + slots
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.clip_names = list(clips or DEFAULT_CLIPS)
        self.clips = {n: Clip(n) for n in self.clip_names}
        self.residual = residual
        self.weights = dict(WEIGHTS, **(weights or {}))   # e.g. {"root_vel": 0.25} if it keeps poses but not pace
        self.exam_start = None        # (clip name, start time s): exams set it; training draws it
        self.body_geoms = {g for g in range(self.model.ngeom) if self.geom_leg[g] < 0 and g != self.floor}
        self.clip, self.tau0 = self.clips[self.clip_names[0]], 0.0

    # ---------- reference ----------
    def tau(self, ahead=0):
        return self.tau0 + (self.t + ahead) * self.dt

    def ref(self, key, ahead=0):
        return self.clip.at(self.tau(ahead), key)

    def heading_paws(self):
        R = self._rotation()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        rel = np.array([self.data.geom_xpos[g] for g in self.foot_ids]) - self.data.qpos[:3]
        return rel @ _rotz(yaw)

    def body_on_floor(self):
        d = self.data
        return any((c.geom1 == self.floor and c.geom2 in self.body_geoms) or
                   (c.geom2 == self.floor and c.geom1 in self.body_geoms) for c in d.contact[:d.ncon])

    def finished(self):
        """Seconds since a sit/lie clip ended (0 while it plays, and always 0 for loops)."""
        return 0.0 if self.clip.loop else max(0.0, self.tau() - self.clip.length)

    # ---------- gym API ----------
    def _obs(self):
        d, R = self.data, self._rotation()
        ph = 2 * np.pi * self.clip.phase(self.tau())
        onehot = np.zeros(self.slots)
        onehot[ALL_CLIPS.index(self.clip.name)] = 1.0
        obs = np.concatenate([
            R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3], [d.qpos[2]],
            d.qpos[7:19] - DOG, d.qvel[6:18] * 0.1, self.last_action,
            [np.sin(ph), np.cos(ph)], onehot,
            self.ref("joints", 1) - DOG, [self.ref("height", 1)], self.ref("v_body", 1), [self.ref("w_body", 1)[2]],
            [self.finished()],
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
        if self.randomize:   # the same physics randomization as the other brains
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)
        if self.exam_start:
            name, self.tau0 = self.exam_start
        else:
            name = self.clip_names[self.rng.integers(len(self.clip_names))]
            c = self.clips[name]
            # anywhere in a loop; sit/lie: anywhere in the clip or its hold (the hold gets ~1/4 of starts)
            self.tau0 = float(self.rng.uniform(0, c.length if c.loop else c.length + 0.3 * HOLD))
        self.clip = self.clips[name]
        self.t = 0
        # reference state initialization: Bolt starts in the dog's pose at that moment, moving as it was
        yaw = self.rng.uniform(-np.pi, np.pi)
        tilt = self.ref("tilt")
        u, s, vt = np.linalg.svd(tilt)          # re-orthonormalize the blended rotation
        Rw = _rotz(yaw) @ (u @ vt)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rw.reshape(-1))
        d.qpos[:3] = [0.0, 0.0, self.ref("height") + 0.005]
        d.qpos[3:7] = q
        d.qpos[7:19] = np.clip(self.ref("joints"), self.lo, self.hi)
        moving = 0.0 if self.finished() > 0 else 1.0
        d.qvel[0:3] = Rw @ self.ref("v_body") * moving
        d.qvel[3:6] = self.ref("w_body") * moving
        d.qvel[6:18] = self.ref("joint_vel") * moving
        d.ctrl[:] = d.qpos[7:19]
        mujoco.mj_forward(m, d)
        # never start with a paw or the body in the floor: lift by the deepest penetration
        pen = max([0.0] + [-c.dist for c in d.contact[:d.ncon]])
        if pen > 0:
            d.qpos[2] += pen + 0.002
            mujoco.mj_forward(m, d)
        self.last_action = np.zeros(12)
        steps_left = (self.clip.length + HOLD - self.tau0) / self.dt if not self.clip.loop else self.max_steps
        self.limit = int(max(25, min(self.max_steps, steps_left)))
        return self._obs(), {}

    def tracking(self):
        """The five tracking terms (each 0..1) and the raw errors used for early termination."""
        d, R = self.data, self._rotation()
        q, qd = d.qpos[7:19], d.qvel[6:18]
        e_pose = np.sum((self.ref("joints") - q) ** 2)
        e_vel = np.sum((self.ref("joint_vel") - qd) ** 2)
        e_paw = np.sum((self.ref("paws") - self.heading_paws()) ** 2)
        e_h = self.ref("height") - d.qpos[2]
        g, g_ref = R.T @ np.array([0, 0, -1.0]), self.ref("gravity")
        tilt = np.arccos(np.clip(np.dot(g, g_ref) / np.linalg.norm(g_ref), -1, 1))
        v, w = R.T @ d.qvel[0:3], d.qvel[3:6]   # MuJoCo free joint: angular velocity is in the body frame
        e_v = np.sum((self.ref("v_body") - v) ** 2)
        e_w = np.sum((self.ref("w_body") - w) ** 2)
        terms = dict(pose=np.exp(-5 * e_pose), vel=np.exp(-0.1 * e_vel), paw=np.exp(-40 * e_paw),
                     root=np.exp(-20 * e_h ** 2 - 10 * tilt ** 2), root_vel=np.exp(-2 * e_v - 0.2 * e_w))
        return terms, dict(height_err=float(e_h), tilt=float(tilt), pose_rms=float(np.sqrt(e_pose / 12)))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        base = self.ref("joints", 1) if self.residual else DOG
        self.data.ctrl[:] = np.clip(base + ACTION_SCALE * a, self.lo, self.hi)
        physics(self.model, self.data, self.frame_skip)
        self.t += 1
        self.env_steps += 1
        terms, err = self.tracking()
        reward = sum(self.weights[k] * v for k, v in terms.items())
        self.last_action = a
        floor_ok = self.ref("height") < 0.3     # sitting and lying: the body may rest on the floor
        lost = abs(err["height_err"]) > 0.15 or err["tilt"] > 0.8
        fell = lost or (self.body_on_floor() and not floor_ok)
        done = self.t >= self.limit
        info = dict(err, clip=self.clip.name, tau=self.tau(), fell=fell, **{f"r_{k}": float(v) for k, v in terms.items()})
        return self._obs(), float(reward), bool(fell), bool(done and not fell), info


class ImitatePolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl"):
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.slots = self.model.observation_space.shape[0] - 66   # 8: a brain from before the getup clip
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: ImitateEnv(slots=self.slots)]))
        self.norm.training = False

    def act(self, obs):
        if len(obs) > 66 + self.slots:   # an older brain: drop the clip slots it never had (they are 0)
            obs = np.concatenate([obs[:48 + self.slots], obs[48 + len(ALL_CLIPS):]])
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, clips=None, episodes=4, seed=9000):
    """Per clip: the mean tracking reward per step over the whole try (0 counted for every step lost after a
    fall), the fall rate, and for gaits how much of the dog's pace Bolt keeps (speed_: forward distance
    over the dog's; for turns, the angle turned over the dog's). Loops run 4 s from evenly spaced points in
    the stride; sit and lie play from the start through their 1 s hold. Physics is randomized as in training.
    A brain can hold the poses while barely moving (the smoke test at 300k: 1/4 of the walk's speed), so the
    pace is part of the score (score())."""
    env = ImitateEnv(randomize=True, clips=clips)
    out = {}
    for name in env.clip_names:
        c = env.clips[name]
        turn = name.startswith("turn")
        scores, falls, pace = [], 0, []
        for k in range(episodes):
            env.exam_start = (name, (k / episodes) * c.length if c.loop else 0.0)
            obs, _ = env.reset(seed=seed + k)
            if c.loop:
                env.limit = 200
            total, fell, done_by, yaw = 0.0, False, 0.0, 0.0
            for _ in range(env.limit):
                R = env._rotation()
                p0, h0 = env.data.qpos[:2].copy(), np.arctan2(R[1, 0], R[0, 0])
                obs, r, fell, done, info = env.step(policy.act(obs))
                R = env._rotation()
                done_by += float(np.dot(env.data.qpos[:2] - p0, [np.cos(h0), np.sin(h0)]))
                yaw += float(np.angle(np.exp(1j * (np.arctan2(R[1, 0], R[0, 0]) - h0))))
                total += r
                if fell or done:
                    break
            falls += fell
            scores.append(total / env.limit)
            if c.loop:
                want = env.limit * env.dt * (c.w_body[:, 2].mean() if turn else c.v_body[:, 0].mean())
                pace.append((yaw if turn else done_by) / want)
        out[f"track_{name}"] = round(float(np.mean(scores)), 3)
        out[f"fall_{name}"] = falls / episodes
        if c.loop:
            out[f"speed_{name}"] = round(float(np.mean(pace)), 2)
    return out


def score(result):
    """One number for keeping the best brain: mean over clips of tracking x (gaits) the share of the dog's
    pace kept, 0..1."""
    names = [k[6:] for k in result if k.startswith("track_")]
    return float(np.mean([result[f"track_{n}"] * float(np.clip(result.get(f"speed_{n}", 1.0), 0.0, 1.0))
                          for n in names]))
