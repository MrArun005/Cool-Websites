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

Landing starts (landing_rate > 0, step 1 of the skill-switch fixes): that share of the tries starts instead in a body
state of the run-up jump brain's landing, from mid-air through touch-down (research/gen_landing_starts.py ->
research/results/landing_starts.npz, at any heading), tracking the clip the skill manager hands over to at the
landing (canter, at its closest-pose moment). For landing_grace s the try is not ended for being off the
reference height (Bolt is still in the air, up to ~35 cm above the canter's body height; the 0.8 rad tilt limit and
the body-on-floor check stay on), and after landing_bridge s the clip switches to landing_then (the jump ->
(canter) -> trot hand-over of test_skills.py). landing_rate 0 (the default) draws nothing extra: tries are exactly
as before.

Posture commands while moving (moving_rate > 0, step 2): that share of the tries starts in a gait (Bolt in the dog's
walk or trot at a random moment, its body moving at a speed drawn from moving_speed, from the gait in moving_gaits whose
clip pace is closest) and at once is told to sit, lie down or get up (moving_clips): the clip switches to the posture
clip at its closest-pose moment, as the skill manager does (closest_moment, the same rule as skills.closest_moment).
Reaching the clip's end pose (pose_reached.py: the one definition the switch test uses) within pose_time s of the
command pays pose_weight every step the pose is held from then on (while that clip is still the one being tracked).
The try runs at least to the time limit. moving_rate 0 (the default) draws nothing extra.
exam() reports pose reached for sit and lie from standing still (the clip's start) and from a moving start in each
of moving_gaits at that gait's own pace.

motor_brake: None = bolt_env's process default (off unless set; train_imitate.py turns it on), True/False for this
env (bolt_env: back-EMF braking above the drive cutoff).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch  # noqa: F401  (imported by the policies below; keeps the import order of the other envs)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import ACTION_SCALE, STANCES, BoltEnv, physics
from pose_reached import CLIP_POSE, pose_ok

DOG = STANCES["dog"]
CLIP_DIR = Path(__file__).with_name("dogmotion") / "clips"
ALL_CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"]   # getup: lie reversed (run 18d)
DEFAULT_CLIPS = ["walk", "trot", "pace", "turn_left", "turn_right", "sit", "lie"]   # canter (3.4 m/s) later
WEIGHTS = dict(pose=0.5, vel=0.05, paw=0.2, root=0.15, root_vel=0.1)
HOLD = 1.0          # s: sit and lie hold their final pose this long
# Switching skills mid-try (run 18e): which clip may follow which. Gaits switch to any gait (joining at the frame
# whose pose is closest) or to sit (the sit clip starts standing, so the brain learns to stop first); a finished
# sit goes on to lie, a finished lie to getting up, a finished getup to walking or trotting.
GAITS = ("walk", "trot", "pace", "canter", "turn_left", "turn_right")
NEXT = {**{g: GAITS + ("sit",) for g in GAITS}, "sit": ("lie",), "lie": ("getup",), "getup": ("walk", "trot")}
LOOP_SECONDS = 5.0  # s: longest try on a looping gait
LANDING_FILE = Path(__file__).with_name("research") / "results" / "landing_starts.npz"   # gen_landing_starts.py
LANDING = dict(rate=0.0, grace=0.5, bridge=0.6, then="trot")   # defaults of the landing-start arguments (see above)
# defaults of the moving-start posture arguments (see above). speed: m/s range (walk clip 1.1, trot 1.9); pose_time: s
# from the command to reach the pose, the switch test's time for each (test_skills.seq_rest: sit 4 s, lie 5.5 s; getup
# as lie, the clip is lie reversed); weight: paid per step in the pose (the tracking terms pay up to 1 per step)
MOVING = dict(rate=0.0, speed=(0.5, 2.0), gaits=("walk", "trot"), clips=("sit", "lie", "getup"),
              pose_time={"sit": 4.0, "lie": 5.5, "getup": 5.5}, weight=0.2)


def closest_moment(clip, q, qd):
    """Clip time (s) whose joint pose is closest to q, joint speeds weighing a little: skills.closest_moment's rule
    (the skill manager's hand-over), so a moving-start command joins the clip where the switch test's does."""
    e = np.sum((clip.joints - q) ** 2, axis=1) + 0.002 * np.sum((clip.joint_vel - qd) ** 2, axis=1)
    return float(np.argmin(e)) / clip.fps


def load_landing(path=LANDING_FILE):
    """The landing states (gen_landing_starts.py): qpos (N, 19) and qvel (N, 18) in their heading frame, the clip
    (ALL_CLIPS index) and clip time to track from each."""
    z = np.load(path)
    return dict(qpos=z["qpos"], qvel=z["qvel"], clip=z["clip"].astype(int), tau0=z["tau0"])


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

    def __init__(self, randomize=True, seed=None, clips=None, residual=True, weights=None, slots=len(ALL_CLIPS), switch_rate=0.0,
                 landing_rate=LANDING["rate"], landing_file=LANDING_FILE, landing_grace=LANDING["grace"],
                 landing_bridge=LANDING["bridge"], landing_then=LANDING["then"], moving_rate=MOVING["rate"],
                 moving_speed=MOVING["speed"], moving_gaits=MOVING["gaits"], moving_clips=MOVING["clips"],
                 pose_time=None, pose_weight=MOVING["weight"], motor_brake=None):
        super().__init__(randomize=randomize, max_steps=int(LOOP_SECONDS * 50), seed=seed, gait_generator="dog",
                         skills=["stand"], motor_brake=motor_brake)
        self.slots = slots   # clip one-hot size: 8 for brains from before the getup clip (74 inputs)
        self.switch_rate = switch_rate   # chance per second of switching to a following clip mid-try (training only)
        self.OBS = 66 + slots
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.clip_names = list(clips or DEFAULT_CLIPS)
        self.clips = {n: Clip(n) for n in self.clip_names}
        self.residual = residual
        self.weights = dict(WEIGHTS, **(weights or {}))   # e.g. {"root_vel": 0.25} if it keeps poses but not pace
        self.exam_start = None        # (clip name, start time s): exams set it; training draws it
        self.body_geoms = {g for g in range(self.model.ngeom) if self.geom_leg[g] < 0 and g != self.floor}
        self.clip, self.tau0 = self.clips[self.clip_names[0]], 0.0
        # landing starts (module docstring): share of tries, the states, s without the height check, s of the bridge
        # clip before landing_then (None or 0 s: no forced switch)
        self.landing_rate, self.landing_grace, self.landing_bridge, self.landing_then = landing_rate, landing_grace, landing_bridge, landing_then
        self.landing = load_landing(landing_file) if landing_rate > 0 else None
        if self.landing is not None:
            need = {ALL_CLIPS[c] for c in np.unique(self.landing["clip"])} | ({landing_then} if landing_then and landing_bridge else set())
            assert need <= set(self.clip_names), f"landing starts need the clips {sorted(need)} (env has {self.clip_names})"
        self.exam_landing = None      # index of a landing state to start from (tests, exams); training draws it
        self.grace_steps, self.land_switch_at = 0, -1
        # moving-start posture commands (module docstring); the gaits and posture clips this env has
        self.moving_rate, self.moving_speed, self.pose_weight = moving_rate, tuple(moving_speed), pose_weight
        self.pose_time = dict(MOVING["pose_time"], **(pose_time or {}))
        self.moving_gaits = [g for g in moving_gaits if g in self.clips and self.clips[g].loop]
        self.moving_clips = [c for c in moving_clips if c in self.clips and c in CLIP_POSE]
        if moving_rate > 0:
            assert self.moving_gaits and self.moving_clips, f"moving starts need gaits {moving_gaits} and clips {moving_clips}"
        self.exam_moving = None       # (posture clip, gait, gait moment s, speed m/s) for tests and exams
        self.pose_clip, self.pose_until, self.pose_hit = None, 0, False

    def gait_speed(self, name):
        """The clip's own pace: mean forward body speed (m/s)."""
        return float(self.clips[name].v_body[:, 0].mean())

    def pose_now(self, name):
        """Is the body in pose `name` (pose_reached.POSE_OK) right now?"""
        R = self._rotation()
        return bool(pose_ok(name, self.data.qpos[2], np.arcsin(np.clip(R[2, 0], -1, 1)), R[2, 2]))

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
        self.pose_clip, self.pose_until, self.pose_hit = None, 0, False
        land = self.exam_landing
        if land is None and self.landing_rate and not self.exam_start and self.rng.random() < self.landing_rate:
            land = int(self.rng.integers(len(self.landing["tau0"])))
        if land is not None:
            return self._landing_start(land)
        self.grace_steps, self.land_switch_at = 0, -1
        mv = self.exam_moving
        if mv is None and self.moving_rate and not self.exam_start and self.rng.random() < self.moving_rate:
            post = self.moving_clips[self.rng.integers(len(self.moving_clips))]
            v = float(self.rng.uniform(*self.moving_speed))
            gait = min(self.moving_gaits, key=lambda g: abs(self.gait_speed(g) - v))   # the gait at about that pace
            mv = (post, gait, float(self.rng.uniform(0, self.clips[gait].length)), v)
        if mv is not None:
            return self._moving_start(*mv)
        if self.exam_start:
            name, self.tau0 = self.exam_start
        else:
            name = self.clip_names[self.rng.integers(len(self.clip_names))]
            c = self.clips[name]
            # anywhere in a loop; sit/lie: anywhere in the clip or its hold (the hold gets ~1/4 of starts)
            self.tau0 = float(self.rng.uniform(0, c.length if c.loop else c.length + 0.3 * HOLD))
        self.clip = self.clips[name]
        self.t = 0
        self._rsi()
        return self._settle()

    def _rsi(self, vscale=1.0):
        """Reference state initialization: Bolt starts in the dog's pose at that moment, moving as it was (the body's
        linear velocity times vscale: moving starts)."""
        m, d = self.model, self.data
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
        if vscale != 1.0:
            d.qvel[0:3] *= vscale
        d.ctrl[:] = d.qpos[7:19]
        mujoco.mj_forward(m, d)

    def _moving_start(self, post, gait, tau_g, v):
        """Bolt in `gait` at moment tau_g, its body at v m/s, told now to do posture clip `post`: that clip from its
        closest-pose moment, the pose to reach within pose_time s."""
        self.clip, self.tau0, self.t = self.clips[gait], float(tau_g), 0
        self._rsi(vscale=v / self.gait_speed(gait))
        c = self.clips[post]
        self.clip, self.tau0 = c, closest_moment(c, self.data.qpos[7:19], self.data.qvel[6:18])
        self.pose_clip, self.pose_until, self.pose_hit = post, int(round(self.pose_time[post] / self.dt)), False
        obs, info = self._settle()
        self.limit = max(self.limit, self.pose_until + 1)   # the try lasts at least to the time limit
        return obs, info

    def _settle(self, floor_only=False):
        m, d = self.model, self.data
        # never start with a paw or the body in the floor: lift by the deepest penetration (landing starts: of the floor
        # contacts only; a recorded state's legs may touch each other, which lifting does not change)
        pen = max([0.0] + [-c.dist for c in d.contact[:d.ncon] if not floor_only or self.floor in (c.geom1, c.geom2)])
        if pen > 0:
            d.qpos[2] += pen + 0.002
            mujoco.mj_forward(m, d)
        self.last_action = np.zeros(12)
        steps_left = (self.clip.length + HOLD - self.tau0) / self.dt if not self.clip.loop else self.max_steps
        self.limit = int(max(25, min(self.max_steps, steps_left)))
        return self._obs(), {}

    def _landing_start(self, i):
        """Start in landing state i (its heading frame turned to a random heading), tracking its clip from its moment."""
        m, d, L = self.model, self.data, self.landing
        self.clip, self.tau0, self.t = self.clips[ALL_CLIPS[L["clip"][i]]], float(L["tau0"][i]), 0
        yaw = self.rng.uniform(-np.pi, np.pi)
        R0 = _mat(L["qpos"][i][3:7])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (_rotz(yaw) @ R0).reshape(-1))
        d.qpos[:3] = [0.0, 0.0, L["qpos"][i][2]]
        d.qpos[3:7] = q
        d.qpos[7:19] = np.clip(L["qpos"][i][7:19], self.lo, self.hi)
        d.qvel[0:3] = _rotz(yaw) @ L["qvel"][i][0:3]
        d.qvel[3:18] = L["qvel"][i][3:18]
        d.ctrl[:] = d.qpos[7:19]
        mujoco.mj_forward(m, d)
        self.grace_steps = int(round(self.landing_grace / self.dt))
        bridge = self.landing_then and self.landing_bridge
        self.land_switch_at = int(round(self.landing_bridge / self.dt)) if bridge else -1
        return self._settle(floor_only=True)

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
        if self.switch_rate and not self.exam_start:
            self._maybe_switch()
        if self.t == self.land_switch_at and self.clip.name != self.landing_then:   # a landing start's bridge is over:
            self._switch_to(self.clips[self.landing_then])                         # on in landing_then (test_skills' trot)
        terms, err = self.tracking()
        reward = sum(self.weights[k] * v for k, v in terms.items())
        pose = None
        if self.pose_clip is not None:   # a moving-start command: paid per step in the pose once reached in time
            pose = self.clip.name == self.pose_clip and self.pose_now(CLIP_POSE[self.pose_clip])
            self.pose_hit = self.pose_hit or (pose and self.t <= self.pose_until)
            if self.pose_hit and pose:
                reward += self.pose_weight
        self.last_action = a
        floor_ok = self.ref("height") < 0.3     # sitting and lying: the body may rest on the floor
        lost = (abs(err["height_err"]) > 0.15 and self.t > self.grace_steps) or err["tilt"] > 0.8   # grace: landing starts
        fell = lost or (self.body_on_floor() and not floor_ok)
        done = self.t >= self.limit
        info = dict(err, clip=self.clip.name, tau=self.tau(), fell=fell, **{f"r_{k}": float(v) for k, v in terms.items()})
        if pose is not None:
            info.update(pose_ok=bool(pose), pose_hit=bool(self.pose_hit))
        return self._obs(), float(reward), bool(fell), bool(done and not fell), info


    def _maybe_switch(self):
        """Now and then hand over to a clip that may follow this one (NEXT), as a dog changes what it does."""
        ready = self.clip.loop or self.finished() > 0.5 * HOLD
        nxt = [n for n in NEXT.get(self.clip.name, ()) if n in self.clips and n != self.clip.name]
        if not ready or not nxt or self.rng.random() > self.switch_rate * self.dt:
            return
        self._switch_to(self.clips[self.rng.choice(nxt)])

    def _switch_to(self, c):
        """Hand over to clip c now."""
        if c.loop:   # join the new gait at the frame whose pose is closest to the current one
            k = int(np.argmin(np.sum((c.joints - self.data.qpos[7:19]) ** 2, 1)))
            start = k / c.fps
        else:        # sit, lie, getup play from their start
            start = 0.0
        self.clip, self.tau0 = c, start - self.t * self.dt
        left = (c.length + HOLD - start) / self.dt if not c.loop else self.max_steps
        self.limit = int(min(self.t + left, self.t + self.max_steps))


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
    pace is part of the score (score()).
    Posture clips (sit, lie): pose_still_<clip> is the share of tries from standing still (the clip's start) that
    reached the clip's end pose (pose_reached.py, the switch test's definition) within the time limit (pose_time) and
    did not fall; pose_moving_<gait>_<clip> (and fall_moving_<gait>_<clip>) the same from a moving start: Bolt in that
    gait at its own pace, told to sit or lie at once (env.exam_moving). Neither enters score()."""
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
    env.exam_start = None
    for name in [n for n in env.clip_names if CLIP_POSE.get(n) in ("sit", "lie")]:
        c = env.clips[name]
        starts = [("still", None)] + [(g, g) for g in env.moving_gaits]
        for tag, gait in starts:
            hits, falls = 0, 0
            for k in range(episodes):
                if gait is None:
                    env.exam_start, env.exam_moving = (name, 0.0), None
                else:
                    env.exam_start, env.exam_moving = None, (name, gait, (k / episodes) * env.clips[gait].length, env.gait_speed(gait))
                obs, _ = env.reset(seed=seed + k)
                until = int(round(env.pose_time[name] / env.dt))
                env.limit = max(env.limit, until + 1)
                hit, fell = False, False
                for _ in range(env.limit):
                    obs, r, fell, done, info = env.step(policy.act(obs))
                    hit = hit or (env.t <= until and env.clip is c and env.pose_now(CLIP_POSE[name]))
                    if fell or done:
                        break
                hits += hit and not fell
                falls += fell
            key = name if gait is None else f"{gait}_{name}"
            out[f"pose_{'still' if gait is None else 'moving'}_{key}"] = hits / episodes
            if gait is not None:
                out[f"fall_moving_{key}"] = falls / episodes
    env.exam_start = env.exam_moving = None
    return out


def score(result):
    """One number for keeping the best brain: mean over clips of tracking x (gaits) the share of the dog's
    pace kept, 0..1."""
    names = [k[6:] for k in result if k.startswith("track_")]
    return float(np.mean([result[f"track_{n}"] * float(np.clip(result.get(f"speed_{n}", 1.0), 0.0, 1.0))
                          for n in names]))
