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

Transition term (transition_weight > 0, step 3): for transition_window s after the clip switches mid-try (a switch
drawn by switch_rate, or a landing start's bridge ending), every step pays -transition_weight x (how far the body's
roll beyond transition_threshold + how far its pitch beyond it), roll and pitch measured against the reference's own
(rad; the gaits' are near 0, the sit clip's pitch is +0.6, so a dog sitting is not charged for sitting). The canter ->
trot falls of skills_report.json go over sideways (roll ~1 rad) 0.5-0.8 s after the switch. The five tracking terms
are unchanged. While 0.8 rad of tilt ends the try, each excess stays under 0.5 rad, so with a weight up to ~0.8 a step
still pays more than falling. transition_weight 0 (the default) computes nothing.

Real-robot gaps (step 3b; mastermind/REPORT.md "Sim-to-real stress test": control latency is the biggest one, its
clean switches 83% -> 67% at 10 ms -> 50% at 20 ms -> 18% at 40 ms), all off by default, the same models as
mastermind/env.py's stress test:
    latency_ms=(lo, hi)  the motors get the targets the brain sent that long ago, drawn per try uniform in [lo, hi] ms,
                         at the 500 Hz physics rate (whole control steps from a queue, the rest of the delay as the
                         first substeps of a step still on the older target; the robot holds its first targets)
    obs_noise=k          sensor noise, uniform in +-k x NOISE (legged_gym's levels, as mastermind.env.NOISE) on gravity,
                         gyro, body velocity, height, joint angles and speeds, on top of the 0.01 noise of randomize
    motor_lag_ms=tau     first-order lag of the joint targets, time constant tau ms, every physics substep
The "previous action" input stays what the brain sent (as on a robot). exam() adds a latency sweep: lat<ms>_score
and lat<ms>_fall at 0/10/20/40 ms (LATENCY_SWEEP).

Rough terrain and pushes (step 6), off by default:
    terrain_cm      bolt_env's height field (TERRAIN: uniform random heights up to terrain_cm per 10 cm cell, 40 x 40 m
                    beside the flat floor). Tries starting in terrain_clips (walk, trot) start on it, at a random spot
                    within terrain_spread m of its middle; every other try starts on the flat floor at the origin, as
                    before. On terrain the body height (input 9, the height term and the 15 cm limit) is measured from
                    the ground under the body (bolt_env.terrain_height), as on flat floor. Only the paws and shins touch
                    the bumps; body, head and thighs meet the flat floor under the field (bolt_env.terrain_xml: box-on-
                    height-field contacts made the GPU env 6x slower).
    push_interval   s between push chances (0: no pushes); each chance pushes with push_prob, a sideways shove of
                    push_speed m/s and push_spin rad/s of roll to a random side (bolt_env.apply_shove: the switch
                    test's shove, skills.Bolt.shove). The switch test's shove is 2.5 m/s and 6 rad/s (it topples Bolt);
                    legged_gym trains with pushes up to 1 m/s.

motor_brake: None = bolt_env's process default (off unless set; train_imitate.py turns it on), True/False for this
env (bolt_env: back-EMF braking above the drive cutoff).

Body (body="go2", or the BOLT_BODY=go2 env var; default "bolt"): a real robot from robot_bodies.BODIES (the Unitree
Go2: Menagerie's model with position servos and the Go2's motor limits), tracking the clips retargeted onto it
(dogmotion/clips/<clip_prefix><clip>.npz, dogmotion/retarget_body.py). The observation, action and reward layout are the
same; the body's own stance replaces DOG (inputs 10 and 57, residual=False), and the body-size limits are the entry's
(Go2: the body may rest on the floor below a 21 cm reference instead of 30 cm, a try ends 10.5 cm off the reference
height instead of 15 cm; its own sit/lie/stand poses for pose_reached). Landing starts and terrain are Bolt-only; the GPU
env (mjx_imitate.py --body go2) has the Go2 too (test_mjx_imitate.py dump-go2 / check-go2: the parity).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch  # noqa: F401  (imported by the policies below; keeps the import order of the other envs)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import ACTION_SCALE, STANCES, TERRAIN, BoltEnv, apply_shove, physics, terrain_height
from pose_reached import CLIP_POSE, pose_ok

DOG = STANCES["dog"]
CLIP_DIR = Path(__file__).with_name("dogmotion") / "clips"
ALL_CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"]   # getup: lie reversed (run 18d)
DEFAULT_CLIPS = ["walk", "trot", "pace", "turn_left", "turn_right", "sit", "lie"]   # canter (3.4 m/s) later
WEIGHTS = dict(pose=0.5, vel=0.05, paw=0.2, root=0.15, root_vel=0.1)
STEER = dict(share=0.6, change=0.3, weight=0.15)   # steering defaults (ImitateEnv steer_*)
STEER_CLIPS = ("walk", "trot", "pace", "canter")
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
# defaults of the transition-term arguments (see above): weight 0 = off; window s after a switch (the falls come 0.5-0.8 s
# after it); threshold rad of roll or pitch away from the reference's before it costs
TRANSITION = dict(weight=0.0, window=1.0, threshold=0.3)
# real-robot gaps (see above): sensor noise levels, legged_gym's (Rudin et al. 2021), copied from mastermind/env.py NOISE
# (base height has no sensor on a real robot; 0.02 m is mastermind's assumption, not a spec)
NOISE = dict(grav=0.05, gyro=0.2, linvel=0.1, height=0.02, qpos=0.01, qvel=1.5)
REAL = dict(latency_ms=(0.0, 0.0), obs_noise=0.0, motor_lag_ms=0.0)   # defaults: off
LATENCY_SWEEP = (0, 10, 20, 40)   # ms, exam(): mastermind's stress-test points
# rough terrain and pushes (see above): defaults off (cm 0, interval 0); clips that start on the terrain, m around its
# middle where they start; push chance, push size (legged_gym's max push speed; no roll spin unless asked)
TERRAIN_ARGS = dict(cm=TERRAIN["cm"], clips=("walk", "trot"), spread=5.0, seed=TERRAIN["seed"])
PUSH = dict(interval=0.0, prob=0.5, speed=1.0, spin=0.0)


def roll_pitch(R):
    """(roll, nose-up pitch) of a body rotation matrix (yaw-free or not), as BoltEnv.posture() and skills' trace."""
    return np.arcsin(np.clip(R[2, 1], -1, 1)), np.arcsin(np.clip(R[2, 0], -1, 1))


def closest_moment(clip, q, qd):
    """Clip time (s) whose joint pose is closest to q (joint speeds weigh a little, to pick the right half of the stride
    when the pose alone is ambiguous). The one hand-over rule: the skill manager (skills.py imports it from here) and
    the moving-start commands both use it, so training joins a clip where the switch test does."""
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

    def __init__(self, name, prefix=""):
        z = np.load(CLIP_DIR / f"{prefix}{name}.npz")   # prefix "go2_": the clip retargeted onto the Go2
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
                 pose_time=None, pose_weight=MOVING["weight"], transition_weight=TRANSITION["weight"],
                 transition_window=TRANSITION["window"], transition_threshold=TRANSITION["threshold"],
                 latency_ms=REAL["latency_ms"], obs_noise=REAL["obs_noise"], motor_lag_ms=REAL["motor_lag_ms"],
                 terrain_cm=TERRAIN_ARGS["cm"], terrain_clips=TERRAIN_ARGS["clips"], terrain_spread=TERRAIN_ARGS["spread"],
                 terrain_seed=TERRAIN_ARGS["seed"], push_interval=PUSH["interval"], push_prob=PUSH["prob"],
                 push_speed=PUSH["speed"], push_spin=PUSH["spin"], motor_brake=None, root_global=0.0,
                 steer_max=0.0, steer_share=STEER["share"], steer_change=STEER["change"], steer_weight=STEER["weight"],
                 body=None):
        super().__init__(randomize=randomize, max_steps=int(LOOP_SECONDS * 50), seed=seed, gait_generator="dog",
                         skills=["stand"], motor_brake=motor_brake, terrain_cm=terrain_cm, terrain_seed=terrain_seed,
                         body=body)
        # the body's stance, clip files, size limits and poses (module docstring, "Body")
        self.dog, prefix, self.floor_ok_h, self.lost_h, self.poses = DOG, "", 0.3, 0.15, "bolt"
        if self.body != "bolt":
            r = self.robot
            self.dog, prefix, self.floor_ok_h, self.lost_h, self.poses = r.stance, r.clip_prefix, r.floor_ok, r.lost, r.pose_ok
            assert not landing_rate, "landing starts are Bolt's (recorded from Bolt's jump brain)"
        self.slots = slots   # clip one-hot size: 8 for brains from before the getup clip (74 inputs)
        # root_global (0..1): blend of our two root terms (height + tilt, body-frame velocity) with Peng et al. 2020's
        # Eq. 8-9 (GLOBAL root position, full rotation incl. heading, global velocity), the reference root integrated
        # from the try's start (research/peng_score.py; 29 Oct: Bolt drifts ~10 m off the trot's path in 10 s, root 0.09).
        # The inputs are unchanged, so the brain cannot see its drift; it can learn to turn and run at the clip's rates.
        self.root_global = root_global
        # steering (30 Oct, Bolt Run: Bolt could not steer at a trot): a yaw-rate command (rad/s) added to the gait's
        # reference spin rate (input 73, the spin-rate terms, the global root's heading) on STEER_CLIPS. Training draws it
        # for steer_share of the tries, uniform in +-steer_max, redrawn at steer_change per second; a "steer" term
        # (steer_weight x exp(-4 (w_z - w_ref_z)^2)) pays for following it. steer_max 0 (the default): exactly as before.
        self.steer_max, self.steer_share, self.steer_change, self.steer_weight = steer_max, steer_share, steer_change, steer_weight
        self.steer, self.exam_steer = 0.0, None   # the command now; exams set exam_steer
        self.switch_rate = switch_rate   # chance per second of switching to a following clip mid-try (training only)
        self.OBS = 66 + slots
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.clip_names = list(clips or DEFAULT_CLIPS)
        self.clips = {n: Clip(n, prefix) for n in self.clip_names}
        self.residual = residual
        self.weights = dict(WEIGHTS, **(weights or {}))   # e.g. {"root_vel": 0.25} if it keeps poses but not pace
        self.exam_start = None        # (clip name, start time s): exams set it; training draws it
        self.body_geoms = {g for g in range(self.model.ngeom) if self.geom_leg[g] < 0 and g != self.floor and g not in self.ground}
        self.terrain_clips, self.terrain_spread = tuple(terrain_clips), terrain_spread
        self.push_steps = int(round(push_interval / self.dt)) if push_interval else 0
        self.push_prob, self.push_speed, self.push_spin = push_prob, push_speed, push_spin
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
        # transition term (module docstring): -weight x excess roll/pitch for window steps after a mid-try switch
        self.transition_weight, self.transition_threshold = transition_weight, transition_threshold
        self.transition_steps = int(round(transition_window / self.dt))
        self.switch_t = None          # step of the last mid-try clip switch (None: none this try)
        # real-robot gaps (module docstring)
        self.latency_ms, self.obs_noise, self.motor_lag_ms = tuple(latency_ms), obs_noise, motor_lag_ms
        self.exam_latency_ms = None   # a fixed latency for exams (the sweep)
        self.delay_sub, self._queue, self._lag_y = 0, None, None

    def gait_speed(self, name):
        """The clip's own pace: mean forward body speed (m/s)."""
        return float(self.clips[name].v_body[:, 0].mean())

    def pose_now(self, name):
        """Is the body in pose `name` (pose_reached.POSE_OK) right now?"""
        R = self._rotation()
        return bool(pose_ok(name, self.data.qpos[2], np.arcsin(np.clip(R[2, 0], -1, 1)), R[2, 2], self.poses))

    # ---------- reference ----------
    def tau(self, ahead=0):
        return self.tau0 + (self.t + ahead) * self.dt

    def ref(self, key, ahead=0):
        return self.clip.at(self.tau(ahead), key)

    def ref_w(self, ahead=0):
        """The reference spin rates (body frame) with the steering command added to the yaw rate on STEER_CLIPS."""
        w = self.ref("w_body", ahead)
        if self.steer and self.clip.name in STEER_CLIPS:
            w = w + np.array([0.0, 0.0, self.steer])
        return w

    def heading_paws(self):
        R = self._rotation()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        rel = np.array([self.data.geom_xpos[g] for g in self.foot_ids]) - self.data.qpos[:3]
        return rel @ _rotz(yaw)

    def ground_z(self):
        """Height of the ground under the body (m): the terrain's (0 off it), or None on a flat-floor env."""
        if self.terrain is None:
            return None
        return float(terrain_height(self.terrain["field"], self.terrain["cm"], self.data.qpos[0], self.data.qpos[1]))

    def body_on_floor(self):
        d = self.data
        if self.terrain is not None:   # the terrain is floor too
            return any((c.geom1 in self.ground and c.geom2 in self.body_geoms) or
                       (c.geom2 in self.ground and c.geom1 in self.body_geoms) for c in d.contact[:d.ncon])
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
            R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3],
            [d.qpos[2] if self.terrain is None else d.qpos[2] - self.ground_z()],
            d.qpos[7:19] - self.dog, d.qvel[6:18] * 0.1, self.last_action,
            [np.sin(ph), np.cos(ph)], onehot,
            self.ref("joints", 1) - self.dog, [self.ref("height", 1)], self.ref("v_body", 1), [self.ref_w(1)[2]],
            [self.finished()],
        ])
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)
        if self.obs_noise:   # the stress test's sensor noise, in observation units (gyro x0.25, joint speed x0.1)
            k, u = self.obs_noise, lambda n: self.rng.uniform(-1, 1, n)
            obs[0:3] += k * NOISE["grav"] * u(3)
            obs[3:6] += k * NOISE["gyro"] * 0.25 * u(3)
            obs[6:9] += k * NOISE["linvel"] * u(3)
            obs[9] += k * NOISE["height"] * u(1)[0]
            obs[10:22] += k * NOISE["qpos"] * u(12)
            obs[22:34] += k * NOISE["qvel"] * 0.1 * u(12)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        self._try_id = getattr(self, "_try_id", 0) + 1
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
        self.switch_t = None
        self.steer = 0.0 if self.exam_steer is None else float(self.exam_steer)
        if self.exam_steer is None and self.steer_max and not self.exam_start and self.rng.random() < self.steer_share:
            self.steer = float(self.rng.uniform(-self.steer_max, self.steer_max))
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
        self._rsi(on_terrain=self.terrain is not None and name in self.terrain_clips)
        return self._settle()

    def _rsi(self, vscale=1.0, on_terrain=False):
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
        if on_terrain:   # a random spot on the rough field, at the ground's height there
            at, w = TERRAIN["at"], self.terrain_spread
            d.qpos[0:2] = [at[0] + self.rng.uniform(-w, w), at[1] + self.rng.uniform(-w, w)]
            d.qpos[2] += self.ground_z()
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
        lat = self.exam_latency_ms
        if lat is None and self.latency_ms[1] > 0:
            lat = self.rng.uniform(*self.latency_ms)
        # latency in physics substeps; the queue holds the targets sent (newest last), the robot holding its first
        self.delay_sub = int(round((lat or 0.0) / 1000.0 / m.opt.timestep))
        if self.delay_sub or self.motor_lag_ms:
            longest = max(self.delay_sub, int(round(self.latency_ms[1] / 1000.0 / m.opt.timestep)))   # as MjxImitate.queue_len
            self._queue = [d.ctrl.copy() for _ in range(longest // self.frame_skip + 2)]
            self._lag_y = d.ctrl.copy()
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
        e_h = self.ref("height") - (d.qpos[2] if self.terrain is None else d.qpos[2] - self.ground_z())
        g, g_ref = R.T @ np.array([0, 0, -1.0]), self.ref("gravity")
        tilt = np.arccos(np.clip(np.dot(g, g_ref) / np.linalg.norm(g_ref), -1, 1))
        v, w = R.T @ d.qvel[0:3], d.qvel[3:6]   # MuJoCo free joint: angular velocity is in the body frame
        e_v = np.sum((self.ref("v_body") - v) ** 2)
        e_w = np.sum((self.ref_w() - w) ** 2)
        terms = dict(pose=np.exp(-5 * e_pose), vel=np.exp(-0.1 * e_vel), paw=np.exp(-40 * e_paw),
                     root=np.exp(-20 * e_h ** 2 - 10 * tilt ** 2), root_vel=np.exp(-2 * e_v - 0.2 * e_w))
        if self.steer_max:   # only steering runs have the term (old runs' rewards unchanged)
            terms["steer"] = np.exp(-4 * (self.ref_w()[2] - w[2]) ** 2)
        return terms, dict(height_err=float(e_h), tilt=float(tilt), pose_rms=float(np.sqrt(e_pose / 12)))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        base = self.ref("joints", 1) if self.residual else self.dog
        self.data.ctrl[:] = np.clip(base + ACTION_SCALE * a, self.lo, self.hi)
        if self.push_steps and self.t > 0 and self.t % self.push_steps == 0 and self.rng.random() < self.push_prob:
            side = 1.0 if self.rng.random() < 0.5 else -1.0   # a push now (the switch test's shove, push-sized)
            self.data.qvel[:] = apply_shove(self.data.qvel, self._rotation(), self.push_speed, self.push_spin, side)
        if self.delay_sub or self.motor_lag_ms:
            self._delayed_physics()
        else:
            physics(self.model, self.data, self.frame_skip)
        self.t += 1
        self.env_steps += 1
        if self.switch_rate and not self.exam_start:
            self._maybe_switch()
        if self.steer_max and self.exam_steer is None and not self.exam_start and self.rng.random() < self.steer_change * self.dt:
            self.steer = float(self.rng.uniform(-self.steer_max, self.steer_max)) if self.rng.random() < self.steer_share else 0.0
        if self.t == self.land_switch_at and self.clip.name != self.landing_then:   # a landing start's bridge is over:
            self._switch_to(self.clips[self.landing_then])                         # on in landing_then (test_skills' trot)
        terms, err = self.tracking()
        if self.root_global:
            g_root, g_vel = self._global_root_terms()
            terms["root"] = (1 - self.root_global) * terms["root"] + self.root_global * g_root
            terms["root_vel"] = (1 - self.root_global) * terms["root_vel"] + self.root_global * g_vel
        reward = sum(self.weights.get(k, self.steer_weight) * v for k, v in terms.items())
        pose = None
        if self.pose_clip is not None:   # a moving-start command: paid per step in the pose once reached in time
            pose = self.clip.name == self.pose_clip and self.pose_now(CLIP_POSE[self.pose_clip])
            self.pose_hit = self.pose_hit or (pose and self.t <= self.pose_until)
            if self.pose_hit and pose:
                reward += self.pose_weight
        trans = None
        if self.transition_weight and self.switch_t is not None and self.t - self.switch_t < self.transition_steps:
            trans = self.transition_excess()
            reward -= self.transition_weight * trans
        self.last_action = a
        floor_ok = self.ref("height") < self.floor_ok_h     # sitting and lying: the body may rest on the floor
        lost = (abs(err["height_err"]) > self.lost_h and self.t > self.grace_steps) or err["tilt"] > 0.8   # grace: landing starts
        fell = lost or (self.body_on_floor() and not floor_ok)
        done = self.t >= self.limit
        info = dict(err, clip=self.clip.name, tau=self.tau(), fell=fell, **{f"r_{k}": float(v) for k, v in terms.items()})
        if pose is not None:
            info.update(pose_ok=bool(pose), pose_hit=bool(self.pose_hit))
        if self.transition_weight:
            info.update(transition=float(trans or 0.0), since_switch=None if self.switch_t is None else self.t - self.switch_t)
        return self._obs(), float(reward), bool(fell), bool(done and not fell), info


    def _global_root_terms(self):
        """Peng 2020 Eq. 8-9 against the reference root integrated since the try began (started by _start_ref_root)."""
        d = self.data
        if getattr(self, "_ref_root", None) is None or self._ref_root[2] != self._try_id:
            R = self._rotation()
            self._ref_root = [d.qpos[:2].copy(), float(np.arctan2(R[1, 0], R[0, 0])), self._try_id]
        u, _, vt = np.linalg.svd(self.ref("tilt"))
        tilt = u @ vt
        R_ref = _rotz(self._ref_root[1]) @ tilt
        v_ref, w_ref = R_ref @ self.ref("v_body"), R_ref @ self.ref_w()
        self._ref_root[0] = self._ref_root[0] + v_ref[:2] * self.dt
        self._ref_root[1] += float(w_ref[2]) * self.dt
        R_ref = _rotz(self._ref_root[1]) @ tilt
        R = self._rotation()
        h = d.qpos[2] if self.terrain is None else d.qpos[2] - self.ground_z()
        dx2 = float(np.sum((self._ref_root[0] - d.qpos[:2]) ** 2) + (self.ref("height") - h) ** 2)
        ang = float(np.arccos(np.clip((np.trace(R_ref.T @ R) - 1) / 2, -1, 1)))
        g_root = np.exp(-20 * dx2 - 10 * ang ** 2)
        g_vel = np.exp(-2 * np.sum((v_ref - d.qvel[0:3]) ** 2) - 0.2 * np.sum((w_ref - R @ d.qvel[3:6]) ** 2))
        return g_root, g_vel

    def _maybe_switch(self):
        """Now and then hand over to a clip that may follow this one (NEXT), as a dog changes what it does."""
        ready = self.clip.loop or self.finished() > 0.5 * HOLD
        nxt = [n for n in NEXT.get(self.clip.name, ()) if n in self.clips and n != self.clip.name]
        if not ready or not nxt or self.rng.random() > self.switch_rate * self.dt:
            return
        self._switch_to(self.clips[self.rng.choice(nxt)])

    def _delayed_physics(self):
        """One control step with the latency and motor lag (mastermind/env.py's _physics): substep i gets the target sent
        back (+1 for the first `sub` substeps) control steps ago, through the first-order lag; ctrl keeps what was sent."""
        m, d = self.model, self.data
        sent = d.ctrl.copy()
        self._queue = self._queue[1:] + [sent]
        back, sub = divmod(self.delay_sub, self.frame_skip)
        target, older = self._queue[-1 - back], self._queue[-2 - back]
        g = 1.0 - np.exp(-m.opt.timestep / (self.motor_lag_ms / 1000.0)) if self.motor_lag_ms else 1.0
        for i in range(self.frame_skip):
            want = older if i < sub else target
            if self.motor_lag_ms:
                self._lag_y = self._lag_y + g * (want - self._lag_y)
                want = self._lag_y
            d.ctrl[:] = want
            physics(m, d, 1)
        d.ctrl[:] = sent

    def transition_excess(self):
        """Roll and pitch (rad) beyond transition_threshold, each against the reference's own, summed."""
        r, p = roll_pitch(self._rotation())
        r_ref, p_ref = roll_pitch(self.ref("tilt"))
        return float(max(abs(r - r_ref) - self.transition_threshold, 0.0) + max(abs(p - p_ref) - self.transition_threshold, 0.0))

    def _switch_to(self, c):
        """Hand over to clip c now."""
        self.switch_t = self.t
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


def exam(policy, clips=None, episodes=4, seed=9000, latency_sweep=LATENCY_SWEEP):
    """Per clip: the mean tracking reward per step over the whole try (0 counted for every step lost after a
    fall), the fall rate, and for gaits how much of the dog's pace Bolt keeps (speed_: forward distance
    over the dog's; for turns, the angle turned over the dog's). Loops run 4 s from evenly spaced points in
    the stride; sit and lie play from the start through their 1 s hold. Physics is randomized as in training.
    A brain can hold the poses while barely moving (the smoke test at 300k: 1/4 of the walk's speed), so the
    pace is part of the score (score()).
    Posture clips (sit, lie): pose_still_<clip> is the share of tries from standing still (the clip's start) that
    reached the clip's end pose (pose_reached.py, the switch test's definition) within the time limit (pose_time) and
    did not fall; pose_moving_<gait>_<clip> (and fall_moving_<gait>_<clip>) the same from a moving start: Bolt in that
    gait at its own pace, told to sit or lie at once (env.exam_moving). Neither enters score().
    Latency sweep (latency_sweep ms, () for none): lat<ms>_track (mean tracking reward per step over every clip's tries,
    as track_) and lat<ms>_fall (share of tries that fell) with the motors that many ms behind the brain."""
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
    # latency sweep (real-robot gap): the loops' and posture clips' tracking exam again at each fixed latency
    for ms in latency_sweep:
        env.exam_latency_ms = float(ms)
        scores, falls = [], []
        for name in env.clip_names:
            c = env.clips[name]
            for k in range(episodes):
                env.exam_start = (name, (k / episodes) * c.length if c.loop else 0.0)
                obs, _ = env.reset(seed=seed + k)
                if c.loop:
                    env.limit = 200
                total, fell = 0.0, False
                for _ in range(env.limit):
                    obs, r, fell, done, info = env.step(policy.act(obs))
                    total += r
                    if fell or done:
                        break
                scores.append(total / env.limit)
                falls.append(fell)
        out[f"lat{ms}_track"] = round(float(np.mean(scores)), 3)
        out[f"lat{ms}_fall"] = round(float(np.mean(falls)), 3)
    env.exam_latency_ms = env.exam_start = None
    return out


def score(result):
    """One number for keeping the best brain: mean over clips of tracking x (gaits) the share of the dog's
    pace kept, 0..1."""
    names = [k[6:] for k in result if k.startswith("track_")]
    return float(np.mean([result[f"track_{n}"] * float(np.clip(result.get(f"speed_{n}", 1.0), 0.0, 1.0))
                          for n in names]))


def exam_steer(policy, clips=("trot", "canter"), rates=(-1.0, -0.5, 0.5, 1.0), episodes=2, seconds=4.0, seed=9500):
    """Steering exam: per clip, Bolt plays the gait with a fixed yaw-rate command (env.exam_steer, rad/s) for `seconds`,
    from evenly spaced stride moments, randomized physics. steer_<clip>: how well the turn follows the command, 1 -
    |turned - commanded| / |commanded| averaged over tries (clipped to 0..1; a fall counts 0); steertrack_<clip> the
    mean tracking reward per step; steerfall_<clip> the share of tries that fell; turn_<clip>_<rate> the mean yaw rate
    reached. steer_mean: the mean of steer_<clip>."""
    env = ImitateEnv(randomize=True, clips=list(clips))
    steps = int(round(seconds / env.dt))
    out = {}
    for name in clips:
        c = env.clips[name]
        follow, track, falls = [], [], 0
        for rate in rates:
            got = []
            for k in range(episodes):
                env.exam_start = (name, (k / episodes) * c.length)
                env.exam_steer = rate
                obs, _ = env.reset(seed=seed + k)
                env.limit = steps
                R = env._rotation()
                h, yaw, total, fell = np.arctan2(R[1, 0], R[0, 0]), 0.0, 0.0, False
                for _ in range(steps):
                    obs, r, fell, done, _ = env.step(policy.act(obs))
                    R = env._rotation()
                    h1 = np.arctan2(R[1, 0], R[0, 0])
                    yaw += float(np.angle(np.exp(1j * (h1 - h))))
                    h, total = h1, total + r
                    if fell or done:
                        break
                falls += fell
                track.append(total / steps)
                want = rate * seconds
                follow.append(0.0 if fell else float(np.clip(1 - abs(yaw - want) / abs(want), 0, 1)))
                got.append(yaw / seconds)
            out[f"turn_{name}_{rate:+g}"] = round(float(np.mean(got)), 3)
        out[f"steer_{name}"] = round(float(np.mean(follow)), 3)
        out[f"steertrack_{name}"] = round(float(np.mean(track)), 3)
        out[f"steerfall_{name}"] = round(falls / (len(rates) * episodes), 3)
    env.exam_steer, env.exam_start = None, None
    out["steer_mean"] = round(float(np.mean([out[f"steer_{n}"] for n in clips])), 3)
    return out


def exam_terrain(policy, cm, clip="trot", tries=30, seconds=4.0, seed=9000, push_speed=0.0, push_spin=0.0):
    """Step 6's target test: `tries` tries of `clip` on the rough field (bolt_env TERRAIN, cm high; 0: the flat floor),
    each starting at an evenly spaced moment of the stride, randomized physics, `seconds` long. A fall is the skill
    manager's (skills.FALL_UP / _after_step): tipped past 60 degrees, or the body on the ground for 0.1 s; a try that
    the env ends early for losing the motion (15 cm off the reference height, 0.8 rad tilt) is counted apart (ended).
    push_speed/push_spin: one sideways shove 1 s in (0: none). Returns fall, ended, track (mean reward per step), speed
    (forward distance over the clip's)."""
    env = ImitateEnv(randomize=True, clips=[clip], terrain_cm=cm)
    c = env.clips[clip]
    steps = int(round(seconds / env.dt))
    falls, ended, track, pace = 0, 0, [], []
    for k in range(tries):
        env.exam_start = (clip, (k / tries) * c.length)
        obs, _ = env.reset(seed=seed + k)
        env.limit = steps
        R = env._rotation()
        h0, p0 = np.arctan2(R[1, 0], R[0, 0]), env.data.qpos[:2].copy()
        floor_steps, fell, early, total = 0, False, False, 0.0
        for t in range(steps):
            if push_speed and t == int(round(1.0 / env.dt)):
                env.data.qvel[:] = apply_shove(env.data.qvel, env._rotation(), push_speed, push_spin, 1.0 if k % 2 else -1.0)
            obs, r, lost, done, info = env.step(policy.act(obs))
            total += r
            floor_steps = floor_steps + 1 if env.body_on_floor() else 0
            if env._rotation()[2, 2] < 0.5 or floor_steps >= 5:
                fell = True
                break
            if lost and not early:
                early = True   # the env would end the try here; keep going to see whether Bolt really falls
        falls += fell
        ended += early or fell
        track.append(total / steps)
        pace.append(float(np.dot(env.data.qpos[:2] - p0, [np.cos(h0), np.sin(h0)])) / (steps * env.dt * c.v_body[:, 0].mean()))
    return dict(cm=cm, clip=clip, tries=tries, fall=falls / tries, ended=ended / tries, track=round(float(np.mean(track)), 3),
                speed=round(float(np.mean(pace)), 2))
