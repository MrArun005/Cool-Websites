"""The robot-side brain loop for a real Go2 (train/deploy/PLAN.md, phase 1 steps 3-5): sensor readings in, joint
targets out, every 20 ms, exactly as imitate_env.ImitateEnv builds the inputs and turns the brain's output into motor
targets during training.

    c = Go2Controller("models/bolt-go2-damp-798M/params.pkl")
    c.command("trot", steer=0.3)                      # any clip of the brain; steer: yaw rate (rad/s) on the gaits
    q_target = c.step(quat, gyro, q, dq, v_body, height)   # all in OUR joint order (FL, FR, RL, RR x hip, thigh, calf)

Inputs (75, imitate_env.ImitateEnv._obs): gravity in the body frame (from the IMU quaternion), gyro x 0.25, body
velocity in the body frame, body height, joint angles minus the stance, joint speeds x 0.1, the previous action, clip
phase (sin, cos), clip one-hot (9), the reference one step ahead (joints minus stance, height, body velocity, yaw rate
with the steering command added), and seconds since a sit/lie clip ended.

Body velocity and height have no sensor on a real Go2: the caller passes estimates (PLAN.md phase 3). Everything else
is a direct reading. Numpy and mujoco only (no torch, SB3 or JAX), so it runs on the Go2's Jetson.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import robot_bodies  # noqa: E402
from bolt_env import ACTION_SCALE  # noqa: E402
from mjx_imitate import MjxImitatePolicy  # noqa: E402

CLIP_DIR = HERE.parent / "dogmotion" / "clips"
# as imitate_env (copied, not imported: imitate_env pulls in torch and SB3, which the robot does not need)
ALL_CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup", "jump"]
MOTION_CLIPS = ALL_CLIPS[:9]   # what every brain knows; "jump" only for a brain with the 10th slot
GETUP_DONE = 0.2                   # seconds past the get-up's end before a move commanded while lying starts
THEN = {"jump": ("trot", 0.2)}   # a clip that ends running: hand over to this gait this long after its end (training's
                                 # hold, imitate_env.CLIP_HOLD), joining at the closest pose
STEER_CLIPS = ("walk", "trot", "pace", "canter")
FAST = ("trot", "pace", "canter")   # sit / lie commanded in one of these: walk for walk_down s first (walk -> sit / lie
SLOW_FIRST = ("sit", "lie")         # never fell in 12 moving tries; trot -> sit did, 2-3 of 12)
DT = 0.02   # s: the brain's step (50 Hz), as BoltEnv (0.002 s physics x 10)


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_mat(q):
    """(w, x, y, z) -> 3x3 rotation, body to world (the Unitree IMU's quaternion and MuJoCo's share this order)."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class Clip:
    """The reference parts of imitate_env.Clip the inputs use (same file, same numbers; test_deploy checks it)."""

    def __init__(self, name, prefix="go2_"):
        z = np.load(CLIP_DIR / f"{prefix}{name}.npz")
        self.name, self.loop, self.fps = name, bool(z["loop"]), float(z["fps"])
        self.n = len(z["joints"])
        self.length = self.n / self.fps
        self.joints, self.joint_vel = z["joints"], z["joint_vel"]
        self.height = z["base_pos"][:, 2]
        R = np.array([quat_to_mat(q) for q in z["base_quat"]])
        self.v_body = np.einsum("tji,tj->ti", R, z["base_linvel"])
        self.w_body = np.einsum("tji,tj->ti", R, z["base_angvel"])

    def frame(self, tau):
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


def predict(quat, gyro, q, dq, ahead):
    """Latency compensation: the state `ahead` seconds from now, extrapolated at constant speed (joint angles with
    their speeds; body rotation with the body-frame gyro). The brain then answers for the moment its command lands,
    not the moment it was measured (PLAN.md 3 Nov: the Lite3 brain falls with 3-15 ms of command delay)."""
    if ahead <= 0:
        return quat, q
    w = np.asarray(gyro, float) * ahead          # body-frame rotation vector
    ang = float(np.linalg.norm(w))
    if ang > 1e-9:
        ax = w / ang
        dqt = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])
        a, b = np.asarray(quat, float), dqt       # q_new = q * dq (body-frame increment)
        quat = np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                         a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                         a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                         a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])
        quat = quat / np.linalg.norm(quat)
    return quat, np.asarray(q, float) + np.asarray(dq, float) * ahead


class Go2Controller:
    def __init__(self, brain, body="go2", clips=None, walk_down=0.0):
        """clips None: the motion clips, plus "jump" when the brain has its slot (a 9-slot brain never saw it).
        walk_down: seconds of walk between a running gait and sit / lie (0: switch at once, as training's moving starts)."""
        self.brain = MjxImitatePolicy(brain)
        if clips is None:
            clips = MOTION_CLIPS + (["jump"] if self.brain.slots >= 10 else [])
        self.body = robot_bodies.get(body)
        self.stance = np.asarray(self.body.stance, float)
        m = self.body.model()
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.clips = {n: Clip(n, self.body.clip_prefix) for n in clips}
        self.slots = len(ALL_CLIPS)
        self.last_action = np.zeros(12)
        self.steer = 0.0
        self.clip, self.tau = self.clips["trot" if "trot" in self.clips else clips[0]], 0.0
        self.walk_down, self.queued = float(walk_down), None   # queued: (posture, seconds of walk left)
        self.after_getup = None   # a move commanded while lying: played once the get-up has finished (see command)

    # ---------- commands ----------
    def command(self, name, steer=None, q=None):
        """Switch to clip `name` (a gait joins at the frame closest to the current pose q, as ImitateEnv._switch_to;
        sit/lie/getup play from their start). steer: yaw rate (rad/s) added on the gaits; None keeps it."""
        if steer is not None:
            self.steer = float(steer)
        if self.queued and name != self.queued[0]:
            self.queued = None          # a new command replaces a posture still waiting for the walk
        self.after_getup = None         # a new command replaces a move still waiting for the get-up
        # lying (or still getting up), any move but lie / getup goes through the get-up first: training only ever
        # leaves lie by getup, and sit straight from lie fell 6 of 6 times on DEEP's Lite3 model (PLAN.md 4 Nov)
        if (name not in ("lie", "getup") and "getup" in self.clips
                and (self.clip.name == "lie" or (self.clip.name == "getup" and self.finished() < GETUP_DONE))):
            self.after_getup = name
            name = "getup"
        if name == self.clip.name:
            return
        if (name in SLOW_FIRST and self.clip.name in FAST and self.walk_down > 0 and "walk" in self.clips
                and not self.queued):
            self.queued = (name, self.walk_down)
            name = "walk"
        c = self.clips[name]
        if c.loop and q is not None:
            self.tau = float(np.argmin(np.sum((c.joints - np.asarray(q)) ** 2, 1))) / c.fps
        else:
            self.tau = 0.0
        self.clip = c

    def start_at(self, name, tau):
        """Set the clip and its time directly (tests, and starting from a known state)."""
        self.clip, self.tau = self.clips[name], float(tau)

    # ---------- the loop ----------
    def ref(self, key, ahead=1):
        return self.clip.at(self.tau + ahead * DT, key)

    def ref_yaw_rate(self, ahead=1):
        w = self.ref("w_body", ahead)[2]
        return w + (self.steer if self.clip.name in STEER_CLIPS else 0.0)

    def finished(self):
        return 0.0 if self.clip.loop else max(0.0, self.tau - self.clip.length)

    def inputs(self, quat, gyro, q, dq, v_body, height):
        R = quat_to_mat(quat)
        ph = 2 * np.pi * self.clip.phase(self.tau)
        onehot = np.zeros(self.slots)
        onehot[ALL_CLIPS.index(self.clip.name)] = 1.0
        return np.concatenate([
            R.T @ np.array([0, 0, -1.0]), np.asarray(gyro, float) * 0.25, np.asarray(v_body, float), [float(height)],
            np.asarray(q, float) - self.stance, np.asarray(dq, float) * 0.1, self.last_action,
            [np.sin(ph), np.cos(ph)], onehot,
            self.ref("joints") - self.stance, [self.ref("height")], self.ref("v_body"), [self.ref_yaw_rate()],
            [self.finished()],
        ])

    def step(self, quat, gyro, q, dq, v_body, height):
        """One 20 ms brain step: the joint targets (our order) for the next 20 ms. Advances the clip clock."""
        obs = self.inputs(quat, gyro, q, dq, v_body, height)
        a = self.brain.act(obs)
        target = np.clip(self.ref("joints") + ACTION_SCALE * a, self.lo, self.hi)   # residual on the reference
        self.last_action = np.asarray(a, float)
        self.tau += DT
        if self.queued:
            self.queued = (self.queued[0], self.queued[1] - DT)
            if self.queued[1] <= 1e-9:
                name, self.queued = self.queued[0], None
                self.command(name, q=q)
        if self.after_getup and self.clip.name == "getup" and self.finished() >= GETUP_DONE:
            self.command(self.after_getup, q=q)
        nxt = THEN.get(self.clip.name)
        if nxt and nxt[0] in self.clips and self.finished() >= nxt[1]:
            self.command(nxt[0], q=q)
        return target
