"""Scripted moves for the hybrid duck (Microduck body + owner's head): the skills Microduck has no brain for.

Each move is a "brain" with the same interface as an ONNX policy in ../microduck_port/harness.py:
`brain(obs61) -> action14` (servo target = DEFAULT_POSE + action), `action_scale = 1.0`, plus `begin(rt)`
when the runtime hands control to it. They read only what the robot has: the IMU (gyro, gravity) in the
observation, and the time since they started.

Balance help (the idea that worked on our own duck): the body is NOT statically stable on Microduck's
home pose (holding DEFAULT_POSE without a brain falls forward in ~1 s), so every scripted leg move adds an
ankle feedback on the trunk pitch from the IMU: ankle += kp * pitch + kd * pitch_rate (kp 3, kd 0.1;
found by a sweep, sign checked: the opposite sign falls in < 1 s).

Leg moves are written as a "squat height" h (m): the hip/knee/ankle offsets that lower the trunk by h
with the soles flat and the feet under the hips (leg_ik(), solved once with MuJoCo kinematics on the
hybrid model, table LEG_IK below; h < 0 extends past home, at most 1 cm: the leg is nearly straight).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "microduck_port"))
from harness import DEFAULT_POSE, JOINT_NAMES  # noqa: E402

# leg_ik(): h -> left (hip_pitch, knee, ankle) offsets from DEFAULT_POSE; right leg = minus (mirrored).
# Solved with leg_ik() below on hybrid_allcollisions_noantenna.xml (foot 2-4 mm off the exact sagittal
# target at 3-5 cm: the leg's pitch axes are not exactly parallel to the trunk's).
LEG_IK_H = np.array([-0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06])
LEG_IK = np.array([[-0.3913, -0.7696, -0.3783], [-0.1434, -0.2799, -0.1365], [0.0, 0.0, 0.0], [0.115, 0.222, 0.1071],
                   [0.2153, 0.4137, 0.1983], [0.3069, 0.586, 0.2791], [0.3926, 0.7449, 0.3523], [0.4748, 0.8939, 0.4191],
                   [0.5548, 1.0353, 0.4804], [0.6343, 1.1707, 0.5364], [0.7149, 1.3013, 0.5865], [0.7987, 1.4281, 0.6294],
                   [0.8894, 1.5515, 0.6621], [0.9936, 1.672, 0.6784], [1.1265, 1.7893, 0.6628]])
L_HIP, L_KNEE, L_ANK, R_HIP, R_KNEE, R_ANK = 2, 3, 4, 11, 12, 13
NECK, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = 5, 6, 7, 8
BAL_KP, BAL_KD = 3.0, 0.1


def ik(h):
    h = float(np.clip(h, LEG_IK_H[0], LEG_IK_H[-1]))
    return np.array([np.interp(h, LEG_IK_H, LEG_IK[:, i]) for i in range(3)])


def legs(q, h, dank=0.0, dhip=0.0, dknee=0.0):
    """Add the squat-h leg offsets (plus extra ankle / hip / knee offsets, left sign) to q in place."""
    dh, dk, da = ik(h)
    dh, dk, da = dh + dhip, dk + dknee, da + dank
    q[L_HIP] += dh; q[L_KNEE] += dk; q[L_ANK] += da
    q[R_HIP] -= dh; q[R_KNEE] -= dk; q[R_ANK] -= da
    return q


def pitch(obs):
    """Trunk pitch from the IMU gravity (rad, + = leaning forward)."""
    return math.atan2(float(obs[3]), -float(obs[5]))


def balance(obs, kp=BAL_KP, kd=BAL_KD):
    return kp * pitch(obs) + kd * float(obs[1])


def smooth(s):
    s = min(1.0, max(0.0, s))
    return 0.5 - 0.5 * math.cos(math.pi * s)


class Script:
    """Base: time since begin() in s, head offsets from the runtime (rt.head_offsets(), 4 values)."""
    action_scale = 1.0
    quiet = False

    def __init__(self):
        self.rt, self.k = None, 0

    def begin(self, rt=None):
        self.rt, self.k = rt, 0
        self.q0 = (DEFAULT_POSE + rt.sim.last_action) if rt is not None else DEFAULT_POSE.copy()

    @property
    def t(self):
        return self.k * 0.02

    def head(self, q):
        if self.rt is not None and not self.quiet:
            q[NECK:HEAD_ROLL + 1] += self.rt.head_offsets()
        return q

    def __call__(self, obs):
        q = self.target(obs, self.t)
        self.k += 1
        return (q - DEFAULT_POSE).astype(np.float32)

    def target(self, obs, t):
        raise NotImplementedError


class Squat(Script):
    """Lower the trunk by `depth` m over `down` s (soles flat, IMU ankle balance), then hold.
    `up=True` goes the other way: from the squat back to home over `down` s (then hand to velstand)."""

    def __init__(self, depth=0.045, down=1.0, up=False, start=None, kp=BAL_KP, kd=BAL_KD, kh=0.0):
        super().__init__()
        self.depth, self.down, self.up, self.start = depth, down, up, start
        self.kp, self.kd, self.kh = kp, kd, kh       # kh: hips also lean the trunk back over the feet

    def target(self, obs, t):
        s = smooth(t / self.down)
        h0 = self.depth if self.up else (self.start or 0.0)
        h1 = 0.0 if self.up else self.depth
        q = DEFAULT_POSE.copy()
        legs(q, h0 + (h1 - h0) * s, balance(obs, self.kp, self.kd), -self.kh * pitch(obs))
        return self.head(q)


class Hop(Script):
    """One hop. Crouch (IMU ankle balance), then push: legs to full extension, ankles push, the head swings
    up (the head is a quarter of the mass). The push ends on the knee ENCODER, not a clock: once the knee has
    opened to `release` rad (offset from home), the legs snap back up (`h_air`) so the feet leave the floor
    while the body still rises, and after `Ta` s the legs go to the landing crouch `ht` (balance on) and hold
    there until `Tland` s after the release. A push that hasn't released after `Tp_max` s releases anyway.
    Parameters from the cross-entropy search in search_moves.py (moves_hybrid.json "hop")."""
    quiet = True

    def __init__(self, P, start_h=0.0):
        super().__init__()
        self.P, self.start_h = P, start_h        # start_h: crouch from the previous hop's landing crouch (chained)
        self.t_rel = None

    def begin(self, rt=None):
        super().begin(rt)
        self.t_rel = None

    @property
    def duration(self):
        return self.P["Tc"] + self.P["Tp_max"] + self.P["Tland"]

    def done(self):
        return self.t_rel is not None and self.t >= self.t_rel + self.P["Tland"]

    def target(self, obs, t):
        P = self.P
        bal = balance(obs, P["kp"], P["kd"])
        q = DEFAULT_POSE.copy()
        if t < P["Tc"]:
            s = smooth(t / P["Tc"])
            q[NECK] += P["neck_c"] * s
            return legs(q, self.start_h + (P["hc"] - self.start_h) * s, bal + P["lean"] * s)
        knee = float(obs[6 + L_KNEE])                       # left knee, offset from home (encoder)
        if self.t_rel is None and (knee < P["release"] or t >= P["Tc"] + P["Tp_max"]):
            self.t_rel = t
        if self.t_rel is None:
            q[NECK] += P["neck_p"]
            return legs(q, -0.01, P["push_ank"] + bal, P["push_hip"])
        tr = t - self.t_rel
        if tr < P["Ta"]:
            q[NECK] += P["neck_p"]
            return legs(q, P["h_air"], P["air_ank"])
        s = smooth((tr - P["Ta"]) / 0.1)
        q[NECK] += P["neck_p"] * (1 - s)
        return legs(q, P["h_air"] + (P["ht"] - P["h_air"]) * s, bal)


class Keyframes(Script):
    """Glide through absolute servo-target keyframes (14 values each), each over its duration (smoothed),
    starting from the targets the robot holds; optional IMU ankle balance added on top (`bal` per key)."""

    def __init__(self, keys, durations, bal=None, head=True):
        super().__init__()
        self.keys = [np.asarray(k, np.float32) for k in keys]
        self.durs = list(durations)
        self.bal = list(bal) if bal is not None else [0.0] * len(self.keys)
        self.quiet = not head

    @property
    def duration(self):
        return float(sum(self.durs))

    def target(self, obs, t):
        prev = self.q0
        for i, (key, dur) in enumerate(zip(self.keys, self.durs)):
            if t < dur or i == len(self.keys) - 1:
                q = prev + (key - prev) * smooth(t / dur)
                b = self.bal[i] * balance(obs)
                q = q.copy()
                q[L_ANK] += b; q[R_ANK] -= b
                return self.head(q)
            t, prev = t - dur, key
        return self.keys[-1].copy()


class Residual:
    """A trained brain plus fixed servo-target offsets, ramped in over `ramp` s (or out, from full to zero):
    the brain keeps the balance, the offsets change the pose (the swag sit: alpha_sitstand seated, legs splayed,
    one knee opened). The brain sees the result in its joint observation and leans against it a little; the
    offsets are searched with the brain in the loop (search_moves.py swag)."""

    def __init__(self, policy, offsets, ramp=1.0, out=False, guard_deg=None):
        self.policy, self.off, self.ramp, self.out = policy, np.asarray(offsets, np.float32), ramp, out
        self.action_scale, self.k, self.quiet = policy.action_scale, 0, False
        self.guard_deg, self.g = guard_deg, 1.0   # lean-back guard: fade the offsets out past guard_deg (IMU)

    def begin(self, rt=None):
        self.k = 0

    @property
    def duration(self):
        return self.ramp

    def __call__(self, obs):
        f = min(1.0, self.k * 0.02 / self.ramp) if self.ramp > 0 else 1.0
        self.k += 1
        if self.guard_deg is not None:
            lean = -math.degrees(pitch(obs))
            want = float(np.clip(1.0 - (lean - self.guard_deg) / 4.0, 0.0, 1.0))
            self.g += float(np.clip(want - self.g, -0.08, 0.005))    # back off fast, return slowly
        return self.policy(obs) + self.off * ((1.0 - f) if self.out else f) * self.g


def leg_ik(model_path, hs):
    """Solve LEG_IK (kinematics only): for each h, the left (hip_pitch, knee, ankle) offsets that keep the foot
    flat and under the hip while the trunk is h lower. Gauss-Newton on finite differences."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in JOINT_NAMES]
    foot = m.body("ankle_left").id

    def fk(x):
        d.qpos[:] = 0; d.qpos[2] = 0.2; d.qpos[3] = 1
        q = DEFAULT_POSE.copy(); q[[L_HIP, L_KNEE, L_ANK]] += x
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        return d.xpos[foot].copy(), d.xmat[foot].reshape(3, 3).copy()

    p0, R0 = fk(np.zeros(3))

    def resid(x, h):
        p, R = fk(x)
        r = R @ R0.T
        return np.r_[(p - p0 - [0, 0, h])[[0, 2]] * 10, (r[0, 2] - r[2, 0]) / 2]

    out, x = [], np.zeros(3)
    for h in hs:
        for _ in range(30):
            r = resid(x, h)
            J = np.stack([(resid(x + e, h) - r) / 1e-5 for e in np.eye(3) * 1e-5], 1)
            x = x - np.linalg.lstsq(J, r, rcond=None)[0]
        out.append(x.copy())
    return np.array(out)
