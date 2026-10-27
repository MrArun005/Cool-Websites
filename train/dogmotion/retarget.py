"""Retarget real dog motion capture onto Bolt: reference clips for imitation (imitate_env.py).

    python dogmotion/retarget.py --bvh-dir <folder with the MANN .bvh files>   # from train/, PYTHONPATH=.

The source is the dog motion capture of Zhang, Starke et al. 2018 (see SOURCES.md): a real dog in a
mocap suit, 60 frames/s, a 27-joint skeleton. This follows the keypoint retargeting of Peng et al. 2020
("Learning Agile Robotic Locomotion Skills by Imitating Animals"), adapted to Bolt:

1. Forward kinematics of the dog skeleton (bvh.py) gives every joint's position; resampled to 50 Hz.
2. The trunk: forward = shoulders' midpoint minus hips' midpoint, left = the average left-right
   direction of the shoulders and of the hips, made orthogonal. That rotation is Bolt's body rotation
   (so the dog's pitch and roll carry over), and the trunk's midpoint, scaled, is Bolt's body position.
3. Scale: Bolt's legs are 0.4 m long, the dog's front and rear legs differ. Each leg pair gets its own
   factor, chosen so the dog's standing shoulder/hip height maps onto Bolt's dog stance (hip 0.385 m
   above the paw centre, bolt_env STANCES["dog"]). Horizontal motion of the body uses their mean.
4. Each paw: target = Bolt's hip + the dog's hip-to-paw vector (scaled), except the height, which is
   the dog's paw height above the floor (scaled, plus the paw ball's radius), so planted paws stay on
   the floor even though Bolt's body proportions differ (Peng's trick).
5. Leg inverse kinematics (reference_gait.leg_ik): the sideways joint rolls the leg into the paw's
   plane, hip and knee solve the two-link triangle, front knees bend backward and rear knees forward
   like Bolt's. Angles are clamped to the joint limits (the amount clamped is reported).
6. Nothing but the paws may go into the floor: when the torso, head or a leg would (sitting, lying),
   the body is raised just enough and the legs are solved again (rump and elbows then rest on the
   floor, as on the real dog).
7. Light smoothing (Gaussian, 1 frame); looping clips have their start/end mismatch spread linearly
   over the cycle, and store how far the body moves and turns per cycle (to chain cycles).

Each clip is written to dogmotion/clips/<name>.npz with 50 Hz arrays:
    base_pos (T,3) m, base_quat (T,4) w x y z, joints (T,12) rad (package.JOINTS order), joint_vel (T,12),
    base_linvel (T,3) world m/s, base_angvel (T,3) world rad/s, foot_pos (T,4,3) world paw centres,
    foot_contact (T,4) planted paws, plus fps, loop, cycle_pos (3,), cycle_yaw, source, frames.
A kinematic check (MuJoCo, qpos set frame by frame) prints paw height errors, joint-limit clamping,
IK misses, floor and self penetration, and writes dogmotion/clips/check.json.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bvh  # noqa: E402
from bolt_env import FEET, LIE_POSE, MODEL, SIT_POSE, STANCES  # noqa: E402
from reference_gait import HIP_X, HIP_Y, LINK, leg_ik  # noqa: E402

FPS = 50
PAW_R = 0.025         # Bolt's paw ball radius (m)
HIP_Z = -0.03         # hips sit 3 cm below the body centre (bolt.xml)
STAND_REACH = 2 * LINK * np.cos(STANCES["dog"][1])   # hip to paw centre in the dog stance: 0.385 m

# Dog keypoints (bvh joint names) for Bolt's legs, in Bolt's order FL, FR, RL, RR.
DOG_HIPS = ["LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"]
DOG_TOES = ["LeftHand_end", "RightHand_end", "LeftFoot_end", "RightFoot_end"]

# The clips: (name, bvh file, start s, end s, loops). The looping segments are the ones Peng et al.
# picked for their dog (motion_imitation/retarget_motion: pace 162-201, trot 448-481, canter 430-459,
# left turn 1085-1124 in 60 Hz frames); sit and lie were picked by looking at body heights (survey in
# REPORT.md): stand ~1 s, go down, hold.
CLIPS = [
    ("walk", "D1_ex01_KAN01_001", 4.0, "cycle", True),          # steady ~1 m/s walk; one stride found automatically
    ("trot", "D1_009_KAN01_001", 448 / 60, 481 / 60, True),
    ("pace", "D1_007_KAN01_001", 162 / 60, 201 / 60, True),
    ("canter", "D1_010_KAN01_001", 430 / 60, 459 / 60, True),   # 3.4 m/s: kept for later, not trained by default
    # Peng et al. call this segment "left turn0"; in the mocap's own (right-handed, Z-up) frame the dog turns
    # clockwise, so it is our right turn, and its mirror image is the left turn.
    ("turn_right", "D1_053_KAN01_001", 1085 / 60, 1124 / 60, True),
    ("turn_left", "D1_053_KAN01_001", 1085 / 60, 1124 / 60, "mirror"),
    ("sit", "D1_013_KAN01_001", 7.5, 11.0, False),
    ("lie", "D1_061z_KAN01_003", 35.5, 41.0, False),
]
STAND_REF = ("D1_001_KAN01_001", 0.0, 10.0)   # a dog standing still: calibrates scale and paw height


def mirror(p, names):
    """The mirror image of a motion (left <-> right): flip y and swap Left*/Right* joints."""
    p = p.copy()
    p[..., 1] *= -1
    swap = [names.index(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right")) for n in names]
    return p[:, swap]


def find_cycle(p, names, t0, dt, lo=0.3, hi=1.2):
    """One stride of a steady gait near t0: the period and start whose end pose best matches the start
    (body-relative paw positions). Returns (start s, end s)."""
    hips_mid = p[:, idx(names, ["LeftUpLeg", "RightUpLeg"])].mean(1)
    rel = p[:, idx(names, DOG_TOES)] - hips_mid[:, None]
    rel[..., :2] -= rel[..., :2].mean(1, keepdims=True)
    a0 = int(t0 / dt)
    best = (np.inf, 0, 0)
    for a in range(a0, a0 + int(0.5 / dt)):
        for per in range(int(lo / dt), int(hi / dt)):
            if a + per >= len(p):
                break
            err = np.abs(rel[a] - rel[a + per]).mean() + 0.002 * per * dt   # prefer the shortest period
            if err < best[0]:
                best = (err, a, per)
    return best[1] * dt, (best[1] + best[2]) * dt


def keypoints(path, t0=None, t1=None, fps=FPS, mirrored=False):
    """Dog joint positions (T, J, 3), Z up, metres, resampled to `fps` from t0 to t1 inclusive.
    t1="cycle": one stride starting near t0 (find_cycle)."""
    joints, frames, dt = bvh.load(path)
    p = bvh.world_positions(joints, frames)
    names = [j.name for j in joints]
    if mirrored:
        p = mirror(p, names)
    if t1 == "cycle":
        t0, t1 = find_cycle(p, names, t0, dt)
    t_src = np.arange(len(p)) * dt
    t0 = 0.0 if t0 is None else t0
    t1 = t_src[-1] if t1 is None else t1
    t = np.arange(t0, t1 + 1e-9, 1.0 / fps)
    t = t[t <= t_src[-1]]
    out = np.empty((len(t), p.shape[1], 3))
    for j in range(p.shape[1]):
        for a in range(3):
            out[:, j, a] = np.interp(t, t_src, p[:, j, a])
    return out, names, (t0, t1)


def idx(names, which):
    return [names.index(n) for n in which]


def calibrate(bvh_dir):
    """Scale per leg pair and the dog's paw height when planted, from a dog standing still."""
    k, names, _ = keypoints(Path(bvh_dir) / f"{STAND_REF[0]}.bvh", STAND_REF[1], STAND_REF[2])
    hips, toes = k[:, idx(names, DOG_HIPS)], k[:, idx(names, DOG_TOES)]
    ground = np.median(toes[..., 2], axis=0)                # planted paw keypoint height, per leg
    ground = np.array([ground[:2].mean()] * 2 + [ground[2:].mean()] * 2)
    reach = np.median(hips[..., 2] - toes[..., 2], axis=0)  # standing hip-over-paw height
    scale = np.array([reach[:2].mean()] * 2 + [reach[2:].mean()] * 2)
    scale = STAND_REACH / scale
    # A dog stands with its paws closer together than its shoulders/hips (under the midline); Bolt's
    # thighs would hit its body if its legs leaned in that far. So the paws' sideways offset from the hip
    # when standing is removed: a standing dog maps to Bolt's paws straight under its hips, and only the
    # dog's sideways movement relative to that is copied.
    R, _, _, _ = trunk(k, names)
    lat = np.median(np.einsum("tj,tlj->tl", R[:, :, 1], toes - hips), axis=0)
    lat = np.array([0.5 * (lat[0] - lat[1])] * 2 + [0.5 * (lat[2] - lat[3])] * 2) * np.array([1, -1, 1, -1])
    return dict(scale=scale, ground=ground, lateral=lat)


def trunk(k, names):
    """Body rotation (T,3,3) and trunk centre (T,3) of the dog."""
    hip_mid = k[:, idx(names, ["LeftUpLeg", "RightUpLeg"])].mean(1)
    sh_mid = k[:, idx(names, ["LeftArm", "RightArm"])].mean(1)
    fwd = sh_mid - hip_mid
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    l_sh = k[:, names.index("LeftArm")] - k[:, names.index("RightArm")]
    l_hp = k[:, names.index("LeftUpLeg")] - k[:, names.index("RightUpLeg")]
    left = l_sh / np.linalg.norm(l_sh, axis=1, keepdims=True) + l_hp / np.linalg.norm(l_hp, axis=1, keepdims=True)
    left -= np.sum(left * fwd, 1, keepdims=True) * fwd
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    up = np.cross(fwd, left)
    return np.stack([fwd, left, up], -1), 0.5 * (hip_mid + sh_mid), hip_mid, sh_mid


def smooth(x, sigma=1.0, wrap=False):
    """Gaussian smoothing along time (axis 0)."""
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    if wrap:
        xp = np.concatenate([x[-r:], x, x[:r]])
    else:
        xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return sum(w[i] * xp[i:i + len(x)] for i in range(2 * r + 1))


class Bolt:
    """Bolt's kinematics in MuJoCo, to place the body, check the floor and verify the result."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.lo, self.hi = self.m.jnt_range[1:13, 0].copy(), self.m.jnt_range[1:13, 1].copy()
        self.foot = [self.m.geom(f).id for f in FEET]
        self.floor = self.m.geom("floor").id
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in self.m.geom_bodyid]
        self.geom_leg = np.array([["FL", "FR", "RL", "RR"].index(n[:2]) if n[:2] in ("FL", "FR", "RL", "RR") else -1
                                  for n in names])

    def set(self, pos, quat, q):
        self.d.qpos[:3], self.d.qpos[3:7], self.d.qpos[7:19] = pos, quat, q
        mujoco.mj_kinematics(self.m, self.d)

    def lowest_nonfoot(self):
        """Lowest point (m above the floor) of any solid geom except the paws."""
        m, d, low = self.m, self.d, np.inf
        for g in range(m.ngeom):
            if g == self.floor or g in self.foot or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R, c, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                z = c[2] - np.abs(R[2]) @ s
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = c[2] - abs(R[2, 2]) * s[1] - s[0]
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = c[2] - s[0]
            else:
                continue
            low = min(low, z)
        return low

    def leg_self_contacts(self):
        """Legs (0-3) with a geom touching the body or another leg."""
        mujoco.mj_collision(self.m, self.d)
        legs = set()
        for c in self.d.contact[:self.d.ncon]:
            if self.floor in (c.geom1, c.geom2) or -c.dist < 0.001:
                continue
            for g in (c.geom1, c.geom2):
                leg = self.geom_leg[g]
                if leg >= 0:
                    legs.add(int(leg))
        return legs

    def contacts(self):
        """(deepest floor penetration by a non-paw geom, deepest self penetration) in m (positive = into)."""
        mujoco.mj_forward(self.m, self.d)
        floor_pen = self_pen = 0.0
        for c in self.d.contact[:self.d.ncon]:
            pen = -c.dist
            if self.floor in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.floor else c.geom1
                if other not in self.foot:
                    floor_pen = max(floor_pen, pen)
            else:
                self_pen = max(self_pen, pen)
        return floor_pen, self_pen


# A leg may roll in under the body by at most 0.1 rad when retargeting: the thigh is only 6 mm from the
# torso's side (a dog's paws come in under its midline; Bolt's would cut through its body).
INWARD = 0.1


def comfort_limits(lo, hi):
    lo, hi = lo.copy(), hi.copy()
    for leg in range(4):
        if leg in (0, 2):    # left legs: inward is negative abduction
            lo[3 * leg] = max(lo[3 * leg], -INWARD)
        else:
            hi[3 * leg] = min(hi[3 * leg], INWARD)
    return lo, hi


# Folding down. A dog folds its legs flat under it to sit and lie, partly with its long paws, which Bolt
# does not have: following the dog's paws there wedged Bolt's knees under its body and held it 10 cm up.
# So as the dog's shoulders or hips come down from FOLD_FROM to FOLD_FROM - FOLD_FULL above its paws
# (scaled), that end's legs blend into Bolt's own measured fold (bolt_env SIT_POSE / LIE_POSE: rear legs
# as in the sit when the body is pitched nose-up, as in the lie when it is level), and the body rests on them.
FOLD_FROM, FOLD_FULL = 0.22, 0.12


def folded(q, fold, pitch):
    """Blend the IK legs into Bolt's own folded legs. fold = (front, rear) in 0-1; pitch nose-up (rad)."""
    up = float(np.clip(-pitch / SIT_PITCH, 0.0, 1.0))   # euler pitch is nose-down positive
    template = LIE_POSE.copy()
    template[6:] = (1 - up) * LIE_POSE[6:] + up * SIT_POSE[6:]
    q = q.copy()
    for leg in range(4):
        w = fold[0] if leg < 2 else fold[1]
        q[3 * leg:3 * leg + 3] = (1 - w) * q[3 * leg:3 * leg + 3] + w * template[3 * leg:3 * leg + 3]
    return q


SIT_PITCH = 0.54   # bolt_env TARGETS["sit"]["pitch"], measured
MAX_PITCH = 0.6


def untangle(bolt, pos, quat_, q):
    """Legs that cut into the body (a deeply folded leg rolled sideways swings its thigh up into the
    torso): first straighten the roll, then unfold the hip a little at a time, until MuJoCo finds no
    contact. Returns the fixed joints and how much was changed (rad)."""
    q0 = q.copy()
    for step in range(30):
        bolt.set(pos, quat_, q)
        legs = bolt.leg_self_contacts()
        if not legs:
            break
        for leg in legs:
            a, h = 3 * leg, 3 * leg + 1
            if abs(q[a]) > 0.02:
                q[a] *= 0.6
            else:
                q[h] += 0.05 * np.sign(STANCES["dog"][h] - q[h])
    return q, float(np.abs(q - q0).max())


def solve_legs(R, c, targets, lo, hi):
    """Joint angles putting each paw centre at its world target; also the clamped amount (limits and
    reach) in radians and metres."""
    q, clamp, short = np.zeros(12), 0.0, 0.0
    for leg in range(4):
        p = R.T @ (targets[leg] - c) - np.array([HIP_X[leg], HIP_Y[leg], HIP_Z])
        x, y, z = p
        raw_abd = np.arctan2(y, -z)
        short = max(short, float(np.linalg.norm(p)) - 2 * LINK)
        # roll first (clamped), then solve hip and knee in the leg's rolled plane: a clamped roll then
        # leaves the paw a little sideways of its target instead of pushing it into the floor
        abd = float(np.clip(raw_abd, lo[3 * leg], hi[3 * leg]))
        ca, sa = np.cos(abd), np.sin(abd)
        hip, knee = leg_ik(x, -(ca * -z + sa * y), front=leg < 2)
        raw = np.array([raw_abd, hip, knee])
        cl = np.clip(raw, lo[3 * leg:3 * leg + 3], hi[3 * leg:3 * leg + 3])
        clamp = max(clamp, float(np.abs(raw - cl).max()))
        q[3 * leg:3 * leg + 3] = cl
    return q, clamp, short


def quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def euler(R):
    """yaw, pitch, roll with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return np.arctan2(R[1, 0], R[0, 0]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[2, 1], R[2, 2])


def from_euler(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return _rotz(yaw) @ Ry @ Rx


def retarget(bvh_dir, name, file, t0, t1, loop, cal, bolt):
    mirrored = loop == "mirror"
    loop = bool(loop)
    k, names, (t0, t1) = keypoints(Path(bvh_dir) / f"{file}.bvh", t0, t1, mirrored=mirrored)
    k = smooth(k, 1.0)
    Rs, centre, hip_mid, sh_mid = trunk(k, names)
    hips, toes = k[:, idx(names, DOG_HIPS)], k[:, idx(names, DOG_TOES)]
    s, ground = cal["scale"], cal["ground"]
    s_body = s.mean()
    f0 = Rs[0][:, 0]                      # start at the origin, facing +x
    Rz = _rotz(-np.arctan2(f0[1], f0[0]))
    xy0 = centre[0, :2].copy()

    def level(p):
        p = p.copy()
        p[..., :2] -= xy0
        return p @ Rz.T

    # A. the body and the paw targets, frame by frame
    T = len(k)
    hip_off = np.array([[HIP_X[i], 0.0, HIP_Z] for i in (0, 2)])   # front, rear hip midpoints (body)
    root = np.zeros((T, 6))              # x y z yaw pitch roll
    targets = np.zeros((T, 4, 3))
    drop = np.zeros(T)
    fold = np.zeros((T, 2))              # how far the front and rear legs are folded under the body (0-1)
    for t in range(T):
        R = Rz @ Rs[t]
        c = level(centre[t:t + 1])[0]
        c[:2] *= s_body
        # body height: Bolt's hips at the dog's (scaled) shoulder and hip heights, on average
        zf, zr = s[0] * (sh_mid[t, 2] - ground[0]), s[2] * (hip_mid[t, 2] - ground[2])
        c[2] = 0.5 * (zf + zr) + PAW_R - 0.5 * (R @ hip_off[0] + R @ hip_off[1])[2]
        fold[t] = np.clip((FOLD_FROM - np.array([zf, zr])) / FOLD_FULL, 0.0, 1.0)
        left = R[:, 1].copy()
        left[2] = 0.0
        left /= np.linalg.norm(left)
        for leg in range(4):
            hip_w = c + R @ np.array([HIP_X[leg], HIP_Y[leg], HIP_Z])
            rel = level(toes[t, leg:leg + 1])[0] - level(hips[t, leg:leg + 1])[0] - cal["lateral"][leg] * left
            targets[t, leg, :2] = hip_w[:2] + s[leg] * rel[:2]
            targets[t, leg, 2] = max(PAW_R, s[leg] * (toes[t, leg, 2] - ground[leg]) + PAW_R)
            # A planted paw out of the leg's reach (the dog strides with straighter legs than Bolt's 0.4 m
            # allow): lower the body until it is reachable, with 1.5 cm to spare.
            if targets[t, leg, 2] < PAW_R + 0.01:
                dxy = np.linalg.norm(targets[t, leg, :2] - hip_w[:2])
                reach = 2 * LINK - 0.015
                if dxy < reach:
                    drop[t] = max(drop[t], (hip_w[2] - targets[t, leg, 2]) - np.sqrt(reach ** 2 - dxy ** 2))
        root[t, :3] = c
        root[t, 3:] = euler(R)
    root[:, 3] = np.unwrap(root[:, 3])
    # A sitting dog's back is steeper (~47 degrees) than Bolt can sit: its rump is 6 cm behind its hips and
    # its folded rear legs would lift their paws off the floor. Nose-up pitch is capped a little above
    # Bolt's own measured sit.
    root[:, 4] = np.maximum(root[:, 4], -MAX_PITCH)
    # lower smoothly: a running max over 0.14 s, then a Gaussian, so the body does not bob frame to frame
    drop = np.array([drop[max(0, t - 3):t + 4].max() for t in range(T)])
    drop = smooth(drop[:, None], 2.0, wrap=loop)[:, 0]
    root[:, 2] -= drop

    # B. paws relative to the body's heading (x forward, y left, z up from the floor), so a loop can be
    # closed and everything smoothed in the space the motion happens in, before solving the legs
    rel = np.zeros_like(targets)
    for t in range(T):
        rel[t] = (targets[t] - [root[t, 0], root[t, 1], 0.0]) @ _rotz(root[t, 3])
    if loop:   # the last frame is the next cycle's first: spread the mismatch over the cycle
        ramp = np.linspace(0, 1, T)
        rel -= ramp[:, None, None] * (rel[-1] - rel[0])
        root[:, 2] -= ramp * (root[-1, 2] - root[0, 2])
        root[:, 4:] -= ramp[:, None] * (root[-1, 4:] - root[0, 4:])
        d = root[-1] - root[0]
        cycle_yaw = float(d[3])
        cycle_pos = np.array([*(_rotz(-root[0, 3]) @ [d[0], d[1], 0.0])[:2], 0.0])   # in the start heading frame
        root, rel, fold = root[:-1], rel[:-1], fold[:-1]
        T -= 1
        # smooth with wrap: unroll the per-cycle drift first
        drift = np.linspace(0, 1, T, endpoint=False)[:, None] * np.r_[d[:2], 0, d[3], 0, 0]
        root = smooth(root - drift, 1.0, wrap=True) + drift
        rel = smooth(rel.reshape(T, -1), 1.0, wrap=True).reshape(T, 4, 3)
    else:
        cycle_pos, cycle_yaw = np.zeros(3), 0.0
        fold = smooth(fold, 2.0)
        root = smooth(root, 1.0)
        rel = smooth(rel.reshape(T, -1), 1.0).reshape(T, 4, 3)
    rel[..., 2] = np.maximum(rel[..., 2], PAW_R)
    targets = np.stack([rel[t] @ _rotz(root[t, 3]).T + [root[t, 0], root[t, 1], 0.0] for t in range(T)])

    # C. legs by IK; raise the body if anything but the paws would go into the floor
    lo, hi = comfort_limits(bolt.lo, bolt.hi)
    out = dict(base_pos=np.zeros((T, 3)), base_quat=np.zeros((T, 4)), joints=np.zeros((T, 12)))
    stats = dict(clamp=0.0, short=0.0, lifted=0.0, lowered=float(drop.max()), untangled=0.0)
    for t in range(T):
        R, c, lift = from_euler(*root[t, 3:]), root[t, :3].copy(), 0.0
        for _ in range(8):
            q, clamp, short = solve_legs(R, c, targets[t], lo, hi)
            q = folded(q, fold[t], root[t, 4])
            q, fix = untangle(bolt, c, quat(R), q)
            stats["untangled"] = max(stats["untangled"], fix)
            bolt.set(c, quat(R), q)
            low = bolt.lowest_nonfoot()
            if low >= 0.004:
                break
            c[2] += 0.004 - low
            lift += 0.004 - low
        stats["clamp"], stats["short"] = max(stats["clamp"], clamp), max(stats["short"], short)
        stats["lifted"] = max(stats["lifted"], lift)
        out["base_pos"][t], out["base_quat"][t], out["joints"][t] = c, quat(R), q
    if stats["lifted"] > 0:   # a lift is a step change: smooth the height, then solve the legs again
        z = smooth(out["base_pos"][:, 2:3], 2.0, wrap=loop)[:, 0]
        out["base_pos"][:, 2] = np.maximum(z, out["base_pos"][:, 2] - 0.002)
        for t in range(T):
            R = _mat(out["base_quat"][t])
            q = folded(solve_legs(R, out["base_pos"][t], targets[t], lo, hi)[0], fold[t], root[t, 4])
            out["joints"][t] = untangle(bolt, out["base_pos"][t], out["base_quat"][t], q)[0]

    # D. velocities (world frame), from the final poses
    dt = 1.0 / FPS
    pos, qt = out["base_pos"], out["base_quat"]
    if loop:
        R0 = _rotz(cycle_yaw)
        start_rot = _rotz(root[0, 3])
        nxt_pos = np.concatenate([pos[1:], [pos[0] + start_rot @ cycle_pos]])
        nxt_q = np.concatenate([qt[1:], [quat(R0 @ _mat(qt[0]))]])
        nxt_j = np.concatenate([out["joints"][1:], out["joints"][:1]])
    else:
        nxt_pos, nxt_q = np.concatenate([pos[1:], pos[-1:]]), np.concatenate([qt[1:], qt[-1:]])
        nxt_j = np.concatenate([out["joints"][1:], out["joints"][-1:]])
    out["base_linvel"] = (nxt_pos - pos) / dt
    out["joint_vel"] = (nxt_j - out["joints"]) / dt
    angvel = np.zeros((T, 3))
    for t in range(T):
        dq, neg, v = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(neg, qt[t])
        mujoco.mju_mulQuat(dq, nxt_q[t], neg)        # world-frame rotation from t to t+1
        mujoco.mju_quat2Vel(v, dq, dt)
        angvel[t] = v
    out["base_angvel"] = angvel
    for key in ("base_linvel", "joint_vel", "base_angvel"):   # central differences
        if loop:
            out[key] = 0.5 * (out[key] + np.roll(out[key], 1, 0))
        else:
            out[key][1:] = 0.5 * (out[key][1:] + out[key][:-1])

    # E. kinematic check in MuJoCo (qpos set frame by frame)
    feet, foot_err, floor_pen, self_pen = np.zeros((T, 4, 3)), [], [], []
    for t in range(T):
        bolt.set(pos[t], qt[t], out["joints"][t])
        feet[t] = [bolt.d.geom_xpos[g] for g in bolt.foot]
        fp, sp = bolt.contacts()
        floor_pen.append(fp)
        self_pen.append(sp)
        foot_err.append(np.linalg.norm(feet[t] - targets[t], axis=1).max())
    planted = targets[..., 2] <= PAW_R + 0.01   # the dog's paw is on the floor
    contact = feet[..., 2] < PAW_R + 0.012
    h_err = (feet[..., 2] - PAW_R)[planted]
    out["foot_pos"], out["foot_contact"] = feet, contact
    span = T * dt
    speed = np.linalg.norm(cycle_pos[:2]) / span if loop else float(np.linalg.norm(pos[-1, :2] - pos[0, :2]) / span)
    check = dict(
        frames=T, seconds=round(span, 2), loop=loop, speed_mps=round(float(speed), 3),
        turn_rate=round(cycle_yaw / span, 3) if loop else 0.0,
        body_height_m=[round(float(pos[:, 2].min()), 3), round(float(pos[:, 2].max()), 3)],
        planted_paw_height_err_mm=dict(mean=round(1000 * float(np.abs(h_err).mean()), 1),
                                       min=round(1000 * float(h_err.min()), 1), max=round(1000 * float(h_err.max()), 1)),
        paw_target_miss_mm=dict(mean=round(1000 * float(np.mean(foot_err)), 1), max=round(1000 * float(np.max(foot_err)), 1)),
        joint_limit_clamp_rad=round(stats["clamp"], 3), out_of_reach_m=round(max(0.0, stats["short"]), 3),
        legs_untangled_rad=round(stats["untangled"], 3), body_lifted_m=round(stats["lifted"], 3), body_lowered_m=round(stats["lowered"], 3),
        floor_penetration_mm=round(1000 * max(floor_pen), 1), self_penetration_mm=round(1000 * max(self_pen), 1),
        self_contact_frames=int(np.sum(np.array(self_pen) > 0.002)),
    )
    meta = dict(fps=FPS, loop=loop, cycle_pos=cycle_pos, cycle_yaw=cycle_yaw, name=name,
                source=f"MANN dog mocap {file}.bvh {t0:.3f}-{t1:.3f} s{' mirrored' if mirrored else ''} "
                       "(CC BY-NC 4.0, Zhang/Starke et al. 2018)")
    return {**out, **meta}, check


def _mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh-dir", required=True, help="folder with the MANN dog .bvh files (MotionCapture.zip)")
    ap.add_argument("--out", default=str(HERE / "clips"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = calibrate(args.bvh_dir)
    print(f"scale per leg {np.round(cal['scale'], 3)}, planted paw keypoint height {np.round(cal['ground'], 3)} m")
    bolt, checks = Bolt(), {}
    for name, file, t0, t1, loop in CLIPS:
        clip, check = retarget(args.bvh_dir, name, file, t0, t1, loop, cal, bolt)
        np.savez_compressed(out / f"{name}.npz", **{k: np.asarray(v) for k, v in clip.items()})
        checks[name] = check
        print(name, json.dumps(check))
    (out / "check.json").write_text(json.dumps(dict(calibration={k: v.round(4).tolist() for k, v in cal.items()},
                                                    clips=checks), indent=2))


if __name__ == "__main__":
    main()
