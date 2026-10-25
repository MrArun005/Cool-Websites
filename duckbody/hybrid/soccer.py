"""Duck soccer: the hybrid duck plays by itself on a small pitch. It finds the ball with its head camera, walks
round behind it, lines up with the goal, kicks it with Microduck's ball_kick_left / ball_kick_right brains, chases
it and repeats; it gets up by itself when it falls. Score: goals in a 2-minute match.

    PY=/home/user/cool-websites/.venv/bin/python
    MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer.py --probe            # where the ball must be for a kick to connect
    MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer.py --seeds 0-9        # 10 matches (randomized bodies / ball starts)
    MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --soccer    # the soccer replay for the page

Pitch: 2.5 x 1.5 m, 15 cm boards (get-up rolls tumbled the duck over 6 cm ones), a goal 50 cm wide and 15 cm high
at the +x end: yellow posts and crossbar, light grey net. Ball: 7 cm, 20 g, rolling friction (from 1 m/s it rolls
~1.7 m). After a goal the duck hops and the ball goes back to the centre spot 3 s later.

The kicks (ball_kick_left / ball_kick_right, Microduck's, unchanged): episodic, 0.5 s window (manifest, cheatsheet),
all-zero command, blind (trained with the ball at 9 cm ahead of the trunk, +-4.2 cm sideways, +-1.5 cm noise). On this
duck (--probe, 3 bodies per cell): ball centre 6-9 cm ahead of the trunk and up to +-4 cm off the foot -> hit every
time, ~1.2 m/s, ~1.6 m in 3 s; 12 cm ahead -> 1-2/3; 15 cm and more -> never. Sideways offset steers it: ball in
line with the foot -> ~12 deg outwards, 2 cm inside -> ~15 deg inwards, 1 cm inside (the aim) -> about straight. And they only kick from a standing duck's pose: with the
head down (the foot view below) they do nothing, so the head goes home for 1 s first.

Perception (10 Hz, 160 x 120 head camera as follow.py):
    ball    magenta blob -> the ray through its centre meets the plane z = ball radius -> ball position relative to
            the trunk (camera pose from the head joints, trunk tilt from the IMU). Accurate to ~1 cm at the feet.
    goal    yellow pixels -> the lowest one at each side of the frame = the two post bases on the floor. The goal is
            a landmark with a known position AND direction, so one sighting gives position and heading
            (+-5 deg near, +-12 deg from the far end); blended in 30 %, sightings far from the belief are dropped.
    between sightings: heading = IMU gyro about gravity integrated at 200 Hz, position = dead reckoning from the
            twist sent to velstand through its measured speed map. A tumble (tilt > 60 deg) invalidates heading:
            the next goal sighting is taken whole.
Behaviour (50 Hz):
    search  turn on the spot towards where it was last seen, head nodding from far to the feet; step back first if
            it was lost up close; after 10 s walk towards the centre for a while
    approach / chase   aim point 24 cm behind the ball on the line ball -> goal centre (kept 12 cm off the boards); if
            not behind the ball, orbit it at 28 cm, 50 deg at a time, on the side with more room
    align   head down (neck -0.6, pitch 1.1: the camera sees 5-15 cm in front of the feet); pick the foot on the
            ball's side (the board side near a board); step (velstand's dead band: >= 0.33 forward, 0.45 sideways,
            0.6 turning) until the ball is at (8, +-3.2) cm and the heading within 0.15 rad, held 0.3 s
    ready   stand, head home 1 s; kick: the kick brain 0.5 s, then velstand; then chase
    dribble when it cannot line up within 8 s (ball against a board): walk into the ball for 2.5 s
    reflex  turning on the spot 1.5 s without the IMU yaw moving (against a board): back up 1 s
Falls: the runtime's get-up (velstand, unstick sequence). Scoring uses the simulator's truth (ball in the goal);
the duck itself never uses it.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_hybrid as rh  # noqa: E402
import follow as F  # noqa: E402
from harness import MicroduckSim, Policy  # noqa: E402

# ---------------------------------------------------------------------------------------------- the pitch
PITCH_X, PITCH_Y = 1.25, 0.75        # half sizes: 2.5 x 1.5 m
WALL_H, WALL_T = 0.15, 0.02          # boards: keep the ball in, and the duck (get-up rolls tumbled over 6 cm walls)
GOAL_W, GOAL_D, GOAL_H, POST = 0.50, 0.20, 0.15, 0.02   # mouth width, depth, height, post thickness
BALL_R, BALL_M = 0.035, 0.020        # 7 cm, 20 g (Microduck's kick task: 7 cm, 15 g hollow ball)
BALL_FRICTION = (0.5, 0.005, 0.002)   # sliding, torsional, rolling: from 1 m/s it rolls ~1.7 m in ~6 s
BALL_RGB = F.BALL_RGB                # magenta, as follow.py
GOAL_RGB = (1.0, 0.85, 0.05)         # yellow posts and crossbar: the camera finds the goal frame by colour
COLORS = dict(wall="#c8553d", goal="#ffd90d", net="#d9dde3", line="#f2f2f2", turf="#2f6b3a")


def pitch_boxes():
    """Static boxes of the pitch (world frame): walls with a gap for the goal, the goal (posts, crossbar, back,
    sides), lines (visual only)."""
    X, Y, t, h = PITCH_X, PITCH_Y, WALL_T, WALL_H
    gw = GOAL_W / 2

    def box(kind, x0, x1, y0, y1, z0, z1, visual=False):
        return dict(kind=kind, pos=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
                    size=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2], quat=[1, 0, 0, 0],
                    color=COLORS[kind if kind in COLORS else "wall"], visual=visual)
    b = [box("turf", -X, X, -Y, Y, -0.002, 0.0005, visual=True),
         box("wall", -X - t, X + t, Y, Y + t, 0, h), box("wall", -X - t, X + t, -Y - t, -Y, 0, h),   # sides
         box("wall", -X - t, -X, -Y, Y, 0, h),                                                       # own end
         box("wall", X, X + t, gw + POST, Y, 0, h), box("wall", X, X + t, -Y, -gw - POST, 0, h),     # goal end
         # goal: posts on the line, crossbar, back panel, side panels
         box("goal", X, X + POST, gw, gw + POST, 0, GOAL_H), box("goal", X, X + POST, -gw - POST, -gw, 0, GOAL_H),
         box("goal", X, X + POST, -gw - POST, gw + POST, GOAL_H, GOAL_H + POST),
         box("net", X + GOAL_D, X + GOAL_D + t, -gw - POST, gw + POST, 0, GOAL_H),
         box("net", X + POST, X + GOAL_D, gw, gw + POST, 0, GOAL_H), box("net", X + POST, X + GOAL_D, -gw - POST, -gw, 0, GOAL_H),
         # lines: halfway, centre spot, goal line
         box("line", -0.005, 0.005, -Y, Y, 0.0005, 0.0012, visual=True),
         box("line", X - 0.006, X, -gw, gw, 0.0005, 0.0012, visual=True)]
    return b


def scene_editor(cam=True):
    import mujoco

    def edit(spec):
        if cam:
            head = spec.body("jaw_soft")
            c = head.add_camera()
            c.name = "eye"
            c.pos, c.quat, c.fovy = list(F.EYE_POS), list(F.EYE_QUAT), F.FOVY
        for i, b in enumerate(pitch_boxes()):
            g = spec.worldbody.add_geom()
            g.name = f"pitch_{b['kind']}_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size, g.pos, g.quat = b["size"], b["pos"], b["quat"]
            c = b["color"].lstrip("#")
            rgb = [int(c[k:k + 2], 16) / 255 for k in (0, 2, 4)]
            if b["kind"] == "goal":
                rgb = list(GOAL_RGB)
            g.rgba = rgb + [1.0]
            g.friction = [0.6, 0.005, 0.0001]
            if b["visual"]:
                g.contype = g.conaffinity = 0
        ball = spec.worldbody.add_body()
        ball.name = "ball"
        ball.pos = [0.5, 0, BALL_R]
        ball.add_freejoint().name = "ball_free"
        g = ball.add_geom()
        g.name = "ball_geom"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [BALL_R, 0, 0]
        g.mass = BALL_M
        g.rgba = list(BALL_RGB) + [1.0]
        g.friction = list(BALL_FRICTION)
        g.condim = 6                 # rolling friction
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 640)
    return edit


class Pitch:
    """The sim with the pitch and a free ball; helpers for the ball state."""

    def __init__(self, seed=0, randomize=True, cam=True):
        self.sim = MicroduckSim(str(rh.SCENE), seed=seed, randomize=randomize, edit=scene_editor(cam))
        m = self.sim.m
        j = m.joint("ball_free")
        self.bq, self.bv = int(m.jnt_qposadr[j.id]), int(m.jnt_dofadr[j.id])
        self.ball_body = m.body("ball").id
        self.ball_geom = m.geom("ball_geom").id
        self.robot_geoms = {i for i in range(m.ngeom) if m.body_rootid[m.geom_bodyid[i]] == self.sim.trunk}

    def set_ball(self, x, y, vx=0.0, vy=0.0):
        import mujoco
        d = self.sim.d
        d.qpos[self.bq:self.bq + 7] = [x, y, BALL_R, 1, 0, 0, 0]
        d.qvel[self.bv:self.bv + 6] = [vx, vy, 0, 0, 0, 0]
        mujoco.mj_forward(self.sim.m, d)

    def ball(self):
        d = self.sim.d
        return d.qpos[self.bq:self.bq + 3].copy(), d.qvel[self.bv:self.bv + 3].copy()

    def touching(self):
        """Robot geoms in contact with the ball now (names)."""
        d, m, out = self.sim.d, self.sim.m, []
        for c in d.contact[:d.ncon]:
            if c.geom1 == self.ball_geom and c.geom2 in self.robot_geoms:
                out.append(m.geom(c.geom2).name)
            elif c.geom2 == self.ball_geom and c.geom1 in self.robot_geoms:
                out.append(m.geom(c.geom1).name)
        return out


def trunk_pose(sim):
    p = sim.d.xpos[sim.trunk]
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    return p.copy(), math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------------------------- kick probe
def kick_trial(seed, foot, dx, dy, window=0.5, stand_s=1.5, after_s=3.0, gain=1.0, head_down=False, head_up_s=None):
    """Stand still, put the ball at (dx, dy) from the kicking foot's nominal spot (trunk frame: x forward,
    0.09 m ahead of the trunk, y = +-0.042), run the kick brain for `window` s, then velstand. Returns the
    ball's speed and direction, the distance it rolled, and whether the duck stayed up."""
    import mujoco
    P = Pitch(seed=seed, randomize=True, cam=False)
    sim = P.sim
    rt = rh.HybridRuntime(sim=sim, seed=seed, log=None)
    rt.reset("standing")
    sim.d.qpos[0:2] = [-0.6, 0.0]
    sim.d.qpos[3:7] = [1, 0, 0, 0]
    P.set_ball(0.6, 0.6)
    mujoco.mj_forward(sim.m, sim.d)
    if head_down:                                   # kick straight from the foot view (head not back at home)
        rt.look(0.0, FOOT_VIEW[1], neck=FOOT_VIEW[0])
    rt.run(stand_s)
    p, yaw = trunk_pose(sim)
    fy = 0.042 if foot == "left" else -0.042
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = 0.09 + dx, fy + dy
    P.set_ball(p[0] + c * bx - s * by, p[1] + s * bx + c * by)
    if head_up_s:                                   # from the foot view: head back home this long, then kick
        rt.look(0.0, 0.0)
        rt.run(head_up_s)
    b0, _ = P.ball()
    pol = Policy(f"ball_kick_{foot}")
    kp0 = sim.bam.model.actuator.kp
    sim.bam.model.actuator.kp = kp0 * gain
    touched, vmax, vdir = set(), 0.0, None
    sim.command[:] = 0
    for _ in range(int(window * rh.HZ)):
        sim.step(pol)
        t = P.touching()
        touched.update(t)
        _, v = P.ball()
        sp = float(np.hypot(v[0], v[1]))
        if sp > vmax:
            vmax, vdir = sp, wrap(math.atan2(v[1], v[0]) - yaw)
    sim.bam.model.actuator.kp = kp0
    rt._start("velstand")
    rt.mode = "stand"
    up = True
    for _ in range(int(after_s * rh.HZ)):
        rt.tick()
        t = P.touching()
        _, v = P.ball()
        sp = float(np.hypot(v[0], v[1]))
        if sp > vmax and not t:
            vmax, vdir = sp, wrap(math.atan2(v[1], v[0]) - yaw)
        up = up and not rt.lying()
    b1, _ = P.ball()
    dd = b1[:2] - b0[:2]
    return dict(seed=seed, foot=foot, dx=dx, dy=dy, window=window, touched=sorted(touched), hit=vmax > 0.15,
                v=round(vmax, 3), dir_deg=round(math.degrees(vdir), 1) if vdir is not None else None,
                rolled=round(float(np.hypot(*dd)), 3), fwd=round(float(dd[0] * c + dd[1] * s), 3), up=up and rt.falls == 0)


def _probe_job(a):
    return kick_trial(*a)


PROBE_DX = (-0.03, 0.0, 0.03, 0.06, 0.09)
PROBE_DY = (-0.04, -0.02, 0.0, 0.02, 0.04)


def probe(jobs=3, seeds=(0, 1, 2)):
    grid = [(s, foot, dx, dy) for foot in ("left", "right") for dx in PROBE_DX for dy in PROBE_DY for s in seeds]
    with ProcessPoolExecutor(jobs) as ex:
        runs = list(ex.map(_probe_job, grid))
    return runs


# ---------------------------------------------------------------------------------------------- perception
W, H, FOVY, CAM_HZ = F.W, F.H, F.FOVY, F.CAM_HZ
FY = (H / 2) / math.tan(math.radians(FOVY) / 2)


def detect_goal(img_rgb):
    """Yellow goal: HSV threshold, the union of the yellow blobs. Returns dict(u0, u1, v1, left=(u, v), right=(u, v))
    with the lowest pixel near each side of the goal (the post bases), or None."""
    import cv2
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (22, 120, 80), (36, 255, 255))     # no opening: far away a post is 1 px wide
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    u0, u1 = int(xs.min()), int(xs.max())
    if u1 - u0 < 4:
        return None
    side = max(2, (u1 - u0) // 8)
    lm, rm = xs <= u0 + side, xs >= u1 - side
    li, ri = np.argmax(np.where(lm, ys, -1)), np.argmax(np.where(rm, ys, -1))
    return dict(u0=u0, u1=u1, v1=int(ys.max()), n=len(xs), left=(float(xs[li]), float(ys[li])),
                right=(float(xs[ri]), float(ys[ri])), edge=u0 <= 1 or u1 >= W - 2)


def pixel_ground(sim, cam_id, u, v, z=0.0):
    """Ground point (height z) seen at pixel (u, v), relative to the trunk in the trunk's yaw frame (x forward,
    y left). Camera pose from the head joints' kinematics, trunk tilt from the IMU (on the robot: encoders + IMU)."""
    ray = sim.d.cam_xmat[cam_id].reshape(3, 3) @ np.array([(u - W / 2) / FY, -(v - H / 2) / FY, -1.0])
    c = sim.d.cam_xpos[cam_id]
    if ray[2] > -1e-3:
        return None
    s = (z - c[2]) / ray[2]
    g = c + s * ray - sim.d.xpos[sim.trunk]
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([cy * g[0] + sy * g[1], -sy * g[0] + cy * g[1]])


# ---------------------------------------------------------------------------------------------- the player
# velstand's measured speed for a drive command (soccer.py calibration on 2 randomized bodies; it does not step
# below ~0.25 forward / ~0.3 sideways / ~0.5 turning): used for dead reckoning only
VX_MAP = ((-0.3, -0.07), (-0.25, 0.0), (0.25, 0.0), (0.3, 0.11), (0.5, 0.23), (0.8, 0.35), (1.0, 0.41))
VY_MAP = ((-0.4, -0.058), (-0.3, 0.0), (0.3, 0.0), (0.4, 0.058))
GOAL_C = np.array([PITCH_X, 0.0])        # aim point: the middle of the goal mouth
POSTS_MID = np.array([PITCH_X + POST / 2, 0.0])
FIX_GAIN = 0.3                           # a goal sighting moves the pose estimate this far towards what it measured
BEHIND = 0.24                            # approach point: this far behind the ball on the line ball -> goal
AROUND = 0.28                            # going round the ball: pass this far to its side
KICK_X, KICK_Y = 0.08, 0.032             # ball centre in the trunk frame for a straight kick (probe: x 0.06-0.10)
TOL_X, TOL_Y, TOL_H = 0.02, 0.02, 0.15   # alignment tolerances (m, m, rad); probe: +-2 cm sideways = +-15 deg
LOST_S = 0.7
LOST_S_LONG = 4.0
ALIGN_MAX_S = 8.0                        # going round a still ball it may be out of view this long
READY_S = 1.0                            # head back home before a kick: from the foot view the kick brains do
                                         # not kick at all (their obs is far from a standing duck's), 0.6 s: 2/4
FOOT_VIEW = (-0.6, 1.1)                  # neck, head pitch: the camera sees the ball at the feet (5-15 cm)
CAM_H = 0.21


def interp(m, x):
    xs, ys = zip(*m)
    return float(np.interp(x, xs, ys))


class Player:
    """The duck's soccer brain: perception (camera at 10 Hz), estimation (IMU heading, dead reckoning, goal
    sightings), and the behaviour state machine search -> approach -> align -> kick -> chase."""

    def __init__(self, rt, cam_id, psi0=0.0, pos0=(0.0, 0.0)):
        self.rt, self.sim, self.cam_id = rt, rt.sim, cam_id
        self.psi = psi0                          # heading, field frame (IMU gyro about gravity, integrated)
        self.pos = np.array(pos0, float)         # dead-reckoned position, fixed by goal sightings
        self.ball_f = np.array([0.0, 0.0])       # ball in the field frame (last sighting)
        self.ball_seen_t = -9.0
        self.goal_seen_t = -9.0
        self.state, self.state_t = "search", 0.0
        self.foot, self.ok_t, self.kicks = None, 0.0, 0
        self.search_dir, self.search_turned = 1.0, 0.0
        self.t = 0.0
        self.last_det = self.last_goal = None
        self.misses = 0
        self.turn_hist, self.backup_until, self.backups = [], -1.0, 0
        self.pose_lost, self.fixes, self.rejects, self.reject_run = False, 0, 0, 0
        rt.on_substep = self._gyro
        self.events = []

    def set_state(self, s):
        if s != self.state:
            self.state, self.state_t = s, self.t

    # -------------------------------------------------------------- estimation
    def _gyro(self, sim):
        """Every physics step (200 Hz, like an IMU's own integration; at 50 Hz the stepping jolts alias into a
        drift of ~0.1 rad/min): yaw rate about gravity = gyro . up, both in the trunk frame from the IMU."""
        up = -sim.gravity()
        self.psi = wrap(self.psi + float(sim.d.sensordata[sim.gyro_adr] @ up) * rh.DT / 4)

    def imu(self):
        rt = self.rt
        if rt.mode == "drive" and not rt.lying():
            vx, vy = interp(VX_MAP, rt.vel[0]), interp(VY_MAP, rt.vel[1])
            c, s = math.cos(self.psi), math.sin(self.psi)
            self.pos += np.array([c * vx - s * vy, s * vx + c * vy]) * rh.DT
        # the boards: the trunk cannot be closer than ~6 cm to them (walking into a board moves nothing); shift the
        # ball estimate along so what the duck saw stays where it was relative to it
        lim = np.array([PITCH_X - 0.06, PITCH_Y - 0.06])
        corr = np.clip(self.pos, -lim, lim) - self.pos
        self.pos += corr
        self.ball_f += corr

    def to_field(self, rel):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return self.pos + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])

    def to_body(self, p):
        d = np.asarray(p) - self.pos
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def expect_visible(self):
        """Would the ball be in the picture if it were where we think? (projection through the camera pose)"""
        sim = self.sim
        rel = self.to_body(self.ball_f)
        R = sim.d.xmat[sim.trunk].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        p = sim.d.xpos[sim.trunk] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], 0.0])
        p[2] = BALL_R
        pc = sim.d.cam_xmat[self.cam_id].reshape(3, 3).T @ (p - sim.d.cam_xpos[self.cam_id])
        if pc[2] > -0.03:
            return False
        u, v = -pc[0] / pc[2] * FY, pc[1] / pc[2] * FY
        return abs(u) < W / 2 - 6 and abs(v) < H / 2 - 6

    def see(self, img):
        det = F.detect(img)
        self.last_det = det
        if det is not None:
            g = pixel_ground(self.sim, self.cam_id, det[0], det[1], BALL_R)
            if g is not None and np.hypot(*g) < 3.5:
                self.ball_f = self.to_field(g)
                self.ball_seen_t = self.t
                self.misses = 0
        elif self.expect_visible():
            self.misses += 1                      # it should be in view but is not: it has moved
        gd = detect_goal(img)
        self.last_goal = gd
        if gd is not None and not gd["edge"]:
            # the two post bases (lowest yellow pixel at each side of the frame) on the floor, in the trunk frame:
            # a landmark with a known position AND direction, so one sighting fixes heading and position
            pl = pixel_ground(self.sim, self.cam_id, *gd["left"])
            pr = pixel_ground(self.sim, self.cam_id, *gd["right"])
            if pl is not None and pr is not None and 0.35 < np.hypot(*((pl + pr) / 2)) < 2.2 and \
                    0.4 < np.hypot(*(pl - pr)) < 0.68:
                m_b, d_b = (pl + pr) / 2, pl - pr
                psi_m = wrap(math.pi / 2 - math.atan2(d_b[1], d_b[0]))          # field: left post -> right post = -y
                c, s = math.cos(psi_m), math.sin(psi_m)
                pos_m = POSTS_MID - np.array([c * m_b[0] - s * m_b[1], s * m_b[0] + c * m_b[1]])
                g = 1.0 if self.pose_lost else FIX_GAIN
                if not self.pose_lost and (abs(wrap(psi_m - self.psi)) > 0.5 or np.hypot(*(pos_m - self.pos)) > 0.6):
                    self.rejects += 1                    # far from what we believe: a bad view (ball in front of a post...)
                    self.reject_run += 1
                    if self.reject_run < 5:              # ...unless the views keep agreeing with each other: then the
                        return det, gd                   # belief is what is wrong, take the sighting whole
                    g = 1.0
                self.reject_run = 0
                self.psi = wrap(self.psi + g * wrap(psi_m - self.psi))
                self.pos = self.pos + g * (pos_m - self.pos)
                if self.pose_lost:
                    self.pose_lost = False
                    self.ball_seen_t = min(self.ball_seen_t, self.t - LOST_S_LONG - 1)   # the ball estimate was in the old frame
                self.goal_seen_t = self.t
                self.fixes += 1
        return det, gd

    # -------------------------------------------------------------- behaviour
    def plan(self):
        """Approach point behind the ball and the kick direction (field frame)."""
        b = np.clip(self.ball_f, [-PITCH_X + BALL_R, -PITCH_Y + BALL_R], [PITCH_X - BALL_R, PITCH_Y - BALL_R])
        u = GOAL_C - b
        u /= max(np.linalg.norm(u), 1e-6)
        P = b - BEHIND * u
        P = np.clip(P, [-PITCH_X + 0.12, -PITCH_Y + 0.12], [PITCH_X - 0.12, PITCH_Y - 0.12])
        k = b - P
        k /= max(np.linalg.norm(k), 1e-6)
        return P, k

    def go_to(self, target_f, vmax=0.9, face=None):
        """Twist to walk to a field point: turn towards it, walk when facing it (velstand's dead band respected)."""
        e = self.to_body(target_f)
        d, b = float(np.hypot(*e)), math.atan2(e[1], e[0])
        if face is not None and d < 0.06:
            b = wrap(face - self.psi)
        wz = float(np.clip(1.6 * b, -1.0, 1.0))
        if abs(b) > 0.12 and abs(wz) < 0.6:          # velstand does not turn below ~0.5
            wz = math.copysign(0.6, b)
        vx = 0.0
        if abs(b) < 0.5 and d > 0.04:
            vx = float(np.clip(0.3 + 1.2 * d, 0.32, vmax))
        if abs(b) > 0.9 and d < 0.3:            # close behind / beside: side-step instead of a wide turn
            vx = 0.0
        return vx, 0.0, wz, d

    def look_at_ball(self):
        rel = self.to_body(self.ball_f)
        d = float(np.hypot(*rel))
        bear = math.atan2(rel[1], rel[0])
        if d < 0.24:
            self.rt.look(float(np.clip(bear, -0.8, 0.8)), FOOT_VIEW[1], neck=FOOT_VIEW[0])
        else:
            elev = math.atan2(CAM_H - BALL_R, max(d - 0.04, 0.05))
            self.rt.look(bear, elev - F.CAM_DOWN)

    def step(self, t):
        """One 50 Hz tick of behaviour, with the blocked-turn reflex: turning on the spot for 1.5 s with < 0.25 rad
        of IMU yaw (a foot against a wall or the goal) -> back up 1 s."""
        vx, vy, wz = self._step(t)
        turning = abs(wz) >= 0.5 and vx <= 0.0 and self.rt.mode == "drive"
        self.turn_hist = (self.turn_hist + [(turning, self.psi)])[-75:]
        if len(self.turn_hist) == 75 and all(a for a, _ in self.turn_hist) and \
                abs(wrap(self.turn_hist[-1][1] - self.turn_hist[0][1])) < 0.25:
            self.backup_until, self.turn_hist = t + 1.0, []
            self.backups += 1
        if t < self.backup_until:
            return -0.5, 0.0, 0.0
        return vx, vy, wz

    def _step(self, t):
        self.t = t
        rt = self.rt
        if rt.mode == "get up" or rt.lying():
            if self.sim.tilt_deg() > 60:              # a tumble: IMU yaw about gravity is not the heading any more
                self.pose_lost = True
            self.set_state("get up")
            return 0.0, 0.0, 0.0
        if rt.mode in ("kick left", "kick right"):
            return 0.0, 0.0, 0.0
        if rt.mode == "hop" or self.state == "goal!":
            return 0.0, 0.0, 0.0
        if self.state in ("get up", "kick"):
            self.set_state("search" if t - self.ball_seen_t > LOST_S else "chase")
        # lost: not where we thought (4 frames it should have been in view), or not seen for a while
        lost = self.misses >= 4 or t - self.ball_seen_t > LOST_S_LONG
        rel = self.to_body(self.ball_f)
        dball = float(np.hypot(*rel))
        if lost and self.state not in ("search",):
            self.set_state("search")
            self.search_turned = 0.0
            self.search_dir = 1.0 if math.atan2(rel[1], rel[0]) >= 0 else -1.0
        if self.state == "search":
            if not lost:
                self.set_state("approach")
            else:
                ph = t - self.state_t
                nod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * ph)      # far (pitch 0.1) .. at the feet (0.9, neck -0.3)
                self.rt.look(0.6 * self.search_dir, 0.1 + 0.8 * nod, neck=-0.3 * nod)
                if dball < 0.3 and ph < 1.5:                 # lost it up close: step back, look down at the feet
                    self.rt.look(0.0, FOOT_VIEW[1], neck=FOOT_VIEW[0])
                    return -0.35, 0.0, 0.0
                if ph > 10 and ph % 10 < 3:                  # turned round without it: walk towards the middle
                    vx, vy, wz, _ = self.go_to(np.array([0.0, 0.0]), 0.8)
                    return vx, vy, wz
                return 0.0, 0.0, 0.9 * self.search_dir
        P, k = self.plan()
        n = np.array([-k[1], k[0]])
        r = self.pos - self.ball_f
        along, lat = float(r @ k), float(r @ n)
        if self.state in ("approach", "chase"):
            self.look_at_ball()
            self.set_state("chase" if dball > 0.6 else "approach")
            phi_d = math.atan2(r[1], r[0])                  # where the duck is, seen from the ball
            phi_p = math.atan2(-k[1], -k[0])                # where it should be: behind it
            dphi = wrap(phi_p - phi_d)
            if abs(dphi) > 0.6:                             # not behind the ball yet: orbit round it, ~50 deg ahead
                if abs(dphi) > 2.6:                         # straight in front: go round the side with more room
                    room = [min(PITCH_Y - abs((self.ball_f + sg * AROUND * n)[1]), PITCH_X - abs((self.ball_f + sg * AROUND * n)[0]))
                            for sg in (1.0, -1.0)]
                    dphi = abs(dphi) if room[0] >= room[1] else -abs(dphi)   # counter-clockwise passes the +n side
                a = phi_d + float(np.clip(dphi, -0.9, 0.9))
                rad = max(AROUND, min(dball, 0.5))
                target = self.ball_f + rad * np.array([math.cos(a), math.sin(a)])
                target = np.clip(target, [-PITCH_X + 0.1, -PITCH_Y + 0.1], [PITCH_X - 0.1, PITCH_Y - 0.1])
                vx, vy, wz, d = self.go_to(target)
            else:
                vx, vy, wz, d = self.go_to(P, face=math.atan2(k[1], k[0]))
                if d < 0.07 or (along < -0.12 and abs(lat) < 0.06 and dball < BEHIND + 0.06):
                    self.set_state("align")
                    self.foot = None
            return vx, vy, wz
        if self.state == "align":
            self.rt.look(0.0, FOOT_VIEW[1], neck=FOOT_VIEW[0])
            if dball > BEHIND + 0.15 or along > 0.0:
                self.set_state("approach")
                return 0.0, 0.0, 0.0
            if t - self.state_t > ALIGN_MAX_S:             # cannot line up (a ball against the boards): barge into it
                self.set_state("dribble")
                return 0.0, 0.0, 0.0
            if self.foot is None:
                by = rel[1]
                self.foot = "left" if by > 0 else "right"
                if self.ball_f[1] > PITCH_Y - 0.15:          # by a side wall: keep the body on the open side
                    self.foot = "left"
                elif self.ball_f[1] < -PITCH_Y + 0.15:
                    self.foot = "right"
            ty = KICK_Y if self.foot == "left" else -KICK_Y
            ex, ey = rel[0] - KICK_X, rel[1] - ty
            eh = wrap(math.atan2(k[1], k[0]) - self.psi)
            fresh = t - self.ball_seen_t < 0.25
            vx = vy = wz = 0.0
            if abs(eh) > 0.3:
                wz = math.copysign(0.65, eh)
                if ex > 0.08:
                    vx = 0.33
            else:
                if abs(ex) > TOL_X:
                    vx = 0.33 if ex > 0 else -0.4
                if abs(ey) > TOL_Y:
                    vy = 0.45 if ey > 0 else -0.45
                if abs(eh) > TOL_H:
                    wz = math.copysign(0.6, eh)
            if vx == vy == wz == 0.0 and fresh:
                self.ok_t += rh.DT
            else:
                self.ok_t = 0.0
            if self.ok_t > 0.3:
                self.set_state("ready")
                self.ok_t = 0.0
                self.kick_at = dict(foot=self.foot, ex=round(float(ex), 3), ey=round(float(ey), 3), eh=round(float(eh), 3))
            return vx, vy, wz
        if self.state == "dribble":
            self.look_at_ball()
            if t - self.state_t > 2.5:
                self.set_state("approach")
            vx, vy, wz, _ = self.go_to(self.ball_f + 0.1 * k, 0.6)
            return vx, vy, wz
        if self.state == "ready":                    # stand still, head home, then the kick brain
            self.rt.look(0.0, 0.0)
            if t - self.state_t >= READY_S:
                self.set_state("kick")
                self.kicks += 1
                self.kick_at["t"] = round(t, 2)
                rt.command(f"kick {self.foot}")
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------------------------- a match
T_MATCH = 120.0


def kickoff(seed):
    """Random start: the duck on its own half facing the goal (+-0.3 rad), the ball somewhere in the middle."""
    r = np.random.default_rng(700 + seed)
    duck = (r.uniform(-0.9, -0.5), r.uniform(-0.4, 0.4), r.uniform(-0.3, 0.3))
    ball = (r.uniform(-0.3, 0.5), r.uniform(-0.5, 0.5))
    return duck, ball, r


def in_goal(b):
    return b[0] > PITCH_X + BALL_R and abs(b[1]) < GOAL_W / 2


def play(seed=0, record=False, t_match=T_MATCH, trace=False, cam_every=4):
    import mujoco
    P = Pitch(seed=seed, randomize=True)
    sim = P.sim
    rt = rh.HybridRuntime(sim=sim, seed=seed, log=None)
    rt.reset("standing")
    (dx, dy, dyaw), (bx, by), rng = kickoff(seed)
    sim.d.qpos[0:2] = [dx, dy]
    sim.d.qpos[3:7] = [math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)]
    P.set_ball(bx, by)
    mujoco.mj_forward(sim.m, sim.d)
    ren = mujoco.Renderer(sim.m, H, W)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    cam_id = sim.m.camera("eye").id
    pl = Player(rt, cam_id, psi0=dyaw, pos0=(dx, dy))   # told where it stands and which way it faces at kick-off
    every = rh.HZ // CAM_HZ
    goals, goal_t, reset_at, goal_hop_end = [], None, None, 0.0
    kicks, frames, balls, states, cams, dets = [], [], [], [], [], []
    seen = frames_n = 0
    falls_t, f0 = [], 0
    timeline = []
    n = int(t_match * rh.HZ)
    for k in range(n):
        t = k * rh.DT
        b, bv = P.ball()
        if goal_t is None and in_goal(b):
            goal_t = t
            goals.append(round(t, 2))
            pl.set_state("goal!")
            if rt.mode == "drive":
                rt.command("hop")
            goal_hop_end = t + 2.5
            reset_at = t + 3.0
        if reset_at is not None and t >= reset_at:          # ball back on the centre spot
            P.set_ball(0.0, 0.0)
            goal_t = reset_at = None
            pl.set_state("search")
            pl.ball_seen_t = -9.0
        if k % every == 0:
            mujoco.mj_forward(sim.m, sim.d)
            ren.update_scene(sim.d, camera="eye")
            img = ren.render()
            det, gd = pl.see(img)
            frames_n += 1
            seen += det is not None
            if record and k % (every * cam_every) == 0:
                cams.append(F._jpeg(img, det))
        vx, vy, wz = pl.step(t)
        if pl.state == "kick" and (not kicks or kicks[-1]["n"] != pl.kicks):
            kicks.append(dict(pl.kick_at, n=pl.kicks, b0=[round(float(v), 3) for v in b[:2]], vmax=0.0, connected=False))
        if pl.state != "goal!" or t > goal_hop_end:     # (during the celebration hop the runtime runs alone)
            rt.drive(vx, vy, wz)
        rt.tick()
        pl.imu()
        if kicks and t - kicks[-1]["t"] < 1.0:
            sp = float(np.hypot(*P.ball()[1][:2]))
            kicks[-1]["vmax"] = round(max(kicks[-1]["vmax"], sp), 3)
            kicks[-1]["connected"] = kicks[-1]["vmax"] > 0.3
        if rt.falls > f0:
            falls_t.append(round(t, 2))
            f0 = rt.falls
        if record and k % 2 == 0:
            frames.append(sim.frame())
            bb = P.ball()[0]
            balls.append([round(float(bb[0]), 4), round(float(bb[1]), 4), round(float(bb[2]), 4)])
            states.append(pl.state)
            dets.append(None if pl.last_det is None else [round(pl.last_det[0]), round(pl.last_det[1]), round(pl.last_det[2], 1)])
        if trace and k % 50 == 0:
            tp, tyaw = trunk_pose(sim)
            print(f"t {t:5.1f} {pl.state:8s} mode {rt.mode:10s} ball {b[0]:+.2f},{b[1]:+.2f} est {pl.ball_f[0]:+.2f},{pl.ball_f[1]:+.2f} "
                  f"duck {tp[0]:+.2f},{tp[1]:+.2f},{tyaw:+.2f} est {pl.pos[0]:+.2f},{pl.pos[1]:+.2f},{pl.psi:+.2f} "
                  f"cmd {vx:+.2f} {vy:+.2f} {wz:+.2f} goals {len(goals)} kicks {pl.kicks} fixes {pl.fixes}")
    from_kick = [any(kc["connected"] and 0 < g - kc["t"] < 6.0 for kc in kicks) for g in goals]
    res = dict(seed=seed, goals=len(goals), goal_t=goals, goals_without_kick=len(goals) - sum(from_kick), kicks=len(kicks), connected=sum(kc["connected"] for kc in kicks),
               kick_log=kicks, falls=rt.falls, falls_t=falls_t, in_view=round(seen / max(1, frames_n), 3), body=sim.dr,
               fixes=pl.fixes, fix_rejects=pl.rejects, backups=pl.backups, unsticks=rt.unsticks,
               kickoff=dict(duck=[round(dx, 3), round(dy, 3), round(dyaw, 3)], ball=[round(bx, 3), round(by, 3)]))
    if record:
        res.update(frames=frames, bodies=sim.body_names(), ball=balls, states=states, cam=cams,
                   cam_fps=CAM_HZ / cam_every, det=dets)
    ren.close()
    return res


def _job(seed):
    t = time.time()
    r = play(seed)
    r["wall_s"] = round(time.time() - t, 1)
    return r


def summarize(runs):
    g = [r["goals"] for r in runs]
    return dict(n=len(runs), goals_median=float(np.median(g)), goals_min=min(g), goals_max=max(g), goals_total=sum(g),
                matches_with_goals=sum(x > 0 for x in g),
                kicks_median=float(np.median([r["kicks"] for r in runs])),
                dribbled_goals=sum(r.get("goals_without_kick", 0) for r in runs),
                kicks=sum(r["kicks"] for r in runs), connected=sum(r["connected"] for r in runs),
                falls=sum(r["falls"] for r in runs), runs_with_falls=sum(r["falls"] > 0 for r in runs),
                in_view_median=float(np.median([r["in_view"] for r in runs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--trace", type=int, default=None, help="one match with a printed trace (seed)")
    ap.add_argument("--t", type=float, default=T_MATCH)
    ap.add_argument("--out", default=str(HERE / "results_soccer_hybrid.json"))
    a = ap.parse_args()
    if a.trace is not None:
        r = play(a.trace, trace=True, t_match=a.t)
        print({k: v for k, v in r.items() if k != "body"})
        return
    if a.probe:
        runs = probe(a.jobs)
        Path(HERE / "probe_kick_hybrid.json").write_text(json.dumps(runs, indent=0))
        return
    lo, hi = (int(v) for v in a.seeds.split("-"))
    with ProcessPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(_job, range(lo, hi + 1)))
    for r in runs:
        print({k: v for k, v in r.items() if k not in ("body", "kick_log")})
    s = summarize(runs)
    print(json.dumps(s, indent=1))
    Path(a.out).write_text(json.dumps(dict(summary=s, runs=runs), indent=1))


if __name__ == "__main__":
    main()
