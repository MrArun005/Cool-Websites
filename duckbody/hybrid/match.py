"""5-a-side duck football: 11 hybrid ducks (Blue 5, Yellow 5, a referee in black) in one MuJoCo model, on a pitch
scaled from RoboCup Humanoid KidSize. Every duck runs the hybrid runtime (Microduck's brains + runtime_hybrid.py), the
servo model is BAM's XL330 for all 11 x 14 servos, contacts between ducks are on.

    PY=/home/user/cool-websites/.venv/bin/python
    MUJOCO_GL=osmesa $PY duckbody/hybrid/match.py --check              # one-duck physics == harness.MicroduckSim, speed
    MUJOCO_GL=osmesa $PY duckbody/hybrid/match.py --seeds 0-2          # matches (3 processes), results_match_hybrid.json
    MUJOCO_GL=osmesa $PY duckbody/hybrid/match.py --trace 0 --t 60     # one match with a printed trace
    MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --match      # the match replay for the page

The pitch (RoboCup Humanoid League KidSize field x 0.5). KidSize robots are 40-90 cm tall (typically ~60 cm) on a
9 x 6 m field; this duck is ~25 cm tall (head top), walks 0.17 m/s and runs ~0.5 m/s, close to what KidSize robots
do, but its legs, steps and kicks (~1.7 m roll) are about half theirs, so x 0.5:
    field 4.5 x 3 m, lines 2.5 cm, centre circle 0.75 m diameter, goal 1.3 m wide, 0.6 m high, 0.3 m deep,
    goal area 0.5 x 1.5 m, penalty area 1.0 x 2.5 m, penalty mark 0.75 m from the goal line, border strip 0.5 m,
    then 15 cm boards (a venue barrier; RoboCup has none, the ball would just be fetched). Ball 7 cm, 20 g
    (KidSize: a size-1 ball, ~15 cm, x 0.5 = 7.5 cm).
Goals: white posts and crossbar (5 cm), a net (back, sides and roof) that collides, so the ball stops in it.

Physics as single-duck (harness.MicroduckSim): 5 ms steps, 50 Hz brains, BAM m6 XL330 per servo (firmware P loop,
DC motor, load-dependent friction, voltage drop per robot, 3-6 step command delay), Microduck's domain randomization per
duck (mass/inertia, CoM, friction, kp, battery). BAM's MujocoController computes a robot's servo friction from the
constraint rows whose efc_id equals the servo's *joint* index (efc_id is a dof index for those rows, so it is offset by
the free joint); MultiBam keeps exactly that per robot, so one duck here steps like one duck in the harness (--check).
"""
from __future__ import annotations

import argparse
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
import harness as Hm  # noqa: E402
from harness import (BAM_KP_FW, BAM_VIN, BAM_VIN_DROP_GAIN, BAM_VIN_MIN, BAM_STIFF_SOLREF_FRICTION,  # noqa: E402
                     BAM_STIFF_SOLIMP_FRICTION, DECIMATION, DEFAULT_POSE, DELAY_LAG, HOME_TRUNK_Z, JOINT_NAMES,
                     NOISE, TIMESTEP, Policy)
import mujoco  # noqa: E402

ROBOT_XML = HERE / "hybrid_allcollisions_noantenna.xml"     # the body every skill was tested on (no antennas)

# ---------------------------------------------------------------------------------------------- the pitch
S = 0.5                                   # KidSize x 0.5
HX, HY = 9.0 * S / 2, 6.0 * S / 2         # 2.25, 1.5
LINE_W = 0.05 * S
GOAL_W, GOAL_D, GOAL_H = 2.6 * S, 0.6 * S, 1.2 * S
POST_R = 0.05 * S                         # 10 cm posts -> 5 cm
GOAL_AREA = (1.0 * S, 3.0 * S)            # length (x), width (y)
PEN_AREA = (2.0 * S, 5.0 * S)
PEN_MARK = 1.5 * S
CIRCLE_R = 1.5 * S / 2
BORDER = 1.0 * S
BOARD_H, BOARD_T = 0.15, 0.03
BALL_R, BALL_M = 0.035, 0.020
BALL_FRICTION = (0.5, 0.005, 0.002)       # soccer.py's ball: from 1 m/s it rolls ~1.7 m
BALL_RGB = F.BALL_RGB                     # magenta: the head camera's colour threshold (follow.py / soccer.py)
TURF_FRICTION = (0.7, 0.005, 0.0001)      # = the lowest of the robots' randomized foot friction (0.7-1.3): MuJoCo
                                          # takes the max of the two geoms, so a duck's own friction factor decides
TURF_RGB, LINE_RGB, GOAL_RGB, NET_RGB, BOARD_RGB = "#2e7d32", "#f4f4f4", "#ffffff", "#cfd6dc", "#1d3b5c"
TEAM_RGB = {"blue": (0.13, 0.38, 0.95), "yellow": (1.0, 0.82, 0.05), "ref": (0.06, 0.06, 0.07)}


def pitch_info():
    """What the page needs to draw the pitch (and what the rules use)."""
    return dict(hx=HX, hy=HY, line_w=LINE_W, goal_w=GOAL_W, goal_d=GOAL_D, goal_h=GOAL_H, post_r=POST_R,
                goal_area=list(GOAL_AREA), pen_area=list(PEN_AREA), pen_mark=PEN_MARK, circle_r=CIRCLE_R,
                border=BORDER, board_h=BOARD_H, board_t=BOARD_T, turf=TURF_RGB, line=LINE_RGB, goal=GOAL_RGB,
                net=NET_RGB, board=BOARD_RGB, ball_r=BALL_R, scale_of_kidsize=S)


def _rgb(h):
    h = h.lstrip("#")
    return [int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)]


def line_boxes():
    """White lines as flat boxes (centre x, y, half-length x, half-width y, yaw): touch and goal lines, halfway line,
    centre circle (segments), goal areas, penalty areas, penalty marks, centre mark."""
    w = LINE_W / 2
    b = [(0, HY, HX + w, w, 0), (0, -HY, HX + w, w, 0), (HX, 0, w, HY + w, 0), (-HX, 0, w, HY + w, 0),
         (0, 0, w, HY, 0), (0, 0, w * 2, w, 0)]
    for sx in (1, -1):
        for (L, Wd) in (GOAL_AREA, PEN_AREA):
            x = sx * (HX - L)
            b += [(x, 0, w, Wd / 2 + w, 0), (sx * (HX - L / 2), Wd / 2, L / 2, w, 0), (sx * (HX - L / 2), -Wd / 2, L / 2, w, 0)]
        b += [(sx * (HX - PEN_MARK), 0, w * 2, w, 0), (sx * (HX - PEN_MARK), 0, w, w * 2, 0)]
    n = 48
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        b.append((CIRCLE_R * math.cos(a), CIRCLE_R * math.sin(a), w, math.pi * CIRCLE_R / n + 0.002, a))
    return b


def add_pitch(spec):
    wb = spec.worldbody
    g = wb.add_geom()
    g.name, g.type, g.size = "turf", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    g.rgba = _rgb(TURF_RGB) + [1]
    g.friction = list(TURF_FRICTION)
    for i, (x, y, hx, hy, yaw) in enumerate(line_boxes()):
        g = wb.add_geom()
        g.name, g.type = f"line_{i}", mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos = [hx, hy, 0.0004], [x, y, 0.0004]
        g.quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        g.rgba = _rgb(LINE_RGB) + [1]
        g.contype = g.conaffinity = 0

    def box(name, x0, x1, y0, y1, z0, z1, rgb):
        g = wb.add_geom()
        g.name, g.type = name, mujoco.mjtGeom.mjGEOM_BOX
        g.size = [(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]
        g.pos = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
        g.rgba = _rgb(rgb) + [1]
        g.friction = [0.6, 0.005, 0.0001]
        return g

    def cyl(name, p0, p1, r, rgb):
        g = wb.add_geom()
        g.name, g.type = name, mujoco.mjtGeom.mjGEOM_CAPSULE
        g.fromto = list(p0) + list(p1)
        g.size = [r, 0, 0]
        g.rgba = _rgb(rgb) + [1]
        return g

    X, Y, Bd, t = HX + BORDER, HY + BORDER, BORDER, BOARD_T
    box("board_n", -X - t, X + t, Y, Y + t, 0, BOARD_H, BOARD_RGB)
    box("board_s", -X - t, X + t, -Y - t, -Y, 0, BOARD_H, BOARD_RGB)
    box("board_e", X, X + t, -Y, Y, 0, BOARD_H, BOARD_RGB)
    box("board_w", -X - t, -X, -Y, Y, 0, BOARD_H, BOARD_RGB)
    gw, nt = GOAL_W / 2 + POST_R, 0.01
    for sx, side in ((1, "e"), (-1, "w")):
        xp = sx * (HX + POST_R)                          # posts stand on the goal line, behind its centre
        for sy in (1, -1):
            cyl(f"post_{side}{'l' if sy > 0 else 'r'}", (xp, sy * gw, 0), (xp, sy * gw, GOAL_H), POST_R, GOAL_RGB)
        cyl(f"bar_{side}", (xp, -gw, GOAL_H), (xp, gw, GOAL_H), POST_R, GOAL_RGB)
        xb = sx * (HX + GOAL_D)
        x0, x1 = sorted((sx * HX, xb))
        box(f"net_back_{side}", min(xb, xb + sx * nt), max(xb, xb + sx * nt), -gw, gw, 0, GOAL_H, NET_RGB)
        box(f"net_l_{side}", x0, x1, gw, gw + nt, 0, GOAL_H, NET_RGB)
        box(f"net_r_{side}", x0, x1, -gw - nt, -gw, 0, GOAL_H, NET_RGB)
        box(f"net_top_{side}", x0, x1, -gw, gw, GOAL_H, GOAL_H + nt, NET_RGB)
    ball = wb.add_body()
    ball.name, ball.pos = "ball", [0, 0, BALL_R]
    ball.add_freejoint().name = "ball_free"
    g = ball.add_geom()
    g.name, g.type, g.size, g.mass = "ball_geom", mujoco.mjtGeom.mjGEOM_SPHERE, [BALL_R, 0, 0], BALL_M
    g.rgba = list(BALL_RGB) + [1.0]
    g.friction = list(BALL_FRICTION)
    g.condim = 6


# ---------------------------------------------------------------------------------------------- the ducks
def robot_spec(force_limit, team, cam=True):
    """One hybrid duck as MicroduckSim prepares it (policy servos -> BAM torque motors), with the head camera and
    the team colour on the light shell parts (the owner's head and bill keep their colours)."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for act in spec.actuators:
        tgt = act.target if isinstance(act.target, str) else act.target.name
        if tgt.startswith("passive_") or tgt not in JOINT_NAMES:
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        j = spec.joint(tgt)
        j.damping = np.zeros((3, 1))
        j.frictionloss = 0.0
        j.solref_friction = BAM_STIFF_SOLREF_FRICTION
        j.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
    # The visible-only meshes (group 2, no contacts; every body has an explicit inertial, so they carry no mass)
    # do nothing in the simulation: the head cameras draw box proxies and the page draws its own mesh. Dropping
    # them (and meshes nothing uses any more) takes the 11-duck model from ~860 MB to a fraction.
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0 and g.group == 2:
            spec.delete(g)
    used = {g.meshname for g in spec.geoms}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
    if cam:
        c = spec.body("jaw_soft").add_camera()
        c.name, c.pos, c.quat, c.fovy = "eye", list(F.EYE_POS), list(F.EYE_QUAT), F.FOVY
    tint = TEAM_RGB.get(team)
    if cam:                     # what the head cameras draw of the other ducks: one box per part (see proxy_boxes)
        for body, (pos, half, rgb) in proxy_boxes().items():
            g = spec.body(body).add_geom()
            g.type, g.pos, g.size = mujoco.mjtGeom.mjGEOM_BOX, list(pos), list(half)
            g.contype = g.conaffinity = 0
            g.group = PROXY_GROUP
            g.rgba = list(tint if tint is not None and tint_part(rgb) else rgb) + [1.0]
    if tint is not None:
        for mat in spec.materials:
            if tint_part(mat.rgba[:3], mat.name):
                mat.rgba = list(tint) + [mat.rgba[3]]
    return spec


PROXY_GROUP = 5
_PROXY = {}


def proxy_boxes():
    """Per body of the duck: the box around its visible meshes (body frame) and the colour of its biggest part. The
    head cameras render these instead of the full meshes: OSMesa (software) takes ~1.2 s per frame for 11 full
    ducks (~3 M triangles), ~10 ms with the boxes. The camera only needs the ball, and the ducks for occlusion."""
    if _PROXY:
        return _PROXY
    m = mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for b in range(1, m.nbody):
        pts, best = [], (0, (0.8, 0.8, 0.8))
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
            rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
            if len(v) > best[0]:
                best = (len(v), tuple(float(c) for c in rgba[:3]))
        if pts:
            P = np.concatenate(pts)
            lo, hi = P.min(0), P.max(0)
            _PROXY[m.body(b).name] = ((lo + hi) / 2, np.maximum((hi - lo) / 2 * 0.9, 0.004), best[1])
    return _PROXY


def tint_part(rgb, name=""):
    """Which parts take the team colour: light, unsaturated shell / bracket parts, not the owner's head and bill
    (the same rule runs on the page)."""
    r, g, b = (float(v) for v in rgb[:3])
    if name.startswith("owner") or "bill" in name:
        return False
    return min(r, g, b) > 0.55 and max(r, g, b) - min(r, g, b) < 0.15


ROBOTS = [("b1", "blue"), ("b2", "blue"), ("b3", "blue"), ("b4", "blue"), ("b5", "blue"),
          ("y1", "yellow"), ("y2", "yellow"), ("y3", "yellow"), ("y4", "yellow"), ("y5", "yellow"), ("ref", "ref")]


def build_model(robots=ROBOTS, pitch=True, cams=True):
    """One MjModel: the pitch, the ball, and a copy of the duck per robot (attached under a frame, names prefixed
    'b1_' ...)."""
    from bam.model import load_model
    bam = load_model(motor_name="xl330", model="m6")
    force_limit = BAM_VIN * bam.kt.value / bam.R.value      # as MicroduckSim (from the nominal 7.4 V)
    world = mujoco.MjSpec()
    world.compiler.autolimits = True
    world.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    world.visual.headlight.ambient = [0.35, 0.35, 0.35]
    world.visual.headlight.specular = [0, 0, 0]
    world.visual.global_.offwidth = 640
    world.memory = 128 << 20                # 11 ducks in a heap (kick-off building) need more than the default arena
    lt = world.worldbody.add_light()
    lt.pos, lt.dir = [0, 0, 3.5], [0, 0, -1]
    lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    if pitch:
        add_pitch(world)
    else:
        g = world.worldbody.add_geom()
        g.name, g.type, g.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    first = None
    for i, (name, team) in enumerate(robots):
        child = robot_spec(force_limit, team, cam=cams)
        if first is not None:                    # one copy of the meshes for all ducks: the others' geoms point at
            for mesh in list(child.meshes):      # the first duck's mesh assets (attach would copy ~200 MB per duck)
                child.delete(mesh)
        fr = world.worldbody.add_frame()
        fr.pos = [0.0, 0.0, 0.0]
        fr.attach_body(child.body("trunk_base"), name + "_", "")
        if first is None:
            first = name + "_"
    if len(robots) > 1:
        for g in world.geoms:
            mn = g.meshname
            if mn and not mn.startswith(first):
                g.meshname = first + mn.split("_", 1)[1]
    m = world.compile()
    m.opt.timestep = TIMESTEP
    return m, bam


class MultiBam:
    """BAM's MujocoController.update() for every robot's 14 servos at once (numpy over 14 x n). Same model and
    maths; the per-robot parts are kept per robot: the supply voltage drop (sum of that robot's |torque|), the
    randomized kp_fw and battery voltage, and the friction lookup (see the module docstring)."""

    def __init__(self, bam_model, m, d, robots):
        self.model, self.m, self.d = bam_model, m, d
        self.n = len(robots)
        self.act, self.jnt, quirk = [], [], []
        for name in robots:
            free = m.joint(f"{name}_trunk_base_freejoint").id
            j0, d0 = free, m.jnt_dofadr[free]
            for jn in JOINT_NAMES:
                a = m.actuator(f"{name}_{jn}").id
                j = m.actuator_trnid[a, 0]
                self.act.append(a)
                self.jnt.append(j)
                quirk.append(d0 + (j - j0))     # = this joint's index in the one-robot model, at this robot's offset
        self.act, self.jnt = np.array(self.act), np.array(self.jnt)
        self.quirk = np.array(quirk)
        self.qadr, self.dof = m.jnt_qposadr[self.jnt], m.jnt_dofadr[self.jnt]
        m.dof_armature[self.dof] = bam_model.actuator.get_extra_inertia()
        mujoco.mj_setConst(m, d)
        self.q_target = np.zeros(14 * self.n)
        self.kp = np.full(self.n, BAM_KP_FW)
        self.vin = np.full(self.n, BAM_VIN)
        self.prev = np.zeros(14 * self.n)
        self.last_ts = d.time
        self.gain, self.vmin = BAM_VIN_DROP_GAIN, BAM_VIN_MIN

    def update(self):
        d, m, act = self.d, self.m, self.model.actuator
        q, dq = d.qpos[self.qadr], d.qvel[self.dof]
        load = np.abs(self.prev).reshape(self.n, 14).sum(1)
        vin = np.maximum(self.vin - self.gain * load, self.vmin)
        vin0, kp0 = act.vin, act.kp
        act.vin, act.kp = np.repeat(vin, 14), np.repeat(self.kp, 14)
        dt = d.time - self.last_ts
        self.last_ts = d.time
        control = act.compute_control(self.q_target, q, dq, dt)
        torque = act.compute_torque(control, True, q, dq)
        act.vin, act.kp = vin0, kp0
        self.prev = np.atleast_1d(torque).copy()
        d.ctrl[self.act] = torque
        ext = -d.qfrc_bias[self.dof] + d.qfrc_constraint[self.dof]
        sel = d.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF.value
        fr = np.zeros(m.nv)
        if sel.any():
            np.add.at(fr, d.efc_id[sel], d.efc_force[sel])
        ext -= fr[self.quirk]
        fl, damp = self.model.compute_frictions(d.qfrc_actuator[self.dof], ext, dq)
        m.dof_frictionloss[self.dof] = fl
        m.dof_damping[self.dof] = damp


class _Model:
    """sim.m for a runtime: the shared model, with actuator names resolved inside this robot."""

    def __init__(self, m, prefix):
        self._m, self._p = m, prefix

    def actuator(self, name):
        return self._m.actuator(self._p + name)

    def __getattr__(self, k):
        return getattr(self._m, k)


class Duck:
    """What runtime_hybrid.HybridRuntime needs from a sim, for one robot of the shared model. step() only runs the
    brain (the action for this tick); World.physics() then steps every robot's servos and the physics together."""

    def __init__(self, world, i, name, team):
        m = world.m
        self.world, self.i, self.name, self.team = world, i, name, team
        self.m, self.d = _Model(m, name + "_"), world.d
        self.trunk = m.body(f"{name}_trunk_base").id
        self.free_q = m.jnt_qposadr[m.joint(f"{name}_trunk_base_freejoint").id]
        self.free_v = m.jnt_dofadr[m.joint(f"{name}_trunk_base_freejoint").id]
        s = m.sensor(f"{name}_imu_ang_vel")
        self.gyro_adr = slice(s.adr[0], s.adr[0] + 3)
        sl = slice(14 * i, 14 * i + 14)
        self.qadr, self.vadr = world.bam.qadr[sl], world.bam.dof[sl]
        self.jrange = m.jnt_range[world.bam.jnt[sl]]
        self.bodies = [b for b in range(m.nbody) if m.body_rootid[b] == self.trunk]
        self.geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == self.trunk}
        self.cam = m.camera(f"{name}_eye").id if world.cams else None
        self.rng = np.random.default_rng(world.seed * 100 + i)
        self.obs_noise, self.delay = True, True
        self.pending, self.on_substep = DEFAULT_POSE.copy(), None
        self.dr = {}
        self.reset_state()

    def reset_state(self):
        self.last_action = np.zeros(14, np.float32)
        self.command = np.zeros(13, np.float32)
        self.lag = int(self.rng.integers(DELAY_LAG[0], DELAY_LAG[1] + 1)) if self.delay else 0
        self.target_hist = []
        self.pending = DEFAULT_POSE.copy()
        self.dr["lag"] = self.lag

    def place(self, kind="standing", **kw):
        """Only resets the brain-side state; World.put() places the body."""
        self.reset_state()

    # --- sensing, as harness.MicroduckSim
    def gravity(self):
        return -self.d.xmat[self.trunk].reshape(3, 3)[2, :]

    def tilt_deg(self):
        return math.degrees(math.acos(float(np.clip(-self.gravity()[2], -1, 1))))

    def trunk_z(self):
        return float(self.d.xpos[self.trunk][2])

    def obs(self):
        o = np.concatenate([self.d.sensordata[self.gyro_adr], self.gravity(), self.d.qpos[self.qadr] - DEFAULT_POSE,
                            self.d.qvel[self.vadr], self.last_action, self.command]).astype(np.float32)
        if self.obs_noise:
            o[:48] += self.rng.uniform(-1, 1, 48).astype(np.float32) * NOISE
        return o

    def step(self, policy, on_substep=None):
        action = policy(self.obs())
        self.last_action = action.astype(np.float32)
        self.pending = DEFAULT_POSE + action * policy.action_scale
        self.on_substep = on_substep
        return action

    def pose(self):
        p = self.d.xpos[self.trunk]
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return np.array([p[0], p[1]]), math.atan2(R[1, 0], R[0, 0])


class World:
    """The shared model + data, the robots, the multi-robot BAM controller."""

    def __init__(self, robots=ROBOTS, seed=0, randomize=True, pitch=True, cams=True):
        self.seed, self.cams = seed, cams
        self.touch_hook = None
        self.m, bam = build_model(robots, pitch=pitch, cams=cams)
        self.d = mujoco.MjData(self.m)
        bam.actuator.kp = BAM_KP_FW
        bam.actuator.vin = BAM_VIN
        bam.actuator.max_current = None
        self.bam = MultiBam(bam, self.m, self.d, [n for n, _ in robots])
        self.ducks = [Duck(self, i, n, t) for i, (n, t) in enumerate(robots)]
        self.rng = np.random.default_rng(seed)
        self.mass0 = self.m.body_mass.copy()
        self.inertia0 = self.m.body_inertia.copy()
        self.ipos0 = self.m.body_ipos.copy()
        self.fric0 = self.m.geom_friction.copy()
        if pitch:
            j = self.m.joint("ball_free")
            self.bq, self.bv = int(self.m.jnt_qposadr[j.id]), int(self.m.jnt_dofadr[j.id])
            self.ball_geom = self.m.geom("ball_geom").id
        if randomize:
            self.randomize()
        for dk in self.ducks:                    # not all on the origin (a heap of 11 ducks: ~35k contacts)
            self.put(dk, -2.0 + 0.4 * dk.i, 0.0, 0.0)
        mujoco.mj_forward(self.m, self.d)

    def randomize(self):
        """harness.MicroduckSim.randomize_body per duck: mass/inertia x U(0.95, 1.05) per part, trunk and head CoM
        +-3 mm, friction x U(0.7, 1.3) on its own geoms, kp_fw x U(0.85, 1.15), battery U(6.5, 8.2) V."""
        m, r = self.m, self.rng
        for k, dk in enumerate(self.ducks):
            b = np.array(dk.bodies)
            s = r.uniform(0.95, 1.05, len(b))
            m.body_mass[b] = self.mass0[b] * s
            m.body_inertia[b] = self.inertia0[b] * s[:, None]
            m.body_ipos[b] = self.ipos0[b]
            for part in ("trunk_base", "jaw_soft"):
                m.body_ipos[m.body(f"{dk.name}_{part}").id] += r.uniform(-0.003, 0.003, 3)
            f = r.uniform(0.7, 1.3)
            g = np.array(sorted(dk.geoms))
            m.geom_friction[g, 0] = self.fric0[g, 0] * f
            self.bam.kp[k] = BAM_KP_FW * r.uniform(0.85, 1.15)
            self.bam.vin[k] = r.uniform(6.5, 8.2)
            dk.dr.update(mass_scale_mean=round(float(s.mean()), 3), friction=round(float(f), 3),
                         kp=round(float(self.bam.kp[k]), 1), vin=round(float(self.bam.vin[k]), 2))
        mujoco.mj_setConst(m, self.d)

    def put(self, duck, x, y, yaw, joints=None):
        """Stand a duck at (x, y) facing yaw, home pose, at rest (kick-off positions, manual placement)."""
        d = self.d
        q = DEFAULT_POSE.astype(float) if joints is None else np.asarray(joints, float)
        d.qpos[duck.free_q:duck.free_q + 7] = [x, y, HOME_TRUNK_Z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[duck.free_v:duck.free_v + 6] = 0
        d.qpos[duck.qadr] = q
        d.qvel[duck.vadr] = 0
        sl = slice(14 * duck.i, 14 * duck.i + 14)
        self.bam.q_target[sl] = q
        duck.pending = q.astype(np.float32).copy()
        duck.target_hist = []

    def physics(self):
        """4 physics steps of 5 ms with every robot's latest servo targets (each through its own command delay)."""
        for _ in range(DECIMATION):
            for dk in self.ducks:
                dk.target_hist.append(dk.pending)
                if len(dk.target_hist) > dk.lag + 1:
                    dk.target_hist.pop(0)
                self.bam.q_target[14 * dk.i:14 * dk.i + 14] = dk.target_hist[0]
            self.bam.update()
            mujoco.mj_step(self.m, self.d)
            for dk in self.ducks:
                if dk.on_substep is not None:
                    dk.on_substep(dk)
            if self.touch_hook is not None:
                self.touch_hook()

    # ball
    def set_ball(self, x, y, vx=0.0, vy=0.0):
        self.d.qpos[self.bq:self.bq + 7] = [x, y, BALL_R, 1, 0, 0, 0]
        self.d.qvel[self.bv:self.bv + 6] = [vx, vy, 0, 0, 0, 0]

    def ball(self):
        return self.d.qpos[self.bq:self.bq + 3].copy(), self.d.qvel[self.bv:self.bv + 3].copy()

    def ball_touchers(self):
        """Indices of the ducks touching the ball now."""
        d, out = self.d, set()
        bg = self.ball_geom
        for c in d.contact[:d.ncon]:
            if c.geom1 == bg or c.geom2 == bg:
                o = c.geom2 if c.geom1 == bg else c.geom1
                for dk in self.ducks:
                    if o in dk.geoms:
                        out.add(dk.i)
        return out


# ---------------------------------------------------------------------------------------------- runtime per duck
_POLICIES = {}


def _shared_policy(name):
    """One ONNX session per brain, shared by the 11 runtimes (the brains are stateless; last_action is per duck)."""
    if name not in _POLICIES:
        _POLICIES[name] = Policy(name)
    return _POLICIES[name]


rh.Policy = _shared_policy


class MatchRuntime(rh.HybridRuntime):
    """HybridRuntime, unchanged, with tick() split around the shared physics: tick_brain() (personality, command,
    the brain's action) for every duck, World.physics(), then tick_after() (IMU yaw, fall / get-up / unstick logic,
    plan steps) for every duck. Same order of operations as HybridRuntime.tick for one duck."""

    def tick(self):
        raise RuntimeError("use tick_brain(); world.physics(); tick_after()")

    def tick_brain(self):
        while self.todo and self.todo[0][0] <= self.steps:
            self.todo.pop(0)[1]()
        self._personality()
        self.sim.command[:] = self._command_vector()
        self.sim.step(self.cur, on_substep=self.on_substep)

    def tick_after(self):
        self.yaw_est += float(self.sim.d.sensordata[self.sim.gyro_adr][2]) * rh.DT
        self.roll_rot = self.roll_rot + float(self.sim.d.sensordata[self.sim.gyro_adr][1]) * rh.DT if self.skill == "roulade" else 0.0
        self.steps += 1
        self.plan_t += 1
        up = self.sim.tilt_deg() < 25
        self.up_ticks = self.up_ticks + 1 if up else 0
        self.down_ticks = 0 if up else self.down_ticks + 1
        if self.up_ticks > rh.HZ:
            self.unstick_k = 0
        if self.lying() and self.mode not in ("lie down", "roll", "get up") and self.skill not in rh.SEATED_SKILLS:
            self._say_log("fell over: getting up")
            self.falls += 1
            self.vel[:] = 0
            self.vel_goal[:] = 0
            self.mode, self.plan, self.plan_all = "get up", [("velstand", None)], [("velstand", None)]
            self._start("velstand")
            return
        if self.mode == "get up" and self.skill == "velstand" and min(self.down_ticks, self.plan_t) > rh.UNSTICK_AFTER_S * rh.HZ:
            step = rh.UNSTICK_SEQ[self.unstick_k % len(rh.UNSTICK_SEQ)]
            self._say_log(f"stuck getting up: {step[0]} {step[1]}, then get up")
            self.unsticks += 1
            self.unstick_k += 1
            self.down_ticks = 0
            self.plan = [step, ("velstand", None)]
            self._start(*step)
            return
        if self.plan and self._step_done(self.plan[0][1]):
            self.plan.pop(0)
            if self.plan:
                self._next()


def make_renderer(m):
    ren = mujoco.Renderer(m, F.H, F.W)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    ren._opt = mujoco.MjvOption()
    ren._opt.geomgroup[:] = 0
    ren._opt.geomgroup[0] = 1               # pitch, lines, goals, ball
    ren._opt.geomgroup[PROXY_GROUP] = 1     # the ducks as boxes
    return ren


def render_eye(ren, d, cam):
    ren.update_scene(d, camera=cam, scene_option=ren._opt)
    return ren.render()


def check(seconds=4.0):
    """One duck in World (no pitch) against harness.MicroduckSim on the same body and the same commands (no noise,
    no delay randomness, no domain randomization): the trajectories must agree."""
    w = World(robots=[("b1", None)], randomize=False, pitch=False, cams=False)
    dk = w.ducks[0]
    dk.obs_noise, dk.delay = False, False
    dk.reset_state()
    sim = Hm.MicroduckSim(str(ROBOT_XML.parent / "scene_hybrid_allcollisions_noantenna.xml"), delay=False, obs_noise=False)
    sim.place("standing", joint_noise=0.0)
    q0 = sim.d.qpos.copy()
    sim.d.qpos[3:7] = [1, 0, 0, 0]
    sim.d.qpos[:3] = [0, 0, HOME_TRUNK_Z]
    sim.d.qpos[sim.qadr] = DEFAULT_POSE
    sim.bam.q_target[:] = DEFAULT_POSE
    mujoco.mj_forward(sim.m, sim.d)
    w.put(dk, 0, 0, 0)
    w.d.qpos[dk.free_q:dk.free_q + 3] = [0, 0, HOME_TRUNK_Z]
    mujoco.mj_forward(w.m, w.d)
    pol = _shared_policy("velstand")
    worst = 0.0
    for k in range(int(seconds * 50)):
        c = np.zeros(13, np.float32)
        c[0] = 0.4 if k > 25 else 0.0
        c[2] = 0.3 if k > 100 else 0.0
        sim.command[:] = c
        dk.command[:] = c
        sim.step(pol)
        dk.step(pol)
        w.physics()
        worst = max(worst, float(np.abs(sim.d.xpos[sim.trunk] - w.d.xpos[dk.trunk]).max()))
    print(f"one duck, World vs MicroduckSim, {seconds} s walk + turn: max trunk position difference {worst * 1000:.4f} mm"
          f" (trunk at {sim.d.xpos[sim.trunk].round(3)} vs {w.d.xpos[dk.trunk].round(3)})")
    del q0
    return worst


def speed(seconds=5.0, n=11, render=0):
    """Real-time factor with n ducks standing / walking on the pitch (velstand), optional head-camera renders per
    50 Hz tick-group of 5 (10 Hz)."""
    w = World(robots=ROBOTS[:n], seed=0, randomize=True)
    rts = []
    for i, dk in enumerate(w.ducks):
        x = -1.5 + 0.3 * (i % 5) * 2
        y = -0.8 + 0.4 * (i // 5)
        w.put(dk, x, y, 0.0)
        rt = MatchRuntime(sim=dk, seed=i, log=None)
        rt.reset("standing")
        rt.drive(0.6, 0.0, 0.2)
        rts.append(rt)
    w.set_ball(0.3, 0.0)
    mujoco.mj_forward(w.m, w.d)
    ren = None
    if render:
        ren = make_renderer(w.m)
    t0 = time.time()
    tb = tp = tr = 0.0
    N = int(seconds * 50)
    for k in range(N):
        a = time.time()
        for rt in rts:
            rt.drive(0.6, 0.0, 0.2)
            rt.tick_brain()
        b = time.time()
        w.physics()
        c = time.time()
        for rt in rts:
            rt.tick_after()
        if ren is not None and k % 5 == 0:
            for dk in w.ducks[:render]:
                render_eye(ren, w.d, dk.cam)
        e = time.time()
        tb += b - a
        tp += c - b
        tr += e - c
    wall = time.time() - t0
    up = sum(not rt.lying() for rt in rts)
    print(f"{n} ducks, {render} cameras at 10 Hz: {seconds:.0f} s sim in {wall:.1f} s wall = {seconds / wall:.2f} x real time "
          f"(brains {tb:.1f} s, physics+BAM {tp:.1f} s, rest+render {tr:.1f} s); upright {up}/{n}; ncon {w.d.ncon}, nefc {w.d.nefc}")
    return seconds / wall




# ================================================================================================ the game
import soccer as SO  # noqa: E402  (camera geometry, kick placement, alignment numbers from the single-duck game)
import tactics as TT  # noqa: E402  (the team tactics: pure Python, see tactics.py)

HALF_S = 180.0              # playing time per half (2 halves, sides swapped at half-time)
CAM_EVERY = rh.HZ // F.CAM_HZ   # head camera at 10 Hz
REC_EVERY = 3               # replay frames at 50/3 = 16.7 fps
WIFI_DELAY, WIFI_NOISE, WIFI_EVERY = 0.2, 0.02, 0.2     # team messages: 200 ms late, 2 cm noise, 5 per second
LOC_SIGMA, LOC_YAW_SIGMA, LOC_TAU = 0.05, 0.05, 2.0     # self-localization error: 5 cm, 3 deg, 2 s correlation
OPP_NOISE = 0.06            # where a duck thinks an opponent is (sim truth + this noise; stands in for robot detection)
OPP_TAU = 1.0               # ... drifting with this correlation time (a tracker's error, not white noise)
READY_S, SET_S, GOAL_S = 14.0, 1.0, 3.0
FREEKICK_S, FREEKICK_D = 10.0, 0.5   # set pieces: opponents keep 0.5 m from the ball for 10 s or until it is played
KICKOFF_S = 10.0
NO_PROGRESS_S = 10.0        # ball untouched / pinned between ducks this long (moved < 15 cm): dropped ball
PICKUP_DOWN_S, PICKUP_OUT_S, PICKUP_S = 12.0, 8.0, 10.0   # incapable player: down 12 s / off the pitch 8 s -> out 10 s
OUT_WHISTLE_S = 0.8         # the ball is placed this long after it went out
ROLES = ("GK", "D1", "D2", "A1", "A2")
assert (TT.HX, TT.HY, TT.GOAL_W, TT.PEN_AREA, TT.BORDER, TT.BALL_R) == (HX, HY, GOAL_W, PEN_AREA, BORDER, BALL_R)
assert (TT.KICK_X, TT.KICK_Y, TT.TOL_X, TT.TOL_Y, TT.TOL_H, TT.BEHIND, TT.AROUND, TT.READY_S, TT.FOOT_VIEW, TT.CAM_H,
        TT.CAM_DOWN, TT.DT) == (SO.KICK_X, SO.KICK_Y, SO.TOL_X, SO.TOL_Y, SO.TOL_H, SO.BEHIND, SO.AROUND, SO.READY_S,
                                SO.FOOT_VIEW, SO.CAM_H, F.CAM_DOWN, rh.DT)
# kick-off spots (team frame: own goal at -x), (x, y): kicking team / other team
KICKOFF_SPOTS = {True: {"GK": (-HX + 0.3, 0.0), "D1": (-1.3, 0.6), "D2": (-1.3, -0.6), "A1": (-0.22, 0.03), "A2": (-0.7, -0.75)},
                 False: {"GK": (-HX + 0.3, 0.0), "D1": (-1.3, 0.6), "D2": (-1.3, -0.6), "A1": (-0.6, 0.0), "A2": (-0.75, 0.75)}}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def seg_dist(a, b, p):
    """Distance of p from segment a-b and the position along it (0 at a, 1 at b)."""
    ab = b - a
    L2 = float(ab @ ab)
    s = float(np.clip((p - a) @ ab / L2, 0, 1)) if L2 > 1e-9 else 0.0
    return float(np.linalg.norm(a + s * ab - p)), s


class Team:
    """A team's shared world model (RoboCup team communication): the ball estimate of whoever saw it last, every
    player's pose, who is the striker. Messages arrive WIFI_DELAY late with WIFI_NOISE on the ball."""

    def __init__(self, name, dir_):
        self.name, self.dir = name, dir_
        self.msgs, self.ball, self.ball_t, self.ball_from = [], None, -99.0, None
        self.poses = {}
        self.striker = None
        self.score = 0
        self.players = []

    def send(self, t, sender, ball, ball_t, rng):
        if ball is not None:
            self.msgs.append((t + WIFI_DELAY, sender, np.asarray(ball) + rng.normal(0, WIFI_NOISE, 2), ball_t))

    def deliver(self, t):
        keep = []
        for m in self.msgs:
            if m[0] <= t:
                if m[3] > self.ball_t:
                    self.ball, self.ball_t, self.ball_from = m[2], m[3], m[1]
            else:
                keep.append(m)
        self.msgs = keep

    def announce(self, ball, t):
        """The referee placed the ball (kick-off, throw-in ...): every player knows where (by the rules / it is
        announced), as when a RoboCup robot sees the set piece being placed."""
        self.msgs = []
        self.ball, self.ball_t, self.ball_from = np.array(ball, float), t, None
        for p in self.players:
            p.ball_f, p.ball_t, p.misses = np.array(ball, float), t, 0


class Player:
    """One duck's soccer brain. Pose: self-localization (sim truth + a slowly drifting error, LOC_*; stands in for
    the line/goal particle filter a RoboCup robot runs). Ball: its own head camera when it renders one (the
    striker and the goalkeeper, 10 Hz, soccer.py's colour blob + floor ray), else the team's shared estimate."""

    def __init__(self, game, duck, rt, team, role, number):
        self.g, self.duck, self.rt, self.team, self.role, self.num = game, duck, rt, team, role, number
        self.rng = np.random.default_rng(1000 * game.seed + duck.i)
        self.err = self.rng.normal(0, 1, 3) * [LOC_SIGMA, LOC_SIGMA, LOC_YAW_SIGMA]
        self.ball_f, self.ball_t, self.misses = None, -99.0, 0
        self.state, self.cmd = "search", None
        self.cam_on = False
        self.penalized_until = None
        self.last_sent = -9.0
        self.t = 0.0
        self.down_since, self.out_since = None, None
        self.stats = dict(kicks=0, connected=0, shots=0, passes=0, passes_completed=0, clearances=0, goals=0,
                          own_goals=0, touches=0, pickups=0, frames_seen=0, frames=0)

    @property
    def name(self):
        return self.duck.name

    # ------------------------------------------------------------ estimation
    def update_pose(self):
        a = rh.DT / LOC_TAU
        self.err += -a * self.err + math.sqrt(2 * a) * self.rng.normal(0, 1, 3) * [LOC_SIGMA, LOC_SIGMA, LOC_YAW_SIGMA]
        p, yaw = self.duck.pose()
        self.pos = p + self.err[:2]
        self.psi = wrap(yaw + self.err[2])

    def to_field(self, rel):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return self.pos + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])

    def to_body(self, p):
        d = np.asarray(p) - self.pos
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def expect_visible(self):
        if self.ball_f is None:
            return False
        sim = self.duck
        rel = self.to_body(self.ball_f)
        R = sim.d.xmat[sim.trunk].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        p = sim.d.xpos[sim.trunk] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], 0.0])
        p[2] = BALL_R
        pc = sim.d.cam_xmat[sim.cam].reshape(3, 3).T @ (p - sim.d.cam_xpos[sim.cam])
        if pc[2] > -0.03:
            return False
        u, v = -pc[0] / pc[2] * SO.FY, pc[1] / pc[2] * SO.FY
        return abs(u) < F.W / 2 - 6 and abs(v) < F.H / 2 - 6

    def see(self, img, t):
        det = F.detect(img)
        self.stats["frames"] += 1
        if det is not None:
            g = SO.pixel_ground(self.duck, self.duck.cam, det[0], det[1], BALL_R)
            if g is not None and np.hypot(*g) < 3.0:
                self.ball_f, self.ball_t, self.misses = self.to_field(g), t, 0
                self.stats["frames_seen"] += 1
                return det
        if self.expect_visible():
            self.misses += 1
        return det

    def ball(self, t):
        """(ball estimate, source): own camera if fresh, else the team's (delayed, noisy), else None."""
        if self.ball_f is not None and t - self.ball_t < 0.7 and self.misses < 4:
            return self.ball_f, "own"
        tm = self.team
        if tm.ball is not None and t - tm.ball_t < 4.0:
            return tm.ball, "team"
        if self.ball_f is not None and t - self.ball_t < 4.0 and self.misses < 4:
            return self.ball_f, "own"
        return None, None

    def share(self, t):
        if self.ball_f is not None and self.ball_t > self.last_sent and t - self.last_sent >= WIFI_EVERY and t - self.ball_t < 0.3:
            self.team.send(t, self.duck.i, self.ball_f, self.ball_t, self.rng)
            self.last_sent = t

    def active(self):
        return self.penalized_until is None


class Referee:
    """The referee duck: walks with play ~0.8 m from the ball, off the line of play (at the ball's side, in y),
    looks at the ball. It is choreography: the decisions come from the GameController (simulator truth), as in
    RoboCup where the human referee's calls go through the GameController."""

    def __init__(self, game, duck, rt):
        self.g, self.duck, self.rt = game, duck, rt
        self.side = -1.0

    def act(self, t):
        rt = self.rt
        if rt.mode == "get up" or rt.lying() or rt.mode == "hop":
            return 0.0, 0.0, 0.0
        b = self.g.world.ball()[0][:2]
        p, yaw = self.duck.pose()
        if b[1] < -HY + 0.9:
            self.side = 1.0
        elif b[1] > HY - 0.9:
            self.side = -1.0
        target = np.array([np.clip(b[0] - 0.15 * np.sign(b[0]), -HX - 0.1, HX + 0.1), np.clip(b[1] + 0.8 * self.side, -HY - 0.35, HY + 0.35)])
        for dk in self.g.world.ducks:                       # keep out of the players' way
            if dk is self.duck:
                continue
            q, _ = dk.pose()
            v = target - q
            dd = float(np.linalg.norm(v))
            if dd < 0.4:
                target = target + unit(v) * (0.4 - dd)
        e = target - p
        c, s = math.cos(yaw), math.sin(yaw)
        eb = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
        d, bear = float(np.hypot(*eb)), math.atan2(eb[1], eb[0])
        rb = b - p
        rbb = np.array([c * rb[0] + s * rb[1], -s * rb[0] + c * rb[1]])
        bb = math.atan2(rbb[1], rbb[0])
        dist = float(np.hypot(*rbb))
        self.rt.look(float(np.clip(bb, -1.2, 1.2)), math.atan2(SO.CAM_H - BALL_R, max(dist, 0.1)) - F.CAM_DOWN)
        if d < 0.15:                                         # there: face the ball
            wz = 0.0 if abs(bb) < 0.3 else math.copysign(0.6, bb)
            return 0.0, 0.0, wz
        wz = float(np.clip(1.6 * bear, -1.0, 1.0))
        if abs(bear) > 0.12 and abs(wz) < 0.6:
            wz = math.copysign(0.6, bear)
        vx = float(np.clip(0.3 + 1.2 * d, 0.32, 1.1)) if abs(bear) < 0.5 else 0.0
        return vx, 0.0, wz


MOVE_V, MOVE_WIN = 0.03, 0.5    # "moving": the trunk travelled > 3 cm/s over the last 0.5 s (steps in place don't count)
MOVE_FAST = 0.10                # "walking briskly": > 10 cm/s over the same window (velstand's top speed ~0.4 m/s)
DUEL_R, DUEL_GAP = 0.3, 1.0     # duel: players of both teams within 0.3 m of the ball (a new one after 1 s without)


class Metrics:
    """Measurement only (simulator truth, no effect on play): per player the distance walked and the share of
    playing time it was moving (trunk displacement over 0.5 s windows), duels for the ball, and body contacts
    between opponents (shoulder-to-shoulder)."""

    def __init__(self, game):
        self.g = game
        w = game.world
        self.owner = np.full(w.m.ngeom, -1)
        for dk in w.ducks:
            self.owner[sorted(dk.geoms)] = dk.i
        self.hist = {p.name: [] for p in game.players}
        self.per = {p.name: dict(dist=0.0, moving=0, fast=0, samples=0) for p in game.players}
        self.duels, self.in_duel, self.duel_last = 0, False, -9.0
        self.bumps, self.bump_last = 0, {}
        self.play_s = 0.0

    def tick(self):
        g = self.g
        if g.phase != "playing":
            for h in self.hist.values():
                h.clear()
            return
        self.play_s += rh.DT
        every = int(round(0.1 / rh.DT))
        if g.k % every == 0:
            n = int(round(MOVE_WIN / 0.1))
            for p in g.players:
                h = self.hist[p.name]
                if not p.active():
                    h.clear()
                    continue
                q = p.duck.pose()[0].copy()
                h.append(q)
                if len(h) > n + 1:
                    h.pop(0)
                if len(h) == n + 1:
                    st = self.per[p.name]
                    disp = float(np.linalg.norm(h[-1] - h[0]))
                    st["samples"] += 1
                    st["moving"] += disp / MOVE_WIN > MOVE_V
                    st["fast"] += disp / MOVE_WIN > MOVE_FAST
                    if g.k % (n * every) == 0:
                        st["dist"] += disp
        b = g.world.ball()[0][:2]
        near = {"blue": 9.0, "yellow": 9.0}
        for p in g.players:
            if p.active():
                near[p.team.name] = min(near[p.team.name], float(np.linalg.norm(p.duck.pose()[0] - b)))
        duel = near["blue"] < DUEL_R and near["yellow"] < DUEL_R
        if duel:
            if not self.in_duel and g.t - self.duel_last > DUEL_GAP:
                self.duels += 1
            self.duel_last = g.t
        self.in_duel = duel
        d = g.world.d
        team = {dk.i: dk.team for dk in g.world.ducks}
        for c in d.contact[:d.ncon]:
            a, b2 = self.owner[c.geom1], self.owner[c.geom2]
            if a < 0 or b2 < 0 or a == b2 or team[a] == "ref" or team[b2] == "ref" or team[a] == team[b2]:
                continue
            key = (min(a, b2), max(a, b2))
            if g.t - self.bump_last.get(key, -9.0) > 1.0:
                self.bumps += 1
            self.bump_last[key] = g.t

    def result(self):
        per = {}
        for n, st in self.per.items():
            per[n] = dict(dist_m=round(st["dist"], 2), moving=round(st["moving"] / max(1, st["samples"]), 3),
                          fast=round(st["fast"] / max(1, st["samples"]), 3))
        return dict(players=per, duels=self.duels, bumps=self.bumps, play_s=round(self.play_s, 1))


class Game:
    """The match: world, 10 players + referee, the GameController (states, rules, clock, score, events)."""

    def __init__(self, seed=0, record=False, half_s=HALF_S, halves=2, trace=False, log=None):
        self.seed, self.record, self.half_s, self.halves, self.trace = seed, record, half_s, halves, trace
        self.world = w = World(ROBOTS, seed=seed, randomize=True)
        self.rng = np.random.default_rng(5000 + seed)
        self.teams = {"blue": Team("blue", 1.0), "yellow": Team("yellow", -1.0)}
        self.players, self.rts = [], []
        for dk in w.ducks:
            rt = MatchRuntime(sim=dk, seed=1000 * seed + dk.i, log=None)
            rt.reset("standing")
            self.rts.append(rt)
            if dk.team == "ref":
                self.ref = Referee(self, dk, rt)
                continue
            tm = self.teams[dk.team]
            num = int(dk.name[1:])
            p = Player(self, dk, rt, tm, ROLES[num - 1], num)
            tm.players.append(p)
            self.players.append(p)
        self.opp_err = {}
        self.tactics = {tn: TT.TeamTactics(seed * 10 + k) for k, tn in enumerate(self.teams)}
        self.ren = make_renderer(w.m)
        w.touch_hook = self._touch
        self.t, self.k = 0.0, 0
        self.half, self.clock = 1, 0.0
        self.phase, self.phase_t = "set", 0.0
        self.kickoff_team = "blue" if seed % 2 == 0 else "yellow"
        self.first_kickoff = self.kickoff_team
        self.setpiece = None
        self.out_pending = None
        self.last_touch = None           # (duck index, team, t)
        self.touch_now = set()
        self.kicks, self.events, self.frames, self.clk = [], [], [], []
        self.ball_moved_from = None
        self.log = log
        self.place_kickoff(teleport=True)
        self.metrics = Metrics(self)

    # ---------------------------------------------------------------- helpers
    def ev(self, kind, team=None, text="", **kw):
        e = dict(t=round(self.t, 2), half=self.half, clock=round(self.clock, 1), kind=kind, team=team, text=text, **kw)
        self.events.append(e)
        if self.log:
            self.log(f"[{self.t:6.1f} s | {self.half}H {int(self.clock) // 60}:{int(self.clock) % 60:02d}] {kind.upper()} {team or ''} {text}")

    def other(self, team):
        return "yellow" if team == "blue" else "blue"

    def by_index(self, i):
        for p in self.players:
            if p.duck.i == i:
                return p
        return None

    # ---------------------------------------------------------------- touches, kicks
    def _touch(self):
        w = self.world
        bg = w.ball_geom
        d = w.d
        for c in d.contact[:d.ncon]:
            if c.geom1 == bg or c.geom2 == bg:
                o = c.geom2 if c.geom1 == bg else c.geom1
                for dk in w.ducks:
                    if o in dk.geoms:
                        self.touch_now.add(dk.i)

    def process_touches(self):
        for i in sorted(self.touch_now):
            dk = self.world.ducks[i]
            if dk.team == "ref":
                continue
            p = self.by_index(i)
            new = self.last_touch is None or self.last_touch[0] != i or self.t - self.last_touch[2] > 0.5
            if new:
                p.stats["touches"] += 1
                for kc in reversed(self.kicks):            # pass result: who touched the ball next
                    if kc["kind"] == "pass" and kc.get("result") is None and kc["who"] != i and self.t - kc["t"] < 6.0:
                        kc["result"] = "completed" if dk.team == kc["team"] else "intercepted"
                        if kc["result"] == "completed":
                            self.by_index(kc["who"]).stats["passes_completed"] += 1
                        break
            self.last_touch = (i, dk.team, self.t)
            sp = self.setpiece
            if sp is not None and dk.team == sp["team"]:
                sp["touched"] = True
        self.touch_now = set()

    def kick(self, p, t, foot, kind, mate):
        b, _ = self.world.ball()
        kind = kind or "forward"
        mate_i = next((q.duck.i for q in self.players if q.name == mate), None) if mate else None
        self.kicks.append(dict(t=round(t, 2), who=p.duck.i, name=p.name, team=p.team.name, foot=foot, kind=kind,
                               mate=mate_i, b0=[round(float(v), 3) for v in b[:2]], vmax=0.0, connected=False))
        p.stats["kicks"] += 1
        if kind in ("shot", "pass", "clearance"):
            p.stats[{"shot": "shots", "pass": "passes", "clearance": "clearances"}[kind]] += 1

    # ---------------------------------------------------------------- set-ups
    def spot(self, team, role, kicking):
        x, y = KICKOFF_SPOTS[kicking][role]
        return np.array([x, y]) * team.dir

    def place_kickoff(self, teleport):
        """Ball on the centre spot; with teleport every duck is put on its kick-off spot (start, half-time)."""
        w = self.world
        w.set_ball(0.0, 0.0)
        for tm in self.teams.values():
            tm.announce((0.0, 0.0), self.t)
        if teleport:
            for p in self.players:
                s = self.spot(p.team, p.role, p.team.name == self.kickoff_team)
                w.put(p.duck, s[0], s[1], 0.0 if p.team.dir > 0 else math.pi)
                self.reset_rt(p.rt)
            w.put(self.ref.duck, 0.3, -HY - 0.25, math.pi / 2)
            self.reset_rt(self.ref.rt)
        mujoco.mj_forward(w.m, w.d)
        for p in self.players:
            p.update_pose()

    def reset_rt(self, rt):
        rt.sim.reset_state()
        rt.mode, rt.plan = "stand", [("velstand", None)]
        rt.plan_all = list(rt.plan)
        rt.vel[:] = 0
        rt.vel_goal[:] = 0
        rt.head_cmd[:] = 0
        rt.head_goal[:] = 0
        rt.yaw_est = 0.0
        rt._start("velstand")

    def manual_placement(self):
        """SET: ducks not on a legal spot (more than 0.35 m off, or down) are carried there (RoboCup manual placement)."""
        w, moved = self.world, []
        for p in self.players:
            if not p.active():
                continue
            s = self.spot(p.team, p.role, p.team.name == self.kickoff_team)
            q, _ = p.duck.pose()
            if np.linalg.norm(q - s) > 0.35 or p.rt.lying() or p.duck.tilt_deg() > 25:
                w.put(p.duck, s[0], s[1], 0.0 if p.team.dir > 0 else math.pi)
                self.reset_rt(p.rt)
                moved.append(p.name)
        if moved:
            self.ev("placement", None, "manual placement: " + ", ".join(moved), who=moved)
        w.set_ball(0.0, 0.0)
        mujoco.mj_forward(w.m, w.d)

    def penalize(self, p, why):
        """RoboCup 'incapable player': taken off for PICKUP_S, then back in at the touch line in its own half."""
        tm = p.team
        n = sum(1 for q in self.players if not q.active())
        x = -tm.dir * (0.6 + 0.35 * n)
        side = -1.0 if self.ref.side > 0 else 1.0            # the side away from the referee
        self.world.put(p.duck, x, side * (HY + 0.25), -side * math.pi / 2)
        self.reset_rt(p.rt)
        p.penalized_until = self.t + PICKUP_S
        p.stats["pickups"] += 1
        p.down_since = p.out_since = None
        self.ev("pickup", tm.name, f"{p.name} {why}: off for {PICKUP_S:.0f} s", who=p.name)

    # ---------------------------------------------------------------- tactics (tactics.py: pure Python)
    def world_model(self, tm):
        """What tactics.TeamTactics gets: this team's own estimates (poses with localization error, own camera / team
        ball), opponents as perceived (truth + a drifting OPP_NOISE error, OPP_TAU correlation), the GameController's state."""
        t = self.t
        phase = self.phase if self.phase in ("playing", "ready", "set") else "other"
        sp = self.setpiece
        players = []
        for p in tm.players:
            b, src = p.ball(t)
            players.append(dict(name=p.name, role=p.role, pos=p.pos, psi=p.psi, active=p.active(),
                                down=p.rt.mode == "get up" or p.rt.lying(),
                                busy=p.rt.mode in ("kick left", "kick right", "hop"),
                                ball=None if b is None else np.asarray(b, float), ball_src=src,
                                ball_age=t - p.ball_t))
        opps = []
        a = rh.DT / OPP_TAU
        for q in self.teams[self.other(tm.name)].players:
            e = self.opp_err.setdefault((tm.name, q.name), self.rng.normal(0, OPP_NOISE, 2))
            e += -a * e + math.sqrt(2 * a) * self.rng.normal(0, OPP_NOISE, 2)
            opps.append(dict(pos=q.duck.pose()[0] + e, role=q.role, active=q.active()))
        return dict(t=t, phase=phase, dir=tm.dir, ball=None if tm.ball is None or t - tm.ball_t > 4.0 else tm.ball,
                    setpiece=None if sp is None else dict(kind=sp["kind"], ours=sp["team"] == tm.name,
                                                          ball=sp["ball"], radius=sp["radius"]),
                    spots={p.name: self.spot(tm, p.role, tm.name == self.kickoff_team) for p in tm.players},
                    players=players, opps=opps)

    def apply(self, p, c, t):
        """One player's tactics.Cmd -> its runtime (velstand twist, head) and, on a kick, the kick brain."""
        p.cmd, p.state = c, c.state
        p.cam_on = bool(c.cam) and p.active()
        if c.head is not None:
            p.rt.look(*c.head)
        if c.kick:
            self.kick(p, t, c.kick, c.kind, c.mate)
            p.rt.command(f"kick {c.kick}")
            return
        p.rt.drive(c.vx, c.vy, c.wz)

    # ---------------------------------------------------------------- the GameController
    def award(self, kind, team, pos):
        """Set piece: the referee puts the ball at pos; the other team keeps FREEKICK_D away for FREEKICK_S."""
        w = self.world
        w.set_ball(pos[0], pos[1])
        mujoco.mj_forward(w.m, w.d)
        for tm in self.teams.values():
            tm.announce(pos, self.t)
        self.setpiece = dict(kind=kind, team=team, ball=list(pos), until=self.t + FREEKICK_S, radius=FREEKICK_D, touched=False)
        self.ev(kind, team, f"{kind} {team or ''}".strip(), pos=[round(float(v), 2) for v in pos])

    def rules(self):
        """Once per tick, on the simulator's truth (the GameController / referee)."""
        w, t = self.world, self.t
        b, bv = w.ball()
        if self.phase == "playing":
            self.clock += rh.DT
            # kick-off / set piece over?
            sp = self.setpiece
            if sp is not None:
                moved = np.linalg.norm(b[:2] - np.asarray(sp["ball"])) > 0.15
                if t > sp["until"] or moved or (sp["touched"] and sp["kind"] != "kick-off"):
                    self.setpiece = None
            # no progress: dropped ball (both teams 0.5 m away for 3 s)
            self.prog = getattr(self, "prog", (t, b[:2].copy()))
            if np.linalg.norm(b[:2] - self.prog[1]) > 0.15 or self.setpiece is not None:
                self.prog = (t, b[:2].copy())
            elif t - self.prog[0] > NO_PROGRESS_S and self.out_pending is None:
                pos = (float(np.clip(b[0], -HX + 0.3, HX - 0.3)), float(np.clip(b[1], -HY + 0.3, HY - 0.3)))
                self.award("dropped ball", None, pos)
                self.setpiece["until"] = t + 3.0
                self.prog = (t, np.array(pos))
            # goal / out
            if self.out_pending is None:
                gx = HX + LINE_W / 2 + BALL_R
                if abs(b[0]) > gx and abs(b[1]) < GOAL_W / 2 and b[2] < GOAL_H:
                    scorer_team = "blue" if (b[0] > 0) == (self.teams["blue"].dir > 0) else "yellow"
                    return self.goal(scorer_team)
                if abs(b[1]) > HY + LINE_W / 2 + BALL_R:
                    lt = self.last_touch[1] if self.last_touch else None
                    team = self.other(lt) if lt in self.teams else ("blue" if b[0] * self.teams["blue"].dir < 0 else "yellow")
                    x = float(np.clip(b[0], -HX + 0.05, HX - 0.05))
                    self.out_pending = (t, "throw-in", team, (x, math.copysign(HY - 0.02, b[1])))
                    self.ev("whistle", None, "ball out over the touch line")
                elif abs(b[0]) > gx:
                    defending = "blue" if (b[0] < 0) == (self.teams["blue"].dir > 0) else "yellow"
                    lt = self.last_touch[1] if self.last_touch else None
                    sx, sy = math.copysign(1, b[0]), math.copysign(1, b[1])
                    if lt == defending:
                        self.out_pending = (t, "corner kick", self.other(defending), (sx * (HX - 0.03), sy * (HY - 0.03)))
                    else:
                        self.out_pending = (t, "goal kick", defending, (sx * (HX - GOAL_AREA[0]), sy * GOAL_AREA[1] / 2))
                    self.ev("whistle", None, "ball out over the goal line")
            elif t - self.out_pending[0] > OUT_WHISTLE_S:
                _, kind, team, pos = self.out_pending
                self.out_pending = None
                self.award(kind, team, pos)
            # incapable players
            for p in self.players:
                if not p.active():
                    if t >= p.penalized_until:
                        p.penalized_until = None
                        self.ev("return", p.team.name, f"{p.name} back in", who=p.name)
                    continue
                q, _ = p.duck.pose()
                down = p.rt.lying() or p.duck.tilt_deg() > 25
                p.down_since = (p.down_since if p.down_since is not None else t) if down else None
                out = abs(q[0]) > HX + 0.35 or abs(q[1]) > HY + 0.35
                p.out_since = (p.out_since if p.out_since is not None else t) if out else None
                if p.down_since is not None and t - p.down_since > PICKUP_DOWN_S:
                    self.penalize(p, "down too long")
                elif p.out_since is not None and t - p.out_since > PICKUP_OUT_S:
                    self.penalize(p, "left the field")
            if self.clock >= self.half_s:
                self.end_half()
        elif self.phase == "goal":
            if t - self.phase_t > GOAL_S:
                self.phase, self.phase_t = "ready", t
                w.set_ball(0.0, 0.0)
                for tm in self.teams.values():
                    tm.announce((0.0, 0.0), t)
                self.ev("ready", self.kickoff_team, "players to kick-off positions")
        elif self.phase == "ready":
            done = all(np.linalg.norm(p.duck.pose()[0] - self.spot(p.team, p.role, p.team.name == self.kickoff_team)) < 0.25
                       for p in self.players if p.active())
            if t - self.phase_t > READY_S or (done and t - self.phase_t > 2.0):
                self.phase, self.phase_t = "set", t
                self.manual_placement()
                for tm in self.teams.values():
                    tm.announce((0.0, 0.0), t)
        elif self.phase == "set":
            if t - self.phase_t > SET_S:
                self.phase, self.phase_t = "playing", t
                self.setpiece = dict(kind="kick-off", team=self.kickoff_team, ball=[0.0, 0.0], until=t + KICKOFF_S,
                                     radius=CIRCLE_R, touched=False)
                self.ev("kick-off", self.kickoff_team, f"kick-off {self.kickoff_team}")

    def goal(self, team):
        lt = self.last_touch
        scorer = self.by_index(lt[0]) if lt is not None else None
        tm = self.teams[team]
        tm.score += 1
        own = scorer is not None and scorer.team is not tm
        if scorer is not None:
            scorer.stats["own_goals" if own else "goals"] += 1
        sc = f"{self.teams['blue'].score}:{self.teams['yellow'].score}"
        self.ev("goal", team, f"GOAL {team}" + (f" ({scorer.name}{', own goal' if own else ''})" if scorer else "") + f", {sc}",
                scorer=scorer.name if scorer else None, own_goal=own, score=[self.teams["blue"].score, self.teams["yellow"].score])
        self.phase, self.phase_t = "goal", self.t
        self.setpiece = None
        self.out_pending = None
        self.kickoff_team = self.other(team)
        if scorer is not None and not own:
            sp = scorer.duck.pose()[0]
            for p in tm.players:
                if p.active() and np.linalg.norm(p.duck.pose()[0] - sp) < 0.8 and not p.rt.lying():
                    p.rt.command("hop")
        if not self.ref.rt.lying():
            self.ref.rt.command("hop")

    def end_half(self):
        self.ev("whistle", None, f"end of half {self.half}")
        if self.half >= self.halves:
            self.phase = "end"
            self.ev("full-time", None, f"full time {self.teams['blue'].score}:{self.teams['yellow'].score}")
            return
        self.half += 1
        self.clock = 0.0
        for tm in self.teams.values():                   # change ends
            tm.dir = -tm.dir
        self.kickoff_team = self.other(self.first_kickoff)
        for p in self.players:
            p.penalized_until = None
            p.down_since = p.out_since = None
        self.setpiece = None
        self.out_pending = None
        self.place_kickoff(teleport=True)
        self.phase, self.phase_t = "set", self.t
        self.ev("half-time", None, "half-time: teams change ends")

    # ---------------------------------------------------------------- the loop
    def tick(self):
        t = self.t
        w = self.world
        for p in self.players:
            p.update_pose()
        for tm in self.teams.values():
            tm.deliver(t)
        playing = self.phase == "playing"
        cmds = {}
        for tm in self.teams.values():
            cmds.update(self.tactics[tm.name].step(self.world_model(tm)))
        for p in self.players:
            p.cam_on = bool(cmds[p.name].cam) and p.active() and playing
        if self.k % CAM_EVERY == 0 and playing:
            mujoco.mj_forward(w.m, w.d)
            for p in self.players:
                if p.cam_on:
                    p.see(render_eye(self.ren, w.d, p.duck.cam), t)
        for p in self.players:
            if p.cam_on:
                p.share(t)
        for p in self.players:
            c = cmds[p.name]
            if not (p.active() and self.phase in ("playing", "ready")):
                c.vx = c.vy = c.wz = 0.0
                c.kick = None
            self.apply(p, c, t)
        self.ref.rt.drive(*self.ref.act(t))
        for rt in self.rts:
            rt.tick_brain()
        w.physics()
        for rt in self.rts:
            rt.tick_after()
        self.process_touches()
        b, bv = w.ball()
        sp = float(np.hypot(bv[0], bv[1]))
        for kc in self.kicks[-3:]:
            if t - kc["t"] < 1.2:
                kc["vmax"] = round(max(kc["vmax"], sp), 3)
                if not kc["connected"] and kc["vmax"] > 0.3:
                    kc["connected"] = True
                    self.by_index(kc["who"]).stats["connected"] += 1
        self.rules()
        self.metrics.tick()
        if self.record and self.k % REC_EVERY == 0:
            self.frames.append(self.frame())
            self.clk.append([int(round(self.clock * 10)), self.half, "gsrpe".index(self.phase[0]) if self.phase[0] in "gsrpe" else 0])
        if self.trace and self.k % 100 == 0:
            st = " ".join(f"{p.name}:{p.state[:5]}{'*' if p.cmd is not None and p.cmd.role == 'striker' else ''}" for p in self.players)
            print(f"t {t:6.1f} {self.phase:8s} {self.half}H {self.clock:5.1f} ball {b[0]:+.2f},{b[1]:+.2f} "
                  f"score {self.teams['blue'].score}:{self.teams['yellow'].score} | {st}", flush=True)
        self.k += 1
        self.t = self.k * rh.DT

    def frame(self):
        """Replay frame: per robot the free joint (mm, quaternion x 1e4) and its 15 hinge angles (mrad), then the
        ball (mm). The page rebuilds every body by forward kinematics (kinematics())."""
        d, out = self.world.d, []
        for dk in self.world.ducks:
            q = d.qpos[dk.free_q:dk.free_q + 7]
            out += [int(round(v * 1000)) for v in q[:3]] + [int(round(v * 1e4)) for v in q[3:7]]
            out += [int(round(v * 1000)) for v in d.qpos[dk.free_q + 7:dk.free_q + 22]]
        b = d.qpos[self.world.bq:self.world.bq + 3]
        return out + [int(round(v * 1000)) for v in b]

    def play(self):
        t0 = time.time()
        while self.phase != "end":
            self.tick()
        self.wall = time.time() - t0
        return self.result()

    def result(self):
        per = {}
        for p in self.players:
            per[p.name] = dict(team=p.team.name, role=p.role, falls=p.rt.falls, unsticks=p.rt.unsticks, **p.stats,
                               ball_in_view=round(p.stats["frames_seen"] / max(1, p.stats["frames"]), 3))
        per["ref"] = dict(team="ref", role="referee", falls=self.ref.rt.falls, unsticks=self.ref.rt.unsticks)
        kk = self.kicks
        cnt = lambda kind: sum(1 for e in self.events if e["kind"] == kind)
        team = {}
        for tn in ("blue", "yellow"):
            ks = [k for k in kk if k["team"] == tn]
            team[tn] = dict(score=self.teams[tn].score, kicks=len(ks), connected=sum(k["connected"] for k in ks),
                            shots=sum(k["kind"] == "shot" for k in ks), passes=sum(k["kind"] == "pass" for k in ks),
                            passes_completed=sum(k.get("result") == "completed" for k in ks),
                            clearances=sum(k["kind"] == "clearance" for k in ks),
                            falls=sum(per[p.name]["falls"] for p in self.players if p.team.name == tn),
                            touches=sum(per[p.name]["touches"] for p in self.players if p.team.name == tn))
        sim_s = self.k * rh.DT
        return dict(seed=self.seed, score=[self.teams["blue"].score, self.teams["yellow"].score], teams=team, players=per,
                    goals=[e for e in self.events if e["kind"] == "goal"], kicks=kk,
                    throw_ins=cnt("throw-in"), goal_kicks=cnt("goal kick"), corners=cnt("corner kick"), pickups=cnt("pickup"),
                    placements=cnt("placement"), events=self.events, sim_s=round(sim_s, 1),
                    playing_s=round(self.half_s * self.halves, 1), wall_s=round(getattr(self, "wall", 0.0), 1),
                    rtf=round(sim_s / max(1e-6, getattr(self, "wall", 1.0)), 3),
                    bodies={dk.name: dk.dr for dk in self.world.ducks}, metrics=self.metrics.result())


def kinematics():
    """The duck's kinematic tree for the page (one robot, body order as in the model): parent, body pos/quat in the
    parent, hinge axis/anchor; the replay frames carry the free joint + 15 hinge angles per robot."""
    m = mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()
    out = dict(bodies=[], parent=[], pos=[], quat=[], axis=[], anchor=[])
    for b in range(1, m.nbody):
        out["bodies"].append(m.body(b).name)
        out["parent"].append(int(m.body_parentid[b]) - 1)
        out["pos"].append([round(float(v), 6) for v in m.body_pos[b]])
        out["quat"].append([round(float(v), 6) for v in m.body_quat[b]])
        j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        if m.jnt_type[j[0]] == mujoco.mjtJoint.mjJNT_FREE:
            out["axis"].append(None)
            out["anchor"].append(None)
        else:
            assert len(j) == 1 and m.jnt_qposadr[j[0]] == 7 + len([a for a in out["axis"] if a is not None])
            out["axis"].append([round(float(v), 6) for v in m.jnt_axis[j[0]]])
            out["anchor"].append([round(float(v), 6) for v in m.jnt_pos[j[0]]])
    return out


def check_fk(game):
    """The page's forward kinematics (kinematics() + a frame) against MuJoCo's body poses, in mm."""
    K = kinematics()
    fr = game.frame()
    d = game.world.d
    worst = 0.0
    for r, dk in enumerate(game.world.ducks):
        a = fr[r * 22:(r + 1) * 22]
        P, Q = [], []
        hinge = 0
        for b in range(len(K["bodies"])):
            if K["parent"][b] < 0:
                p = np.array(a[:3]) / 1000
                q = np.array(a[3:7]) / 1e4
            else:
                pp, pq = P[K["parent"][b]], Q[K["parent"][b]]
                Rp = np.zeros(9)
                mujoco.mju_quat2Mat(Rp, pq)
                p = pp + Rp.reshape(3, 3) @ K["pos"][b]
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, pq, np.array(K["quat"][b]))
                ang = a[7 + hinge] / 1000
                hinge += 1
                R0 = np.zeros(9)
                mujoco.mju_quat2Mat(R0, q)
                anc = p + R0.reshape(3, 3) @ K["anchor"][b]
                qa = np.zeros(4)
                mujoco.mju_axisAngle2Quat(qa, np.array(K["axis"][b]), ang)
                q2 = np.zeros(4)
                mujoco.mju_mulQuat(q2, q, qa)
                R2 = np.zeros(9)
                mujoco.mju_quat2Mat(R2, q2)
                p, q = anc - R2.reshape(3, 3) @ K["anchor"][b], q2
            P.append(p)
            Q.append(q)
        for b, name in enumerate(K["bodies"]):
            bid = game.world.m.body(f"{dk.name}_{name}").id
            worst = max(worst, float(np.abs(d.xpos[bid] - P[b]).max()))
    return worst * 1000


def _job(seed, half_s=HALF_S, halves=2):
    g = Game(seed=seed, half_s=half_s, halves=halves)
    return g.play()


def summarize(runs):
    rows = []
    for r in runs:
        rows.append(dict(seed=r["seed"], score=r["score"],
                         kicks=sum(r["teams"][t]["kicks"] for t in r["teams"]),
                         connected=sum(r["teams"][t]["connected"] for t in r["teams"]),
                         shots=sum(r["teams"][t]["shots"] for t in r["teams"]),
                         passes=sum(r["teams"][t]["passes"] for t in r["teams"]),
                         passes_completed=sum(r["teams"][t]["passes_completed"] for t in r["teams"]),
                         falls=sum(p["falls"] for p in r["players"].values()), throw_ins=r["throw_ins"],
                         goal_kicks=r["goal_kicks"], corners=r["corners"], pickups=r["pickups"], placements=r["placements"],
                         rtf=r["rtf"], wall_s=r["wall_s"], sim_s=r["sim_s"]))
        m = r.get("metrics")
        if m:
            mv = [p["moving"] for p in m["players"].values()]
            rows[-1].update(moving=round(sum(mv) / len(mv), 3), moving_over_70=sum(v > 0.7 for v in mv),
                            dist_m=round(sum(p["dist_m"] for p in m["players"].values()) / len(mv), 1),
                            duels=m["duels"], bumps=m["bumps"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--trace", type=int, default=None)
    ap.add_argument("--t", type=float, default=HALF_S, help="playing time per half (s)")
    ap.add_argument("--halves", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results_match_hybrid.json"))
    ap.add_argument("--one", type=int, default=None, help="play one match (seed) in this process, write --out")
    ap.add_argument("--merge", nargs="*", default=None, help="merge --one result files into --out")
    a = ap.parse_args()
    if a.check:
        check()
        g = Game(seed=0)
        for _ in range(100):
            g.tick()
        print(f"page forward kinematics vs MuJoCo: max body position error {check_fk(g):.3f} mm")
        speed(5.0, 1)
        speed(5.0, 11)
        speed(5.0, 11, render=4)
        return
    if a.trace is not None:
        g = Game(seed=a.trace, half_s=a.t, halves=a.halves, trace=True, log=print)
        r = g.play()
        print(json.dumps({k: v for k, v in r.items() if k not in ("events", "kicks", "bodies", "players")}, indent=1))
        for n, p in r["players"].items():
            print(n, p)
        return
    if a.one is not None:                     # one match in this process -> its own file (merge with --merge)
        g = Game(seed=a.one, half_s=a.t, halves=a.halves, log=print)
        r = g.play()
        Path(a.out).write_text(json.dumps(r))
        print("done", r["score"], r["wall_s"], r["rtf"])
        return
    if a.merge:
        runs = [json.loads(Path(f).read_text()) for f in a.merge]
        rows = summarize(runs)
        for r in rows:
            print(r)
        Path(a.out).write_text(json.dumps(dict(summary=rows, runs=runs), indent=1))
        return
    lo, hi = (int(v) for v in a.seeds.split("-"))
    from functools import partial
    with ProcessPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(partial(_job, half_s=a.t, halves=a.halves), range(lo, hi + 1)))
    rows = summarize(runs)
    for r in rows:
        print(r)
    Path(a.out).write_text(json.dumps(dict(summary=rows, runs=runs), indent=1))


if __name__ == "__main__":
    main()
