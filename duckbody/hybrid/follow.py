"""Follow a ball with the head camera: the hybrid duck sees a coloured ball, tracks it with the head, turns and
walks to it, stops ~15 cm away and hops when it gets there; when the ball goes behind it, it searches by turning.

    PY=/home/user/cool-websites/.venv/bin/python
    MUJOCO_GL=osmesa $PY duckbody/hybrid/follow.py --seeds 0-9 [--jobs 3]     # 10 bodies / ball paths, scored
    MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --replays           # replays (+ camera inset) for the page

Camera. Microduck's MJCF has a `head_camera` in the head, but in the hybrid it sits inside the owner's (longer)
head shell and its frame looks backwards, so the sim adds `eye`: on the front of the owner's head, 3 mm out of
the shell, above the bill, looking along the bill (about 10 deg down when the head is level), vertical field of
view 49 deg like the Pi camera module v2 (62 x 49 deg). Rendered at 160 x 120, 10 Hz (MuJoCo, OSMesa here).

Detection (what would run on the Pi Zero 2 W): OpenCV HSV threshold for the ball's magenta, a 3x3 opening, the
largest blob, its enclosing circle -> centre (u, v) and radius r in pixels. 0.2 ms per frame on this desktop CPU
(a Pi Zero 2 W is ~10-20x slower: a few ms, fine at 10 Hz). Next step would be a tiny CNN detector (e.g. a 96x96 int8 SSD/FOMO-style net, a few ms on the Pi)
for toys that aren't one solid colour.

Control (10 Hz vision, 50 Hz brain, runtime_hybrid.HybridRuntime.drive + look):
    where   the pixel's ray, turned into the trunk frame with the camera pose from the head joints (forward
            kinematics; encoders on the robot) -> bearing and elevation of the ball; between frames the bearing
            is corrected by the IMU yaw the body has turned since. Distance from the ball's pixel radius
            (known 3 cm radius), kept from the last full view while the ball is cut by the image edge.
    head    look(bearing, elevation - 10 deg): the head points straight at it (no integrating servo: with the
            head's lag that overshot and lost the ball, 25 % in view vs 95 %)
    body    wz = 1.5 x bearing (max 1 rad/s); vx = 1.2 x (distance - 0.15 m) in -0.3..0.8, at least 0.15 when
            it should move (velstand does not step below ~0.1), only when facing it (bearing < 0.5 rad)
    lost    after 0.6 s unseen: if it was close, step back and look down; then turn on the spot (0.9 rad/s)
            towards where it was last seen, the head looking that way and nodding up and down
    reached within 0.18 m for 1.5 s with the ball still: a celebration hop (the runtime's searched hop)
Ball paths (per seed): rolls around at 4-12 cm/s (never into the duck) with stops of 3-5 s, once rolls fast
(0.7 m/s) round the duck's side to 0.5 m behind it, and stops for the last 8 s. 60 s per run. Scored on the
simulator's ground truth: % of camera frames with the ball detected, final distance camera -> ball.
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
from harness import MicroduckSim  # noqa: E402

BALL_R = 0.03
BALL_RGB = (0.95, 0.15, 0.65)       # magenta: not in the scene (floor blue/grey, bill orange, body white/grey)
W, H, FOVY = 160, 120, 49.0
CAM_HZ = 10
EYE_POS = (0.020, 0.0, -0.096)      # in the head link (jaw_soft): front of the owner's head, above the bill


def _eye_for_body(eye):
    """A smaller head (build_hybrid.py k < 1) carries the eye on its shell: scale it about the shell centre."""
    rep = json.loads((HERE / "head_report.json").read_text()).get(rh.ROBOT.name, {})
    if "eye_pos_head_frame" in rep:                       # a head shape (build_shapes.py) places its own eye camera
        return tuple(rep["eye_pos_head_frame"])
    k, c = rep.get("head_k", 1.0), np.array(rep.get("shell_centre_head_frame_m", (0.0, 0.0, 0.0)))
    return tuple(float(v) for v in c + k * (np.array(eye) - c))


EYE_POS = _eye_for_body(EYE_POS)
EYE_QUAT = (0.7071068, 0.0, 0.0, -0.7071068)   # camera -z = head -z (forward), camera up = head +x
T_RUN = 60.0
STOP_D = 0.15                       # wanted distance camera -> ball centre
K_TURN, K_FWD, V_MAX, V_MIN, V_BACK = 1.5, 1.2, 0.8, 0.15, -0.3
LOST_S, SEARCH_WZ = 0.6, 0.9
HOP_NEAR, HOP_HOLD_S = 0.18, 1.5
CAM_DOWN = 0.17                     # the eye looks ~10 deg down at head pitch 0 (standing)


def scene_editor():
    import mujoco

    def edit(spec):
        head = spec.body("jaw_soft")
        cam = head.add_camera()
        cam.name = "eye"
        cam.pos = list(EYE_POS)
        cam.quat = list(EYE_QUAT)
        cam.fovy = FOVY
        ball = spec.worldbody.add_body()
        ball.name = "ball"
        ball.mocap = True
        ball.pos = [0.7, 0, BALL_R]
        g = ball.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [BALL_R, 0, 0]
        g.rgba = list(BALL_RGB) + [1.0]
        g.contype = 0
        g.conaffinity = 0
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 640)
    return edit


# ---------------------------------------------------------------------------------------------- detection
def detect(img_rgb):
    """Magenta ball: HSV threshold, largest blob, enclosing circle. Returns (u, v, r_px, area) or None."""
    import cv2
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (140, 90, 60), (172, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 6:
        return None
    pts = cv2.findNonZero((labels == i).astype(np.uint8))
    (u, v), r = cv2.minEnclosingCircle(pts)
    return float(u), float(v), float(max(r, 1.0)), area


def pixel_angles(u, v):
    """Angles of a pixel off the optical axis: (right +, down +) in rad."""
    fy = (H / 2) / math.tan(math.radians(FOVY) / 2)
    return math.atan((u - W / 2) / fy), math.atan((v - H / 2) / fy)


def distance_from_radius(r_px, u, v):
    fy = (H / 2) / math.tan(math.radians(FOVY) / 2)
    ang = math.atan(r_px / fy)
    return BALL_R / max(math.sin(ang), 1e-3)


# ---------------------------------------------------------------------------------------------- ball paths
def ball_path(seed):
    """A scripted path: list of (t_start, kind, args). Kinds: 'to' (x, y, speed), 'wait', 'behind' (speed)."""
    r = np.random.default_rng(500 + seed)
    a0 = r.uniform(-0.5, 0.5)
    d0 = r.uniform(0.6, 0.9)
    start = (d0 * math.cos(a0), d0 * math.sin(a0))
    plan = [("wait", r.uniform(2.0, 3.0))]
    t_behind = r.uniform(24, 34)
    t = plan[0][1]
    while t < T_RUN - 8:
        if t_behind is not None and t >= t_behind:
            plan.append(("behind", 0.7))
            t += 4.0
            plan.append(("wait", r.uniform(3.0, 5.0)))
            t += plan[-1][1]
            t_behind = None
            continue
        ang, dist = r.uniform(-math.pi, math.pi), r.uniform(0.3, 0.6)
        plan.append(("move", (ang, dist, r.uniform(0.04, 0.12))))
        t += dist / plan[-1][1][2]
        plan.append(("wait", r.uniform(3.0, 5.0)))
        t += plan[-1][1]
    plan.append(("wait", 99.0))
    return start, plan


def _seg_dist(a, b, p):
    ab = b - a
    s = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
    return float(np.linalg.norm(a + s * ab - p))


class Ball:
    def __init__(self, seed, area=1.6):
        self.start, self.plan = ball_path(seed)
        self.p = np.array(self.start, float)
        self.k, self.t_k, self.target, self.speed, self.via = 0, 0.0, None, 0.0, []
        self.area = area
        self.moving = False

    def step(self, t, dt, duck_xy, duck_yaw):
        if self.target is None and not self.via:
            kind, arg = self.plan[min(self.k, len(self.plan) - 1)]
            if kind == "wait":
                self.moving = False
                if t - self.t_k >= arg:
                    self.k, self.t_k = self.k + 1, t
                return self.p
            if kind == "move":     # roll somewhere, but not into / under the duck (stays >= 0.3 m from it)
                ang, dist, spd = arg
                for i in range(24):
                    a = ang + i * 0.5
                    tgt = np.clip(self.p + dist * np.array([math.cos(a), math.sin(a)]), -self.area, self.area)
                    if _seg_dist(self.p, tgt, duck_xy) >= 0.3:
                        break
                self.via, self.speed = [tgt], spd
            elif kind == "behind":       # around the duck's side to 0.5 m behind it
                hd = np.array([math.cos(duck_yaw), math.sin(duck_yaw)])
                side = np.array([-hd[1], hd[0]]) * (1 if (self.p - duck_xy) @ np.array([-hd[1], hd[0]]) >= 0 else -1)
                self.via = [duck_xy + 0.45 * side + 0.05 * hd, duck_xy - 0.5 * hd + 0.15 * side]
                self.speed = arg
            self.k, self.t_k = self.k + 1, t
        self.moving = True
        tgt = self.via[0]
        d = tgt - self.p
        n = float(np.linalg.norm(d))
        stepl = self.speed * dt
        if n <= stepl:
            self.p = tgt.copy()
            self.via.pop(0)
            if not self.via:
                self.target = None
                self.t_k = t
        else:
            self.p = self.p + d / n * stepl
        return self.p


# ---------------------------------------------------------------------------------------------- one run
def play(seed=0, record=False, t_run=T_RUN, cam_every=2, trace=False):
    import mujoco
    import cv2
    sim = MicroduckSim(str(rh.SCENE), seed=seed, randomize=True, edit=scene_editor())
    rt = rh.HybridRuntime(sim=sim, seed=seed, log=None)
    rt.reset("standing")
    sim.d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(sim.m, sim.d)
    ren = mujoco.Renderer(sim.m, H, W)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    ball = Ball(seed)
    mid = sim.m.body("ball").mocapid[0]
    cam_id = sim.m.camera("eye").id
    trunk = sim.trunk
    every = rh.HZ // CAM_HZ
    seen_t, last_bearing, det = -9.0, 0.0, None
    f_bear = f_elev = f_dist = f_yaw = 0.0
    near_t, hopped_at = 0.0, -99.0
    log = dict(seen=0, frames=0, seen_behind=0, frames_behind=0, hops=0, reacquire=[], dist_cam=[], dist_trunk=[])
    behind_since = None
    frames, balls, cams, dets = [], [], [], []
    n = int(t_run * rh.HZ)
    for k in range(n):
        t = k * rh.DT
        txy = sim.d.xpos[trunk][:2].copy()
        R = sim.d.xmat[trunk].reshape(3, 3)
        tyaw = math.atan2(R[1, 0], R[0, 0])
        bp = ball.step(t, rh.DT, txy, tyaw)
        sim.d.mocap_pos[mid] = [bp[0], bp[1], BALL_R]
        # ground truth, for scoring only
        cpos = sim.d.cam_xpos[cam_id]
        d_cam = float(np.hypot(*(bp - cpos[:2])))
        d_trunk = float(np.hypot(*(bp - txy)))
        rel = math.atan2(bp[1] - txy[1], bp[0] - txy[0]) - tyaw
        rel = (rel + math.pi) % (2 * math.pi) - math.pi
        if k % every == 0:
            mujoco.mj_forward(sim.m, sim.d)
            ren.update_scene(sim.d, camera="eye")
            img = ren.render()
            det = detect(img)
            log["frames"] += 1
            behind = abs(rel) > 2.0
            log["frames_behind"] += behind
            if det:
                log["seen"] += 1
                log["seen_behind"] += behind
                # the ray to the ball in the trunk frame: camera pose relative to the trunk is forward kinematics
                # of the head joints (encoders on the robot); then bearing (left +) and elevation (down +)
                fy = (H / 2) / math.tan(math.radians(FOVY) / 2)
                ray = sim.d.cam_xmat[cam_id].reshape(3, 3) @ np.array([(det[0] - W / 2) / fy, -(det[1] - H / 2) / fy, -1.0])
                ray = sim.d.xmat[trunk].reshape(3, 3).T @ ray
                f_bear = math.atan2(ray[1], ray[0])
                f_elev = math.atan2(-ray[2], math.hypot(ray[0], ray[1]))
                edge = det[0] - det[2] < 1 or det[0] + det[2] > W - 2 or det[1] - det[2] < 1 or det[1] + det[2] > H - 2
                if not edge or t - seen_t > LOST_S:      # a ball cut by the image edge looks small: keep the last distance
                    f_dist = distance_from_radius(det[2], det[0], det[1]) * math.cos(f_elev)   # horizontal, camera -> ball
                f_yaw = rt.yaw_est                       # IMU heading when the frame was taken
                if t - seen_t > LOST_S and behind_since is not None:
                    log["reacquire"].append(round(t - behind_since, 2))
                    behind_since = None
                seen_t = t
            if record and k % (every * cam_every) == 0:
                cams.append(_jpeg(img, det))
            if det is None and t - seen_t > LOST_S and behind_since is None:
                behind_since = seen_t
        lost = t - seen_t > LOST_S
        if not lost and seen_t >= 0:
            bearing = f_bear - (rt.yaw_est - f_yaw)       # the body has turned since the frame (IMU gyro)
            last_bearing = bearing
            dist = f_dist
            wz = float(np.clip(K_TURN * bearing, -1.0, 1.0))
            vx = float(np.clip(K_FWD * (dist - STOP_D), V_BACK, V_MAX)) if abs(bearing) < 0.5 else 0.0
            if dist > STOP_D + 0.015 and abs(bearing) < 0.5:
                vx = max(vx, V_MIN)                      # velstand does not step below ~0.1
            elif dist < STOP_D - 0.04:
                vx = min(vx, -V_MIN)                     # the ball came at the duck: step back
            if dist < STOP_D + 0.03:
                wz *= 0.5
            rt.look(bearing, f_elev - CAM_DOWN)
            near = dist < HOP_NEAR and not ball.moving
            near_t = near_t + rh.DT if near else 0.0
            if near_t > HOP_HOLD_S and t - hopped_at > 8.0 and rt.mode == "drive":
                rt.command("hop")
                hopped_at, near_t = t, 0.0
                log["hops"] += 1
        else:
            direction = 1.0 if last_bearing >= 0 else -1.0
            wz, vx = SEARCH_WZ * direction, 0.0
            if f_dist < 0.3 and t - seen_t < LOST_S + 1.5:   # lost it up close: step back, look down
                wz, vx = 0.4 * direction, -0.3
                rt.look(0.6 * direction, 0.6)
            else:
                rt.look(0.5 * direction, 0.25 + 0.25 * math.sin(2 * math.pi * 0.4 * t))
        if rt.mode != "hop":
            rt.drive(vx, 0.0, wz)
        rt.tick()
        if trace and k % 25 == 0:
            print(f"t {t:5.1f} true rel {rel:+.2f} d {d_cam:.2f} | det {None if det is None else tuple(round(x) for x in det[:3])} "
                  f"lost {lost} head {rt.head_cmd.round(2)} vx {vx:.2f} wz {wz:+.2f} mode {rt.mode} ball_moving {ball.moving}")
        if record and k % 2 == 0:
            frames.append(sim.frame())
            balls.append([round(float(bp[0]), 4), round(float(bp[1]), 4), BALL_R])
            dets.append(None if det is None or lost else [round(det[0]), round(det[1]), round(det[2], 1)])
        if t > t_run - 4.0:
            log["dist_cam"].append(d_cam)
            log["dist_trunk"].append(d_trunk)
    res = dict(seed=seed, in_view=round(log["seen"] / max(1, log["frames"]), 3),
               in_view_front=round((log["seen"] - log["seen_behind"]) / max(1, log["frames"] - log["frames_behind"]), 3),
               behind_frac=round(log["frames_behind"] / max(1, log["frames"]), 3),
               final_dist_cam=round(float(np.median(log["dist_cam"])), 3),
               final_dist_trunk=round(float(np.median(log["dist_trunk"])), 3),
               falls=rt.falls, hops=log["hops"], reacquire_s=log["reacquire"], body=sim.dr)
    if record:
        res.update(frames=frames, bodies=sim.body_names(), ball=balls, cam=cams, cam_fps=CAM_HZ / cam_every, det=dets)
    ren.close()
    return res


def _jpeg(img, det):
    import cv2
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if det:
        u, v, r, _ = det
        cv2.rectangle(bgr, (int(u - r), int(v - r)), (int(u + r), int(v + r)), (60, 255, 60), 1)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _job(seed):
    t = time.time()
    r = play(seed)
    r["wall_s"] = round(time.time() - t, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "results_follow_hybrid.json"))
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.seeds.split("-"))
    with ProcessPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(_job, range(lo, hi + 1)))
    for r in runs:
        print({k: v for k, v in r.items() if k != "body"})
    s = dict(n=len(runs), in_view_median=float(np.median([r["in_view"] for r in runs])),
             in_view_front_median=float(np.median([r["in_view_front"] for r in runs])),
             final_dist_cam_median=float(np.median([r["final_dist_cam"] for r in runs])),
             final_dist_trunk_median=float(np.median([r["final_dist_trunk"] for r in runs])),
             within_25cm=sum(r["final_dist_cam"] < 0.25 for r in runs), falls=sum(r["falls"] for r in runs),
             hops=sum(r["hops"] for r in runs),
             reacquired=sum(len(r["reacquire_s"]) > 0 for r in runs),
             reacquire_median_s=float(np.median([x for r in runs for x in r["reacquire_s"]])) if any(r["reacquire_s"] for r in runs) else None)
    print(json.dumps(s, indent=1))
    Path(a.out).write_text(json.dumps(dict(summary=s, runs=runs), indent=1))


if __name__ == "__main__":
    main()
