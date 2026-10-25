"""MP4 videos of the hybrid duck's three games (hurdle course, ball follow, 1-duck soccer), for watching.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/make_videos.py sim hurdles|follow|soccer      # play the game (same seed as the results), save states
    $PY duckbody/hybrid/make_videos.py render hurdles|follow|soccer   # render the saved states -> videos/<game>.mp4
    $PY duckbody/hybrid/make_videos.py highlights                     # videos/highlights.mp4 from the three videos
    $PY duckbody/hybrid/make_videos.py all                            # all of the above, one after the other

How: the games (course.py, follow.py, soccer.py) are run unchanged; this script wraps HybridRuntime.tick (and the
ball detector / soccer Player) to save qpos, the ball and the brain's state after every 50 Hz control step. Rendering
then rebuilds the same scene (the game's MjSpec edit + nicer lights, sky and floor) and replays the saved states
offscreen at 1280x720, 30 fps (qpos interpolated between control steps), with a tracking camera, text overlays and,
for follow and soccer, the duck's own head camera ('eye') as a picture-in-picture. ffmpeg encodes H.264 (CRF 23,
yuv420p, faststart). Runs are the published ones: hurdles = the best level-3 run (course_r mode) of
results_course_hybrid.json, follow = a good run of results_follow_hybrid.json, soccer = the best match of
results_soccer_hybrid.json (shown at 2x).
The sim stage forces MUJOCO_GL=osmesa (what the published results used: the head camera images, and so the run, depend
on the renderer). States go to the scratch dir in $MAKE_VIDEOS_CACHE (default videos/.cache, not committed).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "microduck_port"))
import runtime_hybrid as rh  # noqa: E402
import course as C  # noqa: E402
import follow as F  # noqa: E402
import soccer as S  # noqa: E402

OUT = HERE / "videos"
CACHE = Path(os.environ.get("MAKE_VIDEOS_CACHE", OUT / ".cache"))
W, H, FPS = 1280, 720, 30
PIP_W, PIP_H = 320, 240
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# the runs (same seeds as the published results)
HURDLES = dict(level=3, mode="course_r")           # seed: best level-3 score in results_course_hybrid.json
FOLLOW_SEED = 2                                    # 97 % in view, ends 16 cm away, 3 hops, 0 falls
SOCCER_SPEED = 2.0


# ------------------------------------------------------------------------------------------------ recording hooks
class Recorder:
    def __init__(self):
        self.rows, self.player, self.last_det_t, self.last_det, self.k = [], None, -9.0, None, 0

    def tick(self, rt):
        d = rt.sim.d
        self.k += 1
        t = self.k * rh.DT
        row = dict(t=t, qpos=d.qpos.copy(), mocap=d.mocap_pos.copy(), mode=rt.mode, skill=rt.skill, falls=rt.falls,
                   vel=[round(float(v), 2) for v in rt.vel])
        if self.player is not None:
            row["state"] = self.player.state
        row["last_det_t"] = self.last_det_t
        row["det"] = self.last_det
        self.rows.append(row)


REC: Recorder | None = None
_tick0 = rh.HybridRuntime.tick
_detect0 = F.detect
_player_init0 = S.Player.__init__


def _tick(self):
    _tick0(self)
    if REC is not None:
        REC.tick(self)


def _detect(img):
    det = _detect0(img)
    if REC is not None and det is not None:
        REC.last_det_t = (REC.k + 1) * rh.DT           # detect runs before the tick of the same step
        REC.last_det = [round(float(v), 1) for v in det[:3]]
    return det


def _player_init(self, *a, **kw):
    _player_init0(self, *a, **kw)
    if REC is not None:
        REC.player = self


rh.HybridRuntime.tick = _tick
F.detect = _detect
S.Player.__init__ = _player_init


def simulate(game):
    global REC
    REC = Recorder()
    if game == "hurdles":
        runs = json.loads((HERE / "results_course_hybrid.json").read_text())["runs"]
        pick = max((r for r in runs if r["level"] == HURDLES["level"] and r["mode"] == HURDLES["mode"]), key=lambda r: r["score"])
        res = C.play(pick["level"], pick["seed"], pick["mode"])
        ref = {k: pick[k] for k in ("end", "t", "falls", "score")}
        got = {k: res[k] for k in ref}
    elif game == "follow":
        runs = json.loads((HERE / "results_follow_hybrid.json").read_text())["runs"]
        pick = next(r for r in runs if r["seed"] == FOLLOW_SEED)
        res = F.play(FOLLOW_SEED)
        ref = {k: pick[k] for k in ("in_view", "final_dist_cam", "hops", "falls")}
        got = {k: res[k] for k in ref}
    elif game == "soccer":
        runs = json.loads((HERE / "results_soccer_hybrid.json").read_text())["runs"]
        pick = max(runs, key=lambda r: (r["goals"], -r["falls"], r["connected"]))     # as page_data.soccer_replay
        res = S.play(pick["seed"])
        ref = {k: pick[k] for k in ("goals", "goal_t", "kicks", "connected", "falls")}
        got = {k: res[k] for k in ref}
    else:
        raise SystemExit(f"unknown game {game}")
    print(game, "seed", pick["seed"], "published", ref, "now", got, "same" if ref == got else "DIFFERENT")
    res = {k: v for k, v in res.items() if k not in ("frames", "cam", "body")}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{game}.pkl", "wb") as f:
        pickle.dump(dict(game=game, seed=pick["seed"], res=res, rows=REC.rows, same=ref == got), f)
    REC = None


# ------------------------------------------------------------------------------------------------ the look
def _look(spec):
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, H)
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.32, 0.32, 0.34]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.map.znear = 0.005
    spec.visual.map.shadowclip = 2.0
    spec.visual.rgba.haze = [0.80, 0.88, 0.96, 1.0]
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1, tex.rgb2 = [0.52, 0.72, 0.93], [0.93, 0.96, 1.0]
        elif tex.name == "groundplane":
            tex.rgb1, tex.rgb2, tex.markrgb = [0.66, 0.66, 0.62], [0.58, 0.58, 0.55], [0.74, 0.74, 0.70]
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.reflectance = 0.0
            mat.texrepeat = [8, 8]
    for lt in spec.worldbody.lights:
        lt.castshadow = True
        lt.dir = [0.35, 0.45, -1.0]
        lt.pos = [-1.0, -1.3, 3.0]
        lt.diffuse = [0.75, 0.73, 0.68]
        lt.specular = [0.15, 0.15, 0.15]
    fill = spec.worldbody.add_light()
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.dir = [-0.5, 0.6, -0.6]
    fill.pos = [1, -1, 2]
    fill.diffuse = [0.22, 0.24, 0.28]
    fill.castshadow = False


def render_model(game, data):
    spec = mujoco.MjSpec.from_file(str(rh.SCENE))
    if game == "hurdles":
        r = data["res"]
        L = C.layout(r["level"], r["seed"])
        C.scene_editor(L)(spec)
        for b in L["boxes"]:              # the visual-only markers (finish line, the black hole floor)
            if b.get("visual"):
                g = spec.worldbody.add_geom()
                g.type, g.size, g.pos, g.quat = mujoco.mjtGeom.mjGEOM_BOX, b["size"], b["pos"], b["quat"]
                c = b["color"].lstrip("#")
                g.rgba = [int(c[k:k + 2], 16) / 255 for k in (0, 2, 4)] + [1.0]
                g.contype = g.conaffinity = 0
        data["layout"] = L
    elif game == "follow":
        F.scene_editor()(spec)
    else:
        S.scene_editor()(spec)
    _look(spec)
    m = spec.compile()
    assert m.nq == len(data["rows"][0]["qpos"]), (m.nq, len(data["rows"][0]["qpos"]))
    return m


def quat_slices(m):
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = m.jnt_qposadr[j] + 3
            out.append(slice(a, a + 4))
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
            a = m.jnt_qposadr[j]
            out.append(slice(a, a + 4))
    return out


def state_at(rows, t, qs):
    """qpos / mocap interpolated at sim time t, and the row at or before t (for the text)."""
    ts = t / rh.DT - 1                     # rows[k] is the state at (k + 1) * DT
    k0 = int(np.clip(math.floor(ts), 0, len(rows) - 1))
    k1 = min(k0 + 1, len(rows) - 1)
    a = float(np.clip(ts - k0, 0, 1))
    q = (1 - a) * rows[k0]["qpos"] + a * rows[k1]["qpos"]
    for s in qs:
        n = np.linalg.norm(q[s])
        q[s] = q[s] / n if n > 1e-9 else rows[k0]["qpos"][s]
    mc = (1 - a) * rows[k0]["mocap"] + a * rows[k1]["mocap"]
    if np.linalg.norm(rows[k1]["mocap"] - rows[k0]["mocap"]) > 0.2:      # a teleport: no blending
        mc = rows[k0]["mocap"]
    return q, mc, rows[k0]


# ------------------------------------------------------------------------------------------------ overlays
_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[key]


def panel(draw, xy, lines, pad=12, fill=(14, 20, 30, 175), radius=10):
    """lines: list of (text, size, bold, rgb). Draws a rounded translucent box with the lines; returns its box."""
    x, y = xy
    widths, heights = [], []
    for text, size, bold, _ in lines:
        bb = draw.textbbox((0, 0), text, font=font(size, bold))
        widths.append(bb[2] - bb[0])
        heights.append(size + 6)
    w, h = max(widths) + 2 * pad, sum(heights) + 2 * pad - 6
    if x < 0:
        x = W + x - w
    if y < 0:
        y = H + y - h
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
    yy = y + pad
    for (text, size, bold, rgb), hh in zip(lines, heights):
        draw.text((x + pad, yy), text, font=font(size, bold), fill=rgb + (255,))
        yy += hh
    return (x, y, x + w, y + h)


WHITE, GREY, GOLD, GREEN, RED, CYAN = (245, 245, 245), (190, 198, 210), (255, 200, 60), (120, 230, 140), (255, 120, 110), (130, 210, 255)


def compose(img, lines_tl, lines_tr, lines_bl, pip=None, pip_label="", banner=None, card=None):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    if lines_tl:
        panel(dr, (20, 20), lines_tl)
    if lines_tr:
        panel(dr, (-20, 20), lines_tr)
    if lines_bl:
        panel(dr, (20, -20), lines_bl)
    if pip is not None:
        x0, y0 = W - 20 - PIP_W, H - 20 - PIP_H - 26
        dr.rounded_rectangle([x0 - 6, y0 - 6, x0 + PIP_W + 6, y0 + PIP_H + 32], radius=8, fill=(14, 20, 30, 200))
        dr.text((x0 + 2, y0 + PIP_H + 6), pip_label, font=font(16, True), fill=WHITE + (255,))
    if banner:
        bb = dr.textbbox((0, 0), banner[0], font=font(54, True))
        bw = bb[2] - bb[0]
        dr.rounded_rectangle([(W - bw) / 2 - 30, 110, (W + bw) / 2 + 30, 200], radius=16, fill=(14, 20, 30, 190))
        dr.text(((W - bw) / 2, 122), banner[0], font=font(54, True), fill=banner[1] + (255,))
    if card:
        dr.rectangle([0, 0, W, H], fill=(8, 12, 20, 140))
        sizes = [(card[0], 52, True, WHITE)] + [(s, 26, False, GREY) for s in card[1:]]
        tot = sum(s + 16 for _, s, _, _ in sizes)
        y = (H - tot) / 2
        for text, s, b, c in sizes:
            bb = dr.textbbox((0, 0), text, font=font(s, b))
            dr.text(((W - (bb[2] - bb[0])) / 2, y), text, font=font(s, b), fill=c + (255,))
            y += s + 16
    im = Image.alpha_composite(im, ov)
    if pip is not None:
        x0, y0 = W - 20 - PIP_W, H - 20 - PIP_H - 26
        im.paste(Image.fromarray(pip).convert("RGBA"), (x0, y0))
    return np.asarray(im.convert("RGB"))


def draw_det(img, det_box, color=(60, 255, 60)):
    """A box round the detected ball on the head camera image (scaled from the 160 x 120 detector frame)."""
    if det_box is None:
        return img
    u, v, r = det_box
    sx = 1.0
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([(u - r) * sx, (v - r) * sx, (u + r) * sx, (v + r) * sx], outline=color, width=2)
    return np.asarray(im)


# ------------------------------------------------------------------------------------------------ cameras
class Smooth:
    def __init__(self, tau):
        self.tau, self.v = tau, None

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.v is None:
            self.v = x.copy()
        else:
            self.v = self.v + (1 - math.exp(-dt / self.tau)) * (x - self.v)
        return self.v


def trunk_xy_yaw(m, d):
    tb = m.body("trunk_base").id
    p = d.xpos[tb]
    R = d.xmat[tb].reshape(3, 3)
    return p.copy(), math.atan2(R[1, 0], R[0, 0])


# ------------------------------------------------------------------------------------------------ rendering
def encoder(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render(game, only=None):
    """Render videos/<game>.mp4; with only=[frame indices], save just those frames as PNGs in the cache (preview)."""
    data = pickle.loads((CACHE / f"{game}.pkl").read_bytes())
    m = render_model(game, data)
    d = mujoco.MjData(m)
    rows, res = data["rows"], data["res"]
    ren = mujoco.Renderer(m, H, W)
    pip_ren = mujoco.Renderer(m, PIP_H, PIP_W) if game != "hurdles" else None
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    qs = quat_slices(m)
    t_end = rows[-1]["t"]
    speed = SOCCER_SPEED if game == "soccer" else 1.0
    intro_s, outro_s = 2.0, 3.0
    n_main = int(t_end / speed * FPS)
    n = int(intro_s * FPS) + n_main + int(outro_s * FPS)
    look_s, dist_s, az_s = Smooth(0.35), Smooth(0.8), Smooth(1.5)
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    ball_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    path = OUT / f"{game}.mp4"
    enc = encoder(path) if only is None else None
    dt = 1.0 / FPS
    titles = dict(hurdles="The hurdle course", follow="Follow the ball", soccer="Duck soccer")
    seed = data["seed"]
    for i in range(n):
        tv = (i - intro_s * FPS) / FPS
        t = float(np.clip(tv * speed, rh.DT, t_end))
        q, mc, row = state_at(rows, t, qs)
        d.qpos[:] = q
        if m.nmocap:
            d.mocap_pos[:] = mc
        mujoco.mj_forward(m, d)
        p, yaw = trunk_xy_yaw(m, d)
        ball = d.xpos[ball_b].copy() if ball_b >= 0 else None
        # camera per game
        if game == "hurdles":
            la = look_s([p[0] + 0.10, p[1] * 0.6, max(p[2] - 0.04, 0.03)], dt)
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = la, 1.15, 118.0, -26.0
        elif game == "follow":
            sep = float(np.hypot(*(ball[:2] - p[:2])))
            mid = p + 0.45 * (ball - p)
            la = look_s([mid[0], mid[1], 0.06], dt)
            cam.lookat[:] = la
            cam.distance = float(dist_s([np.clip(0.85 + 0.9 * sep, 0.95, 2.0)], dt)[0])
            # azimuth follows the duck's heading slowly: a front-right 3/4 view (unwrapped)
            want = math.degrees(yaw) + 140.0
            if az_s.v is not None:
                want = az_s.v[0] + ((want - az_s.v[0] + 180) % 360 - 180)
            cam.azimuth, cam.elevation = float(az_s([want], dt)[0]), -28.0
        else:
            sep = float(np.hypot(*(ball[:2] - p[:2])))
            mid = 0.5 * (p + ball)
            la = look_s([np.clip(mid[0], -0.55, 0.75), np.clip(0.6 * mid[1], -0.3, 0.3), 0.0], dt)
            cam.lookat[:] = la
            cam.distance = float(dist_s([np.clip(1.25 + 0.7 * sep, 1.55, 2.4)], dt)[0])
            cam.azimuth, cam.elevation = 80.0, -34.0
        if only is not None and i not in only:
            if i > max(only):
                break
            continue
        ren.update_scene(d, camera=cam, scene_option=opt)
        img = ren.render()
        pip = None
        if pip_ren is not None:
            pip_ren.update_scene(d, camera=cam_id)
            pip = pip_ren.render()
        tl, tr, bl, banner, card = [], [], [], None, None
        tl = [(titles[game], 30, True, WHITE), ("Hybrid duck: owner's head on Pollen Robotics' Microduck body, MuJoCo", 16, False, GREY)]
        brain = f"brain: {row['skill']} ({row['mode']})"
        tw = f"twist vx {row['vel'][0]:+.2f} vy {row['vel'][1]:+.2f} wz {row['vel'][2]:+.2f}"
        if game == "hurdles":
            L = data["layout"]
            stage = C.stage_at(L, float(q[0]))
            tr = [(f"t = {t:5.1f} s", 30, True, WHITE), (f"Level {res['level']}  ·  seed {seed}", 18, False, GREY),
                  (f"falls: {row['falls']}", 18, True, RED if row["falls"] else GREEN)]
            bl = [(f"stage: {stage}", 22, True, GOLD), (brain, 18, False, WHITE), (tw, 16, False, GREY)]
            if i < intro_s * FPS:
                card = ["The hurdle course, level 3", "steps 2 cm · ramps 10 and 15 deg · bars up to 2.5 cm · rough ground",
                        "a 12 cm beam · a hole to walk round · 4 shoves of 0.5 m/s",
                        f"body {seed} (randomized), the best level-3 run of 10"]
            elif tv > n_main / FPS:
                card = [f"Finished in {res['t']:.1f} s" if res["finished"] else res["end"],
                        f"{res['falls']} fall(s), got up by itself  ·  score {res['score']}",
                        "level 3 over 10 bodies: 7/10 finish (median 63 s)"]
        elif game == "follow":
            lost = row["t"] - row["last_det_t"] > F.LOST_S
            st = "hop!" if row["mode"] == "hop" else ("searching (turn + look)" if lost else "tracking the ball")
            tr = [(f"t = {t:5.1f} s", 30, True, WHITE), (f"seed {seed}", 18, False, GREY)]
            bl = [(f"state: {st}", 22, True, RED if lost else GOLD), (brain, 18, False, WHITE), (tw, 16, False, GREY)]
            if i < intro_s * FPS:
                card = ["Follow the ball", "the duck only sees through its head camera (160 x 120, 10 Hz)",
                        "colour blob -> head looks at it, body turns and walks to 15 cm, hops when it is close",
                        f"body {seed} (randomized), 60 s"]
            elif tv > n_main / FPS:
                card = ["Done", f"ball in view {100 * res['in_view']:.0f} % of the time, ends {100 * res['final_dist_cam']:.0f} cm away",
                        f"{res['hops']} hops, {res['falls']} falls"]
        else:
            goals = sum(g <= t for g in res["goal_t"])
            kicks = [k for k in res["kick_log"] if k["t"] <= t]
            st = row.get("state", "")
            tr = [(f"GOALS {goals}", 34, True, GOLD), (f"t = {t:5.1f} s / 120", 20, True, WHITE),
                  (f"kicks {sum(k['connected'] for k in kicks)}/{len(kicks)} connected", 16, False, GREY),
                  (f"{SOCCER_SPEED:g}x speed", 16, True, CYAN)]
            bl = [(f"state: {st}", 22, True, GOLD), (brain, 18, False, WHITE), (tw, 16, False, GREY)]
            if any(0 <= t - g < 3.0 for g in res["goal_t"]):
                banner = ("GOAL!", GOLD)
            if i < intro_s * FPS:
                card = ["Duck soccer: one duck, one goal", "finds the ball and the goal with its head camera, walks round behind the ball,",
                        "lines it up and kicks with Microduck's ball_kick brains",
                        f"body {seed} (randomized), 2-minute match shown at {SOCCER_SPEED:g}x"]
            elif tv > n_main / FPS:
                card = [f"{res['goals']} goals in 2 minutes", f"{res['connected']}/{res['kicks']} kicks connected, {res['falls']} falls",
                        "over 10 matches: median 3 goals, 9/10 matches score"]
        if pip is not None:
            det_box = None
            if row["t"] - row["last_det_t"] <= F.LOST_S:
                d2 = _detect0(pip)                    # the duck's detector, run on the inset image (so the box sits on it)
                det_box = d2[:3] if d2 else None
            pip = draw_det(pip, det_box)
        frame = compose(img, tl, tr, bl if card is None else [], pip if card is None else None, "duck's eye (head camera)",
                        banner if card is None else None, card)
        if only is not None:
            Image.fromarray(frame).save(CACHE / f"prev_{game}_{i}.png")
            continue
        enc.stdin.write(frame.tobytes())
        if i % 150 == 0:
            print(game, f"frame {i}/{n}", flush=True)
    ren.close()
    if only is not None:
        return
    enc.stdin.close()
    enc.wait()
    if pip_ren is not None:
        pip_ren.close()
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


# ------------------------------------------------------------------------------------------------ highlights
def highlights():
    """~45 s: the hurdles at 2x, 12 s of following, and the soccer around its first two goals."""
    soc = pickle.loads((CACHE / "soccer.pkl").read_bytes())["res"]
    g = soc["goal_t"]
    parts = [("hurdles.mp4", 2.0, None, 2.0), ("follow.mp4", 2.0 + 8.0, 14.0, 1.0)]
    # soccer video time of a sim time: intro 2 s + t / speed
    for gt in g[:2]:
        a = 2.0 + max(0.0, gt - 14.0) / SOCCER_SPEED
        parts.append(("soccer.mp4", a, 10.0, 1.0))
    tmp = CACHE / "hl"
    tmp.mkdir(parents=True, exist_ok=True)
    listf = tmp / "list.txt"
    names = []
    for i, (name, a, dur, sp) in enumerate(parts):
        out = tmp / f"p{i}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a}", "-i", str(OUT / name)]
        if dur:
            cmd += ["-t", f"{dur}"]
        vf = "setpts=PTS/2" if sp == 2.0 else "setpts=PTS"
        if sp == 2.0:
            vf += ",drawtext=fontfile=%s:text='2x speed':x=w-tw-28:y=h-th-28:fontsize=26:fontcolor=0x82d2ff:box=1:boxcolor=0x0e141ebb:boxborderw=10" % FONT_B
        cmd += ["-vf", vf, "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True)
        names.append(out)
    listf.write_text("".join(f"file '{p}'\n" for p in names))
    path = OUT / "highlights.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listf), "-c:v", "libx264",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    what = sys.argv[1]
    games = sys.argv[2:] or ["hurdles", "follow", "soccer"]
    if what in ("sim", "all"):
        for g in games:
            simulate(g)
    if what in ("render", "all"):
        for g in games:
            render(g)
    if what == "preview":         # preview GAME i,j,k: those frames as PNGs in the cache
        render(sys.argv[2], only=[int(v) for v in sys.argv[3].split(",")])
    if what in ("highlights", "all"):
        highlights()


if __name__ == "__main__":
    main()
