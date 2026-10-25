"""The hardest hurdle game: an obstacle course for the hybrid duck, in MuJoCo, driven by runtime_hybrid.py.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/course.py --levels 1,2,3 --seeds 0-9 [--mode walk|course] [--jobs 3]   # the game, scored
    $PY duckbody/hybrid/course.py --probe                                                      # single obstacles
    MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --replays                            # replays for the page

The course is a deck of boxes (top at z = 0) laid over a floor 6 cm lower, so a hole is a real hole. The
duck starts at x = 0 facing +x. Stages, in order of difficulty (level 1 numbers; levels 2 and 3 scale them):

    start      flat
    steps      1 cm step up, 30 cm on top, 1 cm step down
    ramps      5 deg up, flat, 5 deg down; then 10 deg up, flat, 10 deg down
    bars       low bars across the course: 0.5, 1.0, 1.5 cm high, 3 cm deep
    rough      a patch of random small boxes 0.3-0.8 cm high
    narrow     a 1 cm high beam 18 cm wide (the feet are ~9 cm apart); stepping off = "off the path"
    hole       a 20 x 24 cm hole in the middle (6 cm deep); the duck must walk around it
    shoves     flat, with velocity kicks (0.3 m/s, Microduck's training range) in random directions
    finish     the line

The deck is 1 m wide with 15 cm side walls (velstand's get-ups and the unstick rolls travel up to 0.5 m
sideways and tumbled over 4 and 8 cm rails); only the hole is a real drop.

Steering: pure pursuit on the course's centre line (around the hole on a side lane). It uses the simulator's
position of the trunk (x, y, yaw), standing in for what the robot would need on a real course (the head
camera following the course edges, or a tag); the yaw part is what the IMU gives. Speed per stage comes
from the runtime's COURSE profile ("course" mode) or a fixed walk ("walk" baseline).

Score (the game): 1000 x fraction of the course reached + 10 per second left of the 90 s limit when it
finishes, - 50 per fall. Falls cost time (velstand gets up, then the duck walks on); falling into the
hole ends the run where it is.

Modes: "walk" (the runtime's walk, 0.4 command, everywhere), "course" (runtime COURSE_SPEED per obstacle from the
--probe sweep, turn on the spot when the heading is off, back up when a turn is blocked) and "course_r" (+ the
IMU stumble reflex: tilt > 12 deg -> stand still 0.6 s). All three get up after falls with the runtime's unstick
sequence. Levels 2 and 3 go past what velstand was trained on (Microduck's rough terrain: steps <= 1.5 cm,
random grid <= 1 cm, slopes <= 5.7 deg).
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_hybrid as rh  # noqa: E402
from harness import MicroduckSim  # noqa: E402

FLOOR_Z = -0.06          # floor under the deck (hole depth)
DECK_W = 1.00            # course width (y): get-ups can travel 0.3 m sideways
T_MAX = 90.0
LEVELS = {   # what gets harder per level
    1: dict(step=0.010, ramps=(5.0, 10.0), bars=(0.005, 0.010, 0.015), rough=(0.003, 0.008), narrow=0.18,
            hole=(0.20, 0.24), shoves=2, shove_v=0.3),
    2: dict(step=0.015, ramps=(7.5, 12.5), bars=(0.010, 0.015, 0.020), rough=(0.005, 0.012), narrow=0.15,
            hole=(0.25, 0.30), shoves=3, shove_v=0.4),
    3: dict(step=0.020, ramps=(10.0, 15.0), bars=(0.015, 0.020, 0.025), rough=(0.008, 0.016), narrow=0.12,
            hole=(0.30, 0.36), shoves=4, shove_v=0.5),
}
STAGES = ("start", "steps", "ramps", "bars", "rough", "narrow", "hole", "shoves", "finish")
RAIL_H = 0.15            # side walls (a get-up or roll travels up to 0.5 m sideways; falls tumbled over 4 and 8 cm walls)
COLORS = dict(rail="#c8553d", deck="#3a4250", step="#6b7a8f", ramp="#5f8fb4", bar="#e0b040", rough="#8a6f5a", narrow="#7fae6f",
              hole="#000000", finish="#f2f2f2", start="#ff9b33")


def _box(kind, x0, x1, y0, y1, z0, z1, quat=(1, 0, 0, 0)):
    return dict(kind=kind, pos=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
                size=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2], quat=list(quat), color=COLORS[kind])


def _ramp(x0, x1, z0, z1, w=DECK_W, t=0.02):
    """A tilted slab whose top surface goes from (x0, z0) to (x1, z1)."""
    dx, dz = x1 - x0, z1 - z0
    a = math.atan2(dz, dx)
    L = math.hypot(dx, dz)
    n = np.array([-math.sin(a), math.cos(a)])          # top-surface normal in (x, z)
    c = np.array([(x0 + x1) / 2, (z0 + z1) / 2]) - n * t / 2
    q = (math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0)  # rotate about y by -a (pitch the slab up)
    return dict(kind="ramp", pos=[float(c[0]), 0.0, float(c[1])], size=[L / 2, w / 2, t / 2], quat=list(q),
                color=COLORS["ramp"])


def layout(level=1, seed=0):
    """Course geometry for one level and seed (rough boxes, hole side and shoves are drawn from the seed)."""
    P, r = LEVELS[level], np.random.default_rng(1000 + seed)
    boxes, stages, W = [], [], DECK_W
    x = 0.30
    stages.append(("start", -0.30, x))
    # steps: up, 30 cm on top, down
    s0 = x + 0.10
    boxes.append(_box("step", s0, s0 + 0.30, -W / 2, W / 2, 0.0, P["step"]))
    stages.append(("steps", x, s0 + 0.45)); x = s0 + 0.45
    # ramps (a) up, flat, down; (b) steeper
    for ang, run_ in zip(P["ramps"], (0.30, 0.20)):
        h = run_ * math.tan(math.radians(ang))
        x0 = x + 0.05
        boxes += [_ramp(x0, x0 + run_, 0.0, h), _box("ramp", x0 + run_, x0 + run_ + 0.15, -W / 2, W / 2, h - 0.02, h),
                  _ramp(x0 + run_ + 0.15, x0 + 2 * run_ + 0.15, h, 0.0)]
        x = x0 + 2 * run_ + 0.15 + 0.10
    stages.append(("ramps", stages[-1][2], x))
    # bars
    x0 = x
    for i, hb in enumerate(P["bars"]):
        xb = x + 0.15 + 0.20 * i
        boxes.append(_box("bar", xb - 0.015, xb + 0.015, -W / 2, W / 2, 0.0, hb))
    x = x + 0.15 + 0.20 * (len(P["bars"]) - 1) + 0.15
    stages.append(("bars", x0, x))
    # rough patch
    x0, x1 = x + 0.05, x + 0.55
    for _ in range(int(48 * (x1 - x0) / 0.5)):
        cx, cy = r.uniform(x0 + 0.02, x1 - 0.02), r.uniform(-0.25, 0.25)
        sx, sy, hz, yaw = r.uniform(0.015, 0.035), r.uniform(0.015, 0.035), r.uniform(*P["rough"]), r.uniform(0, math.pi)
        b = _box("rough", cx - sx, cx + sx, cy - sy, cy + sy, 0.0, hz, (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        boxes.append(b)
    x = x1 + 0.10
    stages.append(("rough", x0 - 0.05, x))
    # narrow beam (1 cm high)
    x0, x1, w = x + 0.10, x + 0.60, P["narrow"]
    boxes.append(_box("narrow", x0, x1, -w / 2, w / 2, 0.0, 0.010))
    narrow = (x0, x1, w)
    x = x1 + 0.10
    stages.append(("narrow", x0 - 0.10, x))
    # hole: walk around it on a side lane
    hx, hy = P["hole"]
    h0 = x + 0.40
    hole = (h0, h0 + hx, -hy / 2, hy / 2)
    side = 1.0 if r.random() < 0.5 else -1.0
    lane = side * (hy / 2 + (W / 2 - hy / 2) / 2)
    x = h0 + hx + 0.40
    stages.append(("hole", stages[-1][2], x))
    # shoves
    x0 = x
    shoves = []
    for i in range(P["shoves"]):
        a = r.uniform(-math.pi, math.pi)
        shoves.append(dict(x=round(x0 + 0.10 + 0.15 * i, 3), v=[P["shove_v"] * math.cos(a), P["shove_v"] * math.sin(a)]))
    x = x0 + 0.10 + 0.15 * P["shoves"] + 0.15
    stages.append(("shoves", x0, x))
    finish = x + 0.10
    stages.append(("finish", x, finish))
    # deck: one strip minus the hole
    xs, xe = -0.40, finish + 0.40
    deck = [(xs, hole[0], -W / 2, W / 2), (hole[1], xe, -W / 2, W / 2), (hole[0], hole[1], -W / 2, hole[2]),
            (hole[0], hole[1], hole[3], W / 2)]
    boxes = [_box("deck", a, b, c, d, FLOOR_Z, 0.0) for a, b, c, d in deck] + boxes
    boxes += [_box("rail", xs, xe, sgn * W / 2 - (0.02 if sgn > 0 else 0), sgn * W / 2 + (0.02 if sgn < 0 else 0), 0.0, RAIL_H)
              for sgn in (-1.0, 1.0)]
    boxes.append(dict(kind="finish", pos=[finish, 0, 0.0005], size=[0.01, W / 2, 0.0005], quat=[1, 0, 0, 0],
                      color=COLORS["finish"], visual=True))
    boxes.append(dict(kind="hole", pos=[(hole[0] + hole[1]) / 2, 0, FLOOR_Z + 0.0005], size=[hx / 2, hy / 2, 0.0005],
                      quat=[1, 0, 0, 0], color="#000000", visual=True))
    return dict(level=level, seed=seed, boxes=boxes, stages=stages, hole=hole, lane=lane, narrow=narrow,
                shoves=shoves, finish=finish, floor_z=FLOOR_Z, params=P)


def path_y(L, x):
    """Centre line: y = 0, except a smooth detour onto the side lane around the hole."""
    h0, h1 = L["hole"][0], L["hole"][1]
    a0, a1, b0, b1 = h0 - 0.38, h0 - 0.06, h1 + 0.06, h1 + 0.38
    if x <= a0 or x >= b1:
        return 0.0
    if a1 <= x <= b0:
        return L["lane"]
    s = (x - a0) / (a1 - a0) if x < a1 else (b1 - x) / (b1 - b0)
    return L["lane"] * (0.5 - 0.5 * math.cos(math.pi * s))


def stage_at(L, x):
    for name, a, b in L["stages"]:
        if x < b:
            return name
    return "finish"


def scene_editor(L):
    """MjSpec edit: lower the floor, add the course boxes (static, in the world body)."""
    import mujoco

    def edit(spec):
        for g in spec.worldbody.geoms:
            if g.name == "floor":
                g.pos = [0, 0, FLOOR_Z]
        for i, b in enumerate(L["boxes"]):
            if b.get("visual"):
                continue
            g = spec.worldbody.add_geom()
            g.name = f"course_{b['kind']}_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = b["size"]
            g.pos = b["pos"]
            g.quat = b["quat"]
            c = b["color"].lstrip("#")
            g.rgba = [int(c[k:k + 2], 16) / 255 for k in (0, 2, 4)] + [1.0]
            g.friction = [0.9, 0.005, 0.0001]
    return edit


def make_sim(L, seed, randomize=True):
    return MicroduckSim(str(rh.SCENE), seed=seed, randomize=randomize, edit=scene_editor(L))


def start_pose(sim, x=0.0, y=0.0, yaw=0.0, z_extra=0.0):
    import mujoco
    sim.d.qpos[0:3] = [x, y, sim.d.qpos[2] + z_extra]
    sim.d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(sim.m, sim.d)


def trunk_pose(sim):
    p = sim.d.xpos[sim.trunk]
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    return float(p[0]), float(p[1]), float(p[2]), math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


WZ_COURSE = 1.0


def steer(L, x, y, yaw, look=0.22, k=2.0):
    """Pure pursuit to the centre line: wz to point at the line LOOK m ahead."""
    want = math.atan2(path_y(L, x + look) - y, look)
    return float(np.clip(k * wrap(want - yaw), -WZ_COURSE, WZ_COURSE)), wrap(want - yaw)


def play(level=1, seed=0, mode="course", record=False, t_max=T_MAX, L=None, probe_stop=None):
    """One game. Returns the result dict (and frames for the page if record)."""
    L = L or layout(level, seed)
    sim = make_sim(L, seed)
    rt = rh.HybridRuntime(sim=sim, seed=seed, log=None)
    rt.reset("standing")
    start_pose(sim, 0.0, 0.0, float(sim.rng.uniform(-0.1, 0.1)))
    frames, falls_at, shoves_left = [], [], list(L["shoves"])
    best_x, end, t_finish, offpath, was_off = 0.0, "time", None, 0, False
    stage_reach = {}
    hop_done = set()
    n = int(t_max * rh.HZ)
    for k in range(n):
        x, y, z, yaw = trunk_pose(sim)
        st = stage_at(L, x)
        wz, err = steer(L, x, y, yaw)
        if mode == "walk":
            vx = rh.WALK_VX
        elif mode == "run":
            vx = rh.RUN_VX
        else:     # "course": speed per obstacle (under and just ahead); "course_r": + stumble reflex
            vx, wz = rt.course_twist({stage_at(L, x - 0.08), st, stage_at(L, x + 0.15)}, err, wz, recover=mode == "course_r")
        rt.drive(vx, 0.0, wz)
        # shoves: a velocity kick on the free joint (like Microduck's push events) when the trunk passes x
        while shoves_left and x >= shoves_left[0]["x"] and rt.mode == "drive":
            sv = shoves_left.pop(0)["v"]
            sim.d.qvel[0] += sv[0]
            sim.d.qvel[1] += sv[1]
        f0 = rt.falls
        rt.tick()
        if rt.falls > f0:
            falls_at.append(dict(t=round(k * rh.DT, 2), x=round(x, 3), stage=st))
        if record and k % 2 == 0:
            frames.append(sim.frame())
        if not rt.lying():
            best_x = max(best_x, x)
        stage_reach[st] = True
        nx0, nx1, nw = L["narrow"]
        off = nx0 + 0.05 < x < nx1 - 0.05 and abs(y) > nw / 2
        if off and not was_off:
            offpath += 1
        was_off = off
        if x >= L["finish"] and not rt.lying():
            end, t_finish = "finished", round((k + 1) * rh.DT, 2)
            best_x = L["finish"]
            for _ in range(25):          # a short run-out after the line, for the replay
                rt.drive(0.0, 0.0, 0.0); rt.tick()
                if record and _ % 2 == 0:
                    frames.append(sim.frame())
            break
        if z < FLOOR_Z + 0.07 and (abs(y) > DECK_W / 2 or (L["hole"][0] < x < L["hole"][1] and L["hole"][2] < y < L["hole"][3])):
            end = "fell into the hole" if abs(y) < DECK_W / 2 else "fell off the course"
            for _ in range(50):
                rt.tick()
                if record and _ % 2 == 0:
                    frames.append(sim.frame())
            break
        if probe_stop and probe_stop(x):
            end = "probe"
            break
    t_used = t_finish if t_finish else round(len(frames) * 2 * rh.DT, 2) if record else round(n * rh.DT, 2)
    if end != "finished":
        t_used = round((k + 1) * rh.DT, 2)
    frac = min(1.0, best_x / L["finish"])
    score = round(1000 * frac + (10 * (t_max - t_finish) if t_finish else 0) - 50 * rt.falls)
    reached = [s for s, a, b in L["stages"] if best_x >= b - 1e-6 or (s == "finish" and end == "finished")]
    stuck = stage_at(L, best_x + 1e-3) if end != "finished" else None
    res = dict(level=level, seed=seed, mode=mode, end=end, finished=end == "finished", t=t_used, dist=round(best_x, 3),
               frac=round(frac, 3), falls=rt.falls, stumbles=rt.stumbles, unsticks=rt.unsticks, backups=rt.backups, falls_at=falls_at, off_path=offpath, score=score,
               cleared=reached, failed_at=stuck, body=sim.dr)
    if record:
        res["frames"], res["bodies"] = frames, sim.body_names()
        res["layout"] = L
    return res


def _job(args):
    level, seed, mode = args
    t = time.time()
    r = play(level, seed, mode)
    r["wall_s"] = round(time.time() - t, 1)
    return r


def summarize(runs):
    out = {}
    for (lv, mode) in sorted({(r["level"], r["mode"]) for r in runs}):
        rs = [r for r in runs if r["level"] == lv and r["mode"] == mode]
        stages = {s: sum(s in r["cleared"] for r in rs) for s in STAGES[:-1]}
        out[f"L{lv}_{mode}"] = dict(n=len(rs), finished=sum(r["finished"] for r in rs),
                                    median_dist=float(np.median([r["dist"] for r in rs])),
                                    median_t_finished=float(np.median([r["t"] for r in rs if r["finished"]])) if any(r["finished"] for r in rs) else None,
                                    falls=sum(r["falls"] for r in rs), runs_with_falls=sum(r["falls"] > 0 for r in rs),
                                    off_path=sum(r["off_path"] for r in rs), median_score=float(np.median([r["score"] for r in rs])),
                                    cleared=stages, failed_at={s: sum(r["failed_at"] == s for r in rs) for s in STAGES if any(r["failed_at"] == s for r in rs)},
                                    ends={e: sum(r["end"] == e for r in rs) for e in sorted({r["end"] for r in rs})},
                                    falls_by_stage={s: sum(1 for r in rs for f in r["falls_at"] if f["stage"] == s) for s in STAGES if any(f["stage"] == s for r in rs for f in r["falls_at"])})
    return out


PROBE_GRID = (("step", (0.010, 0.015, 0.020, 0.025), (0.3, 0.4, 0.6, 0.8, 1.35)),
              ("bar", (0.005, 0.010, 0.015, 0.020, 0.025), (0.4, 0.6, 0.8, 1.0, 1.35)),
              ("ramp", (5, 10, 12.5, 15, 20), (0.4, 0.6, 0.8, 1.0, 1.35)),
              ("rough", (0.008, 0.012, 0.016), (0.3, 0.4, 0.6, 0.8, 1.35)))


def probe(jobs=3, seeds=10):
    """Single obstacles on a flat deck at several speed commands: how high a step / bar, how steep a ramp."""
    tests = [(kind, v, spd, seeds) for kind, vals, spds in PROBE_GRID for v in vals for spd in spds]
    with ProcessPoolExecutor(jobs) as ex:
        out = list(ex.map(_probe_one, tests))
    return out


def _probe_layout(kind, v, seed):
    W, r = DECK_W, np.random.default_rng(seed)
    boxes = [_box("deck", -0.4, 2.0, -W / 2, W / 2, FLOOR_Z, 0.0)]
    if kind == "step":
        boxes.append(_box("step", 0.4, 0.9, -W / 2, W / 2, 0.0, v))
    elif kind == "bar":
        boxes.append(_box("bar", 0.4, 0.43, -W / 2, W / 2, 0.0, v))
    elif kind == "ramp":
        h = 0.25 * math.tan(math.radians(v))
        boxes += [_ramp(0.4, 0.65, 0.0, h), _box("ramp", 0.65, 0.85, -W / 2, W / 2, h - 0.02, h), _ramp(0.85, 1.10, h, 0.0)]
    elif kind == "rough":
        for _ in range(48):
            cx, cy = r.uniform(0.42, 0.88), r.uniform(-0.25, 0.25)
            sx, sy, yaw = r.uniform(0.015, 0.035), r.uniform(0.015, 0.035), r.uniform(0, math.pi)
            boxes.append(_box("rough", cx - sx, cx + sx, cy - sy, cy + sy, 0.0, r.uniform(v / 2.5, v),
                              (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))))
    return dict(boxes=boxes, stages=[("start", -0.3, 0.3), ("obstacle", 0.3, 1.2), ("finish", 1.2, 1.3)],
                hole=(9, 9.1, -0.01, 0.01), lane=0.0, narrow=(9, 9.1, 1.0), shoves=[], finish=1.3, floor_z=FLOOR_Z)


def _probe_one(t):
    kind, v, spd, seeds = t
    n_ok, falls, ts = 0, 0, []
    for seed in range(seeds):
        L = _probe_layout(kind, v, seed)
        rh_speed = spd
        r = play_fixed(L, seed, rh_speed)
        n_ok += r["finished"] and r["falls"] == 0
        falls += r["falls"]
        ts.append(r["t"])
    return dict(kind=kind, value=v, vx=spd, n=seeds, clean=int(n_ok), falls=falls, t_med=float(np.median(ts)))


def play_fixed(L, seed, vx, t_max=30.0):
    sim = make_sim(L, seed)
    rt = rh.HybridRuntime(sim=sim, seed=seed, log=None)
    rt.reset("standing")
    start_pose(sim, 0.0, 0.0, 0.0)
    for k in range(int(t_max * rh.HZ)):
        x, y, z, yaw = trunk_pose(sim)
        wz, _ = steer(L, x, y, yaw)
        rt.drive(vx, 0.0, wz)
        rt.tick()
        if x >= L["finish"] and not rt.lying():
            return dict(finished=True, falls=rt.falls, t=round(k * rh.DT, 2))
    return dict(finished=False, falls=rt.falls, t=t_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--modes", default="walk,course,course_r")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_course_hybrid.json"))
    a = ap.parse_args()
    if a.probe:
        out = probe(a.jobs)
        for o in out:
            print(o)
        (HERE / "probe_course_hybrid.json").write_text(json.dumps(out, indent=1))
        return
    lo, hi = (int(v) for v in a.seeds.split("-"))
    jobs = [(int(lv), s, m) for lv in a.levels.split(",") for m in a.modes.split(",") for s in range(lo, hi + 1)]
    with ProcessPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(_job, jobs))
    for r in runs:
        print(r["level"], r["mode"], r["seed"], r["end"], r["dist"], r["t"], "falls", r["falls"], r["falls_at"], "off", r["off_path"], "score", r["score"])
    summ = summarize(runs)
    print(json.dumps(summ, indent=1))
    Path(a.out).write_text(json.dumps(dict(summary=summ, runs=runs), indent=1))


if __name__ == "__main__":
    main()
