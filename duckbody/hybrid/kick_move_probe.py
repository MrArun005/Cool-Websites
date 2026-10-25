"""Kick on the move: can Microduck's ball_kick_left / ball_kick_right brains (trained from standing) be started
straight out of velstand walking, without stopping? And how hard does walking through the ball push it?

One duck on soccer.py's pitch (randomized body), velstand driven straight at command vx for 2 s (steady gait), then
at a gait phase (from the feet: which foot is in the air and whether it is rising or falling) the ball is put at
(KICK spot + dx, +-0.042 + dy) in the trunk frame and the kick brain starts (0.5 s, then velstand standing), or, for
"walk", velstand keeps walking through it. Optional: a decel of `decel` s at drive 0 before the trigger, a head pose.

    MUJOCO_GL=osmesa $PY duckbody/hybrid/kick_move_probe.py --grid main --jobs 2 --out probe_kick_move_hybrid.json
    $PY duckbody/hybrid/kick_move_probe.py --summary probe_kick_move_hybrid.json

Phases (kicking foot = the foot on the ball's side): 0 kick foot lifting, 1 kick foot coming down, 2 support foot
lifting (kick foot in stance), 3 support foot coming down.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_hybrid as rh  # noqa: E402
import soccer as SO  # noqa: E402

HEADS = {"home": (0.0, 0.0, None), "p03": (0.0, 0.3, None), "p06": (0.0, 0.6, None),
         "foot": (0.0, SO.FOOT_VIEW[1], SO.FOOT_VIEW[0])}


def kick_move_trial(seed, foot, vx, phase, dx, dy, mode="kick", head="home", decel=0.0, warm=2.0, after=2.0):
    import mujoco
    P = SO.Pitch(seed=seed, randomize=True, cam=False)
    sim = P.sim
    rt = rh.HybridRuntime(sim=sim, seed=seed, log=None)
    rt.reset("standing")
    sim.d.qpos[0:2] = [-1.0, 0.0]
    sim.d.qpos[3:7] = [1, 0, 0, 0]
    P.set_ball(1.0, 0.6)
    mujoco.mj_forward(sim.m, sim.d)
    la, ra = sim.m.body("ankle_left").id, sim.m.body("ankle_right").id
    kb, sb = (la, ra) if foot == "left" else (ra, la)
    rt.look(*HEADS[head][:2], neck=HEADS[head][2])
    rt.run(0.5)
    rt.drive(vx, 0.0, 0.0)
    hist = []
    n_warm = int(warm * rh.HZ)
    trig = None
    for k in range(n_warm + int((decel + 1.5) * rh.HZ)):
        if k == n_warm and decel > 0:
            rt.drive(0.0, 0.0, 0.0)
        rt.look(*HEADS[head][:2], neck=HEADS[head][2])
        if rt.mode == "drive":
            rt.drive(*(rt.vel_goal))
        rt.tick()
        zk, zs = float(sim.d.xpos[kb][2]), float(sim.d.xpos[sb][2])
        hist.append((zk, zs))
        if k < n_warm + int(decel * rh.HZ) or len(hist) < 3:
            continue
        (k0, s0), (k1, s1) = hist[-2], hist[-1]
        lo = min(min(h) for h in hist[-30:])
        up_k, up_s = k1 - lo > 0.004, s1 - lo > 0.004
        ph = None
        if up_k and not up_s:
            ph = 0 if k1 > k0 else 1
        elif up_s and not up_k:
            ph = 2 if s1 > s0 else 3
        if vx == 0.0 and decel == 0.0:
            ph = phase                                   # standing: no gait
        if ph == phase:
            trig = k
            break
    if rt.lying():
        return dict(seed=seed, foot=foot, vx=vx, phase=phase, dx=dx, dy=dy, mode=mode, head=head, decel=decel,
                    trig=False, hit=False, v=0.0, up=False)
    if trig is None:
        return dict(seed=seed, foot=foot, vx=vx, phase=phase, dx=dx, dy=dy, mode=mode, head=head, decel=decel,
                    trig=False, hit=False, v=0.0, up=True)
    p, yaw = SO.trunk_pose(sim)
    v_trunk = float(np.hypot(*sim.d.qvel[0:2]))
    fy = 0.042 if foot == "left" else -0.042
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = 0.09 + dx, fy + dy
    P.set_ball(p[0] + c * bx - s * by, p[1] + s * bx + c * by)
    b0, _ = P.ball()
    if mode == "kick":
        rt.command(f"kick {foot}")
    vmax, vdir, t_hit, falls0 = 0.0, None, None, rt.falls
    for k in range(int(after * rh.HZ)):
        if mode == "walk":
            rt.drive(vx, 0.0, 0.0)
        elif rt.mode in ("kick left", "kick right") and rt.skill == "velstand":
            rt.drive(0.0, 0.0, 0.0)
        rt.tick()
        _, v = P.ball()
        sp = float(np.hypot(v[0], v[1]))
        if sp > vmax:
            vmax, vdir = sp, SO.wrap(math.atan2(v[1], v[0]) - yaw)
            if t_hit is None and sp > 0.3:
                t_hit = k * rh.DT
    b1, _ = P.ball()
    return dict(seed=seed, foot=foot, vx=vx, phase=phase, dx=dx, dy=dy, mode=mode, head=head, decel=decel, trig=True,
                v_trunk=round(v_trunk, 3), hit=vmax > 0.3, v=round(vmax, 3),
                dir_deg=None if vdir is None else round(math.degrees(vdir), 1), t_hit=t_hit,
                rolled=round(float(np.hypot(*(b1[:2] - b0[:2]))), 3), up=rt.falls == falls0 and not rt.lying())


def _job(a):
    return kick_move_trial(**a)


def grid(name):
    G = []
    if name == "main":            # walk speed x gait phase x ball offset, head home, both feet
        for seed, foot, vx, ph, dx, dy in itertools.product((0, 1), ("left", "right"), (0.0, 0.4, 0.7, 1.0),
                                                            (0, 1, 2, 3), (-0.03, 0.0, 0.03, 0.06, 0.09), (-0.02, 0.0)):
            if vx == 0.0 and ph > 0:
                continue
            G.append(dict(seed=seed, foot=foot, vx=vx, phase=ph, dx=dx, dy=dy))
    elif name == "walk":          # walk-through push: no kick
        for seed, foot, vx, dx, dy in itertools.product((0, 1), ("left", "right"), (0.4, 0.7, 1.0), (0.0, 0.06),
                                                        (-0.02, 0.0, 0.02)):
            G.append(dict(seed=seed, foot=foot, vx=vx, phase=2, dx=dx, dy=dy, mode="walk"))
    elif name == "decel":         # 0.25 / 0.5 s at drive 0 from walking (2-3 steps), then kick at a phase
        for seed, foot, vx, dec, ph, dx in itertools.product((0, 1), ("left", "right"), (0.7, 1.0), (0.25, 0.5),
                                                             (0, 2), (0.0, 0.03, 0.06)):
            G.append(dict(seed=seed, foot=foot, vx=vx, phase=ph, dx=dx, dy=-0.01, decel=dec))
    elif name == "head":          # standing kick with the head not at home
        for seed, foot, hd, dx in itertools.product((0, 1, 2), ("left", "right"), ("home", "p03", "p06", "foot"),
                                                    (0.0, 0.03)):
            G.append(dict(seed=seed, foot=foot, vx=0.0, phase=0, dx=dx, dy=-0.01, head=hd))
    return G


def summary(runs, keys):
    rows = {}
    for r in runs:
        k = tuple(r[x] for x in keys)
        rows.setdefault(k, []).append(r)
    out = []
    for k, rs in sorted(rows.items(), key=lambda kv: str(kv[0])):
        hits = [r for r in rs if r["hit"]]
        out.append(dict(zip(keys, k), n=len(rs), hits=len(hits), v=round(float(np.mean([r["v"] for r in hits])), 2) if hits else 0,
                        falls=sum(not r["up"] for r in rs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--keys", default="vx,phase")
    a = ap.parse_args()
    if a.summary:
        runs = json.loads(Path(a.summary).read_text())
        for row in summary(runs, a.keys.split(",")):
            print(row)
        return
    G = grid(a.grid)
    print(len(G), "trials", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        runs = list(ex.map(_job, G, chunksize=4))
    Path(a.out or HERE / f"probe_kick_move_{a.grid}.json").write_text(json.dumps(runs))
    for row in summary(runs, a.keys.split(",")):
        print(row)


if __name__ == "__main__":
    main()
