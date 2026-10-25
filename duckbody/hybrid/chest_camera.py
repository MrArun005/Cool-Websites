"""Chest camera: where to put a second camera so the duck sees the ball under its chest and between its feet.

One hybrid duck (velstand standing, then walking), the FULL body meshes rendered, a 7 cm magenta ball put on a grid
around the feet (trunk frame, x forward, y left); for each candidate camera pose: is the ball detected (follow.detect,
the same colour blob as every camera here) and how far off is the ball position from the floor ray through the blob
centre (soccer.pixel_ground with the chest camera's focal length).

    PY=/home/user/cool-websites/.venv/bin/python
    MUJOCO_GL=osmesa $PY duckbody/hybrid/chest_camera.py --search        # 8 poses (2 cm grid, standing)
    MUJOCO_GL=osmesa $PY duckbody/hybrid/chest_camera.py --map           # the chosen pose (follow.CHEST_*): 1 cm grid
                                                                          # standing + 2 cm grid at 6 walking phases
Writes results_chest_camera.json and chest_camera_view.png (the chosen camera's picture with the ball at a few spots).
Zones (trunk frame): under = x -0.05..0.05, |y| < 0.10 (under the chest, between / behind the toes); front = x
0.05..0.15, |y| < 0.10; mid = x 0.15..0.30, |y| < 0.15; side = x -0.05..0.15, |y| 0.10..0.20.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_hybrid as rh  # noqa: E402
import follow as F  # noqa: E402
from harness import MicroduckSim  # noqa: E402

BR = 0.035                       # match.py's ball radius
W, H = F.W, F.H


def build(cands):
    def edit(spec):
        F.scene_editor()(spec)
        spec.body("ball").geoms[0].size = [BR, 0, 0]
        for k, (pos, pitch) in enumerate(cands):
            F.add_chest_camera(spec.body(F.CHEST_BODY), name=f"chest{k}", pos=pos, pitch_deg=pitch)
    return MicroduckSim(str(rh.SCENE), seed=0, edit=edit)


def ground(sim, cid, u, v):
    ray = sim.d.cam_xmat[cid].reshape(3, 3) @ np.array([(u - W / 2) / F.CHEST_FY, -(v - H / 2) / F.CHEST_FY, -1.0])
    c = sim.d.cam_xpos[cid]
    if ray[2] > -1e-3:
        return None
    g = c + (BR - c[2]) / ray[2] * ray - sim.d.xpos[sim.trunk]
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([cy * g[0] + sy * g[1], -sy * g[0] + cy * g[1]])


def sweep(sim, ren, cams, xs, ys):
    """seen[k] (len(xs) x len(ys)) and the ranging error (m, nan where not seen) per camera."""
    import mujoco
    mid = sim.m.body("ball").mocapid[0]
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    seen = {k: np.zeros((len(xs), len(ys)), int) for k in cams}
    err = {k: np.full((len(xs), len(ys)), np.nan) for k in cams}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = sim.d.xpos[sim.trunk][:2] + [c * x - s * y, s * x + c * y]
            sim.d.mocap_pos[mid] = [p[0], p[1], BR]
            mujoco.mj_forward(sim.m, sim.d)
            for k in cams:
                ren.update_scene(sim.d, camera=k)
                det = F.detect(ren.render())
                if det is not None:
                    seen[k][i, j] = 1
                    g = ground(sim, sim.m.camera(k).id, det[0], det[1])
                    if g is not None:
                        err[k][i, j] = float(np.hypot(g[0] - x, g[1] - y))
    sim.d.mocap_pos[mid] = [5, 5, BR]
    return seen, err


def zones(xs, ys, seen, err):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = dict(under=(X > -0.05) & (X < 0.05) & (abs(Y) < 0.10), front=(X >= 0.05) & (X < 0.15) & (abs(Y) < 0.10),
             mid=(X >= 0.15) & (X < 0.30) & (abs(Y) < 0.15), side=(X > -0.05) & (X < 0.15) & (abs(Y) >= 0.10))
    out = {}
    for z, m in Z.items():
        e = err[m & (seen == 1)]
        e = e[~np.isnan(e)]
        out[z] = dict(seen=round(float(seen[m].mean()), 3),
                      err_median=round(float(np.median(e)), 3) if len(e) else None,
                      err_p90=round(float(np.percentile(e, 90)), 3) if len(e) else None)
    return out


def ascii_map(xs, ys, seen, err):
    rows = []
    for i, x in enumerate(xs):
        r = ""
        for j in range(len(ys)):
            if not seen[i, j]:
                r += "."
            else:
                e = err[i, j]
                r += "?" if np.isnan(e) else ("0" if e < 0.01 else ("1" if e < 0.02 else ("3" if e < 0.04 else ("6" if e < 0.08 else "X"))))
        rows.append(f"{x:+.2f} {r}")
    return rows


def renderer(sim):
    import mujoco
    ren = mujoco.Renderer(sim.m, H, W)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    return ren


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_chest_camera.json"))
    a = ap.parse_args()
    if a.search and a.map:
        sys.exit("run --search and --map in separate processes (a second OSMesa renderer in one process draws nothing)")
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    res["camera"] = dict(body=F.CHEST_BODY, pos=list(F.CHEST_POS), pitch_deg=F.CHEST_PITCH_DEG, hfov_deg=F.CHEST_HFOV,
                         vfov_deg=round(F.CHEST_FOVY, 1), w=W, h=H, ball_r=BR,
                         zones="under: x -0.05..0.05 |y|<0.10; front: x 0.05..0.15 |y|<0.10; mid: x 0.15..0.30 "
                               "|y|<0.15; side: x -0.05..0.15 |y| 0.10..0.20 (trunk frame, m)")
    if a.search:
        cands = [((x, 0.0, z), th) for (x, z) in ((0.037, -0.042), (0.030, -0.046)) for th in (55, 65, 75, 85)]
        cands += [((0.037, 0.0, -0.042), 60)]
        sim = build(cands)
        rt = rh.HybridRuntime(sim=sim, seed=0, log=None)
        rt.reset("standing")
        rt.run(1.5)
        ren = renderer(sim)
        xs, ys = np.round(np.arange(-0.10, 0.40, 0.02), 3), np.round(np.arange(-0.20, 0.205, 0.02), 3)
        names = [f"chest{k}" for k in range(len(cands))]
        seen, err = sweep(sim, ren, names, xs, ys)
        res["search"] = []
        for k, (pos, th) in zip(names, cands):
            z = zones(xs, ys, seen[k], err[k])
            res["search"].append(dict(pos=pos, pitch_deg=th, zones=z))
            print(pos, th, z, flush=True)
    if a.map:
        sim = build([(F.CHEST_POS, F.CHEST_PITCH_DEG)])
        rt = rh.HybridRuntime(sim=sim, seed=0, log=None)
        rt.reset("standing")
        rt.run(1.5)
        ren = renderer(sim)
        xs, ys = np.round(np.arange(-0.10, 0.40, 0.01), 3), np.round(np.arange(-0.20, 0.205, 0.01), 3)
        seen, err = sweep(sim, ren, ["chest0"], xs, ys)
        res["standing"] = dict(zones=zones(xs, ys, seen["chest0"], err["chest0"]), xs=xs.tolist(), ys=ys.tolist(),
                               map=ascii_map(xs, ys, seen["chest0"], err["chest0"]),
                               legend="'.' not seen; seen with a ranging error of 0 <1 cm, 1 <2, 3 <4, 6 <8, X >= 8 cm")
        print("standing", res["standing"]["zones"], flush=True)
        for row in res["standing"]["map"]:
            print(row)
        # pictures: the ball under the front of the chest, at the toes, ahead, beside a foot, further out
        import PIL.Image as I
        import mujoco
        mid = sim.m.body("ball").mocapid[0]
        R = sim.d.xmat[sim.trunk].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        imgs = []
        for (x, y) in [(0.0, 0.0), (0.08, 0.04), (0.15, -0.05), (0.03, 0.12), (0.25, 0.0)]:
            p = sim.d.xpos[sim.trunk][:2] + [c * x - s * y, s * x + c * y]
            sim.d.mocap_pos[mid] = [p[0], p[1], BR]
            mujoco.mj_forward(sim.m, sim.d)
            ren.update_scene(sim.d, camera="chest0")
            imgs.append(ren.render())
            imgs.append(np.full((H, 4, 3), 255, np.uint8))
        I.fromarray(np.concatenate(imgs[:-1], 1)).save(HERE / "chest_camera_view.png")
        sim.d.mocap_pos[mid] = [5, 5, BR]
        # walking (drive 0.5): the legs swing through the view; 6 gait phases, 2 cm grid
        rt.drive(0.5, 0.0, 0.0)
        rt.run(2.0)
        xs2, ys2 = np.round(np.arange(-0.06, 0.32, 0.02), 3), np.round(np.arange(-0.20, 0.205, 0.02), 3)
        walk = []
        for ph in range(6):
            rt.run(0.07)                # (the sweep only re-renders this pose; the ball is a mocap body without contacts)
            seen2, err2 = sweep(sim, ren, ["chest0"], xs2, ys2)
            walk.append(zones(xs2, ys2, seen2["chest0"], err2["chest0"]))
            print("walking phase", ph, walk[-1], flush=True)
        res["walking"] = dict(phases=walk, mean={z: round(float(np.mean([w[z]["seen"] for w in walk])), 3)
                                                 for z in walk[0]})
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
