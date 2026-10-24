"""Replay of the hybrid duck getting up from its back with Microduck's velstand brain, + a PNG strip.

    MUJOCO_GL=osmesa .venv/bin/python duckbody/hybrid/replay.py [--body hybrid] [--seconds 6]

Runs face-up starts (same randomisation as the evaluation, seeds 0, 1, 2, ...) until one succeeds and
one fails, and writes for each:
  replay_getup_back[_fail]<_body>.json  dict(fps=25, seconds, frames=[flat [x,y,z,qw,qx,qy,qz] per body
                                        per frame], bodies=[names], success, seed, ...)
  getup_back[_fail]<_body>.png          8 frames side by side (the get-up, to 1 s after recovery)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "microduck_port"))
sys.path.insert(0, str(HERE))
from harness import MicroduckSim, Policy  # noqa: E402
from evaluate import getup  # noqa: E402
from evaluate_hybrid import BODIES  # noqa: E402


def strip(sim, frames, path, n=8, size=320):
    from PIL import Image
    m, d = sim.m, sim.d
    nb = m.nbody - 1
    r = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.6, 135, -20
    imgs = []
    for i in np.linspace(0, len(frames) - 1, n).round().astype(int):
        f = np.array(frames[i]).reshape(nb, 7)
        # Put the pose back into qpos via the trunk free joint and joint angles is not stored; set body poses
        # directly for drawing: mocap-free trick, write xpos/xquat then update the scene from them.
        d.xpos[1:] = f[:, :3]
        d.xquat[1:] = f[:, 3:]
        for b in range(1, m.nbody):
            mujoco.mju_quat2Mat(d.xmat[b], d.xquat[b])
        # geoms follow their bodies
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b == 0:
                continue
            Rb = d.xmat[b].reshape(3, 3)
            d.geom_xpos[g] = d.xpos[b] + Rb @ m.geom_pos[g]
            Rg = np.zeros(9)
            mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
            d.geom_xmat[g] = (Rb @ Rg.reshape(3, 3)).ravel()
        cam.lookat[:] = [f[0, 0], f[0, 1], 0.08]
        r.update_scene(d, cam)
        img = r.render().copy()
        imgs.append(img)
    r.close()
    Image.fromarray(np.concatenate(imgs, 1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="hybrid")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--max-seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    pol = Policy("velstand")
    tag = "" if a.body == "hybrid" else "_" + a.body
    want = {True: f"replay_getup_back{tag}", False: f"replay_getup_back_fail{tag}"}
    for seed in range(a.seed0, a.seed0 + a.max_seeds):
        if not want:
            break
        sim = MicroduckSim(BODIES[a.body][0], seed=seed)
        res, frames = getup(sim, pol, "face_up", seconds=a.seconds, record=True)
        print(seed, res, flush=True)
        name = want.pop(res["success"], None)
        if name is None:
            continue
        (HERE / f"{name}.json").write_text(json.dumps(dict(
            fps=25, seconds=a.seconds, body=a.body, seed=seed, policy="velstand.onnx", start="face_up", **res,
            bodies=sim.body_names(), frames=frames), separators=(",", ":")))
        # strip: the get-up itself (start .. 1 s after recovery), or the whole try when it fails
        end = int((res["t_recover"] + 1.0) * 25) if res["success"] else len(frames)
        strip(sim, frames[:end], HERE / f"{name.replace('replay_', '')}.png")
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
