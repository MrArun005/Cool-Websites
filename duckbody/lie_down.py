"""Lie down on the belly, the reverse of the face-down get-up, and check the get-up works from where it ends.

    python lie_down.py [--workers 1] [--seeds 20]

In the real runtime: "sit" (squat -> perch, 20/20), then
  lower  MoveBrain(goal = the face-down tuck pose, lean, t1, t2, sign)   (fold forward onto the belly)
  rest   PoseBrain glide to the resting legs (hip, knee, ankle) over 1 s, hold 1 s
Lying = IMU nose-down x > 0.5 and trunk tilted > 55 deg, landing peak |a_z| reported. Then "get up"
(tuck -> lift into the perch -> perch rise -> stand) must stand again. Best scripts re-run on randomized bodies.
"""
import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import runtime as rtm

HERE = Path(__file__).resolve().parent
BRAINS, CFG = rtm.load_brains()
STAND_H = CFG["stand_height"]


def legs(rt, h, k, a):
    c = rt.home.copy()
    for side, sgn in (("left", 1), ("right", -1)):
        for j, v in (("hip_pitch", sgn * h), ("knee", k), ("ankle", a)):
            c[rt.names.index(f"{side}_{j}")] = v
    return c


def episode(args):
    (lean, t1, t2, sign, rest), seed, randomize = args
    rng = np.random.default_rng(seed)
    be = rtm.SimBackend(CFG["servo_names"])
    m = be.model
    if randomize:
        f = rng.uniform(0.9, 1.1, m.nbody)
        m.body_mass[:] *= f
        m.body_inertia[:] *= f[:, None]
        m.geom_friction[:, 0] = rng.uniform(0.4, 1.0)
    rt = rtm.DuckRuntime(backend=be, brains=dict(BRAINS), config=CFG, log=None)
    rt.run(1.0)
    rt.command("sit")
    rt.run(5.5)
    if rt.skill != "pose:perch":
        return args[0], False, False, 0.0, 0.0
    rt.brains["pose:lower"] = rtm.MoveBrain(legs(rt, *rtm.FD_TUCK), rt, lean, t1, t2, sign)
    rt.brains["pose:rest"] = rtm.PoseBrain(legs(rt, *rest), rt, glide=1.0)
    rt.quiet.update(("pose:lower", "pose:rest"))
    rt.mode = "lie"
    rt.plan = [("pose:lower", t1 + t2 + 0.5), ("pose:rest", None)]
    rt.plan_all = list(rt.plan)
    rt._next_in_plan()
    peak = 0.0
    for _ in range(int((t1 + t2 + 0.5 + 2.0) * rt.hz)):
        rt.tick()
        peak = max(peak, abs(float(be.data.qacc[2])))
    lying = rt.lying_face_down()
    stood = False
    if lying:
        rt.command("get up")
        rt.run(8.0)
        p = be.posture()
        stood = bool(p["up"] > 0.9 and p["height"] > 0.9 * STAND_H)
    return args[0], bool(lying), stood, round(peak / 9.81, 1), round(float(be.read().gravity[0]), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    rests = [(0.0, 2.18, -0.8), (-0.6, 1.4, -0.8), (-1.0, 1.9, 0.0)]
    grid = list(itertools.product((0.0, 0.3, 0.6, 1.0), (0.3, 0.8), (1.0, 2.0), (-1, 1), rests))
    with Pool(a.workers) as pool:
        R = pool.map(episode, [(g, 0, False) for g in grid])
        good = [r for r in R if r[2]]
        print(f"lie down: lying face down {sum(r[1] for r in R)}/{len(R)}, and got up again {len(good)}/{len(R)}", flush=True)
        good.sort(key=lambda r: r[3])                     # softest landing first
        for r in good[:6]:
            print("  ", r, flush=True)
        results = []
        for r in good[:3]:
            RR = pool.map(episode, [(r[0], 100 + s, True) for s in range(a.seeds)])
            n_l, n_s = sum(x[1] for x in RR), sum(x[2] for x in RR)
            results.append(dict(script=dict(zip(("lean", "t1", "t2", "sign", "rest"), r[0])), lying=n_l, got_up=n_s, seeds=a.seeds,
                                peak_g=max(x[3] for x in RR)))
            print(f"randomized: {r[0]} lying {n_l}/{a.seeds}, got up {n_s}/{a.seeds}, peak {max(x[3] for x in RR)} g", flush=True)
    (HERE / "model" / "lie_down.json").write_text(json.dumps(dict(randomized=results), indent=1, default=float))


if __name__ == "__main__":
    main()
