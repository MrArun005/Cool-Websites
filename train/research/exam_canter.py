"""Canter (and other clip) exam: imitate_env.exam's protocol, with a pluggable controller and N tries.

    python research/exam_canter.py --clips canter --n 10 --mode plain|pps [--K 16 --H 30 --M 5 --sigma 0.2]

Same protocol as imitate_env.exam: randomized physics, loops run 200 steps (4 s) from evenly spaced
points in the stride (k/N), seed 9000+k; per try the mean tracking reward over the whole 4 s (0 for every
step after a fall), fall yes/no, and the share of the dog's forward distance covered.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from imitate_env import ImitateEnv, ImitatePolicy   # noqa: E402
from pps import Planner, PlannedController            # noqa: E402

BRAIN = HERE.parent.parent / "models" / "bolt-dog-imitate-18b-17.5M"


def run(ctrl, name, n, seed=9000, policy=None, clips=None, log=None):
    env = ImitateEnv(randomize=True, clips=clips or [name])
    c = env.clips[name]
    turn = name.startswith("turn")
    rows = []
    for k in range(n):
        env.exam_start = (name, (k / n) * c.length if c.loop else 0.0)
        obs, _ = env.reset(seed=seed + k)
        if c.loop:
            env.limit = 200
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        total, fell, done_by, yaw, t0 = 0.0, False, 0.0, 0.0, time.time()
        for _ in range(env.limit):
            R = env._rotation()
            p0, h0 = env.data.qpos[:2].copy(), np.arctan2(R[1, 0], R[0, 0])
            a = ctrl.act(env, obs)
            if log is not None:
                log.append((obs.copy(), np.asarray(a, dtype=np.float32)))
            obs, r, fell, done, info = env.step(a)
            R = env._rotation()
            done_by += float(np.dot(env.data.qpos[:2] - p0, [np.cos(h0), np.sin(h0)]))
            yaw += float(np.angle(np.exp(1j * (np.arctan2(R[1, 0], R[0, 0]) - h0))))
            total += r
            if fell or done:
                break
        row = dict(track=total / env.limit, fell=bool(fell), seconds=round(time.time() - t0, 1))
        if c.loop:
            want = env.limit * env.dt * (c.w_body[:, 2].mean() if turn else c.v_body[:, 0].mean())
            row["speed"] = (yaw if turn else done_by) / want
        rows.append(row)
        print(name, k, {a: round(b, 3) if isinstance(b, float) else b for a, b in row.items()}, flush=True)
    tr = np.array([r["track"] for r in rows])
    out = dict(clip=name, n=n, track_mean=round(float(tr.mean()), 3), track_sd=round(float(tr.std(ddof=1)), 3),
               track_se=round(float(tr.std(ddof=1) / np.sqrt(n)), 3), falls=int(sum(r["fell"] for r in rows)),
               speed_mean=round(float(np.mean([r.get("speed", np.nan) for r in rows])), 3), tries=rows)
    return out


class Plain:
    def __init__(self, policy):
        self.p = policy

    def act(self, env, obs):
        return self.p.act(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=["canter"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", default="plain")
    ap.add_argument("--brain", default=str(BRAIN))
    ap.add_argument("--zip", default="policy.zip")
    ap.add_argument("--stats", default="vecnormalize.pkl")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pol = ImitatePolicy(args.brain, args.zip, args.stats)
    res = dict(args=vars(args), results=[])
    for name in args.clips:
        if args.mode == "pps":
            planner = Planner(pol, [name], K=args.K, H=args.H, M=args.M, sigma=args.sigma, robust=args.robust)
            ctrl = PlannedController(pol, planner)
        else:
            ctrl, planner = Plain(pol), None
        r = run(ctrl, name, args.n)
        if planner:
            r["planner"] = planner.stats
        res["results"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "tries"}), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
