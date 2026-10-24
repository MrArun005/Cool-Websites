"""Find balance gains for the walk teacher (walk_expert.py) so it survives before anything is cloned from it.

    python walk_tune.py                              # walk, 2 rounds, writes runs/walk_gains.json
    python walk_tune.py --task run --rounds 3 --candidates 300 --seeds 8

Why: GAINS["walk"] is empty, so the "expert" is the open-loop reference with no balance; it falls in
1-1.5 s on most commands. Behaviour cloning copies that, PPO then starts from a falling brain.

Score per candidate, averaged over seeds and commands (every rollout on a randomized body):
    survived fraction of the episode  -  err_w * tracking error   (fall first, speed second)
Round 1 samples gains uniformly over +-range (the sign convention of the IMU axes is not assumed).
Later rounds sample around the best so far with a shrinking spread. The result goes into
runs/walk_gains.json; walk_expert.GAINS reads it when present, so walk_bc.py / walk_env.check pick it up.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from walk_expert import DEFAULT, rollout

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "walk_gains.json"
# Search range per gain: +-half-width. dmax (foot shift cap, m) is scaled with the body (L in rollout's env).
RANGE = {"kx": 0.02, "kxd": 0.005, "kv": 0.08, "ky": 0.04, "kyd": 0.005, "kw": 0.5, "dx0": 0.005}


def score_one(args):
    gains, task, seeds, seconds, err_w = args
    out = []
    for s in seeds:
        r = rollout(gains, task=task, seed=s, seconds=seconds)
        out.append(r["T"] / seconds - err_w * r["err"])
    return float(np.mean(out))


def sample(rng, centre, spread):
    g = {}
    for k, w in RANGE.items():
        g[k] = float(np.clip(centre.get(k, 0.0) + rng.uniform(-w, w) * spread, -2 * w, 2 * w))
    g["dmax"] = centre.get("dmax", DEFAULT["dmax"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="walk")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--err-w", type=float, default=0.5, help="weight of the tracking error against survival")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds = list(range(a.seeds))
    base = dict(DEFAULT)
    best = (score_one((base, a.task, seeds, a.seconds, a.err_w)), base)
    print(f"baseline (no feedback): {best[0]:.3f}")
    for rd in range(a.rounds):
        spread = 1.0 if rd == 0 else 0.5 ** rd
        cands = [sample(rng, best[1] if rd else {}, spread) for _ in range(a.candidates)]
        with Pool(a.workers) as pool:
            scores = pool.map(score_one, [(g, a.task, seeds, a.seconds, a.err_w) for g in cands])
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (scores[i], cands[i])
        print(f"round {rd + 1}: best {best[0]:.3f} {json.dumps({k: round(v, 4) for k, v in best[1].items()})}")
    # Confirm on fresh seeds the search never saw.
    fresh = list(range(1000, 1000 + a.seeds))
    rows = [rollout(best[1], task=a.task, seed=s, seconds=a.seconds) for s in fresh]
    surv = np.mean([r["T"] for r in rows])
    print(f"fresh seeds: mean survival {surv:.2f}/{a.seconds:g} s, falls {sum(r['fell'] for r in rows)}/{len(rows)}, "
          f"mean err {np.mean([r['err'] for r in rows]):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    data[a.task] = dict(gains=best[1], score=best[0], fresh_survival=float(surv), seconds=a.seconds)
    Path(a.out).write_text(json.dumps(data, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
