"""Skill-manager baseline (skills.py + test_skills.py sequences) with a chosen dog-motion brain, so the
student can be compared with the hand-over manager running the SAME teachers.

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/baseline.py --imitate 18e --n 10 --out mastermind/baseline_18e.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

torch.set_num_threads(1)

import test_skills  # noqa: E402
from skills import Bolt  # noqa: E402
from teachers import imitate_brain  # noqa: E402


def run(bolt, n, seed, only=None):
    tally, started = test_skills.Tally(), time.time()
    for name in only or test_skills.SEQUENCES:
        for k in range(n):
            test_skills.SEQUENCES[name](bolt, seed + 37 * k, tally)
        print(f"{name}: done ({time.time() - started:.0f} s)", flush=True)
    rows = []
    for s in tally.order:
        t, ok = tally.tries[s], tally.ok[s]
        rows.append(dict(switch=s, ok=ok, tries=t, rate=round(ok / t, 2) if t else None, not_reached=tally.skipped[s],
                         failures=tally.fails[s]))
    return rows, round(time.time() - started)


def show(rows):
    for r in rows:
        rate = "-" if r["rate"] is None else f"{r['rate']:.0%}"
        print(f"{r['switch']:34s} {r['ok']:>3d}/{r['tries']:<3d} {rate:>6s} {r['not_reached']:>4d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imitate", default="18e", help="18b or 18e")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    bolt = Bolt(seed=a.seed, imitate=imitate_brain(a.imitate))
    rows, secs = run(bolt, a.n, a.seed, a.only)
    show(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(imitate=a.imitate, n=a.n, seed=a.seed, seconds=secs, switches=rows), indent=1))
