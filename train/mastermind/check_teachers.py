"""Do the teachers of a set do inside the MasterMind world what they do on their own? (step 1 of the v3 plan)

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/check_teachers.py --teachers v3 --out mastermind/results/teachers_v3.json

* canter: research/exam_canter.py's protocol (randomized physics, 4 s from evenly spaced stride points, seeds 5000+k),
  N=30, the set's canter teacher run by the Master env on the canter command (models/...-18c-24.5M/exam_canter_30.json:
  tracking 65%, 1/30 falls, speed 65%)
* every clip: imitate_env.exam's protocol (4 tries per clip, seed 9000), as best.json of the dog-motion brain
* jump: jump_env.exam's fair protocol (fixed physics, straight, bar 1.5 m ahead, arriving at 2 m/s, seeds 7000+k, a
  try counts if the bar stays in its cups and Bolt lands on its feet) run on the shared world (JumpEnv.reset of the
  world itself, every other env attached), N=30 at 40/50 cm (exam_30.json: 100% / 77%)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from env import Master  # noqa: E402
from test_master import clip_exam  # noqa: E402


def jump_fair(m, heights=(0.40, 0.50), n=30, seed=7000, dist=1.5, speed=2.0):
    w, pol = m.w, m.b.jumper
    out = {}
    for h in heights:
        wins, clean = 0, 0
        for k in range(n):
            w.exam_bar, w.exam_speed, w.exam_askew = (dist, h), speed, (0.0, 0.0)
            obs, _ = w.reset(seed=seed + k)
            info, fell = {}, False
            for _ in range(w.max_steps):
                obs, _, fell, trunc, info = w.step(pol.act(obs))
                if fell or trunc:
                    break
            ok = bool(info["cleared"] and not info.get("dropped", info["touched"]) and not fell and info["up"] > 0.7)
            wins += ok
            clean += ok and not info["touched"]
        out[f"jump_{round(h * 100)}"] = wins / n
        out[f"jump_{round(h * 100)}_no_touch"] = clean / n
        print(h, out, flush=True)
    w.exam_bar = w.exam_speed = w.exam_askew = None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default="v3")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = dict(teachers=a.teachers)
    m = Master(seed=0, teachers=a.teachers)
    res["canter_n30_seed5000"] = clip_exam(m, None, episodes=a.n, seed=5000, clips=("canter",))
    print("canter", res["canter_n30_seed5000"], flush=True)
    res["clips_exam_seed9000"] = clip_exam(m, None)
    print("clips", res["clips_exam_seed9000"], flush=True)
    mf = Master(seed=0, teachers=a.teachers, randomize=False)
    res["jump_fair"] = jump_fair(mf, n=a.n)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
