"""One fair exam for a GPU dog-motion brain (mjx_imitate params.pkl), so a new GPU run and the 360M brain are compared
on exactly the same tests (29 Oct, picking run gpu3):

  exam      imitate_env.exam: per clip tracking/falls/pace, sit/lie reached from still and from a moving walk/trot,
            and the latency sweep (0/10/20/40 ms), --episodes tries per clip on each of --seeds
  terrain   imitate_env.exam_terrain, 30 tries x 4 s: trot on 0/5/10 cm, walk on 10 cm, trot on 5 cm with a 1 m/s shove
            (the Step 6 targets: trot 10 cm < 10% falls, 5 cm + 1 m/s pushes < 10%)

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/gpu_brain_exam.py ../models/bolt-dog-imitate-gpu-360M/params.pkl \
        --out research/results/gpu_exam_360M.json [--motor-brake]

Motor brake off by default (old brains were trained without it; run gpu3 was trained with it, so run both).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from bolt_env import add_brake_args, set_motor_brake  # noqa: E402
import imitate_env  # noqa: E402
from mjx_imitate import MjxImitatePolicy  # noqa: E402

TERRAIN = [(0, "trot", 0.0), (5, "trot", 0.0), (10, "trot", 0.0), (10, "walk", 0.0), (5, "trot", 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9000, 11000])
    ap.add_argument("--terrain-tries", type=int, default=30)
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--out")
    add_brake_args(ap, default=False)
    a = ap.parse_args()
    brake = set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain)
    pol = MjxImitatePolicy(a.brain)
    t0 = time.time()
    runs = [imitate_env.exam(pol, episodes=a.episodes, seed=s) for s in a.seeds]
    exam = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]}
    exam["score"] = round(imitate_env.score({k: v for k, v in exam.items() if k.startswith(("track_", "speed_"))}), 3)
    terrain = [] if a.no_terrain else [
        dict(push=push, **imitate_env.exam_terrain(pol, cm, clip=clip, tries=a.terrain_tries, push_speed=push))
        for cm, clip, push in TERRAIN]
    res = dict(brain=a.brain, motor_brake=brake, episodes=a.episodes, seeds=a.seeds, exam=exam, terrain=terrain,
               minutes=round((time.time() - t0) / 60, 1), command="python " + " ".join(sys.argv))
    print(json.dumps(res, indent=1, default=float))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
