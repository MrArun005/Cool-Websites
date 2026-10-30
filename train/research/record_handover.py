"""Record the real trot -> jump hand-over states of MasterMind's world, for training the jump brain from them.

29 Oct: the jump brain alone, started mid-stride from a synthetic trot pose (jump_env._gait_pose), clears 20/30 cm
100% at 1.7 m/s from 1.1 m, but inside the MasterMind chain only 77/73% (mostly a trailing leg knocking the bar
after a good clearance), and clean and failed hand-overs look the same on speed, distance, heading and leg speed
(research/results/handover_trot55/). So the jump brain should practise from the states it is really handed.

Runs the Master env's teacher chain (--teachers) as test_skills.py's jump sequence: trot 2 s from the trot clip,
a bar --heights (random) placed 1.5-2 m ahead, the approach; at the moment the jump brain takes over it stores the
robot's state and the bar's place relative to it, then moves on to the next try (the jump itself is not run).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/record_handover.py --n 400 --seed 50000 \
        --out research/results/handover_states.npz

Saves: qpos (root z, quaternion, 12 joints: qpos[2:19]), qvel (18), ctrl (12), ahead (m, body centre to bar),
heading_err (rad, jump_env.heading_error), off_line (m, jump_env.off_line), height (bar, m), seed.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "mastermind"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from drive import Driver  # noqa: E402
from env import Master  # noqa: E402


class Captured(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=50000, help="first seed (dev 20000 and test 1000 seeds are kept apart)")
    ap.add_argument("--teachers", default="v6h")
    ap.add_argument("--heights", type=float, nargs=2, default=[0.20, 0.60])
    ap.add_argument("--latency-max-ms", type=float, default=0.0,
                    help="per try a control delay drawn in [0, x] ms for the chain (30 Oct: hand-overs as a delayed robot makes them)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {k: [] for k in ("qpos", "qvel", "ctrl", "ahead", "heading_err", "off_line", "height", "seed", "latency_ms")}
    m = Master(seed=a.seed, teachers=a.teachers)
    orig = m._start_jump_brain

    def grab():
        w, d = m.w, m.data
        rows["qpos"].append(d.qpos[2:19].copy())
        rows["qvel"].append(d.qvel[0:18].copy())
        rows["ctrl"].append(d.ctrl[:12].copy())
        rows["ahead"].append(w.hurdle_ahead())
        rows["heading_err"].append(w.heading_error())
        rows["off_line"].append(w.off_line())
        rows["height"].append(w.hurdle_h)
        raise Captured
    m._start_jump_brain = grab
    d = Driver(m, None)
    missed = 0
    for k in range(a.n):
        seed = a.seed + k
        rng = np.random.default_rng(seed)
        lat = round(rng.uniform(0, a.latency_max_ms) / 2) * 2 if a.latency_max_ms else 0.0
        m.real.update(delay=lat / 1000.0)
        d.start("trot", seed=seed)
        d.do("trot", 2.0)
        try:
            d.do("jump", height=float(rng.uniform(*a.heights)), ahead=float(rng.uniform(1.5, 2.0)))
            missed += 1   # no hand-over (a fall or a run-out before the bar)
        except Captured:
            rows["seed"].append(seed)
            rows["latency_ms"].append(lat)
        m.jump = None
    out = {k: np.array(v) for k, v in rows.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out, teachers=a.teachers)
    n = len(out["seed"])
    print(f"{n} hand-overs ({missed} tries without one) -> {a.out}")
    for k in ("ahead", "heading_err", "off_line"):
        print(f"  {k}: mean {out[k].mean():.3f}, sd {out[k].std():.3f}")
    v = np.linalg.norm(out["qvel"][:, :2], axis=1)
    print(f"  speed: mean {v.mean():.2f} m/s, sd {v.std():.2f}; body height mean {out['qpos'][:, 0].mean():.3f} m")


if __name__ == "__main__":
    main()
