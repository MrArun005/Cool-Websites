"""Re-pick a finished jump run's brain on the mid-stride hand-over exam (and keep the still-legs exam beside it).

A jump run (train_jump.py) picks best/ on the exams it was given; run jump-trot (Kaggle, 28 Oct) picked on the
still-legs exams (dog stance, joints still), but MasterMind hands the jump brain over MID-STRIDE from a trot at
~1.7 m/s with the bar ~1.1 m ahead (research/handover_gap.py). This script examines every checkpoint of a
downloaded run folder on both:

  midstride  jump_env.exam(gait=True)  --mid-speed 1.7 m/s from --mid-dist 1.1 m, legs mid-stride (trot pose)
  still      jump_env.exam(gait=False) --still-speed 2.0 m/s from --still-dist 1.5 m, the model cards' exam

each --episodes tries per --heights bar height on seeds --seed.., tries of --max-steps steps; fair agility
scores (bar stays up, lands on its feet). Writes a JSON table (--out, default <run>/reselect_jump.json) and
copies the winner (highest mid-stride mean, ties broken by the still-legs mean, then the later checkpoint) to
<run>/best_midstride/ (policy.zip, vecnormalize.pkl, best.json with its scores and the command).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/reselect_jump.py runs/kaggle-jump-trot/runs/jumptrot
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/reselect_jump.py <run> --every-k 2 --workers 4 \
        --compare ../models/bolt-dog-jump-runup-16i-35.5M <run>/best     # also score these folders (not candidates)

Motor brake (bolt_env.MOTOR) off by default: run jump-trot was trained without it; --motor-brake for runs trained with it.
<run> must hold checkpoints/policy_<n>_steps.zip with checkpoints/policy_vecnormalize_<n>_steps.pkl.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bolt_env import add_brake_args, set_motor_brake   # noqa: E402


def score_one(job):
    """job = (label, folder, zip name, stats name, settings): both exams' scores for one brain."""
    import torch
    torch.set_num_threads(1)
    from jump_env import JumpPolicy, exam
    label, folder, zname, sname, a = job
    t0 = time.time()
    pol = JumpPolicy(Path(folder), zname, sname)
    row = dict(label=label, folder=str(folder), zip=zname, stats=sname)
    for tag, speed, dist, gait in (("mid", a["mid_speed"], a["mid_dist"], True),
                                   ("still", a["still_speed"], a["still_dist"], False)):
        r = exam(pol, episodes=a["episodes"], seed=a["seed"], heights=tuple(a["heights"]), dist=dist,
                 max_steps=a["max_steps"], speed=speed, strict_too=False, gait=gait, **a["exam_kw"])
        for k, v in r.items():
            row[f"{tag}_{k.split('_')[-1]}"] = v
        row[f"{tag}_mean"] = sum(r.values()) / len(r)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder holding checkpoints/")
    ap.add_argument("--every-k", type=int, default=1, help="examine every k-th checkpoint (default 1 = all); the last is always examined")
    ap.add_argument("--min-steps", type=int, default=0, help="skip checkpoints before this many steps (default 0)")
    ap.add_argument("--episodes", type=int, default=30, help="tries per bar height and exam (default 30)")
    ap.add_argument("--seed", type=int, default=7000, help="first seed (default 7000, as the model cards' exams)")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.40, 0.50], help="bar heights in m (default 0.40 0.50)")
    ap.add_argument("--max-steps", type=int, default=220, help="control steps per try, 50 per second (default 220 = 4.4 s, the model cards')")
    ap.add_argument("--mid-speed", type=float, default=1.7, help="mid-stride exam: speed at the start, m/s (default 1.7, the trot hand-over)")
    ap.add_argument("--mid-dist", type=float, default=1.1, help="mid-stride exam: bar distance at the start, m (default 1.1, the trot hand-over)")
    ap.add_argument("--still-speed", type=float, default=2.0, help="still-legs exam: speed, m/s (default 2.0, the model cards')")
    ap.add_argument("--still-dist", type=float, default=1.5, help="still-legs exam: bar distance, m (default 1.5, the model cards')")
    ap.add_argument("--compare", nargs="*", default=[], help="model folders (policy.zip + vecnormalize.pkl) to score too, not candidates")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes (default: all CPUs)")
    ap.add_argument("--out", help="JSON table (default <run>/reselect_jump.json)")
    ap.add_argument("--no-copy", action="store_true", help="do not write <run>/best_midstride/")
    add_brake_args(ap, default=False)
    a = ap.parse_args()
    print("motor brake (on, rad/s, N·m per rad/s):", set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain), flush=True)
    run = Path(a.run)
    ck = run / "checkpoints"
    cks = []
    for z in ck.glob("policy_*_steps.zip"):
        n = int(z.stem.split("_")[1])
        if n >= a.min_steps and (ck / f"policy_vecnormalize_{n}_steps.pkl").exists():
            cks.append(n)
    cks.sort()
    assert cks, f"no checkpoint pairs in {ck}"
    picked = cks[::a.every_k]
    if picked[-1] != cks[-1]:
        picked.append(cks[-1])
    settings = dict(episodes=a.episodes, seed=a.seed, heights=a.heights, max_steps=a.max_steps, mid_speed=a.mid_speed,
                    mid_dist=a.mid_dist, still_speed=a.still_speed, still_dist=a.still_dist, exam_kw=dict(motor_brake=a.motor_brake))
    jobs = [(n, str(ck), f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl", settings) for n in picked]
    jobs += [(f"compare:{c}", c, "policy.zip", "vecnormalize.pkl", settings) for c in a.compare]
    print(f"{len(picked)} of {len(cks)} checkpoints + {len(a.compare)} comparison folders; mid-stride {a.mid_speed} m/s "
          f"from {a.mid_dist} m, still legs {a.still_speed} m/s from {a.still_dist} m, heights {a.heights}, "
          f"{a.episodes} tries, seed {a.seed}, max steps {a.max_steps}, motor brake {'on' if a.motor_brake else 'off'}, "
          f"{a.workers} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(jobs)))) as pool:
        for row in pool.map(score_one, jobs):
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                              if k not in ("folder", "zip", "stats")}), flush=True)
    cands = [r for r in rows if isinstance(r["label"], int)]
    win = max(cands, key=lambda r: (r["mid_mean"], r["still_mean"], r["label"]))
    out = dict(command="python " + " ".join([sys.argv[0]] + sys.argv[1:]), settings={k: v for k, v in settings.items() if k != "exam_kw"},
               motor_brake=dict(on=a.motor_brake, speed=a.brake_speed, gain=a.brake_gain),
               winner=win["label"], rows=rows, minutes=round((time.time() - t0) / 60, 1))
    path = Path(a.out) if a.out else run / "reselect_jump.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwinner: {win['label']} steps, mid-stride mean {win['mid_mean']:.3f}, still-legs mean {win['still_mean']:.3f}")
    for r in rows:
        print(f"  {str(r['label']):>14}  mid " + " ".join(f"{r[k]:.2f}" for k in r if k.startswith("mid_") and k != "mid_mean")
              + f"  | still " + " ".join(f"{r[k]:.2f}" for k in r if k.startswith("still_") and k != "still_mean"))
    print(f"table: {path}")
    if not a.no_copy:
        dst = run / "best_midstride"
        dst.mkdir(exist_ok=True)
        shutil.copy(ck / win["zip"], dst / "policy.zip")
        shutil.copy(ck / win["stats"], dst / "vecnormalize.pkl")
        (dst / "best.json").write_text(json.dumps(dict(steps=win["label"], scores=win, command=out["command"],
                                                       settings=out["settings"]), indent=1))
        print(f"winner copied to {dst}")


if __name__ == "__main__":
    main()
