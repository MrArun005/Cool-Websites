"""Where does the trot brain hand Bolt to the jump brain? (Step 4: the trot -> jump 40 cm gap)

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_gap.py --height 0.40 --n 10 \
        --out research/results/handover40_16g.json
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_gap.py --height 0.40 --n 10 --jump 16i \
        --out research/results/handover40_16i.json

Runs test_skills.py's jump sequence exactly (trot 2 s from the trot clip, hurdle 1.5-2 m ahead drawn from the
same seeds 1000 + 37 k, skills.py's approach and hand-over at JUMP_FROM) with the dog-motion brain the report
used (runs/imitate18b 15M), and records, at the moment the jump brain takes over: the bar distance
(hurdle_ahead, body centre to bar, as the jump exam measures it), the body's forward speed and its speed along
the jump, heading error, distance off the line, body height, and whether the legs are mid-stride. Then the
jump's result (clean on agility rules: over, bar not dropped, landed) and, for comparison, what the jump
brain's own exam uses (jump_env.exam: from rest pose with the body already moving at --exam-speed, --exam-dist
away).

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_gap.py --exam-start --exam-dist 1.5 --exam-speed 2.0 \
        --out research/results/examstart_16i.json      # the 16i exam's start state, same quantities

--jump 16i swaps skills.py's jump brain for the run-up brain models/bolt-dog-jump-runup-16i-35.5M (with its
200-step clock, as it was trained), keeping everything else the same; --jump <folder> any other jump brain
(e.g. the Kaggle "jump-trot" run's best/).

Measured 2026-10-28 (40 cm bar, results in research/results/):
  hand-over (handover40_16g.json, 10 tries): bar 1.07-1.10 m ahead (mean 1.08), 1.53-1.84 m/s (mean 1.68),
      1-2 paws on the floor, joints moving at 3.3-5.1 rad/s rms (mid-stride), body 0.36-0.40 m, heading -2..+3 deg,
      2-7 cm off the line. Clean: 16g 3/10 (as skills_report.json), 16i at the same hand-over 3/10 (handover40_16i.json).
  16i exam start (examstart_16i.json): 1.50 m, 2.00 m/s, 4 paws down, joints still (0.01 rad/s rms), body 0.44 m.
  16i trained from 0.5-1.5 m at 1-3 m/s (half its tries), always from the still dog stance.
  16i, fair exam at 40 cm, 30 tries (fair_exam.py --seed 7000 --heights 0.40 --gait):
      still legs  1.5 m 2.0 m/s 30/30, 1.5 m 1.7 m/s 27/30, 1.1 m 1.7 m/s 30/30
      mid-stride  1.5 m 2.0 m/s 11/30, 1.5 m 1.7 m/s 14/30, 1.1 m 1.7 m/s 15/30
So the hand-over's distance and speed are inside the 16i range and cost nothing; starting mid-stride (legs
swinging, 1-2 paws down) is what drops it from 100% to ~50% in its own exam and to 3/10 in the skill manager.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import test_skills as ts   # noqa: E402  (torch threads = 1)
import skills              # noqa: E402
from jump_env import JumpPolicy   # noqa: E402

BRAINS = {"16g": (skills.MODELS / "bolt-dog-jump-a1-16g-27M", 100),
          "16i": (skills.MODELS / "bolt-dog-jump-runup-16i-35.5M", 200)}


class Recorder:
    """Wraps the jump brain: the first act() of each jump records the state it was handed."""

    def __init__(self, inner, bolt):
        self.inner, self.bolt, self.at = inner, bolt, None

    def act(self, obs):
        w = self.bolt.world
        if w.t == 0 and self.at is None:
            d, R = w.data, w._rotation()
            v = d.qvel[0:3]
            self.at = dict(dist=round(w.hurdle_ahead(), 3), speed_fwd=round(float((R.T @ v)[0]), 2),
                           speed_along_jump=round(float(np.dot(v[:2], w.hurdle_dir)), 2),
                           heading_err_deg=round(float(np.degrees(w.heading_error())), 1), off_line=round(w.off_line(), 3),
                           height=round(float(d.qpos[2]), 3), joint_speed_rms=round(float(np.sqrt(np.mean(d.qvel[6:18] ** 2))), 2),
                           paws_down=int(w.feet_on_ground().sum()))
        return self.inner.act(obs)


def exam_starts(a):
    """The same quantities at the first step of the jump brain's own exam (jump_env.exam with --exam-dist,
    --exam-speed, fixed physics, straight; --exam-gait: mid-stride trot/canter pose), seeds 7000 + k."""
    from jump_env import JumpEnv
    env = JumpEnv(randomize=False, real=True, max_steps=200)
    env.exam_speed, env.exam_gait, env.exam_askew = a.exam_speed, a.exam_gait, (0.0, 0.0)
    rows = []
    for k in range(a.n):
        env.exam_bar = (a.exam_dist, a.height)
        env.reset(seed=7000 + k)
        d, R = env.data, env._rotation()
        v = d.qvel[0:3]
        rows.append(dict(dist=round(env.hurdle_ahead(), 3), speed_fwd=round(float((R.T @ v)[0]), 2),
                         speed_along_jump=round(float(np.dot(v[:2], env.hurdle_dir)), 2),
                         heading_err_deg=round(float(np.degrees(env.heading_error())), 1), off_line=round(env.off_line(), 3),
                         height=round(float(d.qpos[2]), 3), joint_speed_rms=round(float(np.sqrt(np.mean(d.qvel[6:18] ** 2))), 2),
                         paws_down=int(env.feet_on_ground().sum())))
    return rows


def summarize(h):
    return {k: dict(min=min(x[k] for x in h), mean=round(float(np.mean([x[k] for x in h])), 3), max=max(x[k] for x in h))
            for k in ("dist", "speed_fwd", "speed_along_jump", "heading_err_deg", "off_line", "height", "joint_speed_rms", "paws_down")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=0.40)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--jump", default="16g", help="16g, 16i, or a jump brain folder (policy.zip + vecnormalize.pkl, 200-step clock)")
    ap.add_argument("--imitate-steps", type=int, default=15_000_000)
    ap.add_argument("--exam-start", action="store_true", help="measure the jump exam's start state instead (no skill manager)")
    ap.add_argument("--exam-dist", type=float, default=1.5)
    ap.add_argument("--exam-speed", type=float, default=2.0)
    ap.add_argument("--exam-gait", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.exam_start:
        rows = exam_starts(a)
        out = dict(args=vars(a), start=summarize(rows), all=rows)
        print(json.dumps(out["start"], indent=1))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        return
    n = a.imitate_steps
    bolt = skills.Bolt(seed=a.seed, imitate=(skills.IMITATE_RUN, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"))
    folder, clock = BRAINS.get(a.jump, (Path(a.jump), 200))
    if a.jump != "16g":
        bolt.jumper = JumpPolicy(folder)
    bolt.world.max_steps = clock   # the clock input is t / max_steps
    rec = Recorder(bolt.jumper, bolt)
    bolt.jumper = rec
    tries = []
    for k in range(a.n):
        seed = a.seed + 37 * k
        rec.at = None
        bolt.start("trot", seed=seed)          # test_skills.seq_jump, step by step
        bolt.do("trot", 2.0)
        reached = ts.clean_so_far(bolt)
        ahead = float(np.random.default_rng(seed).uniform(1.5, 2.0))
        r = bolt.do("jump", height=a.height, ahead=ahead)
        row = dict(seed=seed, placed_at=round(ahead, 3), reached=reached, clean=bool(r["clean"] and not r["fell"]),
                   cleared=r["cleared"], touched=r["touched"], dropped=r["dropped"], landed=r["landed"], fell=r["fell"],
                   handover=rec.at)
        tries.append(row)
        print(json.dumps(row), flush=True)
    h = [t["handover"] for t in tries if t["handover"]]
    summary = summarize(h)
    out = dict(args=vars(a), jump_from=skills.JUMP_FROM, clean=sum(t["clean"] for t in tries), tries=len(tries),
               handover=summary, all=tries)
    print(json.dumps({k: out[k] for k in ("jump_from", "clean", "tries", "handover")}, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
