"""Skill-switch test (test_skills.py's sequences, tally and 2 s no-fall window) for the plain skill manager
and for the same manager with the dog-motion brain wrapped in the PPS planner (pps.py).

    python research/run_switches.py --mode plain|pps --n 10 --only gaits rest jump30 shove --out research/results/sw_plain.json

Both use the SAME fixed brains (dog motion: models/bolt-dog-imitate-18b-17.5M, not the still-training
run's newest checkpoint, so the numbers do not move under us), the same seeds (1000 + 37 k) and the same
physics randomization. Only the dog-motion brain is planned; jump, self-righting and posture brains run
as before.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import test_skills as ts            # noqa: E402  (sets torch threads to 1)
from skills import CLIPS, GAITS, Bolt   # noqa: E402
from pps import Planner              # noqa: E402

BRAIN = (HERE.parent.parent / "models" / "bolt-dog-imitate-18b-17.5M", "policy.zip", "vecnormalize.pkl")


class PPSBolt(Bolt):
    """Bolt whose dog-motion skills run brain + planned perturbation (replanned every M steps)."""

    def __init__(self, *a, K=16, H=30, M=5, sigma=0.2, **kw):
        super().__init__(*a, **kw)
        self.planner = Planner(self.dog, CLIPS, K=K, H=H, M=M, sigma=sigma, seed=0)
        self.queue, self.planned_clip = [], None

    def start(self, *a, **kw):
        super().start(*a, **kw)
        self.queue, self.planned_clip = [], None
        self.planner.plan[:] = 0.0

    def _clip(self, name, seconds):
        im, d = self.im, self.data
        clip = im.clips[name]
        same_brain = self.skill in CLIPS
        if name != self.skill or not same_brain:
            im.clip, im.tau0, im.t = clip, self._entry(clip), 0
            if not same_brain:
                im.last_action = np.zeros(12)
        if name != self.planned_clip or not same_brain:   # a new clip: the old plan is for another reference
            self.queue, self.planned_clip = [], name
            self.planner.plan[:] = 0.0
        self.skill = name
        if name in GAITS:
            self.gait = name
        im.limit = 10 ** 9
        for _ in range(int(round(seconds / self.dt))):
            if not self.queue:
                self.queue = list(self.planner.new_plan(im))
            e = self.queue.pop(0)
            im.step(np.clip(self.dog.act(im._obs()) + e, -1, 1))
            if self._after_step(name):
                self.queue = []
                return self._fell(name)
        return True

    def _entry(self, clip):
        from skills import closest_moment
        return closest_moment(clip, self.data.qpos[7:19], self.data.qvel[6:18])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="plain")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*", default=["gaits", "rest", "jump30", "shove"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "pps":
        bolt = PPSBolt(seed=args.seed, imitate=BRAIN, K=args.K, H=args.H, M=args.M, sigma=args.sigma)
    else:
        bolt = Bolt(seed=args.seed, imitate=BRAIN)
    tally, started = ts.Tally(), time.time()
    for name in args.only:
        for k in range(args.n):
            ts.SEQUENCES[name](bolt, args.seed + 37 * k, tally)
        print(f"{name}: done ({time.time() - started:.0f} s)", flush=True)
    rows = []
    for s in tally.order:
        n, ok = tally.tries[s], tally.ok[s]
        rows.append(dict(switch=s, ok=ok, tries=n, rate=round(ok / n, 2) if n else None, not_reached=tally.skipped[s],
                         failures=tally.fails[s]))
        print(f"{s:34s} {ok:>3d}/{n:<3d} not reached {tally.skipped[s]}")
    out = dict(args=vars(args), seconds=round(time.time() - started), switches=rows)
    if args.mode == "pps":
        out["planner"] = bolt.planner.stats
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
