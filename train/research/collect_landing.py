"""DAgger data for jump landings: trot -> (jump brain) -> canter bridge -> trot, with the dog-motion brain
planned (PPSBolt) and every planned step labelled (brain raw mean + chosen offset). Student r3 lost the
landing hand-over (8/9 -> 2/9) after the sit/lie data; this adds the states it forgot.

    python research/collect_landing.py --brain runs/research/student_r3 --tries 16 --seed 40000 --beta 0.3 --out runs/research/land_r0.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run_switches import PPSBolt          # noqa: E402
from pps_distill import raw_mean, slice_obs   # noqa: E402


class Recorder(PPSBolt):
    def __init__(self, *a, beta=1.0, **kw):
        super().__init__(*a, **kw)
        self.beta, self.O, self.E, self.rng = beta, [], [], np.random.default_rng(kw.get("seed", 0))
        self.recording = False

    def _clip(self, name, seconds):
        im = self.im
        clip = im.clips[name]
        same_brain = self.skill in ("walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie")
        if name != self.skill or not same_brain:
            im.clip, im.tau0, im.t = clip, self._entry(clip), 0
            if not same_brain:
                im.last_action = np.zeros(12)
        if name != self.planned_clip or not same_brain:
            self.queue, self.planned_clip = [], name
            self.planner.plan[:] = 0.0
        self.skill = name
        self.gait = name
        im.limit = 10 ** 9
        for _ in range(int(round(seconds / self.dt))):
            if not self.queue:
                self.queue = list(self.planner.new_plan(im))
                self.follow = self.rng.random() < self.beta
            e = self.queue.pop(0)
            obs = im._obs()
            if self.recording:
                o = slice_obs(self.dog, obs)
                self.O.append(o)
                self.E.append(raw_mean(self.dog, o[None])[0] + e)
            im.step(np.clip(self.dog.act(obs) + (e if self.follow else 0.0), -1, 1))
            if self._after_step(name):
                self.queue = []
                return self._fell(name)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--tries", type=int, default=16)
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bolt = Recorder(seed=args.seed, imitate=(Path(args.brain), "policy.zip", "vecnormalize.pkl"), beta=args.beta)
    rng = np.random.default_rng(args.seed)
    for k in range(args.tries):
        bolt.start("trot", seed=args.seed + k)
        bolt.recording = False
        bolt.do("trot", 2.0)
        h = float(rng.uniform(0.2, 0.4))
        # records every planned dog-motion step of the jump: the short trot approach, then (the point) the
        # canter bridge and trot after the jump brain hands back at the landing
        bolt.recording = True
        r = bolt.do("jump", height=h, ahead=float(rng.uniform(1.5, 2.0)), then=2.5)
        print(f"try {k}: h={h:.2f} clean={r.get('clean')} landed={r.get('landed')} fell_after={r.get('fell_after')} "
              f"samples={len(bolt.O)}", flush=True)
    np.savez_compressed(args.out, obs=np.array(bolt.O, dtype=np.float32), label=np.array(bolt.E, dtype=np.float32),
                        anchor_obs=np.zeros((0, len(bolt.O[0])), np.float32), anchor_label=np.zeros((0, 12), np.float32))
    print(f"saved {len(bolt.O)} -> {args.out}")


if __name__ == "__main__":
    main()
