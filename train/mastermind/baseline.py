"""Skill-manager baseline (skills.py + test_skills.py sequences) with a chosen dog-motion brain, so the
student can be compared with the hand-over manager running the SAME teachers.

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/baseline.py --imitate 18e --n 30 --out mastermind/results/manager_18e_n30.json
    ... --teachers v3     the manager with teacher set v3 (teachers.SETS: 18e-36M, 18c-29.5M for the canter clip, the
                          run-up jump brain 16i handed over 1.1 m before the bar, student r4 after landings);
                          skills.py itself is unchanged
--with-50: also the 50 cm jump sequence (trot, 50 cm bar 1.5-2 m ahead, trot), which test_skills.py does not have.
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
import numpy as np  # noqa: E402
from imitate_env import ImitatePolicy  # noqa: E402
from jump_env import JumpPolicy  # noqa: E402
from skills import CLIPS, GAITS, Bolt, closest_moment  # noqa: E402
from teachers import imitate_brain, teacher_set  # noqa: E402

EXTRA = {"jump50": test_skills.seq_jump(0.50)}


class Manager(Bolt):
    """skills.Bolt with a teacher set: a separate canter brain, another jump brain, its hand-over distance, try
    length, time limit and bridge, and a landing brain for land_s after a landing. Same code as Bolt._clip/_jump
    otherwise."""

    def __init__(self, seed=0, teachers="v3", **over):
        ts = self.ts = teacher_set(teachers, **over)
        super().__init__(seed=seed, imitate=imitate_brain(ts["dog"]))
        self.jumper = JumpPolicy(ts["jump"])
        self.world.max_steps = ts["jump_steps"]
        self.canter = ImitatePolicy(*imitate_brain(ts["canter"])) if ts["canter"] else None
        self.land = ImitatePolicy(*imitate_brain(ts["land"])) if ts["land"] else None
        self.land_until = -1

    def start(self, *a, **k):
        super().start(*a, **k)
        self.land_until = -1

    def _clip(self, name, seconds):
        im, d = self.im, self.data
        clip = im.clips[name]
        same_brain = self.skill in CLIPS
        if name != self.skill or not same_brain:
            im.clip, im.tau0, im.t = clip, closest_moment(clip, d.qpos[7:19], d.qvel[6:18]), 0
            if not same_brain:
                im.last_action = np.zeros(12)
        self.skill = name
        if name in GAITS:
            self.gait = name
        im.limit = 10 ** 9
        pol = self.canter if (self.canter is not None and name == "canter") else self.dog
        for _ in range(int(round(seconds / self.dt))):
            p = self.land if (self.land is not None and self.t < self.land_until) else pol
            im.step(p.act(im._obs()))
            if self._after_step(name):
                return self._fell(name)
        return True

    def _jump(self, height=0.3, ahead=1.75, after=None, then=2.0, bridge=None, max_seconds=4.0):
        ts, w, gait = self.ts, self.world, self.skill if self.skill in GAITS else self.gait
        bridge = ts["bridge"] if bridge is None else bridge
        after = after or gait
        self.place_hurdle(ahead, height)
        out = dict(height=height, cleared=False, clean=False, landed=False, fell=False, fell_after=False, dropped=False, touched=False)
        approach = gait if self.skill in GAITS else "walk"
        steps = 0
        while w.hurdle_ahead() > ts["jump_from"] and steps < int(max_seconds / self.dt):
            if not self._clip(approach, self.dt):
                out["fell"] = True
                return out
            steps += 1
            if w.hurdle_ahead() < -0.5:
                break
        self.events.append(dict(t=self.t, kind="switch", skill="jump", prev=self.skill))
        w.t, w.last_action = 0, np.zeros(12)
        w.touched = w.cleared = w.landed = w.dropped = False
        w.clear_t, w.best_past = None, w.paws_past().min()
        self.skill = "jump"
        for _ in range(ts["jump_limit"]):
            w.step(self.jumper.act(w._obs()))
            if self._after_step("jump"):
                out.update(fell=True, cleared=w.cleared, touched=w.touched, dropped=w.dropped)
                self._fell("jump")
                return out
            if w.landed:
                break
        out.update(cleared=bool(w.cleared), touched=bool(w.touched), dropped=bool(w.dropped), landed=bool(w.landed),
                   clean=bool(w.cleared and not w.touched and w.landed), steps=w.t)
        if not w.landed:
            self.events.append(dict(t=self.t, kind="no_landing"))
        out["landed_at"] = self.t
        self.land_until = self.t + int(round(ts["land_s"] / self.dt))
        ok = self._clip("canter", bridge) if bridge else True
        if ok and then:
            self.events.append(dict(t=self.t, kind="switch", skill=after, prev=self.skill))
            ok = self._clip(after, then)
        out["fell_after"] = not ok
        return out


def run(bolt, n, seed, only=None, with50=False):
    tally, started = test_skills.Tally(), time.time()
    seqs = dict(test_skills.SEQUENCES, **EXTRA)
    for name in only or (list(seqs) if with50 else test_skills.SEQUENCES):
        for k in range(n):
            seqs[name](bolt, seed + 37 * k, tally)
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
    ap.add_argument("--imitate", default="18e", help="18b or 18e (without --teachers)")
    ap.add_argument("--teachers", help="a teacher set of teachers.SETS, e.g. v3 (the Manager class)")
    ap.add_argument("--with-50", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    bolt = Manager(seed=a.seed, teachers=a.teachers) if a.teachers else Bolt(seed=a.seed, imitate=imitate_brain(a.imitate))
    rows, secs = run(bolt, a.n, a.seed, a.only, a.with_50)
    show(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(imitate=a.imitate, teachers=a.teachers, n=a.n, seed=a.seed, seconds=secs,
                                               switches=rows), indent=1))
