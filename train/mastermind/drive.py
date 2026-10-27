"""A Bolt-shaped driver over the Master env, so test_skills.py's sequences run unchanged on ONE brain.

    d = Driver(Master(seed=1000), brain)     # brain: Student (one network) or None (the teachers, via the env)
    test_skills.SEQUENCES["gaits"](d, seed, tally)

do(skill) gives the command and runs the brain; a fall starts the recovery (the command becomes "stand") and do
returns False after it, as skills.Bolt with auto_recover. With brain=None the env's teacher chains run, which is the
skill manager again (a check that the unified env reproduces skills.py).
"""
import numpy as np


class Driver:
    def __init__(self, master, brain=None):
        self.m, self.brain = master, brain
        self.dt = master.dt
        self.sr = master.sr

    # Bolt's attributes the tests read
    @property
    def events(self):
        return self.m.b.events

    @property
    def trace(self):
        return self.m.b.trace

    @property
    def t(self):
        return self.m.b.t

    def start(self, clip="walk", tau=None, seed=None):
        self.m.start(clip, tau=tau, seed=seed)

    def shove(self, **kw):
        self.m.shove(**kw)

    def _step(self):
        ctrl = self.brain.act(self.m.obs()) if self.brain is not None else None
        return self.m.step(ctrl)

    def _finish_recovery(self):
        n = 0
        while self.m.recovering() and n < 1000:
            self._step()
            n += 1

    def _run(self, steps):
        for _ in range(steps):
            r = self._step()
            if r["fell"]:
                self._finish_recovery()
                return False
        return True

    def do(self, skill, seconds=None, **kw):
        if skill == "jump":
            return self._jump(**kw)
        self.m.command(skill)
        return self._run(int(round((seconds or 2.0) / self.dt)))

    def _jump(self, height=0.3, ahead=1.75, then=2.0):
        m = self.m
        m.command("jump", height=height, ahead=ahead)
        for _ in range(500):
            r = self._step()
            if r["fell"]:
                out = dict(m.last_jump)
                self._finish_recovery()
                return out
            if r.get("jump_done"):
                break
        out = dict(m.last_jump)
        ok = self._run(int(round(then / self.dt)))
        out["fell_after"] = not ok
        return out
