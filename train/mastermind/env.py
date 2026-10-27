"""MasterMind: one world, one observation, one action space for every behaviour Bolt has.

The world is skills.Bolt's (the real-hurdle world of JumpEnv(real=True), every expert's env attached to it), so
the experts can label any state the student visits with exactly the observation and motor mapping they learned.

    m = Master(seed=1)
    m.start("walk", seed=1)
    m.command("trot"); m.command("jump", height=0.3, ahead=1.7); m.command("sit") ...
    x = m.obs()                 # the student's observation (OBS floats)
    y = m.label()               # the teacher's joint targets for this state (12 radians)
    m.step(ctrl)                # execute joint targets (radians); None = the teacher's own

Action (student and labels): absolute joint position targets in radians. The student outputs u in [-1, 1]^12,
ctrl = clip(DOG + FULL * u, lo, hi) with FULL the joint's full range around the dog stance, so every expert's
motor command (imitation: clip reference + 0.4/1.0/1.2 rad corrections; jump: dog stance +- the same; self-right:
the full range; posture: crouch- or dog-stance centred) is reachable by the one brain.

Who teaches (the active "mode", from the command and the state, the same chains skills.py runs):
    walk trot pace canter turn_left turn_right sit lie   the dog-motion brain on that clip (a switch joins the new
                                                          clip at the moment closest to Bolt's pose, as skills.py)
    jump   the current gait up to JUMP_FROM (1.1 m) before the bar, the jump brain until it has landed (or 3 s),
           0.6 s of canter (BRIDGE), then the command returns to the gait before the jump
    stand  the posture brain's stand
    after a fall (skills.py's detector)  the command becomes "stand" and the teacher is the recovery chain:
           self-righting until belly-down and calm, the posture brain's lie (tuck), its stand for 3 s
At a switch the new teacher labels from the very first step; its bookkeeping "previous action" is the action
actually executed, expressed in its own units.

Observation (OBS = 84), every part present for every behaviour (zeros when it does not apply):
     0 gravity in body frame 3 | angular velocity x0.25 3 | linear velocity (body) 3 | height 1
    10 joint angles - dog stance 12 | joint velocities x0.1 12 | previous action u 12
    46 command one-hot 10 (COMMANDS)
    56 reference motion 21: active flag, clip phase sin/cos, reference joints next step - stance 12, reference
       height, body velocity 3, yaw rate, seconds since clip end   (the imitation brain's inputs; during a jump
       the approach gait's clip keeps playing, so the student itself has to decide when to take off)
    77 hurdle 6: present flag, distance ahead (clipped -1..3 m), bar height, sin heading error, metres off the
       line, time since the bar came within JUMP_FROM (s / 2)
    83 time since the last command (s / 3, clipped at 1)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import bolt_env  # noqa: E402
import getup_env  # noqa: E402
import imitate_env  # noqa: E402
import jump_env  # noqa: E402
from bolt_env import ACTION_SCALE, LIE_POSE, BoltEnv  # noqa: E402
from imitate_env import ALL_CLIPS, DOG  # noqa: E402
from skills import BRIDGE, CLIPS, GAITS, JUMP_FROM, Bolt, closest_moment  # noqa: E402
from teachers import imitate_brain  # noqa: E402

COMMANDS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "stand", "jump"]
OBS = 84

# ---- one physics call for every expert env, with the executed joint targets forced in ----
_REAL_PHYSICS = bolt_env.physics
_FORCE = [None]


def _physics(model, data, n):
    if _FORCE[0] is not None:
        data.ctrl[:] = _FORCE[0]
    _REAL_PHYSICS(model, data, n)


for _mod in (bolt_env, imitate_env, jump_env, getup_env):
    _mod.physics = _physics


class Master:
    def __init__(self, seed=0, imitate="18e", randomize=True):
        self.b = Bolt(seed=seed, randomize=randomize, auto_recover=False, imitate=imitate_brain(imitate))
        b = self.b
        self.w, self.im, self.sr = b.world, b.im, b.sr
        self.lo, self.hi = self.w.lo.copy(), self.w.hi.copy()
        self.full = np.maximum(self.hi - DOG, DOG - self.lo)
        self.dt = b.dt
        self.mode = None
        # patient=True (training): the recovery chain moves on only when the state says so (calm belly-down, legs
        # tucked), never on a timeout, so a state is never labelled "self-right" in one try and "tuck"/"stand" in
        # another (DAgger round 5-6 of the first run: the student never got up, the labels on its back contradicted)
        self.patient = False

    # ---------------- episode ----------------
    def start(self, clip="walk", tau=None, seed=None):
        self.b.start(clip, tau=tau, seed=seed)
        self.rng = self.im.rng
        self.cmd, self.mode, self.gait = clip, ("clip", clip), clip if clip in GAITS else "walk"
        self.cmd_t = 0
        self.last_u = np.clip((self.b.data.ctrl - DOG) / self.full, -1, 1)
        self.jump = None          # the jump in progress: dict(phase, height, after, ...)
        self.hurdle_on = False
        self.rec = None           # the recovery in progress: dict(phase, calm, n)
        self.fallen = False
        self.im.limit = 10 ** 9

    @property
    def data(self):
        return self.b.data

    @property
    def t(self):
        return self.b.t

    # ---------------- commands ----------------
    def _to_clip(self, name):
        im, d = self.im, self.data
        prev = self.mode
        if prev != ("clip", name):
            im.clip = im.clips[name]
            im.tau0, im.t = closest_moment(im.clip, d.qpos[7:19], d.qvel[6:18]), 0
            if prev is None or prev[0] != "clip":   # coming from another brain: its last action in this brain's units
                im.last_action = np.clip((d.ctrl - im.ref("joints")) / ACTION_SCALE, -1, 1)
        self.mode = ("clip", name)
        if name in GAITS:
            self.gait = name

    def _to_posture(self, cmd):
        env = self.sr
        if self.mode != ("posture", cmd):
            env.forced_command = cmd
            env.set_command(cmd)
            env.last_action = np.clip((self.data.ctrl - env.base_pose()) / env.action_scale(), -1, 1)
        self.mode = ("posture", cmd)

    def command(self, name, height=0.3, ahead=1.75, after=None):
        """The handler says something. Ignored while a recovery runs (the command is "stand" until it ends)."""
        if self.rec is not None:
            return
        self.cmd, self.cmd_t = name, 0
        self.b.events.append(dict(t=self.t, kind="switch", skill=name, prev=self.b.skill))
        self.jump = None
        if name in CLIPS:
            self._to_clip(name)
        elif name == "stand":
            self._to_posture("stand")
        elif name == "jump":
            gait = self.mode[1] if self.mode[0] == "clip" and self.mode[1] in GAITS else "walk"
            self.b.place_hurdle(ahead, height)
            self.hurdle_on = True
            self.jump = dict(phase="approach", height=height, after=after or gait, steps=0, out=dict(
                height=height, cleared=False, clean=False, landed=False, fell=False, fell_after=False, dropped=False,
                touched=False))
            self._to_clip(gait)
        else:
            raise ValueError(name)
        self.b.skill = name

    def _start_jump_brain(self):
        w = self.w
        self.b.events.append(dict(t=self.t, kind="switch", skill="jump", prev=self.b.skill))
        w.t = 0
        w.last_action = np.clip((self.data.ctrl - DOG) / ACTION_SCALE, -1, 1)
        w.touched = w.cleared = w.landed = w.dropped = False
        w.clear_t, w.best_past = None, w.paws_past().min()
        self.jump["phase"] = "jump"
        self.mode = ("jump",)

    def _start_recovery(self):
        env = self.sr
        self.b.events.append(dict(t=self.t, kind="recover"))
        env.t, env.best_up, env.stood_at, env.kind = 0, env._rotation()[2, 2], None, "fall"
        env.last_action = np.clip((self.data.ctrl - DOG) / env.scale, -1, 1)
        self.rec = dict(phase="selfright", calm=0, n=0)
        self.mode = ("selfright",)
        self.cmd, self.cmd_t, self.jump = "stand", 0, None
        self.b.skill = "recover"

    # ---------------- the teacher ----------------
    def _expert(self):
        """(obs fn, policy, base, scale, step fn) of the brain in charge in this mode."""
        b, mode = self.b, self.mode
        if mode[0] == "clip":
            im = self.im
            return im._obs, b.dog, im.ref("joints", 1), ACTION_SCALE, im.step
        if mode[0] == "jump":
            w = self.w
            return w._obs, b.jumper, DOG, ACTION_SCALE, w.step
        if mode[0] == "selfright":
            sr = self.sr
            return sr._obs, b.righter, DOG, sr.scale, sr.step
        sr = self.sr   # posture
        return (lambda: BoltEnv._obs(sr)), b.posture, sr.base_pose(), sr.action_scale(), (lambda a: BoltEnv.step(sr, a))

    def label(self):
        """The teacher's joint targets (radians) for the current state."""
        obs_fn, pol, base, scale, _ = self._expert()
        a = np.clip(np.asarray(pol.act(obs_fn()), dtype=float), -1, 1)
        return np.clip(base + scale * a, self.lo, self.hi)

    # ---------------- the student's view ----------------
    def obs(self):
        d, R, b = self.data, self.w._rotation(), self.b
        x = np.zeros(OBS)
        x[0:3] = R.T @ np.array([0, 0, -1.0])
        x[3:6] = d.qvel[3:6] * 0.25
        x[6:9] = R.T @ d.qvel[0:3]
        x[9] = d.qpos[2]
        x[10:22] = d.qpos[7:19] - DOG
        x[22:34] = d.qvel[6:18] * 0.1
        x[34:46] = self.last_u
        x[46 + COMMANDS.index(self.cmd)] = 1.0
        if self.mode[0] in ("clip", "jump"):   # a reference motion is playing (in a jump: the approach gait's)
            im = self.im
            ph = 2 * np.pi * im.clip.phase(im.tau())
            x[56] = 1.0
            x[57:59] = np.sin(ph), np.cos(ph)
            x[59:71] = im.ref("joints", 1) - DOG
            x[71] = im.ref("height", 1)
            x[72:75] = im.ref("v_body", 1)
            x[75] = im.ref("w_body", 1)[2]
            x[76] = im.finished()
        if self.hurdle_on:
            w = self.w
            x[77] = 1.0
            x[78] = np.clip(w.hurdle_ahead(), -1.0, 3.0)
            x[79] = w.hurdle_h
            x[80] = np.sin(w.heading_error())
            x[81] = w.off_line()
            x[82] = (w.t * self.dt / 2.0) if self.mode[0] == "jump" else 0.0
        x[83] = min(1.0, self.cmd_t * self.dt / 3.0)
        if b.randomize:
            x[:46] += self.rng.normal(0, 0.01, 46)
        return x.astype(np.float32)

    def ctrl_of(self, u):
        return np.clip(DOG + self.full * np.clip(u, -1, 1), self.lo, self.hi)

    # ---------------- one control step ----------------
    def step(self, ctrl=None):
        """Execute joint targets `ctrl` (radians; None: the teacher's). Returns dict(fell=bool, events)."""
        obs_fn, pol, base, scale, step_fn = self._expert()
        if ctrl is None:
            a = np.clip(np.asarray(pol.act(obs_fn()), dtype=float), -1, 1)
            ctrl = np.clip(base + scale * a, self.lo, self.hi)
        ctrl = np.clip(np.asarray(ctrl, dtype=float), self.lo, self.hi)
        a_exec = np.clip((ctrl - base) / scale, -1, 1)
        _FORCE[0] = ctrl
        try:
            step_fn(a_exec)
        finally:
            _FORCE[0] = None
        self.last_u = np.clip((ctrl - DOG) / self.full, -1, 1)
        if self.mode[0] != "clip":
            self.im.t += 1         # the reference keeps playing (jump: the approach gait's clip)
        self.cmd_t += 1
        return self._after(ctrl)

    def _skill_name(self):
        m = self.mode
        if m[0] == "clip":
            return m[1]
        return {"jump": "jump", "selfright": "self-righting"}.get(m[0]) or ("stand" if m[1] == "stand" else "posture_" + m[1])

    def _after(self, ctrl):
        b, w = self.b, self.w
        fell = b._after_step(self._skill_name())
        out = dict(fell=False)
        if self.rec is not None:
            self._recovery_tick()
            return out
        if fell:
            b.events.append(dict(t=b.t, kind="fall", skill=self._skill_name(), height=round(float(self.data.qpos[2]), 3),
                                 up=round(float(self.data.xmat[w.torso][8]), 2)))
            if self.jump is not None:
                j = self.jump
                if j["phase"] in ("approach", "jump"):
                    j["out"].update(fell=True, cleared=bool(w.cleared), touched=bool(w.touched), dropped=bool(w.dropped))
                else:
                    j["out"]["fell_after"] = True
                j["phase"] = "failed"
                self.last_jump = j["out"]
            out["fell"] = True
            self._start_recovery()
            return out
        j = self.jump
        if j is not None:
            if j["phase"] == "approach":
                j["steps"] += 1
                if w.hurdle_ahead() <= JUMP_FROM or j["steps"] >= 200 or w.hurdle_ahead() < -0.5:
                    self._start_jump_brain()
            elif j["phase"] == "jump":
                if w.landed or w.t >= 150:
                    o = j["out"]
                    o.update(cleared=bool(w.cleared), touched=bool(w.touched), dropped=bool(w.dropped),
                             landed=bool(w.landed), clean=bool(w.cleared and not w.touched and w.landed), steps=w.t)
                    if not w.landed:
                        b.events.append(dict(t=b.t, kind="no_landing"))
                    o["landed_at"] = b.t
                    j["phase"], j["bridge"] = "bridge", 0
                    self._to_clip("canter")
            elif j["phase"] == "bridge":
                j["bridge"] += 1
                if j["bridge"] >= int(round(BRIDGE / self.dt)):
                    self.last_jump = j["out"]
                    after = j["after"]
                    j["phase"] = "done"
                    self.command(after)   # the handler goes back to the gait before the jump
                    out["jump_done"] = True
        if self.hurdle_on and self.jump is None and w.hurdle_ahead() < -1.5:
            self.hurdle_on = False
        return out

    def _recovery_tick(self):
        r, env, d = self.rec, self.sr, self.data
        r["n"] += 1
        if r["phase"] == "selfright":
            standing = env._rotation()[2, 2] > 0.9 and np.linalg.norm(d.qvel[0:3]) < 0.2
            r["calm"] = r["calm"] + 1 if standing else 0
            if r["calm"] >= 15 or (r["n"] >= 200 and not self.patient):
                r["phase"], r["n"] = "tuck", 0
                self._to_posture("lie")
        elif r["phase"] == "tuck":
            if np.abs(d.qpos[7:19] - LIE_POSE).max() < 0.6 or (r["n"] >= 100 and not self.patient):
                r["phase"], r["n"] = "stand", 0
                self._to_posture("stand")
        elif r["phase"] == "stand" and r["n"] >= 150:
            p = env.posture()
            ok = bool(p["up"] > 0.9 and p["height"] > 0.38)
            self.b.events.append(dict(t=self.b.t, kind="recovered" if ok else "not_recovered", height=round(float(p["height"]), 3)))
            self.rec = None
            self.cmd, self.cmd_t = "stand", 0
            self.b.skill = "stand"
            self.b.floor_steps = 0

    def recovering(self):
        return self.rec is not None

    def shove(self, **kw):
        self.b.shove(**kw)
