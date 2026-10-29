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
    jump   the current gait up to jump_from (v2: 1.1 m, v3: 1.5 m) before the bar, the jump brain until it has
           landed (or jump_limit steps), `bridge` s of canter (0.6), then the command returns to the gait before the
           jump. Teacher sets (which brain teaches what) are in teachers.SETS.
    stand  the posture brain's stand
    after a fall (skills.py's detector)  the command becomes "stand" and the teacher is the recovery chain:
           self-righting until belly-down and calm, the posture brain's lie (tuck), its stand for 3 s. With rec_cmd
           (teacher set v3) the student's command input follows the phase: "stand" while self-righting, "lie" while
           tucking (from calm belly-down until every joint is within 0.6 rad of LIE_POSE, at most 2 s), then
           "stand" again, the command clock restarting at each change
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
       line, time since the jump brain took over (s / 2)
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
from imitate_env import ALL_CLIPS, DOG, ImitatePolicy  # noqa: E402
from jump_env import JumpPolicy  # noqa: E402
from skills import CLIPS, GAITS, JUMP, Bolt, closest_moment  # noqa: E402
from teachers import imitate_brain, imitate_policy, teacher_set  # noqa: E402

# ---- sim-to-real stress test (test_master.py --obs-noise/--latency-ms/--motor-lag-ms/--gain-scale; all off by default) ----
# Observation noise, uniform in +-level (legged_gym's noise model, Rudin et al. 2021, the setup used to take A1/Go1/Go2
# policies to hardware: noise_scales dof_pos 0.01 rad, dof_vel 1.5 rad/s, ang_vel 0.2 rad/s, gravity 0.05, lin_vel 0.1 m/s).
# Base height has no sensor on a real robot (it needs a state estimator); 0.02 m is our assumption, not a spec.
NOISE = dict(grav=0.05, gyro=0.2, linvel=0.1, height=0.02, qpos=0.01, qvel=1.5)
REAL_OFF = dict(noise=0.0, delay=0.0, lag=0.0, gain=1.0,
                # more real-body gaps (29 Oct, all off by default): constant per-try biases of what a real robot
                # estimates rather than measures (IMU tilt rad, gyro rad/s, body speed m/s, body height m), joint
                # encoder rounding (rad), the chance a control packet is lost (the motors keep the last target),
                # payload as a share of the body mass, and foot grip as a multiple of the floor friction
                imu_bias=0.0, gyro_bias=0.0, vel_bias=0.0, height_bias=0.0, quant=0.0, drop=0.0,
                payload=0.0, friction=1.0)

COMMANDS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "stand", "jump"]
OBS = 84

# ---- one physics call for every expert env, with the executed joint targets forced in ----
_REAL_PHYSICS = bolt_env.physics
_FORCE = [None]
_LAG = [None]   # (tau s, filtered targets): the motor-lag stress test (Master.real["lag"]), only inside Master.step
_SPLIT = [None]  # (k, older targets): a delay that is not a whole control step; the first k physics steps get the older


def _physics(model, data, n):
    if _FORCE[0] is not None:
        data.ctrl[:] = _FORCE[0]
    if _LAG[0] is None and _SPLIT[0] is None:
        _REAL_PHYSICS(model, data, n)
        return
    target = data.ctrl.copy()
    k, older = _SPLIT[0] if _SPLIT[0] is not None else (0, target)
    for i in range(n):   # one physics step (500 Hz) at a time
        want = older if i < k else target
        if _LAG[0] is not None:   # first-order lag of the joint targets
            tau, y = _LAG[0]
            y += (1.0 - np.exp(-model.opt.timestep / tau)) * (want - y)
            want = y
        data.ctrl[:] = want
        _REAL_PHYSICS(model, data, 1)
    data.ctrl[:] = target


for _mod in (bolt_env, imitate_env, jump_env, getup_env):
    _mod.physics = _physics


class Master:
    def __init__(self, seed=0, imitate=None, randomize=True, teachers="v2", **over):
        """teachers: a teacher set of teachers.SETS ("v2": the first student's, "v3": 18e-36M + 18c canter + 16i
        run-up jump); `imitate` and keyword arguments override single entries of the set."""
        self.ts = ts = teacher_set(teachers, dog=imitate, **over)
        self.teachers = teachers
        self.b = Bolt(seed=seed, randomize=randomize, auto_recover=False, imitate=imitate_brain(ts["dog"]))
        b = self.b
        self.w, self.im, self.sr = b.world, b.im, b.sr
        if Path(ts["jump"]) != JUMP:
            b.jumper = JumpPolicy(ts["jump"])
        self.w.max_steps = ts["jump_steps"]   # the jump brain's clock input is t / its try length
        self.canter = imitate_policy(ts["canter"]) if ts["canter"] else None
        self.land = imitate_policy(ts["land"]) if ts["land"] else None
        loaded, self.clip_pol = {}, {}   # clips: {clip: brain name}, one ImitatePolicy per brain
        for c, n in (ts.get("clips") or {}).items():
            if n not in loaded:
                loaded[n] = imitate_policy(n)
            self.clip_pol[c] = loaded[n]
        # approach: the brain for the gait clip of a jump's run-up (v4: 18e-36M, whose trot 16i's hand-over was tuned on)
        self.approach = imitate_policy(ts["approach"]) if ts.get("approach") else None
        self.land_left = 0
        self.jump_from, self.bridge = ts["jump_from"], ts["bridge"]
        self.lo, self.hi = self.w.lo.copy(), self.w.hi.copy()
        self.full = np.maximum(self.hi - DOG, DOG - self.lo)
        self.dt = b.dt
        self.mode = None
        # patient=True (training): the recovery chain moves on only when the state says so (calm belly-down, legs
        # tucked), never on a timeout, so a state is never labelled "self-right" in one try and "tuck"/"stand" in
        # another (DAgger round 5-6 of the first run: the student never got up, the labels on its back contradicted)
        self.patient = False
        # the stress test (REAL_OFF: today's numbers): noise = multiple of NOISE, delay = s of action delay (2 ms steps),
        # lag = s of first-order motor lag, gain = PD gain (kp and kv) scale on top of the +-15% kp randomization
        self.real = dict(REAL_OFF)
        m = self.w.model
        self._kp0, self._kv0 = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, 2].copy()

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
        self._real_body()
        self.land_left = 0
        self.im.limit = 10 ** 9
        r, m = self.real, self.w.model
        if r["gain"] != 1.0 or not np.allclose(m.actuator_biasprm[:, 2], self._kv0):
            kp = m.actuator_gainprm[:, 0].copy() if self.b.randomize else self._kp0.copy()   # reset re-drew kp
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp * r["gain"], -kp * r["gain"]
            m.actuator_biasprm[:, 2] = self._kv0 * r["gain"]
        d = int(round(r["delay"] / m.opt.timestep))          # delay in physics steps: whole control steps + a part
        self._back, self._sub = divmod(d, self.w.frame_skip)
        self._queue = [self.data.ctrl.copy() for _ in range(self._back + 1)]   # the robot holds its first targets
        self._lag_y = self.data.ctrl.copy()
        self._nrng = np.random.default_rng(None if seed is None else seed + 7919)

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
            w = self.w   # no flags left over from an earlier jump (a fall in the approach reads them)
            w.touched = w.cleared = w.landed = w.dropped = w.under = False
            self.jump = dict(phase="approach", height=height, after=after or gait, steps=0, out=dict(
                height=height, cleared=False, clean=False, landed=False, fell=False, fell_after=False, dropped=False,
                touched=False))
            self._to_clip(self.ts.get("run_up") or gait)   # run_up: the gait of the approach (v3: canter, ~2 m/s)
        else:
            raise ValueError(name)
        self.b.skill = name

    def _start_jump_brain(self):
        w = self.w
        self.b.events.append(dict(t=self.t, kind="switch", skill="jump", prev=self.b.skill))
        w.t = 0
        w.last_action = np.clip((self.data.ctrl - DOG) / ACTION_SCALE, -1, 1)
        w.start_try()
        self.jump["phase"] = "jump"
        self.jump["out"]["v_handover"] = round(float((w._rotation().T @ self.data.qvel[0:3])[0]), 2)
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
            pol = b.dog
            if self.land is not None and self.land_left > 0:
                pol = self.land
            elif self.approach is not None and self.jump is not None and self.jump["phase"] == "approach":
                pol = self.approach
            elif mode[1] in self.clip_pol:
                pol = self.clip_pol[mode[1]]
            elif self.canter is not None and mode[1] == "canter":
                pol = self.canter
            return im._obs, pol, im.ref("joints", 1), ACTION_SCALE, im.step
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
        k = self.real["noise"]
        if k:   # the stress test's sensor noise, in observation units (gyro x0.25, joint speed x0.1)
            u = lambda n: self._nrng.uniform(-1, 1, n)  # noqa: E731
            x[0:3] += k * NOISE["grav"] * u(3)
            x[3:6] += k * NOISE["gyro"] * 0.25 * u(3)
            x[6:9] += k * NOISE["linvel"] * u(3)
            x[9] += k * NOISE["height"] * u(1)[0]
            x[10:22] += k * NOISE["qpos"] * u(12)
            x[22:34] += k * NOISE["qvel"] * 0.1 * u(12)
        r = self.real
        if r["imu_bias"] or r["gyro_bias"] or r["vel_bias"] or r["height_bias"]:
            B = self._bias
            x[0:3] += B["grav"]
            x[3:6] += B["gyro"] * 0.25
            x[6:9] += B["vel"]
            x[9] += B["height"]
        if r["quant"]:   # joint encoders report in steps
            x[10:22] = np.round(x[10:22] / r["quant"]) * r["quant"]
        return x.astype(np.float32)

    def _real_body(self):
        """Per try: draw the constant sensor biases, and apply payload and foot grip (the stress test's real body)."""
        r, m = self.real, self.w.model
        brng = np.random.default_rng(int(self.rng.integers(1 << 31)))
        sgn = lambda n: brng.choice([-1.0, 1.0], n) * brng.uniform(0.5, 1.0, n)  # noqa: E731
        self._bias = dict(grav=r["imu_bias"] * sgn(3), gyro=r["gyro_bias"] * sgn(3), vel=r["vel_bias"] * sgn(3),
                          height=r["height_bias"] * float(sgn(1)[0]))
        self._last_exec = None
        if not hasattr(self, "_mass0"):
            self._mass0, self._fric0 = m.body_mass.copy(), m.geom_friction[:, 0].copy()
        if not self.b.randomize:   # randomized tries re-draw mass and grip at every start; fixed ones restore them
            m.body_mass[:], m.geom_friction[:, 0] = self._mass0, self._fric0
        if r["payload"]:
            m.body_mass[self.w.torso] += r["payload"] * float(np.sum(self._mass0))
        if r["friction"] != 1.0:
            m.geom_friction[:, 0] = m.geom_friction[:, 0] * r["friction"]

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
        if self.real["drop"] and self._last_exec is not None and self._nrng.random() < self.real["drop"]:
            ctrl = self._last_exec       # the packet was lost: the motors keep the last target they got
        self._last_exec = ctrl
        sent = ctrl                      # what the brain sent (its "previous action" input)
        if self.real["delay"]:           # stress test: the motors get the targets sent `delay` s ago
            self._queue.append(ctrl)
            self._queue = self._queue[-(self._back + 2):]
            ctrl = self._queue[-1 - self._back]
            if self._sub:
                _SPLIT[0] = (self._sub, self._queue[-2 - self._back])
        a_exec = np.clip((ctrl - base) / scale, -1, 1)
        _FORCE[0] = ctrl
        if self.real["lag"]:
            _LAG[0] = (self.real["lag"], self._lag_y)
        try:
            step_fn(a_exec)
        finally:
            _FORCE[0] = _LAG[0] = _SPLIT[0] = None
        self.last_u = np.clip((sent - DOG) / self.full, -1, 1)
        if self.mode[0] != "clip":
            self.im.t += 1         # the reference keeps playing (jump: the approach gait's clip)
        self.cmd_t += 1
        self.land_left = max(0, self.land_left - 1)
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
                ahead = w.hurdle_ahead()
                gate = self.ts.get("jump_phase")
                in_phase = False
                if gate and self.mode[0] == "clip" and ahead <= self.ts["jump_max"]:
                    ph = self.im.clip.phase(self.im.tau())
                    in_phase = gate[0] <= ph < gate[1]
                if ahead <= self.jump_from or in_phase or j["steps"] >= 200 or ahead < -0.5:
                    self._start_jump_brain()
            elif j["phase"] == "jump":
                if w.landed or w.t >= self.ts["jump_limit"]:
                    o = j["out"]
                    o.update(cleared=bool(w.cleared), touched=bool(w.touched), dropped=bool(w.dropped),
                             landed=bool(w.landed), clean=bool(w.cleared and not w.touched and w.landed), steps=w.t)
                    if not w.landed:
                        b.events.append(dict(t=b.t, kind="no_landing"))
                    o["landed_at"] = b.t
                    if self.land is not None:
                        self.land_left = int(round(self.ts["land_s"] / self.dt))
                    j["phase"], j["bridge"] = "bridge", 0
                    if self.ts.get("land_hold", 0) > 0:   # the jump brain lands Bolt itself (Bolt's Day Out: landing falls 12 -> 1)
                        j["phase"] = "hold"
                    elif self.bridge > 0:
                        self._to_clip("canter")
                    else:   # no bridge: straight back to the gait
                        self._end_jump(out)
            elif j["phase"] == "hold":   # still the jump brain, slowing Bolt down, for land_hold s; then the gait
                j["bridge"] += 1
                if j["bridge"] >= int(round(self.ts["land_hold"] / self.dt)):
                    self._end_jump(out)
            elif j["phase"] == "bridge":
                j["bridge"] += 1
                if j["bridge"] >= int(round(self.bridge / self.dt)):
                    self._end_jump(out)
        if self.hurdle_on and self.jump is None and w.hurdle_ahead() < -1.5:
            self.hurdle_on = False
        return out

    def _end_jump(self, out):
        j = self.jump
        self.last_jump = j["out"]
        j["phase"] = "done"
        self.command(j["after"])   # the handler goes back to the gait before the jump
        out["jump_done"] = True

    def _recovery_tick(self):
        r, env, d = self.rec, self.sr, self.data
        r["n"] += 1
        if r["phase"] == "selfright":
            standing = env._rotation()[2, 2] > 0.9 and np.linalg.norm(d.qvel[0:3]) < 0.2
            r["calm"] = r["calm"] + 1 if standing else 0
            if r["calm"] >= 15 or (r["n"] >= self.ts.get("sr_limit", 200) and not self.patient):
                if self.ts.get("tuck", True):
                    r["phase"], r["n"] = "tuck", 0
                    self._to_posture("lie")
                    if self.ts.get("rec_cmd"):   # the student is told the phase: "lie" while tucking
                        self.cmd, self.cmd_t = "lie", 0
                else:   # straight to the stand: no lie/stand labels on the same lying states
                    r["phase"], r["n"] = "stand", 0
                    self._to_posture("stand")
        elif r["phase"] == "tuck":
            if np.abs(d.qpos[7:19] - LIE_POSE).max() < 0.6 or (r["n"] >= 100 and not self.patient):
                r["phase"], r["n"] = "stand", 0
                self._to_posture("stand")
                if self.ts.get("rec_cmd"):   # ... and "stand" (clock from 0) once tucked
                    self.cmd, self.cmd_t = "stand", 0
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
