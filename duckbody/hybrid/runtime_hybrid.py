"""The hybrid duck's controller: commands -> plans of brains (Microduck's ONNX policies and scripted moves),
with the head looking around and the jaw moving when it talks. Here it drives the MuJoCo sim
(../microduck_port/harness.MicroduckSim); on the robot the same plan logic would sit on top of
Microduck's daemon (it runs the same ONNX files with the same 61-D observation at 50 Hz).

    python duckbody/hybrid/runtime_hybrid.py --script "say hello; walk; run; squat; stand; sit; swag; stand; sleep; wake; lie down; get up; hop; roll"

Brains (Microduck's, unchanged, from ../microduck_port/policies):
    velstand        stand, walk, run, turn, get up from any fall; command = [vx, vy, wz, head(4), body(6)]
    alpha_sitstand  sit (twist.vx = 1) and stand up from sitting (0); head steerable while seated
    roulade         forward roll (all-zero command), given a 2 s slot, then velstand
    ball_kick_left / ball_kick_right   kick a ball just in front of that foot (blind, all-zero command, 0.5 s
                    window as the manifest / robot daemon), then velstand (soccer.py)
Scripted moves (moves_hybrid.py, parameters from search_moves.py -> moves_hybrid.json):
    squat, hop, lie down, swag sit
Personality: look(yaw, pitch) / idle look-around (head command into the brain, or head targets for scripts),
say(text) or say_audio(samples, sr) -> jaw (the mouth servo is not a policy joint: it is driven directly).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "microduck_port"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from harness import DEFAULT_POSE, MicroduckSim, Policy  # noqa: E402
from mouth import MAX_OPEN, MouthSync  # noqa: E402
import moves_hybrid as mv  # noqa: E402

# The body: scene_hybrid_allcollisions<BODY>.xml (every part collides; antennas drawn only). HYBRID_BODY=<suffix>
# picks another build_hybrid.py variant for every script that imports this (tests, games, match, page data).
BODY = os.environ.get("HYBRID_BODY", "_noantenna")
SCENE = HERE / f"scene_hybrid_allcollisions{BODY}.xml"
ROBOT = HERE / f"hybrid_allcollisions{BODY}.xml"
MOVES = HERE / "moves_hybrid.json"
HZ, DT = 50, 0.02

WALK_VX = 0.4          # velstand's trained range is +-0.4; sim speed 0.17 m/s with heading hold (Pollen: 0.17)
RUN_VX = 1.35         # beyond its 0.4 training range; 10/10 on randomized bodies at ~0.49 m/s (1.5: 8/10)
RUN_RAMP_S = 1.0
HEADING_K = 1.5        # wz = -K * (yaw integrated from the IMU gyro): velstand drifts +0.1-0.2 rad/s otherwise
WZ_MAX = 0.5
SIT_TO_STAND_S = 2.5   # sitstand with flag 0, then velstand takes over
ROLL_SLOT_S = 2.0      # roulade's slot (manifest says 1 s; the longer hybrid head needs 2 s, README)
ROLL_DONE_DEG = 300    # ...but hand back to velstand as soon as it has rolled this far and is upright (else it rolls on)
HOP_LAND_S = 0.35      # landing crouch held after a hop's release (the search used 0.6)
SQUAT = dict(depth=0.05, kp=3.0, kd=0.1, kh=-1.0)   # trunk down ~5 cm; IMU balance: ankles kp/kd, hips -kh*pitch (10/10)
LYING_COS = 0.6        # trunk "up" axis z below this (tilt > 53 deg) = lying / fallen
HEAD_RATE = 2.0        # rad/s, head command slew
SLEEP_HEAD = (-0.9, 0.5, 0.0, 0.0)   # neck_pitch, head_pitch, head_yaw, head_roll deltas: head drooped
SWAG_GUARD_DEG = 6.0   # swag: leaning back further than this fades the leg offsets out (the sit brain re-balances)
SWAG_HEAD = (0.15, -0.25, 0.45, 0.25)  # chin up, looking aside, head cocked (search_moves.SWAG_HEAD)
DRIVE_ACC = (1.5, 1.0, 3.0)   # drive: twist slew limits per second (vx, vy, wz); the caller steers every tick
# "course" mode (hurdle game, course.py): velstand speed command per obstacle, from course.py --probe (10 bodies per
# obstacle and speed; probe_course_hybrid.json): slow walks stall on ramps past 10 deg and trip on 1.5 cm bars, a run
# trips on steps and rough ground. Plus a stumble reflex: tilt past STUMBLE_DEG -> stand still STUMBLE_HOLD_S.
COURSE_SPEED = dict(start=1.35, steps=0.8, ramps=1.0, bars=0.8, rough=0.7, narrow=0.6, hole=0.6, shoves=0.8, finish=1.35)
COURSE_TURN_SLOW = ((0.3, 0.6), (0.7, 0.0))    # heading error > 0.3 rad -> at most 0.6; > 0.7 rad: turn on the spot
STUMBLE_DEG, STUMBLE_HOLD_S = 12.0, 0.6
TURN_STUCK = (1.5, 0.25, 1.0, -0.4)   # turning (|wz| >= 0.5) for 1.5 s with < 0.25 rad of IMU yaw: back up 1 s at -0.4
                                      # (a foot caught behind a bar or step edge blocks turning on the spot)
GETUP_SETTLE_S = 0.5
# Stuck while getting up (e.g. crouched with the head on the floor and one foot on a step, tilt 30-50 deg, which
# velstand holds forever with any twist command; or lying on its side, propped by the flat 14 cm head): after
# UNSTICK_AFTER_S not upright, try in turn: the lie-down keyframes (fold onto the belly where it is), a short roulade
# kick (0.8 s), a long one (1.5 s); velstand gets up after each. Twist wiggles, head swings and the sit pose did not
# free it (tested on 4 stuck states); the roulades did, but roll 0.1-0.5 m in whatever direction the duck faces.
UNSTICK_AFTER_S = 2.0
UNSTICK_SEQ = (("lie_down", "done"), ("roulade", 0.8), ("roulade", 1.5))
KICK_S = 0.5           # manifest duration_s of ball_kick_left / ball_kick_right (cheatsheet: kick_left 0.5 s)   # drive after a fall: upright (tilt < 25 deg) this long before walking on

# command -> plan: list of (brain, end) where end = seconds, None (stay) or a condition name
PLANS = {
    "stand": [("velstand", None)],
    "idle": [("velstand", None)],
    "walk": [("velstand", None)],
    "run": [("velstand", None)],
    "turn": [("velstand", None)],
    "drive": [("velstand", None)],   # external steering every tick: course.py (hurdles), follow.py (ball)
    "squat": [("velstand", 0.5), ("squat_down", 1.0), ("squat_hold", None)],    # settle still, then squat
    "squat up": [("squat_up", 0.8), ("velstand", None)],
    "settle": [("velstand", 0.8), ("velstand", None)],      # stop walking before a move that starts from standing
    "sit": [("sit", None)],
    "sit up": [("sit_stand", SIT_TO_STAND_S), ("velstand", None)],
    "swag": [("sit", 2.5), ("swag_in", "done"), ("swag_hold", None)],
    "swag up": [("swag_out", "done"), ("sit_stand", SIT_TO_STAND_S), ("velstand", None)],
    "sleep": [("sit", None)],
    "lie down": [("velstand", 0.5), ("lie_squat", 1.2), ("lie_down", "done"), ("lie_hold", None)],
    "get up": [("velstand", None)],
    "hop": [("velstand", 0.4), ("hop", "done"), ("velstand", None)],
    # hop after hop: each next hop crouches on from the last landing crouch (~1 hop per second)
    "jump": [("velstand", 0.4), ("hop", "done"), ("hop_chain", "done"), ("repeat", None)],
    # roulade until it has turned ~1 full roll and is upright again (at most ROLL_SLOT_S), then velstand
    "roll": [("velstand", 0.6), ("roulade", "rolled"), ("velstand", None)],
    # kicks start from wherever the caller stopped (soccer.py stands still and aligns first)
    "kick left": [("kick_left", KICK_S), ("velstand", None)],
    "kick right": [("kick_right", KICK_S), ("velstand", None)],
}
ALIASES = {"wake": "wake", "wake up": "wake", "sit down": "sit", "lie": "lie down", "lay down": "lie down",
           "getup": "get up", "get_up": "get up", "stand up": "stand", "forward roll": "roll", "roulade": "roll",
           "swag sit": "swag", "relax": "swag", "hop once": "hop", "jump around": "jump", "go to sleep": "sleep",
           "stop": "stand", "look around": "idle", "talk": "idle",
           "kick": "kick right", "kick_left": "kick left", "kick_right": "kick right"}
ONNX = {"velstand": "velstand", "sit": "alpha_sitstand", "sit_stand": "alpha_sitstand", "roulade": "roulade",
        "kick_left": "ball_kick_left", "kick_right": "ball_kick_right"}
SEATED_SKILLS = {"sit", "swag_in", "swag_hold", "swag_out"}


def load_moves(path=MOVES):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


class HybridRuntime:
    def __init__(self, sim=None, seed=0, randomize=False, moves=None, log=print):
        self.sim = sim or MicroduckSim(str(SCENE), seed=seed, randomize=randomize)
        self.log = log
        self.policies = {k: Policy(v) for k, v in {"velstand": "velstand", "alpha_sitstand": "alpha_sitstand",
                                                    "roulade": "roulade"}.items()}
        self.moves = load_moves() if moves is None else moves
        self.mouth_id = self.sim.m.actuator("mouth").id
        self.steps = 0
        self.mode, self.skill, self.plan, self.plan_all, self.plan_t = "stand", None, [], [], 0
        self.vel = np.zeros(3)          # requested vx, vy, wz
        self.yaw_est = 0.0              # IMU-integrated heading, for heading hold while walking / running
        self.head_cmd = np.zeros(4)     # head command actually sent (slewed)
        self.head_goal = np.zeros(4)
        self.idle = False
        self.idle_next = 0
        self.rng = np.random.default_rng(seed)
        self.sync, self.audio, self.jaw = None, np.zeros(0), 0.0
        self.syllables = []             # (start tick, end tick, amplitude) from say(text)
        self.events = []                # log of (t, message)
        self.todo = []
        self.last_step = ("velstand", None)
        self.roll_rot = 0.0
        self.on_substep = None          # optional callback(sim) every physics step (tests: contacts at 200 Hz)
        self.vel_goal = np.zeros(3)     # drive: twist the caller asks for (self.vel follows it, slewed)
        self.falls, self.up_ticks, self.down_ticks, self.unsticks, self.unstick_k = 0, 0, 0, 0, 0
        self.stumble_until, self.stumbles = 0, 0
        self.turn_hist, self.backup_until, self.backups = [], 0, 0

    # ------------------------------------------------------------------ setup
    def reset(self, kind="standing", **kw):
        self.sim.place(kind, **kw)
        self.steps, self.yaw_est = 0, 0.0
        self.head_cmd[:] = 0
        self.head_goal[:] = 0
        self.sync, self.audio, self.jaw, self.syllables = None, np.zeros(0), 0.0, []
        self.todo = []
        self.mode, self.plan = "stand", [("velstand", None)]
        self.plan_all = list(self.plan)
        self.vel[:] = 0
        self.vel_goal[:] = 0
        self.falls, self.up_ticks, self.down_ticks, self.unsticks, self.unstick_k = 0, 0, 0, 0, 0
        self.idle = False
        self._start("velstand")
        if kind != "standing":
            self.command("get up")

    # ------------------------------------------------------------------ inputs
    def command(self, name, vx=None, vy=0.0, wz=0.0):
        """Switch to a new skill. Picks the right transition from where the duck is (sitting, squatting, lying)."""
        name = ALIASES.get(name.lower().strip(), name.lower().strip())
        prev = self.mode
        if name == "wake":
            name = "stand"
        seated = self.skill in SEATED_SKILLS
        if name in ("stand", "walk", "run", "turn", "drive", "idle", "hop", "jump", "roll", "squat", "lie down", "get up",
                    "kick left", "kick right"):
            if self.lying() and name not in ("lie down",):
                name = "get up"                                  # velstand gets up from any fall
            elif prev == "swag" and seated:
                return self._chain("swag up", then=name)
            elif seated:
                return self._chain("sit up", then=name)
            elif self.skill in ("squat_down", "squat_hold") and name != "squat":
                return self._chain("squat up", then=name)
        if prev in ("walk", "run", "turn", "drive") and name in ("squat", "sit", "sleep", "swag", "lie down", "hop", "jump", "roll"):
            self.vel[:] = 0
            return self._chain("settle", then=name)
        if name in ("sit", "sleep", "swag") and self.skill in ("squat_down", "squat_hold"):
            return self._chain("squat up", then=name)
        if name == "swag" and prev in ("sit", "sleep") and seated:
            self.plan = [p for p in PLANS["swag"] if p[0] != "sit"]
            self.plan_all, self.mode = list(self.plan), "swag"
            return self._next()
        if name in ("sit", "sleep") and prev in ("sit", "sleep") and seated:
            self.mode = name
            self._set_mode_extras(name)
            return self._say_log(f"{name}: staying seated")
        if name not in PLANS:
            return self._say_log(f"unknown command {name!r}")
        self.mode = name
        self._set_mode_extras(name, vx, vy, wz)
        self.plan, self.plan_all = list(PLANS[name]), list(PLANS[name])
        return self._next()

    def _chain(self, first, then):
        self.mode = first
        self.plan = list(PLANS[first])
        self.plan[-1] = ("then:" + then, None)
        self.plan_all = list(self.plan)
        return self._next()

    def _set_mode_extras(self, name, vx=None, vy=0.0, wz=0.0):
        self.idle = name in ("idle",)
        self.vel[:] = 0
        if name == "walk":
            self.vel[:] = [WALK_VX if vx is None else vx, vy, wz]
        elif name == "run":
            self.vel[:] = [RUN_VX if vx is None else vx, 0.0, wz]
        elif name == "turn":
            self.vel[:] = [0.0, 0.0, wz or 0.6]
        elif name == "drive":
            self.vel_goal[:] = [vx or 0.0, vy, wz]
        self.yaw_est = 0.0
        if name == "sleep":
            self.head_goal[:] = SLEEP_HEAD
        elif name == "swag":
            self.head_goal[:] = SWAG_HEAD
        elif name not in ("idle",):
            self.head_goal[:] = 0

    def drive(self, vx, vy=0.0, wz=0.0):
        """External steering (hurdle course, ball following): velstand with this twist, slewed by DRIVE_ACC.
        Call it every tick. After a fall the duck gets up first and walks on once it has been upright
        GETUP_SETTLE_S (the fall costs time, not the run)."""
        self.vel_goal[:] = [vx, vy, wz]
        if self.mode == "drive" and self.skill == "velstand":
            return
        if self.mode == "get up" and (self.lying() or self.up_ticks < GETUP_SETTLE_S * HZ):
            return
        if self.mode in ("hop", "jump", "roll", "lie down", "squat", "sit", "swag", "sleep", "kick left", "kick right") and \
                (len(self.plan) > 1 or self.skill != "velstand" or self.plan_t < GETUP_SETTLE_S * HZ):
            return              # let a one-off move (e.g. a celebration hop) finish and settle, then drive again
        self.command("drive", vx, vy, wz)

    def course_twist(self, stages, heading_err=0.0, wz=0.0, recover=True):
        """Course mode: (vx, wz) for the obstacles under / just ahead of the duck (the slowest of `stages`), slower
        while turning; backs up when a turn is blocked (IMU yaw does not move); the stumble reflex if `recover`."""
        v = self.course_vx(stages, heading_err, recover)
        n = int(TURN_STUCK[0] * HZ)
        self.turn_hist = (self.turn_hist + [(abs(wz) >= 0.5 and self.mode == "drive", self.yaw_est)])[-n:]
        if len(self.turn_hist) == n and all(t for t, _ in self.turn_hist) and \
                abs(self.turn_hist[-1][1] - self.turn_hist[0][1]) < TURN_STUCK[1]:
            self.backup_until, self.turn_hist = self.steps + int(TURN_STUCK[2] * HZ), []
            self.backups += 1
        if self.steps < self.backup_until:
            return TURN_STUCK[3], 0.0
        return v, wz

    def course_vx(self, stages, heading_err=0.0, recover=True):
        v = min(COURSE_SPEED.get(s, WALK_VX) for s in stages)
        for err, vmax in COURSE_TURN_SLOW:
            if abs(heading_err) > err:
                v = min(v, vmax)
        if recover and self.mode == "drive":
            if self.sim.tilt_deg() > STUMBLE_DEG and self.steps >= self.stumble_until:
                self.stumbles += 1
                self.stumble_until = self.steps + int(STUMBLE_HOLD_S * HZ)
            if self.steps < self.stumble_until:
                v = 0.0
        return v

    def look(self, yaw=0.0, pitch=0.0, neck=None):
        """Point the head: yaw + left, pitch + down (rad). Kept inside what the brains were trained with.
        With `neck` (+ = up, - = forward / down) the pitch may go to 1.1 rad (velstand's trained head range is
        +-1.1 on neck and head pitch): neck -0.6 and pitch 1.1 look at the feet (soccer.py's foot view)."""
        if neck is None:
            self.head_goal[:] = [0.0, float(np.clip(pitch, -0.6, 0.6)), float(np.clip(yaw, -1.2, 1.2)), 0.0]
        else:
            self.head_goal[:] = [float(np.clip(neck, -1.1, 1.1)), float(np.clip(pitch, -1.1, 1.1)),
                                 float(np.clip(yaw, -1.2, 1.2)), 0.0]

    def say(self, text):
        """Talk: syllables of the text drive the jaw (on the robot the TTS audio goes to say_audio instead)."""
        t = self.steps + 2
        for word in re.findall(r"[a-zA-Z']+|[.,!?]", text):
            if word in ".,!?":
                t += 12 if word != "," else 6
                continue
            n = max(1, len(re.findall(r"[aeiouy]+", word.lower())))
            for _ in range(n):
                d = int(self.rng.integers(7, 11))
                self.syllables.append((t, t + d, float(self.rng.uniform(0.55, 1.0))))
                t += d + 1
            t += 4
        self._say_log(f"says {text!r}")
        return t - self.steps

    def say_audio(self, samples, sample_rate):
        if self.sync is None or self.sync.sr != sample_rate:
            self.sync = MouthSync(sample_rate)
        self.audio = np.concatenate([self.audio, np.asarray(samples, float).ravel()])

    @property
    def speaking(self):
        return len(self.audio) > 0 or any(e > self.steps for _, e, _ in self.syllables)

    def after(self, seconds, fn):
        self.todo.append((self.steps + int(round(seconds * HZ)), fn))
        self.todo.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------ state
    def up_z(self):
        return -float(self.sim.gravity()[2])

    def lying(self):
        return self.up_z() < LYING_COS

    def head_offsets(self):
        return self.head_cmd.astype(np.float32)

    def status(self):
        return dict(t=round(self.steps * DT, 2), mode=self.mode, skill=self.skill, vel=self.vel.tolist(),
                    head=np.round(self.head_cmd, 3).tolist(), jaw=round(self.jaw, 3), speaking=self.speaking,
                    tilt=round(self.sim.tilt_deg(), 1), z=round(self.sim.trunk_z(), 3))

    # ------------------------------------------------------------------ brains
    def _brain(self, skill):
        mvs = self.moves
        if skill in ("velstand", "roulade"):
            return self.policies[skill]
        if skill in ("kick_left", "kick_right"):
            if ONNX[skill] not in self.policies:          # loaded on first use
                self.policies[ONNX[skill]] = Policy(ONNX[skill])
            return self.policies[ONNX[skill]]
        if skill in ("sit", "sit_stand"):
            return self.policies["alpha_sitstand"]
        sq = dict(SQUAT, **mvs.get("squat", {}))
        gains = dict(kp=sq["kp"], kd=sq["kd"], kh=sq["kh"])
        if skill == "squat_down":
            return mv.Squat(sq["depth"], down=1.0, **gains)
        if skill == "squat_hold":
            return mv.Squat(sq["depth"], down=0.02, start=sq["depth"], **gains)
        if skill == "squat_up":
            return mv.Squat(sq["depth"], down=0.8, up=True, **gains)
        if skill == "hop" and "hop" in mvs:
            return mv.Hop(dict(mvs["hop"]["params"], Tland=HOP_LAND_S))
        if skill == "hop_chain" and "hop" in mvs:
            P = mvs["hop"]["params"]
            return mv.Hop(dict(P, Tland=HOP_LAND_S), start_h=P["ht"])
        if skill in ("swag_in", "swag_hold", "swag_out") and "swag" in mvs:
            sw, sp = mvs["swag"], self.policies["alpha_sitstand"]
            return mv.Residual(sp, sw["offsets"], {"swag_in": sw["ramp"], "swag_hold": 0.0, "swag_out": 1.0}[skill],
                               out=skill == "swag_out", guard_deg=SWAG_GUARD_DEG)
        if skill == "lie_squat" and "lie" in mvs:
            return mv.Squat(mvs["lie"]["params"]["squat"], down=1.0, **gains)   # squat's gains (the search had kh 0)
        for name in ("lie_down", "lie_hold"):
            if skill == name and name in mvs:
                k = mvs[name]
                return mv.Keyframes(k["keys"], k["durations"], k.get("bal"), head=k.get("head", True))
        return None

    def _next(self):
        skill, end = self.plan[0]
        if skill.startswith("then:"):
            return self.command(skill[5:])
        if skill == "again":
            self.plan = list(self.plan_all)
            skill, end = self.plan[0]
        if skill == "repeat":                      # do the previous step again, forever
            self.plan.insert(0, self.last_step)
            skill, end = self.last_step
        self.last_step = (skill, end)
        return self._start(skill, end)

    def _start(self, skill, end=None):
        brain = self._brain(skill)
        if brain is None:
            self._say_log(f"{self.mode}: no {skill} move yet, standing instead")
            skill, brain, self.plan = "velstand", self.policies["velstand"], [("velstand", None)]
        if isinstance(brain, mv.Script):
            brain.begin(self)
        if (skill in ONNX) != (self.skill in ONNX) or ONNX.get(skill) != ONNX.get(self.skill):
            pass   # switching brains keeps last_action, as Microduck's daemon does
        self.skill, self.cur, self.plan_t = skill, brain, 0
        timing = "" if end is None else (f" for {end:g} s" if isinstance(end, (int, float)) else f" until {end}")
        return self._say_log(f"{self.mode}: {skill}{timing}")

    def _step_done(self, end):
        if end is None:
            return False
        if end == "rolled":
            return (abs(self.roll_rot) >= math.radians(ROLL_DONE_DEG) and self.up_z() > 0.8) or self.plan_t * DT >= ROLL_SLOT_S
        if end == "done":
            if hasattr(self.cur, "done"):
                return self.cur.done() or self.plan_t * DT >= self.cur.duration
            return self.plan_t * DT >= getattr(self.cur, "duration", 0.0)
        return self.plan_t * DT >= end

    def _command_vector(self):
        c = np.zeros(13, np.float32)
        if self.skill == "velstand":
            v = self.vel.copy()
            if self.mode == "run":
                v[0] *= min(1.0, self.plan_t * DT / RUN_RAMP_S)
            if self.mode == "drive":
                self.vel += np.clip(self.vel_goal - self.vel, -np.array(DRIVE_ACC) * DT, np.array(DRIVE_ACC) * DT)
                v = self.vel.copy()
            if self.mode in ("walk", "run") and abs(self.vel[2]) < 1e-6:
                v[2] = float(np.clip(-HEADING_K * self.yaw_est, -WZ_MAX, WZ_MAX))
            c[:3] = v
            c[3:7] = self.head_cmd
        elif self.skill in ("sit", "swag_in", "swag_hold", "swag_out"):
            c[0] = 1.0
            c[3:7] = self.head_cmd
        elif self.skill == "sit_stand":
            c[0] = 0.0
            c[3:7] = self.head_cmd
        return c          # roulade and scripts: all zero (roulade was trained with an all-zero command)

    def _personality(self):
        # idle: pick a new gaze every 1-2.5 s
        if self.idle and self.steps >= self.idle_next:
            self.head_goal[:] = [self.rng.uniform(-0.15, 0.15), self.rng.uniform(-0.35, 0.3), self.rng.uniform(-1.0, 1.0),
                                 self.rng.uniform(-0.2, 0.2)]
            self.idle_next = self.steps + int(self.rng.uniform(1.0, 2.5) * HZ)
        goal = self.head_goal.copy()
        if self.mode == "sleep" and self.skill == "sit":
            goal[0] += 0.08 * math.sin(2 * math.pi * 0.25 * self.steps * DT)      # slow breathing nod
        quiet = getattr(self.cur, "quiet", False) or self.skill in ("roulade", "sit_stand", "squat_up") \
            or self.mode in ("get up", "roll", "hop", "jump", "lie down", "sit up", "swag up", "squat up", "kick left",
                             "kick right") or self.lying()
        if quiet or (self.mode == "swag" and self.skill == "sit"):     # swag head only once seated
            goal[:] = 0
        step = HEAD_RATE * DT
        self.head_cmd += np.clip(goal - self.head_cmd, -step, step)
        # jaw
        target = 0.0
        if len(self.audio):
            hop = self.sync.hop
            chunk, self.audio = self.audio[:hop], self.audio[hop:]
            target = self.sync.feed(chunk)
        else:
            for s, e, a in self.syllables:
                if s <= self.steps < e:
                    target = MAX_OPEN * a * math.sin(math.pi * (self.steps - s) / (e - s))
            self.syllables = [x for x in self.syllables if x[1] > self.steps]
        if self.mode == "swag" and not self.speaking:
            target = 0.12                                         # relaxed, beak a little open
        self.jaw = float(np.clip(target, 0, MAX_OPEN))
        self.sim.d.ctrl[self.mouth_id] = self.jaw

    # ------------------------------------------------------------------ loop
    def tick(self):
        while self.todo and self.todo[0][0] <= self.steps:
            self.todo.pop(0)[1]()
        self._personality()
        self.sim.command[:] = self._command_vector()
        self.sim.step(self.cur, on_substep=self.on_substep)
        self.yaw_est += float(self.sim.d.sensordata[self.sim.gyro_adr][2]) * DT
        self.roll_rot = self.roll_rot + float(self.sim.d.sensordata[self.sim.gyro_adr][1]) * DT if self.skill == "roulade" else 0.0
        self.steps += 1
        self.plan_t += 1
        up = self.sim.tilt_deg() < 25
        self.up_ticks = self.up_ticks + 1 if up else 0
        self.down_ticks = 0 if up else self.down_ticks + 1
        if self.up_ticks > HZ:
            self.unstick_k = 0
        # fell over while doing something upright: velstand gets up (it is trained for any fall)
        if self.lying() and self.mode not in ("lie down", "roll", "get up") and self.skill not in SEATED_SKILLS:
            self._say_log("fell over: getting up")
            self.falls += 1
            self.vel[:] = 0              # get up with a zero twist (a walk command while getting up props it on edges)
            self.vel_goal[:] = 0
            self.mode, self.plan, self.plan_all = "get up", [("velstand", None)], [("velstand", None)]
            self._start("velstand")
            return
        if self.mode == "get up" and self.skill == "velstand" and min(self.down_ticks, self.plan_t) > UNSTICK_AFTER_S * HZ:
            step = UNSTICK_SEQ[self.unstick_k % len(UNSTICK_SEQ)]
            self._say_log(f"stuck getting up: {step[0]} {step[1]}, then get up")
            self.unsticks += 1
            self.unstick_k += 1
            self.down_ticks = 0
            self.plan = [step, ("velstand", None)]
            self._start(*step)
            return
        if self.plan and self._step_done(self.plan[0][1]):
            self.plan.pop(0)
            if self.plan:
                self._next()

    def run(self, seconds, every=None):
        for _ in range(int(round(seconds * HZ))):
            self.tick()
            if every:
                every(self)

    def _say_log(self, msg):
        self.events.append((round(self.steps * DT, 2), msg))
        if self.log:
            self.log(f"[{self.steps * DT:6.2f} s] {msg}")
        return msg


def run_script(rt, script, hold=4.0, every=None):
    """'say hello; walk; sit; ...' : one clause every `hold` s ('say ...' also talks over the next clause)."""
    for clause in [c.strip() for c in script.split(";") if c.strip()]:
        if clause.startswith("say "):
            rt.say(clause[4:])
            continue
        rt.command(clause)
        rt.run(hold, every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="say hello there; idle; walk; run; stand; squat; stand; sit; swag; stand; "
                                        "sleep; wake; lie down; get up; hop; jump; roll")
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rt = HybridRuntime(seed=a.seed, randomize=True)
    rt.reset("standing")
    run_script(rt, a.script, a.hold, every=None)
    print(rt.status())


if __name__ == "__main__":
    main()
