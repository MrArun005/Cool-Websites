"""One Bolt, many brains: a skill manager that switches between them like a dog does.

    from skills import Bolt
    bolt = Bolt(seed=1)
    bolt.start("walk")                    # reference-state start in the walk clip
    bolt.do("walk", 2); bolt.do("trot", 3); bolt.do("jump", height=0.3); bolt.do("trot", 2)
    bolt.do("sit", 4); bolt.do("lie", 5); bolt.do("stand", 3)
    bolt.shove(); bolt.do("walk", 3)      # a fall triggers "recover" by itself

Skills and the brain behind each (the same robot, bolt.xml, and motor model, bolt_env.physics, for all):
    walk trot pace canter turn_left turn_right sit lie
                  the dog-motion brain (imitate_env, run 18b's newest checkpoint): each skill is a reference
                  clip. A switch keeps the physics and the brain and only changes the clip, which starts at
                  the moment whose joint pose (and joint speeds, a little) is closest to Bolt's own, so the
                  plan the motors follow (q_ref + correction) does not jump.
    jump          the real-hurdle jump brain (jump_env, run 16g 27M): the hurdle is placed 1.5-2 m ahead on
                  the current heading, the current gait carries Bolt to 1.1 m before the bar (JUMP_FROM), the
                  jump brain takes over until it has landed (on its feet half a second after clearing,
                  JumpEnv's "landed"), then a short canter (it lands at a gallop) and the previous gait.
    recover       self-righting (getup_env goal="selfright", run 17e) until belly-down and still, then the
                  posture brain's "lie" to tuck the legs and its "stand" (learnwatch.recovery, the chain the
                  games use).
    stand         the posture brain's "stand" (dog stance).
A fall detector runs every step (tipped past 60 degrees, or the body on the floor for 0.1 s in a skill
that should be on its feet) and starts "recover" by itself (auto_recover=True).

How the brains share one world. There is ONE MjModel/MjData: the real-hurdle world of JumpEnv(real=True)
(hurdle.py: stands and bar after the floor/robot, robot qpos 0-18 and qvel 0-17 unchanged; the hurdle is
parked 100 m away until a jump). Every brain keeps its own env object, but each env's model and data are
re-pointed at that world (_attach), together with everything the env had derived from its own model
(geom/body ids, joint ranges, nominal mass/friction/gains). In the hurdle world geom and body ids are not
the same as in the plain robot model (the stands come before the floor), so the ids must come from the
world, not from the env's own model. With the data shared there is nothing to copy before a step and
nothing can drift out of sync: each brain's observation is its env class's own _obs() of the live state,
and its action goes through its env class's own step() (so the motor mapping is exactly the one it
learned: residual on the clip for imitation, full range around the dog stance for jump and self-right,
centred on the stance for the posture brain). Only the bookkeeping a brain sees besides the body (clip and
clip clock, the jump clock and hurdle, last action) is set at each hand-over. Copying qpos/qvel/act into
private worlds instead would also need the contacts' warm-start, the bar and the randomized model
parameters copied every step; sharing avoids that whole class of bug.
Physics randomization (motor strength, torso mass, floor friction) is drawn once per start(), as the envs
do at reset, and sensor noise is on when randomize=True.

Every run is recorded at 25 Hz: Bolt's body pose (qpos 0-18), the bar's pose, and the skill in charge, in
the replay shape cloudwatch.py writes (replay(): flat qpos list, plus "<k>_props" with the hurdle).
"""
from pathlib import Path

import mujoco
import numpy as np

from bolt_env import LIE_POSE, BoltEnv, apply_shove
from evaluate import Policy
from getup_env import GetUpEnv, GetUpPolicy
from imitate_env import ImitateEnv, ImitatePolicy, closest_moment   # noqa: F401  (closest_moment: the one hand-over rule, shared with the training env)
from jump_env import JumpEnv, JumpPolicy

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
JUMP = MODELS / "bolt-dog-jump-a1-16g-27M"
SELFRIGHT = MODELS / "bolt-dog-selfright-17e-16.5M"
POSTURE = MODELS / "bolt-dog-posture-getup-3M"
IMITATE_RUN = HERE / "runs" / "imitate18b" / "checkpoints"
CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie"]
GAITS = {"walk", "trot", "pace", "canter", "turn_left", "turn_right"}
ON_FEET = GAITS | {"stand", "jump"}   # skills where the body touching the floor is a fall
JUMP_FROM = 1.1    # m: the jump brain takes over with the bar this far ahead of the body. It practised from a
                   # standstill 0.45-0.65 m away; from a 1.5-1.9 m/s trot it needs a longer run-in (measured,
                   # clean jumps over 20-40 cm bars: 0.6 m 0 of 9, 1.0 m 2/8, 1.1 and 1.25 m 11/18 each, 1.4 m 1/8)
BRIDGE = 0.6       # s of canter after landing: the jump brain lands at ~3.5 m/s, faster than any other clip
FALL_UP = 0.5      # body z-axis up-component below this (tipped past 60 degrees): a fall


def newest_imitate(folder=IMITATE_RUN):
    """(folder, zip, stats) of the newest checkpoint pair of the dog-motion run."""
    n = max(int(p.stem.split("_")[1]) for p in Path(folder).glob("policy_*_steps.zip")
            if (Path(folder) / f"policy_vecnormalize_{p.stem.split('_')[1]}_steps.pkl").exists())
    return Path(folder), f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"


def _attach(env, world):
    """Point env at the world's model and data, with every id and range it derived from its own model."""
    env.model, env.data = world.model, world.data
    for k in ("lo", "hi", "nominal_kp", "nominal_mass", "nominal_friction", "torso", "foot_ids", "floor", "geom_leg"):
        setattr(env, k, getattr(world, k))
    if hasattr(env, "body_geoms"):   # ImitateEnv: the robot's body geoms only (not the hurdle's)
        env.body_geoms = set(robot_body_geoms(world))


def robot_body_geoms(env):
    m = env.model
    root = m.body_rootid[env.torso]
    return [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and env.geom_leg[g] < 0]


def quat(xmat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, dtype=float))
    return q


class Bolt:
    def __init__(self, seed=0, randomize=True, auto_recover=True, imitate=None, verbose=False):
        self.randomize, self.auto_recover, self.verbose = randomize, auto_recover, verbose
        # the world: the real-hurdle world, owned by the jump brain's env
        self.world = JumpEnv(randomize=randomize, real=True, seed=seed)
        self.model, self.data = self.world.model, self.world.data
        self.imitate_from = imitate or newest_imitate()
        if isinstance(self.imitate_from, (str, Path)) and str(self.imitate_from).endswith(".pkl"):   # a GPU (mjx_imitate) brain
            from mjx_imitate import MjxImitatePolicy
            self.dog = MjxImitatePolicy(self.imitate_from)
        else:
            self.dog = ImitatePolicy(*self.imitate_from)
        self.im = ImitateEnv(randomize=randomize, clips=CLIPS, slots=self.dog.slots)
        self.jumper = JumpPolicy(JUMP)
        self.righter = GetUpPolicy(SELFRIGHT)
        self.sr = GetUpEnv(randomize=randomize, goal="selfright", max_steps=10 ** 6)
        self.posture = Policy(POSTURE)
        for env in (self.im, self.sr):
            _attach(env, self.world)
        self.body_geoms = set(robot_body_geoms(self.world))
        self.hurdle_body = self.model.body("hurdle0").id
        self.dt = self.world.dt
        self.skill, self.gait = None, "walk"
        self.reset_record()

    # ---------- setup ----------
    def start(self, clip="walk", tau=None, seed=None):
        """A new run: physics randomized as at any env reset, Bolt in the clip's pose at moment tau (random
        if None), moving as the dog was (ImitateEnv's reference state initialization), hurdle parked."""
        im = self.im
        rng = np.random.default_rng(seed)
        c = im.clips[clip]
        im.exam_start = (clip, float(rng.uniform(0, c.length)) if tau is None and c.loop else (tau or 0.0))
        obs, _ = im.reset(seed=seed)   # mj_resetData on the shared world parks hurdle and bar (their xml pos)
        for env in (self.world, self.sr):
            env.rng = im.rng           # one noise source per run
        self.world.hurdle_at, self.world.hurdle_h = np.array([100.0, 0.0]), 0.0
        self.world.bar_rest = self.data.qpos[self.world.bar_q:self.world.bar_q + 3].copy()
        self.skill, self.gait, self.obs = clip, clip if clip in GAITS else self.gait, obs
        self.reset_record()
        self.t, self.floor_steps = 0, 0

    def reset_record(self):
        self.frames, self.bars, self.phases, self.trace, self.events = [], [], [], [], []
        self.static = None
        self.t = 0

    # ---------- per step ----------
    def _after_step(self, skill):
        """Record, trace and check for a fall. Returns True when Bolt has fallen."""
        d = self.data
        R = d.xmat[self.world.torso].reshape(3, 3)
        if self.t % 2 == 0:   # 25 Hz
            self.frames.append(d.qpos[:19].copy())
            self.bars.append(d.qpos[self.world.bar_q:self.world.bar_q + 7].copy())
            self.phases.append(skill)
        # (step, skill, height, uprightness, forward speed, roll, nose-up pitch) every step, for the tests
        self.trace.append((self.t, skill, float(d.qpos[2]), float(R[2, 2]), float((R.T @ d.qvel[0:3])[0]),
                           float(np.arcsin(np.clip(R[2, 1], -1, 1))), float(np.arcsin(np.clip(R[2, 0], -1, 1)))))
        self.t += 1
        on_floor = any((c.geom1 == self.world.floor and c.geom2 in self.body_geoms) or
                       (c.geom2 == self.world.floor and c.geom1 in self.body_geoms) for c in d.contact[:d.ncon])
        self.floor_steps = self.floor_steps + 1 if on_floor and skill in ON_FEET else 0
        return R[2, 2] < FALL_UP or self.floor_steps >= 5

    def _fell(self, skill):
        self.events.append(dict(t=self.t, kind="fall", skill=skill, height=round(float(self.data.qpos[2]), 3),
                                up=round(float(self.data.xmat[self.world.torso][8]), 2)))
        if self.verbose:
            print(f"  fall at {self.t * self.dt:.2f}s during {skill}", flush=True)
        if self.auto_recover:
            self.recover()
        return False

    # ---------- skills ----------
    def do(self, skill, seconds=None, **kw):
        """Run a skill for `seconds` (jump and recover: until done). Returns False if Bolt fell (it is then
        already recovered when auto_recover is on), else True (jump: the jump's result dict)."""
        self.events.append(dict(t=self.t, kind="switch", skill=skill, prev=self.skill))
        if skill in CLIPS:
            return self._clip(skill, seconds or 2.0)
        if skill == "jump":
            return self._jump(**kw)
        if skill == "recover":
            return self.recover()
        if skill in ("stand", "posture_sit", "posture_lie"):
            return self._posture(skill.split("_")[-1], seconds or 2.0)
        raise ValueError(f"unknown skill {skill}")

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
        for _ in range(int(round(seconds / self.dt))):
            im.step(self.dog.act(im._obs()))
            if self._after_step(name):
                return self._fell(name)
        return True

    def _posture(self, cmd, seconds, until=None):
        env = self.sr   # the posture brain runs on BoltEnv's obs and step (dog stance), as learnwatch.recovery
        name = cmd if cmd == "stand" else "posture_" + cmd   # not to be mistaken for the "sit"/"lie" clips
        if self.skill != name:
            env.forced_command = cmd   # forced: BoltEnv.step never draws a new command
            env.set_command(cmd)
            env.last_action = np.zeros(12)
        self.skill = name
        for _ in range(int(round(seconds / self.dt))):
            BoltEnv.step(env, self.posture.act(BoltEnv._obs(env)))
            if self._after_step(name) and cmd == "stand":
                return self._fell(name)
            if until and until():
                break
        return True

    def place_hurdle(self, ahead, height):
        w = self.world
        w.place_real(ahead, height, 0.0, 0.0)
        m = self.model
        self.static = [[int(m.geom_type[g]), np.round(m.geom_size[g], 4).tolist(), np.round(self.data.geom_xpos[g], 4).tolist(),
                        np.round(quat(self.data.geom_xmat[g]), 4).tolist()] for g in range(m.ngeom) if m.geom_bodyid[g] == self.hurdle_body]

    def _jump(self, height=0.3, ahead=1.75, after=None, then=2.0, bridge=BRIDGE, max_seconds=4.0):
        """Place the hurdle `ahead` m on the current heading, keep the gait up to it, jump, land, and go on in
        `after` (default: the gait before the jump) for `then` s, through `bridge` s of canter (Bolt lands at a
        gallop). Returns a dict: cleared, clean (over without touching the bar, landed), landed, fell (during
        the approach or the jump), fell_after (in the bridge and the `then` s after)."""
        w, gait = self.world, self.skill if self.skill in GAITS else self.gait
        after = after or gait
        self.place_hurdle(ahead, height)
        out = dict(height=height, cleared=False, clean=False, landed=False, fell=False, fell_after=False, dropped=False, touched=False)
        # approach in the current gait (a stand-still Bolt walks up to it)
        approach = gait if self.skill in GAITS else "walk"
        steps = 0
        while w.hurdle_ahead() > JUMP_FROM and steps < int(max_seconds / self.dt):
            if not self._clip(approach, self.dt):
                out["fell"] = True
                return out
            steps += 1
            if w.hurdle_ahead() < -0.5:   # walked past it sideways
                break
        # the jump brain takes over: its own clock and bookkeeping, as at JumpEnv.reset
        self.events.append(dict(t=self.t, kind="switch", skill="jump", prev=self.skill))
        w.t, w.last_action = 0, np.zeros(12)
        w.touched = w.cleared = w.landed = w.dropped = False
        w.clear_t, w.best_past = None, w.paws_past().min()
        self.skill = "jump"
        info = {}
        for _ in range(150):
            _, _, _, _, info = w.step(self.jumper.act(w._obs()))
            if self._after_step("jump"):
                out.update(fell=True, cleared=w.cleared, touched=w.touched, dropped=w.dropped)
                self._fell("jump")
                return out
            if w.landed:
                break
        out.update(cleared=bool(w.cleared), touched=bool(w.touched), dropped=bool(w.dropped), landed=bool(w.landed),
                   clean=bool(w.cleared and not w.touched and w.landed), steps=w.t)
        if not w.landed:   # never landed in 3 s (refused, or still in front): hand back anyway
            self.events.append(dict(t=self.t, kind="no_landing"))
        out["landed_at"] = self.t
        ok = self._clip("canter", bridge) if bridge else True
        if ok and then:
            self.events.append(dict(t=self.t, kind="switch", skill=after, prev=self.skill))
            ok = self._clip(after, then)
        out["fell_after"] = not ok
        return out

    def recover(self):
        """Self-right until belly-down and still for 0.3 s (at most 4 s), tuck with the posture brain's "lie"
        (until every joint is within 0.6 rad of the lying pose, at most 2 s), then its "stand" for 3 s.
        Returns True if Bolt ends standing (upright, body above 38 cm)."""
        env = self.sr
        self.events.append(dict(t=self.t, kind="recover"))
        env.t, env.last_action, env.best_up, env.stood_at, env.kind = 0, np.zeros(12), env._rotation()[2, 2], None, "fall"
        self.skill, calm = "recover", 0
        for _ in range(200):
            _, _, _, _, info = env.step(self.righter.act(env._obs()))
            self._after_step("self-righting")
            calm = calm + 1 if info["standing"] else 0
            if calm >= 15:
                break
        self._posture("lie", 2.0, until=lambda: np.abs(self.data.qpos[7:19] - LIE_POSE).max() < 0.6)
        auto, self.auto_recover = self.auto_recover, False   # no recovery inside the recovery
        self._posture("stand", 3.0)
        self.auto_recover = auto
        p = env.posture()
        ok = bool(p["up"] > 0.9 and p["height"] > 0.38)
        self.events.append(dict(t=self.t, kind="recovered" if ok else "not_recovered", height=round(float(p["height"]), 3)))
        return ok

    def shove(self, speed=2.5, spin=6.0, side=None):
        """A hard sideways shove (a kick to the flank): body velocity and roll rate changed at once."""
        R = self.data.xmat[self.world.torso].reshape(3, 3)
        s = side if side is not None else (1 if self.im.rng.random() < 0.5 else -1)
        self.data.qvel[:] = apply_shove(self.data.qvel, R, speed, spin, s)   # bolt_env: shared with the training pushes
        self.events.append(dict(t=self.t, kind="shove", side=int(s)))

    # ---------- queries ----------
    def fell_between(self, t0, t1):
        return any(e["kind"] == "fall" and t0 <= e["t"] < t1 for e in self.events)

    def replay(self, key):
        """{key: flat qpos frames, key_props: hurdle, key_phase: skill per frame} as cloudwatch.py's replays."""
        out = {key: np.round(np.concatenate(self.frames), 4).tolist(), key + "_phase": self.phases}
        if self.static is not None:
            bar = self.model.geom_size[self.world.bar_geom]
            out[key + "_props"] = dict(static=self.static, bar=[float(bar[0]), float(bar[1])],
                                       barpose=np.round(np.concatenate(self.bars), 4).tolist())
        return out
