"""The duck's onboard controller: the code the real robot runs, here driving the MuJoCo sim.

    python duckbody/runtime.py --demo                  # scripted show in the sim, per-phase report + video
    python duckbody/runtime.py --script "say hello; go to sleep; wake up; walk forward; turn left; stop"

Layers (top to bottom):

    personality  command("sit") / look(yaw) / say(audio, sr) / hear("go to sleep")
    DuckRuntime  picks the skill brain, builds the observation, runs the brain at 50 Hz, adds the
                 head (look, rate-limited) and jaw (MouthSync on the speech audio), sends servo targets
    Backend      sensors in, servo targets out, one control period at a time.
                 SimBackend = MuJoCo. A real-robot backend implements the same three methods.

The observation is built exactly as SkillEnv builds it (one layout for every skill, LESSONS §3), from
raw sensor values; each brain (models/mid-<skill>/policy.onnx from package.py) has the normalizer
built in. No action filter: the brains were trained without one, so the robot uses none either.
Switching skills is instant: same inputs, same outputs, the previous action carries over.

This file does not import MuJoCo or the training code except inside SimBackend and the demo.
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mouth import MouthSync, demo_voice  # noqa: E402

MODELS = HERE.parent / "models"
ROBOT_XML = HERE / "model" / "duck_mid.xml"
SOLES = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]

# What the personality can ask for → the skill brains to run, in order: (skill, seconds or None = stay).
PLANS = {
    "stand": [("stand", None)], "idle": [("stand", None)],
    # run = the fast walk: the walk brain (trained up to ~0.31 m/s, walk_env.py WALK_FAST) at a higher speed;
    # the separate run gait fell at once on this body
    "walk": [("walk", None)], "run": [("walk", None)],
    # sit is chained: squat first, then settle back onto belly and thighs (how it was trained)
    # this body sits stably as swag (one leg out); the symmetric sit rolled it onto its back
    # sit: squat, then MoveBrain down into the perch (deep crouch on the soles; s2s_roundtrip.py: 20/20 sat and
    # 20/20 stood back up on randomized bodies). The floor seats (swag, sit) cannot be stood up from.
    "sit": [("squat", 2.0), ("pose:perch", None)], "squat": [("squat", None)],
    "perch rise": [("pose:perch_rise", 2.3), ("stand", None)],
    # sleep in the squat: from the floor seat this body cannot stand back up (every push from the seat tips
    # it onto its back: the tail is on the floor, the weight behind the feet), from the squat it can
    "sleep": [("squat", None)],
    # wake from the squat: glide the servo targets to the home pose over 1 s, then the stand brain
    # (the stand brain alone, started in the squat, fell 5/5; with the 0.5-2 s glide first, 5/5 stood)
    "rise": [("pose:rise", 1.0), ("stand", None)],
    # saved poses: get down like sit, then glide into the pose and hold it (statically stable)
    "swag": [("squat", 2.0), ("swag_sit", 3.0), ("pose:swag", None)],
    "swag sit": [("squat", 2.0), ("swag_sit", None)],      # the trained brain that sits down into swag
    "wake": [("sit_to_stand", 3.0), ("stand", None)],
    # lie down on the belly (lie_down.py: 20/20 lying, 20/20 got up again, landing 0.2 g): sit into the perch,
    # fold forward into the face-down tuck (MoveBrain), stay there; "get up" brings it back
    "lie": [("squat", 2.0), ("pose:perch", 3.0), ("pose:lower", 2.3), ("pose:fd_tuck", None)],
    # get up from lying face down (owner's plan; facedown_rise.py: 20/20 on randomized bodies): tuck the legs
    # under, lift the body into the perch (MoveBrain), then the perch rise and the stand brain
    "get up": [("pose:fd_tuck", 1.0), ("pose:fd_lift", 3.5), ("pose:perch_rise", 2.3), ("stand", None)],
    # on its back or side: the trained roll-over brain (rollover_env.py) rolls it onto the belly in 4 s, then
    # the face-down get-up (back -> roll -> get up: 10/10 on randomized bodies)
    "roll get up": [("get_up", 4.0), ("pose:fd_tuck", 1.0), ("pose:fd_lift", 3.5), ("pose:perch_rise", 2.3), ("stand", None)],
    # Paced hops (the jump brain alone bounces at ~4 Hz). The jump brain runs until 1 tick after its
    # 2nd touchdown (a 1-2 mm prep bounce, then the visible 4-14 mm hop: from a still stand its first
    # flight is tiny), then the stand brain settles for 0.8 s: ~1 hop per 1.15 s. "hop" = once, then stand.
    # ("again" restarts the plan; "landedN" ends a step N touchdowns after it began.)
    "jump": [("jump", "landed2"), ("stand", 0.8), ("again", None)],
    "hop": [("jump", "landed2"), ("stand", None)],
    # forward roll (forward_roll.py: 18/20 over the head onto the back on randomized bodies): squat, tumble
    # forward (searched keyframes), then roll onto the belly (roll-over brain) and the face-down get-up
    "roll": [("squat", 2.0), ("pose:tumble", 3.0), ("get_up", 4.0), ("pose:fd_tuck", 1.0), ("pose:fd_lift", 3.5),
             ("pose:perch_rise", 2.3), ("stand", None)],
}
LANDED_TICKS, LANDED_TIMEOUT = 1, 2.0   # a "landed" step ends this many ticks after the touchdown (3 let the
                                        # jump brain push off again first); give up after this many s
ALIASES = {"hop once": "hop", "one hop": "hop", "single hop": "hop", "get_up": "get up", "getup": "get up", "wake up": "wake", "lie down": "lie", "sit down": "sit"}
# The command one-hot each skill brain was trained with (skills_env.TASK_COMMAND).
POSES = HERE / "model" / "poses_mid.json"


class PoseBrain:
    """Holds a saved pose: glides the servo targets there over `glide` s, then holds them."""

    def __init__(self, ctrl, rt, glide=1.5):
        self.goal = np.clip((np.asarray(ctrl)[rt.pol] - rt.home[rt.pol]) / rt.scale, -1, 1)
        self.rt, self.n, self.start = rt, max(1, int(glide * rt.hz)), None

    def begin(self):
        self.start = self.rt.last_action.copy()

    def __call__(self, obs):
        if self.start is None:
            return self.goal
        f = min(1.0, (self.rt.plan_t + 1) / self.n)
        return self.start + (self.goal - self.start) * f


class MoveBrain(PoseBrain):
    """Two-phase leg move with the soles kept flat (s2s_move.py, owner's design): lean the hips to `lean`
    first (t1 s, knees kept), then move hips and knees to `goal` (t2 s); the ankles follow the flat-sole rule
    (home rule + trunk pitch from the IMU gravity, sign -1) the whole time. Hip pitch is mirrored L/R: `lean`
    is the LEFT value, the right hip gets the mirrored offset. Sit-down into the perch and the stand-up from it
    both use it (s2s_roundtrip.py: 20/20 seated and 20/20 stood on randomized bodies)."""

    def __init__(self, goal, rt, lean, t1, t2, sign=-1):
        self.rt, self.goal, self.lean, self.t1, self.t2, self.sign = rt, np.asarray(goal, float), lean, t1, t2, sign
        n = rt.names
        self.leg = {k: [i for i, x in enumerate(n) if x.endswith(k)] for k in ("hip_pitch", "knee", "ankle")}
        h, hp0 = rt.home, self.leg["hip_pitch"][0]
        self.mirror = {i: (1.0 if j == 0 else float(np.sign(h[i] * h[hp0]) or 1.0)) for j, i in enumerate(self.leg["hip_pitch"])}
        self.t0 = self.start = None

    def begin(self):
        self.t0, self.start = self.rt.steps, self.rt.targets.copy()

    def __call__(self, obs):
        rt, h = self.rt, self.rt.home
        if self.t0 is None:
            self.begin()
        t = (rt.steps - self.t0) * rt.dt
        g = obs[:3]
        tilt = float(np.arctan2(g[0], -g[2]))
        tgt = self.goal.copy() if t >= self.t1 + self.t2 else self.start.copy()
        hp0 = self.leg["hip_pitch"][0]
        for hp, kn, an in zip(self.leg["hip_pitch"], self.leg["knee"], self.leg["ankle"]):
            h0, k0 = self.start[hp], self.start[kn]
            hl = h[hp] + self.mirror[hp] * (self.lean - h[hp0])
            if t < self.t1:
                f = t / self.t1
                tgt[hp], tgt[kn] = h0 + (hl - h0) * f, k0
            elif t < self.t1 + self.t2:
                f = (t - self.t1) / self.t2
                tgt[hp], tgt[kn] = hl + (self.goal[hp] - hl) * f, k0 + (self.goal[kn] - k0) * f
            tgt[an] = h[an] - self.mirror[hp] * (tgt[hp] - h[hp]) - (tgt[kn] - h[kn]) + self.sign * tilt
        return np.clip((tgt[rt.pol] - h[rt.pol]) / rt.scale, -1, 1)


FD_TUCK = (0.0, 2.18, -0.8)            # face-down get-up: left hip pitch, knee, ankle (right hip mirrored)
FD_LIFT = (0.3, 0.5, 2.0, -1)          # MoveBrain into the perch: lean, t1, t2, sign (facedown_rise.py)
LIE_LOWER = (0.0, 0.8, 1.0, -1)        # MoveBrain from the perch down onto the belly: lean, t1, t2, sign (lie_down.py)


class KeyframeBrain(PoseBrain):
    """Replays a searched motion: keyframes of policy actions, each reached by a linear ramp over its duration
    (forward_roll.py), starting from the action the robot is holding; then holds the last keyframe."""

    def __init__(self, keys, durations, rt):
        self.rt, self.keys, self.durs = rt, np.asarray(keys, float), np.asarray(durations, float)
        self.t0 = self.start = None

    def begin(self):
        self.t0, self.start = self.rt.steps, self.rt.last_action.copy()

    def __call__(self, obs):
        if self.t0 is None:
            self.begin()
        t, prev = (self.rt.steps - self.t0) * self.rt.dt, self.start
        for key, dur in zip(self.keys, self.durs):
            if t < dur:
                return prev + (key - prev) * max(t, self.rt.dt) / dur
            t, prev = t - dur, key
        return self.keys[-1]


OBS_COMMAND = {"swag_sit": "sit", "stand": "stand", "walk": "walk", "run": "run", "sit": "sit", "squat": "sit", "sit_to_stand": "stand",
               "lie": "lie", "get_up": "stand", "jump": "jump", "roll": "roll"}


# ---------------------------------------------------------------- sensors / servos
@dataclass
class Sensors:
    gravity: np.ndarray     # (3) gravity direction in the IMU frame (unit vector)
    gyro: np.ndarray        # (3) rad/s, IMU frame
    accel: np.ndarray       # (3) m/s², IMU frame (specific force, reads +9.81 z when level and still)
    joint_pos: np.ndarray   # (n servos) rad, in servo_names order
    joint_vel: np.ndarray   # (n servos) rad/s
    feet: np.ndarray        # (2) left, right sole switch, 0/1


class Backend:
    """What the runtime needs from a body. A real robot implements these with its IMU, servo bus,
    foot switches, speaker and a 50 Hz timer."""
    servo_names: list

    def read(self) -> Sensors:
        raise NotImplementedError

    def write(self, targets):
        """Servo position targets (rad), servo_names order."""
        raise NotImplementedError

    def advance(self):
        """Let one control period (1/50 s) pass: sleep until the next tick, or step the simulation."""
        raise NotImplementedError

    def play(self, audio, sample_rate):
        """Send speech to the speaker (the jaw follows the same samples)."""


class SimBackend(Backend):
    """The MuJoCo duck. Pass model/data to drive an existing simulation (e.g. a SkillEnv's)."""

    def __init__(self, servo_names, model_path=ROBOT_XML, model=None, data=None, ctrl_hz=50):
        import mujoco
        self.mj = mujoco
        if model is None:
            model = mujoco.MjModel.from_xml_path(str(model_path))
            data = mujoco.MjData(model)
            fresh = True
        else:
            fresh = False
        self.model, self.data, m = model, data, model
        self.servo_names = list(servo_names)
        self.act = np.array([m.actuator(n).id for n in self.servo_names])
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in self.act])
        self.dadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in self.act])
        self.imu, self.trunk, self.floor = m.site("imu").id, m.body("trunk_assembly").id, m.geom("floor").id
        self.soles = [m.geom(g).id for g in SOLES]
        self.sens = {n: (m.sensor_adr[m.sensor(n).id], m.sensor_dim[m.sensor(n).id]) for n in ("gyro", "accelerometer")}
        self.frame_skip = int(round(1 / ctrl_hz / m.opt.timestep))
        self.renderer = None
        if fresh:
            self.reset()

    def reset(self):
        """Standing in the home pose, soles on the floor."""
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        d.qpos[:] = m.key("home").qpos
        mj.mj_forward(m, d)
        d.qpos[2] -= min(self._bottom(g) for g in self.soles)
        d.ctrl[:] = m.key("home").ctrl
        mj.mj_forward(m, d)

    def _bottom(self, g):
        m, d = self.model, self.data
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()

    def _sensor(self, name):
        a, n = self.sens[name]
        return self.data.sensordata[a: a + n].copy()

    def read(self):
        d = self.data
        R = d.site_xmat[self.imu].reshape(3, 3)
        feet = np.zeros(2)
        for c in d.contact[:d.ncon]:
            for i, s in enumerate(self.soles):
                if s in (c.geom[0], c.geom[1]) and self.floor in (c.geom[0], c.geom[1]):
                    feet[i] = 1
        return Sensors(gravity=R.T @ np.array([0, 0, -1.0]), gyro=self._sensor("gyro"),
                       accel=self._sensor("accelerometer"), joint_pos=d.qpos[self.qadr].copy(),
                       joint_vel=d.qvel[self.dadr].copy(), feet=feet)

    def write(self, targets):
        self.data.ctrl[self.act] = targets

    def advance(self):
        self.mj.mj_step(self.model, self.data, nstep=self.frame_skip)

    # sim-only helpers (the real robot doesn't know these)
    def posture(self):
        R = self.data.xmat[self.trunk].reshape(3, 3)
        return dict(up=float(R[2, 2]), height=float(self.data.qpos[2]))

    def render(self, width=320, height=240):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height, width)
            self.camera = self.mj.MjvCamera()
            self.camera.type = self.mj.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.trunk
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 0.55, -15, 140
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()


# ---------------------------------------------------------------- brains
class OnnxBrain:
    def __init__(self, path):
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])
        self.path = path

    def __call__(self, obs):
        return self.session.run(None, {"obs": np.asarray(obs, np.float32)[None]})[0][0]


def load_brains(models_dir=MODELS):
    """Every models/mid-<skill>/policy.onnx with its model.json. All must share one layout."""
    brains, cfg = {}, None
    for p in sorted(Path(models_dir).glob("mid-*/policy.onnx")):
        c = json.loads((p.parent / "model.json").read_text())
        if cfg is None:
            cfg = c
        for k in ("servo_names", "policy_servos", "obs_size", "action_size", "home", "action_scale", "commands"):
            if c[k] != cfg[k]:
                raise ValueError(f"{p}: {k} differs from the other brains; every skill must share one layout")
        zip_ = p.parent / "policy.zip"
        if zip_.exists() and zip_.stat().st_mtime > p.stat().st_mtime + 1:
            print(f"warning: {zip_} is newer than policy.onnx; re-run duckbody/package.py --skill {c['skill']}")
        brains[c["skill"]] = OnnxBrain(p)
    if cfg is None:
        raise FileNotFoundError(f"no brains in {models_dir}/mid-*/policy.onnx: run duckbody/package.py --all")
    return brains, cfg


# ---------------------------------------------------------------- the runtime
class DuckRuntime:
    def __init__(self, backend=None, models_dir=MODELS, brains=None, config=None, log=print):
        if brains is None:
            brains, config = load_brains(models_dir)
        self.brains, self.cfg, self.log = dict(brains), config, log
        c = config
        self.hz = c["control_hz"]
        self.dt = 1.0 / self.hz
        self.names = list(c["servo_names"])
        self.pol = np.array([self.names.index(n) for n in c["policy_servos"]])
        self.home = np.array(c["home"])
        self.scale = np.array(c["action_scale"])
        self.lo, self.hi = np.array(c["ctrl_low"]), np.array(c["ctrl_high"])
        self.commands, self.gait_hz = list(c["commands"]), dict(c["gait_hz"])
        self.quiet = set(c["quiet_skills"])
        self.L = c["size_L"]
        self.head = self.names.index("head_yaw") if "head_yaw" in self.names else None
        self.jaw = self.names.index("jaw") if "jaw" in self.names else None
        self.head_rate = c.get("head_yaw_rate", 2.0)
        self.backend = backend or SimBackend(self.names)
        assert list(self.backend.servo_names) == self.names, "backend servo order must match the brains"

        self.last_action = np.zeros(len(self.pol))
        self.steps = 0                      # control ticks since start
        self.vel_cmd = np.zeros(3)
        self.gait_t = 0                     # ticks since the current gait started (gait clock)
        self.plan, self.plan_t = [], 0
        self.plan_all = []                  # the whole plan of the current mode (for "again")
        self.air_ticks, self.landed_at, self.landings = 0, None, 0
        self.skill = self.obs_command = None
        self.fallbacks = []                 # skills asked for that have no brain yet
        self.todo = []                      # (due tick, callable): queued personality steps
        self.look_target, self.head_pos = 0.0, 0.0
        self.mouth, self.audio, self.audio_sr = None, np.zeros(0), None
        self.jaw_pos = 0.0
        self.targets = self.home.copy()
        self.mode = "stand"
        if POSES.exists():
            for k, v in json.loads(POSES.read_text()).items():
                if "ctrl" in v and k != "sit":
                    self.brains[f"pose:{k}"] = PoseBrain(v["ctrl"], self)
                    self.quiet.add(f"pose:{k}")
                if k == "perch":                                    # face-down get-up: tuck, then lift into the perch
                    tuck = self.home.copy()
                    for side, sgn in (("left", 1), ("right", -1)):
                        for j, val in (("hip_pitch", sgn * FD_TUCK[0]), ("knee", FD_TUCK[1]), ("ankle", FD_TUCK[2])):
                            tuck[self.names.index(f"{side}_{j}")] = val
                    self.brains["pose:fd_tuck"] = PoseBrain(tuck, self, glide=0.5)
                    self.brains["pose:fd_lift"] = MoveBrain(v["ctrl"], self, *FD_LIFT)
                    self.brains["pose:lower"] = MoveBrain(tuck, self, *LIE_LOWER)
                    self.quiet.add("pose:lower")
                    self.quiet.update(("pose:fd_tuck", "pose:fd_lift"))
                if k == "perch" and "sit" in v and "rise" in v:   # sit-down into the perch and the stand-up from it
                    self.brains["pose:perch"] = MoveBrain(v["ctrl"], self, **v["sit"])
                    r = dict(v["rise"])
                    self.brains["pose:perch_rise"] = MoveBrain(self.home, self, r.pop("lean"), r.pop("t1"), r.pop("t2"), **r)
                    self.quiet.add("pose:perch_rise")
        self.brains["pose:rise"] = PoseBrain(self.home, self, glide=1.0)
        self.quiet.add("get_up")                            # roll over with the head still (as trained)
        fr = HERE / "model" / "forward_roll.json"
        if fr.exists():
            j = json.loads(fr.read_text())
            self.brains["pose:tumble"] = KeyframeBrain(j["keys"], j["durations"], self)
            self.quiet.add("pose:tumble")
        self._start("stand")

    def lying(self):
        """Trunk tilted more than ~55 deg (IMU gravity: upright reads (0, 0, -1))."""
        return -float(self.backend.read().gravity[2]) < 0.6

    def lying_face_down(self):
        g = self.backend.read().gravity
        return self.lying() and g[0] > 0.5 and "pose:fd_tuck" in self.brains     # nose pointing at the floor

    # ---------- personality inputs ----------
    def command(self, name, vx=0.0, vy=0.0, yaw=0.0):
        """Switch skill now. Returns a short report (also logged)."""
        name = ALIASES.get(name.lower().strip(), name.lower().strip())
        if name not in PLANS:
            return self._say_log(f"unknown command {name!r}")
        fallen = name in ("wake", "get up", "stand") and self.lying_face_down()
        if fallen:
            name = "get up"                                  # fallen on its front: tuck, lift into the perch, stand
        elif name in ("wake", "get up", "stand") and self.lying() and "get_up" in self.brains:
            name, fallen = "roll get up", True               # on its back or side: roll onto the belly first
        if name == "wake" and self.mode in ("sleep", "squat"):
            name = "rise"
        if name == "wake" and self.skill == "pose:perch" and "pose:perch_rise" in self.brains:
            name = "perch rise"
        if name == "wake" and self.skill not in ("sit", "swag_sit", "squat", "sit_to_stand", "roll") and not self.skill.startswith("pose:"):
            name = "stand" if self.skill != "lie" else "get up"          # awake already / lying: get up
        if name == "get up" and not fallen and (self.skill in ("sit", "swag_sit", "sit_to_stand") or self.skill.startswith("pose:")):
            name = "wake"
        if name == "get up" and not fallen and self.skill in ("stand", "walk", "run", "jump"):
            name = "stand"
        self.mode = name
        self.vel_cmd = self._velocity("run" if name == "run" else PLANS[name][0][0], vx, vy, yaw)
        self.plan = list(PLANS[name])
        self.plan_all = list(PLANS[name])
        return self._next_in_plan()

    def look(self, yaw):
        """Turn the head to `yaw` rad (+ = left), at most head_yaw_rate rad/s."""
        if self.head is not None:
            self.look_target = float(np.clip(yaw, self.lo[self.head], self.hi[self.head]))

    def say(self, audio, sample_rate):
        """Speak: the audio goes to the speaker and drives the jaw (MouthSync), 20 ms per tick."""
        if self.mouth is None or self.audio_sr != sample_rate:
            self.mouth, self.audio_sr = MouthSync(sample_rate), sample_rate
        self.audio = np.concatenate([self.audio, np.asarray(audio, float).ravel()])
        self.backend.play(audio, sample_rate)

    @property
    def speaking(self):
        return len(self.audio) > 0

    # ---------- control loop ----------
    def observe(self, s=None):
        """The observation, exactly as SkillEnv builds it (without training noise)."""
        s = s or self.backend.read()
        cmd = np.zeros(len(self.commands))
        cmd[self.commands.index(self.obs_command)] = 1
        f = self.gait_hz.get(self.obs_command)
        if f:
            ph = 2 * np.pi * f * self.gait_t * self.dt
            vel, clock = self.vel_cmd, [np.sin(ph), np.cos(ph)]
        else:
            vel, clock = np.zeros(3), [0.0, 0.0]
        return np.concatenate([s.gravity, s.gyro * 0.25, s.accel / 9.81, s.joint_pos - self.home,
                               s.joint_vel * 0.05, self.last_action, cmd, vel, clock, s.feet]).astype(np.float32)

    def tick(self):
        """One 1/50 s control period: sense, think, act."""
        while self.todo and self.todo[0][0] <= self.steps:
            self.todo.pop(0)[1]()
        obs = self.observe()
        action = np.clip(np.asarray(self.brains[self.skill](obs), float), -1, 1)
        target = self.home.copy()
        target[self.pol] += self.scale * action
        self._personality(target)
        self.targets = np.clip(target, self.lo, self.hi)
        self.backend.write(self.targets)
        self.last_action = action
        self.backend.advance()
        self.steps += 1
        self.plan_t += 1
        if self.obs_command in self.gait_hz:
            self.gait_t += 1
        feet = obs[-2:]
        if feet.sum() == 0:
            self.air_ticks += 1
        else:
            if self.air_ticks >= 1 and self.plan_t > 3:   # a touchdown after a flight (not a settling blip)
                self.landings += 1
                self.landed_at = self.plan_t
            self.air_ticks = 0
        if self.plan and self._step_done(self.plan[0][1]):
            self.plan.pop(0)
            if self.plan and self.plan[0][0] == "again":
                self.plan = list(self.plan_all)
            if self.plan:
                self._next_in_plan()
        return obs, action

    def _step_done(self, secs):
        if secs is None:
            return False
        if isinstance(secs, str) and secs.startswith("landed"):     # "landed" / "landed2": after the n-th touchdown
            n = int(secs[6:] or 1)
            return ((self.landings >= n and self.plan_t >= self.landed_at + LANDED_TICKS)
                    or self.plan_t * self.dt >= LANDED_TIMEOUT)
        return self.plan_t * self.dt >= secs

    def run(self, seconds, every=None):
        for _ in range(int(round(seconds * self.hz))):
            self.tick()
            if every:
                every(self)

    def _personality(self, target):
        quiet = self.skill in self.quiet                 # trained with head still and mouth shut
        if self.head is not None:
            want = 0.0 if quiet else self.look_target
            step = self.head_rate * self.dt
            self.head_pos += float(np.clip(want - self.head_pos, -step, step))
            target[self.head] = self.head_pos
        if self.jaw is not None:
            if self.speaking:
                hop = self.mouth.hop
                chunk, self.audio = self.audio[:hop], self.audio[hop:]
                self.jaw_pos = self.mouth.feed(chunk)
            elif self.mouth is not None:
                self.jaw_pos = self.mouth.silence()
            target[self.jaw] = 0.0 if quiet else self.jaw_pos

    # ---------- skills ----------
    def _velocity(self, skill, vx, vy, yaw):
        L = self.L
        if skill == "walk":
            return np.array([np.clip(vx, -0.1 * L, 0.25 * L), np.clip(vy, -0.1 * L, 0.1 * L), np.clip(yaw, -0.6, 0.6)])
        if skill == "run":
            return np.array([np.clip(vx or 0.4 * L, 0.3 * L, 0.45 * L), 0.0, np.clip(yaw, -0.3, 0.3)])
        return np.zeros(3)

    def _next_in_plan(self):
        skill, secs = self.plan[0]
        msg = self._start(skill)
        timing = f" until {secs}" if isinstance(secs, str) else f" for {secs:g} s" if secs else ""
        return self._say_log(f"{self.mode}: {msg}" + (timing if self.skill == skill else ""))

    def _start(self, skill):
        msg = f"brain {skill}"
        if skill not in self.brains:
            self.fallbacks.append(skill)
            msg = f"no {skill} brain yet, holding stand instead"
            skill = "stand"
            self.plan = [("stand", None)]
        new_cmd = OBS_COMMAND.get(skill, "sit" if skill.startswith("pose:") else None)
        if isinstance(self.brains.get(skill), PoseBrain):
            self.brains[skill].begin()
        if new_cmd != self.obs_command:
            self.gait_t = 0                              # gait clock starts with the gait, as in training
        if new_cmd not in self.gait_hz:
            self.vel_cmd = np.zeros(3)
        self.skill, self.obs_command, self.plan_t = skill, new_cmd, 0
        self.landed_at, self.landings = None, 0
        return msg

    def _say_log(self, msg):
        if self.log:
            self.log(f"[{self.steps * self.dt:6.2f} s] {msg}")
        return msg

    def after(self, seconds, fn):
        self.todo.append((self.steps + int(round(seconds * self.hz)), fn))
        self.todo.sort(key=lambda x: x[0])

    def status(self):
        return dict(mode=self.mode, skill=self.skill, command=self.obs_command, vel_cmd=self.vel_cmd.tolist(),
                    head_yaw=self.head_pos, jaw=self.jaw_pos, speaking=self.speaking, brains=sorted(self.brains))

    # ---------- words ----------
    def answered(self, correct=True, speech_seconds=1.2):
        """The conversation layer calls this after the duck answers a question. A right answer earns
        the celebration: it says so (mouth moving) and drops into the swag pose."""
        if not correct:
            return self._say_log("answer was wrong: no swag this time")
        words, _ = demo_voice(seconds=speech_seconds, seed=7)       # stand-in for "Nailed it!" via text-to-speech
        self.say(words, 22050)
        self.after(speech_seconds, lambda: self.command("swag"))
        return self._say_log("right answer! celebrating with the swag pose")

    def hear(self, text, hold=3.0):
        """Plain-language commands, like the pet's brain.js: "go to sleep", "walk forward then turn left".
        Clauses run one after another, `hold` seconds apart. Returns what was understood."""
        steps = [s for s in (parse(c, self.L) for c in clauses(text)) if s]
        if not steps:
            return [self._say_log(f"didn't understand {text!r}")]
        for i, (kind, args) in enumerate(steps):
            self.after(i * hold, lambda k=kind, a=args: self._do(k, a))
        return [f"{k} {a}" if a else k for k, a in steps]

    def _do(self, kind, args):
        if kind == "look":
            for dt_, yaw in args["script"]:
                self.after(dt_, lambda y=yaw: self.look(y))
        elif kind == "celebrate":
            self.answered(True)
        elif kind == "say":
            words = max(1, len(args["text"].split()))
            audio, _ = demo_voice(seconds=0.4 + 0.3 * words, seed=len(args["text"]))  # stand-in for text-to-speech
            self._say_log(f"says {args['text']!r}")
            self.say(audio, 22050)
        else:
            self.command(kind, **args.get("vel", {}))
            if "for" in args:
                self.after(args["for"], lambda: self.command("stand"))


def clauses(text):
    return [c.strip() for c in re.split(r"\s*(?:[,;.!?]|\band then\b|\bthen\b|\bafter that\b|\band\b)\s*", text, flags=re.I)
            if c.strip()]


def parse(clause, L=1.0):
    """One clause → (kind, args) or None. Order matters: the first match wins."""
    t = clause.lower()
    m = re.match(r"^(?:say|tell me)\s+(.+)$", t)
    if m:
        return "say", {"text": m.group(1)}
    if re.search(r"\b(hello|hi|hey|talk|speak|quack)\b", t):
        return "say", {"text": "hello! I'm a duck" if re.search(r"\b(hello|hi|hey)\b", t) else "quack quack, hello there"}
    if re.search(r"\blook\b", t):
        if "around" in t:
            return "look", {"script": [(0, 1.2), (1.5, -1.2), (3.0, 0.0)]}
        yaw = 1.0 if "left" in t else -1.0 if "right" in t else 3.0 if "behind" in t else 0.0
        return "look", {"script": [(0, yaw)]}
    if re.search(r"\b(stop|stay|freeze|halt|still|enough)\b", t):
        return "stand", {}
    if re.search(r"\b(sleep|nap|rest|bed ?time|good night|tired)\b", t):
        return "sleep", {}
    if re.search(r"\bwake\b", t):
        return "wake", {}
    if re.search(r"\bget up\b|\bstand up\b|^up$", t):
        return "get up", {}
    if re.search(r"\b(correct|that'?s right|right answer|you got it|nailed it|well done)\b", t):
        return "celebrate", {}
    if re.search(r"\bswag sit\b|\bsit (?:with )?swag\b", t):
        return "swag sit", {}
    if re.search(r"\bswag\b|\bchill\b|\brelax\b|\blean back\b", t):
        return "swag", {}
    if re.search(r"\bsquat\b|\bsquatting\b|\bcrouch\b|\bduck down\b", t):
        return "squat", {}
    if re.search(r"\bsit\b|\bsitting\b", t):
        return "sit", {}
    if re.search(r"\b(lie|lay|lying|play dead)\b|^down$|\bget down\b", t):
        return "lie", {}
    if re.search(r"\b(jump(?:ing|s)?|hop(?:ping|s)?|leap(?:ing|s)?)\b", t):
        once = re.search(r"\b(once|one|single|a (?:little |small )?(?:hop|jump))\b", t) and not re.search(r"\b(jumping|hopping|keep)\b", t)
        return ("hop" if once else "jump"), {}
    if re.search(r"\b(roll|somersault|tumble)\b", t):
        return "roll", {}
    turn = 0.5 if "left" in t else -0.5 if "right" in t else 0.0
    if re.search(r"\b(run|sprint|dash|hurry)\b", t):
        return "run", {"vel": {"vx": 0.45 * L, "yaw": 0.6 * turn}}
    if re.search(r"\bturn around\b|\bspin\b", t):
        return "walk", {"vel": {"yaw": 0.6}, "for": np.pi / 0.6}
    if re.search(r"\bturn\b", t):
        return "walk", {"vel": {"yaw": turn or 0.5}}
    if re.search(r"\b(walk|go|move|come|step|forward|back(?:ward)?s?)\b", t):
        vx = -0.08 * L if re.search(r"\bback", t) else 0.0 if re.search(r"\b(left|right)\b", t) else 0.15 * L
        vy = 0.08 * L if "left" in t else -0.08 * L if "right" in t else 0.0
        return "walk", {"vel": {"vx": vx, "vy": vy}}
    if re.search(r"\b(stand|idle|wait)\b", t):
        return "stand", {}
    return None


# ---------------------------------------------------------------- demo
DEMO = [  # (name, what the personality does, seconds, expectation)
    ("stand, look around, talk", ["stand", "look around", ("say", 2.5)], 6.0, "upright"),
    ("go to sleep", ["go to sleep"], 6.0, "upright"),
    ("wake up", ["wake up"], 6.0, "upright"),
    ("say hello", ["say hello I am awake"], 3.0, "upright"),
    ("walk forward", ["walk forward"], 5.0, "upright"),
    ("turn left", ["turn left"], 4.0, "upright"),
    ("stop", ["stop"], 3.0, "upright"),
    ("lie down", ["lie down"], 6.0, "low"),
    ("get up", ["get up"], 7.0, "end upright"),
    ("jump", ["jump"], 4.0, "upright"),
    ("stop", ["stop"], 3.0, "upright"),
    ("roll", ["roll"], 8.0, "end upright"),
]


def demo(out_dir=HERE / "renders", render=True, script=None, models_dir=MODELS):
    rt = DuckRuntime(models_dir=models_dir, log=print)
    be = rt.backend
    print(f"brains: {', '.join(sorted(rt.brains))}")
    frames, strip, results = [], [], []
    can_render = render

    def watch(r):
        nonlocal can_render
        p = be.posture()
        stats["min_up"] = min(stats["min_up"], p["up"])
        stats["fell"] |= p["up"] < 0.3
        if can_render and r.steps % 2 == 0:
            try:
                frames.append(be.render())
            except Exception as e:                            # no GL: carry on without video
                print(f"rendering off: {e}")
                can_render = False

    phases = DEMO if script is None else [(c, [c], 4.0, "upright") for c in clauses(script)]
    for name, actions, secs, expect in phases:
        stats = dict(min_up=1.0, fell=False)
        rt.fallbacks = []
        for a in actions:
            if isinstance(a, tuple):
                audio, _ = demo_voice(seconds=a[1], sr=22050)
                rt.after(1.0, lambda au=audio: rt.say(au, 22050))
            else:
                rt.hear(a)
        rt.run(secs, every=watch)
        p = be.posture()
        fb = sorted(set(rt.fallbacks))
        if fb:
            expect = "upright"                                   # its brain isn't trained yet: stand held instead
        ok = {"upright": not stats["fell"] and stats["min_up"] > 0.8,
              "low": not stats["fell"] and p["height"] < 0.8 * rt.cfg["stand_height"],
              "end upright": p["up"] > 0.85}[expect]
        results.append(dict(phase=name, skill=rt.skill, fallback=fb, min_up=round(stats["min_up"], 3),
                            end_up=round(p["up"], 3), end_height=round(p["height"], 4), expect=expect, ok=bool(ok)))
        if can_render and frames:
            strip.append(frames[-1])
    print("\nphase                      brain now     no brain yet (fell back to stand)   min up  end up  height  check")
    for r in results:
        print(f"{r['phase']:26s} {r['skill']:13s} {', '.join(r['fallback']) or '-':37s} {r['min_up']:6.2f}  {r['end_up']:6.2f}"
              f"  {r['end_height']:.3f}  {r['expect']}: {'yes' if r['ok'] else 'NO'}")
    if frames:
        out_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        video = out_dir / "runtime_demo.mp4"
        imageio.mimsave(video, frames, fps=rt.hz // 2, macro_block_size=8)
        rows = [np.concatenate(strip[i:i + 6] + [np.zeros_like(strip[0])] * (6 - len(strip[i:i + 6])), axis=1)
                for i in range(0, len(strip), 6)]
        imageio.imwrite(out_dir / "runtime_demo_strip.png", np.concatenate(rows, axis=0))
        print(f"saved {video} ({len(frames)} frames) and {out_dir / 'runtime_demo_strip.png'} (end of each phase)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="scripted show in the sim")
    ap.add_argument("--script", help='text commands, e.g. "go to sleep; wake up; walk forward" (4 s each)')
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--models", default=str(MODELS), help="folder with mid-<skill>/policy.onnx brains")
    a = ap.parse_args()
    if a.demo or a.script:
        demo(render=not a.no_render, script=a.script, models_dir=Path(a.models))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
