"""Dog imitation on a GPU: imitate_env.ImitateEnv in MuJoCo MJX, trained with Brax PPO, warm-started from the CPU brain.

    python mjx_imitate.py warm runs/imitate18b_wide/policy_16500000_steps.zip warm.pkl   # CPU stack: SB3 brain -> numpy
    python mjx_imitate.py train --warm warm.pkl --steps 2_000_000_000 --envs 4096 --out runs/imitate_gpu   # on a GPU
    python mjx_imitate.py train --warm warm.pkl --steps 20_000 --envs 16 --out /tmp/t --quick              # CPU smoke test

Same as ImitateEnv: the 75 inputs (same order and scaling), the five tracking terms and their weights, reference
state initialization (random clip and moment, Bolt starting in the dog's pose and moving as it was), early
termination (15 cm off the reference height, 0.8 rad tilt, or the body on the floor while the dog's was not), the
A1 motor model every physics substep, physics randomization (motor gain, body mass, floor friction), sensor noise,
and clip switching mid-try (switch_rate 0.3/s, NEXT rules, a gait joins the new gait at its closest-pose frame).
Clips are drawn canter 3x, the others 1x. Differences, for speed: the robot collides with the floor, and its thighs
with the torso (the only self-contacts the CPU env ever has), but nothing else with itself, and the physics is
float32 (test_mjx_imitate.py measures the gap). Each try restarts from a fresh random clip and moment when it ends
(not from the env's first state as Brax's AutoResetWrapper would), and the end of a clip or of the 5 s on a loop is
a time-out (truncation), not a failure, as in the CPU env.
Landing starts (--landing-rate, default 0) as ImitateEnv's: that share of the tries starts in a landing state of the
run-up jump brain (research/results/landing_starts.npz), tracking canter, with --landing-grace s free of the height
check and a switch to --landing-then after --landing-bridge s. At rate 0 nothing extra is drawn or computed: the env
is the same computation as before (test_mjx_imitate.py checks both).
Posture commands while moving (--moving-rate, default 0) as ImitateEnv's: a try starts in walk or trot at a speed drawn
from --moving-speed and is told at once to sit, lie or get up (--moving-clips), joining that clip at its closest-pose
moment (imitate_env.closest_moment, the skill manager's rule); reaching the clip's end pose (pose_reached.py, the switch test's definition)
within --pose-time s pays --pose-weight per step held. Rate 0: the same computation as before.
Transition term (--transition-weight, default 0 = off) as ImitateEnv's: for --transition-window s after a mid-try clip
switch, -weight x (roll and pitch beyond --transition-threshold rad, each against the reference's own) per step.
Real-robot gaps (--latency-min-ms/--latency-max-ms, --obs-noise, --motor-lag-ms; all off by default) as ImitateEnv's
(mastermind/env.py's stress-test models): action latency drawn per try at the 500 Hz physics rate, legged_gym sensor
noise, first-order motor lag. Off: the same computation as before.
Rough terrain and pushes (step 6; --terrain-cm, --push-interval; off by default) as ImitateEnv's: bolt_env's height
field beside the flat floor, walk and trot tries starting on it, body height measured from the ground under the body;
sideways pushes (bolt_env.apply_shove, the switch test's shove). --terrain-cm takes several heights and --terrain-at
the step fractions where each starts (a curriculum, e.g. 0 5 10 at 0 0.3 0.6): training runs in phases, each phase a
fresh env warm-started from the previous phase's brain.
Motor brake (--motor-brake, on by default for training; MjxImitate(motor_brake=False) by default, as bolt_env's process
default): back-EMF braking above --brake-speed (25 rad/s, the drive cutoff), --brake-gain N·m per rad/s (3.5), capped
at the motor's torque, as bolt_env.physics.

The brain: Brax PPO with SB3's network (ELU MLP, plain Gaussian with a learned log std, inputs normalized and clipped
to +-10), so the CPU brain's weights load unchanged (warm), and a trained brain exports back to numpy
(MjxImitatePolicy.act, the same call as imitate_env.ImitatePolicy, for exam() and cloudwatch.record_imitate).

Writes <out>/params.pkl (numpy brain), <out>/checkpoints/params_<steps>.pkl every eval (num_evals evenly over the
run), <out>/progress.json, and "update at N: ..." lines.
"""
import argparse
import functools
import json
import pickle
import re
import time
from pathlib import Path

import mujoco
import numpy as np

try:   # the numpy brain (MjxImitatePolicy) and the clip tables work without JAX (the CPU stack has none)
    import jax
    import jax.numpy as jp
    from mujoco import mjx
except ImportError:   # pragma: no cover
    jax = jp = mjx = None

HERE = Path(__file__).resolve().parent
CLIP_DIR = HERE / "dogmotion" / "clips"
# copied from bolt_env / imitate_env (Kaggle has no CPU training stack to import them from)
DOG = np.array([0.0, 0.2747, -0.5493, 0.0, 0.2747, -0.5493, 0.0, -0.2747, 0.5493, 0.0, -0.2747, 0.5493])
ACTION_SCALE = np.tile([0.4, 1.0, 1.2], 4)
MOTOR = dict(torque=33.5, power=420.0, speed=21.0, cutoff=25.0)
TRANSITION = dict(weight=0.0, window=1.0, threshold=0.3)   # as imitate_env.TRANSITION
NOISE = dict(grav=0.05, gyro=0.2, linvel=0.1, height=0.02, qpos=0.01, qvel=1.5)   # as imitate_env / mastermind.env NOISE
REAL = dict(latency_ms=(0.0, 0.0), obs_noise=0.0, motor_lag_ms=0.0)              # as imitate_env.REAL: off
# rough terrain and pushes, copied from bolt_env (TERRAIN, terrain_field, terrain_xml, terrain_height, apply_shove) and
# imitate_env (TERRAIN_ARGS, PUSH); test_mjx_imitate.py checks the copies give the same field, xml, heights and shoves
TERRAIN = dict(cm=0.0, half=20.0, cell=0.1, at=(40.0, 0.0), seed=0, base=0.05)
TERRAIN_ARGS = dict(cm=TERRAIN["cm"], clips=("walk", "trot"), spread=5.0, seed=TERRAIN["seed"])
PUSH = dict(interval=0.0, prob=0.5, speed=1.0, spin=0.0)


def terrain_field(seed=TERRAIN["seed"], half=TERRAIN["half"], cell=TERRAIN["cell"]):
    n = int(round(2 * half / cell)) + 1
    return np.random.default_rng(seed).uniform(0.0, 1.0, (n, n))


def terrain_xml(xml, cm, half=TERRAIN["half"], cell=TERRAIN["cell"], at=TERRAIN["at"], base=TERRAIN["base"]):
    n = int(round(2 * half / cell)) + 1
    asset = f'<hfield name="terrain" nrow="{n}" ncol="{n}" size="{half} {half} {cm / 100.0} {base}"/>'
    xml = xml.replace("<asset>", "<asset>\n    " + asset, 1)
    # only the paws and shins touch the bumps (contype bit 4); the body, head and thighs meet the flat floor under the
    # field. Box-on-hfield contacts made MJX 6x slower (paws and shins: 1.9x); on 5-10 cm bumps an upright body is 30+ cm
    # above them, and a fall is still caught (height and tilt limits, the floor under the field)
    geom = (f'<geom name="terrain" type="hfield" hfield="terrain" pos="{at[0]} {at[1]} 0" contype="0" conaffinity="4" '
            'rgba="0.55 0.5 0.42 1"/>')
    for cls, typ in (("foot", "sphere"), ("shin", "capsule")):   # CPU: 1 | 4 (mjx_imitate.collisions makes it 2 | 4)
        xml = xml.replace(f'<default class="{cls}"><geom type="{typ}"', f'<default class="{cls}"><geom type="{typ}" contype="5"', 1)
    i = xml.rindex("</worldbody>")
    return xml[:i] + "  " + geom + "\n  " + xml[i:]


def terrain_height(field, cm, x, y, half=TERRAIN["half"], cell=TERRAIN["cell"], at=TERRAIN["at"]):
    """Ground height (m) at (x, y): bilinear in the field's cells, 0 off the field (JAX)."""
    n = field.shape[0]
    u, v = (x - at[0] + half) / cell, (y - at[1] + half) / cell
    on = (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1)
    u, v = jp.clip(u, 0, n - 1 - 1e-6), jp.clip(v, 0, n - 1 - 1e-6)
    c, r = jp.floor(u).astype(jp.int32), jp.floor(v).astype(jp.int32)
    fu, fv = u - c, v - r
    h = (field[r, c] * (1 - fu) * (1 - fv) + field[r, c + 1] * fu * (1 - fv) + field[r + 1, c] * (1 - fu) * fv
         + field[r + 1, c + 1] * fu * fv)
    return jp.where(on, h * cm / 100.0, 0.0)


def apply_shove(qvel, R, speed, spin, side):
    """bolt_env.apply_shove in JAX: body velocity + side x speed along the body's y axis, roll rate - side x spin."""
    return qvel.at[0:3].add(R @ jp.array([0.0, 1.0, 0.0]) * (side * speed)).at[3:6].add(jp.array([-1.0, 0.0, 0.0]) * (side * spin))
BRAKE = dict(speed=MOTOR["cutoff"], gain=3.5)   # bolt_env: brake_speed default the drive cutoff, BRAKE_GAIN
ALL_CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"]
WEIGHTS = dict(pose=0.5, vel=0.05, paw=0.2, root=0.15, root_vel=0.1)
HOLD, LOOP_SECONDS = 1.0, 5.0
GAITS = ("walk", "trot", "pace", "canter", "turn_left", "turn_right")
NEXT = {**{g: GAITS + ("sit",) for g in GAITS}, "sit": ("lie",), "lie": ("getup",), "getup": ("walk", "trot")}
CLIP_WEIGHTS = {"canter": 3.0}   # how often each clip is drawn (1 for the rest): the canter matters most
LANDING_FILE = HERE / "research" / "results" / "landing_starts.npz"   # research/gen_landing_starts.py
LANDING = dict(rate=0.0, grace=0.5, bridge=0.6, then="trot")         # as imitate_env.LANDING
MOVING = dict(rate=0.0, speed=(0.5, 2.0), gaits=("walk", "trot"), clips=("sit", "lie", "getup"),
              pose_time={"sit": 4.0, "lie": 5.5, "getup": 5.5}, weight=0.2)   # as imitate_env.MOVING
KEYS = ("joints", "joint_vel", "height", "gravity", "v_body", "w_body", "paws", "tilt")


def floor_only(xml):
    """Robot geoms collide with the floor only: robot contype 2 / conaffinity 1, floor contype 1 / conaffinity 2."""
    xml = re.sub(r'(<geom name="floor"[^>]*?)(/?>)', r'\1 contype="1" conaffinity="2"\2', xml, count=1)
    return xml.replace('<geom friction="1.0 0.02 0.01" condim="3"', '<geom friction="1.0 0.02 0.01" condim="3" contype="2" conaffinity="1"', 1)


def collisions(xml):
    """floor_only, plus the contacts of the thighs with the torso, which the CPU env has in about a third of all steps
    (pushing at ~130 N): leg-on-leg contacts never happen there, so only these were missing."""
    xml = floor_only(xml).replace('contype="5"', 'contype="6"')   # rough terrain: paws and shins 2 | 4 (terrain_xml)
    for leg in ("FL", "FR", "RL", "RR"):   # the thigh capsules, in this order in bolt.xml
        xml = xml.replace('<geom class="thigh" fromto', f'<geom name="{leg}_thigh_geom" class="thigh" fromto', 1)
    pairs = "".join(f'<pair geom1="torso" geom2="{leg}_thigh_geom" condim="3" friction="1 1 0.02 0.01 0.01"/>' for leg in ("FL", "FR", "RL", "RR"))
    return xml.replace("</mujoco>", f"  <contact>{pairs}</contact>\n</mujoco>")


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def clip_tables(names, clip_dir=CLIP_DIR):
    """imitate_env.Clip's per-frame quantities for each clip, padded (last frame repeated) to equal length."""
    per = []
    for name in names:
        z = np.load(Path(clip_dir) / f"{name}.npz")
        n = len(z["joints"])
        R = np.zeros((n, 9))
        for t, q in enumerate(z["base_quat"]):
            mujoco.mju_quat2Mat(R[t], q)
        R = R.reshape(n, 3, 3)
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        rel = z["foot_pos"] - z["base_pos"][:, None]
        per.append(dict(n=n, fps=float(z["fps"]), loop=bool(z["loop"]), joints=z["joints"], joint_vel=z["joint_vel"],
                        height=z["base_pos"][:, 2], gravity=np.einsum("tji,j->ti", R, [0, 0, -1.0]),
                        v_body=np.einsum("tji,tj->ti", R, z["base_linvel"]), w_body=np.einsum("tji,tj->ti", R, z["base_angvel"]),
                        paws=np.stack([rel[t] @ _rotz(yaw[t]) for t in range(n)]),
                        tilt=np.stack([_rotz(-yaw[t]) @ R[t] for t in range(n)])))
    N = max(p["n"] for p in per)
    out = dict(n=np.array([p["n"] for p in per]), fps=np.array([p["fps"] for p in per]), loop=np.array([p["loop"] for p in per]))
    out["length"] = out["n"] / out["fps"]
    for k in KEYS:
        out[k] = np.stack([np.concatenate([p[k], np.repeat(p[k][-1:], N - p["n"], 0)]) for p in per])
    return out


# ---------- the brain in numpy ----------
class MjxImitatePolicy:
    """A params.pkl brain (from training here, or warm-started from SB3) -> act(obs), as imitate_env.ImitatePolicy:
    normalize and clip to +-10, ELU MLP, the Gaussian's mean clipped to the action range."""

    def __init__(self, path):
        P = pickle.loads(Path(path).read_bytes()) if not isinstance(path, dict) else path
        self.P = P
        self.mean, self.std, self.layers = np.asarray(P["mean"]), np.asarray(P["std"]), P["layers"]
        self.slots = len(self.mean) - 66

    def act(self, obs):
        obs = np.asarray(obs, np.float32)
        if len(obs) > 66 + self.slots:   # an older brain: drop the clip slots it never had
            obs = np.concatenate([obs[:48 + self.slots], obs[48 + len(ALL_CLIPS):]])
        x = np.clip((obs - self.mean) / self.std, -10.0, 10.0).astype(np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))   # ELU
        return np.clip(x, -1.0, 1.0)


def from_sb3(zip_path, stats_path):
    """An SB3 PPO brain (policy zip + VecNormalize pkl) -> the numpy dict params.pkl uses (needs torch + SB3)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from imitate_env import ImitateEnv
    model = PPO.load(zip_path, device="cpu")
    slots = model.observation_space.shape[0] - 66
    norm = VecNormalize.load(str(stats_path), DummyVecEnv([lambda: ImitateEnv(slots=slots)]))
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.policy.state_dict().items()}

    def mlp(prefix, head):
        idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith(prefix)})
        return [(sd[f"{prefix}.{i}.weight"].T, sd[f"{prefix}.{i}.bias"]) for i in idx] + [(sd[f"{head}.weight"].T, sd[f"{head}.bias"])]

    act = model.policy_kwargs.get("activation_fn")
    assert act is None or act.__name__ == "ELU", act
    var = norm.obs_rms.var.astype(np.float64)
    return dict(mean=norm.obs_rms.mean.astype(np.float32), std=np.sqrt(var + norm.epsilon).astype(np.float32), var=var.astype(np.float32),
                count=float(norm.obs_rms.count), clip_obs=float(norm.clip_obs),
                layers=mlp("mlp_extractor.policy_net", "action_net"), log_std=sd["log_std"],
                value=mlp("mlp_extractor.value_net", "value_net"),
                reward_scale=float(1.0 / np.sqrt(norm.ret_rms.var + norm.epsilon)) if norm.norm_reward else 1.0,
                hidden=tuple(int(W.shape[1]) for W, _ in mlp("mlp_extractor.policy_net", "action_net")[:-1]),
                steps=int(model.num_timesteps), source=str(zip_path))


# ---------- the environment ----------
class MjxImitate:
    """Brax-style env (reset(rng) -> State, step(State, action) -> State) equal to imitate_env.ImitateEnv."""

    observation_size = 66 + len(ALL_CLIPS)
    action_size = 12
    backend = "mjx"

    def __init__(self, randomize=True, clips=None, switch_rate=0.3, weights=None, clip_weights=None, clip_dir=CLIP_DIR,
                 iterations=5, ls_iterations=8, landing_rate=LANDING["rate"], landing_file=LANDING_FILE,
                 landing_grace=LANDING["grace"], landing_bridge=LANDING["bridge"], landing_then=LANDING["then"],
                 moving_rate=MOVING["rate"], moving_speed=MOVING["speed"], moving_gaits=MOVING["gaits"], moving_clips=MOVING["clips"],
                 pose_time=None, pose_weight=MOVING["weight"], transition_weight=TRANSITION["weight"],
                 transition_window=TRANSITION["window"], transition_threshold=TRANSITION["threshold"],
                 latency_ms=REAL["latency_ms"], obs_noise=REAL["obs_noise"], motor_lag_ms=REAL["motor_lag_ms"],
                 terrain_cm=TERRAIN_ARGS["cm"], terrain_clips=TERRAIN_ARGS["clips"], terrain_spread=TERRAIN_ARGS["spread"],
                 terrain_seed=TERRAIN_ARGS["seed"], push_interval=PUSH["interval"], push_prob=PUSH["prob"], push_speed=PUSH["speed"],
                 push_spin=PUSH["spin"], motor_brake=False, brake_speed=BRAKE["speed"], brake_gain=BRAKE["gain"]):
        from brax.envs.base import State
        self.State = State
        xml = (HERE / "bolt.xml").read_text()
        if terrain_cm > 0:   # rough terrain beside the floor (bolt_env.terrain_xml)
            xml = terrain_xml(xml, terrain_cm)
        xml = collisions(xml)
        self.m = m = mujoco.MjModel.from_xml_string(xml)
        self.terrain = None
        if terrain_cm > 0:
            field = terrain_field(terrain_seed)
            m.hfield_data[:] = field.ravel()
            self.terrain = dict(field=jp.array(field, jp.float32), cm=float(terrain_cm), geom=m.geom("terrain").id,
                                clips=jp.array([n in terrain_clips for n in (clips or ALL_CLIPS)]), spread=float(terrain_spread))
        self.push_steps = int(round(push_interval / (m.opt.timestep * 10))) if push_interval else 0
        self.push_prob, self.push_speed, self.push_spin = float(push_prob), float(push_speed), float(push_spin)
        # contact solver: MuJoCo's 100/50 iterations run to the slowest env under vmap; 5/8 gives the same physics
        # here (one control step off the CPU env by the same amounts as 100/50) at a third of the cost
        m.opt.iterations, m.opt.ls_iterations = iterations, ls_iterations
        self.mx = mjx.put_model(m)
        self.randomize, self.switch_rate = randomize, switch_rate
        self.weights = dict(WEIGHTS, **(weights or {}))
        self.names = list(clips or ALL_CLIPS)
        T = clip_tables(self.names, clip_dir)
        self.T = {k: jp.array(T[k], jp.float32) for k in KEYS}
        self.n, self.fps, self.length = jp.array(T["n"], jp.int32), jp.array(T["fps"], jp.float32), jp.array(T["length"], jp.float32)
        self.loop = jp.array(T["loop"])
        self.slot = jp.array([ALL_CLIPS.index(n) for n in self.names], jp.int32)
        cw = dict(CLIP_WEIGHTS, **(clip_weights or {}))
        self.clip_logits = jp.log(jp.array([cw.get(n, 1.0) for n in self.names], jp.float32))
        self.next = jp.array([[b in NEXT.get(a, ()) and b != a for b in self.names] for a in self.names])
        self.canter = self.names.index("canter") if "canter" in self.names else -1
        self.torso, self.floor = m.body("torso").id, m.geom("floor").id
        assert all(m.body(m.geom_bodyid[m.geom(f"{l}_thigh_geom").id]).name == f"{l}_thigh" for l in ("FL", "FR", "RL", "RR"))
        self.feet = jp.array([m.geom(f"{l}_foot").id for l in ("FL", "FR", "RL", "RR")])
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in m.geom_bodyid]
        ground = {self.floor} | ({self.terrain["geom"]} if self.terrain else set())
        self.body_mask = jp.array([n[:2] not in ("FL", "FR", "RL", "RR") and g not in ground for g, n in enumerate(bodies)])
        self.lo, self.hi = jp.array(m.actuator_ctrlrange[:, 0]), jp.array(m.actuator_ctrlrange[:, 1])
        self.kp0, self.fric0, self.mass0 = jp.array(m.actuator_gainprm[:, 0]), jp.array(m.geom_friction[:, 0]), float(m.body_mass[self.torso])
        self.qpos0 = jp.array(m.qpos0, jp.float32)
        self.dt = float(m.opt.timestep * 10)
        self.max_steps = int(LOOP_SECONDS * 50)
        self.dog, self.scale = jp.array(DOG, jp.float32), jp.array(ACTION_SCALE, jp.float32)
        # landing starts (as ImitateEnv): None at rate 0, so reset and step are the computation they always were
        self.landing_rate = float(landing_rate)
        self.L = None
        if self.landing_rate > 0:
            z = np.load(landing_file)
            need = {ALL_CLIPS[c] for c in np.unique(z["clip"])} | ({landing_then} if landing_then and landing_bridge else set())
            assert need <= set(self.names), f"landing starts need the clips {sorted(need)} (env has {self.names})"
            self.L = dict(qpos=jp.array(z["qpos"], jp.float32), qvel=jp.array(z["qvel"], jp.float32),
                          clip=jp.array([self.names.index(ALL_CLIPS[c]) for c in z["clip"]], jp.int32), tau0=jp.array(z["tau0"], jp.float32))
            self.grace_steps = int(round(landing_grace / self.dt))
            self.land_switch_at = int(round(landing_bridge / self.dt)) if landing_then and landing_bridge else -1
            self.land_then = self.names.index(landing_then) if landing_then and landing_bridge else -1
        # real-robot gaps (as ImitateEnv): latency (substeps, drawn per try), sensor noise, motor lag; off = not computed
        self.latency_ms, self.obs_noise, self.motor_lag_ms = tuple(float(x) for x in latency_ms), float(obs_noise), float(motor_lag_ms)
        self.real_on = self.latency_ms[1] > 0 or self.motor_lag_ms > 0
        self.sub_dt = float(m.opt.timestep)
        self.queue_len = int(round(self.latency_ms[1] / 1000.0 / self.sub_dt)) // 10 + 2   # control steps of sent targets kept
        # transition term (as ImitateEnv): off at weight 0
        self.transition_weight, self.transition_threshold = float(transition_weight), float(transition_threshold)
        self.transition_steps = int(round(transition_window / self.dt))
        # moving-start posture commands (as ImitateEnv): None at rate 0
        self.brake = (float(brake_speed), float(brake_gain)) if motor_brake else None
        self.moving_rate, self.moving_speed, self.pose_weight = float(moving_rate), tuple(moving_speed), float(pose_weight)
        self.M = None
        if self.moving_rate > 0:
            from pose_reached import CLIP_POSE, POSE_OK, pose_ok   # (only with moving starts: older kernels lack the file)
            self.pose_ok = pose_ok
            gaits = [g for g in moving_gaits if g in self.names and bool(T["loop"][self.names.index(g)])]
            posts = [c for c in moving_clips if c in self.names and c in CLIP_POSE]
            assert gaits and posts, f"moving starts need gaits {moving_gaits} and clips {moving_clips}"
            pt = dict(MOVING["pose_time"], **(pose_time or {}))
            gi = [self.names.index(g) for g in gaits]
            self.pose_names = list(POSE_OK)
            self.M = dict(gaits=jp.array(gi, jp.int32), posts=jp.array([self.names.index(c) for c in posts], jp.int32),
                          speed=jp.array([float(T["v_body"][i, :T["n"][i], 0].mean()) for i in gi], jp.float32),
                          pose=jp.array([self.pose_names.index(CLIP_POSE[n]) if n in CLIP_POSE else -1 for n in self.names], jp.int32),
                          until=jp.array([int(round(pt[n] / self.dt)) if n in pt else 0 for n in self.names], jp.int32))

    @property
    def unwrapped(self):
        return self

    # ---------- reference ----------
    def frame(self, c, tau):
        n = self.n[c]
        u = tau * self.fps[c]
        ul = jp.mod(u, n.astype(jp.float32))
        il = jp.minimum(jp.floor(ul).astype(jp.int32), n - 1)
        un = jp.clip(u, 0.0, (n - 1).astype(jp.float32))
        i_n = jp.floor(un).astype(jp.int32)
        loop = self.loop[c]
        i = jp.where(loop, il, i_n)
        j = jp.where(loop, jp.where(il + 1 >= n, 0, il + 1), jp.minimum(i_n + 1, n - 1))
        b = jp.where(loop, ul - il, un - i_n)
        return i, j, b

    def tau(self, info, ahead=0):
        return info["tau0"] + (info["t"] + ahead).astype(jp.float32) * self.dt

    def ref(self, info, key, ahead=0):
        c = info["clip"]
        i, j, b = self.frame(c, self.tau(info, ahead))
        a = self.T[key][c]
        return (1 - b) * a[i] + b * a[j]

    def finished(self, info):
        return jp.where(self.loop[info["clip"]], 0.0, jp.maximum(0.0, self.tau(info) - self.length[info["clip"]]))

    def phase(self, info):
        c, tau = info["clip"], self.tau(info)
        return jp.where(self.loop[c], jp.mod(tau / self.length[c], 1.0), jp.minimum(tau / self.length[c], 1.0))

    # ---------- physics ----------
    def model_for(self, info):
        mx = self.mx
        kp = self.kp0 * info["kp"]
        return mx.replace(actuator_gainprm=mx.actuator_gainprm.at[:, 0].set(kp), actuator_biasprm=mx.actuator_biasprm.at[:, 1].set(-kp),
                          geom_friction=mx.geom_friction.at[:, 0].set(self.fric0 * info["fric"]),
                          body_mass=mx.body_mass.at[self.torso].set(self.mass0 * info["mass"]))

    def physics(self, mx, d, n=10):
        return jax.lax.scan(lambda d, _: (self.substep(mx, d), None), d, None, length=n)[0]

    def physics_real(self, mx, d, info):
        """One control step with the latency and motor lag (ImitateEnv._delayed_physics): substep i gets the target sent
        back (+1 for the first `sub` substeps) control steps ago, through the first-order lag. Returns (d, info)."""
        sent = d.ctrl
        hist = jp.concatenate([info["qhist"][1:], sent[None]])   # newest last
        back, sub = info["dsub"] // 10, info["dsub"] % 10
        target, older = hist[self.queue_len - 1 - back], hist[self.queue_len - 2 - back]
        g = 1.0 - np.exp(-self.sub_dt / (self.motor_lag_ms / 1000.0)) if self.motor_lag_ms else 1.0

        def one(carry, i):
            d, y = carry
            want = jp.where(i < sub, older, target)
            if self.motor_lag_ms:
                y = y + g * (want - y)
                want = y
            return (self.substep(mx, d.replace(ctrl=want)), y), None

        (d, y), _ = jax.lax.scan(one, (d, info["lagy"]), jp.arange(10))
        return d.replace(ctrl=sent), dict(info, qhist=hist, lagy=y)

    def real_init(self, info, d):
        """The latency queue and lag state of a fresh try: the robot holds its first targets."""
        return dict(info, qhist=jp.tile(d.ctrl[None], (self.queue_len, 1)), lagy=d.ctrl)

    def substep(self, mx, d):
        """One physics substep (2 ms) with the A1 motor model and, if on, the brake."""
        if self.brake is not None:   # back-EMF braking above brake speed, against the motion (bolt_env.brake_torque)
            speed, gain = self.brake
            v = d.qvel[6:18]
            d = d.replace(qfrc_applied=d.qfrc_applied.at[6:18].set(
                -jp.sign(v) * jp.minimum(gain * jp.maximum(jp.abs(v) - speed, 0.0), MOTOR["torque"])))
        w = jp.abs(d.qvel[6:18])
        drive = jp.minimum(MOTOR["torque"], MOTOR["power"] / jp.maximum(w, 1e-6))
        drive = drive * jp.clip((MOTOR["cutoff"] - w) / (MOTOR["cutoff"] - MOTOR["speed"]), 0.0, 1.0)
        lo = jp.where(d.qvel[6:18] < 0, -drive, -MOTOR["torque"])
        hi = jp.where(d.qvel[6:18] >= 0, drive, MOTOR["torque"])
        return mjx.step(mx.replace(actuator_forcerange=jp.stack([lo, hi], -1)), d)

    # ---------- senses and scores ----------
    def obs(self, d, info, key=None):
        R = d.xmat[self.torso]
        ph = 2 * jp.pi * self.phase(info)
        onehot = jp.zeros(len(ALL_CLIPS)).at[self.slot[info["clip"]]].set(1.0)
        o = jp.concatenate([R.T @ jp.array([0.0, 0.0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3],
                            d.qpos[2:3] if self.terrain is None else (d.qpos[2] - self.ground_z(d))[None],
                            d.qpos[7:19] - self.dog, d.qvel[6:18] * 0.1, info["last_action"], jp.stack([jp.sin(ph), jp.cos(ph)]), onehot,
                            self.ref(info, "joints", 1) - self.dog, self.ref(info, "height", 1)[None], self.ref(info, "v_body", 1),
                            self.ref(info, "w_body", 1)[2:3], self.finished(info)[None]])
        if self.obs_noise and key is not None:   # sensor noise (imitate_env.ImitateEnv._obs): its own key
            key, kn = jax.random.split(key)
            k, u = self.obs_noise, jax.random.uniform(kn, (34,), minval=-1.0, maxval=1.0)
            scale = jp.concatenate([jp.full(3, NOISE["grav"]), jp.full(3, NOISE["gyro"] * 0.25), jp.full(3, NOISE["linvel"]),
                                    jp.full(1, NOISE["height"]), jp.full(12, NOISE["qpos"]), jp.full(12, NOISE["qvel"] * 0.1)])
            o = o.at[:34].add(k * scale * u)
        if self.randomize and key is not None:
            o = o + 0.01 * jax.random.normal(key, o.shape)
        return o

    def heading_paws(self, d):
        R = d.xmat[self.torso]
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        Rz = jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return (d.geom_xpos[self.feet] - d.qpos[:3]) @ Rz

    def tracking(self, d, info):
        R = d.xmat[self.torso]
        q, qd = d.qpos[7:19], d.qvel[6:18]
        e_pose = jp.sum((self.ref(info, "joints") - q) ** 2)
        e_vel = jp.sum((self.ref(info, "joint_vel") - qd) ** 2)
        e_paw = jp.sum((self.ref(info, "paws") - self.heading_paws(d)) ** 2)
        e_h = self.ref(info, "height") - (d.qpos[2] if self.terrain is None else d.qpos[2] - self.ground_z(d))
        g, g_ref = R.T @ jp.array([0.0, 0.0, -1.0]), self.ref(info, "gravity")
        tilt = jp.arccos(jp.clip(jp.dot(g, g_ref) / jp.linalg.norm(g_ref), -1.0, 1.0))
        v, w = R.T @ d.qvel[0:3], d.qvel[3:6]
        e_v = jp.sum((self.ref(info, "v_body") - v) ** 2)
        e_w = jp.sum((self.ref(info, "w_body") - w) ** 2)
        terms = dict(pose=jp.exp(-5 * e_pose), vel=jp.exp(-0.1 * e_vel), paw=jp.exp(-40 * e_paw),
                     root=jp.exp(-20 * e_h ** 2 - 10 * tilt ** 2), root_vel=jp.exp(-2 * e_v - 0.2 * e_w))
        return terms, dict(height_err=e_h, tilt=tilt)

    @staticmethod
    def contact(d):
        return d._impl.contact if hasattr(d, "_impl") and hasattr(d._impl, "contact") else d.contact

    def ground_z(self, d):
        """Height of the ground under the body (the terrain's, 0 off it; imitate_env.ImitateEnv.ground_z)."""
        return terrain_height(self.terrain["field"], self.terrain["cm"], d.qpos[0], d.qpos[1])

    def body_on_floor(self, d):
        c = self.contact(d)
        if self.terrain is not None:   # the terrain is floor too
            tg = self.terrain["geom"]
            g0f, g1f = (c.geom[:, 0] == self.floor) | (c.geom[:, 0] == tg), (c.geom[:, 1] == self.floor) | (c.geom[:, 1] == tg)
            other = jp.where(g0f, c.geom[:, 1], c.geom[:, 0])
            return jp.any((c.dist < 0.0) & (g0f | g1f) & self.body_mask[other])
        other = jp.where(c.geom[:, 0] == self.floor, c.geom[:, 1], c.geom[:, 0])
        with_floor = (c.dist < 0.0) & ((c.geom[:, 0] == self.floor) | (c.geom[:, 1] == self.floor))
        return jp.any(with_floor & self.body_mask[other])

    # ---------- episode ----------
    @staticmethod
    def mat2quat(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        k = jp.argmax(jp.array([tr, R[0, 0], R[1, 1], R[2, 2]]))
        s0 = 2 * jp.sqrt(jp.maximum(1 + tr, 1e-12))
        s1 = 2 * jp.sqrt(jp.maximum(1 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
        s2 = 2 * jp.sqrt(jp.maximum(1 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
        s3 = 2 * jp.sqrt(jp.maximum(1 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
        Q = jp.array([[0.25 * s0, (R[2, 1] - R[1, 2]) / s0, (R[0, 2] - R[2, 0]) / s0, (R[1, 0] - R[0, 1]) / s0],
                      [(R[2, 1] - R[1, 2]) / s1, 0.25 * s1, (R[0, 1] + R[1, 0]) / s1, (R[0, 2] + R[2, 0]) / s1],
                      [(R[0, 2] - R[2, 0]) / s2, (R[0, 1] + R[1, 0]) / s2, 0.25 * s2, (R[1, 2] + R[2, 1]) / s2],
                      [(R[1, 0] - R[0, 1]) / s3, (R[0, 2] + R[2, 0]) / s3, (R[1, 2] + R[2, 1]) / s3, 0.25 * s3]])
        q = Q[k]
        q = q * jp.where(q[0] < 0, -1.0, 1.0)
        return q / jp.linalg.norm(q)

    def limit_for(self, c, tau0, t):
        """Last step of the try: loops run 5 s, sit/lie/getup to the end of their 1 s hold (at least 25 steps)."""
        left = jp.floor((self.length[c] + HOLD - tau0) / self.dt)
        return jp.where(self.loop[c], self.max_steps, jp.maximum(25, jp.minimum(self.max_steps, left)).astype(jp.int32)) + t

    def start_info(self, rng):
        if self.terrain is not None:   # terrain on: the start spot's own key
            rng, k_ter = jax.random.split(rng)
        if self.real_on:   # real-robot gaps on: the latency's own key
            rng, k_lat = jax.random.split(rng)
        if self.L is not None:   # landing starts on: their own keys, the rest drawn as always
            rng, k_land = jax.random.split(rng)
        if self.M is not None:   # moving starts on: the same
            rng, k_move = jax.random.split(rng)
        k = jax.random.split(rng, 6)
        one = jp.float32(1.0)
        info = dict(kp=jax.random.uniform(k[0], minval=0.85, maxval=1.15) if self.randomize else one,
                    mass=jax.random.uniform(k[1], minval=0.85, maxval=1.15) if self.randomize else one,
                    fric=jax.random.uniform(k[2], minval=0.6, maxval=1.3) if self.randomize else one,
                    clip=jax.random.categorical(k[3], self.clip_logits).astype(jp.int32), t=jp.int32(0),
                    last_action=jp.zeros(12), trunc=jp.float32(0.0))
        if self.transition_weight:
            info["swt"] = jp.int32(-10 ** 6)   # step of the last mid-try switch (none yet)
        if self.real_on:   # latency in physics substeps, drawn per try
            ms = jax.random.uniform(k_lat, minval=self.latency_ms[0], maxval=self.latency_ms[1]) if self.latency_ms[1] > 0 else jp.float32(0.0)
            info["dsub"] = jp.round(ms / 1000.0 / self.sub_dt).astype(jp.int32)
        c = info["clip"]
        info["tau0"] = jax.random.uniform(k[4]) * jp.where(self.loop[c], self.length[c], self.length[c] + 0.3 * HOLD)
        info["limit"] = self.limit_for(c, info["tau0"], 0)
        yaw = jax.random.uniform(k[5], minval=-jp.pi, maxval=jp.pi)
        if self.L is not None:
            kl = jax.random.split(k_land)
            land = jax.random.uniform(kl[0]) < self.landing_rate
            idx = jax.random.randint(kl[1], (), 0, self.L["tau0"].shape[0])
            info = self.landing_info(info, jp.where(land, idx, -1))
        if self.M is not None:
            km = jax.random.split(k_move, 4)
            on = jax.random.uniform(km[0]) < self.moving_rate
            if self.L is not None:
                on = on & (info["land"] < 0)
            post = self.M["posts"][jax.random.randint(km[1], (), 0, self.M["posts"].shape[0])]
            v = jax.random.uniform(km[2], minval=self.moving_speed[0], maxval=self.moving_speed[1])
            gi = jp.argmin(jp.abs(self.M["speed"] - v))   # the gait at about that pace
            g = self.M["gaits"][gi]
            info = self.moving_info(info, on, post, g, jax.random.uniform(km[3]) * self.length[g], v / self.M["speed"][gi])
        if self.terrain is not None:   # walk/trot tries (not landing or moving starts) start on the rough field
            on = self.terrain["clips"][info["clip"]]
            if self.L is not None:
                on = on & (info["land"] < 0)
            if self.M is not None:
                on = on & (info["mg"] < 0)
            w = self.terrain["spread"]
            xy = jp.array(TERRAIN["at"], jp.float32) + jax.random.uniform(k_ter, (2,), minval=-w, maxval=w)
            info = self.terrain_info(info, on, xy)
        return info, yaw

    def terrain_info(self, info, on, xy):
        return dict(info, ton=jp.asarray(on), txy=jp.where(on, xy, 0.0).astype(jp.float32))

    def moving_info(self, info, on, post, g, tau_g, vscale):
        """info for a moving start (on): Bolt in gait g at tau_g, body speed times vscale, told to do posture clip post
        (joined at its closest-pose moment, imitate_env.closest_moment's rule); off: info unchanged but for the keys."""
        gi = dict(info, clip=g, tau0=tau_g, t=jp.int32(0))
        q, qd = jp.clip(self.ref(gi, "joints"), self.lo, self.hi), self.ref(gi, "joint_vel")
        e = jp.sum((self.T["joints"][post] - q) ** 2, axis=1) + 0.002 * jp.sum((self.T["joint_vel"][post] - qd) ** 2, axis=1)
        e = jp.where(jp.arange(e.shape[0]) < self.n[post], e, jp.inf)
        tau0 = jp.argmin(e).astype(jp.float32) / self.fps[post]
        c = jp.where(on, post, info["clip"]).astype(jp.int32)
        tau0 = jp.where(on, tau0, info["tau0"])
        until = self.M["until"][post]
        limit = jp.where(on, jp.maximum(self.limit_for(c, tau0, 0), until + 1), info["limit"])
        return dict(info, clip=c, tau0=tau0, limit=limit, mg=jp.where(on, g, -1).astype(jp.int32), mtau=jp.float32(tau_g),
                    mscale=jp.float32(vscale), pclip=jp.where(on, post, -1).astype(jp.int32), puntil=jp.where(on, until, 0).astype(jp.int32),
                    phit=jp.int32(0))

    def landing_info(self, info, idx):
        """info for landing state idx (-1: not a landing start; info unchanged but for the landing keys)."""
        land = idx >= 0
        i = jp.maximum(idx, 0)
        c = jp.where(land, self.L["clip"][i], info["clip"])
        tau0 = jp.where(land, self.L["tau0"][i], info["tau0"])
        return dict(info, clip=c, tau0=tau0, limit=jp.where(land, self.limit_for(c, tau0, 0), info["limit"]), land=idx.astype(jp.int32),
                    grace=jp.where(land, self.grace_steps, 0).astype(jp.int32), lswitch=jp.where(land, self.land_switch_at, -1).astype(jp.int32))

    def rsi(self, info, yaw):
        """Reference state initialization: (qpos, qvel, joints) of the dog's pose at that moment (any heading), moving as it was."""
        u, _, vt = jp.linalg.svd(self.ref(info, "tilt"))    # re-orthonormalize the blended rotation
        c, s = jp.cos(yaw), jp.sin(yaw)
        Rw = jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ (u @ vt)
        joints = jp.clip(self.ref(info, "joints"), self.lo, self.hi)
        qpos = self.qpos0.at[0:3].set(jp.stack([0.0, 0.0, self.ref(info, "height") + 0.005])).at[3:7].set(self.mat2quat(Rw)).at[7:19].set(joints)
        moving = jp.where(self.finished(info) > 0, 0.0, 1.0)
        qvel = jp.concatenate([Rw @ self.ref(info, "v_body") * moving, self.ref(info, "w_body") * moving, self.ref(info, "joint_vel") * moving])
        return qpos, qvel, joints

    def start_data(self, info, yaw):
        """The start state: reference state initialization, or a landing or moving start's state."""
        mx = self.model_for(info)
        qpos, qvel, joints = self.rsi(info, yaw)
        c, s = jp.cos(yaw), jp.sin(yaw)
        if self.M is not None:   # a moving start: the gait's reference state, its body speed scaled
            mv = info["mg"] >= 0
            mq, mvel, mj = self.rsi(dict(info, clip=jp.maximum(info["mg"], 0), tau0=info["mtau"], t=jp.int32(0)), yaw)
            mvel = mvel.at[0:3].multiply(info["mscale"])
            qpos, qvel, joints = jp.where(mv, mq, qpos), jp.where(mv, mvel, qvel), jp.where(mv, mj, joints)
        if self.L is not None:   # a landing start: the stored state, turned to this heading
            land, i = info["land"] >= 0, jp.maximum(info["land"], 0)
            lq, lv = self.L["qpos"][i], self.L["qvel"][i]
            Rz = jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            R0 = jp.array([[1 - 2 * (lq[5] ** 2 + lq[6] ** 2), 2 * (lq[4] * lq[5] - lq[3] * lq[6]), 2 * (lq[4] * lq[6] + lq[3] * lq[5])],
                           [2 * (lq[4] * lq[5] + lq[3] * lq[6]), 1 - 2 * (lq[4] ** 2 + lq[6] ** 2), 2 * (lq[5] * lq[6] - lq[3] * lq[4])],
                           [2 * (lq[4] * lq[6] - lq[3] * lq[5]), 2 * (lq[5] * lq[6] + lq[3] * lq[4]), 1 - 2 * (lq[4] ** 2 + lq[5] ** 2)]])
            lj = jp.clip(lq[7:19], self.lo, self.hi)
            lqpos = self.qpos0.at[2].set(lq[2]).at[3:7].set(self.mat2quat(Rz @ R0)).at[7:19].set(lj)
            lqvel = jp.concatenate([Rz @ lv[0:3], lv[3:18]])
            qpos, qvel, joints = jp.where(land, lqpos, qpos), jp.where(land, lqvel, qvel), jp.where(land, lj, joints)
        if self.terrain is not None:   # on the rough field: that spot, at the ground's height there
            xy = info["txy"]
            gz = terrain_height(self.terrain["field"], self.terrain["cm"], xy[0], xy[1])
            qpos = jp.where(info["ton"], qpos.at[0:2].set(xy).at[2].add(gz), qpos)
        d = mjx.make_data(mx).replace(qpos=qpos, qvel=qvel, ctrl=joints)
        d = mjx.forward(mx, d)
        pen = jp.maximum(0.0, jp.max(-self.contact(d).dist))    # never start with a paw or the body in the floor
        if self.L is not None:   # landing starts: floor contacts only (as ImitateEnv; legs touching each other stay)
            cg = self.contact(d).geom
            on_floor = (cg[:, 0] == self.floor) | (cg[:, 1] == self.floor)
            pen = jp.where(land, jp.maximum(0.0, jp.max(jp.where(on_floor, -self.contact(d).dist, 0.0))), pen)
        d = d.replace(qpos=d.qpos.at[2].add(jp.where(pen > 0, pen + 0.002, 0.0)))
        return mjx.forward(mx, d)

    def reset(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        info, yaw = self.start_info(k1)
        d = self.start_data(info, yaw)
        if self.real_on:
            info = self.real_init(info, d)
        info["rng"] = k2
        z = jp.float32(0.0)
        metrics = {**{f"r_{k}": z for k in WEIGHTS}, "fell": z, "canter": z, "canter_steps": z}
        return self.State(pipeline_state=d, obs=self.obs(d, info, k3), reward=z, done=z, metrics=metrics, info=info)

    def pose_term(self, d, info):
        """(bonus, reached in time so far, in the pose now) of a moving-start command (pose_reached.pose_ok)."""
        R = d.xmat[self.torso]
        h, pitch, up = d.qpos[2], jp.arcsin(jp.clip(R[2, 0], -1.0, 1.0)), R[2, 2]
        oks = jp.stack([self.pose_ok(n, h, pitch, up) for n in self.pose_names])
        pc = info["pclip"]
        pose = (pc >= 0) & (info["clip"] == pc) & oks[jp.maximum(self.M["pose"][jp.maximum(pc, 0)], 0)]
        hit = (info["phit"] > 0) | (pose & (info["t"] <= info["puntil"]))
        return self.pose_weight * (hit & pose).astype(jp.float32), hit, pose

    def transition_excess(self, d, info):
        """Roll and pitch (rad) beyond the threshold, each against the reference's own, summed (imitate_env.roll_pitch)."""
        R, Rr = d.xmat[self.torso], self.ref(info, "tilt")
        rp = lambda M: (jp.arcsin(jp.clip(M[2, 1], -1.0, 1.0)), jp.arcsin(jp.clip(M[2, 0], -1.0, 1.0)))
        (r, p), (r0, p0) = rp(R), rp(Rr)
        th = self.transition_threshold
        return jp.maximum(jp.abs(r - r0) - th, 0.0) + jp.maximum(jp.abs(p - p0) - th, 0.0)

    def switch_to(self, info, q, c):
        """Hand over to clip c at step info['t']: a gait joins at the frame whose pose is closest to the current one."""
        dist = jp.sum((self.T["joints"][c] - q) ** 2, axis=1)
        dist = jp.where(jp.arange(dist.shape[0]) < self.n[c], dist, jp.inf)
        start = jp.where(self.loop[c], jp.argmin(dist).astype(jp.float32) / self.fps[c], 0.0)
        tau0 = start - info["t"].astype(jp.float32) * self.dt
        left = jp.where(self.loop[c], float(self.max_steps), jp.floor((self.length[c] + HOLD - start) / self.dt))
        limit = info["t"] + jp.minimum(left, self.max_steps).astype(jp.int32)
        return tau0, limit

    def step(self, state, action):
        info, d = dict(state.info), state.pipeline_state
        rng, k_sw, k_pick, k_obs, k_new, k_obs2 = jax.random.split(info["rng"], 6)
        mx = self.model_for(info)
        a = jp.clip(action, -1.0, 1.0)
        d = d.replace(ctrl=jp.clip(self.ref(info, "joints", 1) + self.scale * a, self.lo, self.hi))
        if self.push_steps:   # now and then a sideways push (imitate_env: the switch test's shove, push-sized)
            rng, k_push, k_side = jax.random.split(rng, 3)
            go = (info["t"] > 0) & (info["t"] % self.push_steps == 0) & (jax.random.uniform(k_push) < self.push_prob)
            side = jp.where(jax.random.uniform(k_side) < 0.5, 1.0, -1.0)
            d = d.replace(qvel=jp.where(go, apply_shove(d.qvel, d.xmat[self.torso], self.push_speed, self.push_spin, side), d.qvel))
        if self.real_on:
            d, info = self.physics_real(mx, d, info)
        else:
            d = self.physics(mx, d)
        info["t"] = info["t"] + 1
        if self.switch_rate:   # now and then hand over to a clip that may follow this one (NEXT)
            c = info["clip"]
            allowed = self.next[c]
            ready = self.loop[c] | (self.finished(info) > 0.5 * HOLD)
            go = ready & jp.any(allowed) & (jax.random.uniform(k_sw) < self.switch_rate * self.dt)
            nc = jax.random.categorical(k_pick, jp.where(allowed, 0.0, -1e9)).astype(jp.int32)
            tau0, limit = self.switch_to(info, d.qpos[7:19], nc)
            info.update(clip=jp.where(go, nc, c), tau0=jp.where(go, tau0, info["tau0"]), limit=jp.where(go, limit, info["limit"]))
            if self.transition_weight:
                info["swt"] = jp.where(go, info["t"], info["swt"])
        if self.L is not None and self.land_then >= 0:   # a landing start's bridge is over: on in landing_then
            go = (info["t"] == info["lswitch"]) & (info["clip"] != self.land_then)
            tau0, limit = self.switch_to(info, d.qpos[7:19], jp.int32(self.land_then))
            info.update(clip=jp.where(go, self.land_then, info["clip"]).astype(jp.int32), tau0=jp.where(go, tau0, info["tau0"]),
                        limit=jp.where(go, limit, info["limit"]))
            if self.transition_weight:
                info["swt"] = jp.where(go, info["t"], info["swt"])
        terms, err = self.tracking(d, info)
        reward = sum(self.weights[k] * v for k, v in terms.items())
        floor_ok = self.ref(info, "height") < 0.3     # sitting and lying: the body may rest on the floor
        high = jp.abs(err["height_err"]) > 0.15
        if self.L is not None:   # landing starts: no height check for the first landing_grace s (still in the air)
            high = high & (info["t"] > info["grace"])
        if self.M is not None:   # a moving-start command: paid per step in the pose once reached in time
            bonus, hit, _ = self.pose_term(d, info)
            reward = reward + bonus
            info["phit"] = hit.astype(jp.int32)
        if self.transition_weight:   # after a mid-try switch: roll and pitch beyond the threshold cost
            on = info["t"] - info["swt"] < self.transition_steps
            reward = reward - self.transition_weight * jp.where(on, self.transition_excess(d, info), 0.0)
        lost = high | (err["tilt"] > 0.8)
        fell = lost | (self.body_on_floor(d) & ~floor_ok)
        trunc = (info["t"] >= info["limit"]) & ~fell
        done = fell | trunc
        on_canter = (info["clip"] == self.canter).astype(jp.float32)
        info.update(last_action=a, trunc=trunc.astype(jp.float32), rng=rng)
        obs = self.obs(d, info, k_obs)
        # a finished try starts afresh: a new clip, moment, heading and physics (as ImitateEnv.reset)
        new_info, yaw = self.start_info(k_new)
        new_d = self.start_data(new_info, yaw)
        if self.real_on:
            new_info = self.real_init(new_info, new_d)
        new_info.update(rng=rng, trunc=info["trunc"])
        new_obs = self.obs(new_d, new_info, k_obs2)
        sel = lambda x, y: jp.where(done, x, y)
        d = jax.tree_util.tree_map(sel, new_d, d)
        info.update({k: jax.tree_util.tree_map(sel, v, info[k]) for k, v in new_info.items()})   # (wrappers' keys untouched)
        obs = sel(new_obs, obs)
        metrics = dict(state.metrics)   # keep what the training wrappers added
        metrics.update({f"r_{k}": v for k, v in terms.items()}, fell=fell.astype(jp.float32), canter=on_canter * reward, canter_steps=on_canter)
        return state.replace(pipeline_state=d, obs=obs, reward=reward, done=done.astype(jp.float32), metrics=metrics, info=info)


def wrap_for_training(env, episode_length=1000, action_repeat=1, randomization_fn=None):
    """Brax's training wrappers, minus the reset to the first state: the env restarts finished tries itself, and its
    own time-outs (info['trunc']) are passed on as truncations. episode_length only sets the eval rollout length."""
    from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

    class Wrap(EpisodeWrapper):
        def step(self, state, action):
            state.info.update(steps=jp.where(state.done, jp.zeros_like(state.info["steps"]), state.info["steps"]))
            state = super().step(state.replace(done=jp.zeros_like(state.done)), action)
            state.info["truncation"] = state.info["trunc"]
            return state

    return Wrap(VmapWrapper(env), 10 ** 9, action_repeat)


def make_networks(hidden=(512, 256), log_std=-1.0):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    def factory(observation_size, action_size, preprocess_observations_fn=None, **_):
        return ppo_networks.make_ppo_networks(
            observation_size, action_size, preprocess_observations_fn=functools.partial(running_statistics.normalize, max_abs_value=10.0),
            policy_hidden_layer_sizes=tuple(hidden), value_hidden_layer_sizes=tuple(hidden), activation=jax.nn.elu,
            distribution_type="normal", noise_std_type="log", init_noise_std=float(np.exp(log_std)))
    return factory


def to_brax(P, count=1e8):
    """numpy brain -> Brax PPO params (normalizer state, policy, value) for ppo.train(restore_params=...)."""
    from brax.training import types
    from brax.training.acme import running_statistics, specs
    n = len(P["mean"])
    var = np.asarray(P.get("var", np.asarray(P["std"], np.float64) ** 2), np.float32)
    st = running_statistics.init_state(specs.Array((n,), jp.dtype("float32")), mode="welford")
    st = st.replace(mean=jp.array(P["mean"], jp.float32), std=jp.array(P["std"], jp.float32),
                    summed_variance=jp.array(var * np.float32(count), jp.float32), count=types.UInt64(hi=int(count) >> 32, lo=int(count) & 0xFFFFFFFF))
    hid = lambda layers: {f"hidden_{i}": dict(kernel=jp.array(W), bias=jp.array(b)) for i, (W, b) in enumerate(layers)}
    policy = {"params": {"MLP_0": hid(P["layers"][:-1]), "Dense_0": dict(kernel=jp.array(P["layers"][-1][0]), bias=jp.array(P["layers"][-1][1])),
                         "std_logparam": {"log_value": jp.array(P["log_std"])}}}
    value = {"params": hid(P["value"])}
    return st, policy, value


def to_numpy(params, extra=None):
    """Brax PPO params (normalizer, policy, value) -> the numpy brain (params.pkl)."""
    st, pol, val = params
    p, v = pol["params"], val["params"]
    hidden = sorted(p["MLP_0"], key=lambda s: int(s.split("_")[-1]))
    layers = [(np.asarray(p["MLP_0"][k]["kernel"]), np.asarray(p["MLP_0"][k]["bias"])) for k in hidden]
    layers.append((np.asarray(p["Dense_0"]["kernel"]), np.asarray(p["Dense_0"]["bias"])))
    vl = [(np.asarray(v[k]["kernel"]), np.asarray(v[k]["bias"])) for k in sorted(v, key=lambda s: int(s.split("_")[-1]))]
    count = float(np.asarray(st.count.hi, np.float64) * 2.0 ** 32 + np.asarray(st.count.lo, np.float64)) if hasattr(st.count, "hi") else float(st.count)
    return dict(mean=np.asarray(st.mean), std=np.asarray(st.std), var=np.asarray(st.summed_variance) / max(count, 1.0), count=count,
                layers=layers, log_std=np.asarray(p["std_logparam"]["log_value"]), value=vl,
                hidden=tuple(int(W.shape[1]) for W, _ in layers[:-1]), **(extra or {}))


def curriculum(levels, at, steps):
    """The terrain curriculum: [(terrain_cm, steps)] phases, level i from step fraction at[i] to at[i+1] (the last to
    the end). levels (0 5 10) and at (0 0.3 0.6) must have the same length, at[0] = 0, increasing."""
    assert len(levels) == len(at) and at[0] == 0 and all(b > a for a, b in zip(at, at[1:])) and at[-1] < 1, (levels, at)
    ends = list(at[1:]) + [1.0]
    return [(float(cm), int(round((b - a) * steps))) for cm, a, b in zip(levels, at, ends)]


def train(args):
    from brax.training.agents.ppo import train as ppo
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    W = pickle.loads(Path(args.warm).read_bytes()) if args.warm else None

    def make_env(cm):
        return MjxImitate(switch_rate=args.switch_rate, landing_rate=args.landing_rate, landing_file=args.landing_file,
                          landing_grace=args.landing_grace, landing_bridge=args.landing_bridge, landing_then=args.landing_then,
                          moving_rate=args.moving_rate, moving_speed=args.moving_speed, moving_gaits=args.moving_gaits, moving_clips=args.moving_clips,
                          pose_time={k: float(v) for k, v in (kv.split("=") for kv in args.pose_time)},
                          pose_weight=args.pose_weight, transition_weight=args.transition_weight, transition_window=args.transition_window,
                          transition_threshold=args.transition_threshold, latency_ms=(args.latency_min_ms, args.latency_max_ms),
                          obs_noise=args.obs_noise, motor_lag_ms=args.motor_lag_ms, terrain_cm=cm, terrain_clips=args.terrain_clips,
                          terrain_spread=args.terrain_spread, terrain_seed=args.terrain_seed, push_interval=args.push_interval,
                          push_prob=args.push_prob, push_speed=args.push_speed, push_spin=args.push_spin,
                          motor_brake=args.motor_brake, brake_speed=args.brake_speed, brake_gain=args.brake_gain)

    phases = curriculum(args.terrain_cm, args.terrain_at, args.steps)
    hidden = W["hidden"] if W else (512, 256)
    reward_scale = args.reward_scaling or (W["reward_scale"] if W else 1.0)
    start, log, base0 = time.time(), [], int(W["steps"]) if W else 0
    done_steps = 0   # steps of the finished phases
    q = args.quick
    try:
        for k, (cm, n_steps) in enumerate(phases):
            env = make_env(cm)
            if k == 0:
                print(f"motor brake: {'on' if env.brake else 'off'}" + (f" above {env.brake[0]:g} rad/s, {env.brake[1]:g} N·m per rad/s" if env.brake else ""), flush=True)
            if len(phases) > 1:
                print(f"phase {k + 1}/{len(phases)}: terrain {cm:g} cm for {n_steps:,} steps", flush=True)
            base = base0 + done_steps
            restore = to_brax(W) if W else None
            last = [0]   # the steps this phase actually ran (Brax rounds up to whole iterations)

            def progress(n, m, cm=cm, base=base, off=done_steps):
                L = max(1.0, float(m.get("eval/avg_episode_length", 1.0)))
                last[0] = max(last[0], int(n))
                row = dict(steps=off + int(n), total=base + int(n), terrain_cm=cm, minutes=round((time.time() - start) / 60, 1), length=round(L, 1),
                           reward_per_step=float(m.get("eval/episode_reward", np.nan)) / L,
                           fell=float(m.get("eval/episode_fell", np.nan)),
                           canter=float(m.get("eval/episode_canter", np.nan)) / max(1e-6, float(m.get("eval/episode_canter_steps", np.nan))),
                           sps=float(m.get("training/sps", np.nan)), **{t: float(m.get(f"eval/episode_r_{t}", np.nan)) / L for t in WEIGHTS})
                log.append(row)
                (out / "progress.json").write_text(json.dumps(log, indent=1))
                print(f"update at {off + int(n)}: reward/step {row['reward_per_step']:.3f} (canter {row['canter']:.3f}), try {L:.0f} steps, fell {row['fell']:.2f}, "
                      f"pose {row['pose']:.2f}, {row['sps']:.0f} steps/s, {row['minutes']} min" + (f", terrain {cm:g} cm" if len(phases) > 1 else ""), flush=True)
                if args.hours and time.time() - start > args.hours * 3600:
                    raise TimeoutError(f"time budget of {args.hours} h used up at {off + int(n)} steps")

            def save(step, make_policy, params, base=base, off=done_steps, cm=cm):
                P = to_numpy(params, dict(steps=base + int(step), source="mjx_imitate", terrain_cm=cm))
                (out / "checkpoints" / f"params_{off + int(step)}.pkl").write_bytes(pickle.dumps(P))
                (out / "params.pkl").write_bytes(pickle.dumps(P))

            evals = args.evals if len(phases) == 1 else max(2, int(round(args.evals * n_steps / args.steps)))
            _, params, _ = ppo.train(
                environment=env, num_timesteps=n_steps, episode_length=env.max_steps, num_envs=args.envs, num_eval_envs=min(args.envs, 128),
                wrap_env_fn=wrap_for_training, learning_rate=args.lr, learning_rate_schedule="ADAPTIVE_KL", desired_kl=args.kl,
                learning_rate_schedule_max_lr=1e-3, entropy_cost=args.entropy, discounting=0.95, gae_lambda=0.95, unroll_length=20,
                batch_size=min(64, args.envs) if q else 256, num_minibatches=4 if q else 32, num_updates_per_batch=4 if q else 5,
                normalize_observations=True, reward_scaling=reward_scale, clipping_epsilon=0.2, max_grad_norm=1.0,
                num_evals=evals, network_factory=make_networks(hidden, float(np.mean(W["log_std"])) if W else -1.0),
                restore_params=restore, restore_value_fn=True, seed=args.seed + k, deterministic_eval=True,
                progress_fn=progress, policy_params_fn=save)
            ran = max(last[0], n_steps)
            W = to_numpy(params, dict(steps=base + ran, source="mjx_imitate", terrain_cm=cm))   # the next phase starts here
            (out / "params.pkl").write_bytes(pickle.dumps(W))
            done_steps += ran
    except TimeoutError as e:
        print(f"stopped: {e}; the last saved brain is {out / 'params.pkl'}", flush=True)
    print(f"done: {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm", help="SB3 brain -> numpy params.pkl (run with the CPU stack)")
    w.add_argument("zip")
    w.add_argument("out")
    w.add_argument("--stats", help="VecNormalize pkl (default: the checkpoint's policy_vecnormalize_<n>_steps.pkl)")
    t = sub.add_parser("train")
    t.add_argument("--warm", help="numpy brain to start from (from `warm`, or a params.pkl from an earlier GPU run)")
    t.add_argument("--steps", type=int, default=1_000_000_000)
    t.add_argument("--envs", type=int, default=4096)
    t.add_argument("--out", default=str(HERE / "runs" / "imitate_gpu"))
    t.add_argument("--evals", type=int, default=21, help="progress reports and saved brains over the run (21: every 5%%)")
    t.add_argument("--switch-rate", type=float, default=0.3)
    t.add_argument("--lr", type=float, default=1e-4, help="start; then adapted to keep the KL near --kl")
    t.add_argument("--kl", type=float, default=0.01)
    t.add_argument("--entropy", type=float, default=0.0, help="as SB3's ent_coef in train_imitate.py")
    t.add_argument("--reward-scaling", type=float, help="default: 1/std of the CPU run's returns (VecNormalize), so the critic fits")
    t.add_argument("--hours", type=float, help="stop after this many hours (the last saved brain stays)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--quick", action="store_true")
    t.add_argument("--landing-rate", type=float, default=LANDING["rate"],
                   help="share of tries that start in the run-up jump brain's landing (mid-air to touch-down); 0: none, as before")
    t.add_argument("--landing-file", default=str(LANDING_FILE), help="landing states (research/gen_landing_starts.py)")
    t.add_argument("--landing-grace", type=float, default=LANDING["grace"],
                   help="s after a landing start without the 15 cm height check (tilt and body-on-floor stay on)")
    t.add_argument("--landing-bridge", type=float, default=LANDING["bridge"],
                   help="s of the landing state's clip (canter) before switching to --landing-then (skills.BRIDGE); 0: no switch")
    t.add_argument("--landing-then", default=LANDING["then"], help="clip after the bridge (test_skills: trot)")
    t.add_argument("--moving-rate", type=float, default=MOVING["rate"],
                   help="share of tries that start moving and are told at once to sit, lie or get up; 0: none, as before")
    t.add_argument("--moving-speed", type=float, nargs=2, default=list(MOVING["speed"]), metavar=("LO", "HI"),
                   help="body speed range (m/s) of a moving start (default %(default)s; walk clip 1.1, trot 1.9)")
    t.add_argument("--moving-gaits", nargs="+", default=list(MOVING["gaits"]), help="gaits a moving start is taken from (the one nearest the speed)")
    t.add_argument("--moving-clips", nargs="+", default=list(MOVING["clips"]), help="posture clips commanded on a moving start")
    t.add_argument("--pose-time", nargs="+", default=[f"{k}={v:g}" for k, v in MOVING["pose_time"].items()], metavar="CLIP=S",
                   help="s from the command to reach the pose, per clip (default %(default)s: the switch test's times)")
    t.add_argument("--pose-weight", type=float, default=MOVING["weight"], help="reward per step in the pose once reached in time")
    t.add_argument("--transition-weight", type=float, default=TRANSITION["weight"],
                   help="cost per rad of roll/pitch beyond --transition-threshold in the window after a clip switch; 0: off, as before")
    t.add_argument("--transition-window", type=float, default=TRANSITION["window"], help="s after a clip switch the term is on (default %(default)s)")
    t.add_argument("--transition-threshold", type=float, default=TRANSITION["threshold"],
                   help="rad of roll or pitch away from the reference's that is free (default %(default)s)")
    t.add_argument("--latency-min-ms", type=float, default=REAL["latency_ms"][0], help="action latency per try, uniform from this (ms)")
    t.add_argument("--latency-max-ms", type=float, default=REAL["latency_ms"][1],
                   help="... to this (ms); 0: no latency, as before (mastermind's stress test: 10/20/40 ms)")
    t.add_argument("--obs-noise", type=float, default=REAL["obs_noise"], help="sensor noise, x legged_gym's levels (NOISE); 0: none")
    t.add_argument("--motor-lag-ms", type=float, default=REAL["motor_lag_ms"], help="first-order motor lag time constant (ms); 0: none")
    t.add_argument("--terrain-cm", type=float, nargs="+", default=[TERRAIN_ARGS["cm"]],
                   help="rough terrain height (cm), or several for a curriculum with --terrain-at (e.g. 0 5 10); 0: flat, as before")
    t.add_argument("--terrain-at", type=float, nargs="+", default=[0.0], help="step fraction where each --terrain-cm level starts (e.g. 0 0.3 0.6)")
    t.add_argument("--terrain-clips", nargs="+", default=list(TERRAIN_ARGS["clips"]), help="clips whose tries start on the terrain")
    t.add_argument("--terrain-spread", type=float, default=TERRAIN_ARGS["spread"], help="m around the field's middle where they start")
    t.add_argument("--terrain-seed", type=int, default=TERRAIN_ARGS["seed"], help="the field's random heights")
    t.add_argument("--push-interval", type=float, default=PUSH["interval"], help="s between push chances; 0: no pushes, as before")
    t.add_argument("--push-prob", type=float, default=PUSH["prob"], help="chance of a push at each (default %(default)s)")
    t.add_argument("--push-speed", type=float, default=PUSH["speed"], help="sideways push, m/s of body speed (default %(default)s, legged_gym's max)")
    t.add_argument("--push-spin", type=float, default=PUSH["spin"], help="roll rate the push adds, rad/s (default %(default)s)")
    t.add_argument("--motor-brake", action=argparse.BooleanOptionalAction, default=True,
                   help="motor braking above --brake-speed (bolt_env back-EMF model); default on for training")
    t.add_argument("--brake-speed", type=float, default=BRAKE["speed"], help="rad/s where braking starts (default %(default)s, the drive cutoff)")
    t.add_argument("--brake-gain", type=float, default=BRAKE["gain"], help="braking N·m per rad/s above --brake-speed (default %(default)s)")
    args = ap.parse_args()
    if args.cmd == "warm":
        z = Path(args.zip)
        stats = Path(args.stats) if args.stats else z.with_name(z.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
        P = from_sb3(z, stats)
        Path(args.out).write_bytes(pickle.dumps(P))
        print(f"{z} ({P['steps']:,} steps, hidden {P['hidden']}, {len(P['mean'])} inputs, reward scale {P['reward_scale']:.4f}) -> {args.out}")
    else:
        train(args)


if __name__ == "__main__":
    main()
