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
ALL_CLIPS = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"]
WEIGHTS = dict(pose=0.5, vel=0.05, paw=0.2, root=0.15, root_vel=0.1)
HOLD, LOOP_SECONDS = 1.0, 5.0
GAITS = ("walk", "trot", "pace", "canter", "turn_left", "turn_right")
NEXT = {**{g: GAITS + ("sit",) for g in GAITS}, "sit": ("lie",), "lie": ("getup",), "getup": ("walk", "trot")}
CLIP_WEIGHTS = {"canter": 3.0}   # how often each clip is drawn (1 for the rest): the canter matters most
KEYS = ("joints", "joint_vel", "height", "gravity", "v_body", "w_body", "paws", "tilt")


def floor_only(xml):
    """Robot geoms collide with the floor only: robot contype 2 / conaffinity 1, floor contype 1 / conaffinity 2."""
    xml = re.sub(r'(<geom name="floor"[^>]*?)(/?>)', r'\1 contype="1" conaffinity="2"\2', xml, count=1)
    return xml.replace('<geom friction="1.0 0.02 0.01" condim="3"', '<geom friction="1.0 0.02 0.01" condim="3" contype="2" conaffinity="1"', 1)


def collisions(xml):
    """floor_only, plus the contacts of the thighs with the torso, which the CPU env has in about a third of all steps
    (pushing at ~130 N): leg-on-leg contacts never happen there, so only these were missing."""
    xml = floor_only(xml)
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
                 iterations=5, ls_iterations=8):
        from brax.envs.base import State
        self.State = State
        xml = collisions((HERE / "bolt.xml").read_text())
        self.m = m = mujoco.MjModel.from_xml_string(xml)
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
        self.body_mask = jp.array([n[:2] not in ("FL", "FR", "RL", "RR") and g != self.floor for g, n in enumerate(bodies)])
        self.lo, self.hi = jp.array(m.actuator_ctrlrange[:, 0]), jp.array(m.actuator_ctrlrange[:, 1])
        self.kp0, self.fric0, self.mass0 = jp.array(m.actuator_gainprm[:, 0]), jp.array(m.geom_friction[:, 0]), float(m.body_mass[self.torso])
        self.qpos0 = jp.array(m.qpos0, jp.float32)
        self.dt = float(m.opt.timestep * 10)
        self.max_steps = int(LOOP_SECONDS * 50)
        self.dog, self.scale = jp.array(DOG, jp.float32), jp.array(ACTION_SCALE, jp.float32)

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

    def physics(self, mx, d):
        def sub(d, _):
            w = jp.abs(d.qvel[6:18])
            drive = jp.minimum(MOTOR["torque"], MOTOR["power"] / jp.maximum(w, 1e-6))
            drive = drive * jp.clip((MOTOR["cutoff"] - w) / (MOTOR["cutoff"] - MOTOR["speed"]), 0.0, 1.0)
            lo = jp.where(d.qvel[6:18] < 0, -drive, -MOTOR["torque"])
            hi = jp.where(d.qvel[6:18] >= 0, drive, MOTOR["torque"])
            return mjx.step(mx.replace(actuator_forcerange=jp.stack([lo, hi], -1)), d), None
        return jax.lax.scan(sub, d, None, length=10)[0]

    # ---------- senses and scores ----------
    def obs(self, d, info, key=None):
        R = d.xmat[self.torso]
        ph = 2 * jp.pi * self.phase(info)
        onehot = jp.zeros(len(ALL_CLIPS)).at[self.slot[info["clip"]]].set(1.0)
        o = jp.concatenate([R.T @ jp.array([0.0, 0.0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3], d.qpos[2:3],
                            d.qpos[7:19] - self.dog, d.qvel[6:18] * 0.1, info["last_action"], jp.stack([jp.sin(ph), jp.cos(ph)]), onehot,
                            self.ref(info, "joints", 1) - self.dog, self.ref(info, "height", 1)[None], self.ref(info, "v_body", 1),
                            self.ref(info, "w_body", 1)[2:3], self.finished(info)[None]])
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
        e_h = self.ref(info, "height") - d.qpos[2]
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

    def body_on_floor(self, d):
        c = self.contact(d)
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
        k = jax.random.split(rng, 6)
        one = jp.float32(1.0)
        info = dict(kp=jax.random.uniform(k[0], minval=0.85, maxval=1.15) if self.randomize else one,
                    mass=jax.random.uniform(k[1], minval=0.85, maxval=1.15) if self.randomize else one,
                    fric=jax.random.uniform(k[2], minval=0.6, maxval=1.3) if self.randomize else one,
                    clip=jax.random.categorical(k[3], self.clip_logits).astype(jp.int32), t=jp.int32(0),
                    last_action=jp.zeros(12), trunc=jp.float32(0.0))
        c = info["clip"]
        info["tau0"] = jax.random.uniform(k[4]) * jp.where(self.loop[c], self.length[c], self.length[c] + 0.3 * HOLD)
        info["limit"] = self.limit_for(c, info["tau0"], 0)
        yaw = jax.random.uniform(k[5], minval=-jp.pi, maxval=jp.pi)
        return info, yaw

    def start_data(self, info, yaw):
        """Reference state initialization: the dog's pose at that moment (any heading), moving as it was."""
        mx = self.model_for(info)
        u, _, vt = jp.linalg.svd(self.ref(info, "tilt"))    # re-orthonormalize the blended rotation
        c, s = jp.cos(yaw), jp.sin(yaw)
        Rw = jp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ (u @ vt)
        joints = jp.clip(self.ref(info, "joints"), self.lo, self.hi)
        qpos = self.qpos0.at[0:3].set(jp.stack([0.0, 0.0, self.ref(info, "height") + 0.005])).at[3:7].set(self.mat2quat(Rw)).at[7:19].set(joints)
        moving = jp.where(self.finished(info) > 0, 0.0, 1.0)
        qvel = jp.concatenate([Rw @ self.ref(info, "v_body") * moving, self.ref(info, "w_body") * moving, self.ref(info, "joint_vel") * moving])
        d = mjx.make_data(mx).replace(qpos=qpos, qvel=qvel, ctrl=joints)
        d = mjx.forward(mx, d)
        pen = jp.maximum(0.0, jp.max(-self.contact(d).dist))    # never start with a paw or the body in the floor
        d = d.replace(qpos=d.qpos.at[2].add(jp.where(pen > 0, pen + 0.002, 0.0)))
        return mjx.forward(mx, d)

    def reset(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        info, yaw = self.start_info(k1)
        d = self.start_data(info, yaw)
        info["rng"] = k2
        z = jp.float32(0.0)
        metrics = {**{f"r_{k}": z for k in WEIGHTS}, "fell": z, "canter": z, "canter_steps": z}
        return self.State(pipeline_state=d, obs=self.obs(d, info, k3), reward=z, done=z, metrics=metrics, info=info)

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
        terms, err = self.tracking(d, info)
        reward = sum(self.weights[k] * v for k, v in terms.items())
        floor_ok = self.ref(info, "height") < 0.3     # sitting and lying: the body may rest on the floor
        lost = (jp.abs(err["height_err"]) > 0.15) | (err["tilt"] > 0.8)
        fell = lost | (self.body_on_floor(d) & ~floor_ok)
        trunc = (info["t"] >= info["limit"]) & ~fell
        done = fell | trunc
        on_canter = (info["clip"] == self.canter).astype(jp.float32)
        info.update(last_action=a, trunc=trunc.astype(jp.float32), rng=rng)
        obs = self.obs(d, info, k_obs)
        # a finished try starts afresh: a new clip, moment, heading and physics (as ImitateEnv.reset)
        new_info, yaw = self.start_info(k_new)
        new_d = self.start_data(new_info, yaw)
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


def train(args):
    from brax.training.agents.ppo import train as ppo
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    W = pickle.loads(Path(args.warm).read_bytes()) if args.warm else None
    env = MjxImitate(switch_rate=args.switch_rate)
    hidden = W["hidden"] if W else (512, 256)
    restore = to_brax(W) if W else None
    reward_scale = args.reward_scaling or (W["reward_scale"] if W else 1.0)
    start, log, base = time.time(), [], int(W["steps"]) if W else 0

    def progress(n, m):
        L = max(1.0, float(m.get("eval/avg_episode_length", 1.0)))
        row = dict(steps=int(n), total=base + int(n), minutes=round((time.time() - start) / 60, 1), length=round(L, 1),
                   reward_per_step=float(m.get("eval/episode_reward", np.nan)) / L,
                   fell=float(m.get("eval/episode_fell", np.nan)),
                   canter=float(m.get("eval/episode_canter", np.nan)) / max(1e-6, float(m.get("eval/episode_canter_steps", np.nan))),
                   sps=float(m.get("training/sps", np.nan)), **{k: float(m.get(f"eval/episode_r_{k}", np.nan)) / L for k in WEIGHTS})
        log.append(row)
        (out / "progress.json").write_text(json.dumps(log, indent=1))
        print(f"update at {n}: reward/step {row['reward_per_step']:.3f} (canter {row['canter']:.3f}), try {L:.0f} steps, fell {row['fell']:.2f}, "
              f"pose {row['pose']:.2f}, {row['sps']:.0f} steps/s, {row['minutes']} min", flush=True)
        if args.hours and time.time() - start > args.hours * 3600:
            raise TimeoutError(f"time budget of {args.hours} h used up at {n} steps")

    def save(step, make_policy, params):
        P = to_numpy(params, dict(steps=base + int(step), source="mjx_imitate"))
        (out / "checkpoints" / f"params_{int(step)}.pkl").write_bytes(pickle.dumps(P))
        (out / "params.pkl").write_bytes(pickle.dumps(P))

    q = args.quick
    try:
        _, params, _ = ppo.train(
            environment=env, num_timesteps=args.steps, episode_length=env.max_steps, num_envs=args.envs, num_eval_envs=min(args.envs, 128),
            wrap_env_fn=wrap_for_training, learning_rate=args.lr, learning_rate_schedule="ADAPTIVE_KL", desired_kl=args.kl,
            learning_rate_schedule_max_lr=1e-3, entropy_cost=args.entropy, discounting=0.95, gae_lambda=0.95, unroll_length=20,
            batch_size=min(64, args.envs) if q else 256, num_minibatches=4 if q else 32, num_updates_per_batch=4 if q else 5,
            normalize_observations=True, reward_scaling=reward_scale, clipping_epsilon=0.2, max_grad_norm=1.0,
            num_evals=args.evals, network_factory=make_networks(hidden, float(np.mean(W["log_std"])) if W else -1.0),
            restore_params=restore, restore_value_fn=True, seed=args.seed, deterministic_eval=True,
            progress_fn=progress, policy_params_fn=save)
        (out / "params.pkl").write_bytes(pickle.dumps(to_numpy(params, dict(steps=base + args.steps, source="mjx_imitate"))))
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
