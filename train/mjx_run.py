"""Run 20 on a GPU: the same running task as run_env.py, in MuJoCo MJX with Brax PPO (thousands of dogs at once).

    python mjx_run.py --steps 300_000_000 --envs 4096 --out runs/run20gpu     # on a GPU (Kaggle, Colab)
    python mjx_run.py --steps 200_000 --envs 16 --out /tmp/t --quick          # CPU smoke test

Same body (bolt.xml), same A1 motor model applied every physics substep, same 49 inputs, rewards, shoves,
speed curriculum and episode rules as run_env.RunEnv, so a brain trained here takes run_env's exam and
drives the pages unchanged. Differences, for speed: the legs and body only collide with the floor (no
leg-on-leg contact), and the physics randomization is friction, payload and motor gain per dog.

Writes <out>/params.pkl (the brain: observation normalizer + policy weights, numpy), <out>/progress.json,
and prints "update at N: ..." lines, plus "checkpoint N ready: {...}" with run_env's exam when --exam-every.
"""
import argparse
import functools
import json
import pickle
import re
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

HERE = Path(__file__).resolve().parent
DOG = np.array([0.0, 0.2747, -0.5493, 0.0, 0.2747, -0.5493, 0.0, -0.2747, 0.5493, 0.0, -0.2747, 0.5493], dtype=np.float32)   # bolt_env.STANCES["dog"]: the dog stance (copied: Kaggle has no CPU stack)
SCALE = np.tile([0.2, 0.5, 0.6], 4).astype(np.float32)
MOTOR = dict(torque=33.5, power=420.0, speed=21.0, cutoff=25.0)   # bolt_env.MOTOR: the Unitree A1 motor
MAX_V, MAX_W = 4.0, 5.0


def floor_only(xml):
    """Robot geoms collide with the floor only: robot contype 2 / conaffinity 1, floor contype 1 / conaffinity 2."""
    xml = re.sub(r'(<geom name="floor"[^>]*?)(/?>)', r'\1 contype="1" conaffinity="2"\2', xml, count=1)
    return xml.replace('<geom friction="1.0 0.02 0.01" condim="3"', '<geom friction="1.0 0.02 0.01" condim="3" contype="2" conaffinity="1"', 1)


class MjxRun:
    """Brax-style env (reset(rng) -> State, step(State, action) -> State)."""

    def __init__(self, pushes=True, episode_length=1000):
        from brax.envs.base import State
        self.State = State
        xml = floor_only((HERE / "bolt.xml").read_text())
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.mx = mjx.put_model(self.m)
        m = self.m
        self.torso = m.body("torso").id
        self.floor = m.geom("floor").id
        self.feet = jp.array([m.geom(f"{l}_foot").id for l in ("FL", "FR", "RL", "RR")])
        legs = {m.body(f"{l}_{p}").id for l in ("FL", "FR", "RL", "RR") for p in ("thigh", "shin")}
        self.knee_mask = jp.array([m.geom_bodyid[g] in legs and g not in list(np.array(self.feet)) for g in range(m.ngeom)])
        self.body_mask = jp.array([m.geom_bodyid[g] == self.torso for g in range(m.ngeom)])
        self.lo, self.hi = jp.array(m.actuator_ctrlrange[:, 0]), jp.array(m.actuator_ctrlrange[:, 1])
        self.kp0, self.fric0, self.mass0 = jp.array(m.actuator_gainprm[:, 0]), jp.array(m.geom_friction[:, 0]), float(m.body_mass[self.torso])
        self.dt = m.opt.timestep * 10
        self.pushes, self.episode_length = pushes, episode_length
        self.dog, self.scale = jp.array(DOG), jp.array(SCALE)

    observation_size = 49
    action_size = 12
    backend = "mjx"

    @property
    def unwrapped(self):
        return self

    # ---------- physics ----------
    def model_for(self, info):
        mx = self.mx
        kp = self.kp0 * info["kp"]
        gain = mx.actuator_gainprm.at[:, 0].set(kp)
        bias = mx.actuator_biasprm.at[:, 1].set(-kp)
        fric = mx.geom_friction.at[:, 0].set(self.fric0 * info["fric"])
        mass = mx.body_mass.at[self.torso].set(self.mass0 + info["payload"])
        return mx.replace(actuator_gainprm=gain, actuator_biasprm=bias, geom_friction=fric, body_mass=mass)

    def physics(self, mx, d):
        def sub(d, _):
            w = jp.abs(d.qvel[6:18])
            drive = jp.minimum(MOTOR["torque"], MOTOR["power"] / jp.maximum(w, 1e-6))
            drive = drive * jp.clip((MOTOR["cutoff"] - w) / (MOTOR["cutoff"] - MOTOR["speed"]), 0.0, 1.0)
            lo = jp.where(d.qvel[6:18] < 0, -drive, -MOTOR["torque"])
            hi = jp.where(d.qvel[6:18] >= 0, drive, MOTOR["torque"])
            return mjx.step(mx.replace(actuator_forcerange=jp.stack([lo, hi], -1)), d), None
        return jax.lax.scan(sub, d, None, length=10)[0]

    # ---------- senses ----------
    @staticmethod
    def R(d, torso):
        return d.xmat[torso]

    def obs(self, d, info):
        R = d.xmat[self.torso]
        return jp.concatenate([R.T @ jp.array([0.0, 0.0, -1.0]), R.T @ d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3],
                               info["cmd"] * jp.array([1.0, 1.0, 0.5]), d.qpos[7:19] - self.dog, d.qvel[6:18] * 0.05,
                               info["last_action"], d.qpos[2:3]])

    def contacts(self, d):
        c = d.contact
        touching = c.dist < 0.0
        other = jp.where(c.geom[:, 0] == self.floor, c.geom[:, 1], c.geom[:, 0])
        with_floor = touching & ((c.geom[:, 0] == self.floor) | (c.geom[:, 1] == self.floor))
        knee = jp.any(with_floor & self.knee_mask[other])
        body = jp.any(with_floor & self.body_mask[other])
        feet = jp.array([jp.any(with_floor & (other == f)) for f in self.feet])
        return knee, body, feet

    # ---------- commands and curriculum ----------
    def command(self, rng, top_v, top_w):
        k = jax.random.split(rng, 7)
        r = jax.random.uniform(k[0])
        vx = jp.where(r > 0.1, jax.random.uniform(k[1], minval=-1.0, maxval=top_v), 0.0)
        vx = jp.where(r > 0.6, jax.random.uniform(k[2], minval=0.6, maxval=1.0) * top_v, vx)
        vy = jp.where(jax.random.uniform(k[3]) < 0.3, jax.random.uniform(k[4], minval=-0.5, maxval=0.5), 0.0)
        w = jp.where(jax.random.uniform(k[5]) < 0.6, jax.random.uniform(k[6], minval=-top_w, maxval=top_w), 0.0)
        return jp.array([vx, vy, w])

    # ---------- episode ----------
    def reset(self, rng):
        k = jax.random.split(rng, 8)
        info = dict(kp=jax.random.uniform(k[0], minval=0.9, maxval=1.1), fric=jax.random.uniform(k[1], minval=0.5, maxval=1.25),
                    payload=jax.random.uniform(k[2], minval=-1.0, maxval=2.0), top_v=jp.float32(1.0), top_w=jp.float32(1.0),
                    ema=jp.float32(0.0), n_top=jp.int32(0), t=jp.int32(0), last_action=jp.zeros(12), air=jp.zeros(4),
                    track_sum=jp.float32(0.0), rng=k[3])
        info["cmd"] = self.command(k[4], info["top_v"], info["top_w"])
        mx = self.model_for(info)
        yaw = jax.random.uniform(k[5], minval=-jp.pi, maxval=jp.pi)
        q = jp.array(self.m.qpos0).at[2].set(0.45).at[3:7].set(jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)]))
        joints = jp.clip(self.dog + 0.03 * jax.random.normal(k[6], (12,)), self.lo, self.hi)
        q = q.at[7:19].set(joints)
        d = mjx.make_data(mx).replace(qpos=q, ctrl=joints)
        d = mjx.forward(mx, d)
        d = jax.lax.fori_loop(0, 20, lambda i, d: self.physics(mx, d), d)
        z = jp.float32(0.0)
        metrics = dict(track_v=z, vx=z, top_v=jp.float32(1.0), fell=z)
        return self.State(pipeline_state=d, obs=self.obs(d, info), reward=z, done=z, metrics=metrics, info=info)

    def step(self, state, action):
        info, d = dict(state.info), state.pipeline_state
        mx = self.model_for(info)
        rng, k1, k2 = jax.random.split(info["rng"], 3)
        a = jp.clip(action, -1.0, 1.0)
        d = d.replace(ctrl=jp.clip(self.dog + self.scale * a, self.lo, self.hi))
        t = info["t"] + 1
        if self.pushes:   # a shove every 8 s
            shove = jp.where((t % 400) == 0, 1.0, 0.0) * jax.random.uniform(k1, (2,), minval=-1.0, maxval=1.0)
            d = d.replace(qvel=d.qvel.at[0:2].add(shove))
        d = self.physics(mx, d)
        cmd = jp.where((t % 500) == 0, self.command(k2, info["top_v"], info["top_w"]), info["cmd"])

        R = d.xmat[self.torso]
        vb, w = R.T @ d.qvel[0:3], d.qvel[3:6]
        knee, body, feet = self.contacts(d)
        first = feet & (info["air"] > 0)
        moving = jp.linalg.norm(cmd) > 0.1
        air_pay = jp.where(moving, jp.sum((info["air"] - 0.25) * first), 0.0)
        air = jp.where(feet, 0.0, info["air"] + self.dt)
        e_v = jp.sum((cmd[:2] - vb[:2]) ** 2)
        track_v = jp.exp(-e_v / jp.maximum(0.25, 0.1 * cmd[0] ** 2))
        track_w = jp.exp(-(cmd[2] - w[2]) ** 2 / 0.25)
        g = R.T @ jp.array([0.0, 0.0, -1.0])
        r = (1.5 * track_v + 0.75 * track_w - 2.0 * vb[2] ** 2 - 0.05 * jp.sum(w[:2] ** 2) - 1.0 * jp.sum(g[:2] ** 2)
             - 2e-4 * jp.sum(d.actuator_force ** 2) - 0.01 * jp.sum((a - info["last_action"]) ** 2)
             - 1.0 * knee + 1.0 * air_pay)
        fell = body | (R[2, 2] < 0.3) | (d.qpos[2] < 0.15)
        r = r - 10.0 * fell
        track_sum = info["track_sum"] + track_v
        # curriculum at the end of an episode (a fall, or the episode wrapper's time limit at episode_length)
        end = fell | (t >= self.episode_length)
        near_top = cmd[0] > 0.7 * info["top_v"]
        avg = track_sum / jp.maximum(t, 1)
        use = end & near_top & (t >= 200)
        ema = jp.where(use, 0.9 * info["ema"] + 0.1 * avg, info["ema"])
        n_top = info["n_top"] + use.astype(jp.int32)
        up = (n_top >= 20) & (ema > 0.75)
        top_v = jp.where(up, jp.minimum(MAX_V, info["top_v"] + 0.5), info["top_v"])
        top_w = jp.where(up, jp.minimum(MAX_W, info["top_w"] + 0.5), info["top_w"])
        ema, n_top = jp.where(up, 0.0, ema), jp.where(up, 0, n_top)
        # the auto-reset wrapper restores the body; per-episode counters restart here
        info.update(rng=rng, cmd=jp.where(end, self.command(k1, top_v, top_w), cmd), t=jp.where(end, 0, t), last_action=jp.where(end, 0.0, a),
                    air=jp.where(end, 0.0, air), track_sum=jp.where(end, 0.0, track_sum), ema=ema, n_top=n_top, top_v=top_v, top_w=top_w)
        metrics = dict(track_v=track_v, vx=vb[0], top_v=top_v, fell=fell.astype(jp.float32))
        return state.replace(pipeline_state=d, obs=self.obs(d, info), reward=r, done=fell.astype(jp.float32), metrics=metrics, info=info)


# ---------- the trained brain, in numpy (for run_env's exam, the pages and ONNX) ----------
class MjxPolicy:
    """params.pkl -> act(obs): normalize, MLP (ELU), tanh of the mean: Brax PPO's deterministic action."""

    def __init__(self, path):
        P = pickle.loads(Path(path).read_bytes())
        self.mean, self.std, self.layers = P["mean"], P["std"], P["layers"]

    def act(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))   # ELU
        return np.tanh(x[:12])


def to_numpy(normalizer_params, policy_params):
    st = normalizer_params
    mean, std = np.asarray(st.mean), np.asarray(st.std)
    layers = []
    p = policy_params["params"]
    for k in sorted(p, key=lambda s: int(s.split("_")[-1])):
        layers.append((np.asarray(p[k]["kernel"]), np.asarray(p[k]["bias"])))
    return dict(mean=mean, std=std, layers=layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--out", default=str(HERE / "runs" / "run20gpu"))
    ap.add_argument("--evals", type=int, default=30, help="progress reports (and saved brains) over the run")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = MjxRun()
    start, log = time.time(), []

    def progress(n, metrics):
        L = max(1.0, float(metrics.get("eval/avg_episode_length", 1.0)))
        row = dict(steps=int(n), minutes=round((time.time() - start) / 60, 1), length=L,
                   reward=float(metrics.get("eval/episode_reward", np.nan)),
                   track_v=float(metrics.get("eval/episode_track_v", np.nan)) / L,
                   top_v=float(metrics.get("eval/episode_top_v", np.nan)) / L,
                   fell=float(metrics.get("eval/episode_fell", np.nan)))
        log.append(row)
        (out / "progress.json").write_text(json.dumps(log, indent=1))
        print(f"update at {n}: reward {row['reward']:.1f}, episode {L:.0f} steps, tracking {row['track_v']:.2f}, "
              f"practising up to {row['top_v']:.2f} m/s, {row['minutes']} min", flush=True)

    def save(step, make_policy, params):
        P = to_numpy(params[0], params[1])
        (out / "checkpoints" / f"params_{int(step)}.pkl").write_bytes(pickle.dumps(P))
        (out / "params.pkl").write_bytes(pickle.dumps(P))

    nets = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(512, 256, 128),
                             value_hidden_layer_sizes=(512, 256, 128), activation=jax.nn.elu)
    make_inference, params, metrics = ppo.train(
        environment=env, num_timesteps=args.steps, episode_length=1000, num_envs=args.envs, num_eval_envs=min(args.envs, 128),
        learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97, unroll_length=20, batch_size=min(256, args.envs) if args.quick else 256,
        num_minibatches=4 if args.quick else 32, num_updates_per_batch=4, normalize_observations=True, reward_scaling=1.0,
        clipping_epsilon=0.2, gae_lambda=0.95, max_grad_norm=1.0, num_evals=args.evals, network_factory=nets, seed=0,
        progress_fn=progress, policy_params_fn=save)
    save(args.steps, None, params)
    print(f"done: {args.steps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
