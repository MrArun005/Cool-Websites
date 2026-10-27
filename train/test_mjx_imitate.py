"""Parity of mjx_imitate.MjxImitate with imitate_env.ImitateEnv, and of the warm-started Brax brain with the SB3 one.

The CPU stack (torch, SB3) and JAX live in different virtualenvs, so this runs in two halves:
    <cpu python> test_mjx_imitate.py dump  <sb3 zip> <warm.pkl> <cases.pkl>   # ImitateEnv states, obs, rewards, actions
    <jax python> test_mjx_imitate.py check <warm.pkl> <cases.pkl>             # the same through MjxImitate and Brax

Compared: the 75 observations and the five reward terms for identical states (qpos, qvel, clip, time, last action),
the reference-state start (same clip, moment and heading), the clip switch (same new clip), one control step of
physics, and the deterministic action of the warm-started Brax policy against ImitatePolicy.act.
"""
import pickle
import sys
from pathlib import Path

import numpy as np


def dump(zip_path, warm_path, out):
    from imitate_env import ALL_CLIPS, ImitateEnv, ImitatePolicy
    from mjx_imitate import MjxImitatePolicy
    z = Path(zip_path)
    pol = ImitatePolicy(z.parent, z.name, z.name.replace("policy_", "policy_vecnormalize_", 1).replace(".zip", ".pkl"))
    npol = MjxImitatePolicy(warm_path)
    rng = np.random.default_rng(0)
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS)
    starts, steps, switches = [], [], []
    for ci, name in enumerate(ALL_CLIPS):
        c = env.clips[name]
        for k in range(3):
            tau0 = float(rng.uniform(0, c.length if c.loop else c.length + 0.3))
            env.exam_start = (name, tau0)
            obs, _ = env.reset(seed=100 * ci + k)
            tilt = env.ref("tilt")
            u, _, vt = np.linalg.svd(tilt)
            M = env._rotation() @ (u @ vt).T
            starts.append(dict(clip=ci, tau0=tau0, yaw=float(np.arctan2(M[1, 0], M[0, 0])), qpos=env.data.qpos.copy(),
                               qvel=env.data.qvel.copy(), obs=obs, limit=env.limit))
            for t in range(40):
                pre = dict(clip=ci, tau0=env.tau0, t=env.t, last_action=env.last_action.copy(), qpos=env.data.qpos.copy(),
                           qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy())
                a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
                obs, r, fell, done, info = env.step(a)
                d = env.data
                legleg = sum(1 for cc in d.contact[:d.ncon] if cc.geom1 != env.floor and cc.geom2 != env.floor)
                steps.append(dict(pre=pre, action=a, obs=obs, reward=r, terms={k: info[f"r_{k}"] for k in ("pose", "vel", "paw", "root", "root_vel")},
                                  height_err=info["height_err"], tilt=info["tilt"], body_on_floor=env.body_on_floor(), fell=fell,
                                  qpos=d.qpos.copy(), qvel=d.qvel.copy(), legleg=legleg, t=env.t))
                if t == 20 and (c.loop or env.finished() > 0.5):   # a forced switch from here
                    env2 = ImitateEnv(randomize=False, clips=ALL_CLIPS, switch_rate=1e9)
                    env2.data.qpos[:] = d.qpos
                    env2.clip, env2.tau0, env2.t, env2.rng = env.clip, env.tau0, env.t, np.random.default_rng(t + ci)
                    env2._maybe_switch()
                    switches.append(dict(clip=ci, tau0=env.tau0, t=env.t, q=d.qpos[7:19].copy(), new=ALL_CLIPS.index(env2.clip.name),
                                         new_tau0=env2.tau0, new_limit=env2.limit))
                if fell or done:
                    break
    # 20 observations from training-like tries (randomized), for the policy check
    envr = ImitateEnv(randomize=True, clips=ALL_CLIPS)
    obs_set = []
    for k in range(20):
        o, _ = envr.reset(seed=500 + k)
        for _ in range(int(rng.integers(0, 30))):
            o, _, f, dn, _ = envr.step(pol.act(o))
            if f or dn:
                break
        obs_set.append(o)
    obs_set = np.array(obs_set)
    acts = np.array([pol.act(o) for o in obs_set])
    nacts = np.array([npol.act(o) for o in obs_set])
    print(f"numpy brain vs ImitatePolicy.act: max |diff| {np.abs(acts - nacts).max():.2e}")
    Path(out).write_bytes(pickle.dumps(dict(starts=starts, steps=steps, switches=switches, obs=obs_set, acts=acts)))
    print(f"{len(starts)} starts, {len(steps)} steps, {len(switches)} switches -> {out}")


def check(warm_path, cases):
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from brax.training.agents.ppo import networks as ppo_networks
    import mjx_imitate as M
    C = pickle.loads(Path(cases).read_bytes())
    env = M.MjxImitate(randomize=False, switch_rate=0.0)

    def info_of(s):
        return dict(kp=jp.float32(1), mass=jp.float32(1), fric=jp.float32(1), clip=jp.int32(s["clip"]), tau0=jp.float32(s["tau0"]),
                    t=jp.int32(s["t"]), last_action=jp.array(s["last_action"], jp.float32), trunc=jp.float32(0), limit=jp.int32(0))

    @jax.jit
    def senses(qpos, qvel, info):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        terms, err = env.tracking(d, info)
        return env.obs(d, info), terms, err, env.body_on_floor(d)

    @jax.jit
    def one_step(qpos, qvel, ctrl, info, a):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        st = env.State(pipeline_state=d, obs=jp.zeros(75), reward=jp.float32(0), done=jp.float32(0),
                       metrics={f"r_{k}": jp.float32(0) for k in M.WEIGHTS} | dict(fell=jp.float32(0), canter=jp.float32(0), canter_steps=jp.float32(0)),
                       info=info | dict(rng=jax.random.PRNGKey(0), limit=jp.int32(10 ** 6)))
        n = env.step(st, a)
        return n.pipeline_state.qpos, n.obs

    start_data = jax.jit(env.start_data)
    # 1. observations and reward terms for identical states
    do, dt, de, flo, n_body = [], {k: [] for k in M.WEIGHTS}, [], 0, 0
    for s in C["steps"]:
        info = info_of(dict(s["pre"], t=s["t"], last_action=s["action"]))
        o, terms, err, bof = senses(jp.array(s["qpos"], jp.float32), jp.array(s["qvel"], jp.float32), info)
        do.append(np.abs(np.asarray(o) - s["obs"]))
        for k in M.WEIGHTS:
            dt[k].append(abs(float(terms[k]) - s["terms"][k]))
        de.append(abs(float(err["tilt"]) - s["tilt"]) + abs(float(err["height_err"]) - s["height_err"]))
        flo += bool(bof) != bool(s["body_on_floor"])
        n_body += bool(s["body_on_floor"])
    do = np.array(do)
    worst = np.argsort(do.max(0))[::-1][:3]
    print(f"[obs] {len(do)} states: max |diff| {do.max():.2e} (worst inputs {list(worst)}: {[f'{do.max(0)[i]:.1e}' for i in worst]}), mean {do.mean():.1e}")
    print("[reward terms] max |diff| " + ", ".join(f"{k} {max(v):.1e}" for k, v in dt.items()) + f"; errors (tilt+height) {max(de):.1e}")
    print(f"[body on floor] disagree in {flo} of {len(do)} states (CPU: body on floor in {n_body})")
    # 2. reference-state start
    dq = []
    for s in C["starts"]:
        info = info_of(dict(s, t=0, last_action=np.zeros(12)))
        d = start_data(info, jp.float32(s["yaw"]))
        q = np.asarray(d.qpos)
        qq = s["qpos"].copy()
        if np.dot(q[3:7], qq[3:7]) < 0:
            q[3:7] *= -1
        dq.append(max(np.abs(q - qq).max(), np.abs(np.asarray(d.qvel) - s["qvel"]).max()))
        o = np.asarray(env.obs(d, info))
        dq[-1] = max(dq[-1], np.abs(o - s["obs"]).max())
    print(f"[start] {len(dq)} reference-state starts: max |diff| qpos/qvel/obs {max(dq):.2e} (median {np.median(dq):.1e})")
    # 3. clip switches
    bad = 0
    for s in C["switches"]:
        info = info_of(dict(s, last_action=np.zeros(12)))
        tau0, limit = env.switch_to(info, jp.array(s["q"], jp.float32), jp.int32(s["new"]))
        bad += abs(float(tau0) - s["new_tau0"]) > 1e-4 or int(limit) != s["new_limit"]
    print(f"[switch] {len(C['switches'])} forced switches: {bad} differ in join time or end step")
    # 4. one control step of physics (float32 MJX, floor-only contacts vs float64 MuJoCo)
    dp, do1, ll, n_ll = [], [], [], 0
    for s in C["steps"][::3]:
        p = s["pre"]
        info = info_of(p)
        q, o = one_step(jp.array(p["qpos"], jp.float32), jp.array(p["qvel"], jp.float32), jp.array(p["ctrl"], jp.float32), info,
                        jp.array(s["action"], jp.float32))
        dp.append(np.abs(np.asarray(q)[7:19] - s["qpos"][7:19]).max())
        do1.append(np.abs(np.asarray(o) - s["obs"]).max())
        n_ll += s["legleg"] > 0
    legleg_all = sum(s["legleg"] > 0 for s in C["steps"])
    print(f"[one step] {len(dp)} steps: joints max |diff| {max(dp):.2e} rad (median {np.median(dp):.1e}), obs max {max(do1):.2e} "
          f"(median {np.median(do1):.1e}); leg-on-leg contacts on the CPU (none in MJX) in {legleg_all} of {len(C['steps'])} states")
    # 5. warm-started Brax policy vs ImitatePolicy.act
    W = pickle.loads(Path(warm_path).read_bytes())
    nets = M.make_networks(W["hidden"], float(np.mean(W["log_std"])))(75, 12)
    params = M.to_brax(W)
    act = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    a = np.array([np.clip(np.asarray(act(jp.array(o), jax.random.PRNGKey(0))[0]), -1, 1) for o in C["obs"]])
    print(f"[warm start] Brax deterministic action vs ImitatePolicy.act on {len(a)} obs: max |diff| {np.abs(a - C['acts']).max():.2e}")
    v = np.asarray(nets.value_network.apply(params[0], params[2], jp.array(C["obs"])))
    print(f"[warm start] critic on them: {v.min():.2f} .. {v.max():.2f} (SB3 normalized-return units)")


if __name__ == "__main__":
    dump(*sys.argv[2:5]) if sys.argv[1] == "dump" else check(*sys.argv[2:4])
