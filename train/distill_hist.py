"""Copy a brain into one that is blind to body velocity but sees its last few inputs (DAgger, as distill_blind.py).

    python distill_hist.py ../models/bolt-lite3-dog-149M out_dir --history 3 [--iters 60 --steps 40000]

Why (PLAN.md 10 Nov): the blind Lite3 brain falls in steered canters. A probe on its teacher showed the cause: with
only the forward-speed input zeroed the teacher falls in 14 of 24 steered canters (3 of 24 with it). A brain with one
frame of input cannot know its speed; with the last frames (gravity, gyro, joints, previous action: HIST_SEL, the
same inputs the Go2 brain uses) it can work it out. The student is the teacher's network with a wider first layer
(new columns start at zero, so it starts as the blind teacher), and writes params.pkl directly (numpy; history and
blind_vel set), which MjxImitatePolicy and the robot program already run. No SB3 policy comes out. Simulation only.
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", help="brain folder: policy.zip, vecnormalize.pkl, params.pkl")
    ap.add_argument("out")
    ap.add_argument("--body", default="lite3")
    ap.add_argument("--clips", nargs="*", default=["walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right", "sit", "lie", "getup"])
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--keep", type=int, default=600_000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--init", help="continue from this params.pkl (same history)")
    ap.add_argument("--motor-speed-x", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["BOLT_BODY"] = a.body
    os.environ.pop("BOLT_BLIND_VEL", None)
    if a.motor_speed_x:
        os.environ["BOLT_MOTOR_SPEED_X"] = str(a.motor_speed_x)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from imitate_env import ImitateEnv
    from mjx_imitate import HIST_SEL

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(2)
    src, out = Path(a.teacher), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = pickle.load(open(a.init or src / "params.pkl", "rb"))
    T = pickle.load(open(src / "params.pkl", "rb"))
    venv = make_vec_env(ImitateEnv, n_envs=a.envs, seed=a.seed, vec_env_cls=SubprocVecEnv, env_kwargs=dict(
        clips=a.clips, switch_rate=0.3, steer_max=0.8, latency_ms=(0.0, 60.0), moving_rate=0.75))
    teacher = PPO.load(src / "policy.zip", device="cpu").policy
    teacher.set_training_mode(False)
    tmean, tstd = np.asarray(T["mean"], np.float32), np.asarray(T["std"], np.float32)
    k, nh = a.history, len(HIST_SEL)
    if a.init:
        assert int(P.get("history", 0)) == k
        mean, std, layers = np.asarray(P["mean"], np.float32), np.asarray(P["std"], np.float32), P["layers"]
    else:
        mean = np.concatenate([tmean, np.tile(tmean[HIST_SEL], k)])
        std = np.concatenate([tstd, np.tile(tstd[HIST_SEL], k)])
        W0, b0 = T["layers"][0]
        layers = [(np.concatenate([W0, np.zeros((k * nh, W0.shape[1]), np.float32)]), b0)] + list(T["layers"][1:])
    value = [(np.array(W, np.float32), np.array(b, np.float32)) for W, b in P["value"]]   # the value head, for PPO afterwards
    if value[0][0].shape[0] < len(mean):
        value[0] = (np.concatenate([value[0][0], np.zeros((len(mean) - value[0][0].shape[0], value[0][0].shape[1]), np.float32)]), value[0][1])
    Vw = [torch.nn.Parameter(torch.as_tensor(W)) for W, _ in value]
    Vb = [torch.nn.Parameter(torch.as_tensor(b)) for _, b in value]
    Ws = [torch.nn.Parameter(torch.as_tensor(np.array(W, np.float32))) for W, _ in layers]
    bs = [torch.nn.Parameter(torch.as_tensor(np.array(b, np.float32))) for _, b in layers]
    opt = torch.optim.Adam(Ws + bs + Vw + Vb, lr=a.lr)
    mean_t, std_t = torch.as_tensor(mean), torch.as_tensor(std)

    def net(x, WW, bb):
        x = torch.clamp((x - mean_t) / std_t, -10.0, 10.0)
        for i, (W, b) in enumerate(zip(WW, bb)):
            x = x @ W + b
            if i < len(WW) - 1:
                x = torch.nn.functional.elu(x)
        return x

    def student(x):
        return net(x, Ws, bs)

    def teach(o):
        x = torch.as_tensor(np.clip((o - tmean) / tstd, -10.0, 10.0), dtype=torch.float32)
        f = teacher.mlp_extractor(teacher.extract_features(x))
        return teacher.action_net(f[0]), teacher.value_net(f[1])

    def save():
        Q = dict(T)
        Q.update(mean=mean, std=std, history=k, blind_vel=True,
                 layers=[(W.detach().numpy().copy(), b.detach().numpy().copy()) for W, b in zip(Ws, bs)],
                 value=[(W.detach().numpy().copy(), b.detach().numpy().copy()) for W, b in zip(Vw, Vb)])
        pickle.dump(Q, open(out / "params.pkl", "wb"))

    lo, hi = venv.action_space.low, venv.action_space.high
    sigma = a.noise * np.exp(np.asarray(T["log_std"], np.float32)) if "log_std" in T else 0.05
    obs = venv.reset()
    hist = np.tile(obs[:, None, HIST_SEL], (1, k, 1))
    X, Y, V = [], [], []
    t0 = time.time()
    for it in range(a.iters):
        falls, n = 0, 0
        while n < a.steps:
            b = obs.astype(np.float32).copy()
            b[:, 6:9] = 0.0
            x = np.concatenate([b, hist.reshape(a.envs, -1)], axis=1)
            with torch.no_grad():
                act_t, val_t = (v.numpy() for v in teach(obs))
                act_s = student(torch.as_tensor(x)).numpy()
            X.append(x); Y.append(act_t); V.append(val_t)
            hist = np.concatenate([hist[:, 1:], obs[:, None, HIST_SEL]], axis=1)
            obs, _, done, infos = venv.step(np.clip(act_s + sigma * np.random.randn(*act_s.shape), lo, hi))
            for e in np.flatnonzero(done):   # a new try: its history starts over (as MjxImitatePolicy.reset)
                hist[e] = np.tile(obs[e, HIST_SEL], (k, 1))
                falls += not infos[e].get("TimeLimit.truncated", False)
            n += a.envs
        Xa, Ya, Va = np.concatenate(X)[-a.keep:], np.concatenate(Y)[-a.keep:], np.concatenate(V)[-a.keep:]
        X, Y, V = [Xa], [Ya], [Va]
        xs, ys, vs = torch.as_tensor(Xa), torch.as_tensor(Ya), torch.as_tensor(Va)
        vvar = max(1e-6, float(vs.var()))
        for _ in range(a.epochs):
            perm = torch.randperm(len(xs))
            tot, m = 0.0, 0
            for i in range(0, len(xs), 4096):
                j = perm[i:i + 4096]
                la = ((student(xs[j]) - ys[j]) ** 2).mean()
                lv = ((net(xs[j], Vw, Vb) - vs[j]) ** 2).mean() / vvar   # separate networks: no effect on the brain
                opt.zero_grad(); (la + 0.05 * lv).backward(); torch.nn.utils.clip_grad_norm_(Ws + bs, 1.0); opt.step()
                tot, m, vtot = tot + float(la.detach()), m + 1, float(lv.detach())
        print(f"iter {it}: {len(xs):,} samples, student ends by falling {1000 * falls / n:.2f} per 1000 steps, action error "
              f"{np.sqrt(tot / m):.4f} (rms), value error {vtot:.3f} of its variance, {(time.time() - t0) / 60:.1f} min", flush=True)
        save()
        if (out / "STOP").exists():
            break
    venv.close()
    to_sb3(out / "params.pkl", src, out, a.history)
    print(f"done -> {out / 'params.pkl'} and policy.zip + vecnormalize.pkl (train_imitate --init-from {out} --history {k} --blind-vel)", flush=True)


def to_sb3(pkl, teacher_dir, out, k):
    """params.pkl (brain + value head) -> an SB3 run folder, so train_imitate can go on with PPO from it."""
    os.environ["BOLT_HISTORY"], os.environ["BOLT_BLIND_VEL"] = str(k), "1"
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from imitate_env import ImitateEnv, make_env
    P = pickle.load(open(pkl, "rb"))
    old = PPO.load(Path(teacher_dir) / "policy.zip", device="cpu")
    tn = VecNormalize.load(str(Path(teacher_dir) / "vecnormalize.pkl"), DummyVecEnv([lambda: ImitateEnv()]))
    venv = VecNormalize(DummyVecEnv([lambda: make_env()]), norm_obs=True, norm_reward=True, clip_obs=tn.clip_obs, gamma=tn.gamma)
    venv.obs_rms.mean = np.asarray(P["mean"], np.float64)
    venv.obs_rms.var = np.maximum(np.asarray(P["std"], np.float64) ** 2 - venv.epsilon, 1e-12)
    venv.obs_rms.count, venv.ret_rms = tn.obs_rms.count, tn.ret_rms
    model = PPO("MlpPolicy", venv, n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs, learning_rate=1e-5,
                gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=0.2, ent_coef=old.ent_coef, max_grad_norm=old.max_grad_norm,
                target_kl=0.01, policy_kwargs=old.policy_kwargs, device="cpu", verbose=0)
    sd = model.policy.state_dict()
    for prefix, head, layers in (("mlp_extractor.policy_net", "action_net", P["layers"]), ("mlp_extractor.value_net", "value_net", P["value"])):
        idx = sorted({int(n.split(".")[2]) for n in sd if n.startswith(prefix)})
        for i, (W, b) in zip(idx, layers[:-1]):
            sd[f"{prefix}.{i}.weight"], sd[f"{prefix}.{i}.bias"] = torch.as_tensor(W.T.copy()), torch.as_tensor(b.copy())
        sd[f"{head}.weight"], sd[f"{head}.bias"] = torch.as_tensor(layers[-1][0].T.copy()), torch.as_tensor(layers[-1][1].copy())
    sd["log_std"] = torch.as_tensor(np.asarray(P["log_std"], np.float32))
    model.policy.load_state_dict(sd)
    model.num_timesteps = int(P["steps"])
    model.save(Path(out) / "policy.zip")
    venv.save(str(Path(out) / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
