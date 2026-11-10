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
    Ws = [torch.nn.Parameter(torch.as_tensor(np.array(W, np.float32))) for W, _ in layers]
    bs = [torch.nn.Parameter(torch.as_tensor(np.array(b, np.float32))) for _, b in layers]
    opt = torch.optim.Adam(Ws + bs, lr=a.lr)
    mean_t, std_t = torch.as_tensor(mean), torch.as_tensor(std)

    def student(x):
        x = torch.clamp((x - mean_t) / std_t, -10.0, 10.0)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            x = x @ W + b
            if i < len(Ws) - 1:
                x = torch.nn.functional.elu(x)
        return x

    def teach(o):
        x = torch.as_tensor(np.clip((o - tmean) / tstd, -10.0, 10.0), dtype=torch.float32)
        f = teacher.mlp_extractor(teacher.extract_features(x))
        return teacher.action_net(f[0])

    def save():
        Q = dict(T)
        Q.update(mean=mean, std=std, history=k, blind_vel=True,
                 layers=[(W.detach().numpy().copy(), b.detach().numpy().copy()) for W, b in zip(Ws, bs)])
        pickle.dump(Q, open(out / "params.pkl", "wb"))

    lo, hi = venv.action_space.low, venv.action_space.high
    sigma = a.noise * np.exp(np.asarray(T["log_std"], np.float32)) if "log_std" in T else 0.05
    obs = venv.reset()
    hist = np.tile(obs[:, None, HIST_SEL], (1, k, 1))
    X, Y = [], []
    t0 = time.time()
    for it in range(a.iters):
        falls, n = 0, 0
        while n < a.steps:
            b = obs.astype(np.float32).copy()
            b[:, 6:9] = 0.0
            x = np.concatenate([b, hist.reshape(a.envs, -1)], axis=1)
            with torch.no_grad():
                act_t = teach(obs).numpy()
                act_s = student(torch.as_tensor(x)).numpy()
            X.append(x); Y.append(act_t)
            hist = np.concatenate([hist[:, 1:], obs[:, None, HIST_SEL]], axis=1)
            obs, _, done, infos = venv.step(np.clip(act_s + sigma * np.random.randn(*act_s.shape), lo, hi))
            for e in np.flatnonzero(done):   # a new try: its history starts over (as MjxImitatePolicy.reset)
                hist[e] = np.tile(obs[e, HIST_SEL], (k, 1))
                falls += not infos[e].get("TimeLimit.truncated", False)
            n += a.envs
        Xa, Ya = np.concatenate(X)[-a.keep:], np.concatenate(Y)[-a.keep:]
        X, Y = [Xa], [Ya]
        xs, ys = torch.as_tensor(Xa), torch.as_tensor(Ya)
        for _ in range(a.epochs):
            perm = torch.randperm(len(xs))
            tot, m = 0.0, 0
            for i in range(0, len(xs), 4096):
                j = perm[i:i + 4096]
                loss = ((student(xs[j]) - ys[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(Ws + bs, 1.0); opt.step()
                tot, m = tot + float(loss.detach()), m + 1
        print(f"iter {it}: {len(xs):,} samples, student ends by falling {1000 * falls / n:.2f} per 1000 steps, action error "
              f"{np.sqrt(tot / m):.4f} (rms), {(time.time() - t0) / 60:.1f} min", flush=True)
        save()
        if (out / "STOP").exists():
            break
    print(f"done -> {out / 'params.pkl'}", flush=True)


if __name__ == "__main__":
    main()
