"""Behaviour-cloning warm start for the walk / run brain: a PPO model that already steps like the
closed-loop reference walker (walk_expert.py), ready for train.py --resume.

The first walk run (runs/mid_walk_v1) learned to stand and absorb shoves: from a brain that does
nothing, every early attempt to step fell and paid nothing, so standing (~0.16 per step) won. Here
the brain starts out walking instead, and PPO only has to make it better.

    python walk_bc.py --out runs/mid_walk_bc                     # walk
    python walk_bc.py --task run --out runs/mid_run_bc           # (run is resumed from the walk brain)
    python train.py --skills --env walk_env:WalkEnv --model model/duck_mid.xml --task walk --name mid_walk \\
        --resume runs/mid_walk_bc/policy.zip ...

Steps (DAgger): roll out a mix of the teacher and the student on randomized bodies over the full
command range, label every visited state with the teacher's action, fit the actor (normalized
obs → action mean) by least squares; repeat with more of the student driving. Then the normalizer's
return statistics and the critic are fitted on the student's own rollouts, so PPO's first updates
don't throw the gait away on a random critic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from walk_env import WalkEnv
from walk_expert import Expert, GAINS

HERE = Path(__file__).resolve().parent
MODEL = str(HERE / "model" / "duck_mid.xml")


def make_envs(task, n, seed):
    return [WalkEnv(task=task, randomize=True, seed=seed + i, model_path=MODEL, practice=False) for i in range(n)]


def collect(envs, experts, steps, student=None, beta=1.0, noise=0.0, rng=None, ramp_s=0.0):
    """Visit states with the teacher (prob. beta per episode) or the student driving; return obs and
    teacher labels. Episodes restart on termination; student actions get Gaussian noise."""
    obs_l, act_l = [], []
    obs = [e.reset(seed=int(rng.integers(1 << 30)))[0] for e in envs]
    for x in experts:
        x.reset()
    driver = [rng.random() < beta for _ in envs]
    per = steps // len(envs)
    for _ in range(per):
        O = np.array(obs)
        A_t = np.array([x.action() for x in experts])
        obs_l.append(O)
        act_l.append(A_t)
        A = A_t.copy()
        if student is not None and not all(driver):
            A_s = student(O)
            A_s = np.clip(A_s + noise * rng.normal(size=A_s.shape), -1, 1)
            for i, d in enumerate(driver):
                if not d:
                    A[i] = A_s[i]
        for i, e in enumerate(envs):
            a = A[i]
            if ramp_s:
                a = a * min(1.0, e.t * e.dt / ramp_s)
            o, r, term, trunc, _ = e.step(a)
            if term or trunc:
                o = e.reset(seed=int(rng.integers(1 << 30)))[0]
                experts[i].reset()
                driver[i] = rng.random() < beta
            obs[i] = o
    return np.concatenate(obs_l), np.concatenate(act_l)


def build_model(venv, log_std, lr):
    return PPO("MlpPolicy", venv, n_steps=512, batch_size=4096, n_epochs=5, learning_rate=lr, gamma=0.99,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=0.03,
               policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.ELU,
                                  log_std_init=log_std), seed=0, device="cpu", verbose=1)


def norm(vn, O):
    return np.clip((O - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs).astype(np.float32)


def actor_mean(policy, X):
    f = policy.extract_features(X, policy.pi_features_extractor)
    return policy.action_net(policy.mlp_extractor.forward_actor(f))


def fit_actor(model, vn, O, A, epochs=30, lr=1e-3, bs=2048, log=print):
    p = model.policy
    params = list(p.mlp_extractor.policy_net.parameters()) + list(p.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    X = torch.as_tensor(norm(vn, O))
    Y = torch.as_tensor(A, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            loss = ((actor_mean(p, X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 9 or ep == epochs - 1:
            log(f"  actor epoch {ep + 1}: mse {tot / n:.5f}")


def student_fn(model, vn):
    def f(O):
        with torch.no_grad():
            return actor_mean(model.policy, torch.as_tensor(norm(vn, O))).numpy().clip(-1, 1)
    return f


def fit_critic(model, vn, envs, steps, noise, rng, gamma=0.99, lam=0.95, iters=3, log=print):
    """Return statistics and critic from the student's own (noisy) rollouts: TD(λ) targets."""
    p = model.policy
    stud = student_fn(model, vn)
    n_env = len(envs)
    T = steps // n_env
    O = np.zeros((T, n_env, envs[0].obs_size), np.float32)
    R = np.zeros((T, n_env))
    D = np.zeros((T, n_env))          # 1 where the episode ended after this step (terminated)
    C = np.zeros((T, n_env))          # 1 where it was cut by time (truncated)
    obs = [e.reset(seed=int(rng.integers(1 << 30)))[0] for e in envs]
    ret = np.zeros(n_env)
    rets = []
    for t in range(T):
        O[t] = np.array(obs)
        A = np.clip(stud(O[t]) + noise * rng.normal(size=(n_env, envs[0].action_space.shape[0])), -1, 1)
        for i, e in enumerate(envs):
            o, r, term, trunc, _ = e.step(A[i])
            R[t, i] = r
            ret[i] = ret[i] * gamma + r
            rets.append(ret[i])
            if term or trunc:
                D[t, i], C[t, i] = float(term), float(trunc and not term)
                ret[i] = 0.0
                o = e.reset(seed=int(rng.integers(1 << 30)))[0]
            obs[i] = o
    rets = np.array(rets)
    vn.ret_rms.mean, vn.ret_rms.var, vn.ret_rms.count = 0.0, float(np.var(rets)), float(len(rets))
    Rn = np.clip(R / np.sqrt(vn.ret_rms.var + vn.epsilon), -vn.clip_reward, vn.clip_reward)
    X = torch.as_tensor(norm(vn, O.reshape(T * n_env, -1)))
    opt = torch.optim.Adam(list(p.mlp_extractor.value_net.parameters()) + list(p.value_net.parameters()), lr=1e-3)
    for it in range(iters):
        with torch.no_grad():
            V = p.predict_values(X).numpy().reshape(T, n_env)
        # TD(λ) returns; at time cuts bootstrap from the value of the cut state itself
        G = np.zeros_like(V)
        nxt_v, nxt_g = V[-1], V[-1]
        for t in reversed(range(T)):
            boot = np.where(C[t] > 0, V[t], (1 - lam) * nxt_v + lam * nxt_g)
            G[t] = Rn[t] + gamma * (1 - D[t]) * boot
            nxt_v, nxt_g = V[t], G[t]
        Y = torch.as_tensor(G.reshape(-1, 1), dtype=torch.float32)
        for ep in range(20):
            perm = torch.randperm(len(X))
            tot = 0.0
            for k in range(0, len(X), 2048):
                idx = perm[k:k + 2048]
                loss = ((p.predict_values(X[idx]) - Y[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
        log(f"  critic pass {it + 1}: mse {tot / len(X):.4f}, return std {np.sqrt(vn.ret_rms.var):.2f}, "
            f"mean reward/step {R.mean():.3f}, falls {int(D.sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="walk")
    ap.add_argument("--out", default="runs/mid_walk_bc")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40_000, help="samples per DAgger round")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--log-std", type=float, default=-2.8)
    ap.add_argument("--lr", type=float, default=2e-4, help="PPO learning rate stored in the model")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    envs = make_envs(a.task, a.envs, a.seed * 1000)
    experts = [Expert(e, **GAINS[a.task]) for e in envs]
    venv = VecNormalize(DummyVecEnv([lambda: WalkEnv(task=a.task, randomize=True, model_path=MODEL)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = build_model(venv, a.log_std, a.lr)
    noise = float(np.exp(a.log_std))
    log = lambda s: print(s, flush=True)

    O, A = collect(envs, experts, a.steps, rng=rng)
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = O.mean(0), O.var(0) + 1e-6, float(len(O))
    log(f"round 0: teacher only, {len(O)} samples")
    fit_actor(model, venv, O, A, log=log)
    betas = np.linspace(0.5, 0.0, a.rounds - 1) if a.rounds > 1 else []
    for k, beta in enumerate(betas, 1):
        o, act = collect(envs, experts, a.steps, student=student_fn(model, venv), beta=beta, noise=noise, rng=rng)
        O, A = np.concatenate([O, o]), np.concatenate([A, act])
        log(f"round {k}: teacher drives {beta:.2f} of episodes, {len(O)} samples")
        fit_actor(model, venv, O, A, epochs=20, log=log)
    fit_critic(model, venv, envs, a.steps, noise, rng, log=log)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    (out / "bc.json").write_text(json.dumps(dict(task=a.task, gains=GAINS[a.task], samples=len(O), rounds=a.rounds,
                                                 log_std=a.log_std, lr=a.lr), indent=1))
    log(f"saved {out / 'policy.zip'}")


if __name__ == "__main__":
    main()
