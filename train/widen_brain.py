"""Give a trained walk brain a new input (the turn command) without losing what it learned.

    python widen_brain.py ../models/bolt-dog-walk-dog-5M runs/turn_start

The network's first layers get one extra input whose weights start at zero, and the input
normalizer gets a matching entry (mean 0, spread 1). At a turn command of 0 the widened brain
behaves exactly like the original, so training only has to learn how to use the new input.
Writes policy.zip + vecnormalize.pkl, ready for train.py --resume <out>/policy.zip --turning.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import BoltEnv


def widen(src, out, gait_generator="dog"):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old = PPO.load(src / "policy.zip", device="cpu")
    n_old = old.observation_space.shape[0]
    env = DummyVecEnv([lambda: BoltEnv(skills=["walk"], gait_generator=gait_generator, turning=True)])
    n_new = env.observation_space.shape[0]
    new = PPO("MlpPolicy", env, device="cpu", policy_kwargs=old.policy_kwargs, learning_rate=old.learning_rate,
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs, gamma=old.gamma,
              gae_lambda=old.gae_lambda, clip_range=old.clip_range, ent_coef=old.ent_coef,
              max_grad_norm=old.max_grad_norm, target_kl=old.target_kl)
    state = old.policy.state_dict()
    widened = {}
    for k, v in new.policy.state_dict().items():
        w = state[k]
        if w.shape != v.shape:  # a first layer: copy the old columns, the new input's column starts at 0
            assert w.shape[1] == n_old and v.shape[1] == n_new, k
            w = torch.cat([w, torch.zeros(w.shape[0], n_new - n_old)], dim=1)
        widened[k] = w
    new.policy.load_state_dict(widened)
    new.num_timesteps = old.num_timesteps
    new.save(out / "policy.zip")

    norm = VecNormalize.load(str(src / "vecnormalize.pkl"), DummyVecEnv([lambda: BoltEnv(gait_generator=gait_generator)]))
    rms = norm.obs_rms
    rms.mean = np.concatenate([rms.mean, np.zeros(n_new - n_old)])
    rms.var = np.concatenate([rms.var, np.ones(n_new - n_old)])
    norm.observation_space = env.observation_space
    norm.save(str(out / "vecnormalize.pkl"))
    print(f"{src.name}: {n_old} -> {n_new} inputs, {old.num_timesteps:,} steps -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    args = ap.parse_args()
    widen(args.src, args.out)
