"""Give a dog-imitation brain a slot for a new clip (74 -> 75 inputs when "getup" joined ALL_CLIPS).

    python widen_imitate.py runs/imitate18b/checkpoints/policy_15000000_steps.zip runs/imitate18d_start

The new one-hot input goes in at index 48 + old slots (after the old clip one-hot); its first-layer weights
start at zero and its normalizer is neutral, so the brain behaves exactly as before until it learns the clip.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from imitate_env import ALL_CLIPS, ImitateEnv

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
old = PPO.load(src, device="cpu")
slots_old = old.observation_space.shape[0] - 66
at, extra = 48 + slots_old, len(ALL_CLIPS) - slots_old
env = DummyVecEnv([lambda: ImitateEnv()])
new = PPO("MlpPolicy", env, device="cpu", policy_kwargs=old.policy_kwargs, n_steps=old.n_steps, batch_size=old.batch_size,
          n_epochs=old.n_epochs, gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=old.clip_range, ent_coef=old.ent_coef,
          max_grad_norm=old.max_grad_norm, target_kl=old.target_kl, learning_rate=old.learning_rate)
state, widened = old.policy.state_dict(), {}
for k, v in new.policy.state_dict().items():
    w = state[k]
    if w.shape != v.shape:   # a first layer: zero columns for the new clip slot(s), inserted at `at`
        w = torch.cat([w[:, :at], torch.zeros(w.shape[0], extra), w[:, at:]], dim=1)
    widened[k] = w
new.policy.load_state_dict(widened)
new.num_timesteps = old.num_timesteps
n = old.num_timesteps
new.save(out / f"policy_{n}_steps.zip")
stats = src.with_name(src.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
norm = VecNormalize.load(str(stats), DummyVecEnv([lambda: ImitateEnv(slots=slots_old)]))
norm.obs_rms.mean = np.insert(norm.obs_rms.mean, at, np.zeros(extra))
norm.obs_rms.var = np.insert(norm.obs_rms.var, at, np.ones(extra))
norm.observation_space = env.observation_space
norm.save(str(out / f"policy_vecnormalize_{n}_steps.pkl"))
print(f"widened {src} ({66 + slots_old} -> {66 + len(ALL_CLIPS)} inputs, {n:,} steps) -> {out}")
