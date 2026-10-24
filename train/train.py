"""Train Bolt's motor skills with reinforcement learning (PPO).

    python train.py --steps 8000000          # ~1 hour on 4 CPU cores
    python train.py --steps 200000 --quick   # smoke test

Saves the policy to runs/<name>/policy.zip and the observation statistics to
runs/<name>/vecnormalize.pkl. Watch progress with:
    tensorboard --logdir runs
"""
import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from bolt_env import BoltEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--envs", type=int, default=8, help="parallel simulated dogs")
    ap.add_argument("--name", default="bolt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    args = ap.parse_args()

    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    venv = make_vec_env(BoltEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=512, batch_size=4096, n_epochs=5,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, max_grad_norm=1.0, target_kl=0.03,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                           activation_fn=torch.nn.ELU, log_std_init=-1.0),
        tensorboard_log=str(out.parent), seed=args.seed, device="cpu", verbose=1,
    )
    every = max(500_000 // args.envs, 1)
    ckpt = CheckpointCallback(every, str(out / "checkpoints"), name_prefix="policy", save_vecnormalize=True)

    start = time.time()
    model.learn(total_timesteps=args.steps, callback=ckpt, tb_log_name=args.name, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {args.steps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
