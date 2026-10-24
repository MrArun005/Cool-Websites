"""Resume the get_up run with Microduck-style exploration (train.py's resume keeps the old noise).

    GETUP_STEP_OFFSET=<get_up steps already done> python train_getup.py --resume <ckpt.zip> --steps 30000000 \\
        --log-std -0.7 --ent-coef 0.005

Why: the run was resumed from the stand brain, whose exploration noise had shrunk to std 0.08 rad.
Microduck's stand-up starts at std 1.0 (action scale 1.0) with entropy bonus 0.01; with 0.08 rad
of noise a roll or a push-up from the floor is never sampled. This resets the policy's log_std
(same network, same obs/action layout) and adds an entropy bonus, then continues PPO as train.py.
"""
import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from backkick import BackKickEnv

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--name", default="mid_get_up")
    ap.add_argument("--log-std", type=float, default=None, help="reset the exploration log std to this")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=500_000)
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = HERE / "runs" / a.name
    venv = make_vec_env(BackKickEnv, n_envs=a.envs, seed=int(time.time()) % 10000, vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(task="get_up", model_path=str(HERE / "model/duck_mid.xml")))
    ckpt = Path(a.resume)
    stats = ckpt.with_name("vecnormalize.pkl")
    if not stats.exists():
        stats = ckpt.with_name(ckpt.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
    venv = VecNormalize.load(str(stats), venv)
    model = PPO.load(ckpt, env=venv, device="cpu", tensorboard_log=str(out.parent),
                     custom_objects=dict(ent_coef=a.ent_coef, learning_rate=a.lr, lr_schedule=lambda _: a.lr))
    if a.log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(a.log_std)
    print(f"resume {ckpt} at {model.num_timesteps} steps; log_std {model.policy.log_std.mean().item():.2f} "
          f"ent_coef {model.ent_coef} lr {a.lr}; GETUP step offset {BackKickEnv.step_offset}", flush=True)
    cb = CheckpointCallback(max(a.ckpt_every // a.envs, 1), str(out / "checkpoints"), name_prefix="policy",
                            save_vecnormalize=True)
    model.learn(total_timesteps=a.steps - model.num_timesteps, callback=cb, tb_log_name=a.name,
                reset_num_timesteps=False, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
