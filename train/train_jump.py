"""Train Bolt's jump brain (jump_env.py) from scratch, examining every checkpoint and keeping the best.

    python train_jump.py --name jump1 --steps 6000000 [--resume runs/jump1/checkpoints/policy_X_steps.zip] [--lr 1e-4]

Writes runs/<name>/checkpoints/, runs/<name>/best/ (policy.zip, vecnormalize.pkl, best.json) and prints
"checkpoint N ready: {...}" / "new best" lines (watch.log style) as it goes.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from jump_env import JumpEnv, JumpPolicy, exam


class Examiner(BaseCallback):
    """Every `every` steps: save a checkpoint, take the exam, keep the best."""

    def __init__(self, out, every):
        super().__init__()
        self.out, self.every, self.next, self.best = out, every, every, -1.0

    def _on_training_start(self):   # after a resume, the next exam is the next multiple of `every`
        self.next = (self.num_timesteps // self.every + 1) * self.every

    def _on_step(self):
        if self.num_timesteps < self.next:
            return not (self.out / "STOP").exists()
        self.next += self.every
        ck = self.out / "checkpoints"
        ck.mkdir(exist_ok=True)
        n = self.num_timesteps
        self.model.save(ck / f"policy_{n}_steps.zip")
        self.training_env.save(str(ck / f"policy_vecnormalize_{n}_steps.pkl"))
        result = exam(JumpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"))
        score = sum(result.values()) / len(result)
        levels = self.training_env.get_attr("bar_level")
        print(f"checkpoint {n} ready: {result} (practice bars now {100 * sum(levels) / len(levels):.0f} cm)", flush=True)
        if result.get("jump_15", 1) == 0 and self.best > 0.3:   # collapsed (run 15c at 7M): say so loudly
            print(f"collapse warning at {n}: no clean 15 cm jumps (best score so far {self.best:.2f})", flush=True)
        if score > self.best:
            self.best = score
            best = self.out / "best"
            best.mkdir(exist_ok=True)
            shutil.copy(ck / f"policy_{n}_steps.zip", best / "policy.zip")
            shutil.copy(ck / f"policy_vecnormalize_{n}_steps.pkl", best / "vecnormalize.pkl")
            (best / "best.json").write_text(json.dumps(dict(steps=n, exam=result, score=score), indent=2))
            print(f"new best at {n}: score {score:.2f}", flush=True)
        return not (self.out / "STOP").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="jump")
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resume")
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (on --resume too)")
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--n-steps", type=int, default=512, help="steps per env per update (x envs = samples per update)")
    args = ap.parse_args()
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    venv = make_vec_env(JumpEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv)
    if args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")), venv)
        model = PPO.load(ck, env=venv, device="cpu", custom_objects=dict(learning_rate=args.lr, target_kl=args.target_kl, n_steps=args.n_steps))
        venv.env_method("__setattr__", "env_steps", model.num_timesteps // args.envs)  # keep the bar-height curriculum
        print(f"resuming from {ck} at {model.num_timesteps:,} steps, learning rate {model.learning_rate}, target kl {model.target_kl}, {model.n_steps * args.envs} samples per update", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=4096, n_epochs=5, learning_rate=args.lr, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=args.target_kl,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.ELU,
                                       log_std_init=-0.5),
                    device="cpu", verbose=0)
    start = time.time()
    model.learn(total_timesteps=args.steps - model.num_timesteps, callback=Examiner(out, 500_000),
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
