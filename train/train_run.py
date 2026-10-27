"""Train Bolt's running brain (run_env.py): fast running, turning and shove recovery, examined every checkpoint.

    python train_run.py --name run20 --steps 40000000 [--resume runs/run20/checkpoints/policy_X_steps.zip]

Writes runs/<name>/checkpoints/, runs/<name>/best/ (policy.zip, vecnormalize.pkl, best.json) and prints
"checkpoint N ready: {...}" / "new best" lines as it goes.
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

from run_env import RunEnv, RunPolicy, exam, score as run_score


class Examiner(BaseCallback):
    """Every `every` steps: save a checkpoint, take the running exam (speeds, turns, a shove), keep the best."""

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
        result = exam(RunPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"))
        score = run_score(result)
        tops = self.training_env.get_attr("top_v")
        print(f"checkpoint {n} ready: {result} (practising up to {max(tops):.1f} m/s, mean {sum(tops) / len(tops):.1f})", flush=True)
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
    ap.add_argument("--name", default="run20")
    ap.add_argument("--steps", type=int, default=40_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resume")
    ap.add_argument("--every", type=int, default=500_000, help="steps between checkpoints (each is tested and shown live)")
    args = ap.parse_args()
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    venv = make_vec_env(RunEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv)
    if args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")), venv)
        model = PPO.load(ck, env=venv, device="cpu")
        print(f"resuming from {ck} at {model.num_timesteps:,} steps", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        # 16k steps per update (2048 per env): runs 15c-15e with 4k steps per update broke down again and again
        model = PPO("MlpPolicy", venv, n_steps=2048, batch_size=4096, n_epochs=5, learning_rate=3e-4, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=0.03,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]), activation_fn=torch.nn.ELU,
                                       log_std_init=0.0),   # legged_gym's network size
                    device="cpu", verbose=0)
    start = time.time()
    model.learn(total_timesteps=args.steps - model.num_timesteps, callback=Examiner(out, args.every),
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
