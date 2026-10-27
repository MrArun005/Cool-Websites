"""Train Bolt's dog-imitation brain (imitate_env.py) from scratch, examining every checkpoint and keeping the best.

    python train_imitate.py --name imitate18 --steps 30000000 [--resume runs/imitate18/checkpoints/policy_X_steps.zip]
    python train_imitate.py --name imitate_smoke --steps 300000 --envs 2 --exam-every 100000   # smoke test

Writes runs/<name>/checkpoints/, runs/<name>/best/ (policy.zip, vecnormalize.pkl, best.json) and prints
"checkpoint N ready: {...}" / "new best" lines as it goes (best = imitate_env.score: tracking x share of the
dog's pace kept), plus one "update" line per PPO update with the
mean tracking reward per step of the tries that ended in it (0..1: 1 = the dog's motion exactly) and the
mean try length.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from imitate_env import DEFAULT_CLIPS, ImitateEnv, ImitatePolicy, exam, score


class Examiner(BaseCallback):
    """Every `every` steps: save a checkpoint, take the exam (tracking score and falls per clip), keep the best.
    After every update: print the training tries' mean reward per step and length."""

    def __init__(self, out, every, clips):
        super().__init__()
        self.out, self.every, self.next, self.best, self.clips = out, every, every, -1.0, clips

    def _on_training_start(self):   # after a resume, the next exam is the next multiple of `every`
        self.next = (self.num_timesteps // self.every + 1) * self.every
        self.t0 = time.time()

    def _on_rollout_end(self):
        eps = list(self.model.ep_info_buffer)
        if eps:
            per_step = np.mean([e["r"] / e["l"] for e in eps])
            length = np.mean([e["l"] for e in eps])
            fps = self.num_timesteps / max(1e-6, time.time() - self.t0)
            print(f"update at {self.num_timesteps}: reward/step {per_step:.3f}, try length {length:.0f} steps "
                  f"({len(eps)} tries), {fps:.0f} steps/s", flush=True)

    def _on_step(self):
        if self.num_timesteps < self.next:
            return not (self.out / "STOP").exists()
        self.next += self.every
        ck = self.out / "checkpoints"
        ck.mkdir(exist_ok=True)
        n = self.num_timesteps
        self.model.save(ck / f"policy_{n}_steps.zip")
        self.training_env.save(str(ck / f"policy_vecnormalize_{n}_steps.pkl"))
        result = exam(ImitatePolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"), clips=self.clips)
        tracks = [v for k, v in result.items() if k.startswith("track_")]
        falls = [v for k, v in result.items() if k.startswith("fall_")]
        s = score(result)
        print(f"checkpoint {n} ready: {result} (mean tracking {np.mean(tracks):.3f}, fall rate {np.mean(falls):.2f}, "
              f"score {s:.3f})", flush=True)
        if s > self.best:
            self.best = s
            best = self.out / "best"
            best.mkdir(exist_ok=True)
            shutil.copy(ck / f"policy_{n}_steps.zip", best / "policy.zip")
            shutil.copy(ck / f"policy_vecnormalize_{n}_steps.pkl", best / "vecnormalize.pkl")
            (best / "best.json").write_text(json.dumps(dict(steps=n, exam=result, score=s, clips=self.clips), indent=2))
            print(f"new best at {n}: score {s:.3f}", flush=True)
        return not (self.out / "STOP").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="imitate")
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resume")
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--exam-every", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=2, help="torch threads")
    ap.add_argument("--root-vel-weight", type=float, help="reward weight of body velocity tracking (default 0.1, "
                    "DeepMimic); raise it if the exam's speed_ stays low while tracking is good")
    args = ap.parse_args()
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    venv = make_vec_env(ImitateEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv, env_kwargs=dict(
        clips=args.clips, weights={"root_vel": args.root_vel_weight} if args.root_vel_weight else None))
    if args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")), venv)
        model = PPO.load(ck, env=venv, device="cpu")
        print(f"resuming from {ck} at {model.num_timesteps:,} steps", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.95)
        # 2048 steps per env per update (16k with 8 envs): runs 15c-15e with 4k per update broke down again and
        # again. gamma 0.95 as in Peng et al. 2020 (tracking pays every step; no far-off goal to plan for).
        model = PPO("MlpPolicy", venv, n_steps=2048, batch_size=4096, n_epochs=5, learning_rate=3e-4, gamma=0.95,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=0.03,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256]), activation_fn=torch.nn.ELU,
                                       log_std_init=-1.0),
                    device="cpu", verbose=0)
    start = time.time()
    print(f"training on clips {args.clips}, {args.envs} envs, {model.n_steps * args.envs} samples per update", flush=True)
    model.learn(total_timesteps=args.steps - model.num_timesteps, callback=Examiner(out, args.exam_every, args.clips),
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
