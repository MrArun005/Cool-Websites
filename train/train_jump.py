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


def widen(src, out):
    """A 49-input jump brain -> 51 inputs for the real hurdle (run 16): the two new inputs (heading error,
    distance off the line) start with zero weights and a neutral normalizer, so it jumps exactly as before."""
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old = PPO.load(src / "policy.zip", device="cpu")
    env = DummyVecEnv([lambda: JumpEnv(real=True)])
    new = PPO("MlpPolicy", env, device="cpu", policy_kwargs=old.policy_kwargs, n_steps=old.n_steps,
              batch_size=old.batch_size, n_epochs=old.n_epochs, gamma=old.gamma, gae_lambda=old.gae_lambda,
              clip_range=old.clip_range, ent_coef=old.ent_coef, max_grad_norm=old.max_grad_norm, target_kl=old.target_kl)
    state, widened = old.policy.state_dict(), {}
    for k, v in new.policy.state_dict().items():
        w = state[k]
        if w.shape != v.shape:   # a first layer: the new inputs' columns start at zero
            w = torch.cat([w, torch.zeros(w.shape[0], v.shape[1] - w.shape[1])], dim=1)
        widened[k] = w
    new.policy.load_state_dict(widened)
    new.num_timesteps = old.num_timesteps
    new.save(out / "policy_{}_steps.zip".format(old.num_timesteps))
    norm = VecNormalize.load(str(src / "vecnormalize.pkl"), DummyVecEnv([lambda: JumpEnv()]))
    norm.obs_rms.mean = np.concatenate([norm.obs_rms.mean, np.zeros(2)])
    norm.obs_rms.var = np.concatenate([norm.obs_rms.var, np.ones(2)])
    norm.observation_space = env.observation_space
    norm.save(str(out / "policy_vecnormalize_{}_steps.pkl".format(old.num_timesteps)))
    print(f"widened {src} (49 -> 51 inputs, {old.num_timesteps:,} steps) -> {out}", flush=True)
    return out / "policy_{}_steps.zip".format(old.num_timesteps)


class Examiner(BaseCallback):
    """Every `every` steps: save a checkpoint, take the exam, keep the best."""

    def __init__(self, out, every, heights=(0.15, 0.20, 0.25, 0.30), dist=0.55, max_steps=100):
        super().__init__()
        self.out, self.every, self.next, self.best, self.heights = out, every, every, -1.0, heights
        self.dist, self.max_steps = dist, max_steps

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
        result = exam(JumpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"), heights=self.heights,
                      dist=self.dist, max_steps=self.max_steps)
        fair = [v for k, v in result.items() if k.startswith("jump_")]   # the best brain is picked on the fair score
        score = sum(fair) / len(fair)
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
    ap.add_argument("--real", action="store_true", help="the real hurdle (hurdle.py) and heading inputs: run 16 on")
    ap.add_argument("--widen-from", help="a 49-input jump brain folder to start from (with --real)")
    ap.add_argument("--approach", type=float, nargs=2, default=[0.45, 0.65], help="bar distance at the start (m), run 16i: 1.5 3.0")
    ap.add_argument("--max-steps", type=int, default=100, help="steps per try (50 per second); a run-up needs ~200")
    ap.add_argument("--approach-speed", type=float, default=0.0, help="half the tries start running at the bar, 1 m/s up to this (run 16i: 3)")
    ap.add_argument("--exam-dist", type=float, default=0.55, help="bar distance in the exam (m), run 16i: 2.0")
    ap.add_argument("--exam-heights", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30],
                    help="bar heights (m) of the exam, e.g. 0.25 0.30 0.40 0.50 for a brain that clears 30 cm")
    ap.add_argument("--bar-start", type=float, help="the coach's starting bar height (m), e.g. 0.20 for a skilled brain")
    args = ap.parse_args()
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    venv = make_vec_env(JumpEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv, env_kwargs=dict(real=args.real, approach=tuple(args.approach), max_steps=args.max_steps,
                                                                                           approach_speed=args.approach_speed))
    if args.widen_from and not args.resume:
        args.resume = str(widen(args.widen_from, out / "widened"))
    if args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")), venv)
        model = PPO.load(ck, env=venv, device="cpu", custom_objects=dict(learning_rate=args.lr, target_kl=args.target_kl, n_steps=args.n_steps))
        venv.env_method("__setattr__", "env_steps", model.num_timesteps // args.envs)  # keep the bar-height curriculum
        if args.bar_start:
            venv.env_method("__setattr__", "bar_level", args.bar_start)
        print(f"resuming from {ck} at {model.num_timesteps:,} steps, learning rate {model.learning_rate}, target kl {model.target_kl}, {model.n_steps * args.envs} samples per update", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=4096, n_epochs=5, learning_rate=args.lr, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=args.target_kl,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.ELU,
                                       log_std_init=-0.5),
                    device="cpu", verbose=0)
    start = time.time()
    model.learn(total_timesteps=args.steps - model.num_timesteps, callback=Examiner(out, 500_000, tuple(args.exam_heights), args.exam_dist, args.max_steps),
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
