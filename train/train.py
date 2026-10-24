"""Train Bolt's motor skills with reinforcement learning (PPO).

    python train.py --steps 8000000          # ~1 hour on 4 CPU cores
    python train.py --steps 200000 --quick   # smoke test
    python train.py --resume runs/bolt/checkpoints/policy_1500000_steps.zip   # continue after a crash
    python train.py --resume ../models/bolt-dog-3M/policy.zip --name bolt-dog   # continue a packaged model

Saves the policy to runs/<name>/policy.zip and the observation statistics to
runs/<name>/vecnormalize.pkl. Watch progress with:
    tensorboard --logdir runs
"""
import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from bolt_env import BoltEnv


class StopFile(BaseCallback):
    """Stop training when watch.py decides the run has stopped improving (it writes <run>/STOP)."""

    def __init__(self, path, every=2048):
        super().__init__()
        self.path, self.every = path, every

    def _on_step(self):
        if self.n_calls % self.every == 0 and self.path.exists():
            print(f"stopping early: {self.path.read_text().strip()}")
            return False
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--envs", type=int, default=8, help="parallel simulated dogs")
    ap.add_argument("--name", default="bolt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--resume", help="checkpoint .zip to continue from; its vecnormalize .pkl must sit next to it")
    ap.add_argument("--skills", nargs="+", help="train a brain for only these commands, e.g. --skills walk")
    ap.add_argument("--gait-generator", nargs="?", const="crouch", default=None, choices=["crouch", "tall", "dog"],
                    help="the walk runs on the built-in trot, around this stance (no value = crouch, as in runs 5-6)")
    ap.add_argument("--turning", action="store_true", help="the walk also gets a turn command (53 inputs; see widen_brain.py)")
    ap.add_argument("--handover-brain", help="posture brain folder: half the walk episodes start with it standing the dog up")
    ap.add_argument("--speed-control", action="store_true", help="the walk also gets a speed command, up to a run (54 inputs)")
    args = ap.parse_args()

    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    venv = make_vec_env(BoltEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(skills=args.skills, gait_generator=args.gait_generator, turning=args.turning,
                                        handover_brain=args.handover_brain, speed_control=args.speed_control))
    if args.resume:
        ckpt_zip = Path(args.resume)
        stats = ckpt_zip.with_name("vecnormalize.pkl")  # packaged model folder (models/<name>/)
        if not stats.exists():                          # training checkpoint (runs/<name>/checkpoints/)
            stats = ckpt_zip.with_name(ckpt_zip.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
        venv = VecNormalize.load(str(stats), venv)
        model = PPO.load(ckpt_zip, env=venv, device="cpu", tensorboard_log=str(out.parent))
        venv.set_attr("total_steps", model.num_timesteps)  # keep the walk-speed curriculum where it was
        print(f"resuming from {ckpt_zip} at {model.num_timesteps:,} steps")
    else:
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
    remaining = args.steps - model.num_timesteps
    stop = StopFile(out / "STOP")
    model.learn(total_timesteps=remaining, callback=CallbackList([ckpt, stop]), tb_log_name=args.name,
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
