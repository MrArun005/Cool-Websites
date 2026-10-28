"""Train Bolt's get-up brain (getup_env.py) from scratch, examining every checkpoint and keeping the best.

    python train_getup.py --name getup1 --steps 15000000 [--resume runs/getup1/checkpoints/policy_X_steps.zip]

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

from train_jump import TimeGuard
from getup_env import EXAM_KINDS, GetUpEnv, GetUpPolicy, exam, exam_full


class Examiner(BaseCallback):
    """Every `every` steps: save a checkpoint, take the exam (stood up from each kind of fall), keep the best."""

    def __init__(self, out, every, goal="stand", full=False, episodes=10):
        super().__init__()
        self.exam_first = False
        self.out, self.every, self.next, self.best, self.goal = out, every, every, -1.0, goal
        # full=True (run "recover"): the exam is the whole recovery (self-right, tuck, stand: getup_env.exam_full)
        # from run "recover"'s tougher falls; the best brain is picked on the share that stood up, ties going
        # to the faster one (0.002 per second: less than one try in 60)
        self.full, self.episodes = full, episodes

    def _on_training_start(self):   # after a resume, the next exam is the next multiple of `every`
        self.next = (self.num_timesteps // self.every + 1) * self.every
        if self.exam_first and not (self.out / "checkpoints" / f"policy_{self.num_timesteps}_steps.zip").exists():
            self._exam(self.num_timesteps)   # the starting brain, on the same exam: the bar to beat
        best = self.out / "best" / "best.json"
        if best.exists():   # a resumed run keeps its best brain
            self.best = json.loads(best.read_text()).get("score", self.best)

    def _on_step(self):
        if self.num_timesteps < self.next:
            return not (self.out / "STOP").exists()
        self.next += self.every
        self._exam(self.num_timesteps)
        return not (self.out / "STOP").exists()

    def _exam(self, n):
        ck = self.out / "checkpoints"
        ck.mkdir(exist_ok=True)
        self.model.save(ck / f"policy_{n}_steps.zip")
        self.training_env.save(str(ck / f"policy_vecnormalize_{n}_steps.pkl"))
        pol = GetUpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl")
        if self.full:
            result = exam_full(pol, episodes=self.episodes)
            rates = [v for k, v in result.items() if k.startswith("stood_")]
            secs = [v if v is not None else 10.0 for k, v in result.items() if k.startswith("tsec_")]
            score = sum(rates) / len(rates) - 0.002 * sum(secs) / len(secs)
            result.update({f"right_{k}": v for k, v in exam(pol, episodes=self.episodes, kinds=EXAM_KINDS, goal=self.goal,
                                                            hard=True).items()})
        else:
            result = exam(pol, goal=self.goal)
            rates = [v for k, v in result.items() if k.startswith("up_")]
            score = sum(rates) / len(rates)
        print(f"checkpoint {n} ready: {result}", flush=True)
        if score > self.best:
            self.best = score
            best = self.out / "best"
            best.mkdir(exist_ok=True)
            shutil.copy(ck / f"policy_{n}_steps.zip", best / "policy.zip")
            shutil.copy(ck / f"policy_vecnormalize_{n}_steps.pkl", best / "vecnormalize.pkl")
            (best / "best.json").write_text(json.dumps(dict(steps=n, exam=result, score=score), indent=2))
            print(f"new best at {n}: score {score:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="getup")
    ap.add_argument("--steps", type=int, default=15_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resume")
    ap.add_argument("--goal", default="stand", choices=["stand", "selfright"])
    ap.add_argument("--every", type=int, default=500_000, help="steps between checkpoints (each is tested and shown live)")
    ap.add_argument("--init-from", help="a model folder (policy.zip + vecnormalize.pkl) to fine-tune, e.g. ../models/bolt-dog-selfright-17e-16.5M")
    ap.add_argument("--hard", action="store_true", help="run 'recover': tougher falls (kicks, drops, ice, a second kick) and wider physics")
    ap.add_argument("--full-exam", action="store_true", help="examine the whole recovery (self-right, tuck, stand) from the tough falls")
    ap.add_argument("--exam-episodes", type=int, default=10)
    ap.add_argument("--lr", type=float, help="learning rate (on --resume/--init-from too; default 3e-4 / the saved one)")
    ap.add_argument("--target-kl", type=float, help="PPO target kl (default 0.03 / the saved one)")
    ap.add_argument("--n-steps", type=int, help="steps per env per update (default 2048 / the saved one)")
    ap.add_argument("--hours", type=float, help="stop cleanly (saving a checkpoint) after this many hours")
    ap.add_argument("--exam-first", action="store_true", help="examine the starting brain too (it can stay the best)")
    args = ap.parse_args()
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    venv = make_vec_env(GetUpEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv, env_kwargs=dict(goal=args.goal, hard=args.hard))
    over = {k: v for k, v in dict(learning_rate=args.lr, target_kl=args.target_kl, n_steps=args.n_steps).items() if v is not None}
    if args.init_from and not args.resume:
        src = Path(args.init_from)
        venv = VecNormalize.load(str(src / "vecnormalize.pkl"), venv)
        model = PPO.load(src / "policy.zip", env=venv, device="cpu", custom_objects=over)
        args.resume = str(src / "policy.zip")
        print(f"fine-tuning {src} from {model.num_timesteps:,} steps, learning rate {model.learning_rate}, target kl {model.target_kl}", flush=True)
    elif args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")), venv)
        model = PPO.load(ck, env=venv, device="cpu", custom_objects=over)
        print(f"resuming from {ck} at {model.num_timesteps:,} steps, learning rate {model.learning_rate}, target kl {model.target_kl}", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        # 16k steps per update (2048 per env): runs 15c-15e with 4k steps per update broke down again and again
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps or 2048, batch_size=4096, n_epochs=5, learning_rate=3e-4, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0, target_kl=0.03,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.ELU,
                                       log_std_init=0.0),   # more exploration: rolling over needs big moves
                    device="cpu", verbose=0)
    start = time.time()
    cbs = [Examiner(out, args.every, args.goal, args.full_exam, args.exam_episodes)]
    cbs[0].exam_first = args.exam_first
    if args.hours:
        cbs.append(TimeGuard(args.hours, out))
    model.learn(total_timesteps=args.steps - model.num_timesteps, callback=cbs,
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
