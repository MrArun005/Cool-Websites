"""Train Bolt's dog-imitation brain (imitate_env.py) from scratch, examining every checkpoint and keeping the best.

    python train_imitate.py --name imitate18 --steps 30000000 [--resume runs/imitate18/checkpoints/policy_X_steps.zip]
    python train_imitate.py --name imitate_smoke --steps 300000 --envs 2 --exam-every 100000   # smoke test
    python train_imitate.py --body go2 --name go2_smoke --clips walk trot --steps 300000 --envs 2 --exam-every 100000

--body go2 trains the Unitree Go2 (robot_bodies.BODIES; its own clips dogmotion/clips/go2_*.npz) instead of Bolt; it sets BOLT_BODY
for this process and its envs, so the exams and ImitatePolicy load the same body. Run a Go2 brain's exams with
BOLT_BODY=go2 too.

Writes runs/<name>/checkpoints/, runs/<name>/best/ (policy.zip, vecnormalize.pkl, best.json) and prints
"checkpoint N ready: {...}" / "new best" lines as it goes (best = imitate_env.score: tracking x share of the
dog's pace kept), plus one "update" line per PPO update with the
mean tracking reward per step of the tries that ended in it (0..1: 1 = the dog's motion exactly) and the
mean try length.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from imitate_env import ALL_CLIPS, DEFAULT_CLIPS, STEER, STEER_CLIPS, ImitateEnv, ImitatePolicy, exam, exam_steer, score
from bolt_env import add_brake_args, set_motor_brake, set_motor_speed
from robot_bodies import BODIES


class Examiner(BaseCallback):
    """Every `every` steps: save a checkpoint, take the exam (tracking score and falls per clip), keep the best.
    After every update: print the training tries' mean reward per step and length."""

    def __init__(self, out, every, clips, steer_max=0.0, hours=None, exam_first=False):
        super().__init__()
        self.deadline = time.time() + hours * 3600 if hours else None   # Kaggle time guard: exam, save and stop
        self.exam_first = exam_first
        self.out, self.every, self.next, self.best, self.clips = out, every, every, -1.0, clips
        self.steer_max = steer_max   # > 0: also the steering exam, and the best brain must steer too

    def _on_training_start(self):   # after a resume, the next exam is the next multiple of `every`
        self.next = self.num_timesteps if self.exam_first else (self.num_timesteps // self.every + 1) * self.every
        self.t0 = time.time()
        best = self.out / "best" / "best.json"
        if best.exists():   # a resumed run keeps its best brain instead of overwriting it with the next exam
            self.best = json.loads(best.read_text()).get("score", self.best)

    def _on_rollout_end(self):
        eps = list(self.model.ep_info_buffer)
        if eps:
            per_step = np.mean([e["r"] / e["l"] for e in eps])
            length = np.mean([e["l"] for e in eps])
            fps = self.num_timesteps / max(1e-6, time.time() - self.t0)
            print(f"update at {self.num_timesteps}: reward/step {per_step:.3f}, try length {length:.0f} steps "
                  f"({len(eps)} tries), {fps:.0f} steps/s", flush=True)

    def _on_step(self):
        late = self.deadline is not None and time.time() > self.deadline
        if self.num_timesteps < self.next and not late:
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
        if self.steer_max:   # steering runs: score x (0.5 + 0.5 x how well the turns follow the command)
            sc = [c for c in self.clips if c in STEER_CLIPS and c in ("trot", "canter", "walk")]
            r = self.steer_max
            st = exam_steer(ImitatePolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"), clips=sc,
                            rates=(-r, -r / 2, r / 2, r))
            result.update(st)
            s = s * (0.5 + 0.5 * st["steer_mean"])
        print(f"checkpoint {n} ready: {result} (mean tracking {np.mean(tracks):.3f}, fall rate {np.mean(falls):.2f}, "
              f"score {s:.3f})", flush=True)
        if s > self.best:
            self.best = s
            best = self.out / "best"
            best.mkdir(exist_ok=True)
            shutil.copy(ck / f"policy_{n}_steps.zip", best / "policy.zip")
            shutil.copy(ck / f"policy_vecnormalize_{n}_steps.pkl", best / "vecnormalize.pkl")
            body = os.environ.get("BOLT_BODY", "bolt")
            (best / "best.json").write_text(json.dumps(dict(steps=n, exam=result, score=s, clips=self.clips,
                                                            **({"body": body} if body != "bolt" else {})), indent=2))
            print(f"new best at {n}: score {s:.3f}", flush=True)
        if late:
            print(f"time guard: stopping at {n}", flush=True)
        return not (self.out / "STOP").exists() and not late


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="imitate")
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resume")
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--switch-rate", type=float, default=0.0, help="chance per second of switching clip mid-try (run 18e: 0.3)")
    ap.add_argument("--slots", type=int, default=len(ALL_CLIPS), help="clip one-hot size: 8 for runs begun before the getup clip")
    ap.add_argument("--exam-every", "--every", type=int, default=1_000_000)
    ap.add_argument("--hours", type=float, help="time guard: exam, save and stop after this long (Kaggle)")
    ap.add_argument("--init-from", help="start from this brain folder (policy.zip, vecnormalize.pkl) at its step count, "
                    "with --lr (default 1e-4): a fine-tune (make_train_kernel.py)")
    ap.add_argument("--exam-first", action="store_true", help="exam the starting brain first (step 0 of the run)")
    ap.add_argument("--lr", type=float, help="on --resume: a new learning rate (fine-tune, e.g. 1e-4; default keeps the saved 3e-4)")
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=2, help="torch threads")
    ap.add_argument("--root-vel-weight", type=float, help="reward weight of body velocity tracking (default 0.1, "
                    "DeepMimic); raise it if the exam's speed_ stays low while tracking is good")
    ap.add_argument("--steer-max", type=float, default=0.0, help="yaw-rate command range (rad/s) for steering at walk/trot/"
                    "pace/canter (30 Oct; 0 = none). Adds the steering exam to every checkpoint")
    ap.add_argument("--steer-share", type=float, default=STEER["share"], help="share of tries (and redraws) with a command")
    ap.add_argument("--steer-change", type=float, default=STEER["change"], help="chance per second of a new command")
    ap.add_argument("--steer-weight", type=float, default=STEER["weight"], help="reward weight of the steer term")
    ap.add_argument("--root-global", type=float, default=0.0, help="0..1 blend toward Peng 2020's global root terms (heading and path); 0 = our terms")
    add_brake_args(ap, default=True)
    ap.add_argument("--motor-speed", type=float, nargs=3, metavar=("RATED", "CUTOFF", "POWER"),
                    help="a faster motor drive (rad/s, rad/s, W), e.g. 30 35 700 for a Go2-class body (default: the A1's 21 25 420)")
    ap.add_argument("--latency-ms", type=float, nargs=2, default=(0.0, 0.0), metavar=("LO", "HI"),
                    help="control delay per try, uniform in [LO, HI] ms (ImitateEnv latency_ms; default none)")
    ap.add_argument("--body", choices=["bolt"] + sorted(BODIES), default=os.environ.get("BOLT_BODY", "bolt"),
                    help="the robot: bolt (default) or a real robot of robot_bodies.BODIES, e.g. go2 (the Unitree Go2: "
                         "its own clips, motors and stance)")
    args = ap.parse_args()
    os.environ["BOLT_BODY"] = args.body   # the training envs (subprocesses), the exams and ImitatePolicy all read it
    if args.body != "bolt":
        assert not args.motor_speed, "--motor-speed changes Bolt's drive; a real-robot body has its own motor model"
        print(f"body: {args.body} (motors, brake speeds and gains: robot_bodies.BODIES; --brake-speed/--brake-gain do not apply)",
              flush=True)
    if args.motor_speed:
        print("motor drive (rated rad/s, cutoff rad/s, W):", set_motor_speed(*args.motor_speed), flush=True)
        args.brake_speed = args.motor_speed[1]   # the brake starts where the faster drive fades out
    print("motor brake (on, rad/s, N·m per rad/s):", set_motor_brake(args.motor_brake, args.brake_speed, args.brake_gain), flush=True)
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    venv = make_vec_env(ImitateEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv, env_kwargs=dict(
        clips=args.clips, slots=args.slots, switch_rate=args.switch_rate, weights={"root_vel": args.root_vel_weight} if args.root_vel_weight else None, root_global=args.root_global, steer_max=args.steer_max,
        steer_share=args.steer_share, steer_change=args.steer_change, steer_weight=args.steer_weight,
        **({"latency_ms": tuple(args.latency_ms)} if args.latency_ms[1] > 0 else {})))
    if args.init_from and not args.resume:
        src = Path(args.init_from)
        venv = VecNormalize.load(str(src / "vecnormalize.pkl"), venv)
        model = PPO.load(src / "policy.zip", env=venv, device="cpu",
                         custom_objects=dict(learning_rate=args.lr or 1e-4, target_kl=args.target_kl))
        args.resume = True   # keep the step count (reset_num_timesteps=False)
        print(f"fine-tuning {src} from {model.num_timesteps:,} steps", flush=True)
    elif args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")), venv)
        model = PPO.load(ck, env=venv, device="cpu", custom_objects=dict(learning_rate=args.lr, target_kl=args.target_kl) if args.lr else None)
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
    print("motor brake in the training envs (on, rad/s, N·m per rad/s):", venv.get_attr("motor_brake")[0], flush=True)
    start = time.time()
    print(f"training on clips {args.clips}, {args.envs} envs, {model.n_steps * args.envs} samples per update", flush=True)
    model.learn(total_timesteps=args.steps - model.num_timesteps, callback=Examiner(out, args.exam_every, args.clips, args.steer_max, args.hours, args.exam_first),
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    print(f"done: {model.num_timesteps:,} steps in {(time.time() - start) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
