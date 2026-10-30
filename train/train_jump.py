"""Train Bolt's jump brain (jump_env.py) from scratch, examining every checkpoint and keeping the best.

    python train_jump.py --name jump1 --steps 6000000 [--resume runs/jump1/checkpoints/policy_X_steps.zip] [--lr 1e-4]
    # pick best/ on the mid-stride hand-over (MasterMind's trot -> jump): 1.7 m/s from 1.1 m, legs mid-stride
    python train_jump.py ... --select midstride --select-speed 1.7 --select-dist 1.1 --select-heights 0.40 0.50 0.60

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
from bolt_env import add_brake_args, add_exam_gap_args, set_exam_gaps, set_motor_brake


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

    def __init__(self, out, every, heights=(0.15, 0.20, 0.25, 0.30), dist=0.55, max_steps=100, speed=None, extra=(),
                 select="still", mid=None):
        super().__init__()
        self.exam_first = False
        self.out, self.every, self.next, self.best, self.heights = out, every, every, -1.0, heights
        self.dist, self.max_steps, self.speed = dist, max_steps, speed
        # extra exams, (speed, dist) each, fair score only, keys "v<speed x10>_jump_<cm>" (e.g. v20_jump_50): run
        # jump-trot examines 1.5 m/s (main) and 2.0 m/s, and picks the best brain on all their fair jump keys
        self.extra = list(extra)
        # the mid-stride exam (28 Oct): the jump brain takes over from a trot with the legs mid-stride (MasterMind's
        # hand-over, research/handover_gap.py), not from the settled still-legs stance of the exams above.
        # mid = dict(speed, dist, heights, episodes, seed) or None (no mid-stride exam); keys "mid_v<speed x10>_<cm>".
        # select: which fair keys pick best/: "still" (the exams above, as before 28 Oct), "midstride" or "both".
        self.select, self.mid = select, mid

    def _on_training_start(self):   # after a resume, the next exam is the next multiple of `every`
        self.next = (self.num_timesteps // self.every + 1) * self.every
        if self.exam_first and not (self.out / "checkpoints" / f"policy_{self.num_timesteps}_steps.zip").exists():
            self._exam(self.num_timesteps)   # the starting brain, on the same exam: the bar to beat
        best = self.out / "best" / "best.json"
        if best.exists():   # a resumed run keeps its best brain (run 16i's reset to 0 at 44.5M and overwrote it)
            old = json.loads(best.read_text())
            if old.get("select", "still") == self.select:
                self.best = old.get("score", self.best)
            else:   # a score on another exam is not comparable: the next checkpoint becomes the best
                print(f"best/ was picked on the {old.get('select', 'still')} exam, now selecting on {self.select}: "
                      f"its score {old.get('score')} is not carried over", flush=True)

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
        result = exam(JumpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"), heights=self.heights,
                      dist=self.dist, max_steps=self.max_steps, speed=self.speed)
        for sp, di in self.extra:
            more = exam(JumpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"), heights=self.heights,
                        dist=di, max_steps=self.max_steps, speed=sp, strict_too=False)
            result.update({f"v{round(sp * 10)}_{k}": v for k, v in more.items()})
        still = [v for k, v in result.items() if "jump_" in k]   # the fair still-legs score(s)
        mid = []
        if self.mid:
            m = self.mid
            more = exam(JumpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl"), episodes=m["episodes"],
                        seed=m["seed"], heights=m["heights"], dist=m["dist"], max_steps=self.max_steps, speed=m["speed"],
                        strict_too=False, gait=True)
            tag = f"mid_v{round(m['speed'] * 10)}"
            mid = list(more.values())
            result.update({f"{tag}_{k.split('_')[-1]}": v for k, v in more.items()})   # no "jump_" in these keys
            result["mid_mean"] = sum(mid) / len(mid)
        fair = {"still": still, "midstride": mid, "both": still + mid}[self.select]   # the best brain is picked on these
        score = sum(fair) / len(fair)
        levels = self.training_env.get_attr("bar_level")
        print(f"checkpoint {n} ready: {result} (practice bars now {100 * sum(levels) / len(levels):.0f} cm)", flush=True)
        if mid:
            print(f"checkpoint {n} mid-stride: {self.mid['speed']} m/s from {self.mid['dist']} m, legs mid-stride, "
                  f"{result['mid_mean']:.2f} mean over {self.mid['heights']} m; still-legs mean {sum(still) / len(still):.2f}; "
                  f"selecting on {self.select}", flush=True)
        if result.get("jump_15", 1) == 0 and self.best > 0.3:   # collapsed (run 15c at 7M): say so loudly
            print(f"collapse warning at {n}: no clean 15 cm jumps (best score so far {self.best:.2f})", flush=True)
        if score > self.best:
            self.best = score
            best = self.out / "best"
            best.mkdir(exist_ok=True)
            shutil.copy(ck / f"policy_{n}_steps.zip", best / "policy.zip")
            shutil.copy(ck / f"policy_vecnormalize_{n}_steps.pkl", best / "vecnormalize.pkl")
            (best / "best.json").write_text(json.dumps(dict(steps=n, exam=result, score=score, select=self.select), indent=2))
            print(f"new best at {n}: score {score:.2f}", flush=True)


class TimeGuard(BaseCallback):
    """Stop cleanly after `hours` (a Kaggle session ends at 12 h): save a checkpoint and end training."""

    def __init__(self, hours, out):
        super().__init__()
        self.end, self.out = time.time() + hours * 3600, out

    def _on_step(self):
        if time.time() < self.end:
            return True
        ck = self.out / "checkpoints"
        ck.mkdir(exist_ok=True)
        n = self.num_timesteps
        self.model.save(ck / f"policy_{n}_steps.zip")
        self.training_env.save(str(ck / f"policy_vecnormalize_{n}_steps.pkl"))
        print(f"time guard: stopping at {n} steps after the time budget", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="jump")
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--resume")
    ap.add_argument("--init-from", help="a model folder (policy.zip + vecnormalize.pkl) to fine-tune, e.g. ../models/bolt-dog-jump-runup-16i-35.5M")
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (on --resume too)")
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--n-steps", type=int, default=512, help="steps per env per update (x envs = samples per update)")
    ap.add_argument("--real", action="store_true", help="the real hurdle (hurdle.py) and heading inputs: run 16 on")
    ap.add_argument("--widen-from", help="a 49-input jump brain folder to start from (with --real)")
    ap.add_argument("--approach", type=float, nargs=2, default=[0.45, 0.65], help="bar distance at the start (m), run 16i: 1.5 3.0")
    ap.add_argument("--max-steps", type=int, default=100, help="steps per try (50 per second); a run-up needs ~200")
    ap.add_argument("--approach-speed", type=float, default=0.0, help="half the tries start running at the bar, 1 m/s up to this (run 16i: 3)")
    ap.add_argument("--exam-speed", type=float, help="exam: arrive at the bar at this speed (m/s), e.g. 2.0 as from a trot or canter")
    ap.add_argument("--exam-dist", type=float, default=0.55, help="bar distance in the exam (m), run 16i: 2.0")
    ap.add_argument("--exam-heights", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30],
                    help="bar heights (m) of the exam, e.g. 0.25 0.30 0.40 0.50 for a brain that clears 30 cm")
    ap.add_argument("--bar-start", type=float, help="the coach's starting bar height (m), e.g. 0.20 for a skilled brain")
    ap.add_argument("--approach-speed-min", type=float, default=1.0, help="running tries start at this speed or more (jump-trot: 1.2)")
    ap.add_argument("--run-frac", type=float, default=0.5, help="share of tries that start running (jump-trot: 0.85)")
    ap.add_argument("--gait-start", type=float, default=0.0, help="share of running tries that start mid-stride in a trot/canter pose")
    ap.add_argument("--bar-max", type=float, default=0.50, help="highest practice bar (m); 16i: 0.50, jump-trot: 0.75")
    ap.add_argument("--exam2", type=float, nargs=2, action="append", default=[], metavar=("SPEED", "DIST"),
                    help="an extra fair exam at SPEED m/s from DIST m (repeatable), e.g. --exam2 2.0 1.5")
    ap.add_argument("--select", choices=["still", "midstride", "both"], default="still",
                    help="which fair exam keys pick best/: still = the still-legs exams (--exam-speed, --exam2; the "
                         "behaviour before 28 Oct), midstride = the mid-stride exam only, both = the mean of all")
    ap.add_argument("--select-gait", action="store_true", help="shorthand for --select midstride")
    ap.add_argument("--select-speed", type=float, help="mid-stride exam speed (m/s); default --exam-speed. Trot hand-over: 1.7")
    ap.add_argument("--select-dist", type=float, help="mid-stride exam bar distance (m); default --exam-dist. Trot hand-over: 1.1")
    ap.add_argument("--select-heights", type=float, nargs="+", help="mid-stride exam bar heights (m); default --exam-heights")
    ap.add_argument("--select-episodes", type=int, default=10, help="mid-stride exam tries per height (default 10, as the still-legs exam)")
    ap.add_argument("--select-seed", type=int, default=5000, help="mid-stride exam first seed (default 5000, as the still-legs exam)")
    ap.add_argument("--no-midstride-log", action="store_true",
                    help="skip the mid-stride exam (it runs at every checkpoint whenever a speed is known, to log its score)")
    ap.add_argument("--handover-file", help="real hand-over states (research/record_handover.py) to start tries from")
    ap.add_argument("--handover-rate", type=float, default=0.0, help="share of training tries started from --handover-file")
    ap.add_argument("--latency-min-ms", type=float, default=0.0, help="per-try control delay drawn in [min, max] ms")
    ap.add_argument("--latency-max-ms", type=float, default=0.0, help="0: off; a real robot has ~10-20 ms")
    ap.add_argument("--motor-lag-max-ms", type=float, default=0.0, help="per-try motor lag drawn in [0, max] ms (0: off)")
    ap.add_argument("--drop-max", type=float, default=0.0, help="per-try share of lost control packets in [0, max] (0: off)")
    ap.add_argument("--latency-jitter-ms", type=float, default=0.0, help="per try the delay wobbles +- up to U(0, x) ms per step")
    ap.add_argument("--latency-spike-p", type=float, default=0.0, help="per try a delay-spike chance per step in U(0, x)")
    ap.add_argument("--every", type=int, default=500_000, help="steps between checkpoints and exams")
    ap.add_argument("--hours", type=float, help="stop cleanly (saving a checkpoint) after this many hours")
    ap.add_argument("--exam-first", action="store_true", help="examine the starting brain too (it can stay the best)")
    add_exam_gap_args(ap)
    add_brake_args(ap, default=True)
    args = ap.parse_args()
    print("motor brake (on, rad/s, N·m per rad/s):", set_motor_brake(args.motor_brake, args.brake_speed, args.brake_gain), flush=True)
    out = Path(__file__).with_name("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    venv = make_vec_env(JumpEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv, env_kwargs=dict(real=args.real, approach=tuple(args.approach), max_steps=args.max_steps,
                                                                                           approach_speed=args.approach_speed,
                                                                                           approach_speed_min=args.approach_speed_min, run_frac=args.run_frac,
                                                                                           gait_start=args.gait_start, bar_max=args.bar_max,
                                                                                           handover_file=args.handover_file, handover_rate=args.handover_rate,
                                                                                           latency_ms=(args.latency_min_ms, args.latency_max_ms),
                                                                                           motor_lag_ms=(0.0, args.motor_lag_max_ms), drop=(0.0, args.drop_max),
                                                                                           latency_jitter_ms=args.latency_jitter_ms, latency_spike_p=args.latency_spike_p))
    print("exam gaps:", set_exam_gaps(args.exam_latency_ms, args.exam_lag_ms, args.exam_drop, args.exam_jitter_ms, args.exam_spike_p), flush=True)
    if args.widen_from and not args.resume:
        args.resume = str(widen(args.widen_from, out / "widened"))
    if args.init_from and not args.resume:   # a model folder: policy.zip + vecnormalize.pkl
        args.resume = str(Path(args.init_from) / "policy.zip")
        stats = Path(args.init_from) / "vecnormalize.pkl"
    elif args.resume:
        ck = Path(args.resume)
        stats = ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
    if args.resume:
        ck = Path(args.resume)
        venv = VecNormalize.load(str(stats), venv)
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
    print("motor brake in the training envs (on, rad/s, N·m per rad/s):", venv.get_attr("motor_brake")[0], flush=True)
    start = time.time()
    select = "midstride" if args.select_gait else args.select
    mid_speed = args.select_speed if args.select_speed is not None else args.exam_speed
    mid = None if (args.no_midstride_log or not mid_speed) else dict(
        speed=mid_speed, dist=args.select_dist if args.select_dist is not None else args.exam_dist,
        heights=tuple(args.select_heights or args.exam_heights), episodes=args.select_episodes, seed=args.select_seed)
    if select != "still" and not mid:
        ap.error("--select midstride/both needs a mid-stride exam: give --select-speed or --exam-speed")
    print(f"best/ picked on the {select} exam; mid-stride exam: {mid}", flush=True)
    cbs = [Examiner(out, args.every, tuple(args.exam_heights), args.exam_dist, args.max_steps, args.exam_speed,
                    [tuple(e) for e in args.exam2], select=select, mid=mid)]
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
