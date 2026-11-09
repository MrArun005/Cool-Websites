"""Make a brain blind to body velocity by copying one that sees it (DAgger, Ross et al. 2011).

    python distill_blind.py ../models/bolt-lite3-dog-149M runs/lite3-blind --body lite3 [--iters 40 --steps 40000]

Why: a real robot has no velocity sensor. The Lite3's CPU brains read the leg estimator's velocity and tolerate its
errors; its GPU brains did not (PLAN.md 9 Nov), and 149M itself falls in 2.5 s with the inputs zeroed. Training a
blind brain with PPO from there means relearning to balance. Copying is cheaper: the student (the teacher's own
network to start with) drives the training env with inputs 6:9 zeroed, the teacher labels every state it reaches
with what it would do given the true velocity, and the student regresses onto those labels (its value head onto the
teacher's value). The states are the student's own, so it learns to recover from its own mistakes.

Out: <out>/policy.zip + vecnormalize.pkl (the teacher's input scaling, unchanged), usable by
train_imitate.py --init-from <out> --blind-vel and `mjx_imitate.py warm` (then set blind_vel in the pkl: --blind).
Each iteration prints the student's falls per 1000 steps and the label error. All of it is simulation.
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", help="brain folder: policy.zip, vecnormalize.pkl")
    ap.add_argument("out")
    ap.add_argument("--body", default="lite3")
    ap.add_argument("--clips", nargs="*", default=["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"])
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--steps", type=int, default=40_000, help="env steps collected per iteration")
    ap.add_argument("--keep", type=int, default=600_000, help="newest samples kept")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--teacher-share", type=float, default=0.5, help="iteration 0: share of envs the teacher drives (0 after)")
    ap.add_argument("--noise", type=float, default=0.3, help="x the brain's own action noise while collecting (state variety)")
    ap.add_argument("--latency-ms", type=float, nargs=2, default=(0.0, 60.0))
    ap.add_argument("--moving-rate", type=float, default=0.75)
    ap.add_argument("--steer-max", type=float, default=0.8)
    ap.add_argument("--switch-rate", type=float, default=0.3)
    ap.add_argument("--motor-speed-x", type=float, default=1.5)
    ap.add_argument("--hours", type=float)
    ap.add_argument("--student", help="continue from this student folder (default: start from the teacher's own network)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["BOLT_BODY"] = a.body
    os.environ.pop("BOLT_BLIND_VEL", None)   # the envs give the true velocity: the teacher needs it, we zero the student's
    if a.motor_speed_x:
        os.environ["BOLT_MOTOR_SPEED_X"] = str(a.motor_speed_x)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from imitate_env import ImitateEnv

    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    src, out = Path(a.teacher), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(ImitateEnv, n_envs=a.envs, seed=a.seed, vec_env_cls=SubprocVecEnv, env_kwargs=dict(
        clips=a.clips, switch_rate=a.switch_rate, steer_max=a.steer_max, latency_ms=tuple(a.latency_ms),
        moving_rate=a.moving_rate))
    norm = VecNormalize.load(str(src / "vecnormalize.pkl"), venv)
    mean, std, clip = norm.obs_rms.mean, np.sqrt(norm.obs_rms.var + norm.epsilon), norm.clip_obs
    teacher = PPO.load(src / "policy.zip", device="cpu")
    student = PPO.load(Path(a.student or src) / "policy.zip", device="cpu")
    tp, sp = teacher.policy, student.policy
    tp.set_training_mode(False)
    opt = torch.optim.Adam(sp.parameters(), lr=a.lr)
    lo, hi = venv.action_space.low, venv.action_space.high

    def scale(o):
        return torch.as_tensor(np.clip((o - mean) / std, -clip, clip), dtype=torch.float32)

    def blind(o):
        o = o.copy()
        o[:, 6:9] = 0.0
        return o

    def heads(policy, x):
        f = policy.mlp_extractor(policy.extract_features(x))
        return policy.action_net(f[0]), policy.value_net(f[1])

    X, Y, V = [], [], []
    obs = venv.reset()
    t0 = time.time()
    for it in range(a.iters):
        falls, n = 0, 0
        drive_teacher = np.arange(a.envs) < round(a.envs * a.teacher_share) if it == 0 and not a.student else np.zeros(a.envs, bool)
        sigma = a.noise * np.exp(sp.log_std.detach().numpy())
        while n < a.steps:
            with torch.no_grad():
                act_t, val_t = heads(tp, scale(obs))
                act_s, _ = heads(sp, scale(blind(obs)))
            X.append(blind(obs).astype(np.float32)); Y.append(act_t.numpy()); V.append(val_t.numpy())
            act = np.where(drive_teacher[:, None], act_t.numpy(), act_s.numpy())
            act = np.clip(act + sigma * np.random.randn(*act.shape), lo, hi)
            obs, _, done, infos = venv.step(act)
            falls += sum(bool(d) and not i.get("TimeLimit.truncated", False) for d, i in zip(done, infos))
            n += a.envs
        Xa, Ya, Va = (np.concatenate(v)[-a.keep:] for v in (X, Y, V))
        X, Y, V = [Xa], [Ya], [Va]
        xs, ys, vs = scale(Xa), torch.as_tensor(Ya), torch.as_tensor(Va)
        sp.set_training_mode(True)
        for _ in range(a.epochs):
            perm = torch.randperm(len(xs))
            tot, k = 0.0, 0
            for i in range(0, len(xs), 4096):
                j = perm[i:i + 4096]
                pa, pv = heads(sp, xs[j])
                la, lv = ((pa - ys[j]) ** 2).mean(), ((pv - vs[j]) ** 2).mean()
                loss = la + 0.05 * lv / max(1e-6, float(vs.var()))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(sp.parameters(), 1.0); opt.step()
                tot, k = tot + float(la.detach()), k + 1
        sp.set_training_mode(False)
        print(f"iter {it}: {len(xs):,} samples, student ends by falling {1000 * falls / n:.2f} per 1000 steps, action error "
              f"{np.sqrt(tot / k):.4f} (rms), {(time.time() - t0) / 60:.1f} min", flush=True)
        student.save(out / "policy.zip")
        shutil.copy(src / "vecnormalize.pkl", out / "vecnormalize.pkl")
        if (out / "STOP").exists() or (a.hours and time.time() - t0 > a.hours * 3600):
            break
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
