"""Test a walk / run checkpoint the way a person would look at it: does it go where it's told, do the
feet take turns, does it stay up?

    python walk_eval.py runs/mid_walk/checkpoints/policy_2000000_steps.zip [--task walk] [--out DIR]

Deterministic rollouts (randomized bodies and shoves, as in the exam) for a fixed set of commands
plus the exam's own random ones (seeds 123…). Per rollout: speed achieved vs commanded (body frame,
averaged after the first second), turn rate vs commanded, foot lifts L/R, alternation (fraction of
lift-offs that follow a lift-off of the other foot), fall, exam success.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from skills_env import GAIT_HZ
from walk_env import WalkEnv

HERE = Path(__file__).resolve().parent
MODEL = str(HERE / "model" / "duck_mid.xml")
FIXED = {"walk": [(0.12, 0, 0), (0.06, 0, 0), (0.17, 0, 0), (-0.05, 0, 0), (0.0, 0.05, 0), (0.0, 0, 0.5),
                  (0.0, 0, -0.5), (0.08, 0, 0.3), (0.0, 0, 0)],
         "run": [(0.21, 0, 0), (0.3, 0, 0), (0.38, 0, 0), (0.3, 0, 0.25), (0.25, 0, -0.25)]}


def load(ckpt):
    ckpt = Path(ckpt)
    stats = ckpt.with_name("vecnormalize.pkl")
    if not stats.exists():
        stats = ckpt.with_name(ckpt.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
    with open(stats, "rb") as f:
        import pickle
        vn = pickle.load(f)
    return PPO.load(ckpt, device="cpu"), vn


def run(model, vn, task, seed, cmd=None, seconds=None):
    env = WalkEnv(task=task, randomize=True, seed=seed, model_path=MODEL, practice=False, max_seconds=seconds)
    obs, _ = env.reset(seed=seed)
    if cmd is not None:
        env.vel_cmd = np.array(cmd, dtype=float)
        env.table = env.ref[task].fast_table(env.vel_cmd, GAIT_HZ[task])
        obs = env._obs()
    d = env.data
    done, prev, lifts, order = False, env.feet_on_ground(), np.zeros(2), []
    vx, vy, wz, ups = [], [], [], []
    total = 0.0
    while not done:
        o = np.clip((obs - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
        a, _ = model.predict(o[None].astype(np.float32), deterministic=True)
        obs, r, term, trunc, info = env.step(a[0])
        total += r
        f = env.feet_on_ground()
        for i in (0, 1):
            if prev[i] == 1 and f[i] == 0:
                lifts[i] += 1
                order.append(i)
        prev = f
        if env.t * env.dt > 1.0:
            vx.append(info["vx"]); vy.append(info["vy"]); wz.append(info["yaw_rate"])
        ups.append(info["up"])
        done = term or trunc
    alt = float(np.mean([order[k] != order[k - 1] for k in range(1, len(order))])) if len(order) > 1 else 0.0
    m = lambda x: round(float(np.mean(x)), 3) if x else 0.0
    return dict(cmd=[round(float(c), 3) for c in env.vel_cmd], vx=m(vx), vy=m(vy), wz=m(wz), lifts=lifts.astype(int).tolist(),
                alternation=round(alt, 2), fell=bool(term), seconds=round(env.t * env.dt, 2), min_up=round(min(ups), 2),
                success=bool(info["success"]), reward=round(total, 1))


def evaluate(ckpt, task="walk", exam_eps=8):
    model, vn = load(ckpt)
    fixed = [run(model, vn, task, 1000 + k, cmd) for k, cmd in enumerate(FIXED[task])]
    exam = [run(model, vn, task, 123 + k) for k in range(exam_eps)]
    summ = lambda rs: dict(success=round(float(np.mean([r["success"] for r in rs])), 3),
                           falls=int(sum(r["fell"] for r in rs)),
                           mean_reward=round(float(np.mean([r["reward"] for r in rs])), 1),
                           speed_err=round(float(np.mean([abs(r["vx"] - r["cmd"][0]) + abs(r["vy"] - r["cmd"][1]) for r in rs])), 3),
                           turn_err=round(float(np.mean([abs(r["wz"] - r["cmd"][2]) for r in rs])), 3),
                           lifts_per_10s=round(float(np.mean([sum(r["lifts"]) / r["seconds"] * 10 for r in rs])), 1),
                           alternation=round(float(np.mean([r["alternation"] for r in rs])), 2))
    return dict(ckpt=str(ckpt), task=task, fixed=fixed, exam=exam, fixed_summary=summ(fixed), exam_summary=summ(exam))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--task", default="walk")
    ap.add_argument("--out")
    ap.add_argument("--eps", type=int, default=8)
    a = ap.parse_args()
    res = evaluate(a.ckpt, a.task, a.eps)
    for r in res["fixed"]:
        print("fixed", r)
    for r in res["exam"]:
        print("exam ", r)
    print("FIXED", res["fixed_summary"])
    print("EXAM ", res["exam_summary"])
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / (Path(a.ckpt).stem + ".json")).write_text(json.dumps(res, indent=1))
