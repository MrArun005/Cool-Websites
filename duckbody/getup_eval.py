"""Honest get_up exam: one checkpoint, each start kind separately, randomized bodies.

    python getup_eval.py runs/mid_get_up/checkpoints/policy_2000000_steps.zip [--seeds 10] [--kinds front,back]

Success = upright (up > 0.95) at > 0.9 x standing height at the END of the 6 s episode.
Also reports how far it got: best up, best height, final up / height, time of first success.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backkick import BackKickEnv

HERE = Path(__file__).resolve().parent
KINDS = ("stand", "sit", "front", "back")


def load(ckpt):
    ckpt = Path(ckpt)
    stats = ckpt.with_name("vecnormalize.pkl")
    if not stats.exists():
        stats = ckpt.with_name(ckpt.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
    model = PPO.load(ckpt, device="cpu")
    dummy = DummyVecEnv([lambda: BackKickEnv(task="get_up", model_path=str(HERE / "model/duck_mid.xml"))])
    vn = VecNormalize.load(str(stats), dummy)
    vn.training = False
    return model, vn


def run(model, vn, kind, seed, env=None, record=None):
    env = env or BackKickEnv(task="get_up", practice=False, randomize=True, model_path=str(HERE / "model/duck_mid.xml"))
    obs, _ = env.reset(seed=seed, options={"kind": kind})
    assert env.start_kind == kind
    best_up, best_h, first_ok, done, info = -1.0, 0.0, None, False, None
    while not done:
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs, r, term, trunc, info = env.step(a[0])
        done = term or trunc
        best_up, best_h = max(best_up, info["up"]), max(best_h, info["height"])
        if info["success"] and first_ok is None:
            first_ok = env.t * env.dt
        if record is not None:
            record.append(env.data.qpos.copy())
    return dict(ok=bool(info["success"]), best_up=best_up, best_h=best_h / env.stand_height,
                up=info["up"], h=info["height"] / env.stand_height, t_ok=first_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--json", help="append the summary as one JSON line to this file")
    a = ap.parse_args()
    model, vn = load(a.ckpt)
    env = BackKickEnv(task="get_up", practice=False, randomize=True, model_path=str(HERE / "model/duck_mid.xml"))
    summary = dict(ckpt=str(a.ckpt))
    for kind in a.kinds.split(","):
        rs = [run(model, vn, kind, 1000 + s, env) for s in range(a.seeds)]
        k = dict(success=sum(r["ok"] for r in rs), n=len(rs),
                 best_up=float(np.mean([r["best_up"] for r in rs])), best_h=float(np.mean([r["best_h"] for r in rs])),
                 final_up=float(np.mean([r["up"] for r in rs])), final_h=float(np.mean([r["h"] for r in rs])),
                 reached=sum(r["t_ok"] is not None for r in rs))
        summary[kind] = k
        print(f"{kind:6s} {k['success']:2d}/{k['n']}  reached-ever {k['reached']:2d}  best up {k['best_up']:+.2f} "
              f"best h {k['best_h']:.2f}  final up {k['final_up']:+.2f} h {k['final_h']:.2f}  "
              f"[{' '.join('%+.1f' % r['up'] for r in rs)}]", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    sys.exit(main())
