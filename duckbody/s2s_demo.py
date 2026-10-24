"""Search for ANY motion that stands the duck up from the swag seat, to show the brain (a demonstration).

    python s2s_demo.py --iters 40 --pop 64 --out runs/s2s_demo.json

Cross-entropy search over open-loop keyframes in the policy's own action space (so whatever it finds,
a brain can output): K keyframes × 11 servo targets in [-1, 1] plus K durations, linearly interpolated,
the last one held. Scored after 5 s in SwingUpEnv (no cut-off for rocking back):
    score = 4 × up + 2 × height progress (seat → squat, capped at 1) + 5 if upright at squat height or above,
(v1 scored 1 × up + 3 × progress up to 1.2: it found the RL brain's trap, 9 cm high leaning back 45°),
averaged over the last 0.5 s so a fall-through does not count. Hand-made scripts (~1,500 variants:
slow, fast, reversed, from the belly, from every stable seat, bigger joints) all tipped over.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

K = 5
DUR = (0.1, 1.2)
SECONDS = 5.0
_env = None


def _get_env():
    global _env
    if _env is None:
        from s2s_swing import SwingUpEnv
        _env = SwingUpEnv(task="sit_to_stand", randomize=False, seed=0, model_path="model/duck_mid.xml",
                          practice=False, max_seconds=SECONDS)
    return _env


def seat_action():
    env = _get_env()
    env.reset(seed=0)
    return np.clip((env.data.ctrl - env.home)[env.pol] / env.action_scale, -1, 1)


def schedule(x, n_act, dt):
    keys = np.clip(x[:K * n_act].reshape(K, n_act), -1, 1)
    durs = np.clip(x[K * n_act:], *DUR)
    return keys, durs


def rollout(x, seed=0, randomize=False, frames=False):
    env = _get_env()
    env.randomize = randomize
    env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    keys, durs = schedule(x, n_act, env.dt)
    a_prev = np.clip((env.data.ctrl - env.home)[env.pol] / env.action_scale, -1, 1)
    seq = []
    for k in range(K):
        n = max(1, int(durs[k] / env.dt))
        seq += [a_prev + (keys[k] - a_prev) * (i + 1) / n for i in range(n)]
        a_prev = keys[k]
    steps = int(SECONDS / env.dt)
    tail, out = [], []
    for t in range(steps):
        a = seq[t] if t < len(seq) else keys[-1]
        obs, r, term, trunc, info = env.step(a)
        if frames:
            out.append((a.copy(), info["height"], info["up"]))
        if t >= steps - 25:
            prog = np.clip((info["height"] - env.seat_q[2]) / (env.crouch_height - env.seat_q[2]), 0, 1.0)
            ok = info["up"] > 0.9 and info["height"] > env.crouch_height - 0.01 * env.L and not env._trunk_on_floor()
            tail.append(4 * info["up"] + 2 * prog + 5 * ok)
        if term or trunc:
            break
    score = float(np.mean(tail)) if tail else -1.0 + info["up"]
    return (score, out) if frames else score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/s2s_demo.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys", type=int, default=5)
    a = ap.parse_args()
    global K
    K = a.keys
    env = _get_env()
    n_act = env.action_space.shape[0]
    seat = seat_action()
    mu = np.concatenate([np.tile(seat, K), np.full(K, 0.5)])
    sd = np.concatenate([np.full(K * n_act, 0.6), np.full(K, 0.3)])
    best = (-1e9, None)
    rng = np.random.default_rng(a.seed)
    with Pool(a.workers) as pool:
        for it in range(a.iters):
            X = mu + sd * rng.standard_normal((a.pop, mu.size))
            S = np.array(pool.map(rollout, list(X)))
            idx = np.argsort(-S)[:a.elite]
            if S[idx[0]] > best[0]:
                best = (float(S[idx[0]]), X[idx[0]].copy())
            mu = 0.7 * X[idx].mean(0) + 0.3 * mu
            sd = np.maximum(0.8 * X[idx].std(0) + 0.2 * sd, 0.15)   # floor 0.15: v1 collapsed to one motion by iter 4
            print(f"iter {it:2d}: best {S[idx[0]]:.2f} elite mean {S[idx].mean():.2f} overall best {best[0]:.2f}", flush=True)
            json.dump(dict(score=best[0], x=best[1].tolist(), K=K), open(a.out, "w"))
            if best[0] > 10.5:
                break
    # robustness: the best motion on randomized bodies
    rob = [rollout(best[1], seed=s, randomize=True) for s in range(10)]
    print("best score", round(best[0], 2), "| on 10 randomized bodies:", [round(r, 1) for r in rob],
          "stood up:", sum(r > 10 for r in rob), "/ 10")


if __name__ == "__main__":
    main()
