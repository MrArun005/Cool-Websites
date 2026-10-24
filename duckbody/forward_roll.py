"""Forward roll (roulade): from the squat, tuck the head and tumble forward over it onto the back; the
runtime's "roll get up" (roll-over brain onto the belly -> face-down get-up, 9/10) then stands.

    python forward_roll.py [--workers 1] [--iters 120] [--keys 5]

Cross-entropy search over K keyframes in the policy's action space (servo targets: the runtime can replay
them), each with a duration, started from the squat brain's crouch. Every candidate is scored on 3 starts
(randomized bodies) so a one-start fluke does not win (back_roll.py's lesson):
    forward pitch rotation of the trunk accumulated over the motion (rad, capped at pi)
  + 2 if it ends lying on its back (IMU gravity x < -0.5, tilted > 55 deg)
The best script is then re-run on 20 randomized bodies.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from skills_env import SkillEnv  # noqa: E402

K = 5
_env = _sq = None


def setup():
    global _env, _sq
    if _env is None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        _env = SkillEnv(task="squat", randomize=True, seed=0, model_path=str(HERE / "model" / "duck_mid.xml"),
                        practice=False, max_seconds=12)
        v = VecNormalize.load(str(HERE.parent / "models" / "mid-squat" / "vecnormalize.pkl"), DummyVecEnv([lambda: _env]))
        v.training = False
        p = PPO.load(str(HERE.parent / "models" / "mid-squat" / "policy.zip"), device="cpu")
        _sq = lambda o: p.predict(v.normalize_obs(o[None]), deterministic=True)[0][0]
    return _env, _sq


def rollout(x, seed=0):
    e, sq = setup()
    obs, _ = e.reset(seed=seed)
    for _ in range(100):                               # squat for 2 s
        obs, *_ = e.step(sq(obs))
    n = e.action_space.shape[0]
    keys = np.clip(x[:K * n].reshape(K, n), -1, 1)
    durs = np.clip(x[K * n:], 0.06, 0.8)
    prev = np.clip((e.data.ctrl - e.home)[e.pol] / e.action_scale, -1, 1)
    seq = []
    for k in range(K):
        s = max(1, int(durs[k] / e.dt))
        seq += [prev + (keys[k] - prev) * (i + 1) / s for i in range(s)]
        prev = keys[k]
    seq += [keys[-1]] * 50
    pitch = 0.0
    for a in seq:
        obs, *_ = e.step(a)
        wy = float(e.data.qvel[4])                     # trunk angular velocity about its own y (free joint, body frame)
        pitch += wy * e.dt
    g = obs[:3]
    on_back = g[0] < -0.5 and -g[2] < 0.6
    return float(min(abs(pitch), np.pi)) + (2.0 if on_back else 0.0), bool(on_back), round(float(pitch), 2)


def score(x):
    return float(np.mean([rollout(x, seed=2000 + i)[0] for i in range(3)]))


def _set_k(k):
    global K
    K = k
    import torch
    torch.set_num_threads(1)


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--keys", type=int, default=5)
    a = ap.parse_args()
    K = a.keys
    n = SkillEnv(task="squat", model_path=str(HERE / "model" / "duck_mid.xml")).action_space.shape[0]
    rng = np.random.default_rng(0)
    mu = np.concatenate([np.zeros(K * n), np.full(K, 0.3)])
    sd = np.concatenate([np.full(K * n, 0.8), np.full(K, 0.2)])
    best = (-9.0, None)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_set_k, initargs=(K,)) as pool:   # spawn: forking after
        # torch is loaded (the squat brain) deadlocked the worker (0% CPU for 12 min)
        for it in range(a.iters):
            X = mu + sd * rng.standard_normal((a.pop, mu.size))
            S = np.array(pool.map(score, list(X)))
            idx = np.argsort(-S)[:6]
            if S[idx[0]] > best[0]:
                best = (float(S[idx[0]]), X[idx[0]].copy())
            mu = 0.7 * X[idx].mean(0) + 0.3 * mu
            sd = np.maximum(0.8 * X[idx].std(0) + 0.2 * sd, 0.1)
            if it % 10 == 0:
                print(f"iter {it}: best {best[0]:.2f}", flush=True)
            if best[0] > 5.0:
                break
        R = pool.starmap(rollout, [(best[1], 100 + s) for s in range(20)])
    ok = sum(r[1] for r in R)
    print(f"best {best[0]:.2f} | onto its back over the head on 20 randomized bodies: {ok}/20 | pitch {[r[2] for r in R]}", flush=True)
    keys = np.clip(best[1][:K * n].reshape(K, n), -1, 1)
    durs = np.clip(best[1][K * n:], 0.06, 0.8)
    (HERE / "model" / "forward_roll.json").write_text(json.dumps(dict(keys=keys.round(4).tolist(), durations=durs.round(3).tolist(),
                                                                      score=best[0], randomized_on_back=f"{ok}/20"), indent=1))


if __name__ == "__main__":
    main()
