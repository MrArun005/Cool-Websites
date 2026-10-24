"""Roll from the back onto the belly, so the face-down get-up (runtime "get up", 20/20) can take over.

    python back_roll.py [--workers 2] [--iters 150] [--keys 6]

Cross-entropy search over K keyframes in the policy's own action space (servo targets, so the runtime can
replay them: runtime.KeyframeBrain), each with its own duration, from lying flat on the back (nose up 90 deg,
legs at home, settled). The legs may move asymmetrically (hip roll / yaw kicks): rolling sideways is the way
over. Score over the last 0.3 s, from the IMU gravity in the observation (obs[0:3], upright = (0, 0, -1)):
    gravity x (nose pointing at the floor, +1 face down) + 2 if face down (x > 0.5 and lying)
The best script is then re-run from randomized back starts (partial roll +-45 deg, randomized bodies).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from backkick import BackKickEnv, _qmul, _quat

HERE = Path(__file__).resolve().parent
K = 6
_env = None


def env():
    global _env
    if _env is None:
        _env = BackKickEnv(task="get_up", randomize=False, seed=0, model_path=str(HERE / "model" / "duck_mid.xml"),
                           practice=False, max_seconds=8)
    return _env


def on_back(e, rng, roll=0.0, randomize=False):
    e.randomize = randomize
    e.reset(seed=int(rng.integers(1 << 30)))
    m, d = e.model, e.data
    q = e.home_q.copy()
    q[3:7] = _qmul(_quat([0, 0, 1], rng.uniform(-np.pi, np.pi)), _qmul(_quat([0, 1, 0], -np.pi / 2), _quat([1, 0, 0], roll)))
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q
    d.qpos[2] = 0.12
    d.ctrl[:] = q[e.qadr]
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=int(1.0 / m.opt.timestep))
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    e.pending = [d.ctrl.copy() for _ in e.pending]
    return e._obs()


def schedule(x, n, dt):
    keys = np.clip(x[:K * n].reshape(K, n), -1, 1)
    durs = np.clip(x[K * n:], 0.06, 0.8)
    return keys, durs


def rollout(x, seed=0, roll=0.0, randomize=False):
    e = env()
    rng = np.random.default_rng(seed)
    obs = on_back(e, rng, roll, randomize)
    n = e.action_space.shape[0]
    keys, durs = schedule(x, n, e.dt)
    prev = np.clip((e.data.ctrl - e.home)[e.pol] / e.action_scale, -1, 1)
    seq = []
    for k in range(K):
        s = max(1, int(durs[k] / e.dt))
        seq += [prev + (keys[k] - prev) * (i + 1) / s for i in range(s)]
        prev = keys[k]
    seq += [keys[-1]] * 40
    tail = []
    for t, a in enumerate(seq):
        obs, _, term, trunc, info = e.step(a)
        if t >= len(seq) - 15:
            gx, gz = float(obs[0]), float(obs[2])
            tail.append(gx + (2.0 if gx > 0.5 and -gz < 0.6 else 0.0))
    return float(np.mean(tail))


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--keys", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "model" / "back_roll.json"))
    a = ap.parse_args()
    K = a.keys
    e = env()
    n = e.action_space.shape[0]
    rng = np.random.default_rng(a.seed)
    mu = np.concatenate([np.zeros(K * n), np.full(K, 0.3)])
    sd = np.concatenate([np.full(K * n, 0.8), np.full(K, 0.2)])
    best = (-9.0, None)
    with Pool(a.workers, initializer=_set_k, initargs=(K,)) as pool:
        for it in range(a.iters):
            X = mu + sd * rng.standard_normal((a.pop, mu.size))
            S = np.array(pool.map(rollout, list(X)))
            idx = np.argsort(-S)[:8]
            if S[idx[0]] > best[0]:
                best = (float(S[idx[0]]), X[idx[0]].copy())
            mu = 0.7 * X[idx].mean(0) + 0.3 * mu
            sd = np.maximum(0.8 * X[idx].std(0) + 0.2 * sd, 0.1)
            if it % 10 == 0:
                print(f"iter {it}: best {best[0]:.2f}", flush=True)
            if best[0] > 2.9:
                break
        rob = pool.starmap(rollout, [(best[1], 100 + s, float(np.random.default_rng(s).uniform(-0.8, 0.8)), True) for s in range(20)])
    ok = sum(r > 2.5 for r in rob)
    print(f"best {best[0]:.2f} | face down from 20 randomized back starts: {ok}/20", flush=True)
    keys, durs = schedule(best[1], n, e.dt)
    Path(a.out).write_text(json.dumps(dict(keys=keys.round(4).tolist(), durations=durs.round(3).tolist(), score=best[0],
                                           randomized_face_down=f"{ok}/20", action_space="policy actions (home + scale x a)"), indent=1))
    print("wrote", a.out)


def _set_k(k):
    global K
    K = k


if __name__ == "__main__":
    main()
