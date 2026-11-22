"""A GPU brain (mjx_imitate params.pkl) -> an SB3 folder the CPU trainer can go on from (train_imitate.py --init-from).

    BOLT_BODY=go2 python gpu_to_sb3.py ../models/bolt-go2-torque-steer/params.pkl ../models/bolt-go2-torque-steer-sb3

The brain itself is copied exactly (checked by a round trip through mjx_imitate.from_sb3). The value network of a
privileged-critic brain saw 8 more inputs (mjx_imitate PRIV_EXTRA: friction, kp, mass, damping, 4 paw contacts) between
the base inputs and the history; those rows are dropped, as the CPU trainer's value network sees what the brain sees.
The return normalizer is set so SB3's reward scale equals the GPU run's reward scale (1/sqrt(var) = reward_scale).
PPO's settings come from a template SB3 folder of the same layout (default: the Lite3 one, made the same way on 18 Nov).
Simulation only.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--template", default=str(HERE.parent / "models" / "bolt-lite3-dog-gpu-1461M-sb3"))
    a = ap.parse_args()
    P = pickle.load(open(a.pkl, "rb"))
    body = P.get("body", "bolt")
    os.environ["BOLT_BODY"] = body
    k = int(P.get("history", 0))
    from mjx_imitate import HIST_SEL, PRIV_EXTRA, from_sb3
    n_obs = len(P["mean"])
    W0, b0 = P["value"][0]
    extra = W0.shape[0] - n_obs
    assert extra in (0, PRIV_EXTRA), (W0.shape, n_obs)
    if extra:
        base = n_obs - k * len(HIST_SEL)
        W0 = np.concatenate([W0[:base], W0[base + extra:]])
    Q = dict(P, value=[(W0, b0)] + list(P["value"][1:]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["BOLT_HISTORY"] = str(k)
    if P.get("blind_vel"):
        os.environ["BOLT_BLIND_VEL"] = "1"
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from imitate_env import make_env
    tpl = PPO.load(Path(a.template) / "policy.zip", device="cpu")
    venv = VecNormalize(DummyVecEnv([lambda: make_env()]), norm_obs=True, norm_reward=True, clip_obs=float(P.get("clip_obs", 10.0)),
                        gamma=tpl.gamma)
    assert venv.observation_space.shape[0] == n_obs, (venv.observation_space.shape, n_obs)
    venv.obs_rms.mean = np.asarray(P["mean"], np.float64)
    venv.obs_rms.var = np.maximum(np.asarray(P["std"], np.float64) ** 2 - venv.epsilon, 1e-12)
    venv.obs_rms.count = float(P.get("count", 1e6))
    # the return normalizer: SB3 divides rewards by sqrt(ret_rms.var + eps); the GPU run multiplied them by reward_scale
    venv.ret_rms.var = np.array(1.0 / float(P.get("reward_scale", 1.0)) ** 2 - venv.epsilon)
    venv.ret_rms.count = 1e6
    model = PPO("MlpPolicy", venv, n_steps=tpl.n_steps, batch_size=tpl.batch_size, n_epochs=tpl.n_epochs, learning_rate=1e-5,
                gamma=tpl.gamma, gae_lambda=tpl.gae_lambda, clip_range=0.2, ent_coef=tpl.ent_coef, max_grad_norm=tpl.max_grad_norm,
                target_kl=0.01, policy_kwargs=tpl.policy_kwargs, device="cpu", verbose=0)
    sd = model.policy.state_dict()
    for prefix, head, layers in (("mlp_extractor.policy_net", "action_net", Q["layers"]), ("mlp_extractor.value_net", "value_net", Q["value"])):
        idx = sorted({int(n.split(".")[2]) for n in sd if n.startswith(prefix)})
        for i, (W, b) in zip(idx, layers[:-1]):
            sd[f"{prefix}.{i}.weight"], sd[f"{prefix}.{i}.bias"] = torch.as_tensor(np.asarray(W).T.copy()), torch.as_tensor(np.asarray(b).copy())
        sd[f"{head}.weight"], sd[f"{head}.bias"] = torch.as_tensor(np.asarray(layers[-1][0]).T.copy()), torch.as_tensor(np.asarray(layers[-1][1]).copy())
    sd["log_std"] = torch.as_tensor(np.asarray(P["log_std"], np.float32))
    model.policy.load_state_dict(sd)
    model.num_timesteps = int(P["steps"])
    model.save(out / "policy.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    # round trip: the SB3 brain must be the same brain
    R = from_sb3(out / "policy.zip", out / "vecnormalize.pkl")
    d = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for (x, _), (y, _) in zip(R["layers"], P["layers"]))
    d = max(d, float(np.max(np.abs(np.asarray(R["mean"]) - np.asarray(P["mean"])))),
            float(np.max(np.abs(np.asarray(R["std"]) - np.asarray(P["std"])) / np.asarray(P["std"]))))
    print(f"{a.pkl} -> {out} (body {body}, history {k}, value inputs {W0.shape[0]}); round trip max difference {d:.2e}")
    assert d < 1e-4, d


if __name__ == "__main__":
    main()
