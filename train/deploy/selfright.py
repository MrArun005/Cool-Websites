"""The self-right brain without torch or SB3 (numpy only), for the robot's own computer.

    python deploy/selfright.py export [models/bolt-go2-selfright-6.9M]   # policy.zip + vecnormalize.pkl -> brain.npz (needs SB3)
    python deploy/selfright.py test   [models/bolt-go2-selfright-6.9M]   # parity vs GetUpPolicy.act (needs SB3)
    (a folder named bolt-lite3-* is a Lite3 brain)

SelfRighter(folder).step(quat, gyro, q, dq, v_body, height) -> the 12 joint targets (our order) for the next 20 ms.
Inputs as getup_env._obs (46): gravity in the body frame, gyro x 0.25, body velocity, height, joints minus the stance,
joint speeds x 0.1, the previous action. Action a in [-1, 1]: target = stance + scale * a, scale = the full motor range
around the stance (getup_env goal "selfright"), clipped to the motor limits.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import robot_bodies  # noqa: E402
from controller import quat_to_mat  # noqa: E402

DEFAULT = HERE.parent.parent / "models" / "bolt-go2-selfright-6.9M"


class SelfRighter:
    def __init__(self, folder=DEFAULT, body="go2"):
        z = np.load(Path(folder) / "brain.npz")
        self.mean, self.std, self.clip_obs = z["mean"], z["std"], float(z["clip_obs"])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        self.body = robot_bodies.get(body)
        self.stance = np.asarray(self.body.stance, float)
        m = self.body.model()
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.scale = np.maximum(self.hi - self.stance, self.stance - self.lo)
        self.last_action = np.zeros(12)

    def act(self, obs):
        x = np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -self.clip_obs, self.clip_obs).astype(np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))   # ELU
        return np.clip(x, -1.0, 1.0)

    def inputs(self, quat, gyro, q, dq, v_body, height):
        R = quat_to_mat(np.asarray(quat, float))
        return np.concatenate([R.T @ np.array([0, 0, -1.0]), np.asarray(gyro) * 0.25, v_body, [height],
                               np.asarray(q) - self.stance, np.asarray(dq) * 0.1, self.last_action]).astype(np.float32)

    def step(self, quat, gyro, q, dq, v_body, height):
        a = self.act(self.inputs(quat, gyro, q, dq, v_body, height))
        self.last_action = np.asarray(a, float)
        return np.clip(self.stance + self.scale * a, self.lo, self.hi)


def export(folder, body="go2"):
    from getup_env import GetUpPolicy
    p = GetUpPolicy(folder, body=body)
    assert not p.delay_input, "a delay-input brain (47 inputs) needs the delay added to inputs()"
    act = p.model.policy_kwargs.get("activation_fn")
    assert act is not None and act.__name__ == "ELU", act
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in p.model.policy.state_dict().items()}
    idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("mlp_extractor.policy_net")})
    layers = [(sd[f"mlp_extractor.policy_net.{i}.weight"].T, sd[f"mlp_extractor.policy_net.{i}.bias"]) for i in idx]
    layers.append((sd["action_net.weight"].T, sd["action_net.bias"]))
    rms = p.norm.obs_rms
    out = dict(mean=rms.mean.astype(np.float32), std=np.sqrt(rms.var + p.norm.epsilon).astype(np.float32),
               clip_obs=np.float32(p.norm.clip_obs), n_layers=len(layers))
    for i, (W, b) in enumerate(layers):
        out[f"W{i}"], out[f"b{i}"] = W, b
    np.savez(Path(folder) / "brain.npz", **out)
    print(f"wrote {Path(folder) / 'brain.npz'}: {len(layers)} layers {[W.shape for W, _ in layers]}")


def test(folder, body="go2"):
    """Same action as the SB3 brain on real self-right inputs (the getup env's own episodes, all fall kinds)."""
    from getup_env import GetUpEnv, GetUpPolicy
    ref, ours = GetUpPolicy(folder, body=body), SelfRighter(folder, body=body)
    env = GetUpEnv(randomize=True, goal="selfright", body=body)
    worst, n = 0.0, 0
    for k, kind in enumerate(("side", "back", "tipped", "standing")):
        env.exam_start = kind
        obs, _ = env.reset(seed=50 + k)
        for _ in range(100):
            a_ref = np.asarray(ref.act(obs), float)
            worst = max(worst, float(np.abs(a_ref - ours.act(obs)).max()))
            n += 1
            obs, _, _, done, _ = env.step(a_ref)
            if done:
                break
    print(f"parity: {n} steps, worst action gap {worst:.2e}")
    assert worst < 1e-4, worst


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    folder = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT)
    body = "lite3" if "lite3" in Path(folder).name else "go2"   # (models/bolt-<body>-selfright-...)
    {"export": export, "test": test}[cmd](folder, body)
