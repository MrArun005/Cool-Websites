"""Bundle what a web page needs to run Bolt live: the robot, both brains, and a reference run.

    python export_live.py --posture ../models/bolt-dog-posture-dog-2.5M \
                          --walk ../models/bolt-dog-walk-run-8.5M -o bolt_live.json \
                          --reference bolt_live_reference.json

The page (train/gaitlab/live.js) runs the same MuJoCo physics in the browser (the official
WebAssembly build, same version as here) and the same control loop as BoltEnv with the dog
gait, turning and speed control. Each brain is stored as its weights plus its observation
normalizer, so the page needs no machine-learning library: two 256-unit layers are a few
lines of JavaScript.

--reference writes a reference run from Python (stand with the posture brain, then walk,
turn and change speed with the walk brain), so the JavaScript port can be checked step by
step against the real thing (live_check.mjs).
"""
import argparse
import base64
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from bolt_env import ACTION_SCALE, MODEL, PAW_GAIT, STAND_POSE, STANCES, BoltEnv
from evaluate import Policy, reset_standing
from record3d import describe
from reference_gait import HIP_X, HIP_Y

# The reference run: (command, steps, turn rad/s, speed m/s); ends by braking, then standing
SCRIPT = [("stand", 100, 0.0, 0.4), ("walk", 150, 0.0, 0.4), ("walk", 150, 0.4, 0.7), ("walk", 150, -0.3, 1.0),
          ("walk", 100, 1.5, 0.3), ("walk", 50, 0.0, 0.2), ("stand", 100, 0.0, 0.2)]


def f32(a):
    """A float array as base64 little-endian float32 (a quarter the size of JSON numbers)."""
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def f64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f8").tobytes()).decode()


def brain(folder):
    model = PPO.load(Path(folder) / "policy.zip", device="cpu")
    with open(Path(folder) / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)  # VecNormalize: (obs - mean) / sqrt(var + eps), clipped to ±clip
    layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    assert all(isinstance(m, torch.nn.ELU) for m in model.policy.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear))
    return dict(inputs=int(model.observation_space.shape[0]),
                mean=f64(norm.obs_rms.mean), var=f64(norm.obs_rms.var), eps=float(norm.epsilon), clip=float(norm.clip_obs),
                layers=[dict(w=f32(m.weight.detach().numpy()), b=f32(m.bias.detach().numpy()),
                             out=int(m.out_features), inp=int(m.in_features)) for m in layers])


def reference(posture, walk):
    """A run in Python with no randomness, recorded step by step for checking the JavaScript port."""
    env = BoltEnv(randomize=False, max_steps=10**6, gait_generator="dog", turning=True, speed_control=True,
                  sharp_turns=walk.sharp_turns)
    env.forced_command = "stand"
    reset_standing(env, 0)
    d = env.data
    start = dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist())
    steps = []
    for command, n, turn, speed in SCRIPT:
        env.forced_command = command
        env.set_command(command)
        env.set_turn(turn)
        env.set_speed(speed)
        for _ in range(n):
            obs = env._obs()
            action = (walk if command == "walk" else posture).act(obs)
            env.step(action)
            steps.append(dict(obs=obs.tolist(), action=np.asarray(action, dtype=float).tolist(), qpos=d.qpos.tolist()))
    return start, dict(start=start, script=SCRIPT, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posture", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--reference", required=True, help="where to write the Python reference run")
    args = ap.parse_args()
    walk = Policy(args.walk)
    start, ref = reference(Policy(args.posture), walk)
    env = BoltEnv(randomize=False, gait_generator="dog", turning=True, speed_control=True, sharp_turns=walk.sharp_turns)
    m = env.model
    bundle = dict(
        xml=MODEL.read_text(), geoms=describe(m),
        constants=dict(stand_pose=STAND_POSE.tolist(), dog_pose=STANCES["dog"].tolist(), action_scale=ACTION_SCALE.tolist(),
                       residual_scale=env.RESIDUAL_SCALE, frame_skip=env.frame_skip, dt=env.dt, paw_gait=PAW_GAIT,
                       turn_gain=env.TURN_GAIN, max_turn=env.MAX_TURN, max_speed=env.MAX_SPEED, speed_ref=env.SPEED_REF,
                       sharp_turns=env.sharp_turns, sharp_turn=env.SHARP_TURN,
                       hip_x=HIP_X.tolist(), hip_y=HIP_Y.tolist(),
                       commands=["stand", "sit", "lie", "walk"]),
        brains=dict(posture=dict(brain(args.posture), source=Path(args.posture).name),
                    walk=dict(brain(args.walk), source=Path(args.walk).name)),
        start=start,  # Bolt standing, as BoltEnv.reset leaves it (seed 0, no randomness)
    )
    Path(args.out).write_text(json.dumps(bundle, separators=(",", ":")))
    Path(args.reference).write_text(json.dumps(ref, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) and {args.reference} ({len(ref['steps'])} steps)")


if __name__ == "__main__":
    main()
