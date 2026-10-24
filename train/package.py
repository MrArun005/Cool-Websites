"""Package a training run into a self-contained model folder.

    python package.py runs/bolt --name bolt-v1 --log train.log

Writes ../models/<name>/:
    policy.onnx        the trained brain, ready for a robot computer (observation
                       normalization is built in, so it takes raw sensor values)
    policy.zip         PPO checkpoint, to continue training
    vecnormalize.pkl   observation statistics that go with policy.zip
    robot.xml          the exact MuJoCo robot model it was trained on
    model.json         machine-readable facts: inputs, outputs, joints, commands
    exam.json          success rate per command (randomized physics)
    curve.csv          learning curve: steps, score per attempt, survival, seconds
    rollouts.npz       recorded episodes: observations, actions, rewards, commands,
                       posture, falls. Training data for imitation or analysis.
    MODEL.md           model card for people

Use --checkpoint to package a specific checkpoint instead of the final policy.
"""
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from bolt_env import ACTION_SCALE, COMMANDS, MODEL, STANCES, STAND_POSE, TARGETS, BoltEnv
from evaluate import Policy, evaluate
from watch import read_curve

JOINTS = ["FL_abd", "FL_hip", "FL_knee", "FR_abd", "FR_hip", "FR_knee",
          "RL_abd", "RL_hip", "RL_knee", "RR_abd", "RR_hip", "RR_knee"]
OBS_LAYOUT = [
    ("gravity_in_body_frame", 3), ("angular_velocity_x0.25", 3), ("linear_velocity_body", 3),
    ("body_height", 1), ("joint_pos_minus_stand", 12), ("joint_vel_x0.1", 12),
    ("previous_action", 12), ("command_one_hot", len(COMMANDS)), ("gait_clock_sin_cos", 2),
]


class OnnxPolicy(torch.nn.Module):
    """Normalize raw observations, run the actor, clip to [-1, 1]: exactly what Policy.act does."""

    def __init__(self, policy):
        super().__init__()
        rms = policy.norm.obs_rms
        self.register_buffer("mean", torch.tensor(rms.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.sqrt(rms.var + policy.norm.epsilon), dtype=torch.float32))
        self.clip = float(policy.norm.clip_obs)
        p = policy.model.policy
        self.actor, self.action_net = p.mlp_extractor.policy_net, p.action_net

    def forward(self, obs):
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        return torch.clamp(self.action_net(self.actor(x)), -1.0, 1.0)


def export_onnx(policy, path):
    net = OnnxPolicy(policy).eval()
    dummy = torch.zeros(1, policy.model.observation_space.shape[0])
    torch.onnx.export(net, (dummy,), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, dynamo=False)


def check_onnx(policy, path, samples=200, gait_generator=False):
    """Run the ONNX file next to the original policy on real observations; they must agree."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path))
    env = BoltEnv(randomize=True, gait_generator=gait_generator, turning=policy.turning)
    obs, _ = env.reset(seed=5)
    worst = 0.0
    for _ in range(samples):
        ref = policy.act(obs)
        out = session.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        worst = max(worst, float(np.abs(out - ref).max()))
        obs, *_ = env.step(ref)
    return worst


def record_rollouts(policy, episodes_per_command, steps=250, seed=2000, commands=COMMANDS, gait_generator=False):
    env = BoltEnv(randomize=True, max_steps=10_000, gait_generator=gait_generator, turning=policy.turning)
    keys = ("obs", "action", "reward", "command", "height", "pitch", "roll", "vx", "fell", "episode")
    data = {k: [] for k in keys}
    episode = 0
    for command in commands:
        c = COMMANDS.index(command)
        for k in range(episodes_per_command):
            env.forced_command = command
            obs, _ = env.reset(seed=seed + episode)
            for _ in range(steps):
                action = policy.act(obs)
                nxt, reward, fell, _, info = env.step(action)
                for key, value in (("obs", obs), ("action", action), ("reward", reward), ("command", c),
                                   ("height", info["height"]), ("pitch", info["pitch"]), ("roll", info["roll"]),
                                   ("vx", info["vx"]), ("fell", fell), ("episode", episode)):
                    data[key].append(value)
                obs = nxt
                if fell:
                    break
            episode += 1
    return {k: np.asarray(v, dtype=np.float32 if k in ("obs", "action", "reward", "height", "pitch", "roll", "vx") else np.int16)
            for k, v in data.items()}


def model_card(name, facts, exam, curve, onnx_err):
    last = curve[-1] if curve else None
    gait_note = ""
    if facts.get("gait_generator", {}).get("used"):
        g = facts["gait_generator"]
        gait_note = (f"\n**This brain walks on top of a built-in trot.** While walking, motor targets are\n"
                     f"`reference_gait.trot(phase, hip_swing=0.15, knee_lift=0.3, base=<{g.get('stance') or 'crouch'} stance>)` + {g['residual_scale']} × action_scale × action,\n"
                     f"where phase = 2π × 2 Hz × time: the same clock the brain sees in its last two inputs (sin, cos).\n"
                     f"Without the trot underneath, this brain will not walk.\n")
    skills = facts.get("skills") or COMMANDS
    rows = "\n".join(f"| {c} | {exam[c]:.0%} |" for c in skills)
    learned = [c for c in skills if exam[c] >= 0.8]
    missing = [c for c in skills if exam[c] < 0.8]
    return f"""# {name}

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged {facts['packaged']} from `{facts['source']}` after **{facts['steps']:,} practice steps**
(about {facts['steps'] / 50 / 3600:.1f} hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; {facts['exam_episodes']} tries per command)

| Command | Passed |
|---|---|
{rows}
{f"| all 7 commands in a row | {exam['sequence']:.0%} |" if 'sequence' in exam else ""}
{f"| fell over during the sequence | {exam['fell']:.0%} |" if 'fell' in exam else ""}

**Learned:** {', '.join(learned) or 'nothing reliably yet'}. **Not yet:** {', '.join(missing) or 'nothing'}.
{f"Final training score per attempt: {last[1]:.0f}, survived {last[2]} of 600 steps." if last else ""}

## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within {onnx_err:.1e}. |
| `policy.zip` + `vecnormalize.pkl` | To continue training (`train.py --resume`) |
| `robot.xml` | The robot it was trained on |
| `model.json` | Input layout, joint order, action scaling, commands and their targets |
| `exam.json`, `curve.csv` | Exam results and the learning curve |
| `rollouts.npz` | {facts['rollout_steps']:,} recorded steps (obs, action, reward, command, posture, falls) |

## Using it

Every 1/50 s: build the 52 inputs in the order listed in `model.json`, run the model, then set
motor targets `stand_pose + action_scale × action` (both in `model.json`), clipped to the joint limits.
{gait_note}
```python
import onnxruntime as ort
brain = ort.InferenceSession("policy.onnx")
action = brain.run(None, {{"obs": obs[None].astype("float32")}})[0][0]
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="training run folder, e.g. runs/bolt")
    ap.add_argument("--name", required=True, help="model name, e.g. bolt-v1")
    ap.add_argument("--log", help="train.py output, for the learning curve")
    ap.add_argument("--checkpoint", help="checkpoint .zip to package instead of the final policy")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--episodes", type=int, default=20, help="exam tries per command")
    ap.add_argument("--rollouts", type=int, default=5, help="recorded episodes per command")
    ap.add_argument("--skills", nargs="+", help="the commands this brain was trained for")
    ap.add_argument("--gait-generator", nargs="?", const="crouch", default=None, choices=["crouch", "tall", "dog"], help="the walk runs on the built-in trot, around this stance (no value = crouch, as in runs 5-6)")
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        stats = ckpt.with_name(ckpt.name.replace("policy_", "policy_vecnormalize_", 1)).with_suffix(".pkl")
        steps = int("".join(ch for ch in ckpt.stem if ch.isdigit()))
    else:
        ckpt, stats, steps = run / "policy.zip", run / "vecnormalize.pkl", None
    shutil.copy(ckpt, out / "policy.zip")
    shutil.copy(stats, out / "vecnormalize.pkl")
    shutil.copy(MODEL, out / "robot.xml")

    policy = Policy(out)
    steps = steps or int(policy.model.num_timesteps)

    export_onnx(policy, out / "policy.onnx")
    onnx_err = check_onnx(policy, out / "policy.onnx", gait_generator=args.gait_generator)
    print(f"onnx export ok, max difference {onnx_err:.2e}")

    exam = {k: round(float(v), 3) for k, v in evaluate(policy, args.episodes, seed=4242, skills=args.skills,
                                                       gait_generator=args.gait_generator).items()}
    (out / "exam.json").write_text(json.dumps(exam, indent=2))
    print("exam:", exam)

    curve = [p for p in read_curve(Path(args.log)) if p[0] <= steps] if args.log else []
    with open(out / "curve.csv", "w") as f:
        f.write("steps,score_per_attempt,steps_survived,seconds\n")
        f.writelines(",".join(map(str, p)) + "\n" for p in curve)

    rollouts = record_rollouts(policy, args.rollouts, commands=args.skills or COMMANDS, gait_generator=args.gait_generator)
    np.savez_compressed(out / "rollouts.npz", **rollouts)

    facts = dict(
        name=args.name, source=str(ckpt), steps=steps, exam_episodes=args.episodes,
        packaged=dt.date.today().isoformat(), rollout_steps=int(len(rollouts["reward"])),
        control_hz=50, obs_size=52, action_size=12,
        obs_layout=[{"name": n, "size": s} for n, s in OBS_LAYOUT],
        joints=JOINTS, stand_pose=STAND_POSE.round(4).tolist(), action_scale=ACTION_SCALE.tolist(),
        commands=COMMANDS, command_targets=TARGETS, skills=args.skills or COMMANDS,
        gait_generator=dict(used=bool(args.gait_generator), stance=args.gait_generator,
                            stance_pose=STANCES[args.gait_generator].round(4).tolist() if args.gait_generator else None,
                            residual_scale=BoltEnv.RESIDUAL_SCALE,
                            how="while walking: motor targets = reference_gait.trot(phase, 0.15, 0.3, base=stance_pose) "
                                "+ residual_scale x action_scale x action"
                                + ("; eased in from stand_pose over the first 0.5 s of walking" if args.gait_generator in ("tall", "dog") else "")
                                + ("; for 'dog' the trot is reference_gait.paw_trot(phase, **PAW_GAIT) instead" if args.gait_generator == "dog" else ""))
        if args.gait_generator else dict(used=False),
        onnx_max_difference=onnx_err,
    )
    (out / "model.json").write_text(json.dumps(facts, indent=2))
    (out / "MODEL.md").write_text(model_card(args.name, facts, exam, curve, onnx_err))
    print(f"packaged -> {out}")


if __name__ == "__main__":
    main()
