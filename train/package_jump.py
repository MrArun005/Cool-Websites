"""Package a jump-brain checkpoint (train_jump.py) into a self-contained model folder.

    python package_jump.py runs/jump15c/checkpoints/policy_6500000_steps.zip --name bolt-dog-jump-6.5M \
                           --log runs/jump15c/train.log

Writes ../models/<name>/: policy.onnx (normalization built in), policy.zip + vecnormalize.pkl (to continue
training), robot.xml, model.json (inputs, outputs, how to run a jump), exam.json (clean jumps per bar
height), curve.csv (the exam and practice bar height at every checkpoint) and MODEL.md.
"""
import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

import numpy as np

from bolt_env import ACTION_SCALE, MODEL
from jump_env import DOG, JumpEnv, JumpPolicy, exam
from package import JOINTS, export_onnx

OBS_LAYOUT = [("gravity_in_body_frame", 3), ("angular_velocity_x0.25", 3), ("linear_velocity_body", 3),
              ("body_height", 1), ("joint_pos_minus_dog_stance", 12), ("joint_vel_x0.1", 12), ("previous_action", 12),
              ("bar_ahead_m", 1), ("bar_height_m", 1), ("time_fraction", 1)]
HEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.30)


def check_onnx(policy, path, samples=100):
    import onnxruntime as ort
    session, env = ort.InferenceSession(str(path)), JumpEnv(randomize=True, real=policy.real)
    obs, _ = env.reset(seed=5)
    worst = 0.0
    for _ in range(samples):
        ref = policy.act(obs)
        worst = max(worst, float(np.abs(session.run(None, {"obs": obs[None].astype(np.float32)})[0][0] - ref).max()))
        obs, _, fell, trunc, _ = env.step(ref)
        if fell or trunc:
            obs, _ = env.reset()
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", help="train_jump.py output, for the curve")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    args = ap.parse_args()
    ck = Path(args.checkpoint)
    stats = ck.name.replace("policy_", "policy_vecnormalize_", 1).replace(".zip", ".pkl")
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ck, out / "policy.zip")
    shutil.copy(ck.with_name(stats), out / "vecnormalize.pkl")
    shutil.copy(MODEL, out / "robot.xml")
    policy = JumpPolicy(out)
    export_onnx(policy, out / "policy.onnx")
    err = check_onnx(policy, out / "policy.onnx")
    result = exam(policy, episodes=args.episodes, heights=HEIGHTS)
    (out / "exam.json").write_text(json.dumps(result, indent=2))
    rows = ["steps,exam_30,exam_40,practice_bar_cm"]
    if args.log:
        for m in re.finditer(r"checkpoint (\d+) ready: \{'jump_30': ([\d.]+), 'jump_40': ([\d.]+)\} \(practice bars now (\d+) cm\)",
                             Path(args.log).read_text()):
            rows.append(",".join(m.groups()))
    (out / "curve.csv").write_text("\n".join(rows) + "\n")
    steps = int(re.search(r"(\d+)_steps", ck.name).group(1))
    layout = OBS_LAYOUT + ([("sin_heading_to_bar_line", 1), ("metres_left_of_bar_line", 1)] if policy.real else [])
    facts = dict(name=args.name, source=str(ck), steps=steps, packaged=dt.date.today().isoformat(), control_hz=50,
                 obs_size=len(layout) and sum(s for _, s in layout), action_size=12, jump_steps=100,
                 hurdle="real (hurdle.py: solid stands, loose bar in cups)" if policy.real else "solid board",
                 obs_layout=[dict(name=n, size=s) for n, s in layout],
                 joints=JOINTS, dog_stance=DOG.tolist(), action_scale=ACTION_SCALE.tolist(),
                 control="joint targets = clip(dog_stance + action_scale * action, joint limits)",
                 start="stand in the dog stance on the motors for 0.5 s, 0.55-0.65 m in front of the bar, facing it",
                 onnx_max_difference=err, exam=result)
    (out / "model.json").write_text(json.dumps(facts, indent=2))
    table = "\n".join(f"| {k.split('_')[1]} cm | {100 * v:.0f}% |" for k, v in result.items())
    (out / "MODEL.md").write_text(f"""# {args.name}

Bolt's jump brain: a real jump over a bar from standing, trained from scratch with PPO in
`train/jump_env.py` ({steps:,} steps, adaptive bar heights). It is separate from the walk and posture
brains: a game walks Bolt up to the bar, stops it, and hands the motors to this brain for 2 s.

## How to run it

1. Stop 0.55-0.65 m in front of the bar, facing it (within ~7 degrees).
2. Hold the joint targets at the dog stance for 0.5 s (training always started like this; after a
   posture-brain stand without the hold, 15 cm bars fell about 1 jump in 3).
3. For 100 control steps (50 Hz): build the {facts["obs_size"]} inputs below, run `policy.onnx`, set the joint
   targets to `clip(dog_stance + action_scale * action)`.
4. Hand back to the posture brain to stand.

Inputs: {", ".join(f"{n} ({s})" for n, s in layout)}.

## Exam

{args.episodes} jumps per height from 55 cm away, randomized physics; a pass is over the bar without
touching it, landed, and standing at the end.

| Bar | Clean jumps |
|---|---|
{table}

ONNX matches the PyTorch policy to {err:.1e}.
""")
    print(f"wrote {out}: exam {result}, onnx error {err:.1e}")


if __name__ == "__main__":
    main()
