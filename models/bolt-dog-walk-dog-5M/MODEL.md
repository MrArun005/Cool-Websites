# bolt-dog-walk-dog-5M

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 17 N·m cap, no power or speed limit (the motor model of 24-25 Oct). How determined: packaged 24 Oct 14:28 UTC, before commit df69a28 (27 Oct 10:08, first power/speed limits) and fd5199b (27 Oct 11:17, 33.5 N·m); its robot.xml snapshot has forcerange="-17 17". Today's code runs 33.5 N·m with limits: examine with BOLT_MOTOR_TORQUE=17 to compare with the numbers here (the power/speed limits cannot be switched off).

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `runs/bolt8b/checkpoints/policy_5000000_steps.zip` after **5,000,000 practice steps**
(about 27.8 hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; 10 tries per command)

| Command | Passed |
|---|---|
| walk | 100% |



**Learned:** walk. **Not yet:** nothing.
Final training score per attempt: 840, survived 600 of 600 steps.

## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within 8.9e-07. |
| `policy.zip` + `vecnormalize.pkl` | To continue training (`train.py --resume`) |
| `robot.xml` | The robot it was trained on |
| `model.json` | Input layout, joint order, action scaling, commands and their targets |
| `exam.json`, `curve.csv` | Exam results and the learning curve |
| `rollouts.npz` | 500 recorded steps (obs, action, reward, command, posture, falls) |

## Using it

Every 1/50 s: build the 52 inputs in the order listed in `model.json`, run the model, then set
motor targets `stand_pose + action_scale × action` (both in `model.json`), clipped to the joint limits.

**This brain walks on top of a built-in trot.** While walking, motor targets are
`reference_gait.trot(phase, hip_swing=0.15, knee_lift=0.3, base=<dog stance>)` + 0.35 × action_scale × action,
where phase = 2π × 2 Hz × time: the same clock the brain sees in its last two inputs (sin, cos).
Without the trot underneath, this brain will not walk.

```python
import onnxruntime as ort
brain = ort.InferenceSession("policy.onnx")
action = brain.run(None, {"obs": obs[None].astype("float32")})[0][0]
```

## Why 5M and not the final 5.5M

Run 8b ran to 5.5M steps and both checkpoints pass the walk exam 100%. But in 10 hand-overs
from the real standing brain (models/bolt-dog-posture-dog-2.5M), the 5M brain walked off
cleanly every time (44 cm tall, no knee on the floor), while the 5.5M brain sank into a
knee-dragging 23 cm crouch in 7 of 10. The exam hands over from a perfect motor-held stand and
missed it, so the 5M checkpoint is the one kept.
