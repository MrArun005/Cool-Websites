# bolt-dog-walk-sharp-11.5M

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 17 N·m cap, no power or speed limit (the motor model of 24-25 Oct). How determined: packaged 24 Oct 18:20 UTC, before commit df69a28 (27 Oct 10:08, first power/speed limits) and fd5199b (27 Oct 11:17, 33.5 N·m); its robot.xml snapshot has forcerange="-17 17". Today's code runs 33.5 N·m with limits: examine with BOLT_MOTOR_TORQUE=17 to compare with the numbers here (the power/speed limits cannot be switched off).

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `runs/bolt12/checkpoints/policy_11500000_steps.zip` after **11,500,000 practice steps**
(about 63.9 hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; 10 tries per command)

| Command | Passed |
|---|---|
| walk | 100% |



**Learned:** walk. **Not yet:** nothing.


## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within 6.6e-07. |
| `policy.zip` + `vecnormalize.pkl` | To continue training (`train.py --resume`) |
| `robot.xml` | The robot it was trained on |
| `model.json` | Input layout, joint order, action scaling, commands and their targets |
| `exam.json`, `curve.csv` | Exam results and the learning curve |
| `rollouts.npz` | 1,250 recorded steps (obs, action, reward, command, posture, falls) |

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

## Sharp turns (gait.json: sharp_turns)

This brain turns by stepping sideways (`reference_gait.paw_trot(..., spin=...)`); build its environment with
`BoltEnv(..., sharp_turns=True)` (`evaluate.Policy` reads `gait.json` and does this).

Turn test (randomised physics, hand-over from bolt-dog-posture-dog-2.5M, 6 tries each):

| Speed | Asked turn | Turned | Falls |
|---|---|---|---|
| 0.2 m/s | 2.0 rad/s | 1.94 rad/s | 0 |
| 0.4 m/s | 1.5 rad/s | 1.54 rad/s | 0 |
| 0.4 m/s | 2.0 rad/s | 1.85 rad/s | 1 |
| 0.7 m/s | 1.5 rad/s | 1.50 rad/s | 0 |
| 1.0 m/s | 1.0 rad/s | 1.00 rad/s | 0 |
| 1.2 m/s | 1.0 rad/s | 0.98 rad/s | 0 |

Speeds 0.2 to 1.2 m/s asked give 0.22, 0.41, 0.68, 0.96 and 1.15 m/s, with 0 falls in 8 at each.
Live page follow test: 0 falls in 40 two-minute sessions, turns capped at 1.5 rad/s and speed at 1.2 m/s.
