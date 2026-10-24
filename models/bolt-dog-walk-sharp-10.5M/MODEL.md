# bolt-dog-walk-sharp-10.5M

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `runs/bolt12/checkpoints/policy_10500000_steps.zip` after **10,500,000 practice steps**
(about 58.3 hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; 10 tries per command)

| Command | Passed |
|---|---|
| walk | 100% |



**Learned:** walk. **Not yet:** nothing.


## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within 8.3e-07. |
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

This brain turns by stepping sideways (`reference_gait.paw_trot(..., spin=...)`). Build its environment with
`BoltEnv(..., sharp_turns=True)`; `evaluate.Policy` reads `gait.json` and does this for you.

Turn test (randomised physics, hand-over from bolt-dog-posture-dog-2.5M, 6 tries each):

| Speed | Asked turn | Turned | Falls |
|---|---|---|---|
| 0.2 m/s | 1.0 rad/s | 1.00 rad/s | 0 |
| 0.2 m/s | 2.0 rad/s | 1.64 rad/s | 0 |
| 0.4 m/s | 1.5 rad/s | 1.36 rad/s | 0 |
| 0.4 m/s | 2.0 rad/s | 1.48 rad/s | 3 |
| 0.7 m/s | 1.5 rad/s | 1.12 rad/s | 2 |
| 1.0 m/s | 1.0 rad/s | 0.93 rad/s | 1 |

Speeds asked 0.2 to 1.2 m/s give 0.22, 0.41, 0.69, 0.99 and 1.18 m/s, the same as bolt-dog-walk-run-10M.
On the live page, follow turns are capped at 1.5 rad/s: 0 falls in 40 two-minute sessions (2 rad/s: 3 falls).
