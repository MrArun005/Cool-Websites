# bolt-dog-walk-dog-5.5M

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `runs/bolt8b/policy.zip` after **5,502,944 practice steps**
(about 30.6 hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; 10 tries per command)

| Command | Passed |
|---|---|
| walk | 100% |



**Learned:** walk. **Not yet:** nothing.
Final training score per attempt: 852, survived 600 of 600 steps.

## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within 6.6e-07. |
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
