# bolt-dog-run2-1M

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `runs/bolt2/checkpoints/policy_1000000_steps.zip` after **1,000,000 practice steps**
(about 5.6 hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; 10 tries per command)

| Command | Passed |
|---|---|
| stand | 100% |
| sit | 100% |
| lie | 100% |
| walk | 0% |
| all 7 commands in a row | 0% |
| fell over during the sequence | 0% |

**Learned:** stand, sit, lie. **Not yet:** walk.
Final training score per attempt: 131, survived 599 of 600 steps.

## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within 6.6e-07. |
| `policy.zip` + `vecnormalize.pkl` | To continue training (`train.py --resume`) |
| `robot.xml` | The robot it was trained on |
| `model.json` | Input layout, joint order, action scaling, commands and their targets |
| `exam.json`, `curve.csv` | Exam results and the learning curve |
| `rollouts.npz` | 3,000 recorded steps (obs, action, reward, command, posture, falls) |

## Using it

Every 1/50 s: build the 52 inputs in the order listed in `model.json`, run the model, then set
motor targets `stand_pose + action_scale × action` (both in `model.json`), clipped to the joint limits.

```python
import onnxruntime as ort
brain = ort.InferenceSession("policy.onnx")
action = brain.run(None, {"obs": obs[None].astype("float32")})[0][0]
```
