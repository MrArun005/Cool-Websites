# bolt-dog-walk-run-10M

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `/tmp/claude-0/-home-user-Cool-Websites/029cd38f-3b1a-5876-9645-932075813c4a/scratchpad/ck10000000/policy.zip` after **10,000,000 practice steps**
(about 55.6 hours of simulated dog time).

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

## Speed test (randomised physics, hand-over from bolt-dog-posture-dog-2.5M, 8 tries each)

```
asked 0.2 m/s: got 0.22 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 23°
asked 0.4 m/s: got 0.41 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 24°
asked 0.7 m/s: got 0.70 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 23°
asked 1.0 m/s: got 0.99 m/s, fell 1/8, knee/shin on floor 0.0%, lower-leg tilt 23°
asked 1.2 m/s: got 1.18 m/s, fell 1/8, knee/shin on floor 0.7%, lower-leg tilt 23°
```

The best of run 11b's checkpoints at speed: 8.5M reached 0.82 and 0.37 m/s when asked for 1.0 and 1.2;
this one reaches 0.99 and 1.18 with no knees on the floor.
