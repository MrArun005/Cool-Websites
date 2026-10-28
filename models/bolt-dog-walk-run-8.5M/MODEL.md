# bolt-dog-walk-run-8.5M

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 17 N·m cap, no power or speed limit (the motor model of 24-25 Oct). How determined: packaged 24 Oct 16:41 UTC, before commit df69a28 (27 Oct 10:08, first power/speed limits) and fd5199b (27 Oct 11:17, 33.5 N·m); its robot.xml snapshot has forcerange="-17 17". Today's code runs 33.5 N·m with limits: examine with BOLT_MOTOR_TORQUE=17 to compare with the numbers here (the power/speed limits cannot be switched off).

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `/tmp/claude-0/-home-user-Cool-Websites/029cd38f-3b1a-5876-9645-932075813c4a/scratchpad/pk11b/policy.zip` after **8,500,000 practice steps**
(about 47.2 hours of simulated dog time).

## Exam (randomized mass, friction and motor strength; 10 tries per command)

| Command | Passed |
|---|---|
| walk | 100% |



**Learned:** walk. **Not yet:** nothing.


## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (1×52 raw sensor values), output `action` (1×12, each −1…1). Normalization is built in. Matches the original to within 7.7e-07. |
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

## Speed and follow tests (randomised physics, hand-over from bolt-dog-posture-dog-2.5M)

| Asked | Got | Falls (of 6) | Knee/shin on floor |
|---|---|---|---|
| 0.2 m/s | 0.21 m/s | 0 | 0% |
| 0.4 m/s | 0.40 m/s | 0 | 0% |
| 0.7 m/s | 0.69 m/s | 0 | 0% |
| 1.0 m/s | 0.76 m/s | 1 | 19% |
| 1.2 m/s | 0.27 m/s | 2 | 68% |

Up to 0.7 m/s is solid; 1.0 m/s and above is not yet reliable on bumpy, randomised physics.

Following a person (`train/follow.py`): the person walks, turns, jogs at 1.1 m/s, turns and stops. On flat
ground Bolt never fell in 5 runs, stayed 1.4 m behind on average and never more than 1.93 m (the walk-turn
brain without speed control fell 6 m behind). It ran at up to 1.19 m/s during the jog.
