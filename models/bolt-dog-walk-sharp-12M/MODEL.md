# bolt-dog-walk-sharp-12M

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 17 N·m cap, no power or speed limit (the motor model of 24-25 Oct). How determined: packaged 24 Oct 18:42 UTC, before commit df69a28 (27 Oct 10:08, first power/speed limits) and fd5199b (27 Oct 11:17, 33.5 N·m); its robot.xml snapshot has forcerange="-17 17". Today's code runs 33.5 N·m with limits: examine with BOLT_MOTOR_TORQUE=17 to compare with the numbers here (the power/speed limits cannot be switched off).

Bolt's movement brain, trained with reinforcement learning (PPO) in MuJoCo.
Packaged 2026-10-24 from `runs/bolt12/checkpoints/policy_12000000_steps.zip` after **12,000,000 practice steps**
(about 66.7 hours of simulated dog time).

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

## Speed and turn tests (randomised physics, hand-over from bolt-dog-posture-dog-2.5M)

This brain turns by stepping sideways (gait.json: sharp_turns). It is the walk brain in Bolt's Castle.

```
asked 0.2 m/s: got 0.23 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 24°
asked 0.4 m/s: got 0.40 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 24°
asked 0.7 m/s: got 0.68 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 24°
asked 1.0 m/s: got 0.97 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 24°
asked 1.2 m/s: got 1.15 m/s, fell 0/8, knee/shin on floor 0.0%, lower-leg tilt 24°
speed 0.2 turn 1.0 rad/s: turned 1.03 rad/s (59°/s), forward 0.24 m/s, fell 0/6, knee/shin on floor 0.0%
speed 0.2 turn 2.0 rad/s: turned 1.94 rad/s (111°/s), forward 0.27 m/s, fell 0/6, knee/shin on floor 0.0%
speed 0.4 turn 1.0 rad/s: turned 1.02 rad/s (58°/s), forward 0.41 m/s, fell 0/6, knee/shin on floor 0.0%
speed 0.4 turn 1.5 rad/s: turned 1.50 rad/s (86°/s), forward 0.41 m/s, fell 0/6, knee/shin on floor 0.0%
speed 0.4 turn 2.0 rad/s: turned 1.89 rad/s (108°/s), forward 0.40 m/s, fell 0/6, knee/shin on floor 0.0%
speed 0.7 turn 1.5 rad/s: turned 1.46 rad/s (84°/s), forward 0.69 m/s, fell 0/6, knee/shin on floor 0.0%
speed 1.0 turn 1.0 rad/s: turned 0.97 rad/s (56°/s), forward 0.96 m/s, fell 0/6, knee/shin on floor 0.0%
speed 1.2 turn 1.0 rad/s: turned 0.96 rad/s (55°/s), forward 1.14 m/s, fell 0/6, knee/shin on floor 0.0%
```

No falls in any speed (8 tries each) or turn test (6 tries each); 11.5M fell once at 2 rad/s and 0.4 m/s.
