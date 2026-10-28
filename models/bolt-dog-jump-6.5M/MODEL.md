# bolt-dog-jump-6.5M

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 17 N·m cap, no power or speed limit (the motor model of 24-25 Oct). How determined: packaged 25 Oct 03:56 UTC, before commit df69a28 (27 Oct 10:08, first power/speed limits) and fd5199b (27 Oct 11:17, 33.5 N·m); its robot.xml snapshot has forcerange="-17 17". Today's code runs 33.5 N·m with limits: examine with BOLT_MOTOR_TORQUE=17 to compare with the numbers here (the power/speed limits cannot be switched off).

Bolt's jump brain: a real jump over a bar from standing, trained from scratch with PPO in
`train/jump_env.py` (6,500,000 steps, adaptive bar heights). It is separate from the walk and posture
brains: a game walks Bolt up to the bar, stops it, and hands the motors to this brain for 2 s.

## How to run it

1. Stop 0.55-0.65 m in front of the bar, facing it (within ~7 degrees).
2. Hold the joint targets at the dog stance for 0.5 s (training always started like this; after a
   posture-brain stand without the hold, 15 cm bars fell about 1 jump in 3).
3. For 100 control steps (50 Hz): build the 49 inputs below, run `policy.onnx`, set the joint
   targets to `clip(dog_stance + action_scale * action)`.
4. Hand back to the posture brain to stand.

Inputs: gravity_in_body_frame (3), angular_velocity_x0.25 (3), linear_velocity_body (3), body_height (1), joint_pos_minus_dog_stance (12), joint_vel_x0.1 (12), previous_action (12), bar_ahead_m (1), bar_height_m (1), time_fraction (1).

## Exam

20 jumps per height from 55 cm away, randomized physics; a pass is over the bar without
touching it, landed, and standing at the end.

| Bar | Clean jumps |
|---|---|
| 10 cm | 80% |
| 15 cm | 80% |
| 20 cm | 70% |
| 25 cm | 40% |
| 30 cm | 0% |

ONNX matches the PyTorch policy to 7.7e-07.
