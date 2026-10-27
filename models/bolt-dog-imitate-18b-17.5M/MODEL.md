# bolt-dog-imitate-18b-17.5M

Bolt's dog-imitation brain: moves like a real dog by tracking retargeted motion-capture clips
(`train/imitate_env.py`, run imitate18b at 17,500,000 steps). It knows walk, trot, pace, canter, turn_left, turn_right, sit, lie.

## How to run it (gaitlab/imitate.js does this)

Every 1/50 s: advance the clip clock, build the 74 inputs below, run the brain (`brain.json` in
the pages, or `policy.onnx`), and set the joint targets to `clip(q_ref(next step) + action_scale * action)`: the
clip's next pose is the plan, the brain adds corrections. Switching clip: a gait joins at the frame whose pose is
closest to Bolt's; sit and lie play from their start and then hold their last pose (the brain practised 1 s).

Inputs: gravity_in_body_frame (3), angular_velocity_x0.25 (3), linear_velocity_body (3), body_height (1), joint_pos_minus_dog_stance (12), joint_vel_x0.1 (12), previous_action (12), clip_phase_sin_cos (2), clip_one_hot (8), ref_joints_next_minus_dog_stance (12), ref_height_next (1), ref_velocity_body_next (3), ref_yaw_rate_next (1), seconds_since_clip_end (1).

## Exam

Training's exam (`imitate_env.exam`, 4 tries per clip, randomized physics, loops 4 s, sit/lie through their
hold): mean tracking reward per step (0..1), falls, and the share of the dog's pace kept. Fixed physics 6 s: the
browser test's protocol in Python (gaitlab/imitate_test.mjs runs the same in the browser physics).

| Clip | Tracking | Falls | Pace | Tracking, fixed 6 s | Falls, fixed 6 s |
|---|---|---|---|---|---|
| walk | 0.908 | 0% | 0.86 | 0.917 | 0% |
| trot | 0.899 | 0% | 0.89 | 0.909 | 0% |
| pace | 0.921 | 0% | 0.86 | - | - |
| canter | 0.426 | 25% | 0.48 | 0.425 | 25% |
| turn_left | 0.890 | 0% | 0.17 | - | - |
| turn_right | 0.898 | 0% | 0.27 | - | - |
| sit | 0.939 | 0% | - | 0.951 | 0% |
| lie | 0.935 | 0% | - | 0.942 | 0% |

ONNX matches the PyTorch policy to 5.4e-07.
