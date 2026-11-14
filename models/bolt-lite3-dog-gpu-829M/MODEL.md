# bolt-lite3-dog-gpu-829M (the Lite3 robot brain from 14 Nov)

DEEP Robotics Lite3 motion brain. No velocity input; it reads its last 3 frames of gravity, gyro, joints and previous
action. numpy params.pkl only (no torch on the robot). Simulation only.

How it was made: the previous robot brain hist-196M (free Kaggle CPU) trained on with the Go2 brain's recipe on the
free Kaggle GPU (mjx_imitate, MJX + Brax PPO): 632 million more steps in 6.5 h, privileged critic, 5% of control
steps dropped, 0-60 ms delay, motor speed x1.0-1.5 and torque limits x0.8-1.4 per try, over-speed penalty 3.0. Why:
the Go2 brain, made that way, had 1 fall in 384 tries of the tests where the Lite3's canter broke (PLAN.md 14 Nov).
Not picked on any exam: it is the run's last checkpoint.

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs, leg estimator), falls:

| Test | hist-196M (11-14 Nov) | this |
|---|---|---|
| canter from a trot (trot:6,canter:8,trot:4,canter:6,walk:4), 3-6 ms | 0/48 | 0/48 |
| the same, 10-19 ms | 3/48 | **0/48** |
| the same, steering left 0.8 rad/s, 10-19 ms | 10/48 | **0/48** |
| the same, 5% of control steps missed | 19/48 | **1/48** |
| the same: 10% missed; 5% missed with 10-19 ms; left 0.8 with 5% missed | not run | 4/48; 3/48; 2/48 |
| the 7 stress tests (trot, routine, sit, turns; delay, missed steps) | 0 falls | 0 falls |
| recovery (side, back, tipped, shove) | 48/48 | 48/48 |
| live in DEEP's simulator, steps over 1.2x rated per 41 s run (6 runs) | 3-12 | **1-2** |
| live with the canter on (49 s plan, 3 runs) | not run | 3/3, 0-3 steps over |
| steering right 0.8 rad/s at 10-19 ms (canter plan) | 26/48 | 15/48 |

Our exam (true physics): tracking 0.779 -> **0.825** (trot 0.67 -> 0.82), steering 0.587 -> 0.541, no falls; trot,
walk, pace 0 falls of 64 fresh tries each; moving sit / lie 0/48; steered canter on fresh seeds 0.540 and 0.524 with
2 falls of 24 each (hist-196M: 0.501 and 0.487 with 1 each).

What is not solved:
- Right turns. With commands 10-19 ms late a trot turning right falls 15 of 48 at 0.8 rad/s and 5 of 48 at 0.6; at
  0.4 it falls 0 of 48, and at 6-10 ms delay 0 of 48 even at 0.8. Left 0.8: 0 of 48. run_go2 caps the Lite3's
  right turns at 0.4 rad/s, and holds them at straight ahead in the canter (which turns left only).
- The canter under 10% missed steps or missed steps with long delay still falls 2-4 of 48, so run_go2 keeps the
  Lite3's canter as a trot by default; `--allow-canter` is now a reasonable choice at normal delay.

Motion data: CC BY-NC 4.0 (non-commercial).
