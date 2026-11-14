# bolt-lite3-dog-gpu-1461M (the Lite3 robot brain from 14 Nov, evening)

DEEP Robotics Lite3 motion brain. No velocity input; it reads its last 3 frames of gravity, gyro, joints and previous
action. numpy params.pkl only (no torch on the robot). Simulation only.

How it was made: gpu-829M trained on for 632 million more steps (6.5 h on the free Kaggle GPU) with the same recipe
(the Go2 brain's: privileged critic, 5% of control steps dropped, 0-60 ms delay, motor speed x1.0-1.5 and torque
x0.8-1.4 per try, over-speed penalty 3.0) and the steering reward doubled to 0.3, aimed at the right-turn falls found
in gpu-829M. The run's last checkpoint, not picked on any exam.

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs, leg estimator), falls of 48:

| Test | hist-196M (11 Nov, CPU) | gpu-829M (14 Nov) | this |
|---|---|---|---|
| trot turning right 0.8 rad/s, 10-19 ms | not run | 15 | **0** |
| trot turning right 0.6 / left 0.8, 10-19 ms | not run | 5 / 0 | 0 / 0 |
| canter plan steering right 0.8, 10-19 ms | 26 | 15 | **0** |
| canter from a trot, 3-6 ms / 10-19 ms | 0 / 3 | 0 / 0 | 0 / 0 |
| canter, steering left 0.8, 10-19 ms | 10 | 0 | 0 |
| canter, 5% of control steps missed | 19 | 1 | **0** |
| canter, 10% missed / 5% missed with 10-19 ms | not run | 4 / 3 | 2 / 1 |

Also: the 7 stress tests (trot, routine, sit, turns; delay, missed steps) 0 falls; recovery 48/48; live in DEEP's
simulator the 41 s routine 6/6 with 0-1 steps over 1.2x rated speed a run (gpu-829M 1-2, hist-196M 3-12); live with
canters and steering (49 s plan, left 0.6 and right 0.8) 6 of 7 finished with 0-1 steps over, and one was ended by
the robot program's stale-data stop (sensor data 81 ms old: a timing hiccup of the simulator on a shared machine,
not a fall).

Our exam (true physics): tracking 0.828 (trot 0.85), steering 0.540 (trot 0.768), no falls; trot, walk, pace 0 falls
of 64 fresh tries each; moving sit / lie 0/48; steered canter on fresh seeds 0.584 and 0.518 with 1 fall of 24 each.

run_go2 now lets the Lite3 canter by default (`--no-canter` turns it into a trot) and has no cap on right turns in
the trot. What is not solved: the canter turns left only (asked for -0.8 rad/s it reaches -0.16), so right-turn
commands are held at straight ahead in the canter; under 10% missed steps the canter still falls 2 of 48.

Motion data: CC BY-NC 4.0 (non-commercial).
