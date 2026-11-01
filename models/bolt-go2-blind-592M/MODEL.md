# bolt-go2-blind-592M (simulation): the Go2 brain for a real robot -- no velocity input (1 Nov)

From models/bolt-go2-gpu-489M, 103M more steps on a Kaggle T4 (3.5 h): the 489M recipe (all 9 clips, switches,
steering +-0.8, root-global 0.5, 0-40 ms delay, moving sit/lie starts 0.3) with **--blind-vel** (the body-velocity
inputs are always 0: a real Go2 has no velocity sensor, and leg odometry was 0.2-0.3 m/s off) and **--drop-p 0.05**
(5% of commands lost). MjxImitatePolicy zeroes the velocity inputs itself (params tag blind_vel).

Real-robot rehearsal (train/deploy/lockstep_vendor.py: Unitree's own go2.xml, their joint damping 0.1 and 5 ms step,
flat floor, only robot-measurable inputs -- IMU, joints, kinematic height; 41 s: trot 10, sit 5, lie 6, getup 6,
walk 8, canter 6; 10 tries):

| | 489M (uses velocity, from leg odometry) | run f (45M blind, 4090) | **this** |
|---|---|---|---|
| falls, plain | 4/10 | 3/10 | **0/10** |
| falls, 0-19 ms command delay | - | - | 1/10 |
| falls, 5% missed control steps | (true velocity: 3/5) | - | 3/10 |

Our own exam (imitate_env.exam, 4 tries per clip; exam_steer): mean 0.809 (489M 0.848), walk/trot/pace
0.811/0.758/0.831, canter 0.485 (489M 0.628), sit/lie/getup 0.939/0.922/0.922, steer trot 0.84 / canter 0.41,
20/40 ms 0.83/0.81 no falls, falls 9.6% (trot -> sit 4/4 and trot -> lie 1/4: the moving-posture falls came back).

Use this one on the robot (train/deploy/run_go2.py --brain models/bolt-go2-blind-592M/params.pkl); 489M stays the
best brain where the true velocity is known (simulation). Next piece: same recipe from here, to win back canter,
trot and the moving sit. Simulation only. Dog motion clips are CC BY-NC (non-commercial).
