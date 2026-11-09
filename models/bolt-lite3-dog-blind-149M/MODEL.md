# bolt-lite3-dog-blind-149M (the Lite3 robot brain from 9 Nov)

DEEP Robotics Lite3 motion brain that does not read body velocity. Simulation only.

Why: a real robot has no velocity sensor. The previous brain (bolt-lite3-dog-149M) reads the velocity that
LegEstimator works out from the legs, and in a trot that estimate is off by 0.3-0.4 m/s on average. With delay or
missed control steps on top, 149M fell.

How: no new reinforcement learning. `train/distill_blind.py` copies 149M (DAgger, Ross et al. 2011): the student,
with inputs 6:9 zeroed, drives the training simulator; 149M, seeing the true velocity, labels every state the student
reaches; the student learns those labels. 60 rounds of 40,000 steps, 68 minutes on 4 CPU cores. params.pkl carries
`blind_vel`, so the robot program zeroes the same inputs.

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs, leg estimator), falls:

| Test | 149M | blind-149M |
|---|---|---|
| trot 20 s, 10-19 ms delay | 2/24 | **0/24** |
| trot 20 s, 3-6 ms delay, 5% of steps missed | 7/24 | **0/24** |
| 41 s routine, 10-19 ms | 2/24 | **0/24** |
| 41 s routine, 3-6 ms | 0/24 | 0/24 |
| routine ending in a canter, 3-6 ms | 13/24 | **1/24** |
| trot -> sit, 5-15 ms | 1/48 | **0/48** |
| pace, walk, turns, 3-6 ms | 0/12 | 0/12 |
| recovery (side, back, tipped, shove) | 20/20 (7 Nov) | 48/48 |
| live on DEEP's simulator, steps over 1.2x rated per 41 s run | 12-15 | **4-7** |

Our exam (true physics, no estimator):

| Measure | 149M | blind-149M |
|---|---|---|
| tracking / fall rate | 0.778 / 0 | 0.775 / 0 |
| moving sit / lie | 0/48 | 0/48 |
| steering in a trot | 0.636 | 0.633 |
| steering in a canter (falls of 8) | **0.433** (1) | 0.255 (4) |
| canter tracking | 0.453 | 0.445 |

What is worse: steering while cantering (it falls in 4 of 8 steered canters in our exam). run_go2 still runs the
Lite3's canter as a trot by default. Motion data: CC BY-NC 4.0, so this brain is for non-commercial use.
