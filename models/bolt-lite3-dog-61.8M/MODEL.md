# bolt-lite3-dog-61.8M (simulation): the DEEP Robotics Lite3 robot brain (3 Nov, replaces 20.9M as the candidate)

From bolt-lite3-dog-38.8M, +23M steps on a free Kaggle CPU session (bolt-lite3-dog-3, ~10.5 h): half the tries start in a
walk or trot and are told at once to sit or lie down (--moving-rate 0.5), steering +-0.8 rad/s, 0-40 ms delay, brake.
params.pkl = numpy export for the GPU trainer and run_go2.py --body lite3 (it reads body velocity: LegEstimator).

| | 20.9M | 38.8M | **61.8M** |
|---|---|---|---|
| our exam: mean tracking | 0.771 | 0.813 | **0.812** |
| canter | 0.36 | 0.49 | **0.52** |
| sit / lie from a moving walk or trot (16 tries) | weak | 11 falls | **0 falls, pose reached 16/16** |
| exam fall rate (incl. delays) | 0.27 | 0.19 | **0.00** |
| DEEP's own MJCF, 41 s plan, estimated v/h (lockstep_vendor) | 0/3 falls | 3/3 (sit -> lie) | **0/3** |
| same, knee/joint steps over 1.2x rated | 993 of 6150 | - | **342 of 6150** |
| same with 5-15 ms command delay | 3/3 falls | - | 3/3 falls |

Open: command delay on DEEP's undamped, unlimited-speed motors; the GPU run (joint damping 0.1-2.0, motor speed x1-4,
0-40 ms delay) is queued for it. Simulation only. Dog motion clips are CC BY-NC (non-commercial).
