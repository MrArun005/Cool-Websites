# bolt-lite3-dog-hist-196M (the Lite3 robot brain from 11 Nov)

DEEP Robotics Lite3 motion brain. No velocity input (a real robot has no velocity sensor); it reads its last 3 frames
of gravity, gyro, joints and previous action instead, which is how it knows its speed. Simulation only.

How it was made, all on free CPU: 149M (read velocity) -> copied with the velocity zeroed and 3 frames of history
(`train/distill_hist.py`) -> PPO on Kaggle CPU with `train_imitate.py --history 3 --blind-vel`: lite3-dog-26 (lr
3e-5, 23M steps), then lite3-dog-28 (over-speed penalty 3.0, lr 1e-5, 25M steps; the last checkpoint, 196,496,976
steps). Judged on seeds the training never used to pick it.

| Test | 149M (9 Nov, reads velocity) | blind-149M (9-11 Nov) | this |
|---|---|---|---|
| steered canter, fresh seeds, 24 tries: score / falls | 0.392 / 3 | 0.270 / 8 | **0.501 / 1** |
| our exam: tracking / steering | 0.778 / 0.534 | 0.775 / 0.444 | 0.779 / **0.587** |
| plain trot, walk, pace in our exam, 64 fresh tries each: falls | not run | 0 | 0 |
| moving sit / lie in our exam | 0/48 | 0/48 | 0/48 |
| DEEP's model, leg estimate: trot 20 s at 10-19 ms | 2/24 | 0/24 | 0/24 |
| trot 20 s, 5% of steps missed | 7/24 | 0/24 | 0/24 |
| 41 s routine at 3-6 ms and at 10-19 ms | 0/24, 2/24 | 0/24, 0/24 | 0/24, 0/24 |
| trot -> sit at 5-15 ms; walk/trot -> sit/lie | 1/48; not run | 0/48; not run | 0/48; 0/72 |
| routine ending in a canter, 3-6 ms | 13/24 | 2/36 | **0/36** |
| recovery (side, back, tipped, shove) | 20/20 | 48/48 | 48/48 |
| live in DEEP's simulator, steps over 1.2x rated per 41 s run | 12-15 | 1-7 (6 runs) | 3-12 (6 runs: 6, 4, 12, 5, 3, 5), no stop |

What is not solved: a steered canter still falls sometimes (1 of 24 on fresh seeds, 2 of 8 on the exam's own seeds),
so run_go2 keeps running the Lite3's canter as a trot by default. Live joint over-speed is a little higher than the
blind brain's. Motion data: CC BY-NC 4.0 (non-commercial).
