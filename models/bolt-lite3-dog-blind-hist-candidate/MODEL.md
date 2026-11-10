# bolt-lite3-dog-blind-hist-candidate (10 Nov; a candidate, not the robot brain)

DEEP Robotics Lite3 motion brain that does not read body velocity but sees its last 3 frames of inputs (gravity,
gyro, joints, previous action). Simulation only. numpy params.pkl only (no SB3 policy).

Why: the blind robot brain (bolt-lite3-dog-blind-149M) falls in steered canters. A probe on its teacher 149M found
the cause: with only the forward-speed input zeroed, 149M falls in 14 of 24 steered canters (3 of 24 with it). One
frame of inputs cannot tell a brain how fast it runs; a few frames can. `train/distill_hist.py` copies 149M into a
network with a wider first layer (60 rounds of 40,000 steps, 54 minutes on 4 CPU cores).

| Test | 149M (reads velocity) | blind-149M (robot brain) | this |
|---|---|---|---|
| steered canter, fresh seeds, 24 tries: score / falls | 0.392 / 3 | 0.270 / 8 | **0.385 / 5** |
| our exam: tracking / steering | 0.778 / 0.534 | 0.775 / 0.444 | 0.777 / 0.540 |
| DEEP's model, leg estimate: trot 20 s at 10-19 ms | 2/24 | 0/24 | 0/24 |
| trot 20 s, 5% of steps missed | 7/24 | 0/24 | 0/24 |
| routine at 3-6 ms and at 10-19 ms | 0/24, 2/24 | 0/24, 0/24 | 0/24, 0/24 |
| trot -> sit, 5-15 ms | 1/48 | 0/48 | 0/48 |
| routine ending in a canter, 3-6 ms | 13/24 | 2/36 | 4/36 |
| recovery (side, back, tipped, shove) | 20/20 | 48/48 | 48/48 |
| live in DEEP's simulator, steps over 1.2x rated per run | 12-15 | **4-7** | 8-12 |

Not the default because the canter runs as a trot on the Lite3 anyway, and on everything the default routine does
the blind brain is as safe with lower joint speeds. Use: `run_go2.py --body lite3 --brain <this>/params.pkl`.
Motion data: CC BY-NC 4.0 (non-commercial).
