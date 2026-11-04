# bolt-lite3-dog-103M-candidate (runner-up, not the default)

DEEP Robotics Lite3 motion brain, simulation only. Kaggle CPU job lite3-dog-6: fine-tuned from bolt-lite3-dog-87.3M
with the canter drawn 2x and 0-80 ms of command delay; checkpoint 103,276,224 steps (numpy params.pkl from
`mjx_imitate.py warm`).

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs, 3-6 / 5-15 ms delay), against 87.3M:

| Test | 87.3M | 103M |
|---|---|---|
| no-canter routine 3-6 ms | 0/12 | 0/24 |
| trot -> sit 5-15 ms | 4/48 | 2/48 |
| routine ending in a canter, 3-6 ms | 2/12 (canter) | 0/12 |
| recovery (side, back, tipped, shove) | 20/20 | 20/20 |
| moving sit / lie (our exam) | 0/48 | 0/48 |
| our exam: tracking / steering | 0.780 / 0.449 | 0.750 / 0.382 |
| live on DEEP's simulator, no canter, 4 runs | no fall; 1 speed stop | no fall |

Not adopted: its gains are mostly in the canter, which run_go2 turns into a trot on the Lite3, while its tracking and
steering are lower. Kept for the hardware day, where the measured delay decides.
