# bolt-lite3-dog-blend-112M-w7 (the Lite3 robot brain from 5 Nov 03:00 UTC)

DEEP Robotics Lite3 motion brain, simulation only. A weight-space blend (train/weight_space.py, no training):
0.75 x bolt-lite3-dog-112M (worker 5: 87.3M + moving sit/lie 0.75 + 0-60 ms delay) + 0.25 x worker 7's 92M checkpoint
(87.3M + lr 3e-5 + steer weight 0.25 + 0-40 ms). Both were fine-tuned from 87.3M, so their weights average cleanly
(model soups, Wortsman et al. 2022). numpy params.pkl only: to fine-tune further, start from 112M or worker 7's run.

    python weight_space.py blend out.pkl ../models/bolt-lite3-dog-112M/params.pkl <w7_92000000.pkl> 0.25

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs):

| Test | 87.3M | 112M | w7 92M | 50/50 blend | **this (75/25)** |
|---|---|---|---|---|---|
| routine without canter, 3-6 ms | 0/12 | 0/24 | 0/12 | 0/12 | **0/24** |
| trot -> sit, 5-15 ms | 4/48 | 0/48 | 6/24 | 3/24 | **0/48** |
| routine ending in a canter, 3-6 ms | 2/12 | 5/12 | | | **2/12** |
| recovery (side, back, tipped, shove) | 20/20 | 20/20 | | | **20/20** |
| moving sit / lie (our exam) | 0/48 | 0/48 | | | **0/48** |
| our exam: tracking / steering | 0.780 / 0.449 | 0.756 / 0.456 | 0.788 / 0.506 | 0.789 / 0.477 | **0.775 / 0.441** |
| live on DEEP's simulator, no canter, 4 runs | no fall, 1 stop | no fall | | | **no fall, no stop** |
