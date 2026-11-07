# bolt-lite3-dog-149M (the Lite3 robot brain from 7 Nov)

DEEP Robotics Lite3 motion brain, simulation only. Kaggle lite3-dog-17: the steering candidate 130M (112M + canter x3,
Kaggle lite3-dog-10) fine-tuned with a x1.5 motor drive and the over-speed penalty (train_imitate --overspeed-weight
1.0), lr 1e-5, 0-60 ms delay; checkpoint 148,981,608 steps (policy.zip + vecnormalize.pkl from SB3; params.pkl, the
numpy brain the robot runs, from `mjx_imitate.py warm --body lite3`).

Why the over-speed penalty: brains that steered well drove the joints past 1.2x rated speed twice as often live on
DEEP's simulator, but not in training, because our motor model fades the drive above rated speed and DEEP's has no
speed limit. Trained with a faster drive and a penalty on speed above rated, the brain learns to hold its joints back
itself (PLAN.md 6-7 Nov).

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs), against the previous robot brain:

| Test | blend-112M-w7 | 149M |
|---|---|---|
| 41 s routine without the canter, 3-6 ms delay | 0/24 falls | 0/24 |
| trot -> sit, 5-15 ms delay | 0/48 | 0/48 |
| recovery (side, back, tipped, shove) | 20/20 | 20/20 |
| moving sit / lie (our exam) | 0/48 | 0/48 |
| our exam: tracking / steering | 0.775 / 0.441 | 0.778 / **0.534** |
| live on DEEP's simulator, steps over 1.2x rated per run | 7-14 | **5-12** |
| routine ending in a canter, 3-6 ms | 2/12 | 9/12 |

Steering up 21% at lower joint speeds, with the same safety record. The canter is worse: run_go2 runs the Lite3's
canter as a trot by default; for runs with --allow-canter use models/bolt-lite3-dog-canter-candidate (0/12).
