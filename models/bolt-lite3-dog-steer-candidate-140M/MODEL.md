# bolt-lite3-dog-steer-candidate-140M (runner-up: better steering at safe joint speeds)

DEEP Robotics Lite3 motion brain, simulation only. Kaggle lite3-dog-15: the steering candidate 130M fine-tuned with a
x1.3 motor drive (closer to DEEP's speed-unlimited model) and the over-speed penalty (train_imitate --overspeed-weight
0.5), lr 1e-5; checkpoint 140,981,608 steps (policy.zip + vecnormalize.pkl; params.pkl from `mjx_imitate.py warm`).

On DEEP's own Lite3 model (1 ms physics, robot-only inputs), against the robot brain and 130M:

| Test | robot brain | 130M | this |
|---|---|---|---|
| routine without canter, 3-6 ms | 0/24 | 0/24 | 0/12 |
| trot -> sit, 5-15 ms | 0/48 | 0/48 | 2/72 |
| routine ending in a canter, 3-6 ms | 2/12 | 7/12 | 5/12 |
| our exam: tracking / steering | 0.775 / 0.441 | 0.777 / 0.516 | 0.784 / 0.517 |
| live, steps over 1.2x rated per run | 7-14 | 17-25 | 9-16 |

The over-speed fix worked (130M's 17-25 down to the robot brain's range, steering kept), at the cost of 2 trot->sit
falls in 72 under 5-15 ms of delay. Not the default for that reason.
