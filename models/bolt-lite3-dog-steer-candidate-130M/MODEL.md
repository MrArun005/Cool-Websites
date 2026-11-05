# bolt-lite3-dog-steer-candidate-130M (runner-up: the best Lite3 steering that keeps the sits)

DEEP Robotics Lite3 motion brain, simulation only. Kaggle lite3-dog-10: 112M fine-tuned with the canter drawn 3x,
0-60 ms delay, moving starts 0.75, lr 2e-5, target-kl 0.01; best checkpoint 129,981,608 steps (policy.zip +
vecnormalize.pkl; params.pkl from `mjx_imitate.py warm --body lite3`).

On DEEP's own Lite3 model (1 ms physics, robot-only inputs), against the robot brain (blend-112M-w7):

| Test | robot brain | this |
|---|---|---|
| routine without canter, 3-6 ms | 0/24 | 0/24 |
| trot -> sit, 5-15 ms | 0/48 | 0/48 |
| routine ending in a canter, 3-6 ms | 2/12 | 7/12 |
| recovery | 20/20 | 20/20 |
| moving sit / lie | 0/48 | 0/48 |
| our exam: tracking / steering | 0.775 / 0.441 | 0.777 / **0.516** |
| live on DEEP's simulator, 4 runs | no fall; 7-14 steps over 1.2x rated | no fall; **17-25** steps over 1.2x rated |

Not the default: it drives the joints past 1.2x their rated speed about twice as often, the thing most likely to trip
the speed stop on a real robot, and it canters worse. If hardware day shows motor speed headroom, it is the better
steerer.
