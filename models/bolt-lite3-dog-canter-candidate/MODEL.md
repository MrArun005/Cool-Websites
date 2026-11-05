# bolt-lite3-dog-canter-candidate (runner-up: the best Lite3 canter under delay)

DEEP Robotics Lite3 motion brain, simulation only. Weight-space blend: 0.75 x bolt-lite3-dog-blend-112M-w7 (the
robot brain) + 0.25 x Kaggle lite3-dog-7's best (87.3M fine-tuned at lr 3e-5, target-kl 0.01, 0-40 ms; 105M).
numpy params.pkl only.

On DEEP's own Lite3 model (1 ms physics, robot-only inputs), against the robot brain:

| Test | robot brain | this |
|---|---|---|
| routine without canter, 3-6 ms | 0/24 | 0/24 |
| trot -> sit, 5-15 ms | 0/48 | 2/48 |
| routine ending in a canter, 3-6 ms | 2/12 | **0/12** |
| recovery | 20/20 | 20/20 |
| moving sit / lie | 0/48 | 0/48 |
| our exam: tracking / steering | 0.775 / 0.441 | 0.791 / 0.475 |

Not the default: it falls in trot -> sit 2/48, the one move the robot brain never fails. For a run with the canter on
(`run_go2.py --body lite3 --allow-canter --brain models/bolt-lite3-dog-canter-candidate/params.pkl`), it is the
better choice.
