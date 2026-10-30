# bolt-dog-steer-48M (simulation): the first Bolt brain that turns on command at a trot (31 Oct)

Kaggle steer-pilot2, pieces 1-2 (5.5 h CPU): from canter-speed 38M, trot + canter, a yaw-rate command of +-0.8 rad/s
on 80% of tries (input 73), steer reward weight 0.6, global root terms 0.5 (train_imitate.py --steer-max 0.8
--steer-share 0.8 --steer-weight 0.6 --root-global 0.5). best/ picked at 48.0M on the steering-weighted exam score.

Local check (imitate_env.exam_steer: commands -1, -0.5, +0.5, +1 rad/s, 6 tries each, 4 s, seed 12000; exam: 12
tries per clip, seed 31000; randomized physics, brake on):

| | canter-speed 38M | this (48M) |
|---|---|---|
| follows the turn command, trot (0-1) | 0.02 | **0.89** (turns -0.94 / -0.51 / +0.40 / +0.84 rad/s) |
| follows the turn command, canter | 0.19 | 0.63 |
| trot tracking / speed | 0.876 / 0.93 | 0.877 / 0.94 |
| canter tracking / speed / falls | 0.710 / 0.77 / 0 | 0.641 / 0.70 / 1 of 12 |

Use it for trot steering (games, following a path). The canter costs: keep canter-speed 38M for the canter until a
later piece wins it back (piece 3 is running).

    cd train
    MUJOCO_GL=osmesa python -c "from imitate_env import ImitatePolicy, exam_steer; print(exam_steer(ImitatePolicy('../models/bolt-dog-steer-48M'), episodes=6, seed=12000))"
