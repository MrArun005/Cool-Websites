# bolt-dog-jump-run60-74M (simulation): long run-up jump, start brain for the run-up work

From Kaggle job jump-run60 (28-29 Oct, research/results/jump_run60/VERDICT.md), checkpoint 74.0M. Not a Bolt brain in
use: the only brain so far that clears 50 cm from a long run-up, kept as the start of a separate run-up jump brain.

Fair exams on Kaggle (fair_exam.py, 30 tries, seed 9000, motor brake on, mid-stride, from 3 m), simulation:
50 cm at 2.0 / 2.5 / 3.0 m/s: 33 / 43 / 13%; 60 cm: 10 / 13 / 0%. Trot hand-over (1.7 m/s from 1.1 m, mid-stride)
40 / 50 cm: 67 / 0% (worse than jump-handover 59M, which stays the hand-over brain).

    cd train
    MUJOCO_GL=osmesa python fair_exam.py jump ../models/bolt-dog-jump-run60-74M --speeds 2.0 2.5 3.0 --dist 3.0 \
        --heights 0.50 0.60 --gait --max-steps 220 --motor-brake --episodes 30
