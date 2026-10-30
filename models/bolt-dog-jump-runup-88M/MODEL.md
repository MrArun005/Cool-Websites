# bolt-dog-jump-runup-88M (simulation): the long run-up jump brain (best 60 cm so far)

Kaggle job jump-runup (29-30 Oct, 12 h CPU, after a 1 h pilot): from jump-run60 74M, long run-ups only (1.5-4 m,
1.8-3.2 m/s, mostly mid-stride), practice bar from 50 cm, motor brake on. Kaggle's best/ (picked on its noisy 10-try
exam) was re-picked here on dev seed 7000 (20 tries, mid-stride, 2.0/2.5 m/s, 50/60 cm summed): 75.85M pilot 1.20,
80M 1.20, 84M 1.50, **88M 1.70**, 92M 1.60, 96.1M 1.20.

Verdict, test seed 9000, 30 tries, mid-stride, from 3 m, brake on (exams/test_*.log):

| | 50 cm at 2.0 / 2.5 / 3.0 m/s | 60 cm at 2.0 / 2.5 / 3.0 m/s |
|---|---|---|
| jump-run60 74M (start) | 33 / 43 / 13% | 10 / 13 / 0% |
| **this (88M)** | **57 / 63 / 17%** | **23 / 33 / 0%** |
| 92M | 53 / 50 / 17% | 20 / 23 / 0% |

A separate brain for long run-ups (not the 1.1 m trot hand-over, which stays jump-handover 59M / jump-delay 76M).
Training flattened after 88M (later checkpoints lower), so no further pieces of this recipe.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python fair_exam.py jump ../models/bolt-dog-jump-runup-88M --speeds 2.0 2.5 3.0 \
        --dist 3.0 --heights 0.50 0.60 --gait --max-steps 220 --motor-brake --episodes 30 --seed 9000 --no-save
