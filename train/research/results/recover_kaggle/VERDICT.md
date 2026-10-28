# Kaggle recovery run (28 Oct, best at 37M) vs 17e-16.5M: keep 17e

    MUJOCO_GL=osmesa python fair_exam.py recover <folder> --easy --episodes 30 --seed 9000   (and --seed 11000)

| | new (seed 9000 / 11000) | 17e (9000 / 11000) |
|---|---|---|
| all six falls, mean stood up | 88% / 90% | 82% / 87% |
| shoved while standing (old exam) | 87% / 70% | 93% / 100% |

+4.5 points on the falls (about 1.6 standard errors) but clearly worse at staying up when shoved while standing,
which Bolt's Day Out uses. Not clearly better, so 17e stays (rule: keep unless clearly better). Files: this folder.
