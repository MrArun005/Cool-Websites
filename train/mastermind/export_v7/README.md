# MasterMind v7 (29 Oct): replaces v4 as the one-network brain

Local DAgger round 1 from the v4 student, teacher set v7 (gpu3 for sit/lie/canter, the jump brain lands Bolt,
jump brain trained on real hand-overs: teacher chain 94.0% on dev seeds). All numbers in simulation.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/dagger.py --teachers v7 --steps0 400000 --steps 250000 --rounds 3 \
        --workers 4 --epochs0 10 --epochs 5 --hidden 1024,512,256 --beta-floor 0.2 --seed 7 \
        --init-student mastermind/export_v4/student.pt --out runs/pilot_mm7        # student_r1.pt = this brain
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v7/student.pt \
        --teachers v7 --n 30 --seed 1000 --with-50 --exam --out mastermind/results/v7/test_pilot_r1.json

Picked on dev seeds 20000 (r1 91.4%, r2 91.3%, v4 86.9%); verdict on fresh test seeds 1000, 30 tries per switch:

| | v4 | v7 |
|---|---|---|
| all switches pooled | 83.3% | **91.3%** |
| without the 50 cm rows | 87.4% | **92.1%** |
| canter -> trot | 26/29 | 30/30 |
| jump -> back to trot (20/30/40/50 cm) | 22/30, 12/30, 14/27, 14/25 | 27/27, 28/28, 24/24, 23/23 |
| trot -> jump 50 cm | 9/30 | 21/30 |
| trot -> jump 20 / 30 cm | 26/30, 26/30 | 20/30, 20/30 (worse) |
| walk -> sit -> lie, pose reached | 25/30 | 22/29 (worse) |
| fall -> stood up | 27/30 | 25/30 (worse) |
| canter tracking (exam) | 0.695 | 0.548 (worse: gpu3 is its canter teacher) |

Not yet the 95% target. Weak spots: low trot -> jump bars, lying down from a walk, the canter's look.
The Kaggle MasterMind v7 run (10.7 h, same teachers) may replace it.
