# MasterMind v12 (30 Oct): the main brain, 96.9% of switches with the fast canter (simulation)

Kaggle job mastermind-v12 (17 DAgger rounds, teachers v11n: v7c with the faster canter 38M, recover-delay 30M,
jump-handover 93M), from export_v9. Dev seeds 20000 (15 tries): r10 96%, r12 96%, r14 97%, r16 96% (exams/dev_r*.log).
Test seeds 1000, 30 tries per switch, with 50 cm (exams/test_r14.log; "[pose reached]" rows not double-counted):

| | v9 | v10 | **v12 = r14** |
|---|---|---|---|
| all switches | 95.9% (591/616) | 94.8% (582/614) | **96.9% (601/620)** |
| tracking walk / trot / canter | 0.909 / 0.892 / 0.558 | 0.910 / 0.890 / 0.714 | **0.908 / 0.889 / 0.715** |
| trot -> jump 20 / 30 / 40 / 50 cm | 27 / 25 / 27 / 22 of 30 | | 27 / 26 / 27 / **27** of 30 |
| canter -> trot | 29/30 | 22/30 | 28/30 |
| fall -> stood up | 25/30 | 29/30 | 26/30 |

v12 replaces v9 as the main brain (no delay) and v10 as the canter brain: v9's switching, v10's canter. With control
delay use v11d (84% at 20 ms). Failing rows: jumps 20-50 cm 3-4 of 30 each, recovery 4/30.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v12/student.pt \
        --teachers v11n --n 30 --seed 1000 --with-50 --exam
