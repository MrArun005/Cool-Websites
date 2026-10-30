# MasterMind v10 (30 Oct): the canter brain (canter tracking 0.714); v9 stays the main brain (simulation)

Kaggle job mastermind-v10 (17 DAgger rounds, teachers v7c = v7s with the faster canter canter-speed 38M, from
export_v9). Dev seeds 20000 (15 tries): r10 95.8%, r12 95.1%, r14 96.4%, r16 96.1%. Test seeds 1000, 30 tries, with
50 cm (exams/test_r*.log):

| | v9 | r14 | **v10 = r16** |
|---|---|---|---|
| all switches | **95.9%** | 94.0% | 94.8% (582/614) |
| tracking walk / trot / canter | 0.909 / 0.892 / 0.558 | 0.908 / 0.889 / 0.715 | 0.910 / 0.890 / **0.714** |
| canter -> trot | 29/30 | 23/30 | 22/30 |

Canter tracking +0.156 (meets the owner's canter >= 70% target, no falls), switching -1.1 (the faster canter is
harder to leave). v9 stays the main brain (switching is the 99% target); use v10 where the canter matters. v12
(teachers v11n, includes this canter) is the next try at both.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v10/student.pt \
        --teachers v7c --n 30 --seed 1000 --with-50 --exam
