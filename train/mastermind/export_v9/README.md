# MasterMind v9 (29 Oct): v8g-level switching with v8's dog-like motion (simulation)

Kaggle job mastermind-v7 (10.5 h CPU, 18 DAgger rounds, teachers v7, from export_v4; exams/summary.json has the
arguments). Round 13 picked on dev seeds 20000 (15 tries per switch: r9 95.8%, r13 96.8%, r15 96.8%, r17 96.1%);
verdict on fresh test seeds 1000, 30 tries per switch, with 50 cm (exams/test_r13.log):

| | v8 | v8g | **v9 (r13)** |
|---|---|---|---|
| all switches | 92.4% | 95.3% | **95.9%** (591/616) |
| tracking walk / trot / canter | 0.908 / 0.888 / 0.698 | 0.839 / 0.784 / 0.548 | **0.909 / 0.892** / 0.558 |
| at 20 ms control delay | 68.3% (322 reached) | **70.4% (540 reached)** | 75.8% (297 reached) |

v9 vs v8g on switching is a tie within chance (+0.6 points, about 4 switches); on dog-like motion it is clearly
better. It falls more under 20 ms delay (fewer switches reached), so v8g stays the delay-robust brain and v9 replaces
v8 as the main brain where delay is not modelled. Failing rows (test): trot -> jump 20/30/40/50 cm 27/25/27/22 of 30,
fall -> stood up 25/30, canter -> trot 29/30. Toward 99%: the jump hand-over and recovery teachers.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v9/student.pt \
        --teachers v7 --n 30 --seed 1000 --with-50 --exam [--latency-ms 20]
