# MasterMind v11d (30 Oct): the delay brain, 84% of switches at 20 ms control delay (simulation)

Kaggle job mastermind-v11d (18 DAgger rounds, 10.5 h): teachers v10d (gpu3 dog motion, recover-delay 30M,
jump-delay 76M), from export_v8g, delay 0-30 ms per try, sensor noise, motor lag 0-20 ms. Round picked on dev seeds
20000 at 20 ms (15 tries): r11 83%, r13 84%, r15 81%, r17 84% (exams/dev20ms_r*.log). Verdict on test seeds 1000, 30
tries per switch, with 50 cm (exams/test_r13_*.log):

| | v8g 0 ms | **v11d 0 ms** | v8g 20 ms | **v11d 20 ms** |
|---|---|---|---|---|
| all switches | 94% (465/496) | 88% (445/503) | 62% (260/420) | **84% (419/500)** |
| trot -> jump 30 / 40 / 50 cm | 90 / 93 / 67% | 80 / 47 / 13% | 3 / 0 / 0% | **97 / 87 / 43%** |
| fall -> stood up | 90% | 97% | 43% | **93%** |
| jump -> (canter) -> trot | 97-100% | 100% | 50-76% | 50-63% |

Use v11d where there is control delay (a real body); without delay v9 (95.9%) stays the main brain, as v11d's
40/50 cm jumps drop at 0 ms. Weak spot under delay: getting back into the trot after landing (50-63%).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v11d/student.pt \
        --teachers v10d --n 30 --seed 1000 --with-50 --exam --latency-ms 20
