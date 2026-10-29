# Teacher chains with the 30 Oct brains (simulation, test_master.py --brain teacher, dev seeds 20000 + 20015,
# 2 x 15 tries per switch, with 50 cm, motor brake on)

| | pooled | trot -> jump 20/30/40/50 cm | landing -> trot (20/30/40/50) | fall -> stood up |
|---|---|---|---|---|
| v7c, no delay | 95.5% (591/619) | 27/27/26/21 of 30 | | 26/30 |
| v10n = v7c + recover-delay 30M, no delay | 95.6% (593/620) | 27/27/26/21 | | 27/30 |
| v8g, 20 ms delay | 69.0% (362/525) | 2/1/1/1 of 30 | 2/10, 4/13, 0/10, 3/7 | 18/30 |
| **v10d** = v8g + jump-delay 76M + recover-delay 30M, 20 ms | **79.6% (483/607)** | **26/22/15/5** | 14/28, 15/28, 12/29, 10/28 | **27/30** |

v10d is the delay teacher set (+10.6 points at 20 ms; jumps from ~0 to 87/73/50/17%). New weak spot under delay:
after landing, the canter -> trot back to the gait (~50%) and 50 cm jumps. v10n = v7c for switching (recovery 26 -> 27,
within chance), kept since recover-delay 30M is better on its own exam.

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain teacher --teachers v10d --n 15 \
        --seed 20000 --with-50 --motor-brake --latency-ms 20
