# bolt-dog-recover-delay-30M (simulation): getting up after a fall, better with and without delay

Kaggle job recover-delay (29 Oct, 12 h CPU): from self-righting 17e 16.5M, tough falls (shoves, drops, ice), 0-30 ms
control delay drawn per try, motor brake on (so examined with the brake on, HANDOFF 0c; 17e without, as trained).
Kaggle's best/ was picked on no-delay exams, so checkpoints were re-tested here.

Picked on dev seeds 7000 at 20 ms (fair_exam recover, 15 tries per fall kind, mean "stood up"): 30M 75.7%, 38M 64.5%,
44M 75.7%, 48M 68.8%, 51.1M 68.8%; 17e 64.3% (brake on 67.8%). Verdict on test seeds 9000, 30 tries per kind
(side, back, tipped, shoved, dropped, ice = 180 tries):

| | stood up, no delay | stood up, 20 ms delay |
|---|---|---|
| 17e (previous) | 81.7% | 63.7% |
| **this (30M)** | **87.3%** | **74.5%** |
| 44M | 79.3% | 62.7% |

Better at both. Still short of the 85%-at-20-ms target: more delay training (in 3.5 h pieces, exams at 20 ms).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python fair_exam.py recover ../models/bolt-dog-recover-delay-30M --episodes 30 \
        --seed 9000 --latency-ms 20 --motor-brake --no-save
