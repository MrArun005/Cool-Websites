# bolt-dog-jump-handover-74M (simulation): the jump brain Bolt uses from 29 Oct

Kaggle job jump-handover (29 Oct, 12 h CPU, from jump-trot 55M, 55.0M -> 74.2M steps): practises from REAL
MasterMind trot -> jump hand-overs (research/record_handover.py, handover_states.npz) plus synthetic starts, motor
brake on. Final checkpoint 74.18M. Kaggle's own best/ never moved off the start (picked on a noisy 10-try exam), so the
checkpoints were re-tested here on held-out hand-over states.

Picked on dev hand-overs (handover_states_dev.npz, seeds 70000+, 100 states, brake on), clean jumps:

| | 20 cm | 30 cm | 40 cm | 50 cm |
|---|---|---|---|---|
| jump-handover 59M (previous) | 86 | 87 | 88 | 79 |
| 62M | 92 | 90 | 90 | 85 |
| 66M | 88 | 88 | 88 | 85 |
| 70M | 86 | 90 | 90 | 86 |
| **74.18M** | 82 | 89 | 93 | **91** |

Verdict on test hand-overs (handover_states_test.npz, seeds 90000+, never used to pick):

| | 20 cm | 30 cm | 40 cm | 50 cm |
|---|---|---|---|---|
| jump-handover 59M | 84 | 90 | 87 | 73 |
| **74.18M** | 87 | 86 | 91 | **90** |

50 cm +17 points (73 -> 90 of 100); 30 cm -4 is within chance on 100 tries. Standard exam (fair_exam, 1.7 m/s from
1.1 m, 30 tries, seed 9000, brake on): still legs 100/100/100% at 30/40/50 cm for both, mid-stride 97/97/97% (59M
100/100/97%). Not delay-proof (like 59M): delay-trained jump is the separate Kaggle job jump-delay.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_exam.py ../models/bolt-dog-jump-handover-74M \
        --states research/results/handover_states_test.npz --heights 0.20 0.30 0.40 0.50 --motor-brake
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python fair_exam.py jump ../models/bolt-dog-jump-handover-74M --speeds 1.7 \
        --dist 1.1 --heights 0.30 0.40 0.50 --gait --max-steps 220 --motor-brake --episodes 30 --no-save
