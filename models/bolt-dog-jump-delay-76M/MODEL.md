# bolt-dog-jump-delay-76M (simulation): the delay-proof jump brain (for a real body)

Kaggle job jump-delay (29 Oct, 12 h CPU): from jump-handover 59M, every try with 0-30 ms control delay drawn per try,
real hand-over practice, motor brake on. Checkpoints re-tested here (Kaggle's best/ was picked on no-delay exams).
Picked on dev hand-overs from the v8g chain at 20 ms (research/results/handover_states_v8g_dev.npz, 100 each, 30/40/50 cm):
64M 42/23/14, 68M 78/59/49, 72M 69/68/50, **76M 88/82/70**, 78.1M 86/84/71.

Verdict on test hand-overs (handover_states_v8g_test.npz, never used to pick), clean jumps of 100:

| | 20 cm | 30 cm | 40 cm | 50 cm |
|---|---|---|---|---|
| jump-handover 59M, 20 ms delay | 7 | 2 | 1 | 0 |
| **this, 20 ms delay** | **85** | **92** | **86** | **67** |
| jump-handover 59M, no delay | 85 | 90 | 85 | 70 |
| this, no delay | 77 | 83 | 81 | 54 |

Use: wherever control delay is modelled or real (a real robot has ~10-20 ms); jump-handover 59M stays the no-delay
simulation brain (Day Out, teacher sets without delay). It jumps better WITH delay than without: it learned to
expect it.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_exam.py ../models/bolt-dog-jump-delay-76M \
        --states research/results/handover_states_v8g_test.npz --heights 0.20 0.30 0.40 0.50 --motor-brake --latency-ms 20
