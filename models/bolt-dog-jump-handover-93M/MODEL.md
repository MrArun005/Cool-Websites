# bolt-dog-jump-handover-93M (simulation): the trot hand-over jump brain from 30 Oct (no-delay sets)

Kaggle job jump-handover-mix (12 h CPU): from jump-handover 74M, practising from 800 recorded hand-overs (400 from the
v6h chain at 1.69 m/s, 400 from the v8g chain at 1.50 m/s; handover_states_mix.npz), motor brake on. Checkpoints
re-tested here. Dev hand-overs from the v8g chain (handover_states_v8g_dev.npz, 100 each, 20/30/40/50 cm):
59M 76/84/86/70, 77M 68/77/94/92, 81M 79/86/98/94, 85M 81/93/92/95, 89M 91/95/97/97, **93M 92/95/97/97**,
97M 90/95/97/98, 100.8M 90/91/97/98.

Verdict on held-out hand-overs (never used to pick), clean jumps of 100:

| | v8g-chain test 20/30/40/50 cm | v6h-chain test 20/30/40/50 cm |
|---|---|---|
| jump-handover 59M | 85 / 90 / 85 / 70 | 84 / 90 / 87 / 73 |
| **this (93M)** | **89 / 93 / 95 / 100** | **85 / 91 / 92 / 99** |

Inside MasterMind's teacher chain (v11n = v10n with this jump; dev seeds 20000 + 20015, 2 x 15 tries, brake on):
pooled 95.6% (587/614), same as v10n 95.6% (593/620); trot -> jump 20/30/40/50 cm 24/26/25/28 of 30 vs 27/27/26/21.
50 cm +7 (beyond chance), lower bars -1 to -3 (within chance). Used in teacher set v11n for the next MasterMind.
Not delay-trained: jump-delay 76M stays the delay brain.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_exam.py ../models/bolt-dog-jump-handover-93M \
        --states research/results/handover_states_v8g_test.npz --heights 0.20 0.30 0.40 0.50 --motor-brake
