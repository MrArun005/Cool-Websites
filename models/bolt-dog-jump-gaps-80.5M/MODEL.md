# bolt-dog-jump-gaps-80.5M (simulation): best jump across delays alone; NOT used in MasterMind yet

Kaggle jump-gaps + piece 2 (from jump-handover 59M; delay 0-30 ms, motor lag 0-25 ms, 10% lost packets per try).
Picked on dev hand-overs (v8g chain) at 20 ms +- 8 ms jitter: 78.5M 84/73/44, 79.5M 87/74/46, **80.5M 93/84/59**,
81.5M 83/77/54, 82.5M 86/78/55, 83.5M 85/71/50 (30/40/50 cm).

Delay sweep (research/delay_sweep.py, v8g test hand-overs, mean of 30/40/50 cm; research/results/delay_sweep/):

| | 0 | 10 | 20 | 30 | 40 ms | 10/20/30/40 jittered | average | worst |
|---|---|---|---|---|---|---|---|---|
| jump-delay 76M | 73 | 89 | 82 | 42 | 8 | 74/68/32/10 | 53% | 8% |
| jump-gaps 76.8M | 36 | 70 | 79 | 67 | 35 | 53/57/47/27 | 53% | 27% |
| **this** | 45 | 75 | 82 | 69 | 41 | 62/66/49/34 | **58%** | **34%** |

Inside MasterMind's chain at 20 ms (teacher set v11d = v10d with this jump; dev seeds, 2 x 15 tries, brake on):
78.4% (451/575) vs v10d 79.6%; trot -> jump 20/30/40/50 cm 17/15/14/5 of 30 vs 26/22/15/5. Worse there: a delayed
trot hands over much messier (research/results/handover_states_v10d_delay*.npz: heading sd 0.24-0.34 rad vs 0.04).
Next: jump-dyndelay2 practises from those delayed hand-overs too. jump-delay 76M stays in v10d.
