# jump-delayaware (30 Oct): the measured delay as an input (input 52), from jump-gaps 80.5M (simulation)

Pilot (Kaggle bolt-jump-delayaware-p1, 2 h, 80.5M -> 83.7M; jump-dyndelay2's recipe + the input). Delayed hand-overs
(handover_states_v10d_delay_dev.npz, 96) at 20 ms +-8, cleared at 30/40/50 cm (dev_*.log): 81M 71/64/40, 82M 78/66/41,
83M 76/62/35, 83.7M 77/64/25 vs 80.5M 80/73/47. Delay sweep of 82M (sweep_82M.json; v8g test hand-overs):

| | 0 | 10 | 20 | 30 | 40 ms | jitter 10/20/30/40 | average / worst |
|---|---|---|---|---|---|---|---|
| 80.5M | 45 | 75 | 82 | 69 | 41 | 62/66/49/34 | 58 / 34 |
| 82M (delay input) | 49 | 78 | 77 | 71 | 42 | 60/63/55/31 | 58 / 31 |

A tie, not a win (flat piece 1 of 2). Only 3.2M steps; piece 2 (3.5 h) decides.
