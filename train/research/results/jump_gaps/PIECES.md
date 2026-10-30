# jump-gaps (from jump-handover 59M; delay 0-30 ms + motor lag 0-25 ms + 10% lost packets per try, both hand-over
# sets, exams at 20 ms, brake on). Simulation, dev hand-overs from the v8g chain (100 each, 30/40/50 cm).

Run 1 (bolt-jump-gaps, 12 h, 59.2M -> 76.8M):

| | 20 ms delay | 20 ms + 20 ms motor lag + 10% lost packets |
|---|---|---|
| jump-delay 76M (in use for delay) | 88 / 82 / 70 | 20 / 11 / 9 |
| 62M / 66M / 70M | 16/5/0, 70/48/18, 78/59/24 | |
| 74M | 82 / 70 / 40 | 57 / 27 / 5 |
| 76.8M (last) | 87 / 76 / 43 | **55 / 32 / 8** |

Still rising at every checkpoint; far better with all gaps at 30/40 cm, worse at 50 cm with delay alone -> not used
yet; piece 2 (bolt-jump-gaps-p2, 3.5 h) continues from 76.8M.

jump-dyndelay piece 1 (bolt-jump-dyndelay-p1, 3.5 h, 0-40 ms per try + jitter 10 ms + spikes, from piece-2 end 83.6M):
dev = delayed hand-overs from the v10d chain (handover_states_v10d_delay_dev.npz, 96 states) at 20 ms +-8 ms jitter,
30/40/50 cm: 85M 70/60/38, 87M 69/57/36, 89M 70/59/36, 89.7M 70/56/28 vs jump-gaps 80.5M 80/73/47 -> worse, not used.
One flat piece; jump-dyndelay2 (also practising from the delayed hand-overs) continues.

## jump-dyndelay2 p1 (30 Oct, Kaggle bolt-jump-dyndelay2-p1, 89.7M-95.4M): flat -> the dyn-delay jump line ends

Delayed hand-overs (handover_states_v10d_delay_dev.npz, 96 states) at 20 ms +-8, jumps cleared at 30 / 40 / 50 cm:
80.5M 80 / 73 / 47; 90.5M 70 / 60 / 40; 92.5M 75 / 65 / 45; 94M 75 / 65 / 46; 95.4M 78 / 67 / 45. Below 80.5M at every
height; with jump-dyndelay p1 that is two flat pieces in a row (HANDOFF 0i). 80.5M stays the delay jump brain; this
run is the control arm for jump-delayaware (the same recipe plus the measured delay as an input).
