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
