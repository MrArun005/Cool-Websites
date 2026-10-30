# recover-gaps pieces (HANDOFF 0i): from recover-delay 30M, delay 0-30 ms + motor lag 0-25 ms + 10% lost packets,
# exams at 20 ms, motor brake on. Simulation.

Piece 1 (bolt-recover-gaps-p1, 3.5 h, 30M -> 37.3M). Dev seed 7000 at 20 ms (15 tries x 6 falls): 32M 77.7%,
34M 73.2%, 36M 81.0%, 37.3M 69.0% (30M: 75.7%). Test seed 9000, 30 tries x 6 falls, stood up:

| | no delay | 20 ms | 20 ms + 20 ms motor lag |
|---|---|---|---|
| recover-delay 30M (in use) | 87.3% | 74.5% | 67.8% |
| piece 1 36M | 78.8% | 71.0% | 72.2% |

Not better -> not used. One flat piece: piece 2 (bolt-recover-gaps-p2) continues; a second flat piece ends the recipe.

Piece 2 (bolt-recover-gaps-p2, 3.5 h, 37.3M -> 46M). Dev seed 7000 at 20 ms, stood up: 38M 70.0%, 39M 71.1%, 40M 74.4%,
41M 65.6%, 42M 71.1%, 43M 74.4%, 44M 65.6%, 45M 72.2%, 46M 70.0% vs recover-delay 30M 75.7%. Second flat piece ->
recipe ended (HANDOFF 0i); recover-delay 30M stays. The slot went to recover-dyndelay (0-40 ms, per-step jitter, spikes).
