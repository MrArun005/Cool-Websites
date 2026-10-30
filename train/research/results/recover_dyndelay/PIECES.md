# recover-dyndelay (30 Oct): ended after two flat pieces (simulation)

From recover-delay 30M; delay 0-40 ms per try, +-10 ms jitter, spikes, motor lag, lost packets; checkpoint exam at
20 ms +-8 (full recovery, 15 tries per fall kind, seed 7000), mean "stood up":

| piece | steps | best checkpoint | vs 30M (80.0%) |
|---|---|---|---|
| p1 (Kaggle bolt-recover-dyndelay) | 30-31.5M | 31M 84.4% | sweep: avg 69.7 / worst 48.9 vs 30M 70.8 / 46.7 (research/results/delay_sweep/recover_dyn31M) - not better |
| p2 (bolt-recover-dyndelay-p2) | 31.5-44.1M | none above 31M (best 43M 83.3%, worst 39M 64.4%) | flat |

Two flat pieces in a row (HANDOFF 0i): the recipe ends; 30M stays the recovery brain. The delay-aware recovery
(recover-delayaware, the measured delay as an input) is the next try at "any delay".
