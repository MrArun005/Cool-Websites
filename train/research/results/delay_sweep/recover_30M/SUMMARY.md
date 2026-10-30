# Recovery delay sweep, recover-delay 30M (the brain in use), simulation
fair_exam recover --episodes 30 --seed 9000 --motor-brake (6 fall kinds x 30 = 180 tries per row), mean "stood up":

| delay | steady | jitter +-8 ms |
|---|---|---|
| 0 ms | 87.2% | |
| 10 ms | 78.9% | 72.2% |
| 20 ms | 74.4% | 76.1% |
| 30 ms | 64.4% | 66.7% |
| 40 ms | 46.7% | |

Worst 46.7% (40 ms), average 70.8%. The bar for recover-dyndelay: worst AND average above these.
