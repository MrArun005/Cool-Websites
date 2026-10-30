# Recovery delay sweep, recover-dyndelay piece 1 best (31M; dev seed 7000 at 20 ms +-8: 84.4% vs 30M 80.0%)
fair_exam recover --episodes 30 --seed 9000 --motor-brake, mean "stood up" (180 tries per row), simulation:

| delay | 30M steady | this steady | 30M jitter +-8 | this jitter +-8 |
|---|---|---|---|---|
| 0 ms | 87.2 | 85.6 | | |
| 10 ms | 78.9 | 75.0 | 72.2 | 76.7 |
| 20 ms | 74.4 | 76.7 | 76.1 | 75.0 |
| 30 ms | 64.4 | 60.6 | 66.7 | 58.3 |
| 40 ms | 46.7 | 48.9 | | |
| average / worst | 70.8 / 46.7 | 69.7 / 48.9 | | |

Not better on both (average -1.1, worst +2.2; both within chance) -> not used; piece 2 (bolt-recover-dyndelay-p2) runs.
