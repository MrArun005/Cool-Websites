# Recovery delay sweep, recover-delayaware 32M (the measured delay as input 47; Kaggle bolt-recover-delayaware-p1)
fair_exam recover --episodes 30 --seed 9000 --motor-brake (180 tries per row), mean "stood up", simulation:

| delay | 30M steady | this steady | 30M jitter +-8 | this jitter +-8 |
|---|---|---|---|---|
| 0 ms | 87.2 | 83.3 | | |
| 10 ms | 78.9 | 76.1 | 72.2 | 79.4 |
| 20 ms | 74.4 | 73.3 | 76.1 | 73.9 |
| 30 ms | 64.4 | 59.4 | 66.7 | 61.7 |
| 40 ms | 46.7 | 51.7 | | |
| average / worst | 70.8 / 46.7 | 69.9 / 51.7 | | |

Worst +5.0, average -0.9: not better on both -> not used (flat piece 1). In-run exam (20 ms +-8): start 83.3%, best 32M 82.2%.
Like recover-dyndelay (69.7 / 48.9), recovery under delay sits at ~70% average whatever the recipe so far.

Piece 2 (Kaggle bolt-recover-delayaware-p2, 34-41M): in-run exam (20 ms +-8) 73-83%, best 41M 83.3% = the start's
83.3%. Second flat piece -> the delay-input recovery line ends; 30M stays.
