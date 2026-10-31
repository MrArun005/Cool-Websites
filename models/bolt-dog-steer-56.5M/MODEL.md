# bolt-dog-steer-56.5M (simulation): the trot-steering brain, piece 3 (31 Oct); replaces steer-48M

Kaggle steer-pilot2 pieces 1-3 (9 h CPU) from canter-speed 38M; recipe as models/bolt-dog-steer-48M/MODEL.md.
Same local check as 48M (exam_steer 6 tries per command, seed 12000; exam 12 tries per clip, seed 31000; brake on):

| | canter-speed 38M | steer 48M | **this (56.5M)** |
|---|---|---|---|
| follows the turn command, trot | 0.02 | 0.89 | 0.88 (turns -1.00 / -0.55 / +0.39 / +0.85 rad/s for -1 / -0.5 / +0.5 / +1) |
| follows the turn command, canter | 0.19 | 0.63 | **0.71** |
| steering mean | 0.11 | 0.76 | **0.79** |
| trot tracking / speed | 0.876 / 0.93 | 0.877 / 0.94 | 0.877 / 0.94 |
| canter tracking / speed / falls | **0.710 / 0.77 / 0** | 0.641 / 0.70 / 1 of 12 | 0.659 / 0.77 / 1 of 12 |

The trot steering brain for games and path following; the canter itself stays on 38M (this canter still trails it).
