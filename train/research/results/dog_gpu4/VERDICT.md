# Kaggle bolt-dog-gpu4 (31 Oct): Bolt dog motion on the Kaggle GPU, motor brake on -- flat, line ended (simulation)

Warm from gpu3 617M (models/bolt-dog-imitate-gpu3-617M), motor brake on (above 25 rad/s, 3.5 N m per rad/s),
10.5 h, 617M more steps at ~16.4k steps/s. Training eval (128 envs), every ~100M steps:

| steps | 0 | 103M | 206M | 309M | 412M | 514M | 617M |
|---|---|---|---|---|---|---|---|
| reward/step | 0.673 | 0.662 | 0.649 | 0.690 | 0.640 | 0.698 | 0.661 |
| canter | 0.365 | 0.284 | 0.346 | 0.367 | 0.368 | 0.407 | 0.394 |
| falls | 15% | 15% | 16% | 13% | 13% | 12% | 14% |

No gain in 10.5 h (noise band only), so no full exam was spent on it: gpu3 617M stays the Bolt GPU motion brain.
More of the same recipe is not the way up for Bolt's canter (0.71 on the CPU canter-speed 38M brain); what moved
the Go2 canter today was a fresh recipe (steering, root-global 0.5, 0-40 ms delay) at full precision.
