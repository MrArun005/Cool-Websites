# Steering pilot 1 (30 Oct, Kaggle bolt-steer-pilot, 2 h): no steering learned (simulation)

From canter-speed 38M, trot + canter, yaw-rate command +-1 rad/s on 60% of tries, steer term weight 0.15. Checkpoint exam
(imitate_env.exam + exam_steer, 2 tries per clip x rate):

| steps | trot | canter (speed) | steer_mean (trot / canter) | falls |
|---|---|---|---|---|
| 38.0M start | 0.876 | 0.715 (0.79) | 0.156 (0.10 / 0.21) | 0 |
| 40.0M | 0.885 | 0.728 (0.82) | 0.132 | 0 |
| 42.0M | 0.883 | 0.730 (0.85) | 0.120 | 0 |
| 42.4M (best/) | 0.880 | 0.728 (0.85) | 0.144 (0.18 / 0.11) | 0 |

Fail (win line steer_mean >= 0.6). The pose and paw terms (0.7 of the reward) pay for the straight reference legs;
a 0.15 steer term does not outweigh them. Next: steer-pilot2 (steer weight 0.6, global root terms 0.5).
Side effect: canter 0.715 -> 0.728 at 85% of the dog's speed (from 79%), trot kept.
