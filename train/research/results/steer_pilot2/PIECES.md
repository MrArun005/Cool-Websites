# Steering pilot 2 (30 Oct): learning to steer at a trot and canter (simulation)

From canter-speed 38M; steer term weight 0.6, global root terms 0.5, commands +-0.8 rad/s on 80% of tries.
Checkpoint exam (exam_steer, 2 tries per clip x rate; imitate_env.exam):

| steps | steer_mean (trot / canter) | trot | canter (speed) | falls |
|---|---|---|---|---|
| 38.0M start | 0.17 (0.10 / 0.25) | 0.876 | 0.715 (0.79) | 0 |
| 40.0M | 0.24 (0.32 / 0.16) | 0.873 | 0.715 (0.78) | 0 |
| 41.0M | 0.42 (0.54 / 0.30) | 0.873 | 0.716 (0.79) | 0 |
| 41.7M | 0.49 (0.61 / 0.38) | 0.874 | 0.719 (0.77) | 0 |

Pilot 1 (weight 0.15) never moved (0.12-0.16). Pilot 2 climbs every checkpoint with tracking kept and no falls: not yet
the 0.6 win line, but clearly working -> piece 2 (bolt-steer-pilot2-p2, 3.5 h) continues it.
