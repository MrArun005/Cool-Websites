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

## Piece 2 (Kaggle bolt-steer-pilot2-p2, 41.7M -> 49.3M): passed

steer_mean 0.53 (42M) -> 0.64 (43M) -> 0.71 (45M) -> **0.74 (48M)** -> 0.70 (49.3M); trot 0.871-0.878, canter
0.708-0.723, 0 falls in the checkpoint exams. Saved as models/bolt-dog-steer-48M (MODEL.md: a bigger local check,
trot steering 0.89, canter steering 0.63, but canter tracking 0.641 with 1 fall in 12). Piece 3 running.

## Piece 3 (Kaggle bolt-steer-pilot2-p3, 49.3M -> 56.5M): small gain, canter partly back

In-run: steer_mean 0.69-0.76, canter 0.711-0.726 at 78-83% speed, 0 falls. Best 56.5M saved as
models/bolt-dog-steer-56.5M: local steering mean 0.79 (48M 0.76), canter steering 0.71 (0.63), canter tracking 0.659 (0.641,
38M 0.710), 1 canter fall in 12. Piece 4 running.

## Piece 4 (Kaggle bolt-steer-pilot2-p4, 56.5M -> 63M): flat -> the CPU steering line ends

In-run steer_mean 0.72-0.75, canter 0.717-0.732 at 83-87% speed, 0 falls. Local check of its best (61M) vs 56.5M: steering
mean 0.81 vs 0.79, canter steering 0.74 vs 0.71, trot 0.879 vs 0.877, canter tracking 0.658 vs 0.659, 1 canter fall
in 12 for both. Within the exam's noise -> flat (owner rule: no more flat runs). steer-56.5M stays the steering brain;
steering continues inside the Go2 GPU run (steer weight 0.6) instead of more CPU pieces.
