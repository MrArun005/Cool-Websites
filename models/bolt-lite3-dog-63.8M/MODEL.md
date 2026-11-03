# bolt-lite3-dog-63.8M (simulation): the DEEP Robotics Lite3 robot brain (3 Nov evening, replaces 61.8M)

From bolt-lite3-dog-61.8M on a free Kaggle CPU session (bolt-lite3-dog-4): the corrected leg collision shapes, 0-60 ms
of control delay per try, moving sit/lie starts 0.5, steering +-0.8. Best checkpoint at 63.8M steps.

DEEP's own MJCF (lockstep_vendor / lockstep_recovery, estimated velocity and height, 1 ms physics):

| test | 61.8M | **63.8M** |
|---|---|---|
| 41 s routine, falls | 0/3 | 0/3 |
| joint steps over 1.2x rated in it | 342 | **296** |
| trot -> sit, 3-6 ms delay, falls | 1/24 | **0/24** |
| trot -> sit, 5-15 ms delay, falls | 6/12 | **4/12** |
| fall recovery (side, back, tipped, shove) | 20/20 | 20/20 |
| sit / lie from a moving walk or trot (our model, 48 tries) | 0 falls | 0 falls |
| our exam, mean tracking | 0.812 | 0.803 |

The A/B run with 4x faster motors (bolt-lite3-dog-4b) fell 2/3 on the routine and 6/12 at 5-15 ms: faster-motor
training does not fix delay on DEEP's model; practising with delay does. Simulation only. Dog clips CC BY-NC.
