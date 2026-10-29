# Canter is capped by the motors' speed, not by training (29 Oct, simulation)

canter-switching 18g 35.5M, 4 s from the clip start, fixed physics (ImitateEnv randomize=False):

| | trot | canter |
|---|---|---|
| peak joint speed | 15.3 rad/s | 26.0 rad/s |
| joint speed, 90th percentile | 12.6 rad/s | 23.5 rad/s |
| torque at >= 95% of 33.5 N·m | 0% | 0% |
| body speed | 1.68 m/s | 1.91 m/s |
| dog's clip speed | 1.91 m/s | 3.37 m/s |

The A1-class body (HANDOFF 0c) is rated 21 rad/s and its drive fades to 0 at 25 rad/s; the canter already runs the
legs into that fade, so the legs cannot swing faster. Canter pilot (runs/pilot_canter, root velocity weight 0.3,
trot+canter, 12-try exam seed 31000): start 0.693 at 68% of the dog's speed -> 36.0M 0.708 / 71% -> 36.5M 0.707 / 71%.
Options: a Go2-class body (~30 rad/s), longer-stride canter, or scoring at a reachable speed (owner's call).
