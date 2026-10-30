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

Pilot continued with root velocity weight 0.6: 37.0M 0.709 / 73%; 37.5M 0.706 / 74%; 38.0M 0.710 / 77%; 38.5M 0.701 /
77% (trot 0.872-0.881, 0 falls everywhere). Verdict: speed +9 points, tracking flat at ~0.71 -> pass line missed, no
12 h run. 38.0M saved as models/bolt-dog-canter-speed-38M (candidate, not used until it passes canter -> trot).

## 30 Oct: faster motors do not lift the canter (canter-fast pilot, Kaggle bolt-canter-fast-p1, 2 h)

canter-speed 38M fine-tuned on a Go2-class drive (30 rad/s rated, out by 35, 700 W; an assumed spec), trot + canter:
canter 0.714 / 79% speed -> 0.725 / 84% (42M), 0.722 / 85% (42.6M); trot 0.876 -> 0.883; 0 falls. The A1-drive
steering pilot over the same 2 h reached 0.728 / 85%, so the drive speed is NOT what holds the canter at ~0.72.
Reward terms of 38M (fixed physics, 4 tries x 4 s): trot pose 0.91, paw 0.94, root 0.99, body velocity 0.81, joint
velocity 0.24; canter pose 0.85, paw 0.73, root 0.89, body velocity 0.22, joint velocity 0.05. The gap is the body's
speed and spin vs the dog's (0.22) and paw placement (0.73), i.e. Bolt's body (forward rear knees, 9 kg torso) moving
like a 3.4 m/s dog. Even near-perfect imitation scores ~0.92 on this measure (walk's best is 0.908), so ">= 0.90"
canter means near-perfect copying of a dog's canter: realistic only on a body built more like the dog (the Go2 port).
