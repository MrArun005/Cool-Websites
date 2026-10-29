# bolt-dog-canter-speed-38M (simulation): faster canter, candidate only (not used yet)

Local pilot (29 Oct, runs/pilot_canter): from canter-switching 18g 35.5M, clips trot + canter, mid-try switches 0.3/s,
learning rate 1e-4; reward weight of body velocity 0.3 to 36.5M, then 0.6. Exam (imitate_env.exam, 12 tries per clip,
seed 31000, randomized physics; train/research/results/canter_limit/NOTE.md has the per-checkpoint table):

| | 18g 35.5M (start) | this (38.0M) |
|---|---|---|
| canter tracking | 0.693 | 0.710 |
| canter speed (share of the dog's 3.37 m/s) | 0.68 | 0.77 |
| trot tracking / speed | 0.885 / 0.89 | 0.876 / 0.92 |
| falls | 0 | 0 |

Missed the pilot's pass line (canter >= 0.72 at >= 0.75 speed): speed passed, tracking stayed at 0.70-0.71, which
fits the motor-speed cap (the canter runs the joints at 23.5-26 rad/s against the A1 drive's 25 rad/s fade). Not in any
teacher set: it must first pass the canter -> trot switching test that 18g was chosen for (93%).
