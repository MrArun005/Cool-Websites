# bolt-dog-canter-speed-38M (simulation): faster canter, the canter teacher from 29 Oct (teacher set v7c)

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
fits the motor-speed cap (the canter runs the joints at 23.5-26 rad/s against the A1 drive's 25 rad/s fade). 

## Switching check (passed): teacher set v7c = v7s with this brain for the canter clip

test_master.py --brain teacher, dev seeds 20000 and 20015 (2 x 15 tries per switch), simulation; logs in
train/research/results/teacher_chain_v7c/:

| | v7s (18g canter) | v7c (this canter) |
|---|---|---|
| all switches | 97.0% (544/561) | 96.8% (542/560) |
| trot -> canter | 29/30 | 30/30 |
| canter -> trot | 27/29 | 26/30 |

Equal within chance, with a canter 9 points closer to the dog's speed. MasterMind v9 learned its canter from gpu3
(0.558 tracking), so the next MasterMind run uses teachers v7c.

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain teacher --teachers v7c --n 15 --seed 20000
