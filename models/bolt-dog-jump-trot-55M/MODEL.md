# Bolt jump brain trained from a trot, Kaggle run jump-trot at 55.04M steps (best jump brain, 28 Oct)

> **Motor torque:** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body), motor brake off in training.
> With `--motor-brake` (25 rad/s, 3.5 N·m·s/rad) the scores below move by 0–7 points either way (table).

All numbers are in simulation (MuJoCo), agility rules: the bar stays up and Bolt lands on its feet.

Started from `bolt-dog-jump-runup-16i-35.5M` and trained 10.7 h on a Kaggle CPU (`train/runs/kaggle-jump-trot/summary.json`
has the full settings: run-ups 0.8–2.0 m at 1.2–2.4 m/s, 70% of tries begun mid-stride, bars 45–75 cm).
The run's own pick (48M) was chosen on still-legs exams; this checkpoint was re-picked on the mid-stride hand-over
exam, the way MasterMind hands it the jump (trot at 1.7 m/s, bar 1.1 m ahead):

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/reselect_jump.py runs/kaggle-jump-trot/runs/jumptrot --every-k 2 --workers 4 \
        --compare ../models/bolt-dog-jump-runup-16i-35.5M runs/kaggle-jump-trot/runs/jumptrot/best

Seed 7000 (the picking seed, so optimistic): mid-stride 40 cm 97%, 50 cm 70% (16i: 50%, 7%; the run's own pick: 83%, 40%).

## Fair 30-try exam on fresh seeds 9000 and 11000 (mean of the two), bar 1.5 m away, 4.4 s tries

    MUJOCO_GL=osmesa python fair_exam.py jump ../models/bolt-dog-jump-trot-55M --seed 9000 --episodes 30 \
        --speeds 1.5 2.0 --heights 0.40 0.50 0.60 --gait --max-steps 220 --no-save [--motor-brake]
    (and --seed 11000; same command on ../models/bolt-dog-jump-runup-16i-35.5M)

| Start | Bar | this brain | with brake | 16i-35.5M |
|---|---|---|---|---|
| 1.5 m/s, still legs | 40 / 50 / 60 cm | 100 / 100 / 0% | 100 / 97 / 0% | 40 / 0 / 0% |
| 1.5 m/s, mid-stride | 40 / 50 / 60 cm | 92 / 65 / 33% | 90 / 68 / 35% | 17 / 5 / 0% |
| 2.0 m/s, still legs | 40 / 50 / 60 cm | 100 / 100 / 13% | 100 / 100 / 17% | 100 / 83 / 7% |
| 2.0 m/s, mid-stride | 40 / 50 / 60 cm | 68 / 52 / 28% | 72 / 55 / 30% | 47 / 38 / 15% |

Per-seed results are in `exams/` (the 2.0 m/s mid-stride row differs a lot by seed: 47/37% on 9000, 90/67% on 11000).
The neighbouring checkpoint (55.0M) scores within a few points (`exams/ck55000000_*.json`), so the pick is not a lucky outlier.

Weak spot: mid-stride at 2.0 m/s (canter speed). 60 cm is not solved.

Load with `jump_env.JumpPolicy(folder, "policy.zip", "vecnormalize.pkl")`.

## In Bolt's Day Out (browser physics, every task a jump, 4 runs x 240 s, seeds 1-4)

    cd train/gaitlab && node doglife_jumptest.mjs <mujoco.js> <data folder> <seed> 240

| | jumps | clean | bar down | falls (of them: in the trot/canter hand-over after landing) |
|---|---|---|---|---|
| this brain | 29 | 21 | 7 | 21 (9) |
| 16i-35.5M | 11 | 5 | 4 | 22 (falls mostly running into the jump: 11) |

Per-run lines: `exams/dayout_jumptest.txt`. Next weak spot: the canter->trot bridge after landing.
