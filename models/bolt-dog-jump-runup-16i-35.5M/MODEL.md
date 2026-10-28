# Bolt jump brain with a run-up, run 16i at 35.5M steps (best so far)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: train/jobs/run16i-jump-runup.json (27 Oct 17:15, after fd5199b), no BOLT_MOTOR_TORQUE override; from the 16f/16g line (both 33.5 N·m; 16f itself started from a brain trained at 17 N·m). The fair 30-try numbers here reproduce exactly on 28 Oct (`fair_exam.py jump ../models/bolt-dog-jump-runup-16i-35.5M --speeds 2.0 --dist 1.5 --heights 0.50 0.60 --seed 7000 --max-steps 220 --no-save`: 50 cm 77%, 60 cm 3%).

Real A1 motor limits, real agility hurdle. Arrives at 2 m/s from 1.5 m away.

- Training exam (10 tries per height): 40 cm 100%, 50 cm 100%, 60 cm 20% clean.
- Fair 30-try exam on fresh seeds (7000+, agility rules, `exam_30.json`): 40 cm 100%, 50 cm 77%, 60 cm 3% clean.
- Strict old test (randomized physics, askew approach, no touch at all): 40 cm 73%, 50 cm 37%, 60 cm 27%.
- Beats 34.5M (fair 50 cm 60%, strict 40 cm 20%). Load with `jump_env.JumpPolicy(folder, "policy.zip", "vecnormalize.pkl")`.
