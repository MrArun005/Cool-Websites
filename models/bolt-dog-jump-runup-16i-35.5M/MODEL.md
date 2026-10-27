# Bolt jump brain with a run-up, run 16i at 35.5M steps (best so far)

Real A1 motor limits, real agility hurdle. Arrives at 2 m/s from 1.5 m away.

- Training exam (10 tries per height): 40 cm 100%, 50 cm 100%, 60 cm 20% clean.
- Fair 30-try exam on fresh seeds (7000+, agility rules, `exam_30.json`): 40 cm 100%, 50 cm 77%, 60 cm 3% clean.
- Strict old test (randomized physics, askew approach, no touch at all): 40 cm 73%, 50 cm 37%, 60 cm 27%.
- Beats 34.5M (fair 50 cm 60%, strict 40 cm 20%). Load with `jump_env.JumpPolicy(folder, "policy.zip", "vecnormalize.pkl")`.
