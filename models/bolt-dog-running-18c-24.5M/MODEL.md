# Bolt dog running brain, run 18c at 24.5M steps (best running brain so far)

- Fair 30-try canter exam (randomized physics, seeds 5000-5029, `research/exam_canter.py`): tracking 65% (±2%), 1/30 falls, 65% of the dog's speed. 23.5M: 61%, 2/30. 18b-17.5M: 42%, 8/30.
- Plan-then-distill (two DAgger rounds, `runs/research18c/chain.sh`) gave no gain on this brain (65%, 1/30): the planner kept the brain's own action 97% of the time, so there was little to teach.
- Load with `ImitatePolicy` (8-slot layout).
