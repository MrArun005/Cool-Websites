# Bolt dog running brain, run 18c at 23.5M steps

Imitation brain (8 clip slots: walk, trot, canter, turn_left, turn_right and others) trained to copy the dog's running gaits.

- Page exam (small sample): canter copies 70%, no falls.
- Fair 30-try canter exam (randomized physics, seeds 5000-5029, `research/exam_canter.py`): tracking 61% (±3%), 2/30 falls, 59% of the dog's speed. Same exam on 18b-17.5M: 42%, 8/30 falls.
- Load with `ImitatePolicy` (it detects the 8-slot layout).
