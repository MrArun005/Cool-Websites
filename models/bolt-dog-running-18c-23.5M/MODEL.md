# Bolt dog running brain, run 18c at 23.5M steps

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

Imitation brain (8 clip slots: walk, trot, canter, turn_left, turn_right and others) trained to copy the dog's running gaits.

- Page exam (small sample): canter copies 70%, no falls.
- Fair 30-try canter exam (randomized physics, seeds 5000-5029, `research/exam_canter.py`): tracking 61% (±3%), 2/30 falls, 59% of the dog's speed. Same exam on 18b-17.5M: 42%, 8/30 falls.
- Load with `ImitatePolicy` (it detects the 8-slot layout).
