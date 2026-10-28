# Bolt dog running brain, run 18c at 24.5M steps (best running brain so far)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

- Fair 30-try canter exam (randomized physics, seeds 5000-5029, `research/exam_canter.py`): tracking 65% (±2%), 1/30 falls, 65% of the dog's speed. 23.5M: 61%, 2/30. 18b-17.5M: 42%, 8/30.
- Plan-then-distill (two DAgger rounds, `runs/research18c/chain.sh`) gave no gain on this brain (65%, 1/30): the planner kept the brain's own action 97% of the time, so there was little to teach.
- Load with `ImitatePolicy` (8-slot layout).
