# Bolt dog switching brain, run 18e best (36M steps; run finished at 45M)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

All 9 clips (walk, trot, pace, canter, turn left/right, sit, lie, get up), trained with clip switches mid-try (switch rate 0.3), 9-slot layout. MasterMind's teacher for the dog motions.

Exam at 36M (tracking): walk 86%, trot 83%, pace 90%, canter 41%, turns 85%, sit 93%, lie 91%, get up 89%. See best.json.
Load with `ImitatePolicy`.
