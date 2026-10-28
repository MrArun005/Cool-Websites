# Bolt dog motion brain, run 18b best (finished at 30M steps)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

The run the owner calls "perfect act". 8-slot layout (walk, trot, pace, canter, turn left/right, sit, lie).

Exam (best.json, tracking): walk 92%, trot 90%, pace 93%, turn left 89%, turn right 90%, sit 95%, lie 93%, canter 54% (no falls).
The earlier saved 18b brain is models/bolt-dog-imitate-18b-17.5M. Load with `ImitatePolicy`.
