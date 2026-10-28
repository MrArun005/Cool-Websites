# Bolt dog motion + getting up like a dog, run 18d best (24.5M steps)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

All 9 clips, 9-slot layout; getting up front-first like a dog (the lie clip reversed). Best all-round dog motion brain.

Exam (best.json, tracking): walk 91%, trot 90%, pace 92%, turn left 90%, turn right 89%, sit 95%, lie 94%, get up 92%, canter 51% (no falls).
Run stopped at ~35.5M: no new best for 11M steps. Load with `ImitatePolicy`.
