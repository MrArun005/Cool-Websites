# Bolt dog running brain, run 18f at 31M steps (best running brain)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

Fine-tune of run 18c's best (29.5M) at learning rate 1e-4, target_kl 0.015. 8-slot layout; real A1 motor limits.

- Fair canter exam, 30 tries, randomized physics, seeds 5000-5029: tracking 71.8% (±0.6%), 0/30 falls, 72% of the dog's speed.
- Second fair exam, seeds 7000-7029: tracking 72.2% (±0.6%), 0/30 falls.
- Run exam (best.json): walk 92%, trot 91%, turns 91%.
- Earlier: 18f-30M 70.8% / 70.4% (0/60 falls); 18c-29.5M 68% (1/30).
- Load with `ImitatePolicy` (8-slot layout).
