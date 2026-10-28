# Bolt canter-switching brain, run 18g at 34M steps

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

Run 18f's 30M canter brain, trained on with mid-try gait switches (walk, trot, canter, turns; switch rate 0.3/s) at learning rate 1e-4. 8-slot layout. Teacher candidate for canter->trot.

Gait-switch test (mastermind/test_master.py --teachers v3 --imitate 18g, sequence walk-trot-canter-trot-walk, 30 tries per seed set):
- seed 1000: walk->trot 100%, trot->canter 100%, canter->trot 80%, trot->walk 96%
- seed 5000: walk->trot 100%, trot->canter 100%, canter->trot 70%, trot->walk 95%
- canter->trot pooled 75% (45/60) vs 18f-31M 59% and MasterMind v3's 68%.
Fair canter exam (30 tries): seeds 5000 68.4% (1 fall), seeds 7000 70.6% (0 falls).
