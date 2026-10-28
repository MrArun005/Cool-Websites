# Bolt canter-switching brain, run 18g at 35.5M steps (best switching teacher)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: run 18 lineage. Run 18 started 27 Oct 08:25 UTC (train/jobs/run18-imitate.json), before commits df69a28 and fd5199b, so its earliest steps had the old 17 N·m cap and no power/speed limit; it was restarted from its checkpoint when it moved machines (e5919ea, 13:52), and run 18b (from run 18's 9M checkpoint, 15:08) and everything after it trained under 33.5 N·m with no BOLT_MOTOR_TORQUE override in any job file. Exams here were taken after fd5199b.

Run 18f's 30M canter brain trained on with mid-try gait switches (walk, trot, canter, turns; 0.3/s) at learning rate 1e-4. 8-slot layout.

Gait-switch test (mastermind/test_master.py --teachers v3 --imitate 18g --over '{"canter":"18g"}', walk-trot-canter-trot-walk, 30 tries per seed set):
- seed 1000: walk->trot 100%, trot->canter 93%, canter->trot 96% (27/28), trot->walk 100%
- seed 5000: walk->trot 100%, trot->canter 100%, canter->trot 90% (27/30), trot->walk 96%
- canter->trot pooled 93% (54/58). Was: 18g-34M 75%, 18f-31M 59%, MasterMind v3 68%.
Fair canter exam (30 tries): seeds 5000 69.4%, seeds 7000 69.6%, 0/60 falls (18f-31M stays the canter-target brain at 71.8/72.2%).
