# Bolt canter-switching brain, run 18g at 35.5M steps (best switching teacher)

Run 18f's 30M canter brain trained on with mid-try gait switches (walk, trot, canter, turns; 0.3/s) at learning rate 1e-4. 8-slot layout.

Gait-switch test (mastermind/test_master.py --teachers v3 --imitate 18g --over '{"canter":"18g"}', walk-trot-canter-trot-walk, 30 tries per seed set):
- seed 1000: walk->trot 100%, trot->canter 93%, canter->trot 96% (27/28), trot->walk 100%
- seed 5000: walk->trot 100%, trot->canter 100%, canter->trot 90% (27/30), trot->walk 96%
- canter->trot pooled 93% (54/58). Was: 18g-34M 75%, 18f-31M 59%, MasterMind v3 68%.
Fair canter exam (30 tries): seeds 5000 69.4%, seeds 7000 69.6%, 0/60 falls (18f-31M stays the canter-target brain at 71.8/72.2%).
