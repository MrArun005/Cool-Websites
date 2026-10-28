# Bolt canter-switching brain, run 18g at 34M steps

Run 18f's 30M canter brain, trained on with mid-try gait switches (walk, trot, canter, turns; switch rate 0.3/s) at learning rate 1e-4. 8-slot layout. Teacher candidate for canter->trot.

Gait-switch test (mastermind/test_master.py --teachers v3 --imitate 18g, sequence walk-trot-canter-trot-walk, 30 tries per seed set):
- seed 1000: walk->trot 100%, trot->canter 100%, canter->trot 80%, trot->walk 96%
- seed 5000: walk->trot 100%, trot->canter 100%, canter->trot 70%, trot->walk 95%
- canter->trot pooled 75% (45/60) vs 18f-31M 59% and MasterMind v3's 68%.
Fair canter exam (30 tries): seeds 5000 68.4% (1 fall), seeds 7000 70.6% (0 falls).
