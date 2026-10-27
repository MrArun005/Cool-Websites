# Bolt dog running brain, run 18f at 31M steps (best running brain)

Fine-tune of run 18c's best (29.5M) at learning rate 1e-4, target_kl 0.015. 8-slot layout; real A1 motor limits.

- Fair canter exam, 30 tries, randomized physics, seeds 5000-5029: tracking 71.8% (±0.6%), 0/30 falls, 72% of the dog's speed.
- Second fair exam, seeds 7000-7029: tracking 72.2% (±0.6%), 0/30 falls.
- Run exam (best.json): walk 92%, trot 91%, turns 91%.
- Earlier: 18f-30M 70.8% / 70.4% (0/60 falls); 18c-29.5M 68% (1/30).
- Load with `ImitatePolicy` (8-slot layout).
