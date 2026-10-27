# Bolt dog running brain, run 18f at 30M steps — canter target reached

Fine-tune of run 18c's best (29.5M) at a lower learning rate (1e-4, target_kl 0.015). 8-slot layout; real A1 motor limits.

- Fair canter exam, 30 tries, randomized physics, seeds 5000-5029: **tracking 70.8% (±0.5%), 0/30 falls**, 73% of the dog's speed.
- Second fair exam, 30 fresh tries, seeds 7000-7029: **tracking 70.4% (±0.8%), 0/30 falls**, 72% of the dog's speed.
- Run exam (best.json): walk 91%, trot 90%, turns 89%, no falls.
- Before: 18c-29.5M 68% with 1/30 falls; 18b-17.5M 42% with 8/30 falls.
- Load with `ImitatePolicy` (8-slot layout).
