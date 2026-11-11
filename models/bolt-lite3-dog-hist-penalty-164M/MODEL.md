# bolt-lite3-dog-hist-penalty-164M (11 Nov; the best Lite3 candidate, not yet the robot brain)

Lite3 motion brain without a velocity input, with 3 frames of input history. Kaggle CPU lite3-dog-25: PPO
(`train_imitate.py --history 3 --blind-vel`, over-speed penalty weight 2.0, lr 1e-5) for 15M steps from
bolt-lite3-dog-blind-hist-start. Simulation only.

| Test | blind-149M (robot brain) | this |
|---|---|---|
| steered canter, fresh seeds, 24 tries: score / falls | 0.270 / 8 | **0.408 / 6** |
| our exam: tracking / steering | 0.775 / 0.444 | **0.783 / 0.618** |
| plain trot in our exam, 64 fresh tries (+12): falls | **0** (+0) | 1 (+1) |
| DEEP's model, leg estimate: the 7 stress tests (trot, routine, sit, turns; delay, missed steps) | 0 falls, canter routine 1/12 | 0 falls, canter routine 0/12 |
| routine ending in a canter, 3-6 ms | 2/36 | **0/36** |
| moving sit / lie on DEEP's model, 5-15 ms | 0/48 (trot -> sit) | 0/120 |
| recovery (side, back, tipped, shove) | 48/48 | 48/48 |
| live in DEEP's simulator, steps over 1.2x rated per run | 4-7 | 5-9 (6 runs) |

Not the default yet: in our own randomized exam it fell in 2 plain trots of 76 where the robot brain fell in none.
Everything else is equal or better. Motion data: CC BY-NC 4.0 (non-commercial).
