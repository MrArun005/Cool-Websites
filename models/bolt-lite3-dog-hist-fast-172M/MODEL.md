# bolt-lite3-dog-hist-fast-172M (11 Nov; a starting point for training, not a robot brain)

Kaggle CPU lite3-dog-26 (as bolt-lite3-dog-hist-penalty-164M but lr 3e-5 and over-speed penalty 1.0, 23M steps). The
best steered canter so far on fresh seeds (0.457 with 1 fall of 24; 149M: 0.392 with 3), but live in DEEP's simulator
its joints run over 1.2x rated speed on 13-21 steps a run (robot brain: 4-7), and it fell in 1 of 24 trots with 5% of
steps missed. Kept to train on with a stronger speed penalty. Simulation only. CC BY-NC 4.0 motion data.
