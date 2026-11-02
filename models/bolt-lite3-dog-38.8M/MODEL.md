# bolt-lite3-dog-38.8M (simulation): DEEP Robotics Lite3 dog-motion brain, second CPU session

From bolt-lite3-dog-20.9M, +18M steps on a free Kaggle CPU session (bolt-lite3-dog-2, 637 min, same recipe: 9 clips,
switches 0.3/s, brake, 0-40 ms delay). params.pkl = the numpy export (mjx_imitate.py warm) for the GPU trainer and the
robot program.

| exam (4 tries per clip) | 20.9M | **38.8M** |
|---|---|---|
| mean tracking | 0.771 | **0.813** |
| walk / trot / pace | 0.77 / 0.69 / 0.79 | **0.82 / 0.75 / 0.82** |
| canter | 0.36, 1 fall in 4 | **0.49, no falls** |
| turns L / R | 0.78 / 0.79 | 0.81 / 0.82 |
| sit / lie / getup | 0.94 / 0.91 / 0.91 | 0.95 / 0.94 / 0.94 |
| falls on the 9 clips | 1 in 36 | **0 in 36** |
| with 10 / 20 / 40 ms delay | - | no falls (tracking 0.85 / 0.84 / 0.77) |
| sit / lie from a moving start | weak | still weak: trot -> sit 3 falls in 4, walk -> lie 3/4, trot -> lie 4/4 |

Win line (mean tracking >= 0.80 and canter >= 0.45): met. Next: the GPU trainer with moving starts, steering and joint
damping 0.1-2.0, as the Go2. Simulation only. Dog motion clips are CC BY-NC (non-commercial).
