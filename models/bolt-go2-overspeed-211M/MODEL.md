# bolt-go2-overspeed-211M (the Go2 robot brain from 8 Nov)

Unitree Go2 motion brain, simulation only, numpy (params.pkl; the robot program needs no torch). Trained on a Kaggle GPU
with MJX + Brax PPO, in two steps from bolt-go2-damp-798M:
1. bolt-go2-privhist: privileged critic (the value network sees the simulator's true velocity, friction, gains, mass,
   damping, paw contacts; the brain does not) + 3 steps of input history, damping 0.1-2.0, blind velocity, 5% dropped steps.
2. bolt-go2-gpu-overspeed (5 h, 211M steps): the same plus a motor drive drawn x1.0-1.5 per try and the over-speed
   penalty (mjx_imitate --overspeed-weight 1.0: reward minus the summed joint speed above rated, as a fraction).

Why: live in Unitree's own simulator in real time, 798M and privhist tripped run_go2's speed stop (3 control steps in a
row over 1.2x rated) within 6-10 s. Lockstep rehearsals never apply that stop, which hid it.

| Test (Unitree's go2.xml, robot-only inputs) | 798M | privhist | **211M** |
|---|---|---|---|
| 21 s routine, 10 tries | 0/10 | 0/10 | 0/10 |
| with 0-19 ms delay / 5% missed steps | 0/10 / 0/10 | 0/10 / 0/10 | 0/10 / 0/10 |
| with 15-19 ms delay / 15% missed steps (12 tries) | 0/12 / 1/12 | 0/12 / 1/12 | 0/12 / 1/12 |
| steps over 1.2x rated (of 10500), worst | 20, 1.84x | 10, 1.65x | 11, **1.48x** |
| recovery (side, back, tipped, shove) | 40/40 | 20/20 | 20/20 |
| our exam: tracking / falls / steering | 0.817 / 1.9% / 0.708 | 0.833 / 0% / 0.750 | **0.840 / 0%** / 0.686 |
| **live in Unitree's simulator, full routine** | speed stop at 6-10 s | speed stop at 10 s | **3/3, no stop** |

The deploy test now runs the brain once per step and resets its input history per try (a brain with history keeps state).
