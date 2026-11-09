# bolt-go2-torque-steer (the Go2 robot brain from 9 Nov)

Unitree Go2 motion brain, simulation only, numpy params.pkl (3 steps of input history; no torch on the robot). Kaggle
GPU (MJX + Brax PPO), 5 h, from bolt-go2-overspeed-211M: privileged critic + history 3, damping 0.1-2.0, blind
velocity, 5% dropped steps, the over-speed penalty (weight 1.0) with a drive x1.0-1.5, AND per-try torque limits x0.8-1.4
per joint type (mjx_imitate --motor-torque-range: a brain trained at one torque limit learns to lean on it, PLAN.md
8 Nov), steering weight 0.8 (was 0.6).

| Test (Unitree's go2.xml, robot-only inputs) | 798M | overspeed-211M | **this** |
|---|---|---|---|
| live in Unitree's simulator, full routine | speed stop at 6-10 s | 3/3, no stop | **3/3, no stop** |
| live steps over 1.2x rated per run | n/a | 2-5 | 3-4 |
| 21 s routine / 0-19 ms / 5% missed steps | 0/10 each | 0/10 each | 0/10 each |
| 15-19 ms / 15% missed steps (12 tries) | 0/12 / 1/12 | 0/12 / 1/12 | 0/12 / 1/12 |
| steps over 1.2x rated (of 10500), worst | 20, 1.84x | 11, 1.48x | 12, **1.38x** |
| recovery | 40/40 | 20/20 | 20/20 |
| our exam: tracking / falls / steering | 0.817 / 1.9% / 0.708 | 0.840 / 0% / 0.686 | **0.845 / 0% / 0.702** |
