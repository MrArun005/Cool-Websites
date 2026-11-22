# bolt-lite3-dog-terrain6-1500M (22 Nov; a CANDIDATE, not yet the robot brain)

DEEP Robotics Lite3 motion brain, simulation only, numpy params.pkl (3 steps of input history, no velocity input).
bolt-lite3-dog-terrain6-1481M (job lite3-dog-45: gpu-1461M + ~20M PPO steps on the random rough field up to 6 cm) plus
another 19M steps the same way (Kaggle CPU bolt-lite3-dog-45-p2, 10.6 h). blind_vel and steer_max 0.8 set in the pkl
after `mjx_imitate.py warm`.

Local tests (DEEP's Lite3 model, lockstep_vendor --dt 0.001 --estimate; terrain6-1481M in brackets): two 3 cm bars,
trot, 5% missed steps, 200 tries: 2 (13); 4 cm bars, 200 tries: 1 (0); flat 50 s no-canter plan, 10% missed, 100 tries:
7 (7). Kaggle sweeps against gpu-1461M on the same seeds decide whether it replaces it (PLAN.md 22 Nov).
