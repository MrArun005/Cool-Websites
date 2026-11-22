# bolt-lite3-dog-terrain6-1481M (22 Nov; a CANDIDATE, not the robot brain)

DEEP Robotics Lite3 motion brain, simulation only, numpy params.pkl (3 steps of input history, no velocity input).
From the robot brain bolt-lite3-dog-gpu-1461M (as SB3), ~20M more PPO steps on Kaggle CPU (job lite3-dog-45, 10.6 h)
with walk and trot tries started on the random rough field up to 6 cm (train_imitate --terrain-cm 6; every robot
geom touches it), 5% missed steps, else the gpu-1461M recipe. blind_vel and steer_max 0.8 set in the pkl after
`mjx_imitate.py warm`.

First local test (DEEP's Lite3 model, lockstep_vendor --dt 0.001 --estimate, 100 tries each, seeds 50000+ / 60000+;
gpu-1461M in brackets): two bars across the path, trot: 3 cm 0/100 (0), 4 cm 0/100 (12), 5 cm 17/100 (36), 6 cm 87/100 (72); walk 4 cm 8/100 (12); flat 50 s plan with 10% missed steps 1/100 (7).
Kaggle sweeps against gpu-1461M on the same seeds decide whether it replaces it (PLAN.md 22 Nov).
