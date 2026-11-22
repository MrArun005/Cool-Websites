# bolt-go2-terrain6-1458M (22 Nov; a CANDIDATE, not the robot brain)

Unitree Go2 motion brain, simulation only, numpy params.pkl (3 steps of input history, no velocity input). The robot
brain bolt-go2-torque-steer as SB3 (gpu_to_sb3.py, exact) plus ~16M PPO steps on Kaggle CPU (job go2-dog-50, 10.6 h)
with walk and trot tries started on the random rough field up to 6 cm (every robot geom touches it), 5% missed steps,
else the torque-steer recipe as far as the CPU trainer has it. blind_vel and steer_max 0.8 set after `mjx_imitate.py warm`.

Local tests on Unitree's go2.xml (lockstep_vendor --estimate, 100 tries, seeds 50000+; torque-steer in brackets): two
bars across the path, "trot:8,walk:4": 3 cm 11 (4), 5 cm 14 (17); "walk:12,trot:2": 3 cm 5 (28), 5 cm 46 (67). Flat 50 s
plan (seeds 60000+): 0-19 ms 0/100, 10% missed 0/100 (0/100), left 0.8 at 10-19 ms 0/100.
Kaggle sweeps against torque-steer on the same seeds decide whether it replaces it (PLAN.md 22-23 Nov).
