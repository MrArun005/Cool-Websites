# bolt-lite3-dog-20.9M (simulation): first dog-motion brain for the DEEP Robotics Lite3

Kaggle CPU run bolt-lite3-dog-pilot (1-2 Nov, 637 min, 20.9M steps, from scratch): train_imitate.py --body lite3, all 9
clips, switches 0.3/s, motor brake, 0-40 ms delay (kaggle/make_train_kernel.py job lite3-dog-pilot). Body:
robot_bodies "lite3" (robots/deeprobotics_lite3, DEEP Robotics' MJCF, BSD-3-Clause).

Exam at 20.9M (4 tries per clip): mean tracking 0.771 (Go2's first CPU pilot: 0.767). Walk 0.77, trot 0.69, pace 0.79,
turns 0.78 / 0.79, sit 0.94, lie 0.91, getup 0.91, all without falls; canter 0.36 with 1 fall in 4. Overall fall rate
0.27 counts the moving sit/lie starts and the delay sweep, which this pilot has not learned yet. Win line (tracking
>= 0.70, no falls): tracking yes, falls no. Next: export to the GPU trainer (mjx_imitate.py warm) and train as the Go2.
Simulation only. Dog motion clips are CC BY-NC (non-commercial).
