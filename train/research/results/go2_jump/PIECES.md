# Go2 hurdle jumps (simulation)

## Pilot 1 (Kaggle bolt-go2-jump-pilot, 31 Oct - 1 Nov): 0 bars cleared -- the still-start line from scratch ended

train_jump.py --body go2 from scratch, still starts 0.39-0.56 m from the bar (Bolt's sizes x 0.86), 0-40 ms delay,
10.6 h, 18.4M steps. Exam every 1M steps (bars 10 / 14 / 17 / 21 cm from 0.47 m): **0.0 at every height, every
checkpoint** (strict too); the practice-bar coach fell to 7 cm and stayed there.

What it learned (local rollout of the last checkpoint, 3 tries at 7 and 10 cm): it leaps straight up (body 0.19 ->
0.45-0.48 m, ~18 cm of jump) but barely moves forward (0.05-0.30 m in 2 s; the bar is 0.39-0.56 m away), so it never
reaches the bar. Bolt's first jump brains learned from the same still starts; the Go2 did not.

Next: pilot 2 (bolt-go2-jump-pilot2, started 04:15 IST 1 Nov, 3.5 h): jump-trot's recipe Go2-sized -- 85% of tries
start running at 0.8-1.8 m/s (70% mid-stride), bar 0.6-1.5 m away, exam at 1.2 m/s from 1.0 m. Gate: any bar > 0.
