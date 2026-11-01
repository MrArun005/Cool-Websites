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

## Pilot 2 (Kaggle bolt-go2-jump-pilot2, 1 Nov): running starts -- still 0 bars; it refuses the bar

jump-trot's recipe Go2-sized (85% running starts 0.8-1.8 m/s, 70% mid-stride, bar 0.6-1.5 m away), 3.5 h, 6.3M steps:
**0.0 at every bar height, every checkpoint**; the coach again sat at its floor (7 cm).

What it learned (local rollouts of the last checkpoint, exam setup 1.2 m/s from 1.0 m, 10 cm bar): it runs at the
bar, slows down and **stops ~0.3 m in front of it** (3 of 4 tries; mean speed 0.31 m/s; the 4th fell). A refusal:
standing in front of the bar costs 0.1 a step, a try at the bar risks -5 (knock / under) or a fall.

Likely cause: the coach's floor (10 cm x kz = 7 cm) is already a real jump for the Go2 (its trot lifts the paws
~5 cm), so the brain never meets a bar it can simply step over and never finds the paying side of the bar.
Fix: `--bar-floor` (new; default unchanged for every body) so the coach can start the Go2 at 2-3 cm and raise the bar
as clean crossings come. Pilot 3 = pilot 2 + --bar-floor 0.02 --bar-start 0.03.

## Pilot 3 (Kaggle bolt-go2-jump-pilot3, 1 Nov): coach down to 2 cm -- still 0 bars; the from-scratch line stops

Pilot 2 + `--bar-floor 0.02 --bar-start 0.03`, 3.5 h, 7.7M steps: **0.0 at every exam height, every checkpoint**;
the practice-bar coach sat at its 2 cm floor the whole run. Gate (any exam bar > 0 or coach > 7 cm): failed.

What it learned (local rollouts of the last checkpoint, 1.2 m/s from 1.0 m): it no longer refuses. It runs in, takes
off ~0.2 m before the bar and leaps high (body 0.36 m over a 2 cm bar, 0.32-0.34 m over 10 cm), but almost straight
up: the body ends only ~0.1 m past the bar's line, all four paws land on the line (paws past -0.18..+0.05 m; "cleared"
needs all four 6.9 cm past), forward speed is gone, and it tips over (upright 0.84 -> 0.19 within 0.5 s). The reward
pays height over the bar and paws past it, but nothing makes it keep its run going through the air and land running.

Decision (gate rule): the from-scratch Go2 jump line stops here (3 pilots, ~32M steps, 0 bars). Next approach, the
one every working Go2 skill came from: **imitate a real dog's jump**. The MANN mocap (dogmotion/SOURCES.md) has
jumping; retarget a running jump to the Go2 (dogmotion/retarget.py), add it as a clip, and train it with the motion
brain's imitation recipe (mjx_imitate.py, from 695M) so take-off, flight and a running landing are all in the
reference. Then the bar exam.
