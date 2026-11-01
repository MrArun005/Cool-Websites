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

## The imitation route: a real dog's jump, retargeted (1 Nov)

A scan of all 51 MANN files for moments with all four paws off the floor found six floor-to-floor running jumps
(0.28-0.52 s of flight, 2.4-3.0 m/s, the dog's body rising 0.34-0.50 m, landing still running at 1.2-2.3 m/s; the
long "flights" in D1_ex03 are the dog standing on a platform). `retarget_body.py --only jump` writes
dogmotion/clips/go2_jump.npz from D1_ex04_KAN02_004 7.9-9.4 s (the other Go2 clips untouched, md5 checked): 1.52 s,
approach -> crouch -> take-off -> flight (Go2 body 0.27 -> 0.54 m) -> front-paws-first landing -> run-out
(go2_jump_clip_frames.png). Kinematic check: paw target miss 30.6 mm mean (trot 32.9), joint-limit clamping up to
1.02 rad (canter 0.38: the stretched flight legs), no floor or self penetration. Next: train it (a 10th clip slot in
the imitation brain, warm-started from 695M by widening its clip one-hot; the jump hands over to trot when it ends).

## Jump as the imitation brain's 10th clip (1 Nov, c2c9aae) -> Kaggle bolt-go2-jump-imitate

Feasibility first (inverse dynamics on the reference, Go2 model): the full dog jump needs up to 1.87x the thigh
torque, 23 of 76 frames over a limit (canter: 1.34x, 2 of 24 frames -- and canter trains). So the brain has to find a
smaller jump than the dog's; the residual policy can only use the torque the motors have.
Plumbing: ALL_CLIPS + "jump" (76 inputs; MOTION_CLIPS = the 9 every body has, Bolt untouched), jump hold 0.2 s
(it ends running), widened warm start (a 9-slot brain gets zero-weight slots and acts identically: 3.9e-7),
`--clips`, `--clip-weight`; the controller hands a finished jump to trot. Parity CPU env vs MJX with the jump clip:
inputs/rewards within 1.1e-6 on 1102 states, 0 end-of-try disagreements. Local smoke of the kernel trained.
Run: from 695M, blind2's recipe + `--clips (9 motion) jump --clip-weight canter=3 jump=4`, 3.5 h T4, started 11:50
IST, ends ~15:35; -p scratchpad/kout_go2jumpim. Gate: jump-clip tracking >= 0.5 with falls on jump tries <= 20%,
the 9 motion clips no worse than 695M (exam), and the rehearsal still 0/10; then a bar test (a hurdle on the jump
clip's path) before any "jumps work" claim.
Exam (scratchpad go2_exam.py): MEAN TRACKING stays on the 9 motion clips; a 10-slot brain also gets the jump exam
apart (imitate_env.exam on ["jump"], 8 tries, from the clip's start through its 0.2 s hold). Before the run:
695M widened to 10 slots (never trained on the jump) **track_jump 0.095, fall_jump 1.00 (8/8)**. The 76-input exam
draws its 0.01 observation noise over 76 numbers, so its random stream differs from the 75-input exam: 695M scores
0.819 under it (0.820 recorded); with the env forced back to 75 inputs it reproduces the recorded exam exactly (every
number, difference 0). New brains are compared under the 76-input exam.

## Kaggle bolt-go2-jump-imitate (1 Nov): the jump never learned -- every try ended at take-off (a rule, fixed)

From 695M widened, 103M steps, 9 motion clips + jump (weight 4). Exam: **track_jump 0.097, fall_jump 1.00** (the
untrained baseline was 0.095 / 1.00). The other skills held or improved a little: 9-clip mean 0.819 -> 0.822, steer
0.678 -> 0.705, canter 0.52 -> 0.535, falls 0; rehearsal 0/10; recovery 40/40; knee speed unchanged (1482 steps
over 1.2x). Cause (checked try by try): every jump try ended at 0.5 s, upright (tilt 0.39), because the reference body
rises to 0.36 m while Bolt is at 0.24: |height error| 0.12 m > the Go2's "lost" limit 0.105 m. The brain never met a
take-off or a flight, so it could not learn one. Fix: CLIP_LOST {"jump": 0.35} in imitate_env and mjx_imitate (falls
still end the try by tilt and body-on-floor). With it the same brain's tries run to 0.92 s (the landing) and end on a
real tilt error. Next: bolt-go2-jump-imitate2, from this brain, same recipe, with the fix (after the CPU/MJX parity
check passes).
