# MasterMind: one brain for every behaviour (DAgger distillation of Bolt's experts)

**Result.** One 84-input MLP (1024-512-256, ELU) drives Bolt through every behaviour: walk, trot, pace, canter,
turns, sit, lie, stand, jumping a real hurdle, and getting up after a shove. It switches between them without
any hand-over. On the same switch tests as `test_skills.py` (same sequences, windows and seeds, N=30 per sequence
on seed sets 1000 and 5000, so N=60), it succeeds at **72% of switches pooled (589/818)**. The skill manager gets
**64% (501/788)** with the same teachers (dog-motion run 18e) and **61% (241/394, N=30)** with today's 18b brain.
The gain comes from jumps (40 cm clean **55% vs 22%**) and walk->sit (the sit pose reached **52% vs 33%**). Every
other switch is level, within the binomial error. Nothing got clearly worse. Landing->trot is still only about 50%.

## Method

* **Unified env** (`env.py`, `Master`). This is the real-hurdle world of `skills.Bolt` (JumpEnv(real=True)), with
  every expert's env attached to it. Every behaviour uses one observation (84 floats):
  * proprioception (46), as the experts see it, with the student's own previous action
  * a command one-hot (10): walk trot pace canter turn_left turn_right sit lie stand jump
  * the reference-motion block (21): the imitation brain's clip phase, the next reference joints, height, velocity
    and yaw rate, the time since the clip ended, and an active flag
  * a hurdle block (6): present, distance ahead, bar height, sin of the heading error, offset from the line, and
    the time since the bar came within 1.1 m
  * the time since the last command

  During a jump the approach gait's clip keeps playing in the observation. The student itself decides when to
  take off, from the hurdle distance.

  **Action:** absolute joint targets in radians, `ctrl = clip(DOG + full_range*u, lo, hi)`. This covers every
  expert's motor mapping: the imitation residual on the clip, the jump/self-right full range, and the posture
  brain's stance-centred mapping.
* **Teachers.** For the active command, the teacher is the expert that `skills.py` would run, via its own
  `_obs()` of the live shared state:
  * gaits/sit/lie: run 18e on the clip, joined at the closest pose
  * jump: the gait until 1.1 m before the bar, then the jump brain (16g) until landed, then a 0.6 s canter
    bridge, then back to the gait
  * stand: the posture brain
  * after a fall (skills.py's detector): the recovery chain, which is self-right 17e, then the posture brain's
    lie, then its stand

  The **student's** action is the one executed. It is forced into the physics, and each expert's bookkeeping
  (its "previous action") gets the executed action in that expert's own units. That lets every visited state get
  a label. At a switch the new teacher labels from the first step.
  * Sanity check: driving the env with the teachers reproduces the skill manager (`results/teacher_in_env_18e.json`
    vs `results/manager_18e_n10.json`; same rates within +-1/10).
* **Handler** (`dagger.py`). It gives commands at random times along the dog's transitions:
  * gait <-> gait
  * gait -> sit -> lie -> stand -> gait
  * gait -> jump over a 15-45 cm bar placed 1.5-2 m ahead -> back to the gait
  * shoves that topple Bolt -> recovery

  Physics randomization and sensor noise are as in the experts' training.
* **DAgger** (Ross 2011), supervised MSE on joint targets.
  * Round 0: teacher rollouts with 0-0.08 rad motor noise (DART), 300k states.
  * Rounds 1+: student rollouts (teacher executed with probability beta), 150k states per round, aggregated, then
    retrain.
  * Loss weights: x4 on the jump brain's second, x2 on the first second after every command.
* **v1 -> v2.**
  * v1 (7 rounds, beta 1->0, 512-512-256): good until round 3, then the student lost recovery (0/10 stood up at
    round 6). The recovery chain moved on after *timeouts*, so states on its back were labelled "self-right" in
    one try and "tuck/stand" in another. With beta=0 these contradictory states flooded the data.
  * v2: the chain is state-driven in training (`patient`: tuck only once calm and belly-down), beta >= 0.2, a
    wider net, and it starts from v1's first 750k states. Then 5 more rounds: 1.5M states, about 1 h of 2 CPU
    cores in all.
* The exported student is **v2 round 5** (the last round, not picked on the test): `export/brain.json`.
  * Same format as `export_live.brain`: normalizer with var=std^2, eps 0, clip 10, ELU layers. Parity with torch
    is 4e-6.
  * `export/model.json` holds the input layout, the command list and the action mapping.
  * The reference clips the inputs need are the imitation brain's `clips.json` (export_imitate.py).

## Results: switch success (rate +- binomial s.e.)

The skill-manager rows come from `baseline.py`, which runs `test_skills.SEQUENCES` on `skills.Bolt`. The student
rows come from `test_master.py`, which runs the same sequences through `drive.Driver`. The "still:" rows are 100%
for all three and are left out.

| switch | manager 18b (N=30) | manager 18e (N=60) | **MasterMind student (N=60)** |
|---|---|---|---|
| walk->trot | 30/30 100% | 60/60 100% | 59/60 98% +- 2 |
| trot->canter | 27/30 90% +- 5 | 45/60 75% +- 6 | 46/59 78% +- 5 |
| canter->trot | 22/27 81% +- 7 | 39/45 87% +- 5 | 39/46 85% +- 5 |
| trot->walk | 22/22 100% | 39/39 100% | 39/39 100% |
| walk->sit, pose reached | 6/30 20% +- 7 | 20/60 33% +- 6 | **31/60 52% +- 6** |
| sit->lie (pose reached) | 6/30 20% +- 7 | 20/60 33% +- 6 | **31/60 52% +- 6** |
| lie->stand | 6/6 100% | 20/20 100% | 31/31 100% |
| trot->jump 20 cm, clean | 16/30 53% +- 9 | 33/60 55% +- 6 | **42/60 70% +- 6** |
| jump 20 cm->(canter)->trot | 11/22 50% +- 11 | 20/48 42% +- 7 | 26/52 50% +- 7 |
| trot->jump 30 cm, clean | 16/30 53% +- 9 | 30/60 50% +- 6 | **45/60 75% +- 6** |
| jump 30 cm->(canter)->trot | 11/25 44% +- 10 | 29/50 58% +- 7 | 30/58 52% +- 7 |
| trot->jump 40 cm, clean | 5/30 17% +- 7 | 13/60 22% +- 5 | **33/60 55% +- 6** |
| jump 40 cm->(canter)->trot | 11/26 42% +- 10 | 31/55 56% +- 7 | 31/60 52% +- 6 |
| fall->recover (stood up) | 26/30 87% +- 6 | 51/60 85% +- 5 | 53/60 88% +- 4 |
| recover->walk | 26/26 100% | 51/51 100% | 53/53 100% |
| **pooled** | **241/394 61%** | **501/788 64%** | **589/818 72%** |

* Per seed set the picture is the same. With seed 1000 (N=30) the student's clean jumps are 20/21/15 of 30 at
  20/30/40 cm, against the manager's 15/15/5. With seed 5000 they are 22/24/18, against 18/15/8. Raw files are
  in `results/`.
* `skills_report.json` (N=10) said trot->jump 8/7/3 of 10. At N=30 the same 18b manager gets 16/16/5 of 30. The
  N=10 numbers were optimistic; N=10 is too few to rank anything here.
* Round 3 of v2 scores the same as round 5 (pooled 333 vs 327 of about 432 on seed 1000).

**Per-clip exam** (imitate_env.exam's measure through the unified env, 4 starts per clip, tracking 0..1). The
student matches its teacher:

| clip | walk | trot | pace | canter | turn L | turn R | sit | lie |
|---|---|---|---|---|---|---|---|---|
| teacher 18e | .851 | .835 | .856 | .300 (falls 1/4) | .825 | .821 | .819 | .845 |
| student v2 r5 | .865 | .843 | .875 | .254 (falls 2/4) | .826 | .821 | .820 | .847 |

Speed kept matches the teacher too: walk .85, trot .89, pace .86, turns .24/.37.

## Why it switches better (and where not)

* **Jumps.** The manager hands the jump brain a body at a fixed 1.1 m, mid-stride, with that brain's history
  reset. The student sees the bar coming in the same input it trots with. It learned the take-off from states
  just around the hand-over, including its own slightly-off ones that DAgger relabelled, so its take-off is
  smoother. Clean jumps go up at every height, most at 40 cm (+33 points). This is a hypothesis from the
  numbers; the take-off timing has not been measured.
* **walk->sit.** The same teacher, but the student was trained on its own braking states, labelled by 18e (which
  learned gait->sit switches), so the pose is reached more often.
* **Not better:**
  * landing->trot (~50%, all three). Labels after landing come from the dog-motion brain, which cannot hold
    3-4 m/s, and a student cannot beat its teacher there.
  * canter (the clip is not a real canter in either brain)
  * trot->canter (75-78%)

## What failed / caveats

* v1's recovery collapse: inconsistent timeout-driven labels plus beta->0, described under Method. Fixed in v2.
* Supervised fit stays at about 0.11 rad RMS overall. Imitation states are fit to about 0.05 rad. The self-right
  (chaotic, full range) and jump-brain states are fit to about 0.16 rad.
* Only one training seed of the student was run. Seed-to-seed spread of the distillation itself is unmeasured.
  The +- above covers only the test tries.
* PPO fine-tuning (step 3 of the plan) was not run.

## Next steps (with compute)

1. **PPO fine-tune of the student** on the unified env. Use the active command's expert reward (ImitateEnv
   tracking, JumpEnv's jump reward, GetUpEnv's) plus a no-fall/smoothness term for 2 s after each switch. Aim at
   landing->trot and trot->canter, where the teachers are the ceiling. The student is the initial policy, with a
   BC/KL term to the teachers against forgetting. The env runs at about 250 steps/s per core with teacher labels,
   or about 400 without. 20M steps is about 14 h on 4 cores, or about 2 h on 32 cores.
2. **Bigger DAgger on a cloud machine.** The job file is `train/jobs/mastermind-dagger.json` (16 workers, 1M + 19
   x 500k states, 1024-512-256), about 3 h on 16 cores. Before starting it, put run 18e's files into
   `runs/mastermind/imitate18e` (env `IMITATE18E`):

       D=runs/mastermind/imitate18e; mkdir -p $D; for f in policy_20000000_steps.zip policy_vecnormalize_20000000_steps.pkl; do git show origin/claude/train-walk19:train/remote/imitate18e/$f > $D/$f; done
       python ../tools/keepalive/keepalive.py run jobs/mastermind-dagger.json      # or directly:
       MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/dagger.py --rounds 20 --steps0 1000000 --steps 500000 --workers 16 --epochs0 40 --epochs 10 --hidden 1024,512,256 --out runs/mastermind/cloud

3. **Better teachers where the student is capped.** A landing-aware gait brain (dog-motion starts from the jump
   brain's 3-4 m/s landings), or letting the jump brain brake for about 1 s after landing before the gait takes
   over. Both are a one-line teacher change in `env.py` and need another DAgger pass.
4. **GRU/small-transformer student** over about 0.5 s of history, if PPO shows the MLP lacks memory (e.g. the
   tuck/stand phase of the recovery).

## Files

* `env.py`: the unified env and teachers
* `dagger.py`: handler, collection and training
* `student.py`: the student network
* `drive.py`: Bolt-shaped driver for test_skills sequences
* `test_master.py`: switch tests and per-clip exam
* `baseline.py`: the skill manager with a chosen dog-motion brain
* `compare.py`: the results table
* `export.py`: browser export
* `teachers.py`: brain paths
* `results/*.json`: every test run and the DAgger histories
* `export/`: `brain.json`, `model.json`, `student.pt`

Datasets and checkpoints are in `train/runs/mastermind/` (gitignored).

Reproduce (from `train/`, `MUJOCO_GL=osmesa OMP_NUM_THREADS=1`, run 18e at `$IMITATE18E`):

    python mastermind/dagger.py --rounds 4 --steps0 300000 --steps 150000 --out runs/mastermind/out       # v1 rounds 0-3
    python mastermind/dagger.py --init-data runs/mastermind/out/data.npz --init-rows 750000 --rounds 6 --steps 150000 \
        --beta-floor 0.2 --hidden 1024,512,256 --epochs0 15 --epochs 6 --out runs/mastermind/v2 --seed 2
    python mastermind/test_master.py --brain runs/mastermind/v2/student_r5.pt --n 30 --exam --out mastermind/results/student.json
    python mastermind/baseline.py --imitate 18e --n 30 --out mastermind/results/manager_18e.json

## v3: the student retrained with the best teachers (2026-10-27)

**Result.** With the new teachers the student succeeds at **84.5% of switches (721/853, ± 1.2)** on REPORT's v2
rows (N=60, seed sets 1000 and 5000), up from the v2 student's **72% (589/818)** and above the skill manager running the
same new brains (**79.8%, 676/847**). With the new 50 cm jump rows added it is **81% (792/972)**, against 67% and 75%.
Per seed set: 85.9% (seed 1000) and 83.2% (seed 5000), so both seed sets agree. Every switch is level or better than the
v2 student except **canter->trot, which got worse (85% -> 68%)**. The owner's targets are **not** met: 95% of switches
(we are at 81-85%), a 70% canter copy with no falls (66%, 1 fall in 30), and 50 cm jumps (43% clean from a trot).

### Teachers (teacher set "v3", `teachers.SETS`)

| behaviour | v2 teacher | v3 teacher |
|---|---|---|
| walk, trot, pace, turns, sit, lie | 18e at 20M | 18e at 36M (`models/bolt-dog-switching-18e-36M`) |
| canter clip (also the 0.6 s bridge after a landing) | 18e | running brain 18c at 29.5M (`models/bolt-dog-running-18c-29.5M`; the coordinator's update over 24.5M) |
| jump | 16g, handed over 1.1 m before the bar | run-up brain 16i at 35.5M (`models/bolt-dog-jump-runup-16i-35.5M`), handed over 1.1 m before the bar, clock input t/200, at most 200 steps |
| the 2.6 s after a landing (bridge + first 2 s of trot) | 18e | I'm-possible's research student r4 (`models/bolt-dog-imitate-pps-student-r4`) |
| stand, recovery | posture brain, self-right 17e | the same, with the recovery phase shown on the command input (`rec_cmd`, below) |

**Step 1: each new teacher inside the MasterMind world** (`check_teachers.py`, `results/teachers_v3.json`):

* 18c-29.5M canter, exam_canter protocol (N=30, seeds 5000+): tracking **69.7%, 0/30 falls**, speed 72%. Standalone:
  68%, 1/30. (24.5M in the world: 65.0%, 1/30, the same as its standalone 65%, 1/30.)
* 18e-36M, imitate_env.exam (4 per clip, seed 9000): walk .865, trot .827, pace .895, turns .850/.851, sit .923, lie .915.
  Its best.json: .864, .83, .896, .849/.854, .925, .913. Same.
* 16i, jump_env's fair protocol run on the shared world (1.5 m, 2 m/s, fixed physics, N=30, seeds 7000+):
  40 cm **100%**, 50 cm **83%**. Standalone exam_30.json: 100%, 77%.

So the world was right for all three; no fix was needed there.

**The jump hand-over had to change, but not the way planned.** 16i trained from 0.5-1.5 m, arriving at up to 3 m/s
(`jobs/run16i-jump-runup.json`: `--approach 0.5 1.5 --max-steps 200 --approach-speed 3.0`), and its exam is 1.5 m at
2 m/s. It turned out to need that speed. On the same world with randomized physics, at 1.5 m: 2.0 m/s gives 40 cm 100%
and 50 cm 45%; 1.6 m/s gives 35% and 20%; from standing it gets 0% at every height. At 1.1 m and 1.7 m/s: 100% and 85%.
A 2 s trot reaches the hand-over at only 1.52 ± 0.08 m/s. Clean jumps (no touch, landed) from the trot, teacher chain,
N=30, seed 1000, for 20/30/40/50 cm:

| hand-over | clean 20 / 30 / 40 / 50 cm |
|---|---|
| 1.5 m (its exam distance) | 57 / 27 / 20 / 7% |
| 1.3 m | 63 / 53 / 20 / 3% |
| 1.2 m | - / 60 / 40 / 13% |
| **1.1 m** | **80 / 70 / 53 / 20%** |
| 1.0 m | - / 63 / 40 / 7% |
| 0.9 m | - / 50 / 20 / 3% |
| canter run-up (switch to canter for the approach), 1.1 / 1.3 / 1.5 m | 30 cm 23 / 17 / 20%: falls in the approach |
| hand-over gated on the trot's phase (0.5-0.75 of the stride, within 1.35 m) | 83 / 67 / 47 / 27% (no clear gain) |

Almost every failure knocked the bar out of its cups. So v3 hands over at 1.1 m, the same distance as before. The
handler bars are 15-55 cm, placed 1.5-2.2 m ahead. Landing teacher (jump_from 1.1, N=30, seed 1000), landing->trot over
20/30/40/50 cm: 18e alone 74/118; with r4 for 2.6 s **94/118**; a 1.0 s bridge 78/118; no bridge 53/118 (at 1.5 m).
These settings were chosen on seed 1000, the same seeds as the test's first half. Seed 5000 was not looked at, and it
gives the same picture.

### DAgger (`dagger.py --teachers v3`)

Same recipe as v2: round 0 is 400k teacher states with DART noise; rounds 1-7 add 200k each (beta 0.5 -> floor 0.2,
1.8M states); the net is 1024-512-256 ELU; 3 workers; about 70 min of 3 cores per run.

* **Run a (lost the recovery).** Shoves stood up 0 / 85 / 0 / 70 / 0 / 35% of 20 after rounds 0-5 (seed 1000),
  and 2 of 12 after round 7 (dev seeds). Once lying belly-down, the posture brain's "keep tucking" and "stand up" labels fall on the same
  states; the switch is by joint distance, or by a timeout in the test. The command input said "stand" throughout, so
  a student hovering at the boundary stayed down. Its gaits and jumps were fine.
* **Run b (`rec_cmd`).** The recovery phase is on the command input: "stand" while self-righting, "lie" while tucking,
  "stand" again once tucked, and the command clock restarts at each change. The fit dropped from 0.111 to 0.071 rad RMS
  and the recovery held: 75-100% from round 3 on. **The page must reproduce this** (`export_v3/model.json` "recovery"
  spells the rule out: upright > 0.9 and still for 15 steps or 4 s -> "lie" until joints within 0.6 rad of LIE_POSE or
  2 s -> "stand" 3 s).
* **Round choice.** It was made on separate dev seeds (20000+37k, N=12 per sequence), never on the test seeds. Pooled:
  r3 79%, r4 82%, **r5 82%** (recovery 12/12), r6 81%, r7 79%. We took r5.
* A bug fix on the way: a fall in the approach read the jump flags of an earlier jump (or crashed in a fresh world).
  The flags are now reset at the jump command. This did not change any earlier number, because a fall in the approach
  already counted as a failed jump.

### Step 3: switch success (N=60: seed sets 1000 and 5000, 30 each; rate ± binomial s.e. in points)

The old student ran in its own world (teacher set v2). Its 50 cm rows are new runs (`student_v2_r5_jump50*.json`).
"Manager, v3 brains" is `baseline.py --teachers v3` (the `Manager` subclass of skills.Bolt, with skills.py unchanged).
"Teacher chain in env" is the same teachers driving the MasterMind env.

| switch | old student v2 r5 | manager, v3 brains | teacher chain in env, v3 | **new student v3 r5** |
|---|---|---|---|---|
| walk->trot | 59/60 98% ± 2 | 59/60 98% ± 2 | 59/60 98% ± 2 | 60/60 100% ± 0 |
| trot->canter | 46/59 78% ± 5 | 54/59 92% ± 4 | 55/59 93% ± 3 | 57/60 95% ± 3 |
| canter->trot | 39/46 85% ± 5 | 40/54 74% ± 6 | 36/54 67% ± 6 | 39/57 68% ± 6 |
| trot->walk | 39/39 100% ± 0 | 38/38 100% ± 0 | 36/36 100% ± 0 | 39/39 100% ± 0 |
| walk->sit [pose reached] | 31/60 52% ± 6 | 49/60 82% ± 5 | 49/60 82% ± 5 | 49/60 82% ± 5 |
| sit->lie [pose reached] | 31/60 52% ± 6 | 49/60 82% ± 5 | 49/60 82% ± 5 | 49/60 82% ± 5 |
| lie->stand (posture) | 31/31 100% ± 0 | 49/49 100% ± 0 | 49/49 100% ± 0 | 49/49 100% ± 0 |
| trot->jump 20 cm | 42/60 70% ± 6 | 43/60 72% ± 6 | 45/60 75% ± 6 | 47/60 78% ± 5 |
| jump 20 cm->(canter)->trot | 26/52 50% ± 7 | 47/60 78% ± 5 | 43/59 73% ± 6 | 47/58 81% ± 5 |
| trot->jump 30 cm | 45/60 75% ± 6 | 34/60 57% ± 6 | 41/60 68% ± 6 | 47/60 78% ± 5 |
| jump 30 cm->(canter)->trot | 30/58 52% ± 7 | 43/57 75% ± 6 | 49/60 82% ± 5 | 51/60 85% ± 5 |
| trot->jump 40 cm | 33/60 55% ± 6 | 18/60 30% ± 6 | 26/60 43% ± 6 | 37/60 62% ± 6 |
| jump 40 cm->(canter)->trot | 31/60 52% ± 6 | 45/56 80% ± 5 | 44/59 75% ± 6 | 48/59 81% ± 5 |
| fall -> recover (stood up) | 53/60 88% ± 4 | 54/60 90% ± 4 | 50/60 83% ± 5 | 51/60 85% ± 5 |
| recover -> walk | 53/53 100% ± 0 | 54/54 100% ± 0 | 50/50 100% ± 0 | 51/51 100% ± 0 |
| trot->jump 50 cm | 11/60 18% ± 5 | 8/60 13% ± 4 | 9/60 15% ± 5 | 26/60 43% ± 6 |
| jump 50 cm->(canter)->trot | 30/57 53% ± 7 | 40/57 70% ± 6 | 44/55 80% ± 5 | 45/59 76% ± 6 |
| **pooled** | **630/935 67% ± 2** | **724/964 75% ± 1** | **734/961 76% ± 1** | **792/972 81% ± 1** |

On REPORT's v2 rows (no 50 cm): old student 589/818 **72.0 ± 1.6%**, manager with the v2 brains 501/788 63.6 ± 1.7%,
manager with the v3 brains 676/847 **79.8 ± 1.4%**, teacher chain in env 681/846 80.5 ± 1.4%, **new student 721/853
84.5 ± 1.2%**.

Canter copy, exam_canter protocol (N=30, seeds 5000+, `canter_exam.py`; tracking / falls / speed):

* old student: **0.33** / 6 of 30 / 0.38
* new student: **0.66** / 1 of 30 / 0.68
* its teacher 18c-29.5M in the world: 0.70 / 0 of 30 / 0.72

Per-clip exam of the new student (4 per clip): walk .870, trot .829, pace .905, canter .697 (0/4 falls), turns
.855/.862, sit .927, lie .890. Trot and walk keep the 0.83-0.87 of v2.

### What got better, what got worse

* **Better:**
  * sit/lie pose reached 52% -> 82%. That is all teacher: 18e-36M.
  * trot->canter 78% -> 95%
  * every landing->trot: 50-53% -> 76-85%. This is 18c in the bridge and r4 after the landing.
  * 50 cm jumps 18% -> 43%
  * canter copy 33% -> 66% with 1 fall in 30 instead of 6
* **Distillation beat its teachers again on jumps.** The manager with the same brains jumps clean 72/57/30/13%; the
  student does 78/78/62/43% at 20/30/40/50 cm. As in v2, the take-off is learned from the bar distance in the same input
  it trots with.
* **Worse: canter->trot 85% -> 68% ± 6.** The manager with the v3 brains also dropped (74%). The new canter is real and
  faster (72% of the dog's speed against 38%), and 18e's trot, which takes over at the switch, falls sideways out of it.
  No teacher here has learned that switch: 18c never trained switches, and 18e never saw 18c's canter.
* **Level, within error:**
  * recovery 88% -> 85%
  * 40 cm jump 55% -> 62%
  * 20 and 30 cm: 70 -> 78%, 75 -> 78%
* **Targets not met:**
  * 95% of switches (81% with the 50 cm rows, 84.5% without)
  * canter copy 66% (target 70%+, no falls). The teacher itself is at 70%, and the student copies it to within 3 points.
  * 50 cm 43%. The jump teacher only clears 50 cm reliably at about 2 m/s, which a trot never reaches.

### Caveats

* One training seed of the student per configuration; the seed-to-seed spread of distillation is still unmeasured.
* The jump hand-over and landing teacher were tuned on test seed set 1000 (teacher chain only). Seed set 5000 was
  untouched and agrees (83.2% vs 85.9% pooled).
* The student was run only through the Python env. In the browser, the page needs the `rec_cmd` command rule for
  recovery (in model.json).

### Next

1. **Canter teacher upgrade:** `models/bolt-dog-running-18f-30M` (fair canter about 70.6%, 0/60 falls vs 18c-29.5M's
   68%, 1/30) arrived after this DAgger run. Set `SETS["v3"]["canter"]` to it (add it to `imitate_brain`) and rerun
   DAgger, about 70 min.
2. **canter->trot:** label it with a brain that has learned that switch: 18c/18f fine-tuned with switches, as 18e was.
   Or keep the canter brain in charge for about 0.5 s while it slows down.
3. **Jumps from a trot:** fine-tune 16i on hand-overs from recorded trot states at 1.0-1.5 m and 1.4-1.7 m/s. Its run-up
   training only started from a standing stance with a velocity kick. Or train a speed-up into the approach.

### Files (v3)

* `teachers.py` (teacher sets), `env.py` (sets, run_up / land / jump_phase / rec_cmd options), `dagger.py`
  (`--teachers --over`), `baseline.py` (`Manager`, `--teachers`, `--with-50`), `test_master.py` (`--teachers --over --with-50`),
  `compare.py` (`--col` merges seed sets, pooled ± s.e.), `check_teachers.py`, `canter_exam.py`, `export.py` (teacher set,
  recovery rule)
* results: `teachers_v3.json`, `manager_v3_n30*.json`, `teacher_in_env_v3_n30*.json`, `student_v3_r5_n30*.json`,
  `student_v2_r5_jump50*.json`, `canter30_student_v2/v3.json`, `dagger_v3_history.json`
* `export_v3/`: brain.json (torch parity 1e-5), model.json, student.pt (run b round 5)
* Big files: `train/runs/mastermind/v3/` (gitignored): run a `dagger/`, run b `daggerb/`, variant tests `var/`, dev
  tests `dev/`, `devb/`

Reproduce (from `train/`, `MUJOCO_GL=osmesa OMP_NUM_THREADS=1`):

    python mastermind/check_teachers.py --teachers v3 --out mastermind/results/teachers_v3.json
    python mastermind/dagger.py --teachers v3 --rounds 8 --steps0 400000 --steps 200000 --workers 3 --beta-floor 0.2 \
        --hidden 1024,512,256 --epochs0 20 --epochs 6 --out runs/mastermind/v3/daggerb --seed 4
    python mastermind/test_master.py --brain runs/mastermind/v3/daggerb/student_r5.pt --teachers v3 --n 30 --seed 1000 --with-50 --exam --out mastermind/results/student_v3_r5_n30.json
    python mastermind/baseline.py --teachers v3 --n 30 --seed 1000 --with-50 --out mastermind/results/manager_v3_n30.json
    python mastermind/compare.py --col "new=mastermind/results/student_v3_r5_n30.json,mastermind/results/student_v3_r5_n30_seed5000.json"

## v4: the student retrained with a gait-switching teacher (2026-10-28)

**Result.** The v4 student succeeds at **83.0% of switches (818/985, ± 1.2)** with the 50 cm rows and **86.6% (752/868)**
on REPORT's v2 rows (N=60, seed sets 1000 and 5000). v3 was 81.5% (792/972) and 84.5% (721/853). The manager running the v4
brains gets 74.2% / 78.8%. Per seed set: 83.2% (seed 1000) and 82.9% (seed 5000). **canter->trot went from 68% to 86%**,
which was the aim. But the pooled gain over v3 is only about 1.5-2 points, about one standard error of the difference, so
it is **not clearly better overall**. The owner's **95% target is not met**. The remaining failures are where every
teacher is weak: sitting from a walk, jumps of 40-50 cm from a trot, and the 2.6 s after a landing.

### Teachers (teacher set "v4", `teachers.SETS`)

| behaviour | v3 teacher | v4 teacher |
|---|---|---|
| walk, trot, canter, turns (also the 0.6 s canter bridge) | 18e-36M; canter 18c-29.5M | **18g at 35.5M** (`models/bolt-dog-canter-switching-18g-35.5M`, `clips` map, key `18g35`) |
| pace, sit, lie | 18e-36M | 18e-36M (unchanged) |
| jump, landing, stand, recovery | 16i-35.5M at 1.1 m, r4 for 2.6 s, posture, self-right 17e, `rec_cmd` | unchanged |

New in the code: `SETS[...]["clips"]` ({clip: brain}, used by env.py and baseline.Manager), `approach` (a brain for the
gait during a jump's run-up), `sr_limit` (the self-right timeout when not training), `teachers.imitate_policy()` (also
loads the Kaggle GPU brain `gpu360`), and `dagger.py --init-student`.

**Step 1: the teachers in the MasterMind world** (`results/teachers_v4.json`, `teachers_v4b.json`):

* 18g-35.5M canter, exam_canter protocol (N=30, seeds 5000+): **69.7%, 0/30 falls**, speed 68%. Standalone: 69.4%.
* 18g walk/trot/turns .902/.883/.882/.877 (18e-36M had .865/.827/.850/.851). pace/sit/lie from 18e-36M .895/.923/.915, the same as in v3.
* 16i fair jump: 40 cm 100%, 50 cm 83%, the same as in v3.
* GPU brain (360M, arrived during the run): canter 76.1%, 0/30 falls; walk .931, trot .920, sit .967, lie .937. Its standalone numbers are 76%, 93, 92, 97, 94.

**Choosing the sit/lie teacher.** This used the teacher chain on dev seeds 20000 (N=30, `results/dev_v4/teach_*.json`),
never the test seeds. With 18g for the gaits:

| sit/lie teacher | walk->sit pose reached | pooled (with 50 cm) |
|---|---|---|
| 18e-36M (v4) | **24/30** | **77%** |
| 18d-24.5M (v4d) | 2/30 | 67% |
| 18g itself (v4g, 8-slot) | 2/30 | 67% |
| v3 set, for reference | 21/30 | 74% |

18d's sit is the best when it starts from its own standing pose (95%). Handed 18g's walking body, it almost never
sits. Only 18e-36M learned clip switches that include sit, so it stays.

**Jumps got harder for the teachers.** With 18g's trot, the teacher chain's clean jumps on dev seeds were 73/57/23/0%
at 20/30/40/50 cm. With v3 (18e's trot) they were 67/70/40/10%. Running the approach on 18e-36M (`v4a`) gave no clear
gain over two dev seed sets (104 vs 98 clean jumps of 240), so v4 keeps 18g for the approach. A longer self-right
timeout (400 steps) changed nothing (26/30 and 28/30 either way). Recovery failures are self-righting that stays on
its back, or a tuck that never reaches the lie pose, not the timeout.

### DAgger (`dagger.py --teachers v4`)

v3's recipe, warm-started from the v3 student's weights (`--init-student runs/mastermind/v3/daggerb/student_r5.pt`).
Round 0 is 400k teacher states; rounds 1-8 add 200k each (beta 0.5 -> 0.2). The net is 1024-512-256, 3 workers, about
75 min. The fit is 0.051 rad RMS at round 4 (v3: 0.071).

**Round choice on dev seeds 20000 (N=30 per sequence, with 50 cm; `results/dev_v4/`):** r3 77%, **r4 82.3%**, r5 81.6%,
r6 81%, r7 77%, r8 79%. The v3 student on the same seeds gets 78%. **r4** was taken. It was the only round tested on
the test seeds.

**Second run with the GPU brain (`v4b`)**, as the coordinator suggested. This set uses the GPU brain for every clip,
including sit/lie. Its teacher chain on dev seeds: canter->trot 59/60, sit 48/60 vs v4's 47/60, so it looked at least
as good. It was warm-started from v4 r8, with the same recipe. **The student copied the gaits perfectly but not the
sit:** dev canter->trot 30/30 in every round, but sit pose 37-60% (the teacher gets 87% on the same seeds), and clean
jumps fell too. Pooled on dev: r3 76%, r4 74%, r5-r8 73-75%. It lost to v4 r4 on dev and was not tested on the test
seeds. The fit was even lower (0.041 rad), so the cause is not capacity. The GPU brain's sit from a moving walk is
probably harder to copy, but this has not been checked. Manager with the v4b brains on the test seeds:
76.2% / 81.4% (`manager_v4b_*.json`).

### Step 3: switch success (N=60: seed sets 1000 and 5000; rate ± binomial s.e. in points)

| switch | v3 student r5 | manager, v3 brains | manager, v4 brains | teacher chain in env, v4 | **v4 student r4** |
|---|---|---|---|---|---|
| walk->trot | 60/60 100% ± 0 | 59/60 98% ± 2 | 60/60 100% ± 0 | 60/60 100% ± 0 | 60/60 100% ± 0 |
| trot->canter | 57/60 95% ± 3 | 54/59 92% ± 4 | 60/60 100% ± 0 | 60/60 100% ± 0 | 59/60 98% ± 2 |
| canter->trot | 39/57 68% ± 6 | 40/54 74% ± 6 | 56/60 93% ± 3 | 53/60 88% ± 4 | **51/59 86% ± 4** |
| trot->walk | 39/39 100% ± 0 | 38/38 100% ± 0 | 55/56 98% ± 2 | 53/53 100% ± 0 | 49/50 98% ± 2 |
| walk->sit [pose reached] | 49/60 82% ± 5 | 49/60 82% ± 5 | 47/60 78% ± 5 | 47/60 78% ± 5 | 47/60 78% ± 5 |
| sit->lie [pose reached] | 49/60 82% ± 5 | 49/60 82% ± 5 | 47/60 78% ± 5 | 47/60 78% ± 5 | 47/60 78% ± 5 |
| lie->stand (posture) | 49/49 100% ± 0 | 49/49 100% ± 0 | 47/47 100% ± 0 | 47/47 100% ± 0 | 47/47 100% ± 0 |
| trot->jump 20 cm | 47/60 78% ± 5 | 43/60 72% ± 6 | 37/60 62% ± 6 | 43/60 72% ± 6 | 51/60 85% ± 5 |
| jump 20 cm->(canter)->trot | 47/58 81% ± 5 | 47/60 78% ± 5 | 42/58 72% ± 6 | 49/60 82% ± 5 | 44/60 73% ± 6 |
| trot->jump 30 cm | 47/60 78% ± 5 | 34/60 57% ± 6 | 27/60 45% ± 6 | 38/60 63% ± 6 | 52/60 87% ± 4 |
| jump 30 cm->(canter)->trot | 51/60 85% ± 5 | 43/57 75% ± 6 | 42/58 72% ± 6 | 47/59 80% ± 5 | 48/60 80% ± 5 |
| trot->jump 40 cm | 37/60 62% ± 6 | 18/60 30% ± 6 | 17/60 28% ± 6 | 25/60 42% ± 6 | 39/60 65% ± 6 |
| jump 40 cm->(canter)->trot | 48/59 81% ± 5 | 45/56 80% ± 5 | 39/58 67% ± 6 | 43/58 74% ± 6 | 50/58 86% ± 5 |
| fall -> recover (stood up) | 51/60 85% ± 5 | 54/60 90% ± 4 | 56/60 93% ± 3 | 57/60 95% ± 3 | 54/60 90% ± 4 |
| recover -> walk | 51/51 100% ± 0 | 54/54 100% ± 0 | 56/56 100% ± 0 | 57/57 100% ± 0 | 54/54 100% ± 0 |
| trot->jump 50 cm | 26/60 43% ± 6 | 8/60 13% ± 4 | 5/60 8% ± 4 | 3/60 5% ± 3 | 21/60 35% ± 6 |
| jump 50 cm->(canter)->trot | 45/59 76% ± 6 | 40/57 70% ± 6 | 44/60 73% ± 6 | 48/58 83% ± 5 | 45/57 79% ± 5 |
| **pooled** | **792/972 81.5% ± 1.2** | **724/964 75.1% ± 1.4** | **737/993 74.2% ± 1.4** | **777/992 78.3% ± 1.3** | **818/985 83.0% ± 1.2** |

On REPORT's v2 rows (no 50 cm): v3 student 721/853 84.5%, manager with the v3 brains 79.8%, manager with the v4 brains
688/873 78.8%, teacher chain with the v4 brains 726/874 83.1%, **v4 student 752/868 86.6% ± 1.2**.

Canter copy, exam_canter protocol (N=30, seeds 5000+, `canter_exam.py`; tracking / falls / speed):

* v3 student: 0.66 / 1 of 30 / 0.68
* **v4 student: 0.672 / 1 of 30 / 0.65**
* its teacher 18g in the world: 0.697 / 0 of 30 / 0.68

Per-clip exam (4 per clip, seed 9000): walk .908, trot .886, pace .907, canter .695 (0/4 falls), turns .888/.881,
sit .926, lie .904. Walk and trot are up from v3's .870/.829, following 18g.

### What got better, what got worse

* **Better:**
  * canter->trot 68% -> 86% ± 4 (the teacher is at 88-93%)
  * walk and trot tracking .87/.83 -> .91/.89
  * clean jumps 20/30 cm 78/78% -> 85/87%. The student again beats its teachers here (chain 72/63%).
  * recovery 85% -> 90%
* **Worse (all within about 1 s.e., but the direction is down):**
  * landing->trot after 20 cm: 81% -> 73% ± 6
  * after 30 cm: 85% -> 80%
  * 50 cm clean: 43% -> 35% ± 6
  * sit/lie pose reached: 82% -> 78%. The teacher chain itself is at 78%: 18g's walk brakes less well into 18e's sit.
* **Level:**
  * 40 cm (62 -> 65%)
  * trot->canter (95 -> 98%)
  * canter copy 0.66 -> 0.67, still 1 fall in 30. The target of 70%+ with no falls is not met; the teacher is at 0.70.
* **Targets not met:**
  * 95% of switches: 83% with 50 cm, 87% without
  * canter 70% with no falls
  * 50 cm jumps (35%)

### Caveats

* One training seed per configuration. The rounds were chosen on dev seeds 20000. The teacher choices (sit/lie
  teacher, approach brain, self-right timeout) were made on dev seeds 20000-22000 only. Test seed sets 1000/5000 were
  run once, for v4 r4 only.
* The page needs the same `rec_cmd` recovery rule as v3 (`export_v4/model.json` "recovery").

### Next

1. **The hybrid set**: the GPU brain for the gaits (its chain is 100% on canter->trot on dev, and the student copies
   its gaits perfectly) plus 18e-36M for sit/lie. It needs a teacher-chain check of GPU walk -> 18e sit, then about
   75 min of DAgger.
2. **Sit from a walk** (78-82% in every teacher here): a dog-motion brain fine-tuned with walk->sit switches from 18g's/the
   GPU brain's walking states, or a braking "stop" clip before sit.
3. **Jumps and landings** are as in v3's Next: fine-tune 16i on hand-overs from recorded 18g trot states, and a
   landing-aware gait brain.

### Files (v4)

* code: `teachers.py` (sets v4, v4a, v4d, v4g, v4b/v4gpu_*, keys `18g35`, `18d`, `imitate_policy`/`gpu360`), `env.py`
  (`clips`, `approach`, `sr_limit`), `baseline.py` (the same for the Manager), `dagger.py` (`--init-student`, v4 handler)
* results: `teachers_v4.json`, `teachers_v4b.json`, `manager_v4_n30*.json`, `manager_v4b_n30*.json`,
  `teacher_in_env_v4_n30*.json`, `student_v4_r4_n30*.json`, `canter30_student_v4.json`, `dagger_v4_history.json`,
  `dagger_v4b_history.json`, and `dev_v4/` (every dev-seed run: teacher choices, rounds of both runs, v3 student on dev)
* `export_v4/`: brain.json (torch parity 1.2e-5), model.json, student.pt (v4 round 4)
* Big files: `train/runs/mastermind/v4/` (gitignored): `dagger/` (v4), `daggerb/` (v4b), `dev*/`, `final/`

Reproduce (from `train/`, `MUJOCO_GL=osmesa OMP_NUM_THREADS=1`):

    python mastermind/check_teachers.py --teachers v4 --out mastermind/results/teachers_v4.json
    python mastermind/dagger.py --teachers v4 --rounds 9 --steps0 400000 --steps 200000 --workers 3 --beta-floor 0.2 \
        --hidden 1024,512,256 --epochs0 20 --epochs 6 --init-student runs/mastermind/v3/daggerb/student_r5.pt --out runs/mastermind/v4/dagger --seed 6
    python mastermind/test_master.py --brain runs/mastermind/v4/dagger/student_r4.pt --teachers v4 --n 30 --seed 20000 --with-50 --out dev.json   # round choice
    python mastermind/test_master.py --brain mastermind/export_v4/student.pt --teachers v4 --n 30 --seed 1000 --with-50 --exam --out mastermind/results/student_v4_r4_n30.json
    python mastermind/baseline.py --teachers v4 --n 30 --seed 1000 --with-50 --out mastermind/results/manager_v4_n30.json
    python mastermind/canter_exam.py --brain mastermind/export_v4/student.pt --teachers v4 --out mastermind/results/canter30_student_v4.json

## Second simulator: the student in PyBullet (2026-10-28, in simulation)

**Question.** Does MasterMind only work in MuJoCo, or does it still work in a second physics engine on the same body?
This is not hardware. It is a cheap check for brains that over-fit one simulator's contact model.

**Result.** Most of it works in PyBullet.
* Walk, trot, pace, the turns, sit and lie run without a fall. Tracking drops from 0.89-0.93 to 0.71-0.84, and speed
  from 0.88-0.89 to 0.75-0.80 of the dog's.
* walk->trot is 30/30 in both engines.
* The canter breaks: tracking 0.15, 3 of 4 starts fall, and trot->canter goes from 100% to 60%.
* Recovery after a shove drops from 100% to 70%. Sit pose reached drops from 90% to 73%.
* Pooled switches (gaits, rest, shove; N=30, seed 1000): **v4 258/266 97% in MuJoCo vs 186/223 83% in PyBullet**, with
  83% of the planned switches reached instead of 99%.
* The v3 student loses more: 88% -> 70%, and only 6 of 18 lie->stand.

So the slow and medium gaits do not depend on MuJoCo's contacts. The canter, self-righting and the sit do, at least
in part. The canter was already the weakest skill (its teacher scores 0.70).

**How** (`sim2sim.py`, `bullet_server.py`).
* The Master env and the test sequences run unchanged. Only the physics call is swapped.
* Each 20 ms the joint targets go to a PyBullet process. It simulates 10 steps of 2 ms with bolt.xml's motor model in
  torque control: a kp 45 / kv 1.2 servo, clipped to the 33.5 N·m / 420 W / 21-25 rad/s limits, plus the joint damping
  and friction.
* The robot state comes back into MuJoCo's data and is `mj_forward`ed. So the observation, the fall detector, the
  scores and the bookkeeping are the same code in both engines.
* The body is bolt.xml converted to URDF by `write_urdf()` (`robots/bolt.urdf`): same masses, full inertias, joint
  frames, axes and limits, the collision primitives, friction 1.0, and self-collision except parent-child.
* Sanity check: holding the dog stance open-loop for 2 s settles at 0.438 m in MuJoCo and 0.436 m in PyBullet.
* Both engines run the **nominal** body here (no randomization, no sensor noise), so the MuJoCo column is higher than
  the randomized test rows above.
* Not modelled in PyBullet: joint armature (0.01 kg m² rotor inertia per joint). `--armature-approx` adds it to each
  leg link's inertia instead. It changes the pooled rate by nothing (186/225) and helps the canter and sit somewhat
  (canter 0.43, 1 of 4 falls; sit 0.89).
* No hurdle in PyBullet, so no jump rows. pybullet has no wheel for Python 3.13, so it runs in a Python 3.11 venv
  (`BULLET_PYTHON`), talking JSON over a pipe.

| switch | v4 MuJoCo | v4 PyBullet | v4 PyBullet, armature approx. | v3 MuJoCo | v3 PyBullet |
|---|---|---|---|---|---|
| walk->trot | 30/30 100% ± 0 | 30/30 100% ± 0 | 30/30 100% ± 0 | 28/30 93% ± 5 | 30/30 100% ± 0 |
| trot->canter | 30/30 100% ± 0 | 18/30 60% ± 9 | 19/30 63% ± 9 | 28/28 100% ± 0 | 16/30 53% ± 9 |
| canter->trot | 29/30 97% ± 3 | 15/16 94% ± 6 | 14/18 78% ± 10 | 12/24 50% ± 10 | 11/11 100% ± 0 |
| trot->walk | 28/29 97% ± 3 | 15/15 100% ± 0 | 14/14 100% ± 0 | 12/12 100% ± 0 | 11/11 100% ± 0 |
| walk->sit [pose reached] | 27/30 90% ± 5 | 22/30 73% ± 8 | 23/30 77% ± 8 | 24/30 80% ± 7 | 18/30 60% ± 9 |
| sit->lie [pose reached] | 27/30 90% ± 5 | 22/29 76% ± 8 | 23/30 77% ± 8 | 24/30 80% ± 7 | 18/18 100% ± 0 |
| lie->stand (posture) | 27/27 100% ± 0 | 22/22 100% ± 0 | 23/23 100% ± 0 | 24/24 100% ± 0 | 6/18 33% ± 11 |
| fall -> recover (stood up) | 30/30 100% ± 0 | 21/30 70% ± 8 | 20/30 67% ± 9 | 28/30 93% ± 5 | 11/30 37% ± 9 |
| recover -> walk | 30/30 100% ± 0 | 21/21 100% ± 0 | 20/20 100% ± 0 | 28/28 100% ± 0 | 11/11 100% ± 0 |
| **pooled** | **258/266 97% ± 1** | **186/223 83% ± 2** | **186/225 83% ± 3** | **208/236 88% ± 2** | **132/189 70% ± 3** |
| switches reached (tried / planned) | 266/270 99% | 223/270 83% | 225/270 83% | 236/270 87% | 189/270 70% |

Per-clip exam (4 starts per clip, seed 9000; tracking / share of starts that fell / speed kept):

| clip | v4 MuJoCo | v4 PyBullet | v4 PyBullet, armature | v3 MuJoCo | v3 PyBullet |
|---|---|---|---|---|---|
| walk | .91 / 0 / .88 | .82 / 0 / .75 | .83 / 0 / .78 | .87 / 0 / .92 | .79 / 0 / .82 |
| trot | .89 / 0 / .89 | .75 / 0 / .80 | .74 / 0 / .80 | .83 / 0 / .83 | .71 / 0 / .79 |
| pace | .91 / 0 / .88 | .82 / 0 / .78 | .84 / 0 / .84 | .91 / 0 / .88 | .80 / 0 / .76 |
| canter | .70 / 0 / .68 | **.15 / .75 / .11** | .43 / .25 / .41 | .70 / 0 / .73 | .16 / .75 / .15 |
| turn left / right | .89 / .89 | .77 / .77 | .83 / .80 | .85 / .86 | .77 / .75 |
| sit / lie | .93 / .91 | .71 / .76 | .89 / .85 | .93 / .89 | .65 / .72 |

Caveats:
* N=30 on one seed set (1000); the ± is binomial over the tries reached. A harder engine also means fewer switches are
  reached, because a fall before a switch skips it (last row).
* PyBullet's contact parameters are its defaults (50 solver iterations, no tuning). That is deliberate: tuning them
  until the brain works would defeat the check.

Reproduce (from `train/`, `MUJOCO_GL=osmesa OMP_NUM_THREADS=1`, `BULLET_PYTHON` = a Python with pybullet + numpy, e.g.
`python3.11 -m venv venv311 && venv311/bin/pip install pybullet numpy`):

    python mastermind/sim2sim.py --engine mujoco --brain mastermind/export_v4/student.pt --teachers v4 --n 30 --exam --out mastermind/results/sim2sim/v4_mujoco.json
    python mastermind/sim2sim.py --engine bullet --brain mastermind/export_v4/student.pt --teachers v4 --n 30 --exam --out mastermind/results/sim2sim/v4_bullet.json
    python mastermind/sim2sim.py --engine bullet --armature-approx --brain mastermind/export_v4/student.pt --teachers v4 --n 30 --exam --out mastermind/results/sim2sim/v4_bullet_armature.json
    python mastermind/sim2sim.py --engine mujoco --brain mastermind/export_v3/student.pt --teachers v3 --n 30 --exam --out mastermind/results/sim2sim/v3_mujoco.json
    python mastermind/sim2sim.py --engine bullet --brain mastermind/export_v3/student.pt --teachers v3 --n 30 --exam --out mastermind/results/sim2sim/v3_bullet.json
    python mastermind/compare.py --reached --col "v4 MuJoCo=mastermind/results/sim2sim/v4_mujoco.json" --col "v4 PyBullet=mastermind/results/sim2sim/v4_bullet.json"
