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
