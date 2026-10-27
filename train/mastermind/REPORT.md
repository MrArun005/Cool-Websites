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
