# Look before you leap: plan around the dog-motion brain, then distill the plan back in

Research agent "I'm-possible", 2026-10-27. Code: `train/research/` (new files only; existing modules are
imported, never changed). Large files (datasets, student brains) live in `train/runs/research/` (gitignored);
the best student is `train/runs/research/student_r4/` (policy.zip + vecnormalize.pkl, same 74 inputs as 18b).

## TL;DR

**Idea: policy-prior predictive sampling (PPS) + DAgger distillation ("plan, then distill").**
Every 0.1 s, copy Bolt's body state into 16 simulated "shadow" Bolts with *nominal* physics (not the
randomized physics Bolt is really in), let the existing run-18b dog-motion brain drive each for 0.6 s with
a small smooth random offset added to its actions (one shadow gets none: the plain brain), score each
future with the brain's own training reward (DeepMimic tracking) minus a fall penalty, and run the real
Bolt with the winning offset. The brain still closes the loop every 1/50 s; the planner only nudges it,
and keeps the plain brain's action 93-95 % of the time. Then record what the planner made the brain do
on the states the *student* visits (DAgger) and fine-tune the brain's actor on it, so the gain survives
with **no planner**: one forward pass, same inputs, exportable to the browser/ONNX like 18b.

**Headline (no planner at run time, distilled student r4 vs the brain it came from, 18b-17.5M), on seeds
never used for training or for choosing between students:**

| held-out test (seeds 5000+37k, canter 7000+k) | baseline 18b-17.5M | student r4 | Fisher / sign p |
|---|---|---|---|
| walk -> sit, sit pose reached | 2/10 | **9/10** | 0.006 |
| sit -> lie, pose reached / lie -> stand | 2/10 / 2/2 (8 not reached) | **9/10 / 9/9** | 0.006 |
| landing after a 20/30/40 cm jump -> canter -> trot | 9/25 | **19/26** | 0.012 |
| gaits walk->trot->canter->trot->walk (4 switches) | 36/38 | 38/39 | n.s. |
| canter falls (4 s, randomized physics) | 8/30 | **1/30** | 0.026 |
| canter tracking mean ± sd | 0.419 ± 0.224 | **0.529 ± 0.099** | paired +0.111 ± 0.035 se, 25 wins / 4 losses, p = 1e-4 |
| canter pace kept (share of the dog's distance) | 0.47 | 0.58 | |
| trot -> jump clean (jump brain unchanged) | 13/30 | 9/30 | 0.42 (n.s., watch) |
| shove -> self-right + stand / -> walk | 8/10 / 8/8 | 10/10 / 9/10 | n.s. |
| from standing still: sit, lie, stand | 10/10 each | 10/10 each | |
| other clips' tracking (walk, trot, pace, turns, sit, lie; N = 10, seeds 9000+) | 0.89-0.94, 0 falls | within 0.013 of baseline (sit 0.941 -> 0.922, lie 0.934 -> 0.921), 0 falls | |

**Development seeds (test_skills seeds 1000+37k, canter 9000+k; these were seen while iterating r0-r4):**

| test | baseline 18b-17.5M | PPS planner on top of 18b | student r1 (canter only) | student r4 (final) |
|---|---|---|---|---|
| walk -> sit, pose reached | 1/10 | **10/10** (p = 1e-4) | 1/10 | **10/10** |
| sit -> lie pose / lie -> stand | 1/10 / 1/1 | 10/10 / 10/10 | 1/10 | 10/10 / 10/10 |
| gaits, 4 switches | 35/37 | **40/40** | 38/39 | 37/38 |
| canter tracking (N = 30) | 0.483 ± 0.164 | **0.552 ± 0.026** | 0.551 ± 0.029 | **0.554 ± 0.026** |
| canter falls (N = 30) | 3/30 | **0/30** | 0/30 | **0/30** |
| canter paired gain vs baseline | | +0.069 ± 0.028, 17/3, p = 0.003 | +0.068 ± 0.026, 22/6, p = 0.004 | |
| trot -> jump 30 cm (not planned) | 7/10 | 7/10 | 6/10 | 6/10 |
| landing 30 cm -> trot | 8/9 | 8/9 | 6/10 | 7/10 |
| shove -> recover -> walk | 9/10, 9/9 | 8/9, 8/8 | 9/10, 9/9 | 10/10, 9/10 |
| wall-clock for the 4-sequence suite (N = 10) | 58 s | 2056 s (~35x, loaded CPU) | 41 s | ~60 s |

**Is it a breakthrough?** For this project, yes in the narrow sense: the worst transition we had (walk ->
sit -> lie, 1-2 in 10) now works 9-10 in 10, landings after a jump go from about 1 in 3 to 3 in 4, and the
canter stops falling, with the **same network size, same inputs, no new RL run**, ~4 CPU-hours in total
(2 cores), and it holds on held-out seeds. It is not a revolution: the planner can only choose among what
the brain can already do, so the canter is still a slow canter (58-60 % of the dog's pace; the tracking of
tries that do not fall rose only 0.534 -> 0.552). It is a cheap, general way to turn "the brain almost
can" into "the brain does", and it composes with every other plan (MasterMind's generalist can be the
proposal policy just the same).

## Why this idea (and not the others)

The ask: a new method that beats per-skill PPO, CPU only, a few hours, our data and brains. The "one
generalist distilled from 18b/16g/17e" option belongs to the MasterMind agent, so it was excluded.

| candidate | expected gain | cost here | verdict |
|---|---|---|---|
| Motion matching over the mocap, feeding the tracking brain references for any speed/turn | high long-term | the brain knows 9 clip one-hots; a new reference stream is out of distribution, so it needs a new tracking run (~20-30M steps) and the raw BVH re-retargeted | not provable in hours; step 4 below |
| Latent motion prior (VAE / transformer over mocap windows) as action space | high (ASE/CALM/PULSE) | tracker on all 30 min, then prior, then high-level policy: GPU-weeks in the papers | no |
| World-model RL (DreamerV3 / TD-MPC2) from scratch | uncertain for mocap imitation | GPU-days | no |
| **Planning with the trained brain as proposal, then DAgger distillation** | aims exactly at the measured failures | no RL; MuJoCo state copy is free; CPU | **chosen** |

Every failure in skills_report.json is "walks into a state it cannot recover from within a second" (falls
0.5-0.8 s after a switch, nose-down sit, sideways fall after a landing). A reflex policy cannot see that
coming; a 0.6 s look-ahead can. It is the cheapest form of the policy + planning idea behind TD-MPC2, LOOP,
Expert Iteration and MuZero: the dynamics model is the simulator (with deliberately wrong physics), the
proposal is the PPO brain, the objective is its training reward.

## Method (research/pps.py, run_switches.py, pps_distill.py, collect_landing.py)

* Planner: every M = 5 control steps, K = 16 candidates, horizon H = 30 (0.6 s). Offsets piecewise-linear
  with knots every 5 steps, N(0, 0.2) per knot (actions in [-1, 1], ~0.2 rad of joint target).
  Candidate 0 = no offset, candidate 1 = previous winner shifted (warm start), 6 around the previous
  winner, 8 around zero. Pick the best (predictive sampling, not MPPI averaging).
* Shadows are `ImitateEnv(randomize=False)`: nominal motor strength, mass, friction, no sensor noise, plain
  world (no hurdle). Only qpos[0:19], qvel[0:18] and the solver warm start are copied, plus clip, clip
  clock and last action. Bolt's real kp is 0.85-1.15x, torso mass 0.85-1.15x, friction 0.6-1.3x of what the
  planner assumes, and it still works.
* Score: sum of the training reward (0.5 pose, 0.05 joint vel, 0.2 paws, 0.15 body pose, 0.1 body vel)
  over the horizon, minus 10 on ImitateEnv's termination (15 cm off the reference height, 0.8 rad tilt,
  body on the floor while the clip is not lying).
* The planner keeps the plain brain's action most of the time (canter: 1123 of 1200 plans; it saw the plain
  brain fall in its model 13 times and always found a non-falling alternative).
* In the skill manager (`PPSBolt`) only the dog-motion brain is planned; jump, self-righting and posture
  brains run unchanged; clip entry is unchanged (closest moment); the plan resets at every clip change.
* Distillation labels: brain raw mean + chosen offset (pre-clip action units). Only the actor MLP and action
  head are fine-tuned (MSE, Adam 1e-4, 20-30 epochs, planner samples weighted 3-5x vs anchor samples of
  the original brain on the other clips). Critic and VecNormalize are untouched: drop-in policy.zip.
  DAgger rounds: the student drives, following the plan only 30 % of the 0.1 s chunks (beta = 0.3), and
  every step is labelled by the planner.

| student | added data (labelled steps) | result |
|---|---|---|
| r0 | 30 canter tries driven by the planner (6000) + 9.5k anchors | canter 0.493 ± 0.155, 3/30 falls: **no gain**. Cloning planner rollouts never visits the states where the brain goes wrong. |
| r1 | DAgger: 40 canter tries driven by r0 (6961) | canter 0.551 ± 0.029, 0/30: the whole planner gain |
| r2 | 20 walk/trot -> sit -> lie tries with the planner, driven by r1 (7209) | sit pose 4/10 |
| r3 | DAgger: 24 sit/lie tries driven by r2 | sit pose 8/10, gaits 40/40, but **jump landing -> trot 2/9** (forgot the fast-landing canter states) and canter 1/10 falls |
| r4 | DAgger: 20 trot -> jump (20-40 cm) -> canter -> trot tries driven by r3, planned dog-motion steps labelled (2846) | the table above |

Total: ~27k labelled steps, ~4 CPU-hours of planning on 2 cores; each fit takes ~1 min.

## What failed or did not help

* Pure behaviour cloning of planner rollouts (r0): zero gain. DAgger was necessary.
* Catastrophic forgetting between transitions: adding sit/lie data (r3) broke jump landings (8/9 -> 2/9)
  until landing states were added (r4). Distilling transition by transition needs every transition in
  the data at once: the GPU run below.
* Longer, noisier planning (H = 50, sigma = 0.4): canter 0.494 ± 0.173, 1/10 falls; worse than H = 30,
  sigma = 0.2 (noisy candidates rarely beat the plain brain, and the nominal model's errors pile up).
* No real canter: pace stays ~60 % of the dog's. That is a brain/training limit (run 18c/22 territory).
* Sitting down from a **trot** (not in the test suite, only in the distillation data): 5 of 10 tries fell
  0.25-0.4 s into the sit even with the planner (walk starts: 0/10). Braking from a trot needs a skill the
  brain does not have.
* The jump itself is not planned (jump brain unchanged): trot -> jump clean 13/30 -> 9/30 with r4 on held-out
  seeds (p = 0.42, not significant; the approach trot comes from the dog-motion brain, so it is worth a
  look). Planning the jump brain with jump_env's reward is the obvious next CPU test.
* The planner itself is ~35x the cost of the plain brain in Python on this loaded CPU: not real-time on one
  core. It is embarrassingly parallel (16 independent rollouts), and the distilled student costs nothing.
* Honesty notes: development seeds were looked at while choosing r3 -> r4, which is why the held-out table
  exists; N = 10 per switch leaves the gait-switch differences non-significant.

## Related work (with what has not been done)

Motion imitation and skill embeddings
* DeepMimic, Peng et al. 2018, https://arxiv.org/abs/1804.02717 (our tracking reward).
* Learning Agile Robotic Locomotion Skills by Imitating Animals, Peng et al. 2020, https://arxiv.org/abs/2004.00784 (our clips' approach; one policy per clip).
* AMP https://arxiv.org/abs/2104.02180 ; ASE https://arxiv.org/abs/2205.01906 ; CALM https://arxiv.org/abs/2305.02195 (adversarial priors, latent skills; transitions from the latent).
* Quadruped follow-ups: CAMP (conditional AMP, one policy, Go2/A1) https://arxiv.org/abs/2509.21810 ;
  Seamless multi-skill learning (AMP + self-trajectory augmentation, Solo8) https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1542692 ;
  dog-mocap basic behaviour controller with semi-supervised GAIL and latent switching https://arxiv.org/abs/2505.09979 ;
  PASIST https://arxiv.org/abs/2503.03476 ; Barkour https://arxiv.org/abs/2305.14654 .

Universal trackers and distillation
* PHC https://arxiv.org/abs/2305.06456 ; PULSE (latent prior distilled from a tracker with DAgger) https://arxiv.org/abs/2310.04582 ;
  MaskedMimic https://arxiv.org/abs/2409.14393 ; UniTracker (CVAE + DAgger) https://arxiv.org/abs/2507.07356 .
* Multi-expert DAgger distillation + RL fine-tuning, quadruped parkour https://arxiv.org/abs/2505.11164 ; DAgger https://arxiv.org/abs/1011.0686 .

Generative motion priors and planners
* DiffuseLoco https://arxiv.org/abs/2404.19264 ; CLoSD (diffusion planner + RL tracker in closed loop) https://arxiv.org/abs/2410.03441 ;
  Learned Motion Matching https://theorangeduck.com/page/learned-motion-matching ; MANN / our mocap https://github.com/sebastianstarke/AI4Animation .

Behaviour foundation models and world models
* Forward-backward representations https://arxiv.org/abs/2103.07945 ; Meta Motivo (FB-CPR) https://arxiv.org/abs/2504.11054 .
* DreamerV3 https://arxiv.org/abs/2301.04104 ; TD-MPC2 https://arxiv.org/abs/2310.16828 (learned latent model + MPPI around a policy prior: the learned-model version of what is done here).

Planning around policies; sampling MPC on legged robots
* LOOP https://arxiv.org/abs/2008.10066 ; Expert Iteration (plan, then distill) https://arxiv.org/abs/1705.08439 .
* MuJoCo MPC / predictive sampling https://arxiv.org/abs/2212.00541 ; whole-body sampling MPC on real legged hardware https://arxiv.org/abs/2409.10469 ;
  reference-free sampling MPC https://arxiv.org/abs/2511.19204 ; safe locomotion learning from an MPC teacher with DAgger https://arxiv.org/abs/2407.11673 .

Not found (searches of Oct 2026): for a quadruped imitating animal mocap with **real motor torque, power
and speed limits**, (a) a sampling planner that uses the imitation policy itself as its proposal and its
tracking reward as the objective to repair *skill transitions*, followed by (b) DAgger distillation of that
planner back into the same policy. Sampling MPC on legged robots plans from scratch or around hand-made
gait references; policy-prior planning (TD-MPC2, LOOP) is shown on benchmark tasks; quadruped AMP/CAMP work
gets transitions by retraining at full RL cost. One simulated robot, N = 10-30: suggestive, not a paper.

## Next step and compute estimate

1. **All transitions at once, at scale (Kaggle T4/P100, ~3-4 h):** port the planner to MJX (16 candidates x
   512 envs = 8k parallel rollouts per plan) and run 5-10 DAgger rounds over the whole transition graph
   (gaits <-> each other, walk/trot -> sit -> lie -> getup, landings at 3-4 m/s, shoves) in one dataset,
   ~2M labelled steps (here: 27k). Expected: the student at the planner's level everywhere, no forgetting
   between transitions. Then a short PPO fine-tune from the student (critic warm-started).
2. **Plan the jump and self-righting brains** with their own env rewards (CPU, ~1-2 h to measure).
3. **Swap the simulator for a learned model** (TD-MPC2-style latent dynamics trained on the ~1e8 steps the
   PPO runs already produced) so the planner could run on a real A1-class robot: GPU, ~1 day.
4. **Data side:** motion matching over the full 30 min of mocap as the reference generator for commanded
   speed/turn, with plan-then-distill absorbing the out-of-distribution references (alone, it would need
   a new tracking run: ~1 GPU-day on MJX).

## Reproduce

From `train/`, with `MUJOCO_GL=osmesa OMP_NUM_THREADS=1` and the scratchpad venv python:

    python research/exam_canter.py --clips canter --n 30 --mode plain --out research/results/canter_plain_n30.json
    python research/exam_canter.py --clips canter --n 30 --mode pps   --out research/results/canter_pps_n30.json
    python research/run_switches.py --mode plain --n 10 --out research/results/sw_plain.json
    python research/run_switches.py --mode pps   --n 10 --out research/results/sw_pps.json
    python research/pps_distill.py collect --tries 30 --seed 20000 --out runs/research/distill_r0.npz
    python research/pps_distill.py fit --data runs/research/distill_r0.npz --w-plan 3 --out runs/research/student_r0
    python research/pps_distill.py collect --tries 40 --seed 21000 --beta 0.3 --student runs/research/student_r0 --anchor-tries 0 --out runs/research/distill_r1.npz
    python research/pps_distill.py fit --data <r0 r1> --init runs/research/student_r0 --epochs 30 --w-plan 5 --out runs/research/student_r1
    python research/pps_distill.py collect --scenario rest --tries 20 --seed 30000 --student runs/research/student_r1 --anchor-tries 0 --out runs/research/rest_r0.npz
    python research/pps_distill.py fit --data <r0 r1 rest_r0> --init runs/research/student_r1 --epochs 30 --w-plan 5 --out runs/research/student_r2
    python research/pps_distill.py collect --scenario rest --tries 12 --seed {31000,32000} --beta 0.3 --student runs/research/student_r2 --anchor-tries 0 --out runs/research/rest_r1_<seed>.npz
    python research/pps_distill.py fit --data <all above> --init runs/research/student_r2 --epochs 30 --w-plan 5 --out runs/research/student_r3
    python research/collect_landing.py --brain runs/research/student_r3 --tries 10 --seed {40000,41000} --beta 0.3 --out runs/research/land_r0_<seed>.npz
    python research/pps_distill.py fit --data <all above> --init runs/research/student_r3 --epochs 30 --w-plan 5 --out runs/research/student_r4
    python research/run_switches.py --mode plain --brain runs/research/student_r4 --seed 5000 --n 10 --only gaits rest rest_still jump20 jump30 jump40 shove --out research/results/heldout_r4.json
    python research/exam_canter.py --brain runs/research/student_r4 --clips canter --n 30 --seed 7000 --out research/results/heldout_canter_r4.json

Per-try numbers: `research/results/*.json`. Data and brains derived from the CC BY-NC 4.0 dog mocap
(dogmotion/SOURCES.md) carry the same non-commercial terms.
