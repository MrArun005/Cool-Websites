# Published quadruped benchmarks vs Bolt: an honest comparison

Compiled 2026-10-29. Every number below was read from the paper's own text (arXiv PDF, converted with `pdftotext`)
unless it is marked otherwise. "not found" means I could not find the number in the source; I did not guess it.
Code and weights availability was checked with `git ls-remote` and a listing of each repository's files.

**Bolt (ours, for reference):** simulated A1-class quadruped in MuJoCo. About 12 kg, 33.5 N·m / 420 W / 21 rad/s
motors, legs about 0.4 m long, base (torso) height about 0.43 m in its "dog" stance (`bolt_env.py`:
`stand height=0.436`). **All Bolt numbers are from simulation only.** Dog mocap source: MANN (Zhang, Starke et al.
2018), the same dataset Peng et al. 2020 used, and the same looping segments (see `train/dogmotion/REPORT.md`).
Tracking reward: Peng 2020 weights (0.5 / 0.05 / 0.2 / 0.15 / 0.1), with identical pose, joint-velocity and paw
terms. The two root terms are different; see section 1.

Legend: **Sim** = simulation result, **Real** = hardware result.

---

## 1. Dog motion-capture imitation

| System (year) | Robot | Skills | Metric (their definition) | Number | Sim/Real | Code / weights | Source |
|---|---|---|---|---|---|---|---|
| Peng et al., "Learning Agile Robotic Locomotion Skills by Imitating Animals" (RSS 2020) | Laikago, 18-DoF (12 actuated). Mass and height not stated in the paper. | Dog Pace, Dog Trot, Dog Backwards Pace/Trot, Dog Spin (from dog mocap); In-Place Steps, Side-Steps, Turn, Hop-Turn, Running Man (artist-animated) | Average normalized return in [0,1] (0 = minimum, 1 = maximum possible return per episode), averaged over 3 seeds. Real: 5 episodes per seed, 15 trials per method. | **Sim (Table V, canonical dynamics), robust / adaptive:** Dog Pace 0.820 / 0.812; Dog Trot 0.727 / 0.718; Dog Spin 0.839 / 0.839; Turn 0.666 / 0.675; Hop-Turn 0.606 / 0.597. No-randomization best: Pace 0.839, Trot 0.752, Spin 0.859. **Real (Table IV, adaptive after AWR adaptation):** Pace 0.827±0.020, Trot 0.593±0.070, Backwards Trot 0.656±0.071, Spin 0.751±0.116, Side-Steps 0.710, Turn 0.606, Hop-Turn 0.518. Real Dog Trot speed 1.08 m/s (manufacturer gait 0.84 m/s). | Sim + Real | Code: yes (google-research/motion_imitation, PyBullet, Laikago and A1). Weights: pretrained `dog_pace`, `dog_trot`, `dog_spin` .zip files in the repo. | https://arxiv.org/abs/2004.00784 · https://github.com/google-research/motion_imitation |
| Escontrela et al., AMP for quadrupeds (2022) | Unitree A1 | Velocity-commanded locomotion in the style of 4.5 s of German Shepherd mocap. Pace-to-canter transition emerges at 1 to 2 m/s. | Tracking of commanded velocity; mechanical cost of transport (COT). **No imitation-accuracy score.** | Sim (Table II): measured speed 0.36 / 0.77 / 1.11 / 1.52 m/s at commanded 0.4 / 0.8 / 1.2 / 1.6. COT 1.07 / 0.93 / 1.02 / 1.12 (AMP), against 1.37 to 1.54 for the hand-designed style reward. Real: canter shown qualitatively (Fig. 2); success rate not found. | Sim (table) + Real (qualitative) | Code: yes (escontra/AMP_for_hardware) | https://arxiv.org/abs/2203.15103 · https://github.com/escontra/AMP_for_hardware |
| Han et al. (Tencent), "Lifelike Agility and Play" (2023/24) | Tencent MAX, 14 kg, 22 N·m continuous / 30 N·m max per actuator | One primitive network tracks all clips of about 0.5 h of Labrador mocap (walk, run, jump, sit) plus about 9 min of stairs. Reused for creeping, hurdles, blocks, stairs and a chase-tag game. | Imitation reward curves only (Fig. 2B). Real-world success over 10 repetitions per setting (Fig. 3E). | Per-clip tracking numbers: not found (figure only). Real hurdles were 10 cm and 15 cm high. They state 100% success in simulation "because the reward and convergence conditions guarantee this". Real success values: figure only, not in the text. | Sim + Real | Code: yes. Pretrained `.model` files and retargeted dog mocap are in the repo. | https://arxiv.org/abs/2308.15143 · https://github.com/Tencent-RoboticsX/lifelike-agility-and-play |
| Li et al. (Tencent), "Terrain-Adaptive Locomotion ... by Imitating Animals" (2023) | Tencent Max | Dog mocap with terrain data; stairs, slopes, blocks | Average return (unnormalized) over 1000 episodes | Sim, plane: 323.88 ± 30.16 (unnormalized, so not comparable). Real: 1.1 m/s stair climbing. | Sim + Real | not found | https://arxiv.org/abs/2308.03273 |
| Klipfel et al., "Single Policy for Diverse Behaviors ... Scalable Motion Imitation" (2023) | Unitree A1 (sim) | 701 retargeted dog-derived clips: stand, step, pace, trot, gallop, jump, turns, **sit, lie** | Clips tracked without early termination | Tracks all 701 training clips. About 90% of 47 held-out "random mixed" clips. | **Sim only** | not checked | https://arxiv.org/abs/2303.15331 |
| Bohez et al. (DeepMind), "Imitate and Repurpose" (2022) | ANYmal B300, about 33 kg, 70 cm tall at rest | Dog mocap (mostly walking/turning), zero-shot imitation, then reuse for walking and dribbling | Qualitative trajectory overlays (Fig. 3) | Numeric tracking error: not found | Sim + Real | not found | https://arxiv.org/abs/2203.17138 |
| Kang et al. (ETH Zurich / Korea Univ.), "Walk like Dogs" (2025) | Unitree Go2 | Steerable gaits from unlabeled dog mocap (sit-pace-sit, spin, gallop tested) | Learning curves and penetration artifacts (figures) | Tabulated numbers: not found | Sim + Real | not found | https://arxiv.org/abs/2507.00677 |
| DecAP (Sood et al., 2023) | (torque-control legged robots) | Imitation of a position-control policy, **not dog mocap** | Not comparable | n/a | | Code: yes (marmotlab/decaying_action_priors) | https://arxiv.org/abs/2310.05714 |
| Disney dog-like quadruped work | | | | **not found.** No Disney-authored dog-mocap quadruped paper with numbers turned up. | | | |
| **Bolt** | sim A1-class, 12 kg | walk, trot, pace, canter, turns, sit, lie, get-up (MANN dog mocap, Peng's segments) | Mean per-step tracking reward in [0,1], 0 for steps lost after a fall | 91 to 97% per clip; canter 76% with 0/90 falls | **Sim only** | (ours) | `train/imitate_env.py`, `train/dogmotion/REPORT.md` |

**Metric caveat (Peng vs Bolt).** Bolt's pose (exp −5Σ), joint-velocity (exp −0.1Σ) and paw (exp −40Σ) terms and
all five weights match Peng's Eq. 4 to 7 exactly. The root terms do not match:

- Peng Eq. 8 penalizes the **global** root position and **full root rotation** (exp(−20‖x̂−x‖² − 10‖q̂−q‖²)).
  Bolt uses **only height and tilt**, so global drift and heading error are free.
- Peng Eq. 9 uses global root linear and angular velocity. Bolt uses body-frame velocity.
- Peng's robot is Laikago, simulated in PyBullet. Episode length, and whether evaluation returns are discounted, were
  not found in the paper (PPO γ = 0.95; AWR γ = 1.0).

So Bolt's 91 to 97% is scored under a more lenient root term, and **cannot be claimed as better than Peng's
0.72 to 0.86 sim numbers** until Bolt is re-scored with Eq. 8 and 9 (see section 9).

---

## 2. Barkour (agility course)

| Item | Value | Source |
|---|---|---|
| Robot | Custom Google quadruped, **11.5 kg**, 220 mm upper and 190 mm lower limbs, hip-to-hip 380 mm × 290 mm | https://arxiv.org/abs/2305.14654 (Sec. V) |
| Course | Start/end pause tables, weave poles, 30° A-frame, broad jump (jump board **0.5 m long**, flush with the floor; a horizontal long jump, not a bar). Fits in about 5 m × 5 m. | same, Fig. 2, App. B |
| Score | R = 1.0 − 0.01 × (full seconds over t_allotted) − 0.1 × (failed or skipped obstacles). t_allotted = Σ d_obstacle / 1.69 m/s = 18 m / 1.69 = **10.64 s** | same, Eq. 1–2, Table I |
| Specialists + high-level navigator (Real) | **0.77 ± 0.064**, 24.6 ± 1.1 s, 0.74 m/s mean over 71 runs (6 falls). Best single run **0.91** (20.4 s). Broad-jump success **38%** (run-up from about 2 m, 2.3 m/s at take-off). Weave poles and A-frame 100%. | same, Table III–IV |
| Generalist Locomotion-Transformer, distilled (Real) | **0.73 ± 0.062**, 25.8 ± 1.6 s, over 19 trials. Smoother transitions than switching specialists (qualitative). | same, Table III |
| Small dogs (Real) | 1.00, 9.02 ± 0.65 s (3.2 kg and 4 kg dogs) | same |
| Open? | Barkour robot models are open in MuJoCo Menagerie (`google_barkour_v0`, `google_barkour_vb`). Policy code and weights from the paper: not found. | https://github.com/google-deepmind/mujoco_menagerie |

---

## 3. Parkour and jumping (obstacle size relative to robot size)

| System (year) | Robot | Skill | Metric | Number | Sim/Real | Code / weights | Source |
|---|---|---|---|---|---|---|---|
| Zhuang et al., Robot Parkour Learning (CoRL 2023) | Unitree A1 and Go1 (onboard depth) | Climb onto a box, leap a gap, crawl, tilt | Success over 10 trials per difficulty (Real). Success over 100 trials × 3 seeds (Sim). | **Real A1:** climb **0.40 m (1.53× robot height)** 80%; leap **0.60 m gap (1.5× robot length)** 80%; crawl 0.2 m 90%; tilt 0.28 m slit. **Sim (harder test set):** climb 86%, leap 80%, crawl 100%, tilt 73%. | Sim + Real | Code: yes (ZiwenZhuang/parkour). Trained policy weights: not found in the repo tree. | https://arxiv.org/abs/2309.05665 · https://github.com/ZiwenZhuang/parkour |
| Cheng et al., Extreme Parkour (2023) | Unitree A1. Standing thigh-joint height 26 cm, body length 40 cm. | High jump **onto** a box, long jump, ramps, handstand | Real: 5 trials per terrain per difficulty (Fig. 7). Sim: mean waypoints reached, normalized [0,1], 256 robots. | **Real:** climbs a **0.5 m box (about 2× hip height)**; long jump **0.8 m (2× body length)**. Per-difficulty real success: figure only, not in the text. **Sim:** hurdle 0.99 ± 0.05, gap 0.96, total 0.98 (waypoints). | Sim + Real | Code: yes (chengxuxin/extreme-parkour). Weights: not found in the repo tree. | https://arxiv.org/abs/2309.14341 · https://github.com/chengxuxin/extreme-parkour |
| Hoeller et al., ANYmal Parkour (2023/24) | ANYmal D, **about 55 kg**, 85 N·m actuators | Jump gaps, climb up and down, crouch, walk, plus a navigation module that switches skills | Success vs difficulty (Fig. 4F, sim). Course completion (Table I). | Trained for gaps up to 1 m and boxes up to 1 m. **Real:** climbs onto a **0.9 m** platform, up to 2 m/s. Sim course success with the learned navigator: **98.2 / 96.3 / 97.6%** against 95.3 / 60.9 / 75.3% for a hand-coded trajectory. Extreme Parkour's Table 1 lists it as climb 2× height, gap 1.5× length. | Sim + Real | not found | https://arxiv.org/abs/2306.14874 |
| Yang et al., CAJun (CoRL 2023) | Unitree Go1 | Continuous jumping across gaps | Max gap crossed | **0.70 m (1.4 body lengths)**, Real. A 60 cm mat was cleared without foot contact. Success rate: not found. | Real | Code: yes (yxyang/cajun) | https://arxiv.org/abs/2306.09557 |
| Chane-Sane et al., SoloParkour (CoRL 2024) | Solo-12. 26 cm standing height, 45 cm length. Mass: not found. | Step up, leap, crawl | Real success, 2 seeds × 10 trials | Step **0.40 m (1.5× height)** 85%; leap 0.35 m gap 70%; crawl 0.20 m 100% | Real | Code: yes (Gepetto/SoloParkour) | https://arxiv.org/abs/2409.13678 |
| Lifelike agility (Tencent MAX) | 14 kg | **Clearing hurdles (a bar)** | Real success, 10 reps | Hurdles of **10 cm and 15 cm**. Rates: figure only. | Real | yes | https://arxiv.org/abs/2308.15143 |
| Barkour jumping policy | 11.5 kg | Broad jump 0.5 m, horizontal | Real success | 38% | Real | not found | https://arxiv.org/abs/2305.14654 |
| **Bolt** | sim, 12 kg, base about 0.43 m | **Clearing a real bar** after a run-up | Clean clear rate | 50 cm (about 1.16× base height) about 100% from still legs at 2 m/s; 60 cm (about 1.4×) up to 70% | **Sim only** | (ours) | |

**Caveat.** Most published "climb" and "high jump" numbers are box **climb-ups**: the robot lands its front legs on
top and pulls itself up. They are not bar clearances, and they are normalized by hip or thigh-joint height, not by
torso height. Bolt's task is to fly over a bar without touching it. The only published bar-clearing quadruped
results I found on hardware are Tencent MAX's 10 and 15 cm hurdles. No same-task comparison exists.

---

## 4. Fall recovery

| System (year) | Robot | Protocol | Number | Sim/Real | Code | Source |
|---|---|---|---|---|---|---|
| Lee, Hwangbo & Hutter, "Robust Recovery Controller" (2019) | ANYmal (about 32 kg per Hwangbo 2019) | (a) 50 arbitrary lying configurations; (b) 50 runs of disturbances while walking or standing. More than 100 falls in total. | **>97%** (3 failures). In (a), upright within 5 s in all 50. Failures were caused by joint angle ≥ 2π. | Real | not found | https://arxiv.org/abs/1901.07517 |
| Hwangbo et al., Science Robotics (2019) | ANYmal | Random initial fall configurations | Success "improved to 100%". Trial count: not found. | Real | not found | https://arxiv.org/abs/1901.08652 |
| Smith et al., "Legged Robots that Keep on Learning" (2021) | Unitree A1 | Learned reset policy between fine-tuning trials, across several terrains | "Over all experiments, the recovery policy was 100% successful". Count: not found. Faster than Unitree's built-in roll-over. | Real | not checked | https://arxiv.org/abs/2110.05457 |
| Lu et al., Adaptive Fall Recovery (AFR) (2024) | Unitree Go1 (Isaac Gym, Gazebo) | Prone starts on 8 complex terrains | Slope 98%, discrete obstacles 86%, stairs 76%, single gaps 62%, air beams 60%, beams 64%, sparse stones 92%, dense stones 66%. Recovery in 1.1 to 2.6 s. | **Sim** | not checked | https://arxiv.org/abs/2412.16924 |
| FR-Net (RA-L 2025) | Unitree Go2 (+ sim Spot, Lite3, Aliengo, Go1) | Supine starts, 10 difficulty levels, 2048 robots. Success = stable and upright within 5 s with < 1 m displacement. | Above 90% at levels 1 to 3. Level 10: rough terrain 69.5%, beams 64.1%, stairs and slopes ≥ 58%. Real: 11 scenarios, qualitative only (no rate found). | Sim (rates) + Real (qualitative) | not checked | https://arxiv.org/abs/2509.11504 |
| Rudin et al., "Learning to Walk in Minutes" (2021) | ANYmal C | Not a recovery paper. Pushes up to ±1 m/s every 10 s during training; terrain success rates only. | Fall-recovery rate: not found | Sim | yes (legged_gym) | https://arxiv.org/abs/2109.11978 |
| **Bolt** | sim, 12 kg | 6 fall types (Bolt's own test) | about 85 to 90% | **Sim only** | | |

---

## 5. Push, disturbance and latency robustness

| System (year) | Robot | Test | Number | Sim/Real | Code / weights | Source |
|---|---|---|---|---|---|---|
| DreamWaQ (Nahrendra et al., ICRA 2023) | Unitree A1 | Random velocity pushes along x, y and z every 1 s, uniform in [−v, v], v raised until fall. Survival = % of a 30-min random walk survived. Trained with 0 to 15 ms system-delay randomization. | Max push **1.121 ± 0.164 m/s**, survival **95.23 ± 1.61%** (baseline 0.511 m/s, 20.5%) | **Sim** | Official code: not found | https://arxiv.org/abs/2301.10602 |
| Walk These Ways (Margolis & Agrawal, CoRL 2022) | Unitree Go1 | They identified about **20 ms** latency on the Go1 and modeled it as a constant action delay in sim. Shoves, curbs and bushes shown qualitatively. Crawled under a 22 cm bar. | No push or latency sweep numbers found | Real (qualitative) | Code + pretrained weights: yes (Improbable-AI/walk-these-ways, `checkpoints/*.jit`) | https://arxiv.org/abs/2212.03238 |
| RMA (Kumar et al., RSS 2021) | Unitree A1 (12 kg per paper) | Payload: carries up to **12 kg (100% of body weight)**. Step-down 15 cm at 80%. Memory foam 100%. | Latency or push sweep: not found | Real + Sim | Official code: not found | https://arxiv.org/abs/2107.04034 |
| Extreme Parkour | A1 | Fixed depth latency 80 ms, proprioception latency 16 ms (enforced) | (a setting, not a sweep) | Real | yes | https://arxiv.org/abs/2309.14341 |
| **Bolt** | sim | Control delay swept 0 to 40 ms | A separate brain: 0% falls at 20 ms | **Sim only** | | |

No published fall-rate-vs-latency **sweep** for A1-class robots was found. One web search surfaced MMDR
(https://arxiv.org/abs/2109.14549), which randomizes delay; I did not open it, and the search snippet shows no
sweep.

---

## 6. Multi-skill single policies and switching

| System (year) | Robot | # skills | Switching metric | Number | Sim/Real | Source |
|---|---|---|---|---|---|---|
| Barkour Locomotion-Transformer (2023) | 11.5 kg custom | Distilled from 3 specialists (walk/turn, slope, jump) | Course score (no per-switch rate) | 0.73 (generalist) vs 0.77 (specialists + switching) | Real | https://arxiv.org/abs/2305.14654 |
| ANYmal Parkour (2023/24) | ANYmal D | 5 skills + navigation policy | Course success (no per-switch rate) | 96.3 to 98.2%. Switches "smooth and unnoticeable" (qualitative). | Sim (rates) + Real | https://arxiv.org/abs/2306.14874 |
| Robot Parkour Learning | A1/Go1 | 5 skills distilled into 1 vision policy | Per-skill success | see section 3 | Sim + Real | https://arxiv.org/abs/2309.05665 |
| Lifelike Agility and Play (Tencent) | MAX 14 kg | Primitive (all mocap) → environmental (creep, hurdle, block, stair) → strategic (chase tag) | Elo between checkpoints; per-switch rate not found | n/a | Sim + Real | https://arxiv.org/abs/2308.15143 |
| Klipfel et al. 2023 | A1 | 701 clips incl. sit and lie in 1 policy | Clip completion | 100% of training clips; about 90% of 47 held-out mixed clips | Sim | https://arxiv.org/abs/2303.15331 |
| "Teaching a Robot Dog New Tricks" (arXiv, Oct 2026) | Unitree B1 | 22 tasks distilled from 8 teachers | **Survival rate after a mid-episode task switch (A→B)**, 0 to 300 N pushes, 40 trials per pair | Student **99.05%** (target switch 99.47%). Teacher-router baseline **80.82%** (91.15%). | Sim (rates) + Real (deployment) | https://arxiv.org/abs/2610.10601 |
| Lee et al. 2019 | ANYmal | 3 behaviors (self-right, stand-up, walk) + selector | Recovery success | >97% | Real | https://arxiv.org/abs/1901.07517 |
| **Bolt MasterMind** | sim | about 10 commands (gaits, turns, sit, lie, stand, jump, get-up) | Bolt's own per-switch success (each switch must reach its target, e.g. clean jump or pose reached, not just survive) | 92.4% | **Sim only** | `train/mastermind/REPORT.md` |

---

## 7. Where Bolt stands, honestly

1. **Every Bolt number is simulation-only.** Every headline system above except Klipfel 2023 also reports hardware
   results. A sim number cannot be compared with a real one, so none of Bolt's results can be claimed to match or beat
   any real-robot result above.
2. **Dog imitation (closest to a like-for-like test).** Bolt uses the same mocap dataset (MANN), the same segments
   and the same pose, velocity and paw reward terms as Peng 2020. Peng's **sim** scores (Table V) are 0.72 to 0.75
   for trot, 0.81 to 0.84 for pace and 0.84 to 0.86 for spin. Bolt's 91 to 97% is higher. However, Bolt's root
   terms ignore global position and heading drift, and the robot, simulator and episode length differ. **Supported
   claim:** "on a Peng-style reward with a relaxed root term, Bolt's sim tracking is in the same range as or above
   Peng's sim numbers." **Not supported:** "Bolt imitates the dog better than Peng/Laikago." Peng's real-robot numbers
   (trot 0.59, pace 0.83) are a different regime altogether.
3. **Canter.** No published dog-mocap paper found reports a canter tracking score. AMP shows an emergent canter on a
   real A1 only qualitatively. Bolt's canter (76%, 0/90 falls) has no published comparator.
4. **Jumping.** Bolt clears a 50 cm bar (about 1.16× its 0.43 m base height) in sim. Published A1-class hardware
   results are box **climbs** (0.40 to 0.50 m, 1.5 to 2× hip height), gap **leaps** (0.6 to 0.8 m), a Barkour
   **broad jump** (0.5 m, 38%), or low **hurdles** (MAX, 10 to 15 cm). These are different tasks with different
   normalizations. No claim can be made. One plain note: Bolt's base height is much taller than A1's 26 cm thigh
   height. Normalized by base height, Bolt's 50 cm is about 1.16×, not 2×.
5. **Fall recovery.** Bolt's 85 to 90% (sim, 6 fall types) is below ANYmal's >97% on real hardware (Lee 2019) and
   the 100% reported for A1 resets (Smith 2021). It is in the range of the Go1/Go2 complex-terrain sim numbers (60
   to 98%, AFR and FR-Net), but those are on rough terrain and use different success definitions. No direct claim.
6. **Robustness.** Being fall-free at 20 ms delay matches the latency real Go1 policies are trained for (WTW: about
   20 ms), but no paper reports a comparable sweep. Bolt has no push-test number in DreamWaQ's format.
7. **Switching.** Bolt's 92.4% uses a strict "target reached" success. The only published per-switch number found
   (Oct 2026, B1) is a **survival** rate (99.05% student vs 80.82% router). The metrics differ, so no claim can be
   made. The qualitative finding agrees with Bolt's own: distilled single policies switch better than teacher
   routers. Barkour found the opposite on score (generalist 0.73 < specialists 0.77) but smoother transitions.

---

## 8. Open code and data most useful to us

- Peng: `google-research/motion_imitation` has the same dog clips and pretrained Laikago policies (PyBullet). It
  can be used to re-run their evaluation.
- Lifelike (Tencent): code, `.model` weights and retargeted dog mocap.
- Walk These Ways: Go1 code + pretrained `.jit` weights.
- Robot Parkour Learning, Extreme Parkour, CAJun, SoloParkour, AMP_for_hardware, DecAP: code.
- MuJoCo Menagerie: `google_barkour_v0`, `google_barkour_vb`, `unitree_a1`, `unitree_go1`, `unitree_go2` models.

---

## 9. Tests we would need to run to compare on their terms

1. **Peng re-score (cheapest, most direct).** Re-score Bolt's pace, trot and turn/spin rollouts with Peng's exact
   Eq. 4 to 9, including global root position, full root rotation and global root velocity. Use undiscounted
   per-step mean over a fixed episode length, 3 seeds × 5 episodes, with canonical (non-randomized) dynamics. Report
   next to Table V. Better still, run Peng's pretrained `dog_pace`/`dog_trot`/`dog_spin` policies through the same
   scorer, or port the Laikago model to MuJoCo, so both robots are scored by one script.
2. **Barkour in MuJoCo.** Build the course to App. B dimensions (weave poles, 30° A-frame, 0.5 m broad jump, pause
   tables). Score with R = 1 − 0.01·⌊t − 10.64⌋₊ − 0.1·failures. Report the mean ± std over at least 70 runs and the
   broad-jump success rate. Label it clearly as sim against their real hardware.
3. **Parkour-style jump tests.** Run box climb-up and gap leap at increasing difficulty, 10 trials per difficulty
   (Robot Parkour Learning protocol), plus a 0.5 m broad jump from 2 m with no board contact. Normalize by **thigh
   or hip-joint height** and body length, as those papers do.
4. **Fall recovery, Lee 2019 protocol.** Run 50 random lying configurations plus 50 falls induced by disturbances
   while walking or standing. Success = upright within 5 s. Also run FR-Net's definition (stable upright ≤ 5 s,
   < 1 m displacement).
5. **DreamWaQ push test.** Apply random velocity kicks in x, y and z every 1 s, uniform in [−v, v]. Increase v to
   find the max survivable push, and report survival % over a 30-min random walk.
6. **Switching, Oct-2026 protocol.** Report **survival after switch** (in addition to our strict success), with
   0 to 300 N pushes of 0.1 to 0.2 s and 40 trials per A→B pair, so both metrics are on record.
7. **Hardware.** Any "matches published" claim ultimately needs a real robot. Until then every row stays "sim vs
   their sim" at best.

### Sources consulted (all primary PDFs from arxiv.org unless stated)
2004.00784, 2203.15103, 2305.14654, 2309.05665, 2309.14341, 2306.14874, 1901.07517, 1901.08652, 2110.05457,
2109.11978, 2212.03238, 2301.10602, 2107.04034, 2308.15143, 2308.03273, 2303.15331, 2203.17138, 2507.00677,
2306.09557, 2310.05714 (abstract only), 2409.13678, 2412.16924, 2509.11504, 2610.10601. GitHub repositories as
linked above (existence and file lists checked).
