# Bolt on a real Unitree Go2 EDU: the plan (started 31 Oct 2026)

Goal: the software we train in simulation runs on a real Go2 **as is**, through the vendor's own control interface,
so the day a Go2 EDU arrives it walks within hours, not weeks. Everything up to the hardware day is done without a
robot, against Unitree's own simulator (`unitree_mujoco`), which speaks exactly the same interface (`unitree_sdk2`,
DDS topics `rt/lowstate` / `rt/lowcmd`) as the real robot. Moving from that simulator to the robot changes one
setting: the network interface.

Brain to deploy first: `models/bolt-go2-gpu-489M` (dog motion: walk, trot, pace, canter, turns, sit, lie, getup,
steering; 0-40 ms delay trained; falls 1.9% in our exam). Jumps, recovery and MasterMind follow the same path later.

The older notes in `train/research/real_robot_port.md` (MasterMind on an A1/Go2) still hold; this file is the
step-by-step version for the Go2 motion brain.

## Where we stand (honest, 31 Oct)

| piece | status |
|---|---|
| Robot body | none yet: a Go2 **EDU** is needed (other Go2 models block low-level motor control) |
| Trained brain for the Go2 body | done for motion (489M); jumps, recovery training; MasterMind not started |
| Bridge: robot sensors -> brain -> motor commands | **started**: `controller.py` builds the inputs and targets exactly as training (`test_deploy.py`: inputs 1e-7, targets 1e-15, clip switch identical); SDK side not yet |
| Safety layer | **not started** (phase 1) |
| Rehearsal in the vendor simulator | **started**: passes in lockstep (5/5, 41 s plan) and 2/5 full runs in real time through DDS (see Log) |
| Inputs the real robot cannot measure | **open**: the brain reads body speed (3) and body height (1), which no Go2 sensor gives (phase 3) |
| Real-world tuning | needs the robot (phase 4-5) |

## Phase 1: the bridge (no hardware; ~3-5 days)

All code in `train/deploy/`, plain Python + numpy (no torch, no JAX), so it runs on the Go2's own Jetson or on a
laptop on wired Ethernet.

1. **Toolchain.** `unitree_sdk2_python` (+ CycloneDDS) and `unitree_mujoco` (its Python simulator) installed and a
   LowState -> LowCmd loop running against it. Gate: we read the simulated robot's joints and IMU and can hold it in
   a stand with plain PD.
2. **Joint map.** Our order FL, FR, RL, RR x (hip, thigh, calf) <-> SDK order FR, FL, RR, RL
   (`robot_bodies.BODIES["go2"]["sdk_order"]`). Signs and zero offsets checked by commanding one joint at a time.
   Gate: a commanded pose in our order shows up as the same pose in the vendor simulator.
3. **Observation builder** (`obs.py`): the 75 inputs exactly as `imitate_env.ImitateEnv._obs` builds them, from
   LowState: gravity from the IMU quaternion, gyro, joint angles and speeds minus the stance, the previous action,
   clip phase and one-hot, and the reference motion from the clip files shipped with the brain. Body speed and height
   come from an estimator (phase 3); first version: leg-kinematics height and a simple velocity filter. Gate: fed the
   same state, `obs.py` and `ImitateEnv._obs` agree to 1e-6 (a unit test on recorded states).
4. **Brain runner** (`brain.py`): `params.pkl` -> numpy MLP (normalize, clip +-10, ELU, mean), ~1 ms. Gate: the same
   action as `mjx_imitate.MjxImitatePolicy` on 1000 recorded inputs.
5. **Control loop** (`run_go2.py`): read state -> build inputs -> brain -> targets = stance + scale x action -> clamp
   -> LowCmd (q target, dq 0, kp 40, kd 1, tau 0, CRC) every 20 ms; the brain's latest targets repeated at 500 Hz.
   Commands: stand / walk / trot / canter / turn / sit / lie / getup, steering rate, from the keyboard or a gamepad.
6. **Safety layer** (in the loop, never in the brain):
   * startup: from the robot's lying pose, blend over 2-3 s to the stance, then hand over to the brain;
   * joint targets 0.05-0.1 rad inside the hardware limits; torque below the rated peak (hip/thigh 23.7, calf 45.4 N·m);
   * damping mode (kp 0, kd 1-3) on: tilt past 60 degrees, no new brain output for 60 ms, joint speed over rated,
     motor over temperature, estimator diverging, or the e-stop key;
   * sport mode released first (the motion switcher), or the robot's own controller fights ours.
7. **Logger**: every 50 Hz frame (inputs, outputs, state, timing) to a file, replayable through our simulator.

## Phase 2: dress rehearsal in the vendor simulator (~2-3 days)

Run `run_go2.py` unchanged against `unitree_mujoco`. It is a different simulator from ours (its own model files,
contact settings and control timing), so passing here is the first real sim-to-sim test.

Exam (same pass lines as our own exam, 10 tries each): stand 30 s; walk, trot, canter 10 s each; turns; walk -> sit,
trot -> lie, getup; steering left/right; with 0 and 20 ms added delay. Pass: no falls in gaits, sit/lie reached,
tracking within 0.05 of our own exam. Every gap found here (gains, damping, timing, joint limits) is fixed in our
training setup and the brain retrained, not patched in the bridge.

## Phase 3: close the input gap (GPU time; ~1-2 days of compute)

The brain reads body speed and height, which a real Go2 only gets from an estimator. Two routes, test both in
phase 2 and keep the better:

* **A. Estimator + noise.** Body height from the legs in contact (kinematics), speed from a Kalman filter on IMU +
  leg kinematics (+ foot contact). Retrain (fine-tune from 489M) with the estimator's realistic error injected.
* **B. Teacher -> student (the stronger, standard route).** The 489M brain becomes the teacher; a student that sees
  only what the robot measures (IMU, joints, previous actions, a short history) learns to act like it. This is the
  "privileged teacher" method the labs use; it also removes the need for the estimator.

Gate: the chosen brain passes the phase 2 exam with only real sensors.

## Phase 4: hardware day (1-3 days, needs a Go2 EDU)

1. Unbox, charge, update firmware, wired Ethernet (192.168.123.x), confirm low-level access (EDU).
2. Read-only: run the loop with the motors off; check the inputs look like the simulator's (IMU, joints, timing).
3. Hanging in a harness, feet just off the floor: stand, then slow walk in the air. Check motor directions again.
4. On a flat high-grip mat at 80% gains: stand, then walk, trot, sit, lie. E-stop in hand, a second person watching.
5. Full gains; then steering; record every run (video + logs).

## Phase 5: make it reliable (2-6 weeks)

Measure the real delay, motor response and friction from the logs; put them into our simulator (the "actuator
network" step, from motor logs); retrain; repeat. Then, one at a time and each with its own exam: getting up after
falls, the companion mind on the Jetson (speech, mood), jumps (needs a depth camera or the LiDAR to see the bar),
MasterMind switching.

## Order of work and gates

| step | needs | done when |
|---|---|---|
| 1.1 toolchain | network, CPU | stand held by plain PD in unitree_mujoco |
| 1.2 joint map | 1.1 | pose test passes |
| 1.3 obs builder | 1.2 | matches ImitateEnv to 1e-6: **PASS** (31 Oct, `test_deploy.py`) |
| 1.4 brain runner | - | matches MjxImitatePolicy: **PASS** (the controller uses it directly) |
| 1.5-1.7 loop, safety, logger | 1.1-1.4 | runs 10 min without a fault in unitree_mujoco |
| 2 rehearsal | phase 1 | exam passes, or gaps listed and fed back |
| 3 input gap | GPU | brain with real sensors only passes phase 2 |
| 4 hardware day | a Go2 EDU | walks on the floor |
| 5 reliability | 4 | 30 min of mixed commands without a fall |

All results are simulation until phase 4. Nothing here calls an external AI service.

## Log

* 31 Oct: `controller.py` + `test_deploy.py` pass (480 steps over 8 clips, a walk -> trot switch, steering).
  Found while reading the vendor files: unitree_mujoco's Go2 model has joint damping 0.1 (+ armature 0.01, friction
  loss 0.2); ours trains with Menagerie's 2.0. A real mismatch to settle in phase 2 (our damping A/B on the CPU
  pilots said the choice matters). Installing the vendor SDK (cyclonedds + unitree_sdk2_python) in this cloud session
  needs the owner's OK (installing downloaded code is gated); step 1.1 waits on that.
* 31 Oct (late): toolchain up (CycloneDDS 0.10.x built from source, unitree_sdk2_python, unitree_mujoco; owner's OK).
  `sim_vendor.py` (Unitree's simulator, headless), `run_go2.py` (the robot program), `lockstep_vendor.py` (gap finder).
  * First runs fell after 2-5 s of trot. Not physics: **Unitree's go2 scene has two 8 cm bumps and a staircase 1.2-3.4 m
    ahead**, and the dog trotted into them. Ruled out on the way, each by an A/B: joint damping 0.1 vs 2.0, physics step
    5 vs 2 ms, Euler vs implicitfast, foot friction 0.4 vs 0.8, servo vs torque motors, shin collisions. Field by field
    the vendor go2.xml and our Menagerie model are identical in bodies, masses, inertias and joints; only the joint
    damping and the geom friction (0.4 vs 0.6/0.8) differ. `--flat` (default) switches the obstacles off.
  * Lockstep on Unitree's model, flat floor, plan trot 10 s, sit 5, lie 6, getup 6, walk 8, canter 6 (41 s):
    **0/5 falls** at the vendor's own settings (damping 0.1, 5 ms step), at 2 ms, and at damping 2.0.
  * Real time through DDS (rt/lowstate, rt/lowcmd at 500 Hz, brain at 50 Hz, true velocity/height): 2/5 runs complete
    the 41 s plan; 1 stopped by the safety layer on a 63 ms stale state (the Python simulator stalls: it runs at
    0.95x real time on this 4-core cloud VM); 2 fell in the canter (the last 6 s), which does not fall in lockstep.
    Brain loop timing: median 20.1 ms, p95 21.3 ms. Next: run the simulator on a dedicated core (or the C++
    unitree_mujoco) to separate VM stalls from real canter sensitivity; then phase 3 (no true velocity/height).
  * Harness lesson: never start a simulator while another still publishes on the same DDS domain.
* 1 Nov (night): real-time runs with the simulator and the controller pinned to separate cores were worse (1/5
  complete): the Python vendor simulator cannot hold real time on this VM, so real-time results here measure the VM.
  Timing sensitivity measured instead in lockstep (`lockstep_vendor.py --delay-ms LO HI --skip-p P`, vendor settings,
  41 s plan, 5 tries): delay 0-10 ms **0/5** falls; 0-19 ms 1/5; **5% of brain steps missed 3/5**; 5-19 ms + 10%
  missed 3/5. The brain handles delay (it was trained with 0-40 ms) but **not missed control steps** (motors holding
  old targets for 40 ms while the brain's "previous action" input says otherwise). Two consequences:
  (1) on the robot the 50 Hz loop must never miss a step: brain on the Jetson, loop at real-time priority, watchdog
  logs every late step; (2) the next Go2 training run adds missed steps (mastermind/env.py's packet-drop model,
  drop 0-10%) so the brain learns to tolerate them. Same as the A1 port plan's packet-drop line.
* 1 Nov (night): phase 3 first measurements (`estimator.py`, `lockstep_vendor.py --estimate / --vel-ref`; vendor
  settings, flat floor, 41 s plan): **height from leg kinematics is good (mean error 0.3 cm)**; body velocity from leg
  odometry is not: 0.18-0.29 m/s mean forward error (differencing paw positions beats the Jacobian version: the paw
  spheres roll and slip). Falls: true velocity 0/5; **leg-odometry velocity 4/10** (mostly canter); velocity = the
  clip's reference 3/5 (trot). The 489M brain leans on its velocity input. Decision: route B -- retrain (fine-tune
  from 489M) with the velocity input taken away, as most published Go2/A1 policies run (and keep the kinematic height,
  which works). Needs GPU time (Kaggle T4/P100 is fine: no TF32).
* 1 Nov (night): `mjx_imitate.py --blind-vel --drop-p` (8da700c). Kaggle GPU run **bolt-go2-blind** started ~02:05
  IST: from 489M, same recipe + no velocity inputs + 5% lost commands, 3.5 h (warm dataset bolt-go2-489m). Gate: the
  lockstep rehearsal with kinematic height and no velocity, vendor settings, 41 s plan, 0/5 falls (489M with the true
  velocity: 0/5; with leg odometry: 4/10).
* 1 Nov 02:00-02:55: RunPod freed a GPU; run f on the 4090 (last ~$1.15): from 489M, --blind-vel --drop-p 0.05,
  45M steps. Training eval 0 / 22.6 / 45.2M: reward 1.205 / 1.258 / 1.288, falls (with 5% lost commands) 35 / 27 / 23%,
  canter 0.76 / 0.82 / 0.86. Gate (lockstep, vendor settings, kinematic height, 41 s plan, 10 tries): **3/10 falls**
  (all trot) vs 489M with leg-odometry velocity 4/10 -- better direction, not passed, not saved. Pod stopped, $0.17
  left. The Kaggle run (bolt-go2-blind, same recipe, 3.5 h) continues it.
* 1 Nov 03:15: run_go2.py gets `--interactive` (type a clip, "steer X", "stop"), the kinematic estimator in place of
  the --truth shortcut (height from the legs; velocity from leg odometry, ignored by a --blind-vel brain), and a late-
  step watchdog (counts steps > 25 ms; damping mode after 3 in a row > 40 ms). Test in Unitree's simulator through DDS,
  real time, run f's blind brain, no truth: trot -> walk -> steer 0.4 -> sit -> lie -> stop by typed commands, 23 s,
  no fall, 0 of 1152 steps late; an unknown command is refused with the list of valid ones.
* 1 Nov 06:15: **phase 3 gate PASSED.** Kaggle bolt-go2-blind (103M blind steps from 489M, T4 3.5 h) ->
  models/bolt-go2-blind-592M: rehearsal with robot-only inputs **0/10 falls** (489M 4/10, run f 3/10); with 0-19 ms
  delay 1/10; with 5% missed steps 3/10. Cost in our own exam: mean 0.809 vs 0.848, canter 0.485 vs 0.628, trot ->
  sit falls again (4/4). The real-robot candidate; a next blind piece continues it.
* 1 Nov 06:35: `HARDWARE_DAY.md`, the phase 4 checklist (connect read-only, harness, floor at 0.8 gains, full gains,
  logs back into the simulator), each step with its stop condition. `read_go2.py` (read-only state printer for step 1) tested on
  Unitree's simulator: 200 Hz, largest gap 9.9 ms, left/right hips mirrored as expected.
* 1 Nov 07:10: `replay_log.py` (phase 5 tool): replays a run_go2.py log's targets open-loop in Unitree's model from
  the logged state every 0.2-0.5 s and reports the joint gap per joint type. On the interactive DDS log (Unitree's
  simulator, damping 0.1): hip / thigh / calf 0.019 / 0.034 / 0.045 rad after 0.5 s; replayed with the wrong damping
  (2.0) the thigh gap is 0.091 -- it finds a motor-model mismatch. On a real Go2's logs this is how the simulator
  gets fixed (damping, delay, motor strength), joint type by joint type.
* 1 Nov 08:55: `lockstep_recovery.py`: fall recovery on Unitree's go2.xml (damping 0.1, 5 ms), robot-only inputs
  (kinematic height; self-right velocity from leg odometry, zero, or true): models/bolt-go2-selfright-6.9M rolls it
  belly-down, then bolt-go2-blind-592M's getup clip stands it up -- **18/18** (side 6/6, back 6/6, tipped 6/6) in all
  three velocity variants. Next: the hand-over in run_go2.py (tilt > 60 deg -> self-right -> getup -> resume), which
  needs the self-right brain without torch (numpy export).
* 1 Nov 09:20-10:00: **fall hand-over in the robot program.** `selfright.py`: the self-right brain as numpy
  (models/bolt-go2-selfright-6.9M/brain.npz, 46 -> 256 -> 256 -> 12 ELU), parity with the SB3 brain over 400 steps of
  real self-right inputs: worst action gap 7e-7. `recovery.py` (Recovery): tilt past 60 deg -> self-right -> upright
  > 0.9 held 0.3 s -> getup clip -> back to the command; gives up after 4 s unrighted or 3 falls in a row. ("Still"
  cannot use the gyro: the brain keeps rocking on its belly at 3-7 rad/s; training's test was body speed, which a real
  Go2 cannot measure well.) `run_go2.py --recover` uses it (plan clock paused while recovering); `sim_vendor.py
  --shove-at T...` pushes the dog over. Results, robot-only inputs, no torch anywhere:
  - lockstep_recovery.py (Unitree's go2.xml, damping 0.1, 5 ms), fall -> roll -> stand -> trotting again, 10 tries
    each: side 10/10, back 10/10, tipped 10/10, shoved over while trotting 10/10 (push 2.5 and 3.5 m/s); median
    6.7 s from the fall to trotting again.
  - DDS rehearsal, real time (sim_vendor damping 2.0, 5 ms, shoves at 13 s and 30 s, plan trot 14 s + walk 8 s,
    speed stop on): both falls recovered, resumed trot then walk, plan done, 2 late steps of 1735.
* 1 Nov 09:40: **found while doing it: 592M shakes its calves at low joint damping.** The DDS rehearsal with
  Unitree's default physics (damping 0.1, 5 ms) stops at the second brain step on the joint-speed stop (FR calf 22
  rad/s, rated 15.7): with the stop off, calves reach 2.0-2.6x rated, alternating sign step to step (a chatter), 84
  of 750 steps over 1.2x. Calf peak / rated: damping 2.0 & 2 ms (ours) 0.45-0.69; 2.0 & 5 ms 0.56-0.76; 0.5 & 5 ms
  0.91-1.28; 0.1 & 2 ms 0.73-1.25; 0.1 & 5 ms 2.0-2.6. Our own env (randomized or not): <= 0.51. The lockstep gate
  missed it (it has no speed check); run f's brain ran the same defaults on 03:15 without tripping it. Not known yet:
  the real Go2's joint damping -- HARDWARE_DAY step 2 (hanging, speed stop on, logs into replay_log.py) measures it
  safely. If it is low, the fix is a training piece with joint damping randomized down to 0.1 and the 5 ms step
  (and a speed term), before the floor steps. The lockstep gate gets a speed check from now on.
* 1 Nov 09:55-10:15: **one-command setup** (the "one day with the robot" goal: no hand-fixing builds on the day).
  `setup_go2.sh [--sim]` installs CycloneDDS 0.10.x (built from source), a Python 3.11 venv with cyclonedds 0.10.2,
  unitree_sdk2_python, numpy, mujoco 3.17.0, gymnasium (and unitree_mujoco with --sim), every repo pinned to the commit
  the rehearsals ran on, into ./bolt-deps; writes bolt-deps/env.sh; ends with `check_ready.py` (PASS/FAIL per check
  with the fix: packages, both brains load, parity, step time, DDS, and with --iface the 192.168.123.x link; READY or
  NOT READY). Parity no longer needs torch on the robot's machine: `test_deploy.py <brain> --save` records the training
  env's inputs and targets (deploy/golden_bolt-go2-blind-592M.npz, 480 steps), `--golden` replays them through the
  controller (max diff 1.2e-7 / 4.6e-15). Tested from an empty folder: install ~10 min, READY 7/7; brain + self-right +
  estimator 0.15 ms median, 0.27 ms 99th percentile of the 20 ms step; then the DDS rehearsal on the fresh install
  (pushed over mid-plan): recovered, resumed sit, plan done, 0 of 1515 steps late.
* 1 Nov 10:20-10:40: **the Go2's own remote drives Bolt** (`remote.py`, `run_go2.py --remote`): left stick walk /
  trot (R1 + full stick canter), release 0.5 s -> sit, right stick steer (0.6 rad/s max), A sit, B lie, Y get up,
  L2+B or SELECT stop; hysteresis on the stick bands. It feeds the same text commands as --interactive (one path).
  Decoding checked against the SDK's own reader on the same bytes. `sim_vendor.py --remote-script` scripts the remote
  inside Unitree's simulator (the bytes the real remote sends, through the real DDS path). Test, real time, damping
  2.0, with --recover: sit -> walk -> trot -> steer 0.3 -> release (sit) -> lie -> getup -> walk -> L2+B stop: every
  command acted on, no fall, 2 of 2349 steps late.
* 1 Nov 10:35: `mjx_imitate.py --damping-range LO HI` (joint damping drawn per try, log-uniform; CPU smoke test: 8
  tries drew 0.12-1.82, no NaN). Kaggle **bolt-go2-damp** started: blind2's recipe (from 592M, moving-rate 0.5, blind,
  5% lost commands, seed 11) + `--damping-range 0.1 2.0`, 3.5 h on a T4, ends ~14:15 IST; -p scratchpad/kout_go2damp.
  Gate: lockstep_vendor at Unitree's defaults (damping 0.1, 5 ms) 0/10 falls AND its speed line passes (592M: worst
  2.23x, 367 steps over 1.2x), + --skip-p 0.05 and --delay-ms 0 19 no worse than 592M, + our exam no worse.
* 1 Nov 10:40-11:05: **Kaggle bolt-go2-blind2 -> models/bolt-go2-blind-695M, the new real-robot brain** (592M + 103M
  steps, moving sit/lie starts 0.5). Rehearsal (Unitree defaults, robot-only inputs, 41 s, 10 tries) 592M -> 695M:
  falls plain 0/10 -> 0/10; 0-19 ms delay 1/10 -> 0/10; 5% missed steps 3/10 -> 2/10; recovery 40/40 -> 40/40.
  Exam: mean 0.809 -> 0.820, falls 9.6% -> 0.0% (every moving sit/lie reached), trot 0.758 -> 0.790, canter 0.485 ->
  0.524, steer canter 0.41 -> 0.54. Knee shaking unchanged at damping 0.1 (steps over 1.2x rated 1257 -> 1448 of
  20500), better at 0.5 (132 -> 76): bolt-go2-damp (started from 592M) is still the piece for it. Every deploy default
  now points at 695M -- run_go2.py and lockstep_vendor.py defaulted to 489M, a brain that needs a velocity sensor the
  real Go2 lacks (a forgotten --brain would have run it). Golden parity record for 695M saved; check_ready READY 7/7;
  DDS run with the default brain, the scripted remote and a push: recovered, resumed walk, stopped by L2+B.
* 1 Nov 11:15-11:40: **pet intelligence on the Go2** (`pet_go2.py`, `run_go2.py --pet [--speak]`): the browser pet's
  brain (pet/js/brain.js: 29 language rules, lines, jokes, mood, memory, feedback labels) ported to Python, built
  in-house, no external AI service. Every trick maps onto trained Go2 skills (dance = walk/trot/pace with steering,
  ends sitting; spin = steered gait; come, walk left/right, turn around, zoomies, bow, play dead, stand = getup);
  tricks this body has not learned (flips, roll over, shake, beg, jump) get an honest line and a safe substitute.
  Mood changes lines and moves (grumpy: walks, no zoomies; fantastic: a happy wiggle). Battery: the robot's real
  charge when passed in. Memory + feedback labels in a JSON file (training data). Voice out: espeak-ng if installed
  (local). Selftest: every move is a real robot-program command. DDS test, real time (damping 2.0): "hi Bolt", "sit
  then dance", joke, "good boy" (labelled the dance as liked), "walk left", "do a backflip" (-> spin, said so),
  "stop" (cut the spin short, sat), "lie down", "how are you" (mood fantastic), "!stop": all acted on, no fall,
  2 of 2796 steps late. Not yet: voice input (an offline speech recognizer on the robot), the browser-only tricks.
* 1 Nov 14:30: **Kaggle bolt-go2-damp verdict: the damping range works on the knees, but this piece does not replace
  695M.** From 592M + 103M steps with joint damping drawn 0.1-2.0 per try. Rehearsal (Unitree defaults, robot-only
  inputs, 41 s, 10 tries) 695M -> damp: falls 0/10 -> 0/10; knee steps over 1.2x rated 1448 -> **580** (worst 2.17x ->
  1.93x); 5% missed steps 2/10 -> **0/10**; 0-19 ms delay 0/10 -> 0/10; damping 0.5: 76 -> 50 steps; recovery 40/40
  -> 40/40. But our exam: mean 0.819 -> 0.807, falls 0% -> 5.8% (trot -> sit 2/4), steer 0.678 -> 0.555, canter
  0.52 -> 0.50: it started from 592M and missed blind2's gains. Still over the speed check (580 steps). Next: the same
  damping range from 695M (Kaggle **bolt-go2-damp2**, started 14:30 IST, ends ~18:15; -p scratchpad/kout_go2damp2),
  same gate. Kept as a result only (scratchpad/kout_go2damp), not in models/.
- **1 Nov ~21:30 IST: damp2 adopted as the real-robot brain: `models/bolt-go2-damp-798M`.** From 695M + 103M steps,
  joint damping 0.1-2.0 per try. Rehearsal 695M -> 798M: falls 0/10 -> 0/10; with 5% missed steps 2/10 -> **0/10**; with
  0-19 ms delay 0/10 -> 0/10; knee steps over 1.2x rated 1448 -> **96** (worst 2.17x -> 1.89x); at damping 0.5 76 -> 11;
  recovery 40/40 both. Exam mean 0.819 -> 0.817, steer 0.678 -> 0.708, falls 0% -> 1.9%. Re-test, 12 tries per moving
  posture start: trot -> sit falls 2/12 -> 3/12, trot -> lie 0/12 -> 1/12, walk -> sit/lie 0/12 both: slightly worse on
  sitting from a trot (within chance at 12 tries), much better on the speed limit that would stop the real robot.
  Golden parity saved; check_ready READY 7/7; all deploy defaults switched. Open: sitting from a trot (both brains).

- **1 Nov ~22:30 IST: walk before sit/lie (controller `walk_down`), tested, left off.** Kaggle GPU quota for the week is
  used up (pilots for jump and trot -> sit could not start), so a software-only fix was tried: from trot/pace/canter,
  walk 1 s before sit or lie. Rehearsal (798M, Unitree defaults, 52 s plan with 4 trot/canter -> sit/lie, 10 tries):
  falls 0/10 without and 0/10 with; knee steps over 1.2x rated 258 vs 243 (worst 2.27x vs 2.28x; canter is the source
  in this plan). No measurable gain in the vendor simulator, so it stays off (default 0); the trot -> sit weakness shows
  only in our exam's instant 1.9 m/s moving starts. Pilots `scratchpad/k_jumppilot`, `k_sitpilot` (40 min each) are
  ready for when the quota resets.
- **1 Nov ~23:50 IST: paused by the owner.** State to resume from: robot brain `models/bolt-go2-damp-798M` (deploy
  defaults, check_ready 7/7). Kaggle GPU quota used up for the week; pilots ready to push (`kaggle/make_kernel.py`
  flags in train/LOCAL_GPU_PROMPT.md: sit-from-trot from 798M, jump from `models/bolt-go2-jump-warm`). Running on
  Kaggle CPU (free, ~10.5 h): `arunmallikarjuna/bolt-lite3-dog-pilot`, the Lite3's first dog-motion brain
  (`kaggle kernels output arunmallikarjuna/bolt-lite3-dog-pilot -p runs/kaggle-lite3dog`). Next ideas, after those
  pilots: a short history of past inputs and a privileged critic (40-min pilots each). Own-GPU route:
  train/LOCAL_GPU_PROMPT.md.
- **2 Nov: work while the Kaggle GPU quota is out.** (1) Lite3 on Kaggle's free CPU: `bolt-lite3-dog-2` (20.9M brain
  continued; win: mean tracking >= 0.80, canter >= 0.45) and `bolt-lite3-selfright-pilot` (win: righted from side and
  back >= 0.5), both ~10.5 h. (2) GPU trainer: `--privileged-critic` (value net sees true velocity, physics factors,
  paw contacts) and `--history K` (brain reads its last K measured frames; MjxImitatePolicy keeps the history, so the
  robot program is unchanged). Warm start from 798M with history 3 acts identically (max action diff 0); CPU smoke
  runs pass for both flags and for neither. (3) Lite3 on the GPU trainer: the 20.9M brain's numpy export scores 0.772
  on the CPU exam (SB3 original 0.771); a GPU-trainer smoke run on the Lite3 body passes. Queued for the quota reset:
  privhist vs base798 A/B (40 min each), Lite3 GPU run 1 (3.5 h), sit and jump pilots.
- **2 Nov evening: real-time rehearsal fix (Go2) and the Lite3 robot program.**
  * Found: since the speed stop went in (1 Nov), `run_go2.py` against `sim_vendor.py` stopped on the brain's first or
    second step (695M and 798M alike: a knee at 19-22 rad/s, rated 15.7): a gait's first targets from a standstill, and
    single 20 ms spikes (1.5-2.4x rated) at switches between moves. On hardware day it would have stopped every run.
    Fix: ease in from the stance over 0.5 s at the start (--ramp; easing at every switch was tried and made trot -> sit
    fall, so not); the speed stop now trips on 3 steps in a row over 1.2x rated (60 ms: a shake or runaway) or any step
    over 3x (--speed-steps, --speed-hard). Result: the 29 s plan trot / sit / lie / getup / walk runs to the end in
    Unitree's simulator over DDS (15 steps over 1.2x, none sustained). A second try stopped on "state 63 ms old": this
    1.3-core machine running simulator and program together.
  * Lite3: `run_go2.py --body lite3` speaks DEEP Robotics' UDP joint protocol (`lite3_link.py`, selftest PASS) to their
    own simulator (`sim_lite3.py`, headless, real-time factor 0.97) or, on the robot, to `lite3_bridge/bridge.cpp`
    (their Lite3_MotionSDK; compiles and links against their x86_64 library; it cannot start off the robot's network).
    Stand-up from lying works with the right joint signs. The 20.9M CPU brain falls 0.9 s into a live trot: lockstep
    shows why, it is not robust to 5-15 ms of command delay on DEEP's undamped model (3/3 falls; 0/3 at 1-4 ms, 0/3
    lying-start, 0/3 DEEP's simulator stepped in lockstep). The queued Lite3 GPU run trains 0-40 ms delay and joint
    damping 0.1-2.0, the recipe that fixed both on the Go2.
- **3 Nov ~02:45 IST: Lite3 CPU results.** dog-2 (38.8M): tracking 0.771 -> 0.813, canter 0.36 -> 0.49, falls on the 9
  clips 1/36 -> 0/36, no falls at 10/20/40 ms delay; moving sit/lie still weak (trot -> sit 3/4). But on DEEP's own
  model (lockstep, estimated v/h, 41 s plan) it falls 3/3 at sit -> lie, where 20.9M fell 0/3: better in our exam, worse
  in the vendor rehearsal, so 20.9M stays the Lite3 robot candidate for now. Self-right 28M: 10/10 from side, back and
  tipped (0.64 s). Running: lite3-dog-3 (moving sit/lie starts, steering), Kaggle CPU. GPU quota still out.
- **3 Nov morning: why the Lite3 38.8M brain falls sit -> lie on DEEP's model (3/3) but not on ours (0/8).** Ruled
  out, each by a direct test: the robot program's inputs (test_deploy.py now takes the brain's body: Lite3 inputs match
  training to 1.2e-7, targets 4.8e-15, PASS), the velocity/height estimate (3/3 falls with true, estimated or reference
  velocity), torque limits (24/24/36 as the URDF: 3/3), armature (3/3), the trunk mesh (collision off: 3/3). Found:
  (1) our Lite3 collision legs were too thin and short: DEEP's shank mesh runs down over the foot (to -0.233 m) and its
  thigh is thick and offset toward the body; lite3_nomesh.xml now matches their bounds (thigh capsule r 0.042 offset
  0.031 inward, shank capsule r 0.021 down to the foot; clip check: no self-penetration, legs resting on the floor only
  when sitting/lying, 1-8 mm). (2) DEEP's simulator gives every motor a flat 30 N·m at any speed; with a torque-speed
  curve like ours (full to rated speed, 0 at 25/21 of it) 38.8M falls 1/3 at sit -> lie instead of 3/3: the brain
  rises from sitting much harder on DEEP's unlimited motors. The real Lite3's motors are speed-limited (URDF 26.2 /
  17.3 rad/s), so DEEP's model is the pessimistic case. Remaining gap: unexplained, small.
- **3 Nov ~09:00 IST: Lite3 robot brain -> `bolt-lite3-dog-61.8M`** (lite3-dog-3, CPU, moving sit/lie starts + steering):
  sit/lie from a moving walk or trot 0 falls in 16 (38.8M: 11), exam fall rate 0.00, tracking 0.812, canter 0.52; on
  DEEP's MJCF 0/3 falls over the 41 s plan (38.8M 3/3, 20.9M 0/3) with a third of 20.9M's over-speed steps. Still 3/3
  falls with 5-15 ms delay there. Also new: `--motor-speed-range` in the GPU trainer; the queued Lite3 GPU run now
  warms from 61.8M with damping 0.1-2.0, motor speed x1-4, 0-40 ms delay and the corrected leg collision shapes.
- **3 Nov: fall recovery for the Lite3, 40/40 on DEEP's model; Go2 still 40/40.** Lite3 self-right 28M exported to numpy
  (selfright.py export/test take a Lite3 folder; parity 9.5e-7). lockstep_recovery.py --body lite3 (DEEP's MJCF, joint
  sign, HL/HR names): first 4/20: the self-right rolled it over every time (0.5-2.3 s), but the getup clip took over from
  wherever the legs were and it tipped past 60 degrees. recovery.py now tucks the legs into the body's lie pose (1.0 s
  blend + 0.3 s hold) before the getup: 0.4 s gave 11/20 (falls during the tuck), 1.0 s 20/20, then 10 tries each:
  Lite3 side / back / tipped / shove 10/10 each (fall to trotting again median 7.9 s); Go2 10/10 each (median 7.9 s,
  was 6.8). run_go2.py --body lite3 --recover now allowed (self-right brain models/bolt-lite3-selfright-28M).
- **3 Nov: live Lite3 run in DEEP's simulator with --recover.** 61.8M falls 1/3 on DEEP's model with 3-6 ms of delay
  (0/3 at 1-3 ms), so the live link now sends each new target at once (lite3_link.set) instead of on the next 2 ms
  resend; the speed stop ignores the self-right and tuck (fast by design; --speed-hard still applies). Live, 2 runs of
  the 29 s plan: run 1 fell at the trot start, rolled over, got up, resumed and finished the plan (37.6 s, standing);
  run 2 recovered a fall the same way, then stopped on a sustained knee over-speed at 10 s. The trot start on DEEP's
  undamped, unlimited motors is the Lite3's open problem: the queued GPU run trains delay, damping and motor speed.
- **3 Nov: weight-space averaging/blending of Lite3 brains (train/weight_space.py), CPU only. Verdict: not adopted.**
  DEEP's model: 61.8M | soup of last 5 checkpoints | soup of last 3 | 0.7 x 61.8M + 0.3 x 38.8M: 41 s plan falls 0/3 |
  1/3 | 1/3 | 0/3; 3-6 ms delay 1/6 | 0/6 | 0/6 | 0/6. Confirmation, 61.8M vs the blend: 3-6 ms 1/12 vs 0/12, 5-15 ms
  1/6 vs 3/6, recovery 20/20 both, moving sit/lie (12 tries each, our model) 0/48 vs 8/48 falls. The blend trades the
  moving postures and long delays for short delays: worse overall, so 61.8M stays.
- **3 Nov evening: latency compensation (controller.predict; run_go2 / lockstep_vendor --predict-ms).** The brain gets the
  state extrapolated by the expected command delay (joints at their speed, body by its gyro). Lite3 61.8M on DEEP's
  model, trot -> sit: 3-6 ms delay 1/24 -> 0/24 falls (4.5 ms ahead); 5-15 ms delay 6/12 -> 2/12 (20 ms), 3/12 (10 ms);
  but with no delay, 5 ms ahead gives 6/12 falls (0/12 without). Over-predicting is as harmful as the delay, so the
  default is off; set it to the delay measured on the robot (HARDWARE_DAY.md).
- **3 Nov ~00:45 IST: Lite3 robot brain -> `bolt-lite3-dog-63.8M`** (lite3-dog-4: corrected leg shapes, 0-60 ms delay).
  DEEP's model, 61.8M -> 63.8M: routine 0/3 -> 0/3, over-speed steps 342 -> 296, 3-6 ms delay 1/24 -> 0/24, 5-15 ms
  6/12 -> 4/12, recovery 20/20 both, moving sit/lie 0/48 both; our exam tracking 0.812 -> 0.803. The A/B with 4x faster
  motors (4b): routine 2/3 falls, 5-15 ms 6/12: not the fix. Next Kaggle CPU job lite3-dog-5: 0-80 ms delay from 63.8M.
  Workers (cloud sessions) running: w1 motor speed x2, w2 10-60 ms delay.
- **4 Nov ~05:00 IST: Lite3 robot brain -> `bolt-lite3-dog-87.3M`** (the final checkpoint of lite3-dog-4). Lesson: the
  run's "best" (63.8M) is picked by our exam, which has no delay; later checkpoints handle delay better on DEEP's model.
  Under realistic delay (DEEP's MJCF): 41 s routine with 3-6 ms 3/6 -> 1/6 falls, trot -> sit with 5-15 ms 4/24 -> 2/24;
  recovery 20/20 and moving sit/lie 0/48 both; our exam tracking 0.803 -> 0.780; no-delay routine 0/3 -> 1/6. Blends of
  63.8M and 87.3M and a four-checkpoint average did not beat it. From now on every run's late checkpoints are tested on
  DEEP's model with delay, not only its exam-best. Open: the canter under delay (every remaining routine fall).
- **4 Nov ~08:00 IST: workers.** w1 (61.8M + motor speed x2): best stayed at its start (exam 0.808), dead end, closed.
  w2 (61.8M + 10-60 ms delay): its last checkpoint (109.4M) on DEEP's model with delay: routine with 3-6 ms 3/6 falls,
  5-15 ms 5/24 (87.3M: 1/6, 2/24): not adopted. Started w3 (87.3M + canter drawn 3x + 0-60 ms) and w4 (87.3M continued,
  0-60 ms), 9.5 h each, pushing best and latest checkpoints.
- **4 Nov afternoon.** Tested on DEEP's model with delay: w3 101M (canter x3) routine 1/6, 5-15 ms 6/24; w4 104M routine
  0/6, 5-15 ms 2/24, but on 12 more delayed routines 4/12 vs 87.3M's 3/12 (with no delay 0/6 vs 1/6): a tie, 87.3M
  stays; lite3-dog-5 (63.8M + 0-80 ms) late checkpoints 3/6 and 6-10/24: worse, dropped. Nearly every fall under delay
  is in the canter, so run_go2.py now runs canter requests as trot on the Lite3 by default (--no-canter / --allow-canter).
  Live in DEEP's simulator (5 runs, --recover): 2 finished the plan after a recovered fall; 3 fell at the trot start and
  the self-right stuck at ~57 degrees (8 s timeout did not help, back to 4 s). The self-right brain predates the leg-shape
  fix: Kaggle job lite3-selfright-2 fine-tunes it on the corrected body. Kaggle lite3-dog-6 (87.3M + canter x2 + 0-80 ms)
  running.
- **4 Nov evening: worker finals under delay (DEEP's model).** w3 115.6M (canter x3): routine with 3-6 ms 1/6, 5-15 ms 8/24.
  w4 119.6M (87.3M continued): collapsed, routine 4/6 (all in the trot), 5-15 ms 19/24, though its own exam still read
  0.772: long CPU fine-tuning with delay drifts past ~105M, and the exam does not see it. 87.3M stays. New workers w5
  (moving sit/lie 0.75) and w6 (switch rate 0.6), both from 87.3M with 0-60 ms; their checkpoints are tested as they come.
- **4 Nov ~22:30 IST.** With the canter off (as run_go2 now does on the Lite3), 87.3M on DEEP's model: a no-canter
  41 s routine with 3-6 ms delay 0/12 falls; trot -> sit 5-15 ms 2/24 again (4/48 over two runs). w5 93M (moving 0.75):
  no-canter routine 2/12 (trot), trot -> sit 1/24 + 1/24 = 2/48: better sits, worse routine, not adopted. w6 92M
  (switch 0.6): routine 1/6, 5-15 ms 5/24, no. 87.3M stays; CPU fine-tuning has plateaued since 87.3M.
- **4 Nov ~22:25 IST (16:55 UTC).** Live pet mode on DEEP's simulator (run_go2 --body lite3 --pet --recover) answers and
  acts; its 2 falls were sit commanded while lying. Lockstep on DEEP's model: trot:4,lie:6,sit:5 fell 6/6 (in sit, with
  or without delay), trot:4,lie:6,getup:6,sit:5 0/6 - training only leaves lie by getup. Controller now plays the get-up
  first for any move commanded while lying (or mid-get-up), then the move 0.2 s after it ends: trot:4,lie:6,sit:5 0/6,
  and 0/6 with 3-6 ms delay; Go2 trot/lie/sit/walk 0/4; test_deploy PASS.
  Live on DEEP's simulator in real time (sim_lite3 + run_go2 --recover, getup:4,trot:4,lie:6,sit:5,walk:5): 3/3 runs with
  no fall and nothing to recover (body up 0.91-0.94 at the end; 0-5 of 1200 steps late; 4-8 steps over 1.2x rated).
- **5 Nov ~02:30 IST (4 Nov 21:00 UTC).** Kaggle lite3-dog-6 (87.3M + canter x2 + 0-80 ms) on DEEP's model, no-canter
  routine 3-6 ms (12) / trot->sit 5-15 ms (24): best/ 88.3M 1/12, 7/24; 98M 0/12, 2/24; **103M 0/12, 1/24** (being
  confirmed); 108M 1/12, 7/24. Worker 5 104M 1/12, 1/24. Kaggle lite3-selfright-2 (28M fine-tuned on the corrected legs):
  its exam saturates (best/ stays 28M); the faster 67M checkpoint recovers 32/40 on DEEP's model (1 ms) against 39/40 for
  28M: not adopted. lockstep_recovery at 5 ms physics fails most Lite3 tries (12/40) because the motion brain's get-up
  tips over at 5 ms, with either controller; DEEP's simulator steps 1 ms, so the Lite3 default is now --dt 0.001.
  Kaggle GPU quota still out. Started lite3-dog-7/8: gentler fine-tunes of 87.3M (lr 3e-5, target-kl 0.01, 0-40 ms;
  8 with sit x2, moving 0.6).
  103M confirmation (all with the per-try controller reset): no-canter routine 3-6 ms 0/24, trot->sit 5-15 ms 2/48
  (87.3M 4/48), canter routine 3-6 ms 0/12 (87.3M 2/12, canter), recovery 20/20, moving 0/48; our exam tracking 0.750 vs
  0.780 and steering 0.382 vs 0.449; live on DEEP's simulator without the canter, 4 runs each: no falls for either
  (87.3M one speed stop entering lie). **87.3M stays**; 103M kept as models/bolt-lite3-dog-103M-candidate. (An
  earlier 7/12 for 103M's canter routine was a test artifact: the rehearsal reused the controller between tries, so a
  try that fell lying started the next one in lie, and the new lie rule sent its trot through a get-up from standing.
  Rehearsals now reset the controller every try.)
- **5 Nov ~04:50 IST (4 Nov 23:20 UTC). New Lite3 brain: worker 5's final, 112M (models/bolt-lite3-dog-112M).** Workers
  5 and 6 finished (9.5 h, ~112M). On DEEP's model: w5 112M no-canter routine 3-6 ms 0/12 + 0/12, trot->sit 5-15 ms
  0/24 + 0/24 (87.3M 4/48), canter routine 5/12 (87.3M 2/12), recovery 20/20, moving 0/48; exam tracking 0.756 / steer
  0.456 (87.3M 0.780 / 0.449); live on DEEP's simulator without the canter 4/4 with no fall and no stop. w6 111.5M:
  steer 0.509 but trot->sit 5/24, no. Adopted 112M as the default (run_go2, lockstep_vendor, lockstep_recovery,
  check_ready; golden recorded; check_ready --body lite3 8/8) since the Lite3 canter already runs as a trot; 87.3M
  stays in models/ for --allow-canter runs. Workers 7 (lr 3e-5, steer weight 0.25, 0-40 ms) and 8 (lr 3e-5, canter x2,
  0-80 ms) started from 87.3M at 22:38 UTC.
- **5 Nov 01:30 UTC check.** Kaggle dog-7/8 running; GPU quota still out. First worker checkpoints on DEEP's model:
  w7 92M (lr 3e-5, steer weight 0.25, 0-40 ms) no-canter routine 0/12, trot->sit 5-15 ms 6/24, but our exam tracking
  0.788 / steer 0.506 (best yet; 112M 0.756 / 0.456); w8 94M (lr 3e-5, canter x2, 0-80 ms) routine 1/12 (trot),
  trot->sit 0/24, tracking 0.764 / steer 0.439. Neither replaces 112M; later checkpoints are tested as they come.
- **5 Nov ~03:00 UTC. New Lite3 brain: a weight-space blend, 0.75 x 112M + 0.25 x worker 7's 92M
  (models/bolt-lite3-dog-blend-112M-w7).** w7 92M tracks and steers best (0.788 / 0.506) but trot->sit 6/24; 50/50
  blend 0.789 / 0.477 but 3/24; 75/25 blend: no-canter routine 0/12 + 0/12, trot->sit 0/24 + 0/24, canter routine 2/12
  (112M 5/12), recovery 20/20, moving 0/48, tracking 0.775 / steer 0.441 (112M 0.756 / 0.456), live 4/4 with no fall
  and no stop. Adopted (defaults swapped, golden recorded, check_ready --body lite3 8/8). First weight-space win: the
  blend keeps one parent's delay robustness and takes part of the other's tracking.
- **5 Nov 05:30 UTC check.** Kaggle dog-7/8 still running; GPU quota out. On DEEP's model: w7 109M routine 0/12,
  trot->sit 9/24, exam tracking 0.800 / steer 0.583 (best tracking and steering yet, worst sits); w8 106M routine 0/12,
  trot->sit 0/24, 0.754 / 0.471. Blends of 112M with w7 109M: 25% routine 2/12 (trot), sits 0/24, 0.776 / 0.442; 35%
  routine 1/12, sits 4/24. w7 has drifted too far from 112M for a blend to keep both; the current blend stays.
- **5 Nov 05:25 UTC: paused.** No robot to test on yet, and simulation training has reached diminishing returns, so
  the paid compute is stopped: the 4-hourly loop is disabled and workers 7/8 were stopped and archived (their last
  checkpoints, w7 109M and w8 106M, are on claude/bolt-worker-7/-8). Kaggle dog-7/8 (free) finish on their own; their
  outputs can be fetched later with `kaggle kernels output arunmallikarjuna/bolt-lite3-dog-7` (and -8). Robot brains as
  of now: Go2 models/bolt-go2-damp-798M, Lite3 models/bolt-lite3-dog-blend-112M-w7. Next real step: a hardware day
  (HARDWARE_DAY.md) on a borrowed or owned robot.
- **5 Nov ~10:30 UTC. Free-compute training (this session + Kaggle).** Local keepalive job l1 (112M + canter x2,
  lr 3e-5, 0-60 ms, 9 h) and Kaggle lite3-dog-9 (112M + steer weight 0.25, lr 2e-5) and -10 (112M + canter x3, lr 2e-5)
  running. Kaggle dog-7/8 done: on DEEP's model with delay, dog-7 best 105M routine 3/12, trot->sit 4/24 (tracking
  0.807 / steer 0.485); dog-8 104M 2/12, 4/24; dog-8 108M 4/12, 9/24. Blend robot brain + 25% dog-7 105M: routine 0/24,
  trot->sit 2/48, **canter routine 0/12** (robot brain 2/12), recovery 20/20, moving 0/48, 0.791 / 0.475: not the
  default (sits), kept as models/bolt-lite3-dog-canter-candidate for --allow-canter runs. 15% blend: trot->sit 1/48,
  canter 2/12: nothing gained.
- **5 Nov ~22:30 UTC.** Kaggle dog-9/10 (from 112M, lr 2e-5) on DEEP's model with delay: dog-9 132M (steer weight
  0.25) routine 0/12, trot->sit 3/24, steer 0.580; dog-9 122M 0/12, 1/24, 0.475; dog-10 120M 0/12, 2/24; **dog-10 130M
  (canter x3) routine 0/24, trot->sit 0/48, recovery 20/20, moving 0/48, tracking 0.777 / steer 0.516, but canter
  routine 7/12 and live 17-25 steps over 1.2x rated (robot brain 7-14)**: kept as
  models/bolt-lite3-dog-steer-candidate-130M, not the default. Blends with the robot brain: +25% dog-9 132M sits 2/24;
  50/50 with 130M sits 0/48, canter 4/12, 0.782 / 0.462: no. Robot brain unchanged. Started Kaggle lite3-dog-11
  (steering + sit x2) and -12 (canter x2, 2-20 ms) from 112M. The local job only trains while this session is busy
  (the idle cloud machine is reclaimed ~20 min after a check), so Kaggle carries the training.
- **6 Nov ~09:30 UTC.** Kaggle dog-11 (112M + steer weight 0.2 + sit x2) and dog-12 (112M + canter x2, delay 2-20 ms)
  on DEEP's model: dog-11 134M routine 0/12, trot->sit 2/24, steer 0.559; dog-11 124M 0/12, 1/24, 0.481; dog-12 134M
  6/12, 14/24; dog-12 124M 2/12, 6/24 - training without short delays (2-20 ms instead of 0-60) hurts badly. No new
  default. Started Kaggle lite3-dog-13 (dog-10's recipe with motors x0.85, from 112M) and -14 (same from the steering
  candidate 130M, lr 1e-5) to cut the steerer's joint over-speed.
- **6 Nov 17:45 UTC. Over-speed: a training gap, not a brain trait.** In our training env (overspeed measured per
  step, 6 tries x 400 steps with switching and steering), 130M has joints over rated in 6.6% of steps (robot brain
  9.5%), yet live on DEEP's simulator it over-speeds twice as often. Our motor model fades the drive above rated speed
  (robot_bodies cutoff), DEEP's model has flat +-30 N·m with no speed limit, so training never showed the brain its
  excess. New: imitate_env overspeed_weight / train_imitate --overspeed-weight (reward minus weight x summed excess over
  rated). Kaggle lite3-dog-15: 130M with a x1.3 drive + overspeed weight 0.5, lr 1e-5. (dog-13/14 run the opposite
  test, a x0.85 drive.)
- **6 Nov 21:30 UTC. Slower training motors make over-speed worse (confirms the training-gap reading).** Kaggle dog-13
  (112M + canter x3 + motors x0.85) and dog-14 (130M + same) on DEEP's model with delay: dog-14 150M routine 0/12,
  trot->sit 0/24, steer 0.462, but live over-speed 35-41 steps a run (130M 17-25, robot brain 7-14); dog-13 126M 0/12,
  0/24, steer 0.486, live 14-16; dog-13 118M sits 1/24; dog-14 140M sits 1/24. A harder speed cap in training teaches
  harder commands, which an uncapped motor turns into more over-speed. No change of brain. The direction is the
  opposite: Kaggle lite3-dog-15 (130M, drive x1.3, overspeed 0.5), -16 (112M, drive x1.3, overspeed 0.5, steer 0.2)
  and -17 (130M, drive x1.5, overspeed 1.0) are running.
- **7 Nov 05:30 UTC. The over-speed fix works.** Kaggle dog-15 (the steering candidate 130M trained with a x1.3 drive
  and overspeed weight 0.5, lr 1e-5): live on DEEP's simulator 147M has 7-15 steps a run over 1.2x rated (130M 17-25,
  robot brain 7-14) and keeps the steering (exam 0.778 / 0.508). But under delay on DEEP's model it lost some
  stability: 147M routine 1/12, trot->sit 2/24; 140M 0/12, 1/24 (live 9-16). Blend robot brain + 25% 147M: 0/12, 1/24,
  steer 0.458: no. Robot brain unchanged. dog-16 (the same fix from the sit-safe 112M) and the GPU run (from the robot
  brain) test the fix on bases that hold the sits.
  dog-15 140M confirmation: trot->sit 5-15 ms 1/48 more (2/72 in all), canter routine 5/12: kept as
  models/bolt-lite3-dog-steer-candidate-140M (steering 0.517 at the robot brain's joint speeds), not the default.
- **7 Nov ~10:00 UTC. New Lite3 brain: 149M (models/bolt-lite3-dog-149M), the over-speed fix.** Kaggle dog-17 (130M with
  a x1.5 drive and overspeed weight 1.0, lr 1e-5): 149M on DEEP's model routine 3-6 ms 0/12 + 0/12, trot->sit 5-15 ms
  0/24 + 0/24, recovery 20/20, moving 0/48, exam 0.778 / steer 0.534 (robot brain 0.775 / 0.441), live 4/4 with no fall
  and 5-12 steps a run over 1.2x rated (robot brain 7-14); canter routine 9/12 (robot brain 2/12; the canter runs as a
  trot by default). Adopted: defaults swapped, golden recorded, check_ready --body lite3 8/8. Others: dog-17 141M live
  13-18 and one live fall; dog-16 (from 112M, x1.3 / 0.5) 140M steer 0.533 but sits 2/24, 127M sits 1/24. Started
  Kaggle dog-18 (the strong fix from 112M) and dog-19 (stronger still from 130M).
- **7 Nov 13:30 UTC. First Lite3 GPU run: does not transfer to DEEP's model.** Kaggle GPU bolt-lite3-gpu-overspeed
  (MJX, warm = the robot brain blend-112M-w7, drive x1.0-1.5, overspeed 0.5, 0-60 ms, 8 h, 823M steps at 28k steps/s).
  Our exam (CPU env): final 0.813 / steer 0.572, no falls. On DEEP's model (trot -> sit, 3-6 ms, 4 tries): checkpoint 0
  (= the robot brain, export identical) 0/3; 51M 1/4; 103M 0/4; 206M 3/4; 412M 4/4; 823M routine 12/12 and trot->sit
  23/24, all within ~3 s of trotting. The brain learns something the MJX and CPU envs allow and DEEP's model does not
  (candidates: torque limits 24/24/36 vs DEEP's flat 30 N·m, our speed fade, MJX's floor-only collisions, float32).
  CPU-trained Lite3 brains and GPU-trained Go2 brains do transfer. No more Lite3 GPU time until the gap is found;
  dog-20 re-pushed (its first push ran before the dataset version with 149M was ready).
- **7 Nov ~15:00 UTC. Go2: GPU privileged critic + history (privhist) beats 798M; both trip the speed stop live.**
  Kaggle GPU bolt-go2-privhist (from 798M, --privileged-critic --history 3): our exam tracking 0.833 / falls 0% / steer
  0.750 (798M 0.817 / 1.9% / 0.708). On Unitree's go2.xml (lockstep, 21 s routine, 10 tries each): plain 0/10 both,
  0-19 ms 0/10 both, 5% skipped steps 0/10 both, 15-19 ms 0/12 both, 15% skips 1/12 both; steps over 1.2x rated 10 of
  10500 (798M 20), worst 1.65x (1.84x); recovery 20/20. BUT live in Unitree's simulator in real time (sim_vendor.py,
  damping 0.1) both trip run_go2's speed stop (3 steps in a row over 1.2x rated) within 6-10 s: 798M at 6.3 s (1.81x)
  and 10.1 s (1.58x); privhist at 10.1 s twice (1.49x, 1.30x). The lockstep tests never apply the speed stop, so the
  "10/10" rehearsals hid this. The Lite3's over-speed fix applies: Kaggle GPU bolt-go2-gpu-overspeed (warm = privhist,
  same flags + --overspeed-weight 1.0 --motor-speed-range 1.0 1.5, 5 h) started. Win: live in Unitree's simulator the
  full routine with no speed stop, and the lockstep results kept.
- **8 Nov 01:30 UTC. New Go2 brain: 211M (models/bolt-go2-overspeed-211M), the first Go2 brain that runs the live
  routine in Unitree's simulator without the speed stop.** Kaggle GPU bolt-go2-gpu-overspeed (warm = privhist, drive
  x1.0-1.5, overspeed weight 1.0, 5 h): live 3/3 full routines, no stop, 2-5 steps over 1.2x rated a run (its 105M
  checkpoint: stop at 10.1 s twice of 3). Lockstep on go2.xml: routine 0/10, 0-19 ms 0/10, 5% skips 0/10, 15-19 ms 0/12,
  15% skips 1/12 (as 798M), over 1.2x 11 of 10500 (798M 20), worst 1.48x (1.84x), recovery 20/20, exam 0.840 / 0% /
  steer 0.686 (798M 0.817 / 1.9% / 0.708). Adopted: go2 defaults swapped in run_go2/lockstep_vendor/lockstep_recovery/
  check_ready; test_deploy fixed for brains with input history (it called act() twice per step, advancing the history
  twice; now the brain runs once, inside the controller, and resets per try; the golden records the try starts);
  goldens re-checked for 211M, 798M and Lite3 149M; check_ready Go2 7/7, Lite3 8/8.
- **8 Nov 03:00 UTC. Lite3: no change.** Kaggle dog-18 (112M, drive x1.5, overspeed 1.0, steer 0.2), dog-19 (130M, x1.7,
  1.5), dog-20 (149M + canter x3 + the fix) on DEEP's model: dog-18 131M and 123M routine 0/12, trot->sit 0/24, exam
  0.774 / steer 0.520 and 0.519, live 6-11 steps over 1.2x; dog-19 150M 0/12, 0/24, 0.779 / 0.507, live 5-13; dog-20 169M
  0/12, 1/24, steer 0.569; dog-20 168M 1/12, 1/24. Blend 149M + 50% dog-20 169M: trot->sit 0/48 but routine 2/12 (trot).
  None beats 149M (0.778 / 0.534, live 5-12). The penalty strength (x1.3/0.5 to x1.7/1.5) changes little once on.
- **8 Nov 05:00 UTC. Lite3 GPU gap, first probe.** The GPU brain (gpu-overspeed 823M) on DEEP's model, trot 6 s, 6 tries:
  falls 2/6 with no delay, 4/6 at 1-2 ms, 6/6 at 3-6 ms and 10-15 ms (CPU-trained brains: 0/6 at all of these); joint
  damping 1.0 makes it worse (4/4), 2 ms physics no different from 1 ms, true velocity input no better. So the gap is
  mostly about delay: the MJX trainer's latency (whole 2 ms substeps of the sent targets, drawn 0-60 ms per try,
  mjx_imitate _delayed_physics) is not teaching what the CPU trainer's does. Next: compare the two latency
  implementations step by step on one state (test_mjx_imitate.py style) before any more Lite3 GPU time.
- **8 Nov 09:30 UTC. Lite3 GPU gap: the torque limit, not the delay code.** mjx_imitate's latency is the same as
  imitate_env's (same queue, same substep split, same lag). The GPU brain 823M in our CPU env: trot 0/8 falls at 0, 3-6 and
  10-15 ms (149M also 0/8): it is robust there. On DEEP's model at 3-6 ms (lockstep_vendor, new diagnostic env var
  LOCKSTEP_TORQUE): DEEP's flat 30 N·m 6/6 falls; our limits 24/24/36 3/6; 36/36/36 6/6; 149M with our limits 0/6. The
  GPU brain learned to lean on our exact limits (commands past them and lets the motor clip). New: mjx_imitate
  --motor-torque-range LO HI (per try and joint type, peak torque and power x log-uniform factor). Short Kaggle GPU
  check started from 149M with --motor-torque-range 0.8 1.4.
- **8 Nov 13:30 UTC. Lite3 GPU torque check: half the gap.** Kaggle GPU bolt-lite3-gpu-torque (warm 149M, torque
  x0.8-1.4 per joint type, drive x1.0-1.5, overspeed 1.0, 0-60 ms, 2.3 h, 316M steps). Trot 6 s at 3-6 ms on DEEP's
  model: 105M 0/6, 211M 1/6, 316M 0/6 (the earlier GPU run: 206M 2/6 here, 412M 4/4, 823M 6/6): the drift no longer grows
  as fast. But 316M on the full tests: no-canter routine 3-6 ms 7/12 (trot), trot->sit 5-15 ms 9/24 (149M 0/12, 0/24).
  Varying the torque limit is part of the fix, not all of it. Next suspects for the rest: our motor model's speed fade
  (DEEP's model has none), the capsule legs vs DEEP's meshes, MJX's floor-only contacts. No more Lite3 GPU runs until a
  probe finds the next cause; the robot brain stays 149M.
- **8 Nov 17:00 UTC. Lite3 GPU gap probe 2: not the motor model.** lockstep_vendor gains LOCKSTEP_FADE (our robot_bodies
  motor model: torque, power and speed fade, on DEEP's model). gtq_316 trot -> sit at 5-15 ms, 8 tries: DEEP's motors 2/8,
  our torque limits 2/8, our full motor model 2/8 (149M with our motor model 0/8). With the torque range in training, the
  rest of the gap is the body: DEEP's mesh legs and contacts vs our capsules and MJX's floor-only contacts, or mass /
  inertia / joint friction. Next probe: the GPU brain on our own lite3_nomesh.xml through the same lockstep loop.
- **8 Nov 21:30 UTC. Lite3 GPU gap probe 3: not DEEP's body.** New lockstep_vendor diagnostic LOCKSTEP_OURS=1 (our training
  body, Bolt's signs, same deploy loop). gtq_316 trot -> sit at 5-15 ms, 8 tries: our body 3/8 (estimated vel/height) and
  2/8 (true); DEEP's body 2/8 (estimated), 0/8 (true); 149M 0/8 everywhere. In our training env (ImitateEnv, trot 8 s ->
  sit, 10 tries): gtq_316 0/10 with no delay and with 5-15 ms, as 149M. So the GPU brains are fine in the training env and
  fragile in the DEPLOY LOOP on either body: the gap sits between how training and the robot program drive the robot (PD
  torques computed every 1 ms from the controller's inputs vs the env's actuators, the leg estimator, settle-in start),
  not in DEEP's model; CPU-trained brains absorb it. That matters for the real robot too. Next: make the deploy loop's
  actuation selectable (env-style actuators at 2 ms) and find which piece flips gtq_316.
- **9 Nov 06:00 UTC. New Go2 brain: torque-steer (models/bolt-go2-torque-steer).** Kaggle GPU (slug bolt-go2-privhist
  v3): 211M + torque limits x0.8-1.4 per joint type + steer weight 0.8, 5 h. Live in Unitree's simulator 3/3 full
  routines, no speed stop, 3-4 steps over 1.2x a run. go2.xml lockstep: 0/10 plain, 0-19 ms, 5% skips; 15-19 ms 0/12,
  15% skips 1/12; over 1.2x 12 of 10500, worst 1.38x (211M 1.48x); recovery 20/20; exam 0.845 / 0% / steer 0.702 (211M
  0.840 / 0% / 0.686). Adopted: go2 defaults swapped, golden recorded, check_ready 7/7.
- **9 Nov 09:30 UTC. Lite3 GPU gap probe 4 (deploy loop on our body, true inputs, 12 tries each).** gtq_316: no delay
  0/12; 5-15 ms per-step jitter 2/12 (trot 13 s, the falls are in the trot); fixed 10 ms 1/12; fixed 15 ms 4/12. So it
  is the delay itself in the deploy loop, not its jitter. 149M also 1/12 at 5-15 ms on a 13 s trot here (0/8 before on
  8 s): the deploy loop is harsher for every brain, the GPU ones more. Ruled out so far: DEEP's body, our motor model
  (fade, power), torque limits beyond the first half, 1 vs 2 ms physics, delay jitter. Left: the loop's actuation (an
  explicit PD torque each substep clipped at the ctrl range, vs the training env's actuators and motor model), and the
  start (1 s settled on the stance vs the env's start inside the clip). (A no-settle switch, LOCKSTEP_NOSETTLE, only
  shows a t=0 drop artifact.)
- **9 Nov 13:30 UTC. CORRECTION to probes 3 and 4.** LOCKSTEP_OURS=1 was invalid: this loop writes motor TORQUES into
  d.ctrl, but our training model's actuators are position servos (gain kp, bias -kp q - kv qd, ctrlrange = joint range),
  so the "our body" numbers of 8 Nov 21:30 and 9 Nov 09:30 measured a broken robot; disregard them. Converting our
  actuators to plain motors (now in the switch) still gives gtq_316 11/12 falls on a 13 s trot at 15 ms with either
  integrator, so the switch is still not a faithful copy of our training body: do not use it until it reproduces the
  training env (149M and gtq_316 0/12 there). Facts that stand: our model uses the implicitfast integrator, DEEP's Euler
  (LOCKSTEP_INTEGRATOR switch); on DEEP's model gtq_316's 13 s trot at a fixed 15 ms is 0/12 with either integrator,
  while the 41 s routine (7/12) and trot -> sit (9/24) fail. The failures are in the transitions and longer routines,
  not in a steady trot. Next: probe on DEEP's model only, with the routine and trot -> sit plans.
- **9 Nov 17:30 UTC. Lite3 GPU gap FOUND: the velocity estimate.** On DEEP's model, gtq_316, 14 s trot, 12 tries: true
  velocity and height 0/12 at 0 ms and 0/12 at 3-6 ms; the robot program's leg estimate (--estimate) 1/12 at 0 ms and
  9/12 at 3-6 ms (falls ~10 s in); clip reference velocity (no measurement) 4/12. 149M with the estimate at 3-6 ms 0/12.
  The GPU-trained Lite3 brain leans on its velocity input; the estimate's errors plus delay tip it. The Go2's GPU brains
  are trained blind to velocity (--blind-vel) and transfer: the Lite3 GPU run gets the same. Earlier "trot passes" were
  without --estimate or shorter than ~8 s. Kaggle GPU bolt-lite3-gpu-blind prepared: warm 149M, --blind-vel, torque
  x0.8-1.4, drive x1.0-1.5, overspeed 1.0, 0-60 ms, 3 h -- NOT started: the weekly GPU quota ran out (30 h; the
  accidental privhist re-run of 8 Nov cost part of it). Kernel kept in scratchpad/k_lite3bl for the next reset.
- **9 Nov 19:30 UTC. Probe before training: 149M without its velocity input.** DEEP's model, leg estimate, 3-6 ms, 12
  tries: 149M as is, 14 s trot 0/12 and the routine 0/12; the same brain with inputs 6:9 zeroed falls 12/12 in both, at
  ~2.5 s. So a --blind-vel run warm-started from 149M starts from a brain that falls at once. Instead of spending GPU
  hours on relearning: `train/distill_blind.py` copies 149M into a student that never sees velocity (the student drives,
  149M with true velocity labels every step; DAgger). `--blind-vel` now also exists in the CPU trainer
  (BOLT_BLIND_VEL=1 zeroes inputs 6:9 in ImitateEnv for training and exams). Results below when measured.
- **9 Nov 20:00 UTC. Blind student, snapshot at iteration 7 of 60 (still copying).** Falls on DEEP's Lite3 model with
  the leg estimate, 149M -> student: trot 20 s at 10-19 ms 2/24 -> 0/24; trot 20 s at 3-6 ms with 5% missed steps
  7/24 -> 0/24; routine at 10-19 ms 2/24 -> 0/24; canter routine at 3-6 ms 9/12 -> 2/12; trot -> sit 1/48 -> 0/48;
  routine at 3-6 ms 0/24 both. Our exam (true physics): tracking 0.778 -> 0.762, steer 0.534 -> 0.521, canter
  tracking 0.453 -> 0.349 with canter falls 0 -> 0.25, moving sit/lie 0/48 both. DEEP's simulator live (41 s routine):
  steps with a joint over 1.2x rated 12 and 15 (149M, 2 runs) against 11 and 11 (student, 2 runs); all finished.
  Reading: not seeing velocity makes the brain sturdier under delay and missed steps (the estimate's error can no
  longer reach it) at a small cost in tracking; the canter is the open part. Final snapshot is tested before any
  change of the default brain. All simulation.
- **9 Nov 20:40 UTC. Blind student, final (60 rounds, 68 min on 4 cores): adopted as the Lite3 robot brain,
  `models/bolt-lite3-dog-blind-149M`.** Falls on DEEP's model with the leg estimate, 149M -> blind: trot 20 s at
  10-19 ms 2/24 -> 0/24; trot 20 s at 3-6 ms with 5% missed steps 7/24 -> 0/24; routine at 10-19 ms 2/24 -> 0/24;
  routine at 3-6 ms 0/24 both; routine ending in a canter at 3-6 ms 13/24 -> 1/24; trot -> sit at 5-15 ms 1/48 ->
  0/48; pace/walk/turns 0/12 both; recovery 48/48. Live in DEEP's simulator, steps with a joint over 1.2x rated per
  41 s run: 12, 15, 12 -> 5, 7, 4. Our exam: tracking 0.778 -> 0.775, falls 0 both, moving sit/lie 0/48 both, trot
  steering 0.636 -> 0.633. Worse: steering in a canter 0.433 -> 0.255, with 4 of 8 steered canters falling against
  1 of 8; so the canter stays a trot by default. Defaults switched in run_go2, check_ready, lockstep_vendor,
  lockstep_recovery; golden recorded and in the push checks; 149M kept. Next: Kaggle CPU fine-tune with
  `--blind-vel` from this brain, canter drawn 3x, to win back the steered canter. All simulation.
- **9 Nov 21:50 UTC. More copying does not fix the canter (not adopted).** 60 more rounds from the blind brain with
  the canter drawn 3x (`distill_blind.py --student`): action error 0.024 -> 0.023; our exam tracking 0.766, canter
  falls 1 of 4, steered canter 0.318 with 3 of 8 falling (blind brain: 0.255, 4 of 8; 149M: 0.433, 1 of 8); canter
  routine on DEEP's model 2/24 (blind brain 1/24). Within the noise of 8 tries: no gain. The copy is as close to
  149M as it gets; a canter that works without velocity has to be learned, which is what Kaggle lite3-dog-21..23
  (PPO with --blind-vel, pushed 20:38 UTC) are for. Simulation.
- **10 Nov 07:30 UTC. Kaggle lite3-dog-21..23 (PPO with --blind-vel from the blind brain, 24-28M steps each): not
  adopted, and why the first look misled.** Each run's best/ showed the steered canter fixed (steer_canter 0.42-0.45,
  falls 0-2 of 8), but best/ is picked on that same 8-try exam. On fresh seeds (exam_steer seed 7777, 24 tries):
  149M 0.392 with 3 falls; blind brain 0.270 with 8; dog-21 0.273 with 6; dog-22 0.309 with 7; dog-23 0.278 with 6;
  last checkpoints no better. So no gain. On DEEP's model dog-23 has no fall in all 7 stress tests (canter routine
  0/12), dog-21 one trot fall in 24 routines, dog-22 one canter fall in 12: the same record as the blind brain.
  Lesson: a brain picked on an exam must be judged on other seeds.
- **10 Nov 07:50 UTC. Cause of the steered-canter falls: the forward-speed input.** 149M in the fresh-seed steered
  canter with single inputs zeroed: forward speed 14 falls of 24 (steer 0.151); sideways speed 5 (0.289); vertical
  3 (0.360, as untouched); forward and sideways 23. A brain that sees one frame cannot know how fast it runs. The Go2
  brain has the same blindness but sees its last 3 frames. Next: `distill_hist.py`, the copy with 3 frames of history
  (the runtime already supports it). Simulation.
- **10 Nov 08:50 UTC. Blind copy with 3 frames of history: the steered canter comes back; kept as a candidate.**
  `distill_hist.py` from 149M, 60 rounds, 54 min on 4 cores, action error 0.018 (blind copy: 0.024):
  `models/bolt-lite3-dog-blind-hist-candidate`. Fresh-seed steered canter (24 tries): 0.385 with 5 falls (149M 0.392
  with 3; blind brain 0.270 with 8). Our exam: tracking 0.777, steering 0.540 (149M 0.778 / 0.534), no falls, moving
  sit/lie 0/48. DEEP's model with the leg estimate: 0 falls in all the trot, routine and sit tests; routine ending in
  a canter 4/36 (blind brain 2/36, 149M 13/24); recovery 48/48. Live in DEEP's simulator 3/3, steps over 1.2x rated
  8-12 a run (blind brain 4-7, 149M 12-15). Not made the default: the Lite3's canter runs as a trot anyway, and for
  the default routine the blind brain is as safe at lower joint speeds. Next: when the GPU quota is back, train
  from this brain with history, the over-speed penalty and --blind-vel. Simulation.
- **10 Nov 16:15 UTC. The CPU trainer learns with input history; Kaggle lite3-dog-24..26 pushed.** `train_imitate.py
  --history K` (imitate_env.HistoryEnv: the last K steps' gravity, gyro, joints and previous action, the layout the
  robot program already builds); `distill_hist.py` now also copies the value head and writes an SB3 folder, checked
  equal to its params.pkl (differences 0). Start brain `models/bolt-lite3-dog-blind-hist-start` (25 more copying
  rounds; fresh-seed steered canter 0.349 with 6 falls of 24, the candidate 0.385 with 5: the same within noise).
  Jobs, ~10.5 h each on Kaggle CPU: 24 = 149M's settings with --history 3 --blind-vel; 25 = over-speed penalty x2
  (the history brain runs its joints faster live: 8-12 steps against the blind brain's 4-7); 26 = lr 3e-5. To be
  judged on fresh seeds. Simulation.
- **11 Nov 03:50 UTC. Kaggle lite3-dog-24..26 (PPO with history, 15-23M steps): a real gain, judged on fresh seeds.**
  Fresh-seed steered canter (24 tries; blind robot brain 0.270 / 8 falls, 149M 0.392 / 3): dog-24 0.402 / 2, dog-25
  (over-speed penalty x2) 0.408 / 6, dog-26 (lr 3e-5) 0.457 / 1. Live in DEEP's simulator, steps over 1.2x rated a run
  (robot brain 4-7): dog-24 12-18 and ONE SPEED STOP at 8 s (joint 11 at 1.37x rated), dog-26 13-21, dog-25 5-9 over 6
  runs. So history lets PPO fix the steered canter, and the penalty at 2.0 is what keeps the joints slow.
  dog-25 in full (`models/bolt-lite3-dog-hist-penalty-164M`): the 7 stress tests 0 falls, canter routine 0/36 (robot
  brain 2/36), moving sit/lie on DEEP's model 0/120, recovery 48/48, our exam tracking 0.783 and steering 0.618
  (robot brain 0.775 / 0.444). Its flaw: 2 plain-trot falls of 76 in our randomized exam (robot brain 0 of 76). Not
  adopted for that reason; best candidate. dog-26 kept as `models/bolt-lite3-dog-hist-fast-172M` to train on.
  Next (pushed): dog-27 = dog-25 on with the trot drawn 3x; dog-28 = dog-26 on with the penalty at 3.0; dog-29 =
  dog-25 at lr 3e-5. Simulation.
- **11 Nov 15:20 UTC. Kaggle lite3-dog-27..29; dog-28's last checkpoint is the new Lite3 robot brain,
  `models/bolt-lite3-dog-hist-196M`.** dog-28 = hist-fast-172M on with the over-speed penalty at 3.0, lr 1e-5, 25M
  steps. Judged on fresh seeds (last checkpoint, which no exam picked): steered canter 0.501 with 1 fall of 24 (blind
  robot brain 0.270 / 8; 149M 0.392 / 3); plain trot, walk, pace 0 falls of 64 each. DEEP's model with the leg
  estimate: the 7 stress tests 0 falls; routine ending in a canter 0/36; walk/trot -> sit/lie 0/72; recovery 48/48.
  Our exam: tracking 0.779, steering 0.587, no falls, moving sit/lie 0/48. Live in DEEP's simulator 6/6, steps over
  1.2x rated 6, 4, 12, 5, 3, 5 (blind brain over 6 runs: 1-7), no stop: the penalty at 3.0 brought 13-21 down.
  Not solved: steered canter falls 2 of 8 on the exam's own seeds; the canter stays a trot by default. Defaults,
  golden and push checks switched; blind-149M kept.
  The others: dog-27 (164M with trot x3) steered canter 0.331 / 6 falls, trot 0 of 64; dog-29 (164M at lr 3e-5)
  0.391 / 6, trot 0 of 64; dog-28's best/ 0.439 / 4. Not adopted. Simulation.
- **12 Nov 02:40 UTC. Kaggle lite3-dog-30..32 (on from hist-196M: same settings, penalty 4.0, steering reward 0.3):
  no gain, not adopted.** Fresh-seed steered canter, best/ and last checkpoint: dog-30 0.431 / 3 falls and 0.444 / 4;
  dog-31 0.431 / 5 and 0.362 / 6; dog-32 0.454 / 3 and 0.487 / 4. The robot brain hist-196M: 0.501 / 1, and on a
  second fresh seed (555) 0.487 / 1, so its number holds. More of the same training has stopped helping.
- **12 Nov 02:40 UTC. Cause of the remaining steered-canter falls: the canter only turns left.** hist-196M, 24 tries
  per command (seed 31337): asked for +0.5 / +1.0 rad/s it turns 0.41 / 0.61 with 0 falls of 48; asked for -0.5 /
  -1.0 it turns -0.07 / -0.32 with 2 falls of 24 each. The dog's canter clip leads with one leg. (The trot is lopsided
  the other way: -0.8 gives -0.65, +0.8 gives 0.37, no falls.) Next: `train_imitate.py --steer-canter-right p` draws
  right turns in most steered canters; Kaggle lite3-dog-33..35. Simulation.
- **12 Nov 14:05 UTC. Kaggle lite3-dog-33..35 (right-turn canter practice from hist-196M): no gain, not adopted.**
  Last checkpoints, 24 tries per command: canter asked for -0.5 rad/s still turns -0.08 / -0.05 / +0.02, for -1.0
  -0.31 / -0.32 / -0.16, with 1-2 falls of 24; and left turns, 0 falls of 48 before, now fall 2-3 of 24 each.
- **12 Nov 14:05 UTC. What the canter exams were really measuring, and the canter's real weakness.**
  (1) Our exam starts every try mid-stride. Straight canter, 64 fresh tries: 149M 4 falls, blind-149M 11, hist-196M
  5 (the usual 4-try exam showed 0). All 12 falls in 192 probed tries of hist-196M began from stride phase 0.29-0.62
  and came within 1.4 s; none from the other two thirds of the stride. A robot never starts a canter in mid-flight.
  (2) The realistic test, on DEEP's model with the leg estimate, canter entered from a trot (new `lockstep_vendor.py
  --steer`): trot:6,canter:8 at 3-6 ms, 24 tries each: steer +0.5 0 falls, +0.8 0, -0.5 0, -0.8 1.
  (3) Plan trot:6,canter:8,trot:4,canter:6,walk:4, 48 tries each: straight at 3-6 ms 0 falls; straight at 10-19 ms 3;
  left 0.8 at 10-19 ms 10; straight at 3-6 ms with 5% of steps missed **19**. The trot alone has 0 of 24 in those
  conditions. So the canter breaks when control steps are missed or late, and the CPU training env never dropped a
  step. The canter stays a trot by default.
  Done: `train_imitate.py --drop-p` (ImitateEnv: with that chance the motors keep their last targets; never in
  exams); run_go2 holds right-turn commands at straight ahead in the Lite3's canter (controller.steer_limits).
  Pushed: Kaggle lite3-dog-36..38 (drop 5%, drop 10%, drop 5% at lr 3e-5). Simulation.
- **13 Nov 01:25 UTC. Kaggle lite3-dog-36..38 (5% / 10% of steps dropped in training, from hist-196M): the canter
  improves, not solved, nothing adopted.** Canter from a trot on DEEP's model, 48 tries, 5% of steps missed (hist-196M:
  19 falls): dog-36 15, dog-37 13 (best/ 17), dog-38 (lr 3e-5) **8**. dog-38's last checkpoint in the other conditions:
  3-6 ms 1 (196M 0), 10-19 ms 0 (3), left 0.8 at 10-19 ms 2 (10); trot 20 s with missed steps 0/24. Kept as
  `models/bolt-lite3-dog-hist-drop-215M` to train on. Pushed: lite3-dog-39..41 from it (same; 10% dropped; canter
  6x). The robot brain stays hist-196M. Simulation.
- **13 Nov 12:45 UTC. Kaggle lite3-dog-39..41 (on from hist-drop-215M with dropped steps; 20-23M more steps each): no
  further gain; this line is stopped.** Canter from a trot on DEEP's model with 5% of steps missed, 48 tries, last
  checkpoints: dog-39 9 falls, dog-40 (10% dropped) 8, dog-41 (canter 6x) 9; hist-drop-215M had 8, hist-196M 19.
  Dropped-step training halved the falls in one round and then stopped helping. Nothing adopted; the robot brain
  stays hist-196M and the Lite3's canter stays a trot by default. No new Kaggle round is started: the cause of the
  remaining falls is not known, and more Lite3 simulator work no longer moves the project (the open question is a
  real robot). Kaggle CPU is idle on purpose until there is a new measured cause or a real-robot report. Simulation.
- **14 Nov 00:55 UTC. The Go2 brain on the tougher tests built this week: 1 fall in 384 tries.** Unitree's go2.xml,
  robot inputs with the leg estimate, plan trot:6,canter:8,trot:4,canter:6,walk:4,sit:5,lie:6,getup:6,trot:5 (50 s),
  48 tries per condition, falls: 0-6 ms 0; 10-19 ms 0; 5% of steps missed 0; 10% missed 1 (in the lie); 5% missed
  with 10-19 ms 0; steering left 0.8 at 10-19 ms 0; right 0.8 at 10-19 ms 0; right 0.8 with 5% missed 0. The Go2's
  canter, both turns included, holds where the Lite3's breaks. What differs is the recipe: the Go2 brain is from the
  GPU trainer (1.4 billion steps, privileged critic, 5% dropped steps, 3 frames of history, varied torque and speed
  limits); the Lite3 brain is from 200M CPU steps without a privileged critic.
- **14 Nov 00:55 UTC. GPU quota back; Lite3 GPU run with the Go2's recipe started.** Kaggle `bolt-lite3-gpu-hist`:
  mjx_imitate from hist-196M (warm start checked locally: reward 1.02 at step 0), --history 3 --blind-vel --drop-p 0.05
  --privileged-critic --overspeed-weight 3.0, speed x1.0-1.5, torque x0.8-1.4, 0-60 ms, 6 h. Win: the canter from a
  trot on DEEP's model <= 2 falls of 48 with 5% missed steps and <= 1 in the other three conditions, everything else
  kept. (The old `bolt-lite3-gpu-blind` kernel, one frame and no dropped steps, was also started by the 4-hourly
  check at 00:42 and uses 3 GPU hours; its recipe is the one already shown to be limited.) Simulation.
- **14 Nov 07:50 UTC. Lite3 GPU run with the Go2's recipe: adopted as the Lite3 robot brain,
  `models/bolt-lite3-dog-gpu-829M`.** Kaggle `bolt-lite3-gpu-hist`, 632M steps in 6.5 GPU hours from hist-196M; the
  last checkpoint. Canter from a trot on DEEP's model, 48 tries, hist-196M -> this: 3-6 ms 0 -> 0; 10-19 ms 3 -> 0;
  left 0.8 at 10-19 ms 10 -> 0; 5% missed steps 19 -> 1. Harder: 10% missed 4; 5% missed with 10-19 ms 3; left 0.8
  with 5% missed 2. The 7 stress tests 0 falls; recovery 48/48; live in DEEP's simulator 6/6 with 1-2 steps over 1.2x
  rated (hist-196M 3-12), and 3/3 with the canter on (0-3 steps). Our exam: tracking 0.825 (0.779), steering 0.541
  (0.587), no falls; trot/walk/pace 0 of 64 fresh tries; moving sit/lie 0/48; fresh-seed steered canter 0.540 and
  0.524 with 2 falls of 24 each.
  New weakness found while testing it (present in hist-196M too, never tested before): steering RIGHT at 10-19 ms.
  Trot turning right 0.8: 15 falls of 48 (hist-196M on the canter plan: 26); right 0.6: 5; right 0.4: 0; right 0.8
  at 6-10 ms: 0; left 0.8 at 10-19 ms: 0. run_go2 now caps the Lite3's right turns at 0.4 rad/s (canter: held at
  straight ahead). The canter stays a trot by default (still 2-4 falls of 48 under the hardest conditions).
  GPU spent this week so far: 6.5 h on this run + 3 h on the old `bolt-lite3-gpu-blind` kernel (not examined beyond
  this note; its recipe lacks history and dropped steps) = 9.5 of 30 h. Simulation.
- **14 Nov 07:55 UTC. Second Lite3 GPU run started: `bolt-lite3-gpu-hist2`.** From gpu-829M, the same recipe with the
  steering reward at 0.3 (default 0.15), 6 h. Aimed at the measured weak spot: right turns at 10-19 ms (trot right
  0.8: 15 falls of 48; 0.6: 5). Win on DEEP's model: trot right 0.8 at 10-19 ms <= 2 of 48 with everything else
  kept. Simulation.
- **14 Nov 15:45 UTC. Second Lite3 GPU run: right turns fixed; adopted, `models/bolt-lite3-dog-gpu-1461M`; the
  Lite3's canter is on by default.** `bolt-lite3-gpu-hist2`, 632M more steps in 6.5 GPU hours from gpu-829M with the
  steering reward at 0.3; the last checkpoint. DEEP's model with the leg estimate, falls of 48, gpu-829M -> this:
  trot right 0.8 at 10-19 ms 15 -> 0; right 0.6 5 -> 0; left 0.8 0 -> 0; canter plan steering right 0.8 at 10-19 ms
  15 -> 0; canter 3-6 ms 0 -> 0; 10-19 ms 0 -> 0; left 0.8 at 10-19 ms 0 -> 0; 5% missed 1 -> 0; 10% missed 4 -> 2;
  5% missed with 10-19 ms 3 -> 1. The 7 stress tests 0 falls; recovery 48/48; live routine 6/6 with 0-1 steps over
  1.2x rated (829M: 1-2). Live with canters and steering (49 s, left 0.6 and right 0.8), with the new defaults: 6 of 7
  finished (0-1 steps over); one was ended at 42.8 s by run_go2's stale-data stop (state 81 ms old: the simulator
  hiccuped on this shared machine; not a fall). Our exam: tracking 0.828, steering 0.540 (trot 0.768), no falls;
  trot/walk/pace 0 of 64 fresh; moving sit/lie 0/48; fresh-seed steered canter 0.584 and 0.518 with 1 fall of 24.
  run_go2: the Lite3 canters by default; no cap on right turns in the trot; right turns still held at straight
  ahead in the canter (it turns left only: -0.8 asked, -0.16 reached). Still open: the canter with 10% missed steps
  (2 of 48), the canter's right turn. GPU used this week: 16 of 30 h. No further run is planned without a measured
  reason: both robots' brains now pass every test in this file. Simulation.
- **9 Nov 19:30 UTC. First-run kit for the Go2: `deploy/first_run.py`.** Listens only (no publisher in the file). Checks
  with PASS/FAIL: message rate and gaps; flat and still; guided tilts (gravity direction as in our simulator, gyro
  consistent with the orientation); 12 guided hand moves (right joint, right sign; expected signs from our leg
  kinematics); a 10 s shadow run of the brain with its answers thrown away (timing, limits). Writes report.json +
  samples.npz. `--selftest` (made-up data: flipped signs, swapped legs, swapped joints, wrong gyro sign, a roll given as
  a pitch, a 50 ms gap must all FAIL) passes; against Unitree's simulator with `--auto`: link 200/s, still, shadow 0
  late steps of 500, brain step 0.5 ms. The hand moves themselves are untested on hardware (no robot). Lite3: not
  covered. HARDWARE_DAY.md step 1 now uses it. `.github/workflows/bolt-checks.yml` runs the two golden checks and this
  self-test on every push (numpy + mujoco + gymnasium only).
