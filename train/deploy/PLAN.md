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
