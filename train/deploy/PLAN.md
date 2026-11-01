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
