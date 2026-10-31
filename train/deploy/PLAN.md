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
| Bridge: robot sensors -> brain -> motor commands | **not started** (this plan, phase 1) |
| Safety layer | **not started** (phase 1) |
| Rehearsal in the vendor simulator | **not started** (phase 2) |
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
| 1.3 obs builder | 1.2 | matches ImitateEnv to 1e-6 |
| 1.4 brain runner | - | matches MjxImitatePolicy |
| 1.5-1.7 loop, safety, logger | 1.1-1.4 | runs 10 min without a fault in unitree_mujoco |
| 2 rehearsal | phase 1 | exam passes, or gaps listed and fed back |
| 3 input gap | GPU | brain with real sensors only passes phase 2 |
| 4 hardware day | a Go2 EDU | walks on the floor |
| 5 reliability | 4 | 30 min of mixed commands without a fall |

All results are simulation until phase 4. Nothing here calls an external AI service.
