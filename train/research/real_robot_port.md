# Porting MasterMind to a real Unitree A1 or Go2: the plan

Status, 2026-10-28: **a plan only, nothing retrained.** Every number below is **in simulation** and comes with the command that
re-makes it. The robot models are the official MuJoCo Menagerie ones, copied into `train/robots/`.

## TL;DR

* Bolt is **not** an A1 or a Go2. It weighs about the same (12.0 kg, A1 12.45 kg, Go2 15.2 kg) and has the same 0.2 m
  thigh and shin as the A1. It also uses the A1's 33.5 N·m motor model. But four things differ, and no brain carries
  over without retraining:
  * **Knee direction.** Bolt's rear knees bend forward, like a dog's hind leg. Every A1 and Go2 knee bends backward.
  * **Mass split.** Bolt has 9 kg in the torso and 3 kg in the legs. The A1 has 4.7 kg and 7.7 kg (the motors sit in
    the legs). Swinging a leg is about 2.5 times heavier.
  * **Stance.** Bolt stands about 0.44 m tall on a 0.30 m wide hip line. The A1 and Go2 stand about 0.27-0.32 m tall,
    with thigh joints 0.26-0.28 m apart and a 8-10 cm hip offset.
  * **Inputs a real robot does not measure.** The brain reads body linear velocity and body height directly.
* So the port is: new body xml, re-retargeted clips, **retrain every expert**, then re-run the MasterMind DAgger recipe on
  the new experts. The code path is the same and only body-specific constants change. Estimate: about 2-3 Kaggle GPU
  sessions plus 20-30 CPU-core-hours (table below).
* What is ready now: the stress test (`test_master.py` flags) and the second-engine check (`sim2sim.py`) are tools that
  can be pointed at the new body. They tell us how much margin a retrained brain has before any hardware.
  * On Bolt today: noise, ±20% gains and the brake cost at most 3 points.
  * **20 ms of action latency drops switches from 83% to 50%** (section 4).

## 1. The bodies side by side

`cd train && python robots/bodies.py --json robots/bodies.json` measures every row from the MuJoCo models: bolt.xml,
`robots/unitree_a1/a1_nomesh.xml` and `robots/unitree_go2/go2_nomesh.xml`. The `_nomesh` files are the Menagerie files
with the visual meshes removed; the physics is unchanged. Motor speed ratings are not in the xml files, and their sources
are cited in `bodies.py`.

| | Bolt | Unitree A1 | Unitree Go2 |
|---|---|---|---|
| total mass | 12.0 kg | 12.45 kg | 15.2 kg |
| torso / legs | 9.0 / 3.0 kg | 4.7 / 7.7 kg | 6.9 / 8.3 kg |
| thigh / shin | 0.200 / 0.200 m | 0.200 / 0.200 m | 0.213 / 0.213 m |
| hip spacing, front-back × left-right (abduction joints) | 0.44 × 0.30 m | 0.366 × 0.094 m | 0.387 × 0.093 m |
| thigh-joint width (with the hip offset) | 0.30 m | 0.264 m | 0.284 m |
| knees | front backward, rear **forward** | all backward | all backward |
| standing height (body) | ~0.44 m (dog stance) | 0.27 m (Menagerie "home") | 0.27 m ("home"); ~0.30-0.32 m typical stand |
| abduction range | ±0.84 rad | ±0.80 rad | ±1.05 rad |
| front hip range | -1.2 … 2.6 rad | -1.05 … 4.19 rad | -1.57 … 3.49 rad |
| knee range | -2.75 … -0.20 rad | -2.70 … -0.92 rad | -2.72 … -0.84 rad |
| actuator in the model | position servo kp 45, kv 1.2 | position servo kp 100 (Menagerie) | torque |
| torque limit | 33.5 N·m | 33.5 N·m | 23.7 N·m hip/thigh, 45.4 N·m knee |
| joint speed | 21 rad/s rated, drive fades to 0 at 25 rad/s; 420 W | 21 rad/s | 30.1 hip/thigh, 15.7 knee (URDF limits) |
| foot | sphere r 2.5 cm, μ 1.0, condim 3 | sphere r 2.0 cm, μ 0.8, soft, condim 6 | sphere r 2.2 cm, μ 0.8, soft, condim 6 |
| joint damping / dry friction | 0.05 / 0.02 | 2.0 / 0.2 | 2.0 / 0.2 |

Licence: the Menagerie A1 and Go2 directories are **BSD-3-Clause, © Unitree Robotics** (each `LICENSE` file), not
Apache-2.0. Menagerie's top-level code is Apache-2.0, but each robot directory carries its own licence. Keep the
`LICENSE` files next to the xml.

Size: the xml, scene, README, CHANGELOG and LICENSE files are about 70 kB. The visual meshes (19 MB for the A1, 28 MB for
the Go2) are kept out of git (`.gitignore`). `bash train/robots/fetch_meshes.sh` fetches them at the pinned Menagerie
commit f3b6013 if you want the published `a1.xml`/`scene.xml` to load with visuals.

Two smaller differences:

* **Knee range.** The A1 and Go2 knees cannot straighten past about 50°: their upper knee limit is -0.92/-0.84 rad,
  against Bolt's -0.20. Clips that use a nearly straight leg will be clamped: stretching, the sit's front legs, and the
  take-off of a jump.
* **Joint friction.** Their joints have 40x Bolt's damping and 10x its dry friction. That is closer to a real geared
  motor, and Bolt's joints are optimistic here.

## 2. What changes to retarget and retrain

### Environment (train/bolt_env.py and the envs built on it)

* **Body.** Add a `--body a1|go2` that loads the Menagerie xml.
  * Replace the Go2's torque `<motor>`s with our position servo: kp and kd as below, and `forcerange` per joint.
  * Keep `bolt_env.physics()` (speed-dependent torque limit, brake). Give `MOTOR` per-joint torque and speed: the Go2 knee
    is 45.4 N·m at 15.7 rad/s, its hips 23.7 N·m at 30.1 rad/s.
  * Keep the Menagerie foot contacts (soft, condim 6) and their joint damping and friction.
* **Gains.** Bolt's kp 45 / kv 1.2 were tuned for Bolt. Public A1/Go2 policies are deployed with kp ≈ 20-40, kd ≈ 0.5-1
  (legged_gym, unitree_rl_gym). Train with the gains the robot will run, randomized ±20%.
* **Constants that encode Bolt's shape.** These are all in a few files (bolt_env.py, imitate_env.py, skills.py,
  jump_env.py, getup_env.py) and need re-measuring with `poses.py` on the new body:
  * the poses STAND_POSE, SIT_POSE, LIE_POSE and `STANCES["dog"]` (= imitate_env.DOG)
  * the target heights in `TARGETS`
  * the fall detector: `FALL_UP` stays, but the body geoms touching the floor need the new geom ids
  * the "recovered" test (height > 0.38 m becomes about 0.25 m)
  * hurdle heights: 50 cm is 1.1x Bolt's standing height but 1.7x the Go2's. Start the Go2 at 10-25 cm.
* **Joint order and signs.**
  * Bolt: FL, FR, RL, RR × (abd, hip, knee), with the rear hip and knee mirrored (`bolt_env.pose`).
  * Unitree SDK: FR, FL, RR, RL × (hip, thigh, calf), all knees negative.
  * Write one mapping (index permutation + sign + offset) and test it both ways in the env: the stand pose must look
    identical in MuJoCo after the round trip.
* **Robust training is on by default.**
  * The motor brake (HANDOFF 0c).
  * Observation noise at the `env.NOISE` levels.
  * **An action delay of 0-40 ms, randomized per episode.** This is the most important one (section 4): today's
    brains lose a third of their switches at 10 ms.
  * kp ±20%, mass ±15% plus a 0-2 kg payload, friction 0.4-1.3.

### Observations the real robot cannot give

| input (MasterMind's 84) | real A1/Go2 | plan |
|---|---|---|
| gravity in body frame (3) | yes, from the IMU quaternion | as is; noise 0.05 |
| angular velocity (3) | yes, the gyro | as is; noise 0.2 rad/s |
| **body linear velocity (3)** | **no sensor** | a state estimator (Kalman filter on IMU + leg kinematics + contact), or retrain the experts and student without it. Its noise must be trained with: 0.1 m/s at least, plus drift |
| **body height (1)** | **no sensor** | estimate from leg kinematics of the feet in contact, or drop it (most published A1/Go2 policies do without it) |
| joint angles, speeds (24) | yes, the encoders | as is; noise 0.01 rad, 1.5 rad/s |
| previous output (12), command (10), time since command (1) | yes, computed | as is |
| reference motion (21) | yes, computed from the clip file on the robot | ship `clips.json` with the brain |
| **hurdle (6)** | **needs perception** | a depth camera or the Go2's LiDAR plus a bar detector. Until then, no jumps on hardware |

### Retargeting the dog clips (train/dogmotion/retarget.py)

The retargeter already follows Peng et al. 2020, which retargeted the same kind of dog motion capture onto Unitree's
Laikago, so most of it carries over. It works from body pose plus paw keypoints, scaled leg pairs, and leg IK.

* **Scale (step 3).** Map the dog's shoulder and hip height onto the new stance (about 0.27-0.30 m), not Bolt's 0.385 m.
* **IK (step 5).** All knees backward. The rear legs solve the same triangle as the front ones, with the knee on the
  other side. Add the 8-10 cm hip offset between the abduction axis and the thigh plane.
* **Joint limits.** Re-run the kinematic check (`clips/check.json`) and read the clamping report. Expect losses where
  the knee nearly straightens (sit, jump take-off) and on the A1's ±0.80 rad abduction.
* **Licence.** The dog motion data stays CC BY-NC 4.0, so a robot running these brains is non-commercial.

### Which brains need retraining

All of them. Each expert's observation and action are in Bolt's joint frame and body dynamics, and the MasterMind
student only copies its teachers.

| brain (Bolt name) | what it does | engine | Bolt cost | A1/Go2 estimate |
|---|---|---|---|---|
| dog-motion imitation (18g/18e type, or the MJX GPU brain) | walk, trot, pace, canter, turns, sit, lie, with clip switches | Kaggle GPU, MJX (`mjx_imitate.py`) | 360M steps in 6.3 h on 2× T4 (~16.5k steps/s) | 1-2 GPU sessions (~6-11 h each): the new body plus a switch fine-tune. Menagerie ships `go2_mjx.xml` |
| posture (stand/sit/lie, getup) | stand, the tuck in the recovery | CPU PPO | 3M steps | ~1 CPU-core-hour ×4 cores |
| self-right (17e) | roll back to belly-down | CPU PPO | 16.5M steps | ~2-3 h on 4 cores |
| run-up jump (16i) | hurdle | CPU PPO (curriculum 16e→16g→16i) | ~35M steps on the last stage alone | 6-10 h on 4 cores, after the gaits; lower bars first |
| landing student (r4) | the 2.6 s after landing | CPU (PPS + DAgger) | ~3-4 h | ~3-4 h, or drop it if the new imitation brain is trained with landing starts |
| **MasterMind student** | everything, one MLP | CPU DAgger (`dagger.py --teachers <new set>`) | ~75 min on 3 cores | ~1.5-3 h on 3-16 cores |

Total: about 2-3 Kaggle GPU sessions and 20-30 CPU-core-hours (4-core Kaggle CPU sessions: about one long session).
These are scaled from Bolt's runs. A new body usually needs one or two reward and curriculum retries, so plan for
1.5x. CPU throughput is the ~2,000 steps/s of 8 envs on 4 cores (HANDOFF).

Order:
1. Body xml, the joint mapping and poses; retarget the clips.
2. Imitation on GPU.
3. Posture and self-right on CPU, in parallel with step 2.
4. Jump, once the trot exists.
5. Teacher-chain check in the MasterMind world (`check_teachers.py`).
6. DAgger.
7. Stress test and second-engine check on the new body (this commit's tools).
8. Hardware.

## 3. Deployment checklist for a real Go2 (EDU) or A1

**Control.**

* Brain at **50 Hz**, as trained. The PD loop runs on the motor drivers.
* Keep the time from reading the state to the motors acting as short as possible: brain on the robot or on wired
  Ethernet.
* Send low-level commands at 500 Hz (Unitree's usual rate), repeating the brain's latest targets.
* Each motor gets q target, dq target 0, Kp, Kd and feed-forward torque 0, with the trained gains.
* Measure the latency from reading state to sending the command (log time stamps) and keep it inside the trained range.

**Software path.**

* Go2: `unitree_sdk2` (C++) or `unitree_sdk2_python`, DDS over wired Ethernet (192.168.123.x).
  * Topics `rt/lowstate` (IMU quaternion, gyro, accelerometer; motor q, dq, estimated torque, temperature; foot force)
    and `rt/lowcmd` (with its CRC).
  * The built-in sport mode must be released first (the motion-switcher service), or it will fight the commands.
  * Low-level access is an EDU feature. Check this for the unit you get.
* A1: `unitree_legged_sdk` 3.x over UDP. LowCmd and LowState hold the same fields; there is a Python binding.

**Where the brain runs.**

* The brain is 84→1024→512→256→12, about 0.7M weights.
  * `export_v4/brain.json` is a plain normalizer plus MLP, so it needs numpy or ONNX only.
  * About 1 ms per step on a laptop CPU.
* It can run on the robot's own computer (Go2 EDU: a Jetson module; A1: its onboard x86 board), or on a laptop on the
  wired Ethernet. Avoid Wi-Fi for low-level control.
* The robot-side loop does this every 20 ms:
  1. read the state
  2. build the 84 inputs, with the estimator for velocity and height
  3. run the brain
  4. map Bolt's joint order to the robot's
  5. clamp
  6. send

**Safety limits, in the loop and not in the brain.**

* **Supports.**
  * First runs hang in a harness or gantry, feet just touching.
  * Then on the floor at a reduced gain scale (`--gain-scale 0.8` was measured in simulation).
  * On a flat, high-friction mat, with no hurdles.
* **Soft limits.**
  * Joint targets stay 0.05-0.1 rad inside the hardware limits.
  * Clamp the torque below the rated value: hip 23.7 and knee 45.4 N·m on the Go2, 33.5 N·m on the A1.
  * Stop on joint speed over the rated value or on motor temperature (in LowState).
* **Damping mode** (kp 0, kd 1-3) on any of:
  * tilt past 60° (the same FALL_UP as in simulation)
  * no new brain output for 60 ms (watchdog)
  * an estimator that diverges
  * the remote's emergency stop
* **Commands, phase 1.** Stand, walk, trot, sit, lie only.
  * Self-righting thrashes the legs through the full range: test it only in a clear cage.
  * Jumps need the perception block first.
* **Startup.** Begin from the robot's lying pose, interpolate over 2-3 s to the brain's first targets, then hand over.
* **Logging.** Log every 50 Hz frame (inputs, outputs, state, timing) and replay the log through the simulator
  (`sim2sim.py`'s bridge pattern) to compare.

## 4. Where Bolt's MasterMind stands today (in simulation)

Details and commands are in `train/mastermind/REPORT.md`, under "Sim-to-real stress test" and "Second simulator".

**Stress test** (v4, N=30, seed set 1000, share of switches that succeed):

| | clean | noise | 6 ms | 10 ms | 20 ms | 40 ms | lag 10 ms | gains ×0.8 | gains ×1.2 | brake | all (10 ms) | all (20 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | 83% | 83% | 77% | 67% | 50% | 18% | 69% | 84% | 80% | 82% | 52% | 43% |
| v3 | 83% | 77% | 76% | 65% | 49% | 18% | 74% | 80% | 79% | 83% | 50% | 44% |
| share of planned switches reached, v4 | 97% | 98% | 95% | 89% | 37% | 6% | 95% | 98% | 95% | 97% | 72% | 33% |

**Second engine** (PyBullet, nominal body, gaits, rest and shove sequences only):
* v4: 97% of switches in MuJoCo, 83% in PyBullet.
* Walk, trot, pace, the turns, sit and lie run without a fall in PyBullet.
* The canter falls 3 of 4 times, and recovery after a shove drops from 100% to 70%.

What this means for hardware:
* **Sensor noise, gain errors and the brake are not the problem.**
* **Latency is.** One control step of delay (20 ms) halves the success rate and makes Bolt fall during the walk itself.
  No expert or student was trained with delay. The first item of the port is to add it to every training run.
* The canter and self-righting are the skills most tied to MuJoCo's contacts. Expect them to transfer last.

These numbers describe the recipe's robustness on Bolt's body. Hardware readiness needs the retrained A1/Go2 brain,
and then the same two tests on that body.

## Top risks for a real-robot test

1. **Latency.** Measured: 10 ms of action delay takes v4 from 83% to 67%, and 20 ms to 50%, with falls in the plain
   walk. A real loop (state read → network → brain → motor driver) is easily 5-20 ms.
   * Fix: train every expert and the student with a random 0-40 ms delay.
   * Measure the robot's real loop time and run the stress table at it. Do not go to hardware until 20 ms is within a
     few points of clean.
2. **State estimation.** Body linear velocity and height are perfect inputs in simulation. The stress test's
   ±0.1 m/s / ±0.02 m white noise is harmless, but a real estimator drifts, worst after landings and shoves. That is
   not tested.
   * Fix: train with an estimator in the loop (or its error model), or retrain without these two inputs.
3. **The body and contact gap.** A second engine already costs 14 points and breaks the canter. The real A1/Go2 adds
   different leg inertia, knee direction, foot softness and 10-40x the joint friction.
   * Expect the first hardware gaits to be slower and stiffer.
   * Gate every step: stress test, then second engine, then harness, then floor.
   * Start with stand, walk, trot, sit and lie only. No jumps until perception exists. Self-righting only in a cage.
