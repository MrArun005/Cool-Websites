# Hardware day: first run of a Bolt brain on a real Unitree Go2 EDU

PLAN.md phase 4, as a checklist. Every step has a stop condition; do not skip ahead. Two people: one at the
keyboard, one next to the robot with the remote (its emergency stop) in hand. Nothing here calls an external AI
service; everything runs on the robot's Jetson or a laptop on wired Ethernet.

Brain: `models/bolt-go2-torque-steer/params.pkl` (no velocity input; 3 steps of input history; trained with a penalty on
joint speed above rated, a faster drive and varied torque limits; on Unitree's model with robot-only inputs: 0/10 falls,
0/10 with 0-19 ms delay, 0/10 with 5% missed steps, recovery 20/20, and live in Unitree's simulator in real time the full
routine 3/3 with no speed stop. Earlier brains: `bolt-go2-overspeed-211M` (same safety, steers less), `bolt-go2-damp-798M`
(tripped the speed stop live within 6-10 s). Program: `train/deploy/run_go2.py`. All numbers so far are simulation.

## 0. Before the robot (the day before)

- [ ] It is a Go2 **EDU** (low-level motor access). Firmware updated with Unitree's app.
- [ ] Laptop or Jetson (Linux; git, cmake, a C compiler, Python 3.11 or uv), this repo checked out:
      `bash train/deploy/setup_go2.sh --sim` (about 10 minutes the first time, seconds after). It installs everything
      pinned into `bolt-deps/` and ends with `check_ready.py`: it must print **READY**.
- [ ] Rehearsal on that machine (`source bolt-deps/env.sh; cd train`): `python deploy/sim_vendor.py --seconds 60 &` then
      `python deploy/run_go2.py --plan "trot:6,sit:5,lie:6,getup:6,walk:6"` completes; then the same with
      `--interactive`. Its "late steps" count must be 0.
- [ ] On the day, robot cabled: `python deploy/check_ready.py --iface <eth>` prints READY (adds the 192.168.123.x check).
- [ ] Space: flat, high-grip mat at least 4 x 2 m, nothing within 1 m; a harness or gantry to hang the robot.

## 1. Connect, read only (motors off)

- [ ] Wired Ethernet; laptop on 192.168.123.x (Unitree's subnet). Find the interface name (`ip a`).
- [ ] Robot lying down, powered on. Sport mode will be released by `--release-sport`.
- [ ] Guided check, listens only: `python deploy/first_run.py --iface <eth> --domain 0` (about 5 minutes; it has no
      command publisher at all). It prints PASS or FAIL for each of:
  - [ ] link: state messages at 400/s or more, no gap of 20 ms.
  - [ ] still: gravity straight down, gyro and joint speeds near 0, joints in range.
  - [ ] tilt: you lift the nose, then the right side; gravity must move as in our simulator and agree with the gyro.
  - [ ] legs: you move each joint by hand when asked (12 moves; `--quick`: 4); the right joint must change, in our
        sign. Legs lying in sport mode are damped: move slowly, about a hand's width.
  - [ ] shadow: the brain runs on the live readings for 10 s, its answers thrown away: 0 late steps, answers inside
        the joint limits.
- [ ] It ends with "ready for the hanging test" and writes `first_run_<time>/report.json` + `samples.npz`. Keep the
      folder. Rehearse it the day before with `--selftest` (no robot) and against `sim_vendor.py` with
      `--domain 1 --auto`. `read_go2.py` still prints the raw readings if you want to look by hand.
- Stop if: any joint maps to the wrong leg or sign, or the IMU frame differs. Fix `robot_bodies.py`, re-run step 0.

## 2. Hanging in the harness (feet 2-3 cm off the floor)

- [ ] `run_go2.py --domain 0 --iface <eth> --release-sport --gain-scale 0.5 --plan "trot:3" --log hang1.npz`
      Startup blend to the stance (2.5 s), hold 1 s, then 3 s of trot in the air.
- [ ] Watch: legs move smoothly in trot order, no buzzing, no joint near its limit; motor temperatures (LowState) flat.
- [ ] Repeat at `--gain-scale 0.8` and `1.0`.
- Known risk: in Unitree's simulator with its default joint damping (0.1) the 592M and 695M brains shake their calves past
  their rated speed (798M much less: 96 steps over 1.2x in 41 s, worst 1.89x) (PLAN.md log, 1 Nov 09:40); with damping 0.5-2.0 it does not. The hanging run tells which the
  real robot is: `replay_log.py hang1.npz --damping 0.1` vs `2.0`, and the calf speeds in the log.
- run_go2.py's speed stop (2 Nov): damping mode when a joint stays over 1.2x its rated speed for 3 brain steps in a
  row (60 ms) or goes over 3x once; the brain is eased in from the stance over its first 0.5 s (--ramp). Single
  20 ms spikes at switches between moves are counted ("steps with a joint over 1.2x rated" at the end), not stopped.
- Stop if: oscillation, a safety stop, or a joint over its rated speed. Compare `hang1.npz` against the same plan in
  `lockstep_vendor.py` before going on.

## 3. On the floor, reduced gains

- [ ] Robot lying on the mat. `--gain-scale 0.8 --plan "sit:5,lie:6,getup:6"`: stands up from the blend, sits, lies,
      gets up. Person with the remote ready.
- [ ] `--gain-scale 0.8 --plan "walk:6"`, then `"trot:6"`. Short runs first.
- Stop if: a fall, a safety stop, or the "late steps" count above 0. Read the log before the next try.

## 4. Full gains, then commands

- [ ] `--gain-scale 1.0`, `--interactive`: walk, trot, steer 0.3, sit, lie, getup, stop. Video every run.
- [ ] Then `--remote` (the Go2's own remote; `deploy/remote.py` has the map: left stick walk/trot, right stick steer,
      A sit, B lie, Y get up, L2+B stop). The keyboard person keeps `--interactive` on too, with "stop" ready.
- [ ] Only then canter (R1 + left stick fully forward), on the longest clear stretch.
- Not today: jumps (need a camera to see the bar), self-righting (needs a cleared cage), MasterMind.
- A later session, in a cleared cage: `--recover` (a fall -> self-right -> get up -> resume; rehearsed on Unitree's
  simulator with pushes, PLAN.md log). First with the robot laid on its side by hand, gains 0.8, remote in hand.

## 5. After the session

- [ ] Copy every `.npz` log and video into `research/results/real_go2/<date>/` with a short note per run.
- [ ] Measure from the logs: `python deploy/replay_log.py <log>.npz --damping 0.1` (and 2.0, 0.5: the setting with
      the smallest joint gap is the robot's), real loop delay, motor response (target vs reached angle), foot slip. These go back into
      the simulator (PLAN.md phase 5) before the next training run.
- [ ] Write what happened, falls included, in PLAN.md's log. Real-robot numbers replace "simulation" only for what
      was actually run.

## Lite3 (DEEP Robotics), the second robot: what differs

- Brain: `models/bolt-lite3-dog-149M/params.pkl` (on DEEP's model with 3-15 ms of delay: no falls in 24 routines and 48 trot -> sit; steering 0.534; live joint over-speed 5-12 steps a run; the canter runs as a trot by default). `models/bolt-lite3-dog-canter-candidate` for runs with `--allow-canter`; `-blend-112M-w7` is the previous brain.
  It reads body velocity, from LegEstimator on the robot. Self-right brain: `models/bolt-lite3-selfright-28M`.
- Link: DEEP's Lite3_MotionSDK through our bridge, not unitree_sdk2. Build `deploy/lite3_bridge/bridge.cpp` on the
  robot's motion host (or a laptop on its network; the build line is at the top of the file), run `./lite3_bridge
  --robot <robot ip> --bolt <laptop ip>`, then `python deploy/run_go2.py --body lite3 --host <bridge host>`.
- Before the robot: `sim_lite3.py` (DEEP's own simulator) + `run_go2.py --body lite3` must stand the robot up from
  lying; `lockstep_vendor.py --body lite3 --estimate` over the 41 s plan.
- Measure the command delay in the hanging test (log a small target step on one joint and the time until the joint
  angle starts to move, minus the motor's own response in sim_lite3.py), then run with `--predict-ms <that delay>`.
  Only with a measured value: on DEEP's model 20 ms of prediction cut falls at 5-15 ms delay from 6/12 to 2/12, but
  5 ms of prediction with no delay caused 6/12 (PLAN.md 3 Nov).
- First check on the robot (motors off): turn it 90 degrees by hand and read the logged yaw rate. DEEP's header says
  degrees/s, their own deploy code treats it as rad/s; if the log shows ~90 instead of ~1.6, restart the bridge with
  `--gyro-deg`. Then lift a leg by hand: our joint angles are minus DEEP's (lite3_link.py), the logged thigh angle must
  grow as the leg swings back.
- Known gaps (PLAN.md 3 Nov): DEEP's simulator gives every motor a flat 30 N·m at any speed; the 38.8M brain falls
  sit -> lie there (less with speed-limited motors).

