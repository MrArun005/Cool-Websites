# Hardware day: first run of a Bolt brain on a real Unitree Go2 EDU

PLAN.md phase 4, as a checklist. Every step has a stop condition; do not skip ahead. Two people: one at the
keyboard, one next to the robot with the remote (its emergency stop) in hand. Nothing here calls an external AI
service; everything runs on the robot's Jetson or a laptop on wired Ethernet.

Brain: `models/bolt-go2-blind-592M/params.pkl` (no velocity input; passed the rehearsal with robot-only inputs, 0/10
falls). Program: `train/deploy/run_go2.py`. All numbers so far are simulation.

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
- [ ] Read-only check: `python deploy/read_go2.py --iface <eth> --domain 0` (never sends a command; prints the
      gravity reading, gyro and the 12 joints in OUR order 5 times a second, then the message rate). Check by hand:
  - [ ] Tilt the robot nose-up: the gravity reading changes sign the way the simulator's does.
  - [ ] Move each leg by hand: the joint in OUR order (FL, FR, RL, RR x hip, thigh, calf) changes, with the
        simulator's sign (registry `sdk_order`).
  - [ ] Message rate ~500 Hz on the robot (Unitree's simulator: 200 Hz), largest gap under 10 ms.
- Stop if: any joint maps to the wrong leg or sign, or the IMU frame differs. Fix `robot_bodies.py`, re-run step 0.

## 2. Hanging in the harness (feet 2-3 cm off the floor)

- [ ] `run_go2.py --domain 0 --iface <eth> --release-sport --gain-scale 0.5 --plan "trot:3" --log hang1.npz`
      Startup blend to the stance (2.5 s), hold 1 s, then 3 s of trot in the air.
- [ ] Watch: legs move smoothly in trot order, no buzzing, no joint near its limit; motor temperatures (LowState) flat.
- [ ] Repeat at `--gain-scale 0.8` and `1.0`.
- Known risk: in Unitree's simulator with its default joint damping (0.1) the 592M brain shakes its calves past
  their rated speed (PLAN.md log, 1 Nov 09:40); with damping 0.5-2.0 it does not. The hanging run tells which the
  real robot is: `replay_log.py hang1.npz --damping 0.1` vs `2.0`, and the calf speeds in the log.
- Stop if: oscillation, a safety stop, or a joint over its rated speed. Compare `hang1.npz` against the same plan in
  `lockstep_vendor.py` before going on.

## 3. On the floor, reduced gains

- [ ] Robot lying on the mat. `--gain-scale 0.8 --plan "sit:5,lie:6,getup:6"`: stands up from the blend, sits, lies,
      gets up. Person with the remote ready.
- [ ] `--gain-scale 0.8 --plan "walk:6"`, then `"trot:6"`. Short runs first.
- Stop if: a fall, a safety stop, or the "late steps" count above 0. Read the log before the next try.

## 4. Full gains, then commands

- [ ] `--gain-scale 1.0`, `--interactive`: walk, trot, steer 0.3, sit, lie, getup, stop. Video every run.
- [ ] Only then canter, on the longest clear stretch.
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
