# bolt-go2-damp-798M (simulation): the Go2 brain for a real robot (1 Nov, replaces 695M)

From models/bolt-go2-blind-695M, 103M more steps on a Kaggle T4 (3.5 h, run bolt-go2-damp2): the same recipe (all 9
clips, switches, steering +-0.8, root-global 0.5, 0-40 ms delay, --blind-vel, 5% lost commands, moving sit/lie starts
0.5) with **joint damping drawn per try, log-uniform 0.1-2.0** (`--damping-range 0.1 2.0`): the real Go2's joint
damping is unknown, and at Unitree's default 0.1 the 695M brain shook its calves past their rated speed.
76 inputs (10 clip slots; the 10th, jump, is untrained here). No velocity input.

Real-robot rehearsal (train/deploy/lockstep_vendor.py: Unitree's go2.xml, damping 0.1, 5 ms, robot-only inputs; 41 s
trot / sit / lie / getup / walk / canter; 10 tries each):

| | 695M | **798M (this)** |
|---|---|---|
| falls, plain | 0/10 | **0/10** |
| falls, 5% missed control steps | 2/10 | **0/10** |
| falls, 0-19 ms command delay | 0/10 | **0/10** |
| knee steps over 1.2x rated (run_go2.py would stop) | 1448 of 20500, worst 2.17x | **96, worst 1.89x** |
| the same with 5% missed steps / with delay | - | 181 / 59 |
| the same at damping 0.5 | 76, worst 1.80x | **11, worst 1.54x** |
| fall -> self-right -> stand -> trot again (side/back/tipped/shove) | 40/40 | **40/40** |

Our exam (76-input exam; imitate_env.exam, 4 tries per clip): mean 0.819 -> 0.817, steer 0.678 -> **0.708**, canter
0.52 -> 0.531, trot 0.79 -> 0.79, falls 0% -> 1.9% (one trot -> sit). With 12 tries of each moving posture start:
trot -> sit falls 695M 2/12, 798M 3/12; trot -> lie 0/12 vs 1/12; walk -> sit / lie 0/12 both. Sitting down from a
trot is the open weakness of both brains.

Still not zero: 96 steps over 1.2x rated at Unitree's default damping: the hanging test on hardware day (speed stop
on) tells whether the real robot is that loose. Use: `train/deploy/run_go2.py` (default brain). Simulation only. Dog
motion clips are CC BY-NC (non-commercial).
