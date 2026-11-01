# bolt-go2-blind-695M (simulation): the Go2 brain for a real robot (1 Nov, replaces 592M)

From models/bolt-go2-blind-592M, 103M more steps on a Kaggle T4 (3.5 h, run bolt-go2-blind2): the same recipe (all 9
clips, switches 0.3/s, steering +-0.8, root-global 0.5, 0-40 ms delay, --blind-vel, --drop-p 0.05) with moving
sit/lie starts raised from 0.3 to 0.5 (walk/trot -> sit/lie). No velocity input (a real Go2 has no velocity sensor).

Real-robot rehearsal (train/deploy/lockstep_vendor.py: Unitree's own go2.xml, damping 0.1, 5 ms step, flat floor,
only robot-measurable inputs; 41 s: trot 10, sit 5, lie 6, getup 6, walk 8, canter 6; 10 tries each):

| | 592M | **695M (this)** |
|---|---|---|
| falls, plain | 0/10 | **0/10** |
| falls, 0-19 ms command delay | 1/10 | **0/10** |
| falls, 5% missed control steps | 3/10 | **2/10** |
| fall -> self-right -> stand -> trot again (lockstep_recovery.py, side/back/tipped/shove) | 40/40 | **40/40** |
| joint speed over 1.2x rated (steps), Unitree damping 0.1 | 1257 of 20500 (worst 2.23x) | 1448 of 20500 (worst 2.17x) |
| the same at damping 0.5 | 132 (worst 1.80x) | **76** (worst 1.63x) |

Our own exam (imitate_env.exam, 4 tries per clip):

| | 592M | **695M** |
|---|---|---|
| mean tracking | 0.809 | **0.820** |
| falls | 9.6% (trot -> sit 4/4) | **0.0%** (every moving sit/lie reached, no fall) |
| walk / trot / pace | 0.811 / 0.758 / 0.831 | 0.817 / **0.790** / 0.836 |
| canter | 0.485 | **0.524** |
| sit / lie / getup | 0.939 / 0.922 / 0.922 | 0.937 / 0.926 / 0.917 |
| steer trot / canter | 0.84 / 0.41 | 0.825 / **0.54** |
| 20 / 40 ms delay | 0.83 / 0.81, no falls | 0.846 / 0.821, no falls |

Still open: knee (calf) shaking past rated speed at Unitree's default damping 0.1 -- unchanged by this piece (the
damping-range run bolt-go2-damp targets it; deploy/PLAN.md 1 Nov 09:40). Canter 0.524 is still the weakest skill.
Use: `train/deploy/run_go2.py` (default brain). Simulation only. Dog motion clips are CC BY-NC (non-commercial).
