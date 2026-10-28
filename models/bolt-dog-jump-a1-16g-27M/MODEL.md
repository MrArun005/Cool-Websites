# Bolt hurdle-jump brain, run 16g best (27M steps, real Unitree A1 motors)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: runs 16f (job added with fd5199b itself, 27 Oct 11:17) and 16g (13:14) have no BOLT_MOTOR_TORQUE override. Lineage: 16f started from run 16e's 12M start brain, trained at 17 N·m.

Jumps the real agility hurdle (solid stands, a loose bar resting in cups) from a run-up, with the real A1 motor
model (33.5 N·m, 420 W, 21 rad/s per joint). 51 inputs (real-hurdle brain), from run 16f's best (19M).

Exam (10 tries per height, half of them 6° askew and 5 cm off the line; clean = over without touching the bar,
landed, standing, facing within 20°): 25 cm 10/10, 30 cm 9/10, 40 cm 7/10, 50 cm 1/10 (best.json).
After landing it gallops off at 3-3.9 m/s: that run-out is the reference gait for run 22 (dogmotion/clips/bolt_gallop.npz).
