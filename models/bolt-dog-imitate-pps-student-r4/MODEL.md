# Bolt dog-motion student r4 (PPS planner distilled into the 18b-17.5M brain)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: saved 27 Oct 19:16 UTC (commit 705be3c), after fd5199b; distilled from 18b-17.5M (see that card's note) by train/research/pps_distill.py, no BOLT_MOTOR_TORQUE override.

Fine-tuned actor of bolt-dog-imitate-18b-17.5M (same 74 inputs; critic and vecnormalize.pkl untouched) on what the
PPS planner made it do (train/research/pps_distill.py, DAgger rounds r0-r4). Results, commands and seeds:
train/research/REPORT.md. Load like any 18b brain (`imitate_env`, policy.zip + vecnormalize.pkl).
