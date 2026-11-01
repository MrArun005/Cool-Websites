# Go2 recovery: getting up after a fall (simulation)

## Pilot 1 (Kaggle bolt-go2-getup-pilot, 31 Oct - 1 Nov): stand goal, flat -- never up from the side or back

train_getup.py --body go2 --goal stand --hard from scratch, 0-40 ms delay, 10.6 h, 23M steps. Exam every 1M steps:

| from | 1M | 10M | 23M |
|---|---|---|---|
| side | 0.0 | 0.0 | 0.0 |
| back | 0.0 | 0.0 | 0.0 |
| tipped | 0.5 | 0.5 | 0.4 |
| standing (stay up) | 1.0 | 1.0 | 1.0 |

Flat for 22M steps. Likely cause: the stand goal moves the joints only ACTION_SCALE (0.4 / 1.0 / 1.2 rad) around
the stance, too little to swing the legs far enough to roll the body over (Go2 shoves land it on the back ~70% of
the time). Bolt got up from falls with the selfright goal (full motor range) and then its posture brain.

Next: pilot 2 (bolt-go2-selfright-pilot, started ~04:45 IST 1 Nov, 3.5 h): selfright goal, then the motion brain's
getup clip (489M: getup 0.93) to stand. Gate: righted from side or back > 0.

## Pilot 2 (Kaggle bolt-go2-selfright-pilot, 1 Nov): selfright goal -- PASSED

Same falls, the selfright goal (full motor range). Exam righted from side / back / tipped / standing: 1.0 / 1.0 / 1.0
/ 1.0 already at 1M steps and at 6M (0.86 / 0.96 / 0.70 s). Chained with a motion brain's getup clip it stands back up
30/30 (side, back, shoved, dropped, ice; 6 each) with both bolt-go2-blind-592M and bolt-go2-gpu-489M. Saved as
models/bolt-go2-selfright-6.9M. The Go2 recovery step of the plan is done in simulation.
