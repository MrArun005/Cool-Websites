# bolt-lite3-dog-112M (the Lite3 robot brain 4 Nov 23:20 - 5 Nov 03:00 UTC; now 75% of the blend bolt-lite3-dog-blend-112M-w7)

DEEP Robotics Lite3 motion brain, simulation only. Cloud worker 5: fine-tuned from bolt-lite3-dog-87.3M with moving
sit/lie starts 0.75 and 0-60 ms of command delay; its final checkpoint, 111,981,608 steps (policy.zip +
vecnormalize.pkl from SB3; params.pkl, the numpy brain the robot program runs, from `mjx_imitate.py warm`).

On DEEP's own Lite3 MuJoCo model (1 ms physics, robot-only inputs), against 87.3M:

| Test | 87.3M | 112M |
|---|---|---|
| 41 s routine without the canter, 3-6 ms delay | 0/12 falls | 0/24 |
| trot -> sit, 5-15 ms delay | 4/48 | **0/48** |
| routine ending in a canter, 3-6 ms | 2/12 (canter) | 5/12 (canter) |
| recovery (side, back, tipped, shove) | 20/20 | 20/20 |
| moving sit / lie (our exam) | 0/48 | 0/48 |
| our exam: tracking / steering | 0.780 / 0.449 | 0.756 / 0.456 |
| live on DEEP's simulator, no canter, 4 runs | no fall; 1 speed stop | no fall, no stop |

Adopted because the robot program runs the Lite3's canter as a trot (run_go2 --no-canter, the default), and with the
canter off it removes the last falls under delay. For runs with --allow-canter, 87.3M cantered better. It drives the
knees over 1.2x rated speed in more control steps (10-17 per 38 s live run against 5-11), under the speed stop's limit
of 3 in a row.
