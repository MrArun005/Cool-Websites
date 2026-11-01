# bolt-go2-selfright-6.9M (simulation): the Go2 rolls back onto its belly after any fall (1 Nov)

Kaggle bolt-go2-selfright-pilot: train_getup.py --body go2 --goal selfright --hard from scratch (full motor range;
belly down, legs tucked, still), tough falls (kicks, drops, ice, a second kick), 0-40 ms delay, 3.5 h, 6.9M steps.

Exam (getup_env.exam, 10 tries each): righted from the side / back / tipped / standing **1.0 / 1.0 / 1.0 / 1.0** at
6M (0.86 / 0.96 / 0.70 s); already 1.0 everywhere at 1M. (Pilot 1, the stand goal, got 0.0 from side and back.)

Full recovery chain (this brain until righted, then a Go2 motion brain's getup clip; 6 tries per fall kind; stood =
upright > 0.9 and body above 0.22 m at the end of the clip + 1 s):

| fall | with bolt-go2-blind-592M | with bolt-go2-gpu-489M |
|---|---|---|
| side | 6/6 | 6/6 |
| back | 6/6 | 6/6 |
| shoved | 6/6 | 6/6 |
| dropped | 6/6 | 6/6 |
| on ice | 6/6 | 6/6 |

An SB3 brain (policy.zip + vecnormalize.pkl, getup_env.GetUpPolicy with body go2). For the robot program it still needs
a numpy export and a hand-over rule in run_go2.py (fallen -> this brain -> getup clip). Self-righting thrashes the legs
through their full range: on hardware only in a cleared cage. Simulation only.
