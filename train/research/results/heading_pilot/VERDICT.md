# Heading pilot (--root-global 1.0, Peng 2020's global root terms in the reward): passed, narrowly (simulation)

runs/pilot_rg: from 18g 35.5M at 36M -> 39.5M steps, train_imitate.py --root-global 1.0 --switch-rate 0.3 --lr 1e-4
(default clips, no canter), local CPU. Scored with research/peng_score.py (Peng et al. 2020 Eq. 4-9, 10 s, seeds 1 2,
5 episodes each, canonical physics); files research/results/peng_score_pilot_rg_*.json.

| | 18g 35.5M (start) | 36M | 38M | 39M | **39.5M** | Peng 2020 sim |
|---|---|---|---|---|---|---|
| trot | 0.700 | 0.716 | 0.712 | 0.723 | **0.731** | 0.727-0.752 |
| pace | 0.668 | 0.704 | 0.727 | 0.733 | **0.759** | 0.820-0.839 |
| trot drift off the dog's path in 10 s | 17.5 m | 6.4 m | 5.0 m | 4.5 m | **3.7 m** | |
| pace drift | 10.8 m | 10.4 m | 8.2 m | 7.5 m | **4.2 m** | |
| trot / pace pose term | 0.943 / 0.892 | 0.926 / 0.929 | 0.912 / 0.944 | 0.913 / 0.940 | 0.915 / 0.942 | |

Pass line was trot +0.03 (>= 0.730) with the pose held: trot 0.731 just meets it (sd 0.004, so narrowly), pace
+0.091. Trot's pose term fell 0.943 -> 0.915 while pace's rose 0.892 -> 0.942. Verdict: the global root terms fix the
drift (17.5 m -> 3.7 m) and lift Peng's score, so the next big GPU dog-motion run uses --root-global 1.0 with all 9
clips. This pilot brain itself is not handed over (no canter, short run).
