# Bolt dog motion brain trained on a Kaggle GPU (MJX + Brax PPO), 360M steps — best all-round dog brain

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: mjx_imitate.py's MOTOR is torque=33.5 (file added 27 Oct 17:31, after fd5199b); warm-started from run 18b (see the 18b cards' note); CPU exams after fd5199b.

Warm-started from run 18b at 16.5M (widened to 9 slots), then 344M steps on 2x T4 (4096 parallel Bolts, ~16.5k steps/s, 6.3 h).
All 9 clips; real A1 motor limits. Load with `mjx_imitate.MjxImitatePolicy("params.pkl")` (numpy, no GPU needed).

Exams on the CPU simulator (randomized physics):
- Fair canter exam (research/exam_canter.py protocol, 30 tries each): seeds 5000 76.0%, seeds 7000 76.6%, seeds 9000 76.6%; 0/90 falls; 78% of the dog's speed.
- 9-clip exam (imitate_env.exam, 8 tries per clip, seed 9000): walk 93%, trot 92%, pace 94%, turn left 91%, turn right 91%, sit 97%, lie 94%, get up 93%, all 0 falls.
- Caveat: in a 2-clip exam env (getup + canter, 8 tries) the canter scored 67.8% with 1 fall in 8, so its canter is not fall-proof in every start condition.
Compare: 18f-31M canter 71.8/72.2/72.4% (0/90 falls); 18d-24.5M walk 91, trot 90, sit 95, lie 94, get up 92.
