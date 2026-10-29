# Kaggle jump-run60b (29 Oct, 10.7 h CPU, 55.0M -> 75.2M steps): failed, jump-handover 59M stays

Mixed run-ups (0.8-4 m, 1.2-3.2 m/s), practice bar starting at 40 cm, from jump-trot 55M, brake on (summary.json has
the full arguments). Fair exams on Kaggle (fair_exam.py, 30 tries, seed 9000, brake on; files here), simulation:

| | jump-trot 55M (start) | run60b "best" (56M) |
|---|---|---|
| from 3 m, still or mid-stride, 2.0 / 2.5 / 3.0 m/s, 50-70 cm | 0% everywhere | 0% everywhere |
| MasterMind hand-over (1.7 m/s, 1.1 m) still legs 40 / 50 cm | 100 / 100% | 100 / 100% |
| same, mid-stride 40 / 50 cm | 90 / 60% | 97 / 53% |

Every checkpoint's in-training exam from 3 m at 2.5 m/s was 0% at 50/60/70 cm (19 of 21 checkpoints exactly 0), the
practice bar never moved off 40 cm in 20M steps ("practice bars now 40 cm" at all 21 checkpoints), and the 1.7 m/s
hand-over exam at 50 cm fell from 100% to 0-40% for 61M-66M before partly coming back. "Best" (56M) was picked on a
0.07 score, i.e. noise. Run60 (the first try) did reach 33-43% at 50 cm from 3 m by 74M; run60b, which mixed in the
short run-ups, did not learn the long run-up at all.

Reading: the long, fast run-up is a different skill from the 1.1 m hand-over jump, and fine-tuning the hand-over brain
on a mix lets the easy short tries dominate while the long ones never succeed, so the bar never rises. Two runs (21 h
CPU) have not moved 60 cm. Next for the 50 cm target, cheaper than another blind 12 h run (rule 0e: pilot first):
a separate run-up brain trained from run60's 74M (the only one that clears 50 cm from 3 m), with the bar starting at
50 cm and a 30-60 min local pilot whose pass line is mid-stride 50 cm from 3 m at 2.5 m/s >= 50% (run60 74M: 43%).
