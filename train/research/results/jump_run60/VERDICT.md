# Kaggle jump-run60 (28-29 Oct, 10.7 h CPU, 55.0M -> 76.3M steps): not better, jump-trot 55M stays

Run-up-only training (1.8-3.2 m/s from 1.5-4 m, bars 55-75 cm, brake on), from jump-trot 55M. Final fair exams on
Kaggle (fair_exam.py, 30 tries, seed 9000, brake on; files here):

| | jump-trot 55M (start) | run60 best (74M) |
|---|---|---|
| from 3 m, mid-stride, 2.0 / 2.5 / 3.0 m/s, 50 cm | 0 / 0 / 0% | 33 / 43 / 13% |
| from 3 m, mid-stride, 2.0 / 2.5 / 3.0 m/s, 60 cm | 0 / 0 / 0% | 10 / 13 / 0% |
| MasterMind hand-over (1.7 m/s, 1.1 m, mid-stride) 40 / 50 cm | 90 / 60% | 67 / 0% |

It began to use a long run-up but lost the trot hand-over, and 60 cm stayed at 10-13%. Why (train.log): the practice
bar never left its 55 cm start ("practice bars now 55 cm" at every checkpoint), so long run-ups AND high bars came
at once; the mid-stride exam only moved off 0 in the last 6M steps. Next: same run-ups mixed with the old short
ones (0.8-4 m, 1.2-3.2 m/s) and the bar starting at 40 cm (job jump-run60b).
