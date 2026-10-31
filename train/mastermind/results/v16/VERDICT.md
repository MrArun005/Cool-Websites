# MasterMind v16f (31 Oct): tie with v12 on the final test -- v12 stays the main brain (simulation)

Kaggle bolt-mastermind-v16f: teachers v11n, from v14 r13, DAgger focus flags --canter-trot-boost 3 --jump-weight 33,
17 rounds (11 h). Pooled count: every switch row, "[pose reached]" rows left out.

Practice (dev seeds 20000, 15 tries per switch, with 50 cm, exam): v12 r14 97.8% (v12 range 96.8-97.8) vs
v16f r12 **98.4%**, r14 98.1%, r16 97.4%. r12 earned the final test.

Final test (seeds 1000, 30 tries per switch, with 50 cm, exam):

| switch | v12 r14 | v16f r12 |
|---|---|---|
| canter -> trot | **28/30** | 23/30 |
| trot -> walk | 27/27 | 23/23 |
| jump 20 / 30 / 40 / 50 cm | 27 / 26 / 27 / 27 of 30 | 27 / **27** / 27 / 27 of 30 |
| fall -> recover (stood up) | 26/30 | **30/30** |
| recover -> walk | 26/26 | **30/30** |
| all switches | **96.9% (601/620)** | **96.9% (599/618)** |

Same picture as v14: recovery goes to 30/30 but canter -> trot drops (here 28 -> 23 of 30, despite the canter-trot
boost). A tie keeps v12. student_r12.pt is kept here as a start brain (best recovery). logs/ holds every test log.
