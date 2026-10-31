# MasterMind v15d (31 Oct): delay-brain retrain, not better than v11d -- v11d stays (simulation)

Kaggle bolt-mastermind-v15d: teachers v10d, 17 DAgger rounds (10.7 h). Scored with the same pooled count as v11d
(every switch row, "[pose reached]" rows left out).

Practice (dev seeds 20000, 15 tries per switch, with 50 cm, exam):

| brain | 20 ms delay | 0 ms delay |
|---|---|---|
| v11d r13 (current delay brain) | 86.9% (272/313) | 89.3% (276/309) |
| v15d r10 | 88.1% | |
| **v15d r12** | **89.1% (278/312)** | 89.3% (274/307) |
| v15d r14 | 87.3% | |
| v15d r16 (last) | 86.2% | |

r12 earned the final test (seeds 1000, 30 tries per switch, with 50 cm, exam):

| brain | 20 ms delay | 0 ms delay |
|---|---|---|
| v11d r13 | **86.8%** | 90.7% |
| v15d r12 | 84.8% (524/618) | 90.6% (561/619) |

The +2.2 points on practice did not hold on fresh seeds (-2.0 at 20 ms, tie at 0 ms): within the noise of 15 tries.
v11d stays the delay brain. logs/ holds every test log.
