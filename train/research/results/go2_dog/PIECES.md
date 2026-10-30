# Go2 dog motion (30-31 Oct): first brain on the Unitree Go2 body, from scratch (simulation)

Kaggle bolt-go2-dog-pilot (2 h, 4 CPUs): all 9 clips, switches 0.3/s, 0-40 ms control delay per try, Menagerie Go2
physics with joint damping 2.0 (assumed). Checkpoint exam (imitate_env.exam, 4 tries per clip, randomized physics):

| steps | mean tracking | walk | trot | pace | canter | turns L/R | sit | lie | getup | fall share |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0M | 0.687 | 0.698 | 0.554 | 0.647 | 0.172 | 0.775 / 0.747 | 0.892 | 0.857 | 0.839 | 13.5% |
| 2.0M | 0.749 | 0.774 | 0.721 | 0.773 | 0.240 | 0.799 / 0.779 | 0.900 | 0.886 | 0.869 | 7.7% |
| 3.0M | 0.767 | 0.791 | 0.745 | 0.794 | 0.315 | 0.784 / 0.793 | 0.909 | 0.884 | 0.886 | 5.8% |
| 3.3M | 0.766 | 0.780 | 0.758 | 0.786 | 0.322 | 0.787 / 0.782 | 0.913 | 0.880 | 0.889 | 7.7% |

Falls at 3.0M: canter 1 in 4 and trot-to-sit/lie moving starts 1 in 4; everything else 0. Speeds kept: walk 49%, trot
63%, canter 27% of the dog's. Win line was tracking >= 0.70 with no falls: tracking passed in 3M steps, falls not yet.
Still climbing -> piece 2 (bolt-go2-dog-p2, 3.5 h) and an A/B with joint damping 0.5 (bolt-go2-damp-pilot).

## Joint damping A/B (Kaggle bolt-go2-damp-pilot, 2 h, damping 0.5 vs Menagerie's assumed 2.0)

| steps | damping 2.0: mean / falls / canter | damping 0.5: mean / falls / canter |
|---|---|---|
| 1.0M | 0.687 / 13.5% / 0.172 | 0.694 / 13.5% / 0.281 |
| 2.0M | 0.749 / 7.7% / 0.240 | 0.722 / 15.4% / 0.347 |
| 3.0M | 0.767 / 5.8% / 0.315 | 0.746 / 11.5% / 0.378 |
| end | 0.766 / 7.7% / 0.322 (3.3M) | 0.763 / 17.3% / 0.377 (3.2M) |

Lower damping: a slightly better canter, about twice the falls, the same mean. The choice changes little at this stage;
training stays on 2.0 (Menagerie's file) until the real Go2's damping is measured on hardware.
