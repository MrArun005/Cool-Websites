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

## Piece 2 (Kaggle bolt-go2-dog-p2, 3.3M -> 8.9M): still climbing

| steps | mean tracking | walk | trot | pace | canter | sit | lie | fall share | trot / canter speed |
|---|---|---|---|---|---|---|---|---|---|
| 4.0M | 0.777 | 0.782 | 0.767 | 0.804 | 0.342 | 0.919 | 0.891 | 5.8% | 73% / 29% |
| 6.0M | 0.790 | 0.789 | 0.773 | 0.810 | 0.364 | 0.944 | 0.910 | 7.7% | 75% / 29% |
| 8.0M | 0.803 | 0.801 | 0.776 | 0.820 | 0.410 | 0.947 | 0.918 | 5.8% | 74% / 37% |
| 8.9M | 0.808 | 0.800 | 0.795 | 0.815 | 0.406 | 0.958 | 0.923 | 9.6% | 77% / 32% |

+0.04 mean in 3.5 h; falls still in the canter and moving sit/lie starts. Piece 3 running (bolt-go2-dog-p3).

## Piece 3 (Kaggle bolt-go2-dog-p3, 8.9M -> 14.6M): slowing

10M 0.803, 11M 0.807, 12M 0.804, 13M 0.815, 14M 0.809, **14.6M 0.819** (walk 0.817, trot 0.819, pace 0.833, canter 0.412,
sit 0.947, lie 0.929; trot speed 80%, canter 38%; falls 9.6%, canter and moving sit/lie). +0.011 in 3.5 h (piece 2:
+0.042). The canter is the bottleneck (flat at ~0.40; the Go2 knee drive is rated 15.7 rad/s). Piece 4 running; if flat,
the next Go2 step is the GPU trainer (mjx) port for far more steps, not more CPU pieces.

## Piece 4 (Kaggle bolt-go2-dog-p4, 14.6M -> 21.4M): flat, CPU line ended

15M 0.817, 16M 0.816, 17M 0.809, 18M 0.816, 19M 0.816, 20M 0.820, 21.4M 0.816 (canter 0.411, falls 8%). +0.00 in 3.5 h:
no more CPU pieces. Go2 continues on the GPU trainer only.

## GPU (RunPod RTX 4090, mjx_imitate, 31 Oct): run b, warm from CPU 14.6M

4096 envs, all 9 clips, switches 0.3/s, steering +-0.8 rad/s (weight 0.6), root-global 0.5, 0-40 ms control delay,
4.0 h at ~16k steps/s.

Bug found first: the 4090's default TF32 matmuls break MJX's contact solver. The same warm brain fell in 59% of eval
tries on TF32 (nan scores before the blow-up guard) against 7% at full float32 (local CPU check: 6%). Fixed in
mjx_imitate (29338fa: jax_default_matmul_precision "highest"; 9804028: a blown-up env ends its try at no reward).
Kaggle P100/T4 have no TF32, which is why earlier GPU runs were fine. ~55 min of pod time went to the broken runs.

Training eval (128 envs): reward/step 1.044 (0) -> 1.274 (45M) -> 1.318 (136M) -> 1.334 (181M); falls 7% -> 0-2%;
steer 0.51 -> 0.94.

Real exam (imitate_env.exam, all clips, 4 tries, + exam_steer trot/canter +-0.4/0.8; scratchpad go2_exam.py):

| brain | mean tracking | walk / trot / pace | canter | sit / lie / getup | steer (trot / canter) | 40 ms delay | falls |
|---|---|---|---|---|---|---|---|
| CPU 14.6M (p3) | 0.819 | 0.817 / 0.819 / 0.833 | 0.412 | 0.947 / 0.929 / - | - | - | 9.6% |
| GPU 45M | 0.816 | 0.794 / 0.808 / 0.816 | 0.554 | 0.947 / 0.908 / 0.924 | 0.78 / 0.46 | 0.81, 0 falls | 13.5% |
| GPU 136M | **0.827** | 0.805 / **0.821** / 0.827 | **0.578** | **0.957** / 0.922 / 0.928 | **0.83 / 0.52** | **0.83, 0 falls** | 13.5% |

No falls in any gait; every fall is a posture command given on the move (trot -> sit 4/4, trot -> lie 2/4,
walk -> sit 1/4). Run c (from run b's final brain) adds moving starts (--moving-rate 0.3, walk/trot -> sit/lie) for
exactly these.

Final run b brain (241M total, models/bolt-go2-gpu-241M): mean 0.838, walk/trot/pace 0.820/0.823/0.839, canter 0.597
(51% pace), sit/lie/getup 0.960/0.926/0.928, steer trot 0.84 / canter 0.50, 0/20/40 ms 0.84/0.87/0.84 with no falls,
falls 11.5% (moving trot->sit 4/4, trot->lie 1/4, walk->sit 1/4). Best Go2 brain so far; run c started from it.

Run c (moving starts walk/trot -> sit/lie, from 241M), checkpoint at 158M more (models/bolt-go2-gpu-399M): mean 0.845,
walk/trot/pace 0.829/0.834/0.855, canter 0.617, falls 11.5% -> **1.9%** (moving sit/lie: from a walk 4/4 + 4/4, from a
trot sit 3/4 (1 fell) and lie 4/4), steer trot 0.79 / canter 0.57, 20/40 ms 0.88/0.85 no falls.
