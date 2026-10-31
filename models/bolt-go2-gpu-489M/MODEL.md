# bolt-go2-gpu-489M (simulation): the best Go2 dog-motion brain (31 Oct); replaces go2-gpu-399M

Run c to the end: from models/bolt-go2-gpu-241M, 249M more steps on a RunPod RTX 4090 (4.4 h, 489M steps in all),
the 241M recipe (all 9 clips, switches 0.3/s, steering +-0.8 rad/s, root-global 0.5, 0-40 ms delay) plus moving
starts (30% of tries begin mid walk or trot and are told to sit or lie). Full float32 (TF32 breaks MJX contacts).

Exam (imitate_env.exam, all clips, 4 tries each, randomized physics; exam_steer trot/canter +-0.4/0.8 rad/s):

| | CPU 14.6M | GPU 241M | GPU 399M | **this (489M)** |
|---|---|---|---|---|
| mean tracking | 0.819 | 0.838 | 0.845 | **0.848** |
| walk / trot / pace | 0.817 / 0.819 / 0.833 | 0.820 / 0.823 / 0.839 | 0.829 / **0.834** / 0.855 | **0.836** / 0.827 / **0.864** |
| canter | 0.412 | 0.597 | 0.617 | **0.628** |
| sit / lie | 0.947 / 0.929 | **0.960** / 0.926 | 0.953 / 0.931 | 0.951 / **0.935** |
| steering trot / canter | - | 0.84 / 0.50 | 0.79 / 0.57 | **0.84 / 0.58** |
| delay 20 / 40 ms | - | 0.87 / 0.84 | 0.88 / 0.85 | **0.88 / 0.85**, no falls |
| falls | 9.6% | 11.5% | **1.9%** | **1.9%** |

The one fall left: 1 of 4 sits commanded from a trot. Sit and lie from a walk, and lie from a trot, all reached.
Use: mjx_imitate.MjxImitatePolicy("params.pkl") with BOLT_BODY=go2 and imitate_env.ImitateEnv (numpy only).
Simulation only: not run on a real Go2. Dog motion clips are CC BY-NC (non-commercial).
