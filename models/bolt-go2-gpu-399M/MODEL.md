# bolt-go2-gpu-399M (simulation): Go2 dog motion that sits and lies down on the move (31 Oct); replaces go2-gpu-241M

From models/bolt-go2-gpu-241M, 158M more steps on a RunPod RTX 4090 (run c, 398969648 steps in all): the same recipe plus
moving starts (30% of tries begin mid walk or trot and are told to sit or lie: --moving-rate 0.3 --moving-gaits walk
trot --moving-clips sit lie). Full float32, 4096 robots, ~16k steps/s.

Exam (imitate_env.exam, all clips, 4 tries each, randomized physics; exam_steer trot/canter +-0.4/0.8 rad/s):

| | CPU 14.6M | GPU 241M | **this (399M)** |
|---|---|---|---|
| mean tracking | 0.819 | 0.838 | **0.845** |
| walk / trot / pace | 0.817 / 0.819 / 0.833 | 0.820 / 0.823 / 0.839 | **0.829 / 0.834 / 0.855** |
| canter | 0.412 | 0.597 | **0.617** |
| sit / lie / getup | 0.947 / 0.929 / - | 0.960 / 0.926 / 0.928 | 0.953 / **0.931** / 0.927 |
| sit / lie reached from a walk | - | 3/4 sit (1 fell), 4/4 lie | **4/4, 4/4** |
| sit / lie reached from a trot | - | 0/4 sit (4 fell), 3/4 lie (1 fell) | **3/4 sit (1 fell), 4/4 lie** |
| steering trot / canter | - | 0.84 / 0.50 | 0.79 / **0.57** |
| delay 20 / 40 ms | - | 0.87 / 0.84, no falls | **0.88 / 0.85**, no falls |
| **falls** | 9.6% | 11.5% | **1.9%** |

The only fall left: one of 4 sits commanded from a trot. Trot steering dipped 0.84 -> 0.79; canter steering rose.
Use: mjx_imitate.MjxImitatePolicy("params.pkl") with BOLT_BODY=go2 and imitate_env.ImitateEnv (numpy only).
Simulation only: not run on a real Go2. Dog motion clips are CC BY-NC (non-commercial).
