# bolt-go2-gpu-241M (simulation): first Unitree Go2 dog-motion brain trained on a GPU (31 Oct)

Body: Unitree Go2 (MuJoCo Menagerie model, `train/robot_bodies.py`). Warm-started from the Kaggle CPU brain at 14.6M
steps, then 226M steps of PPO in MJX on a RunPod RTX 4090 (4.0 h, 4096 robots, ~16k steps/s, full float32: the 4090's
default TF32 breaks the contact solver, see `train/research/results/go2_dog/PIECES.md`).

Recipe: all 9 dog clips (walk, trot, pace, canter, turns, sit, lie, getup), clip switches 0.3/s, steering commands
+-0.8 rad/s on 60% of tries (weight 0.6), root-global 0.5, 0-40 ms control delay per try, sensor noise.

Exam (imitate_env.exam, all clips, 4 tries each, randomized physics; exam_steer trot/canter at +-0.4/0.8 rad/s):

| | CPU 14.6M | GPU 136M | **this (241M)** |
|---|---|---|---|
| mean tracking | 0.819 | 0.827 | **0.838** |
| walk / trot / pace | 0.817 / 0.819 / 0.833 | 0.805 / 0.821 / 0.827 | **0.820 / 0.823 / 0.839** |
| canter (tracking / pace kept) | 0.412 / 38% | 0.578 / 50% | **0.597 / 51%** |
| turns left / right | - | - | 0.825 / 0.828 |
| sit / lie / getup | 0.947 / 0.929 / - | 0.957 / 0.922 / 0.928 | **0.960 / 0.926 / 0.928** |
| steering trot / canter | - | 0.83 / 0.52 | **0.84** / 0.50 |
| delay 0 / 20 / 40 ms | - | 0.83 / - / 0.83 | 0.84 / **0.87** / 0.84, no falls |
| falls | 9.6% | 13.5% | 11.5% |

No falls in any gait or at any delay. Every fall is a posture command given on the move: trot -> sit 4 of 4,
trot -> lie 1 of 4, walk -> sit 1 of 4. Run c (moving sit/lie starts) trains on exactly this.

Use: `mjx_imitate.MjxImitatePolicy("params.pkl").act(obs)` with `BOLT_BODY=go2` and `imitate_env.ImitateEnv`
(numpy only, no JAX needed). exam.json is the full exam; progress.json the training curve.
Simulation only: not run on a real Go2. Dog motion clips are CC BY-NC (non-commercial).
