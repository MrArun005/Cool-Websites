# Bolt dog motion brain, Kaggle GPU run gpu3 at 617M steps: the real-robot candidate (delay-proof), 29 Oct

> **Motor torque:** 33.5 N·m, 420 W, 21 rad/s; trained with the motor brake on (25 rad/s, 3.5 N·m·s/rad).
> All numbers are in simulation.

Warm-started from `bolt-dog-imitate-gpu-360M`, 617M more steps on 2x T4 (10.8 h). New in training: control delay
0-40 ms, sensor noise, landing starts, sit/lie commands while moving, the canter->trot transition term, pushes.
Load with `mjx_imitate.MjxImitatePolicy("params.pkl")`.

## Fair exam against 360M (same code, same seeds)

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/gpu_brain_exam.py ../models/bolt-dog-imitate-gpu3-617M/params.pkl \
        --motor-brake --out exam.json          # and ../models/bolt-dog-imitate-gpu-360M/params.pkl (brake off, as trained)

10 tries per clip on seeds 9000 and 11000; terrain 30 tries, 4 s each.

| | 360M | this brain |
|---|---|---|
| Tracking walk / trot / pace / turns (0 ms delay) | 91-94% | 78-85% |
| Pace kept, walk / trot | 90% / 90% | 79% / 78% |
| Sit / lie tracking | 97% / 94% | 96% / 92% |
| Sit from a walk / from a trot (reached, no fall) | 85% / 15% | 100% / 100% |
| Lie from a walk / from a trot | 5% / 0% | 100% / 100% |
| Falls with 10 / 20 / 40 ms control delay | 0 / 41 / 92% | **0 / 0 / 2%** |
| Tracking with 20 ms delay | 34% | **89%** |
| Trot, 5 cm rough ground + 1 m/s shove: falls | 33% | 20% |
| Trot on 10 cm rough ground: falls | 57% | 67% |
| Exam score (tracking x pace) | 0.716 | 0.602 |

Verdict: not a replacement for 360M as the "copies the dog" brain (tracking fell from 91-94% to 78-85%, below the
90% target), but by far the better brain for a real robot: with 20 ms of control delay (a real robot has 10-20 ms)
360M falls 41% of the time and this one never; it also sits and lies down from a walk or trot, which 360M could not.
Brake on/off changes nothing measurable (exams/exam_gpu3_617M.json vs _brake). The 566M checkpoint scores within
a point (exams/exam_gpu3_566M_brake.json). Not solved: 10 cm rough ground (Step 6 target: trot < 10% falls).
