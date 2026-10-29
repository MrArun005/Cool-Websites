# Bolt jump brain that practised from REAL MasterMind hand-overs, local pilot at 59.2M steps (29 Oct)

> **Motor torque:** 33.5 N·m, 420 W, 21 rad/s, motor brake on in training. All numbers in simulation.

From `bolt-dog-jump-trot-55M`, 4.1M more steps (local, 4 envs, ~1.4 h): 60% of practice tries start from one of
400 real trot -> jump hand-over states recorded in MasterMind's world (research/record_handover.py, seeds 50000+),
practice bars 30-65 cm. Last checkpoint of the pilot (not picked among checkpoints).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python train_jump.py --name pilot_ho --init-from ../models/bolt-dog-jump-trot-55M \
        --steps 400000000 --real --lr 1e-4 --target-kl 0.015 --n-steps 2048 --max-steps 200 --approach 0.8 2.0 \
        --approach-speed 2.4 --approach-speed-min 1.2 --run-frac 0.85 --gait-start 0.7 --bar-start 0.30 --bar-max 0.65 \
        --handover-file research/results/handover_states.npz --handover-rate 0.6 --exam-speed 1.7 --exam-dist 1.1 \
        --exam-heights 0.30 0.40 0.50 --envs 4 --every 500000 --hours 1.4

## Held-out real hand-overs (100 states, seeds 90000+, brake on)

    MUJOCO_GL=osmesa python research/handover_exam.py ../models/bolt-dog-jump-handover-59M \
        --states research/results/handover_states_test.npz --motor-brake

| bar | jump-trot 55M | this brain |
|---|---|---|
| 20 cm | 81% | 84% |
| 30 cm | 77% | 90% |
| 40 cm | 61% | 87% |
| 50 cm | 31% | 73% |

In MasterMind's teacher chain (teachers v7p = v6h + this brain, dev seeds 20000, N=30): trot -> jump 20/30/40/50 cm
80/83/70/57% (v6h with 55M: 77/73/63/37%); all switches pooled 94.0% (v6h 91.9%).

Trade-off, fair_exam.py seed 9000, brake on, bar 1.5 m away: still legs 2.0 m/s 60 cm 70% (55M: 17%), but the
synthetic mid-stride start at 1.5 m/s 40/50 cm 60/13% (55M: 83/57%). Real hand-overs are what MasterMind and Bolt's
Day Out give it, so this brain is the MasterMind jump teacher; the Kaggle jump-handover run (29 Oct) may replace it.

Load with `jump_env.JumpPolicy(folder, "policy.zip", "vecnormalize.pkl")`.

## In Bolt's Day Out (browser physics, every task a jump, 4 runs x 240 s, seeds 1-4, the jump brain lands Bolt)

    cd train/gaitlab && JUMP=jump.json LAND=land node doglife_jumptest.mjs <mujoco.js> <data folder> <seed> 240

| jump brain | jumps | clean | bar down | falls |
|---|---|---|---|---|
| this brain | 29 | 29 | 0 | 6 (1 in a jump) |
| jump-trot 55M | 29 | 23 | 4 | 7 |

Bolt's Day Out uses this brain since 29 Oct. Per-run lines: exams/dayout_jumptest.txt.
