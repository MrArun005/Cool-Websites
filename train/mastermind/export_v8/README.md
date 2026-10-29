# MasterMind v8 (29 Oct): the main one-network brain (replaces v7)

From mastermind/export_v7s, 3 more local DAgger rounds with teachers v7s (gpu3 for sit/lie, 18g canter, the jump brain
lands Bolt, jump brain trained on real hand-overs; teacher chain 93.7% on dev seeds). All numbers in simulation.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/dagger.py --teachers v7s --steps0 400000 --steps 300000 --rounds 5 \
        --workers 4 --epochs0 10 --epochs 5 --hidden 1024,512,256 --beta-floor 0.2 --seed 9 \
        --init-student mastermind/export_v7s/student.pt --out runs/pilot_mm7s2       # student_r3.pt = this brain
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v8/student.pt \
        --teachers v7s --n 30 --seed 1000 --with-50 --exam --out mastermind/results/v7/test_pilot7s2_r3.json

Picked on dev seeds 20000 (r2 90.2%, r3 92.3%, r4 92.2%); verdict on fresh test seeds 1000, 30 tries per switch:

| | v4 | v7 | v8 |
|---|---|---|---|
| all switches | 83.3% | 91.3% | 92.4% |
| without the 50 cm rows | 87.4% | 92.1% | 94.0% |
| canter tracking | 0.695 | 0.548 | 0.698 |
| walk -> sit -> lie, pose reached | 25/30 | 22/29 | 27/29 |
| fall -> stood up | 27/30 | 25/30 | 28/30 |
| trot -> jump 20/30/40/50 cm | 26/26/20/9 | 20/20/19/21 | 21/24/21/17 |
| canter -> trot | 26/29 | 30/30 | 25/29 |

Not yet the 95% target. Weak spots: trot -> jump (21-24/30 at 20-40 cm), canter -> trot (25/29).
