# MasterMind v8g (29 Oct): first one-network brain at the 95% switching target (simulation)

From mastermind/export_v8, 4 local DAgger rounds with teachers v8g: gpu3 (the delay-trained dog-motion brain) for
every clip, the jump brain trained on real hand-overs (lands Bolt itself), 17e recovery; teacher chain 95.8% (dev).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/dagger.py --teachers v8g --steps0 400000 --steps 300000 --rounds 5 \
        --workers 4 --epochs0 10 --epochs 5 --hidden 1024,512,256 --beta-floor 0.2 --seed 11 \
        --init-student mastermind/export_v8/student.pt --out runs/pilot_mm8g        # student_r4.pt = this brain
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v8g/student.pt \
        --teachers v8g --n 30 --seed 1000 --with-50 --exam [--latency-ms 20]

Picked on dev seeds 20000 (r2 95.8%, r3 97.1%, r4 98.2%); verdict on fresh test seeds 1000, 30 tries per switch:

| | v8 | v8g |
|---|---|---|
| all switches | 92.4% | **95.3%** |
| without the 50 cm rows | 94.0% | 96.6% |
| at 20 ms control delay | 68.3% (322 switches reached) | 70.4% (540 reached) |
| tracking walk / trot / canter | 0.908 / 0.888 / 0.698 | 0.839 / 0.784 / 0.548 |

Use v8g where switching matters most; v8 when the motion should look most like the dog's. Not delay-proof: at 20 ms
the jump and recovery brains (its teachers) fail; they need training with delay next.
