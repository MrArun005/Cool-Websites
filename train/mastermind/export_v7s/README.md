# MasterMind v7s (29 Oct): v7 with the canter taught by 18g again: the better-looking canter, a bit fewer switches

From mastermind/export_v7 (v7), 2 local DAgger rounds with teachers v7s (v7 with canter teacher 18g instead of gpu3).

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/dagger.py --teachers v7s --steps0 400000 --steps 250000 --rounds 3 \
        --workers 4 --epochs0 10 --epochs 5 --hidden 1024,512,256 --beta-floor 0.2 --seed 8 \
        --init-student mastermind/export_v7/student.pt --out runs/pilot_mm7s        # student_r2.pt = this brain
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/export_v7s/student.pt \
        --teachers v7s --n 30 --seed 1000 --with-50 --exam --out mastermind/results/v7/test_pilot7s_r2.json

Fresh test seeds 1000, 30 tries per switch (simulation): all switches 89.6% (v7 91.3%, v4 83.3%), without the 50 cm
rows 91.9% (v7 92.1%), trot -> 50 cm 12/30 (v7 21/30); canter tracking 0.680 (v7 0.548, v4 0.695).
v7 stays the main MasterMind (switching is the target furthest off); this one is for when the canter's look matters.
