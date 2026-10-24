# Gait Lab

A 3D replay page for Bolt's trained brains (stand, sit, lie, walk). The page draws a robot-dog body
on top of the simulator's own part positions, and marks paws, knees and shins that touch the floor.

Make the clips (run from `train/`), then serve this folder (e.g. `python -m http.server`):

    P="--brain posture=../models/bolt-dog-run2-1M --brain walk2M=<run>/checkpoints/<2M> --brain walk500k=../models/bolt-dog-walk-500k --gait-generator"
    python record3d.py $P --label "Full routine" -o gaitlab/data/routine.json \
        --script stand:2:posture sit:4:posture stand:3:posture lie:4:posture stand:3:posture walk:8:walk500k
    python record3d.py $P --label "Stand" --script stand:6:posture -o gaitlab/data/stand.json
    python record3d.py $P --label "Sit" --script stand:1.5:posture sit:5:posture -o gaitlab/data/sit.json
    python record3d.py $P --label "Lie down" --script stand:1.5:posture lie:5:posture -o gaitlab/data/lie.json
    python record3d.py $P --label "Walk, 2M brain" --script walk:8:walk2M -o gaitlab/data/walk.json
    python record3d.py $P --label "Walk, 500k brain" --script walk:8:walk500k -o gaitlab/data/walk-500k.json
    python record3d.py $P --label "Stand, then walk" --script stand:3:posture walk:6:walk2M -o gaitlab/data/handover.json

The tail, ears and blinking are decoration; the simulated dog has none of them.
