"""DEEP Robotics' own Lite3 simulator (Lite3_rl_deploy's mujoco_simulation.py), run headless for our rehearsal: the
robot side of deploy/lite3_link.py's UDP protocol, as sim_vendor.py is for the Go2.

    LITE3_RL_DEPLOY=<Lite3_rl_deploy checkout> LITE3_MJCF=<URDF_model>/Lite3/mjcf/Lite3.xml \\
        python deploy/sim_lite3.py --seconds 60 &
    python deploy/run_go2.py --body lite3 --plan "trot:6,sit:5,lie:6,getup:6,walk:6"

Their code unchanged (imported, not copied: BSD-3-Clause, DeepRoboticsLab): only its settings are set before it
starts. No viewer; the flat-floor Lite3 MJCF from DEEP's URDF_model repo (their default scene includes a staircase
and a model submodule); the robot starts lying folded on the floor, in their own initial joint pose, as a real Lite3
does. Their loop steps 1 ms of physics whenever 1 ms of wall time has passed: on a slow machine it runs slower than
real time, and the end prints how much.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60.0)
a = ap.parse_args()

sys.path.insert(0, str(Path(os.environ["LITE3_RL_DEPLOY"]) / "interface" / "robot" / "simulation"))
import mujoco_simulation as ms  # noqa: E402

ms.USE_VIEWER = False
sim = ms.MuJoCoSimulation(xml_relpath=str(Path(os.environ["LITE3_MJCF"]).resolve()))
sim.data.qpos[2] = 0.12            # lying on the floor (their default drops it from 1 m)
ms.mujoco.mj_forward(sim.model, sim.data)
sim.print_debug_info = lambda: None   # their 0.5 Hz state printout
t0 = time.time()
threading.Thread(target=sim.start, daemon=True).start()
time.sleep(a.seconds)
print(f"sim_lite3: {a.seconds:g} s wall, {sim.timestamp:.1f} s simulated (real-time factor {sim.timestamp / (time.time() - t0):.2f}); "
      f"body height {sim.data.qpos[2]:.3f} m, up {sim.data.xmat[1][8]:.3f}", flush=True)
