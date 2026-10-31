"""Unitree's own Go2 simulator (unitree_mujoco's Python simulator), headless: the real robot's control interface
(unitree_sdk2 DDS topics rt/lowstate, rt/lowcmd, rt/sportmodestate) on a MuJoCo Go2, for the PLAN.md phase 2 dress
rehearsal. Same physics loop as unitree_mujoco.py without the viewer window and the gamepad.

    UNITREE_MUJOCO=/path/to/unitree_mujoco python deploy/sim_vendor.py [--seconds 60] [--domain 1] [--iface lo]

Needs the deploy environment (cyclonedds + unitree_sdk2_python, see PLAN.md); unitree_mujoco is not vendored here,
point UNITREE_MUJOCO at a checkout of github.com/unitreerobotics/unitree_mujoco.
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

import mujoco


def flatten(m):
    """Unitree's go2 scene has two 8 cm bumps and a staircase 1.2-3.4 m ahead; our brains are trained on flat floor:
    switch every world geom but the floor plane off (no contact, invisible)."""
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rgba[g][3] = 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--damping", type=float, default=None, help="override the model's joint damping (ours trains with 2.0)")
    ap.add_argument("--dt", type=float, default=0.005, help="physics step (the vendor's 0.005; ours trains at 0.002)")
    ap.add_argument("--stairs", action="store_true", help="keep the scene's bumps and staircase (default: flat floor)")
    a = ap.parse_args()
    root = Path(os.environ["UNITREE_MUJOCO"]) / "simulate_python"
    # the vendor bridge reads a module named config: give it ours (no gamepad, Go2, the vendor's 5 ms step)
    cfg = types.ModuleType("config")
    cfg.ROBOT, cfg.DOMAIN_ID, cfg.INTERFACE = "go2", a.domain, a.iface
    cfg.ROBOT_SCENE = str(root.parent / "unitree_robots" / "go2" / "scene.xml")
    cfg.USE_JOYSTICK, cfg.PRINT_SCENE_INFORMATION, cfg.ENABLE_ELASTIC_BAND = 0, False, False
    cfg.SIMULATE_DT, cfg.VIEWER_DT = a.dt, 0.02
    sys.modules["config"] = cfg
    sys.path.insert(0, str(root))
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge

    m = mujoco.MjModel.from_xml_path(cfg.ROBOT_SCENE)
    m.opt.timestep = cfg.SIMULATE_DT
    if not a.stairs:
        flatten(m)
    if a.damping is not None:
        m.dof_damping[6:] = a.damping
    d = mujoco.MjData(m)
    ChannelFactoryInitialize(cfg.DOMAIN_ID, cfg.INTERFACE)
    UnitreeSdk2Bridge(m, d)
    print(f"vendor sim up: Go2, dt {m.opt.timestep}, joint damping {m.dof_damping[6]:g}, domain {a.domain} on {a.iface}", flush=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        s = time.perf_counter()
        mujoco.mj_step(m, d)
        left = m.opt.timestep - (time.perf_counter() - s)
        if left > 0:
            time.sleep(left)
    wall = time.perf_counter() - t0
    print(f"vendor sim done: {d.time:.1f} s simulated in {wall:.1f} s wall ({d.time / wall:.2f}x real time)", flush=True)


if __name__ == "__main__":
    main()
