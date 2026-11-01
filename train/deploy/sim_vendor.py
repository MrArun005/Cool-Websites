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
    ap.add_argument("--shove-at", type=float, nargs="*", default=[], help="sim times (s) to push the dog over sideways")
    ap.add_argument("--remote-script", default="", help='a scripted wireless remote: "t:keys;..." e.g. '
                    '"12:ly=0.5;16:ly=0.8,rx=0.4;20:;24:B;30:L2+B" (sticks hold until changed, buttons press 0.3 s)')
    ap.add_argument("--shove", type=float, default=3.0, help="the push: sideways m/s, plus a roll kick of 8 rad/s")
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
    bridge = UnitreeSdk2Bridge(m, d)
    script = []
    for item in filter(None, a.remote_script.split(";")):
        t, keys = item.split(":")
        sticks, buttons = {}, []
        for k in filter(None, keys.split(",")):
            if "=" in k:
                sticks[k.split("=")[0]] = float(k.split("=")[1])
            else:
                buttons += k.split("+")
        script.append((float(t), sticks, buttons))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from remote import encode
    held, press_until, cur_buttons = {}, -1.0, []
    print(f"vendor sim up: Go2, dt {m.opt.timestep}, joint damping {m.dof_damping[6]:g}, domain {a.domain} on {a.iface}", flush=True)
    shoves = sorted(a.shove_at)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        s = time.perf_counter()
        if script and d.time >= script[0][0]:   # the scripted remote, into the state the bridge publishes
            _, sticks, cur_buttons = script.pop(0)
            held = {k: 0.0 for k in ("lx", "rx", "ry", "ly")} if not sticks and not cur_buttons else {**held, **sticks}
            press_until = d.time + 0.3
            print(f"remote at {d.time:.1f} s: {sticks or ''} {'+'.join(cur_buttons)}", flush=True)
        if a.remote_script:
            raw = encode(cur_buttons if d.time < press_until else [], **held)
            bridge.low_state.wireless_remote[:] = list(raw)
        if shoves and d.time >= shoves[0]:   # a push: sideways speed and a roll kick, in the body frame
            shoves.pop(0)
            side = 1 if len(shoves) % 2 == 0 else -1
            d.qvel[0:3] += d.xmat[1].reshape(3, 3) @ [0, side * a.shove, 0]
            d.qvel[3:6] += [side * -8.0, 0, 0]
            print(f"shove at {d.time:.1f} s", flush=True)
        mujoco.mj_step(m, d)
        left = m.opt.timestep - (time.perf_counter() - s)
        if left > 0:
            time.sleep(left)
    wall = time.perf_counter() - t0
    print(f"vendor sim done: {d.time:.1f} s simulated in {wall:.1f} s wall ({d.time / wall:.2f}x real time)", flush=True)


if __name__ == "__main__":
    main()
