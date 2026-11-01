"""Read-only look at a Go2 (HARDWARE_DAY.md step 1): prints the IMU and the joints in OUR order, never sends a command.

    python deploy/read_go2.py --iface <eth> --domain 0 [--seconds 30]     # real robot
    python deploy/read_go2.py --iface lo --domain 1                       # Unitree's simulator (sim_vendor.py)

Each line (5 per second): gravity in the body frame (nose-up tilt makes x negative, as in our simulator), gyro, the
12 joint angles in our order FL, FR, RL, RR x (hip, thigh, calf) and how far each is from the stance, and the gap
since the previous state message. At the end: message rate and the largest gap.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import robot_bodies  # noqa: E402
from controller import quat_to_mat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    ChannelFactoryInitialize(a.domain, a.iface)
    body = robot_bodies.get("go2")
    order, stance = np.asarray(body.entry["sdk_order"]), np.asarray(body.stance)
    lock, last, stamps = threading.Lock(), [None], []

    def on_state(msg):
        with lock:
            last[0] = msg
            stamps.append(time.perf_counter())

    ChannelSubscriber("rt/lowstate", LowState_).Init(on_state, 10)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        time.sleep(0.2)
        with lock:
            s, ts = last[0], list(stamps[-50:])
        if s is None:
            print("no rt/lowstate yet", flush=True)
            continue
        q = np.zeros(12)
        for i in range(12):
            q[order[i]] = s.motor_state[i].q
        g = quat_to_mat(np.array(s.imu_state.quaternion, float)).T @ np.array([0, 0, -1.0])
        gap = (np.diff(ts).max() * 1000) if len(ts) > 1 else float("nan")
        print(f"grav {np.round(g, 2)} gyro {np.round(np.array(s.imu_state.gyroscope, float), 2)} "
              f"q {np.round(q, 2)} dq-from-stance max {np.abs(q - stance).max():.2f} rad, gap {gap:.1f} ms", flush=True)
    with lock:
        ts = np.array(stamps)
    if len(ts) > 1:
        print(f"{len(ts)} messages in {ts[-1] - ts[0]:.1f} s = {len(ts) / (ts[-1] - ts[0]):.0f} Hz; largest gap "
              f"{np.diff(ts).max() * 1000:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
