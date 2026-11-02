"""The Lite3 side of run_go2.py (--body lite3): DEEP Robotics' UDP joint protocol, as their Lite3_rl_deploy speaks it.

DEEP's own simulator (Lite3_rl_deploy/interface/robot/simulation/mujoco_simulation.py; our sim_lite3.py runs it
headless) and our bridge to the real robot (deploy/lite3_bridge/: their Lite3_MotionSDK Receiver/Sender) both:
  * send the robot state to us, UDP port 30010: float64 time, then float32 roll, pitch, yaw (rad), body acceleration
    (3, m/s^2), body angular velocity (3, rad/s), joint angles (12), joint speeds (12), joint torques (12);
  * take joint commands from us, UDP port 20001: float32 x 12 each, kp | position | kd | velocity | feed-forward torque
    (240 bytes); the motor drivers apply tau = kp (q_cmd - q) + kd (dq_cmd - dq) + tau_ff.
Joint order FL, FR, HL, HR x (HipX, HipY, Knee) = ours (HL/HR = our RL/RR); DEEP's angles and speeds are minus ours
(robot_bodies "lite3": sdk_sign -1; lite3_nomesh.xml flips the axes to Bolt's convention).

    python deploy/lite3_link.py selftest     # packs and unpacks both messages, checks signs and the quaternion
"""
import socket
import struct
import sys
import threading
import time

import numpy as np

STATE_FMT = "<d45f"            # time, rpy 3, acc 3, gyro 3, q 12, dq 12, tau 12
CMD_FMT = "<60f"               # kp 12, q 12, kd 12, dq 12, tau 12
STATE_PORT, CMD_PORT = 30010, 20001


def rpy_to_quat(r, p, y):
    """ZYX Euler (DEEP's IMU and their simulator's quaternion_to_euler) -> quaternion w, x, y, z."""
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def pack_state(t, rpy, acc, gyro, q, dq, tau):
    return struct.pack(STATE_FMT, float(t), *np.concatenate([rpy, acc, gyro, q, dq, tau]).astype(np.float32))


def unpack_state(buf):
    v = struct.unpack(STATE_FMT, buf[:struct.calcsize(STATE_FMT)])
    f = np.asarray(v[1:], float)
    return v[0], f[0:3], f[3:6], f[6:9], f[9:21], f[21:33], f[33:45]


def pack_cmd(kp, q, kd, dq, tau):
    return struct.pack(CMD_FMT, *np.concatenate([np.broadcast_to(kp, 12), q, np.broadcast_to(kd, 12), dq, tau]).astype(np.float32))


class Lite3Robot:
    """run_go2.Robot's interface (read, set, running, lock) over DEEP's UDP protocol: latest state in our joint order and
    sign, a 500 Hz command writer. host: where the simulator or the bridge listens for commands."""

    def __init__(self, sign, kp, kd, host="127.0.0.1", state_port=STATE_PORT, cmd_port=CMD_PORT):
        self.sign = float(sign)
        self.state, self.sport, self.t_state = None, None, 0.0
        self.target, self.kp, self.kd = None, kp, kd
        self.lock, self.dropped = threading.Lock(), 0
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", state_port))
        self.rx.settimeout(0.2)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, cmd_port)
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._writer, daemon=True).start()

    def _reader(self):
        n = struct.calcsize(STATE_FMT)
        while self.running:
            try:
                buf, _ = self.rx.recvfrom(4096)
            except socket.timeout:
                continue
            # the robot sends 1000 states a second: drain the socket and keep only the newest (a reader that falls
            # behind would otherwise hand the brain states from a growing backlog)
            self.rx.setblocking(False)
            try:
                while True:
                    buf, _ = self.rx.recvfrom(4096)
                    self.dropped += 1
            except (BlockingIOError, OSError):
                pass
            self.rx.settimeout(0.2)
            if len(buf) >= n:
                with self.lock:
                    self.state, self.t_state = unpack_state(buf), time.perf_counter()

    def read(self):
        """(quat wxyz, gyro, q, dq, age s) in our joint order and sign, or None before the first state."""
        with self.lock:
            s, t = self.state, self.t_state
        if s is None:
            return None
        _, rpy, _, gyro, q, dq, _ = s
        return rpy_to_quat(*rpy), gyro.copy(), self.sign * q, self.sign * dq, time.perf_counter() - t

    def set(self, target, kp=None, kd=None):
        with self.lock:
            self.target = None if target is None else np.asarray(target, float).copy()
            if kp is not None:
                self.kp, self.kd = kp, kd

    def _writer(self):
        z = np.zeros(12)
        while self.running:
            t = time.perf_counter()
            with self.lock:
                tgt, kp, kd = self.target, self.kp, self.kd
            if tgt is None:   # damping mode: no position gain, only kd
                msg = pack_cmd(0.0, z, kd, z, z)
            else:
                msg = pack_cmd(kp, self.sign * tgt, kd, z, z)
            try:
                self.tx.sendto(msg, self.addr)
            except OSError:
                pass
            time.sleep(max(0.0, 0.002 - (time.perf_counter() - t)))


def selftest():
    rng = np.random.default_rng(0)
    rpy, acc, gyro = rng.normal(size=3) * 0.3, rng.normal(size=3), rng.normal(size=3)
    q, dq, tau = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
    t, *back = unpack_state(pack_state(1.5, rpy, acc, gyro, q, dq, tau))
    assert t == 1.5 and all(np.allclose(a, b, atol=1e-6) for a, b in zip(back, (rpy, acc, gyro, q, dq, tau)))
    assert len(pack_cmd(30.0, q, 1.0, dq, tau)) == 240
    c = struct.unpack(CMD_FMT, pack_cmd(30.0, q, 1.0, dq, tau))
    assert np.allclose(c[12:24], q, atol=1e-6) and c[0] == 30.0 and c[24] == 1.0
    # the quaternion against the rotation DEEP's simulator would report rpy for (mujoco's own conversion)
    import mujoco
    for _ in range(20):
        r = rng.uniform(-1, 1, 3) * [0.6, 0.6, 3.0]
        quat = rpy_to_quat(*r)
        w, x, y, z = quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        assert np.allclose([roll, pitch, yaw], r, atol=1e-6), (r, roll, pitch, yaw)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        assert abs(np.linalg.norm(quat) - 1) < 1e-9
    print("lite3_link selftest: state 1+45 values and 240-byte commands round-trip; rpy <-> quaternion matches DEEP's "
          "simulator conversion (20 random poses) -> PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
