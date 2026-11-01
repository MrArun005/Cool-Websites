"""Is this machine ready to run Bolt on a Go2? (HARDWARE_DAY.md step 0; setup_go2.sh runs it at the end.)

    python deploy/check_ready.py [--iface <eth>]        # --iface: also check the robot's network link

Each check prints PASS / FAIL and what to do on a FAIL; the last line is READY or NOT READY. Read-only: never sends a
command to a robot. Checks: Python and packages; both brains load (motion: models/bolt-go2-damp-798M, self-right:
models/bolt-go2-selfright-6.9M brain.npz); the motion brain's parity test (controller vs the training env); the brain
step time on this machine (the loop has 20 ms; HARDWARE_DAY wants it well under); CycloneDDS can open a channel; with
--iface, that interface exists and has a 192.168.123.x address (Unitree's subnet).
"""
import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
MODELS = HERE.parent.parent / "models"
BRAIN = MODELS / "bolt-go2-damp-798M" / "params.pkl"
SELFRIGHT = MODELS / "bolt-go2-selfright-6.9M"
results = []


def check(name, fix):
    def wrap(fn):
        try:
            note = fn() or ""
            results.append(True)
            print(f"PASS  {name}{'  (' + note + ')' if note else ''}", flush=True)
        except Exception as e:   # noqa: BLE001 -- every failure is reported, none stops the other checks
            results.append(False)
            print(f"FAIL  {name}: {type(e).__name__}: {e}\n      fix: {fix}", flush=True)
        return fn
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", help="the robot's Ethernet interface (e.g. enp3s0); omit before the robot is connected")
    a = ap.parse_args()

    @check("Python 3.10+", "use setup_go2.sh's venv (Python 3.11)")
    def _():
        assert sys.version_info >= (3, 10), sys.version
        return sys.version.split()[0]

    @check("packages numpy, mujoco, gymnasium, cyclonedds, unitree_sdk2py", "re-run setup_go2.sh; source bolt-deps/env.sh")
    def _():
        v = {}
        for m in ("numpy", "mujoco", "gymnasium", "cyclonedds", "unitree_sdk2py"):
            v[m] = getattr(importlib.import_module(m), "__version__", "?")
        assert v["mujoco"].startswith("3."), v
        return ", ".join(f"{k} {x}" for k, x in v.items())

    @check("motion brain loads", f"is {BRAIN} in this checkout (git lfs / full clone)?")
    def _():
        from controller import Go2Controller
        ctl = Go2Controller(str(BRAIN))
        assert ctl.brain.blind_vel, "not a --blind-vel brain: it needs a velocity sensor a real Go2 lacks"
        return f"{BRAIN.parent.name}, no velocity input"

    @check("self-right brain loads (numpy)", f"run: python deploy/selfright.py export {SELFRIGHT} (needs SB3) or pull brain.npz")
    def _():
        from selfright import SelfRighter
        sr = SelfRighter(SELFRIGHT)
        assert len(sr.layers) == 3
        return SELFRIGHT.name

    @check("parity: controller == training env (recorded reference)", "the controller and the env disagree: do not run "
           "the robot; on a dev machine with torch: python deploy/test_deploy.py <brain> --save")
    def _():
        out = subprocess.run([sys.executable, str(HERE / "test_deploy.py"), str(BRAIN), "--golden"], capture_output=True, text=True,
                             timeout=600, cwd=HERE.parent)
        assert out.returncode == 0, (out.stdout + out.stderr)[-400:]
        return out.stdout.strip().splitlines()[-1][:90] if out.stdout.strip() else "ok"

    @check("brain step time on this machine", "too slow: close other programs, or run on a faster machine")
    def _():
        import numpy as np
        from controller import Go2Controller
        from estimator import LegEstimator
        from recovery import Recovery
        ctl = Go2Controller(str(BRAIN))
        rec, est = Recovery(ctl, SELFRIGHT), LegEstimator()
        quat, q = np.array([1.0, 0, 0, 0]), ctl.stance.copy()
        ts = []
        for k in range(200):
            t = time.perf_counter()
            v, h = est.update(quat, q)
            tgt = rec.step(quat, np.zeros(3), q, np.zeros(12), v, h, resume="trot")
            if tgt is None:
                ctl.step(quat, np.zeros(3), q, np.zeros(12), v, h)
            rec.sr.step(quat, np.zeros(3), q, np.zeros(12), v, h)
            ts.append(time.perf_counter() - t)
        p99 = float(np.percentile(ts, 99)) * 1000
        assert p99 < 10.0, f"99th percentile {p99:.1f} ms of the 20 ms budget"
        return f"motion + self-right + estimator: median {np.median(ts) * 1000:.2f} ms, 99% {p99:.2f} ms of 20 ms"

    @check("CycloneDDS opens a channel", "source bolt-deps/env.sh (CYCLONEDDS_HOME, LD_LIBRARY_PATH)")
    def _():
        out = subprocess.run([sys.executable, "-c", "from unitree_sdk2py.core.channel import ChannelFactoryInitialize;"
                              "ChannelFactoryInitialize(7, 'lo'); print('ok')"], capture_output=True, text=True, timeout=30)
        assert "ok" in out.stdout, (out.stdout + out.stderr)[-300:]

    if a.iface:
        @check(f"network {a.iface} on Unitree's subnet 192.168.123.x", f"set {a.iface} to e.g. 192.168.123.222/24")
        def _():
            import fcntl
            import socket
            import struct
            assert Path(f"/sys/class/net/{a.iface}").exists(), f"no interface {a.iface} (see: ls /sys/class/net)"
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:   # SIOCGIFADDR: the interface's IPv4 address
                ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", a.iface[:15].encode()))[20:24])
            except OSError:
                raise AssertionError(f"{a.iface} has no IPv4 address") from None
            assert ip.startswith("192.168.123."), f"{a.iface} is {ip}, not 192.168.123.x"
            return ip

    ok = all(results)
    print(f"\n{'READY' if ok else 'NOT READY'}: {sum(results)}/{len(results)} checks passed"
          + ("" if ok else " -- fix the FAILs above, then run this again"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
