"""Real-body stress suite: one real-robot issue at a time, the whole system (MasterMind's switch test), so we know which
issues break Bolt and must go into training (owner, 29 Oct: "whatever issues, latency or devices a physical body has,
we should be ready for all of it"). Simulation only; each level is a realistic one for an A1/Go1-class robot.

    cd train
    MUJOCO_GL=osmesa python research/realbody_stress.py mastermind/export_v8g/student.pt --teachers v8g --n 20 \
        --out research/results/realbody_v8g.json

Each row runs mastermind/test_master.py (seed 1000, with the 50 cm jump) in a subprocess with one issue switched on;
reported: switches OK / switches reached (pooled, as the switch test), and how many were reached at all (falls end a
sequence early, so fewer reached = more falls).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent

# (name, what a real robot has, test_master flags, extra environment)
ISSUES = [
    ("baseline", "no issue", [], {}),
    ("delay 10 ms", "control loop latency (A1/Go1: ~10-20 ms)", ["--latency-ms", "10"], {}),
    ("delay 20 ms", "", ["--latency-ms", "20"], {}),
    ("sensor noise", "IMU and encoder noise (legged_gym levels)", ["--obs-noise"], {}),
    ("IMU tilt bias 3 deg", "a mis-mounted or uncalibrated IMU", ["--imu-bias", "0.05"], {}),
    ("gyro bias 0.1 rad/s", "gyro offset drift", ["--gyro-bias", "0.1"], {}),
    ("speed estimate off 0.2 m/s", "a real robot estimates its body speed (no sensor measures it)", ["--vel-bias", "0.2"], {}),
    ("height estimate off 3 cm", "body height is estimated from leg kinematics", ["--height-bias", "0.03"], {}),
    ("encoder step 0.01 rad", "coarse joint encoders", ["--quant", "0.01"], {}),
    ("10% packets lost", "a lossy motor bus", ["--drop", "0.1"], {}),
    ("motor lag 20 ms", "motor current loop and gearbox response", ["--motor-lag-ms", "20"], {}),
    ("PD gains -20%", "motor constants off from the datasheet", ["--gain-scale", "0.8"], {}),
    ("motors 80% torque", "battery sag, hot motors", [], {"BOLT_MOTOR_TORQUE": "26.8"}),
    ("payload +20%", "a battery pack or sensor head", ["--payload", "0.2"], {}),
    ("slippery floor (grip x0.5)", "tiles, wet floor", ["--friction", "0.5"], {}),
]


def pooled(log):
    ok = n = 0
    for line in Path(log).read_text().splitlines():
        m = re.match(r"(.+?)\s+(\d+)/(\d+)\s+\d+%", line)
        if m and "[pose reached]" not in m.group(1):
            ok += int(m.group(2))
            n += int(m.group(3))
    return ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--teachers", default="v8g")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", help="issue names to run (default all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    logs = out.with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    issues = [i for i in ISSUES if not a.only or i[0] in a.only]

    def run(issue):
        name, why, flags, env = issue
        tag = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        log = logs / f"{tag}.log"
        cmd = [sys.executable, "mastermind/test_master.py", "--brain", a.brain, "--teachers", a.teachers, "--n", str(a.n),
               "--seed", str(a.seed), "--with-50"] + flags
        t0 = time.time()
        with open(log, "w") as f:
            subprocess.run(cmd, cwd=TRAIN, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, MUJOCO_GL="osmesa", OMP_NUM_THREADS="1", **env))
        ok, n = pooled(log)
        row = dict(issue=name, real_robot=why, flags=flags, env=env, ok=ok, reached=n,
                   rate=round(ok / n, 3) if n else None, minutes=round((time.time() - t0) / 60, 1))
        print(f"{name:30s} {ok}/{n} = {100 * ok / max(n, 1):.1f}%", flush=True)
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, issues))
    res = dict(brain=a.brain, teachers=a.teachers, n=a.n, seed=a.seed, rows=rows, command="python " + " ".join(sys.argv))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
