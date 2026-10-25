"""The reference for the WASM check: one hybrid duck, velstand, BAM servos, 10 s of standing / walking / turning, in
Python MuJoCo (the same exported one-duck model the page's code is checked on, and match.py's MultiBam), no noise,
no delay, no randomization. compare_walk.mjs runs the same in MuJoCo WASM + onnxruntime-web; --compare prints both.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/live/export_live.py DIR --one
    $PY duckbody/hybrid/live/compare_walk.py DIR                       # -> DIR/walk_py.json
    (cd NODE_DIR && node .../compare_walk.mjs DIR)                     # -> DIR/walk_js.json
    $PY duckbody/hybrid/live/compare_walk.py DIR --compare
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "microduck_port"))
sys.path.insert(0, str(HERE.parent))
import mujoco  # noqa: E402
import match as M  # noqa: E402
import harness as Hm  # noqa: E402
from bam.model import load_model  # noqa: E402

SECONDS = 10.0


def schedule(k):
    """Twist command at tick k (50 Hz): stand 0.5 s, walk 0.4, walk + turn, sidestep, stand."""
    t = k * 0.02
    if t < 0.5:
        return 0.0, 0.0, 0.0
    if t < 4.0:
        return 0.4, 0.0, 0.0
    if t < 6.5:
        return 0.4, 0.0, 0.4
    if t < 8.5:
        return 0.0, 0.35, 0.0
    return 0.0, 0.0, 0.0


def run(d_dir):
    xml = (d_dir / "live_model_one.xml").read_text()
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = Hm.TIMESTEP
    d = mujoco.MjData(m)
    bam = load_model(motor_name="xl330", model="m6")
    bam.actuator.kp, bam.actuator.vin, bam.actuator.max_current = Hm.BAM_KP_FW, Hm.BAM_VIN, None
    mb = M.MultiBam(bam, m, d, ["b1"])
    d.qpos[0:7] = [0, 0, M.HOME_TRUNK_Z, 1, 0, 0, 0]
    d.qpos[mb.qadr] = Hm.DEFAULT_POSE
    mb.q_target[:] = Hm.DEFAULT_POSE
    mujoco.mj_forward(m, d)
    pol = Hm.Policy("velstand")
    tr, gy = m.body("b1_trunk_base").id, m.sensor("b1_imu_ang_vel").adr[0]
    last = np.zeros(14, np.float32)
    out = dict(trunk=[], yaw=[], q=[], obs0=None, act0=None)
    for k in range(int(SECONDS * 50)):
        c = np.zeros(13, np.float32)
        c[:3] = schedule(k)
        o = np.concatenate([d.sensordata[gy:gy + 3], -d.xmat[tr].reshape(3, 3)[2, :], d.qpos[mb.qadr] - Hm.DEFAULT_POSE,
                            d.qvel[mb.dof], last, c]).astype(np.float32)
        last = pol(o)
        if k == 0:
            out["obs0"], out["act0"] = o.tolist(), last.tolist()
        mb.q_target[:] = Hm.DEFAULT_POSE + last
        for _ in range(4):
            mb.update()
            mujoco.mj_step(m, d)
        R = d.xmat[tr].reshape(3, 3)
        out["trunk"].append(d.xpos[tr].round(5).tolist())
        out["yaw"].append(round(math.atan2(R[1, 0], R[0, 0]), 5))
        out["q"].append(d.qpos[mb.qadr].round(4).tolist())
    out["mujoco"] = mujoco.__version__
    return out


def summary(r):
    P = np.array(r["trunk"])
    yaw = np.unwrap(np.array(r["yaw"]))
    q = np.array(r["q"])
    seg = {}
    for name, (a, b) in dict(walk=(1.0, 4.0), walk_turn=(4.5, 6.5), sidestep=(7.0, 8.5)).items():
        i, j = int(a * 50), int(b * 50)
        dt = b - a
        dx = P[j] - P[i]
        # speed in the duck's own frame at the start of the segment
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        seg[name] = dict(v_fwd=round((c * dx[0] + s * dx[1]) / dt, 3), v_side=round((-s * dx[0] + c * dx[1]) / dt, 3),
                         yaw_rate=round((yaw[j] - yaw[i]) / dt, 3))
    knee = q[50:200, 3]
    spec = np.abs(np.fft.rfft(knee - knee.mean()))
    f = np.fft.rfftfreq(len(knee), 0.02)
    return dict(segments=seg, trunk_z_mean=round(float(P[50:, 2].mean()), 4), trunk_z_std=round(float(P[50:, 2].std()), 4),
                step_hz=round(float(f[1 + np.argmax(spec[1:])]), 2), knee_amp=round(float(knee.std() * 2 ** 0.5), 3),
                end=[round(v, 3) for v in P[-1]], fell=bool(P[:, 2].min() < 0.08))


def main():
    d_dir = Path(sys.argv[1])
    if "--compare" in sys.argv:
        py = json.loads((d_dir / "walk_py.json").read_text())
        js = json.loads((d_dir / "walk_js.json").read_text())
        o0 = np.abs(np.array(py["obs0"]) - np.array(js["obs0"])).max()
        a0 = np.abs(np.array(py["act0"]) - np.array(js["act0"])).max()
        P, J = np.array(py["trunk"]), np.array(js["trunk"])
        dd = np.linalg.norm(P - J, axis=1)
        res = dict(python=dict(mujoco=py["mujoco"], **summary(py)), wasm=dict(mujoco=js["mujoco"], **summary(js)),
                   first_tick=dict(obs_max_diff=float(o0), action_max_diff=float(a0)),
                   trunk_distance_mm={f"{t}s": round(float(dd[int(t * 50) - 1]) * 1000, 2) for t in (0.1, 0.5, 1, 2, 4, 6, 8, 10)})
        print(json.dumps(res, indent=1))
        (d_dir / "walk_compare.json").write_text(json.dumps(res, indent=1))
        return
    r = run(d_dir)
    (d_dir / "walk_py.json").write_text(json.dumps(r))
    print("python", json.dumps(summary(r)))


if __name__ == "__main__":
    main()
