"""Sit to stand that works: a floor seat the duck can get up from ("perch"), and the stand-up test.

    python s2s_perch.py                 # search perch poses, test wake-up from each, save the best to poses_mid.json
    python s2s_perch.py --test-only     # re-test the saved perch (e.g. after retraining the stand brain)

The measured swag / sit seats cannot be stood up from: the feet hang in the air and the trunk pivots on
its tail, so every push flips the body onto its back (s2s_script.py, s2s_plan.py: 0 rises in ~350 scripts,
also at 2x torque). A perch is the opposite: bum (belly / thighs) on the floor AND both soles planted.
From there the stand-up is the existing rise plan (glide the servos to home over ~1 s, then the stand brain).

Stage 1  poses.try_pose over a grid of mirrored leg angles (hip pitch, knee, ankle): keep poses that settle
         upright and still, touching the floor with the feet and with belly or thighs. Lowest trunk first.
Stage 2  for each candidate: load it into the real runtime (SimBackend + ONNX stand brain), randomize the
         start (joint noise, heading, friction, mass), run "rise" and count how often it is standing after
         `--seconds`. The first candidate at or above `--pass` is saved as "perch" (qpos, ctrl, up, height)
         and the runtime's PLANS pick it up: sit = squat -> pose:perch, wake from a pose = rise.
"""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from poses import leg_q, try_pose
from runtime import DuckRuntime, PoseBrain, SimBackend, load_brains

HERE = Path(__file__).resolve().parent
LEG = ("hip_pitch", "knee", "ankle")


def perch_candidates(m, d, n_hip=9, n_knee=6, n_ankle=11, min_up=0.85, max_drift=0.003):
    rng = {k: m.actuator_ctrlrange[m.actuator(f"left_{k}").id] for k in LEG}
    home = {k: m.key("home").ctrl[m.actuator(f"left_{k}").id] for k in LEG}
    hips = np.linspace(rng["hip_pitch"][0], rng["hip_pitch"][1], n_hip)
    knees = np.linspace(home["knee"], rng["knee"][1], n_knee)          # from the home knee to the fold limit
    ankles = np.linspace(rng["ankle"][0], rng["ankle"][1], n_ankle)
    out = []
    for h, k, an in itertools.product(hips, knees, ankles):
        r = try_pose(m, d, h, k, an)
        if r["up"] > min_up and r["drift"] < max_drift and "feet" in r["touch"] and ({"belly", "thighs"} & set(r["touch"])):
            out.append(r)
    out.sort(key=lambda r: (r["trunk_bottom"], r["torque"]))
    return out


def perch_ctrl(m, r):
    """Servo targets for a candidate: the measured leg angles (both legs mirrored), everything else home."""
    c = m.key("home").ctrl.copy()
    q = np.zeros(m.nq)
    leg_q(m, q, r["hip"], r["knee"], r["ankle"])
    for i in range(m.nu):
        n = m.actuator(i).name
        if any(n.endswith(k) for k in LEG):
            c[i] = q[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    return c


def test_rise(r, brains, cfg, seeds, seconds, randomize=True, log=None):
    """Start the runtime in this seat, run the rise plan, report how many end standing."""
    ok, ends = 0, []
    for s in seeds:
        rng = np.random.default_rng(s)
        be = SimBackend(cfg["servo_names"])
        m, d = be.model, be.data
        if randomize:
            f = rng.uniform(0.9, 1.1, m.nbody)
            m.body_mass[:] = m.body_mass * f
            m.body_inertia[:] = m.body_inertia * f[:, None]
            m.geom_friction[:, 0] = rng.uniform(0.4, 1.0)
        rt = DuckRuntime(backend=be, brains=dict(brains), config=cfg, log=log)
        ctrl = perch_ctrl(m, r)
        q = np.array(r["qpos"])
        if randomize:
            q[be.qadr] = np.clip(q[be.qadr] + rng.normal(0, 0.02, m.nu), rt.lo, rt.hi)
            yaw = rng.uniform(-np.pi, np.pi)
            q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        rt.brains["pose:perch"] = PoseBrain(ctrl, rt)
        rt.skill = "pose:perch"
        rt.last_action = np.clip((ctrl[rt.pol] - rt.home[rt.pol]) / rt.scale, -1, 1)
        rt.run(0.5)                                    # settle in the seat first
        rt.command("rise")
        min_up = 1.0
        for _ in range(int(seconds * rt.hz)):
            rt.tick()
            min_up = min(min_up, be.posture()["up"])
        p = be.posture()
        stood = p["up"] > 0.9 and p["height"] > 0.9 * cfg["stand_height"] and min_up > 0.3
        ok += stood
        ends.append((round(p["up"], 2), round(p["height"], 3)))
    return ok, ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model" / "duck_mid.xml"))
    ap.add_argument("--poses", default=str(HERE / "model" / "poses_mid.json"))
    ap.add_argument("--models", default=str(HERE.parent / "models"))
    ap.add_argument("--candidates", type=int, default=6, help="how many perch candidates to test for wake-up")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--pass", dest="pass_frac", type=float, default=0.9)
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args()
    brains, cfg = load_brains(Path(a.models))
    poses = json.loads(Path(a.poses).read_text())
    seeds = list(range(100, 100 + a.seeds))
    if a.test_only:
        r = poses["perch"]
        ok, ends = test_rise(r, brains, cfg, seeds, a.seconds)
        print(f"perch wake-up: {ok}/{len(seeds)} stood", ends)
        return
    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    cands = perch_candidates(m, d)
    print(f"{len(cands)} perch candidates (feet planted, bum on the floor, upright, still); lowest first")
    best = None
    for r in cands[: a.candidates]:
        ok, ends = test_rise(r, brains, cfg, seeds, a.seconds)
        print(f"  hip {r['hip']:+.2f} knee {r['knee']:.2f} ankle {r['ankle']:+.2f} | up {r['up']:.2f} base {r['height'] * 1000:.0f} mm "
              f"belly {r['trunk_bottom'] * 1000:.0f} mm | touch {r['touch']} | leg torque {r['torque']:.3f} | wake-up {ok}/{len(seeds)}")
        if best is None or ok > best[0]:
            best = (ok, r)
        if ok >= a.pass_frac * len(seeds):
            break
    ok, r = best
    if ok < a.pass_frac * len(seeds):
        print(f"best perch only woke {ok}/{len(seeds)}: not saved (bar is {a.pass_frac:.0%}); widen the grid or check the rise glide")
        return
    ctrl = perch_ctrl(m, r)
    poses["perch"] = dict(qpos=r["qpos"], ctrl=ctrl.round(5).tolist(), up=r["up"], height=r["height"],
                          legs=dict(hip=r["hip"], knee=r["knee"], ankle=r["ankle"]), touch=r["touch"],
                          wake_up=f"{ok}/{len(seeds)} stood ({a.seconds:g} s, randomized)",
                          note="floor seat with both soles planted: sit = squat -> pose:perch, wake = rise (glide home, stand brain)")
    Path(a.poses).write_text(json.dumps(poses, indent=1))
    print(f"saved perch to {a.poses}: wake-up {ok}/{len(seeds)}")


if __name__ == "__main__":
    main()
