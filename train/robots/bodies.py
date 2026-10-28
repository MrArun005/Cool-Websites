"""Bolt next to the Unitree A1 and Go2 (MuJoCo Menagerie models): masses, leg lengths, joint ranges, motor limits, feet.

    cd train && python robots/bodies.py            # writes robots/*_nomesh.xml (visual meshes stripped) and prints the table
    python robots/bodies.py --json robots/bodies.json

The Menagerie xml files are copied unchanged (BSD-3-Clause, Unitree Robotics; see each LICENSE). Their visual meshes are
not in git (robots/fetch_meshes.sh); *_nomesh.xml drop the mesh assets and the visual geoms only, so the physics (bodies,
inertias, joints, actuators, collision primitives) is the published model's.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
sys.path.insert(0, str(TRAIN))

# Joint motor ratings (N·m peak, rad/s max joint speed) as Unitree publishes them; the Menagerie files carry only the
# torque limit. A1: 33.5 N·m, 21 rad/s (also bolt_env.MOTOR). Go2: 23.7 N·m hip/thigh, 45.43 N·m knee (go2.xml ctrlrange;
# the knee has a 1.5x extra gear), 30.1 rad/s hip/thigh and 15.7 rad/s knee (Unitree's go2_description URDF velocity limits).
RATED = {"bolt": dict(speed="21 (fades to 0 at 25)", power="420 W"),
         "a1": dict(speed="21", power="-"),
         "go2": dict(speed="30.1 hip/thigh, 15.7 knee", power="-")}


def strip(src, dst):
    s = Path(src).read_text()
    s = re.sub(r'\s*<mesh [^>]*/>', "", s)
    s = re.sub(r'\s*<texture type="2d" name="trunk_A1"[^>]*/>', "", s)
    s = re.sub(r'\s*<material name="carbonfibre"[^>]*/>', "", s)
    s = re.sub(r'\s*<geom [^>]*class="visual"[^>]*/>', "", s)
    s = re.sub(r'\s*<geom [^>]*mesh="[^"]*"[^>]*/>', "", s)
    Path(dst).write_text(s)
    return dst


def describe(name, path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = [b for b in range(m.nbody) if m.body_parentid[b] == 0 and m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == 0][0]
    robot = [b for b in range(m.nbody) if m.body_rootid[b] == root]
    total = float(sum(m.body_mass[b] for b in robot))
    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == 3 and m.body_rootid[m.jnt_bodyid[j]] == root]
    names = [m.joint(j).name for j in hinges]
    # leg lengths: front-left leg, hip(thigh) joint -> knee joint -> foot centre, at qpos0
    fl = [j for j in hinges if m.joint(j).name.upper().startswith("FL")]
    abd, hip, knee = fl[0], fl[1], fl[2]
    pj = lambda j: d.xanchor[j].copy()  # noqa: E731
    feet = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.body_rootid[m.geom_bodyid[g]] == root
            and m.geom_bodyid[g] == m.jnt_bodyid[knee] and m.geom_contype[g] | m.geom_conaffinity[g]]
    foot = feet[-1]
    thigh = float(np.linalg.norm(pj(knee) - pj(hip)))
    shin = float(np.linalg.norm(d.geom_xpos[foot] - pj(knee)))
    # hip spacing: front-back and left-right distance between the FL and RR abduction anchors
    rr = [j for j in hinges if m.joint(j).name.upper().startswith("RR")]
    span = d.xanchor[abd] - d.xanchor[rr[0]]
    span_hip = d.xanchor[hip] - d.xanchor[rr[1]]   # thigh joints (A1/Go2 have a 8-10 cm hip offset outboard of abd)
    stand = None
    if m.nkey:
        stand = round(float(m.key_qpos[0][2]), 3)
    acts = list(range(m.nu))
    fr = m.actuator_forcerange[acts] if m.actuator_forcelimited.any() else m.actuator_ctrlrange[acts]
    kind = "position (kp %g, kv %g)" % (m.actuator_gainprm[0, 0], abs(m.actuator_biasprm[0, 2])) if m.actuator_biastype[0] == 1 else "torque"
    rng = {k: [round(float(v), 3) for v in m.jnt_range[j]] for k, j in zip(("abd", "hip", "knee"), fl)}
    return dict(robot=name, xml=str(Path(path).relative_to(TRAIN)), mass_kg=round(total, 2),
                torso_kg=round(float(m.body_mass[root]), 2), legs_kg=round(total - float(m.body_mass[root]), 2),
                thigh_m=round(thigh, 3), shin_m=round(shin, 3), hip_spacing_m=[round(abs(float(span[0])), 3), round(abs(float(span[1])), 3)],
                thigh_joint_width_m=round(abs(float(span_hip[1])), 3), keyframe_height_m=stand,
                knees="all backward" if all(m.jnt_range[j][1] <= 0 for j in hinges[2::3]) else "front backward, rear forward",
                range_FL_rad=rng, actuator=kind,
                torque_limit_Nm=sorted({round(float(abs(v)), 2) for v in np.abs(fr).max(axis=1)}),
                motor_speed_rad_s=RATED[name]["speed"], motor_power=RATED[name]["power"],
                foot=f"sphere r={m.geom_size[foot][0]:.3f} m, friction {m.geom_friction[foot][0]:g}, condim {m.geom_condim[foot]}",
                joint_damping=float(m.dof_damping[m.jnt_dofadr[knee]]), frictionloss=float(m.dof_frictionloss[m.jnt_dofadr[knee]]),
                timestep=float(m.opt.timestep), joints=names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    a1 = strip(HERE / "unitree_a1" / "a1.xml", HERE / "unitree_a1" / "a1_nomesh.xml")
    go2 = strip(HERE / "unitree_go2" / "go2.xml", HERE / "unitree_go2" / "go2_nomesh.xml")
    rows = [describe("bolt", TRAIN / "bolt.xml"), describe("a1", a1), describe("go2", go2)]
    keys = ["mass_kg", "torso_kg", "legs_kg", "thigh_m", "shin_m", "hip_spacing_m", "thigh_joint_width_m", "knees", "keyframe_height_m", "range_FL_rad", "actuator",
            "torque_limit_Nm", "motor_speed_rad_s", "motor_power", "foot", "joint_damping", "frictionloss", "timestep"]
    print("| | " + " | ".join(r["robot"] for r in rows) + " |")
    print("|---" * (len(rows) + 1) + "|")
    for k in keys:
        print(f"| {k} | " + " | ".join(str(r[k]) for r in rows) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
