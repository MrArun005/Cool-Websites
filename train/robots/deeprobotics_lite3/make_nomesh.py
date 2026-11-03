"""Lite3_vendor.xml (DEEP Robotics' MJCF, URDF_model repo, BSD-3-Clause, see LICENSE) -> lite3_nomesh.xml for Bolt.

    python make_nomesh.py

Kept exactly: every body position, mass and inertia, joint position and range (sign-flipped, below), foot spheres.
Changed:
  * joint axes to Bolt's convention (robot_bodies.measure): abduction +x, thigh and knee +y, knee bends with negative
    angles. The vendor uses -x / -y, so each axis is negated and its range negated and swapped: thigh [-2.67, 0.314] ->
    [-0.314, 2.67], knee [0.524, 2.792] -> [-2.792, -0.524]. A Bolt angle is minus the vendor's (the robot program
    flips the signs back).
  * the STL collision meshes -> boxes / capsules sized from the meshes' bounds in each body's frame (the GPU trainer,
    MJX, is slow with meshes; Go2's nomesh file does the same): trunk box, hip cylinder, a thick thigh capsule offset
    toward the body, and a shank capsule down to the foot (SHAPES below; the first version's thin, short legs never
    touched the floor where DEEP's do, and a brain trained on it fell going sit -> lie on DEEP's model).
  * meshes, cameras, the lidar / depth sites and sensors dropped (robot_bodies adds Bolt's floor, light and servos);
    the IMU site kept. Rear legs renamed HL_ / HR_ -> RL_ / RR_ (Bolt finds a leg's geoms by that prefix). armature 0.01 added on every joint (rotor inertia; the vendor file has none, Go2's 0.01).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = ET.parse(HERE / "Lite3_vendor.xml")
root = t.getroot()
for tag in ("asset", "sensor", "actuator", "contact"):   # contact: parent-child excludes, moot (robot geoms are contype 0)
    for e in root.findall(tag):
        root.remove(e)
ET.SubElement(root, "asset")
wb = root.find("worldbody")
for g in list(wb.findall("geom")) + list(wb.findall("light")):
    wb.remove(g)
for d in root.iter("default"):
    for g in d.findall("geom"):
        g.attrib.pop("material", None)
    if d.get("class") == "motor":
        d.find("joint").set("armature", "0.01")


def flip(j):
    ax = [float(v) for v in j.get("axis").split()]
    lo, hi = (float(v) for v in j.get("range").split())
    j.set("axis", " ".join(str(-a + 0.0) for a in ax))
    j.set("range", f"{-hi:g} {-lo:g}")


# body suffix -> collision geoms; sizes from the vendor meshes' bounds in each body's own frame (measured 3 Nov: trunk x
# -0.269..0.279, z -0.054..0.073; thigh y -0.090..0.027 on a left leg (the motor housing sits toward the body), z down
# to -0.215; shank x -0.022..0.028, z down to -0.233, i.e. the shank mesh runs down over the foot: on DEEP's model the
# shanks, not only the foot spheres, carry the robot whenever a leg is angled, and folded rear legs sit on them).
# "{y}" = the thigh's inward offset, -0.031 on left legs, +0.031 on right.
SHAPES = {
    "TORSO": [dict(type="box", size="0.274 0.108 0.0635", pos="0.005 0 0.0095")],
    "HIP": [dict(type="cylinder", size="0.045 0.03", quat="0.707107 0.707107 0 0")],
    "THIGH": [dict(type="capsule", fromto="0 {y} -0.01 0 {y} -0.18", size="0.042")],
    "SHANK": [dict(type="capsule", fromto="0.003 0 0.0 0 0 -0.206", size="0.021")],   # ends where the foot ball takes over
}
for b in wb.iter("body"):
    name = b.get("name")
    for c in list(b):
        if c.tag == "camera" or (c.tag == "site" and c.get("name") != "imu_site"):
            b.remove(c)
        elif c.tag == "geom" and (c.get("class") == "visual" or c.get("type") == "mesh"):
            b.remove(c)
        elif c.tag == "joint":
            flip(c)
    kind = name.split("_")[-1]
    for i, a in enumerate(SHAPES.get(kind, [])):
        y = "-0.031" if name[1:2] == "L" else "0.031"
        a = {k: v.replace("{y}", y) for k, v in a.items()}
        g = ET.Element("geom", dict(name=f"{name}_col{i}", **a), **{"class": "collision"})
        b.insert(0 if kind != "TORSO" else 3, g)
for e in root.iter():   # rear legs HL / HR -> Bolt's RL / RR (Bolt's envs tell legs apart by the name prefix)
    n = e.get("name")
    if n and n[:3] in ("HL_", "HR_"):
        e.set("name", "R" + n[1:])
ET.indent(t)
out = HERE / "lite3_nomesh.xml"
t.write(out, encoding="unicode")
print("wrote", out)
