"""Shrink (or grow) Bolt while keeping it a dynamically similar copy.

    python scale_robot.py 0.25 -o bolt_s025.xml

If every length is multiplied by s, a robot behaves like a sped-up (or slowed-down)
copy of the original when the other quantities follow the matching physics:

    length ×s      mass ×s³      motor torque ×s⁴      time ×√s
    speed ×√s      angular speed ×1/√s      inertia ×s⁵

Joint angles don't change at all. So a brain trained on the big robot can drive the
small one, provided its sensor readings are converted back into big-robot units
(BoltEnv(scale=s) does that). Real small motors won't follow these rules exactly;
measure them and fine-tune.
"""
import argparse
import math
import xml.etree.ElementTree as ET

from bolt_env import MODEL


def _scaled(value, factor):
    return " ".join(f"{float(v) * factor:.6g}" for v in value.split())


def scaled_xml(s, path=MODEL):
    """Return the robot model, scaled by s, as an MJCF string."""
    root = ET.parse(path).getroot()
    t = math.sqrt(s)  # time scale

    for opt in root.iter("option"):
        opt.set("timestep", f"{float(opt.get('timestep', 0.002)) * t:.6g}")

    for joint in root.iter("joint"):
        for attr, power in (("damping", 4.5), ("armature", 5), ("frictionloss", 4)):
            if joint.get(attr):
                joint.set(attr, _scaled(joint.get(attr), s ** power))

    for act in list(root.iter("position")):
        if act.get("kp"):
            act.set("kp", _scaled(act.get("kp"), s ** 4))
        if act.get("kv"):
            act.set("kv", _scaled(act.get("kv"), s ** 4.5))
        if act.get("forcerange"):
            act.set("forcerange", _scaled(act.get("forcerange"), s ** 4))

    for default_geom in root.find("default").iter("geom"):
        # contact softness is a time constant: scale it with time
        default_geom.set("solref", f"{0.02 * t:.6g} 1")

    for el in root.iter():
        if el.tag in ("light", "texture", "material", "global", "headlight"):
            continue
        if el.tag == "geom" and el.get("type") == "plane":
            continue
        for attr in ("pos", "fromto"):
            if el.get(attr):
                el.set(attr, _scaled(el.get(attr), s))
        if el.tag == "geom":
            if el.get("size"):
                el.set("size", _scaled(el.get("size"), s))
            if el.get("mass"):
                el.set("mass", _scaled(el.get("mass"), s ** 3))
    return ET.tostring(root, encoding="unicode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", type=float, help="size factor, e.g. 0.25 for a hand-sized robot")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        f.write(scaled_xml(args.scale))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
