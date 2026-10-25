"""A real agility hurdle for Bolt: two solid wing stands and a loose bar resting in cups.

Like a real one: each stand is an upright pole with a wing panel outside it, both hard (nothing passes
through). The bar is its own light pole lying in two cups with small lips, so a light brush leaves it in
place and a real knock lifts it over a lip and drops it, which is the only fault in agility.

Sized for Bolt (body 56 cm long, 24 cm wide): 1.2 m bar, uprights 1.28 m apart, wings out to 0.94 m.
The stands are mocap bodies (moved by the game, never by contacts), the bars free bodies. Bars go at
the end of the world, after the robot, so the robot's qpos (0-18) and qvel (0-17) keep their places.

One definition for training (jump_env.py), the browser physics (export_live.py) and the page drawing
(the numbers go into the bundle as constants.real_hurdle).
"""
import numpy as np

# Everything is measured from the stand body's origin, which sits at the bar's top height.
GEO = dict(
    post_y=0.64, post_r=0.02, post_top=0.10, bottom=-0.75,      # uprights: from 75 cm below bar top to 10 cm above
    wing_y=(0.66, 0.94), wing_t=0.015, wing_top=0.05,            # wing panels outside the uprights
    cup_y=0.585, cup=(0.03, 0.03, 0.005), cup_top=-0.03,         # cups the bar rests in (half sizes)
    lip=(0.004, 0.03, 0.003), lip_x=0.034,                       # 6 mm lips in front of and behind the bar
    bar_r=0.015, bar_half=0.6, bar_mass=0.25,
)
STAND_RGBA = "0.94 0.94 0.92 1"
WING_RGBA = "0.18 0.44 0.71 1"
BAR_RGBA = "0.77 0.22 0.18 1"
PARK = (100.0, 0.0, 0.15)   # where unused hurdles wait (bar top 15 cm up), 3 m apart along y


def stand_geoms(k):
    """The geoms of stand k (uprights, wings, cups, lips), local to its body."""
    g, out = GEO, []
    post_h = (g["post_top"] - g["bottom"]) / 2
    wing_h = (g["wing_top"] - g["bottom"]) / 2
    for s, side in ((1, "L"), (-1, "R")):
        out.append(f'<geom name="h{k}_post{side}" type="cylinder" size="{g["post_r"]} {post_h:.4f}" '
                   f'pos="0 {s * g["post_y"]} {g["bottom"] + post_h:.4f}" rgba="{STAND_RGBA}"/>')
        wy = sum(g["wing_y"]) / 2
        out.append(f'<geom name="h{k}_wing{side}" type="box" size="{g["wing_t"]} {(g["wing_y"][1] - g["wing_y"][0]) / 2:.4f} {wing_h:.4f}" '
                   f'pos="0 {s * wy:.4f} {g["bottom"] + wing_h:.4f}" rgba="{WING_RGBA}"/>')
        cx, cy, cz = g["cup"]
        out.append(f'<geom name="h{k}_cup{side}" type="box" size="{cx} {cy} {cz}" '
                   f'pos="0 {s * g["cup_y"]} {g["cup_top"] - cz:.4f}" rgba="{STAND_RGBA}"/>')
        lx, ly, lz = g["lip"]
        for f in (1, -1):
            out.append(f'<geom name="h{k}_lip{side}{"F" if f > 0 else "B"}" type="box" size="{lx} {ly} {lz}" '
                       f'pos="{f * g["lip_x"]} {s * g["cup_y"]} {g["cup_top"] + lz:.4f}" rgba="{STAND_RGBA}"/>')
    return out


def stands_xml(n):
    """n hurdle stands, parked away from the course (goes with the floor, before the robot)."""
    return "\n    ".join(
        f'<body name="hurdle{k}" mocap="true" pos="{PARK[0]} {PARK[1] + 3 * k} {PARK[2]}">' + "".join(stand_geoms(k)) + "</body>"
        for k in range(n))


def bars_xml(n):
    """n loose bars, each resting in its parked stand's cups (goes at the end of the world, after the robot)."""
    g = GEO
    return "\n    ".join(
        f'<body name="bar{k}" pos="{PARK[0]} {PARK[1] + 3 * k} {PARK[2] - g["bar_r"]}"><freejoint name="bar{k}"/>'
        f'<geom name="bar{k}" type="capsule" size="{g["bar_r"]}" fromto="0 {-g["bar_half"]} 0 0 {g["bar_half"]} 0" '
        f'mass="{g["bar_mass"]}" rgba="{BAR_RGBA}"/></body>'
        for k in range(n))


def world_xml(robot_xml, n=1, extra_before_floor=""):
    """The robot's model with n real hurdles added."""
    xml = robot_xml.replace('<geom name="floor"', (extra_before_floor + "\n    " if extra_before_floor else "")
                            + stands_xml(n) + '\n    <geom name="floor"', 1)
    assert xml.count("</worldbody>") == 1
    return xml.replace("</worldbody>", "  " + bars_xml(n) + "\n  </worldbody>")


def rest_pose(at, yaw, h):
    """Stand position and quaternion, and the bar's resting position (centre) and quaternion."""
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return np.array([at[0], at[1], h]), q, np.array([at[0], at[1], h - GEO["bar_r"]]), q


def constants():
    """What a page needs to draw the hurdles exactly as the physics has them."""
    return dict(GEO, stand_rgba=STAND_RGBA, wing_rgba=WING_RGBA, bar_rgba=BAR_RGBA)
