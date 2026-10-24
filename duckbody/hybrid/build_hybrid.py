"""Build the hybrid duck: Pollen Robotics' Microduck body with the owner's duck head, bill and mouth.

    .venv/bin/python duckbody/hybrid/build_hybrid.py

Everything below the head is Microduck's MJCF, unchanged: trunk, legs, neck, joints, joint ranges,
armature/damping/friction classes, IMU site, sensors, default pose, collision sets. Only the head
link (Microduck body `jaw_soft`, the link after the `head_roll` joint) is edited:

  * visual: Microduck's head shell, face, beak and electronics are hidden (moved to geom group 4, so
    they are still there to check the packaging) and the owner's head is drawn instead: the v3 head
    shell + bottom sheet + antennas + antenna holders (Open Duck Mini v2 parts at the owner's 0.7 "mid"
    scale, the same parts that make up v3/duck_mesh_v3_head60_jaw22.glb) and the owner's bill, an
    upper bill fixed to the head and a lower jaw on a hinge (`mouth`, 0-35 deg, small position servo).
  * placement: the owner's head shell is centred on Microduck's head shell (same orientation; centre
    of the two shells' bounding boxes made to coincide in the head frame), so it sits where Microduck's
    head sits instead of 6 cm further back where the Open Duck neck carries it.
  * collision: Microduck's head collision meshes are replaced by the convex hulls of the owner's head
    shell, bottom sheet, antenna holders and antennas, plus the two bill ellipsoids.
  * mass: two variants
      matched   (default)  head + jaw total mass, centre of mass and inertia equal to Microduck's head
                           link (the jaw's share is taken out of the head link, so the sum is exact
                           with the mouth closed).
      ownermass            the owner's head and jaw mass/CoM/inertia from model/duck_mid.xml
                           (estimated 0.158 kg), moved with the shell.

Writes, next to this file:
  hybrid_allcollisions[_ownermass].xml, hybrid_groundcontact[_ownermass].xml   robot
  scene_hybrid_allcollisions[_ownermass].xml, scene_hybrid[_ownermass].xml     floor + robot
  head_report.json                                                             mass / CoM / size numbers
Sources: microduck/ (copied from pollen-robotics/microduck_rl, see microduck/SOURCE.md) and
owner_head/ (copied from duckbody/model/upstream/assets + duckbody/model/duck_mid.xml).
"""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MD = HERE / "microduck"                     # copy of microduck_rl/src/mjlab_microduck/robot/microduck
OWNER_MODEL = HERE.parent / "model" / "duck_mid.xml"
OWNER_MESH_DIR = "owner_head"
OWNER_SCALE = 0.7
OWNER_HEAD_BODY = "head_assembly"
MD_HEAD_BODY = "jaw_soft"
MOUTH = dict(kp=2.0, force=0.2, damping=0.02, armature=0.0005, range=0.61)  # same as build_mid.JAW


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def parse(path):
    return ET.parse(path, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def home(m, key):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    return d


def mesh_points(m, d, g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def bbox_in(m, d, body, meshes, group_not=3):
    """Bounding box, in `body`'s frame, of the visual mesh geoms of `body` whose mesh is in `meshes`."""
    b = m.body(body).id
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    pts = [(mesh_points(m, d, g) - p) @ R for g in range(m.ngeom)
           if m.geom_bodyid[g] == b and m.geom_group[g] != group_not and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
           and m.mesh(m.geom_dataid[g]).name in meshes]
    P = np.vstack(pts)
    return P.min(0), P.max(0)


def subtree_inertia(m, d, root):
    """Total mass, CoM and inertia about the CoM (world frame) of `root`'s subtree."""
    ids = [i for i in range(m.nbody) if _under(m, i, root)]
    M = sum(m.body_mass[i] for i in ids)
    c = sum(m.body_mass[i] * d.xipos[i] for i in ids) / M
    I = np.zeros((3, 3))
    for i in ids:
        R = d.ximat[i].reshape(3, 3)
        r = d.xipos[i] - c
        I += R @ np.diag(m.body_inertia[i]) @ R.T + m.body_mass[i] * (r @ r * np.eye(3) - np.outer(r, r))
    return M, c, I


def _under(m, i, root):
    while i > 0:
        if i == root:
            return True
        i = m.body_parentid[i]
    return False


def world_box(R, lo, hi):
    a, b = R @ lo, R @ hi
    return dict(lo=(1000 * np.minimum(a, b)).round(1).tolist(), hi=(1000 * np.maximum(a, b)).round(1).tolist())


def inertial_el(mass, pos, I):
    return {"pos": fmt(pos), "mass": f"{mass:.6g}",
            "fullinertia": fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])}


def build(md_robot: str, out_name: str, variant: str, report: dict, collision: str = "owner", k: float = 1.0,
          suffix: str | None = None):
    """collision: 'owner' (owner's head hulls incl. antennas), 'owner_noantenna' (antennas visual only),
    'microduck' (keep Microduck's own head collision meshes: a pure visual swap). k: owner head scale
    factor on top of the 0.7 mid scale, about the shell centre (1.0 = the owner's mid head)."""
    # ---- reference models -------------------------------------------------------------------
    md_scene = {"robot_allcollisions.xml": "scene_allcollisions.xml", "robot_groundcontact.xml": "scene.xml"}[md_robot]
    mm = mujoco.MjModel.from_xml_path(str(MD / md_scene))
    md = home(mm, mm.key("STAND").id)
    om = mujoco.MjModel.from_xml_path(str(OWNER_MODEL))
    od = home(om, 0)

    hb_m, hb_o = mm.body(MD_HEAD_BODY).id, om.body(OWNER_HEAD_BODY).id
    Rm, pm = md.xmat[hb_m].reshape(3, 3), md.xpos[hb_m]
    Ro, po = od.xmat[hb_o].reshape(3, 3), od.xpos[hb_o]
    # Owner head frame -> Microduck head frame: same world orientation at the home pose, then shift so the
    # two head shells' bounding-box centres coincide.
    R_om = Rm.T @ Ro                                         # owner-head-frame vectors into md head frame
    o_lo, o_hi = bbox_in(om, od, OWNER_HEAD_BODY, {"head", "head_bot_sheet"})
    m_lo, m_hi = bbox_in(mm, md, MD_HEAD_BODY, {"top_head_shell", "bottom_head_shell"})
    shift = (m_lo + m_hi) / 2 - R_om @ ((o_lo + o_hi) / 2)

    def to_md(pos, quat=(1, 0, 0, 0)):
        oc = (o_lo + o_hi) / 2
        return R_om @ (oc + k * (np.asarray(pos, float) - oc)) + shift, mat2quat(R_om @ quat2mat(quat))

    def ksize(v):
        return fmt(k * np.array([float(x) for x in v.split()]))

    # ---- Microduck robot XML ----------------------------------------------------------------
    tree = parse(MD / md_robot)
    root = tree.getroot()
    root.set("model", "microduck_hybrid")
    root.insert(0, ET.Comment(
        f" HYBRID: generated by duckbody/hybrid/build_hybrid.py from microduck/{md_robot} (pollen-robotics/"
        f"microduck_rl, Apache-2.0; meshes CC BY-NC-SA 4.0, see microduck/SOURCE.md). Only the head link "
        f"'{MD_HEAD_BODY}' is changed: owner's duck head + bill + mouth, mass variant '{variant}'. "))
    comp = root.find("compiler")
    comp.set("meshdir", ".")
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        f = mesh.get("file")
        mesh.set("name", mesh.get("name") or Path(f).stem)
        mesh.set("file", f"microduck/assets/{f}")
    for name in ("head", "head_bot_sheet", "antenna", "left_antenna_holder", "right_antenna_holder"):
        ET.SubElement(asset, "mesh", {"name": f"owner_{name}", "file": f"{OWNER_MESH_DIR}/{name}.stl",
                                      "scale": fmt([OWNER_SCALE * k] * 3)})
    for mat in range(om.nmat):
        nm = om.mat(mat).name
        if nm in ("head_material", "head_bot_sheet_material", "antenna_material",
                  "left_antenna_holder_material", "right_antenna_holder_material"):
            ET.SubElement(asset, "material", {"name": f"owner_{nm}", "rgba": fmt(om.mat_rgba[mat])})
    ET.SubElement(asset, "material", {"name": "owner_bill_material", "rgba": "1 0.62 0.1 1"})

    head = next(b for b in root.iter("body") if b.get("name") == MD_HEAD_BODY)
    md_inertial = head.find("inertial")
    # Hide Microduck's head parts (kept for packaging checks), drop their collision geoms.
    for g in list(head.findall("geom")):
        if g.get("class") == "visual":
            g.set("group", "4")
            g.set("name", "md_" + g.get("mesh") + f"_{list(head).index(g)}")
        elif collision != "microduck":
            head.remove(g)

    # Owner's head geoms, from duck_mid.xml's head_assembly (visual + hull collision).
    otree = parse(OWNER_MODEL)
    ohead = next(b for b in otree.getroot().iter("body") if b.get("name") == OWNER_HEAD_BODY)
    insert_at = list(head).index(md_inertial) + 1
    new = []
    for g in ohead.findall("geom"):
        mesh = g.get("mesh")
        if g.get("class") == "visual" and mesh:
            p, q = to_md([float(x) for x in g.get("pos").split()], [float(x) for x in g.get("quat").split()])
            new.append(ET.Element("geom", {"type": "mesh", "class": "visual", "name": f"owner_{mesh}_vis_{len(new)}",
                                           "pos": fmt(p), "quat": fmt(q), "mesh": f"owner_{mesh}",
                                           "material": f"owner_{g.get('material')}"}))
            if collision == "owner" or (collision == "owner_noantenna" and "antenna" not in mesh):
                new.append(ET.Element("geom", {"type": "mesh", "class": "collision", "name": f"owner_{mesh}_col_{len(new)}",
                                               "pos": fmt(p), "quat": fmt(q), "mesh": f"owner_{mesh}"}))
        elif g.get("name") == "upper_bill":
            p, q = to_md([float(x) for x in g.get("pos").split()], [float(x) for x in g.get("quat").split()])
            for cls, nm in (("visual_bill", "owner_upper_bill"), ("collision", "owner_upper_bill_col")):
                if cls == "collision" and collision == "microduck":
                    continue
                el = ET.Element("geom", {"name": nm, "type": "ellipsoid", "size": ksize(g.get("size")), "pos": fmt(p),
                                         "quat": fmt(q)})
                if cls == "collision":
                    el.set("class", "collision")
                else:
                    el.attrib.update({"contype": "0", "conaffinity": "0", "group": "2", "material": "owner_bill_material"})
                new.append(el)
    for i, el in enumerate(new):
        head.insert(insert_at + i, el)

    # Mouth: the owner's lower jaw on a hinge (Microduck's head carries a mouth servo too).
    ojaw = ohead.find("body[@name='jaw']")
    jp, jq = to_md([float(x) for x in ojaw.get("pos").split()], [float(x) for x in ojaw.get("quat").split()])
    jaw = ET.SubElement(head, "body", {"name": "owner_jaw", "pos": fmt(jp), "quat": fmt(jq)})
    oji = ojaw.find("inertial")
    ET.SubElement(jaw, "inertial", dict(oji.attrib, pos=ksize(oji.get("pos"))))
    ET.SubElement(jaw, "joint", {"name": "mouth", "type": "hinge", "axis": "0 1 0", "range": f"0 {MOUTH['range']}",
                                 "damping": str(MOUTH["damping"]), "armature": str(MOUTH["armature"]),
                                 "frictionloss": "0"})
    olb = ojaw.find("geom[@name='lower_bill']")
    ET.SubElement(jaw, "geom", {"name": "owner_lower_bill", "type": "ellipsoid", "size": ksize(olb.get("size")),
                                "pos": ksize(olb.get("pos")), "contype": "0", "conaffinity": "0", "group": "2",
                                "material": "owner_bill_material"})
    if collision != "microduck":
        ET.SubElement(jaw, "geom", {"name": "owner_lower_bill_col", "class": "collision", "type": "ellipsoid",
                                    "size": ksize(olb.get("size")), "pos": ksize(olb.get("pos"))})
    act = root.find("actuator")
    ET.SubElement(act, "position", {"name": "mouth", "joint": "mouth", "kp": str(MOUTH["kp"]),
                                    "forcerange": f"-{MOUTH['force']} {MOUTH['force']}", "ctrlrange": f"0 {MOUTH['range']}"})

    # ---- mass -------------------------------------------------------------------------------
    # Microduck head link (reference): mass, CoM, inertia about CoM in its own frame.
    M0 = mm.body_mass[hb_m]
    c0 = mm.body_ipos[hb_m].copy()
    Rq = quat2mat(mm.body_iquat[hb_m])
    I0 = Rq @ np.diag(mm.body_inertia[hb_m]) @ Rq.T
    # Jaw (owner's) in the head frame, mouth closed.
    mj = float(oji.get("mass"))
    cj = jp + quat2mat(jq) @ (k * np.array([float(x) for x in oji.get("pos").split()]))
    Ij = quat2mat(jq) @ np.diag([float(x) for x in oji.get("diaginertia").split()]) @ quat2mat(jq).T
    if variant == "matched":
        Mh = M0 - mj
        ch = (M0 * c0 - mj * cj) / Mh

        def shift_I(I, m_, r):
            return I + m_ * (r @ r * np.eye(3) - np.outer(r, r))
        # I0 about c0 = Ih about ch shifted to c0 + Ij about cj shifted to c0
        Ih = I0 - shift_I(Ij, mj, cj - c0) - Mh * ((ch - c0) @ (ch - c0) * np.eye(3) - np.outer(ch - c0, ch - c0))
    else:
        oi = om.body_ipos[hb_o]
        Mh = om.body_mass[hb_o]
        ch = R_om @ oi + shift
        Ro_i = R_om @ quat2mat(om.body_iquat[hb_o])
        Ih = Ro_i @ np.diag(om.body_inertia[hb_o]) @ Ro_i.T
    for key, val in inertial_el(Mh, ch, Ih).items():
        md_inertial.set(key, val)
    md_inertial.attrib.pop("quat", None)
    md_inertial.attrib.pop("diaginertia", None)

    # Keep the jaw from touching Microduck's neck bracket (exclude like Microduck does for its own head).
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": "neck_pitch", "body2": "owner_jaw"})
    if md_robot == "robot_groundcontact.xml" and contact.find("exclude[@body1='neck_pitch'][@body2='jaw_soft']") is None:
        # Microduck excludes this pair only in allcollisions because its groundcontact head meshes never reach
        # the bracket; the owner's head hull does, so exclude it here too (a permanent phantom contact otherwise).
        ET.SubElement(contact, "exclude", {"body1": "neck_pitch", "body2": "jaw_soft"})

    if suffix is None:
        suffix = "" if variant == "matched" else "_" + variant
    robot_file = HERE / f"{out_name}{suffix}.xml"
    ET.indent(tree, "  ")
    tree.write(robot_file, encoding="unicode")

    # ---- scene ------------------------------------------------------------------------------
    stree = parse(MD / md_scene)
    sroot = stree.getroot()
    sroot.set("model", "scene_hybrid")
    sroot.find("include").set("file", robot_file.name)
    # Keyframes: insert the mouth (closed) after head_roll, and its ctrl at the end.
    mouth_after = 7 + 9          # qpos index right after head_roll (freejoint 7 + 9 policy joints before it)
    for key in sroot.iter("key"):
        q = key.get("qpos").split()
        q.insert(mouth_after, "0")
        key.set("qpos", " ".join(q))
        if key.get("ctrl"):
            key.set("ctrl", " ".join(key.get("ctrl").split() + ["0"]))
    scene_file = HERE / (("scene_hybrid_allcollisions" if "allcollisions" in md_robot else "scene_hybrid") + suffix + ".xml")
    ET.indent(stree, "  ")
    stree.write(scene_file, encoding="unicode")

    # ---- check + report ---------------------------------------------------------------------
    hm = mujoco.MjModel.from_xml_path(str(scene_file))
    hd = home(hm, hm.key("STAND").id)
    assert hm.joint("mouth").qposadr[0] == mouth_after
    Mm, cm_, Im = subtree_inertia(mm, md, hb_m)
    Mhh, ch_, Ih_ = subtree_inertia(hm, hd, hm.body(MD_HEAD_BODY).id)
    Rh = hd.xmat[hm.body(MD_HEAD_BODY).id].reshape(3, 3)
    ph = hd.xpos[hm.body(MD_HEAD_BODY).id]
    hl_lo, hl_hi = bbox_in(hm, hd, MD_HEAD_BODY, {"owner_head", "owner_head_bot_sheet"}, group_not=3)
    ant_lo, ant_hi = bbox_in(hm, hd, MD_HEAD_BODY, {"owner_head", "owner_head_bot_sheet", "owner_antenna",
                                                    "owner_left_antenna_holder", "owner_right_antenna_holder"}, group_not=3)
    # Microduck's internals (every hidden md_ geom except its outer shells/face) vs the owner's shell box.
    shells = {"top_head_shell", "bottom_head_shell", "face_part", "jaw", "jaw_soft", "soft_mouth_top", "noenoeil"}
    internals = {hm.mesh(hm.geom_dataid[g]).name for g in range(hm.ngeom)
                 if hm.geom_bodyid[g] == hm.body(MD_HEAD_BODY).id and hm.geom_group[g] == 4} - shells
    in_lo, in_hi = bbox_in(hm, hd, MD_HEAD_BODY, internals, group_not=3)
    to_world_dirs = lambda v: Rh @ v                                  # noqa: E731
    # Contacts at the STAND keyframe (should be none inside the robot).
    self_contacts = [(hm.body(hm.geom_bodyid[c.geom1]).name, hm.body(hm.geom_bodyid[c.geom2]).name)
                     for c in hd.contact[:hd.ncon]]
    report[f"{robot_file.name}"] = dict(
        variant=variant, md_robot=md_robot, collision=collision, owner_head_scale=round(OWNER_SCALE * k, 4),
        total_mass_kg=dict(microduck=round(float(mm.body_subtreemass[1]), 4), hybrid=round(float(hm.body_subtreemass[1]), 4)),
        head_mass_kg=dict(microduck=round(float(Mm), 4), hybrid=round(float(Mhh), 4), diff_g=round(1000 * float(Mhh - Mm), 1)),
        head_com_world_STAND_m=dict(microduck=cm_.round(4).tolist(), hybrid=ch_.round(4).tolist(),
                                    diff_mm=(1000 * (ch_ - cm_)).round(1).tolist(),
                                    dist_mm=round(1000 * float(np.linalg.norm(ch_ - cm_)), 1)),
        head_inertia_diag_about_com=dict(microduck=np.diag(Im).round(7).tolist(), hybrid=np.diag(Ih_).round(7).tolist()),
        whole_robot_com_world_STAND_m=dict(
            microduck=md.subtree_com[1].round(4).tolist(), hybrid=hd.subtree_com[1].round(4).tolist(),
            diff_mm=(1000 * (hd.subtree_com[1] - md.subtree_com[1])).round(1).tolist()),
        owner_shell_offset_in_head_frame_mm=(1000 * shift).round(1).tolist(),
        owner_shell_size_world_mm=(1000 * np.abs(to_world_dirs(hl_hi - hl_lo))).round(1).tolist(),
        owner_head_with_antennas_size_world_mm=(1000 * np.abs(to_world_dirs(ant_hi - ant_lo))).round(1).tolist(),
        microduck_shell_size_world_mm=(1000 * np.abs(to_world_dirs(m_hi - m_lo))).round(1).tolist(),
        microduck_internals=sorted(internals),
        microduck_internals_box_rel_head_roll_world_mm=world_box(Rh, in_lo, in_hi),
        owner_shell_box_rel_head_roll_world_mm=world_box(Rh, hl_lo, hl_hi),
        contacts_at_STAND=self_contacts,
        head_roll_joint_world_STAND_m=ph.round(4).tolist(),
    )
    return robot_file, scene_file


# name -> how it differs from the main hybrid (scene files: scene_hybrid[_allcollisions]<suffix>.xml)
VARIANTS = {
    "hybrid": dict(),                                            # owner head at mid scale, matched mass, hull collision
    "hybrid_ownermass": dict(mass="ownermass"),                  # owner's estimated head mass/CoM/inertia
    "hybrid_noantenna": dict(collision="owner_noantenna"),       # antennas don't collide (removed / flexible)
    "hybrid_ownermass_noantenna": dict(mass="ownermass", collision="owner_noantenna"),
    "hybrid_mdcollision": dict(collision="microduck"),           # owner head drawn, Microduck's head collides (control)
    "hybrid_small": dict(k=2 / 3, collision="owner_noantenna"),  # owner head shrunk to Microduck's head width, no antennas
}


def main():
    report = {}
    for variant, kw in VARIANTS.items():
        for md_robot, out in (("robot_allcollisions.xml", "hybrid_allcollisions"), ("robot_groundcontact.xml", "hybrid_groundcontact")):
            r, s = build(md_robot, out, kw.get("mass", "matched"), report, kw.get("collision", "owner"), kw.get("k", 1.0),
                         "" if variant == "hybrid" else variant[len("hybrid"):])
            print("wrote", r.name, s.name)
    (HERE / "head_report.json").write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(k, v["head_mass_kg"], v["head_com_world_STAND_m"]["diff_mm"], v["contacts_at_STAND"])


if __name__ == "__main__":
    main()
