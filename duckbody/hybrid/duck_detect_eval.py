"""Score duck_detect.py (Pollen's Microduck duck detector, and the team-colour fallback) on rendered head-camera
frames of hybrid ducks, against the simulator's ground truth.

    PY=/home/user/cool-websites/.venv/bin/python
    MUJOCO_GL=osmesa $PY duckbody/hybrid/duck_detect_eval.py --scenes 120      # -> results_duck_detect_hybrid.json
    MUJOCO_GL=osmesa $PY duckbody/hybrid/duck_detect_eval.py --sheet DIR       # + contact sheets of frames + boxes

Scenes (seeded): the observing duck (itself untinted or in a team colour: its own legs can enter the picture) stands at the origin looking along +x (head home, or tilted down / sideways a
little), 1-4 other hybrid ducks (Microduck body, the owner's head; standing pose, random yaw) at 0.3-2.5 m within
the camera's field of view, >= 0.25 m apart; 15 % of scenes have no other duck (false positives). Floor: match.py's
green turf with a white line, or the hybrid scene's blue checker. Colouring per duck: untinted (the hybrid as built)
or a team colour on the light shell parts (match.py's rule). Full meshes (not match.py's proxy boxes).
Camera: the head 'eye' (49 deg vertical FOV), rendered natively at 640x480, 320x240 and 160x120 (the games' size).
Ground truth: segmentation render -> each duck's visible pixels and box; a duck counts as visible with >= 0.05 % of
the image (~150 px at 640x480) and a box >= 6 % of the image height. Detection = IoU >= 0.3 with a visible duck's
box, or >= 70 % of the detection's box inside it (a box on part of a duck: the colour detector boxes only the
team-coloured parts), one detection per duck; every other box is a false positive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import duck_detect as DD  # noqa: E402

ROBOT_XML = HERE / "hybrid_allcollisions_noantenna.xml"
EYE_POS = (0.020, 0.0, -0.096)          # follow.py's head camera
EYE_QUAT = (0.7071068, 0.0, 0.0, -0.7071068)
FOVY = 49.0
TEAM_RGB = {"blue": (0.13, 0.38, 0.95), "yellow": (1.0, 0.82, 0.05)}     # match.py
STAND = {"left_hip_roll": -0.0873, "left_hip_pitch": -0.4579, "left_knee": -0.0049, "left_ankle": 0.4530,
         "neck_pitch": 0.3491, "head_pitch": 0.3491,
         "right_hip_roll": 0.0873, "right_hip_pitch": 0.4579, "right_knee": 0.0049, "right_ankle": -0.4530}
STAND_Z = 0.12
RES = ((640, 480), (320, 240), (160, 120))
MAX_OTHERS = 4
FLOOR_Z_TRY = (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)


def tint_part(rgb, name=""):
    """match.py's rule: light, unsaturated shell / bracket parts take the team colour, not the owner's head / bill."""
    r, g, b = (float(v) for v in rgb[:3])
    if name.startswith("owner") or "bill" in name:
        return False
    return min(r, g, b) > 0.55 and max(r, g, b) - min(r, g, b) < 0.15


CHECKER_GROUP = 5


def build():
    """World with the observer ('o_', with the eye camera) and MAX_OTHERS other ducks ('d0_'...), all kinematic
    (rendering only). Two floors: match.py's turf + a white line (group 0) and the hybrid scene's blue checker (group
    5, under the turf); a scene shows one through the renderer's geom groups. Team tints: set_tints()."""
    import mujoco
    w = mujoco.MjSpec()
    w.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    w.visual.headlight.ambient = [0.35, 0.35, 0.35]
    w.visual.headlight.specular = [0, 0, 0]
    w.visual.global_.offwidth, w.visual.global_.offheight = 640, 480
    lt = w.worldbody.add_light()
    lt.pos, lt.dir, lt.type = [0, 0, 3.5], [0, 0, -1], mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    g = w.worldbody.add_geom()
    g.name, g.type, g.size = "turf", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    g.rgba = [0x2e / 255, 0x7d / 255, 0x32 / 255, 1]
    ln = w.worldbody.add_geom()          # a white pitch line across the view
    ln.name, ln.type, ln.size, ln.pos = "line", mujoco.mjtGeom.mjGEOM_BOX, [0.0125, 3.0, 0.0005], [1.2, 0, 0.0005]
    ln.rgba = [0.96, 0.96, 0.96, 1]
    t = w.add_texture()
    t.name, t.type, t.builtin = "grid", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    t.rgb1, t.rgb2, t.mark, t.markrgb = [0.2, 0.3, 0.4], [0.1, 0.2, 0.3], mujoco.mjtMark.mjMARK_EDGE, [0.8, 0.8, 0.8]
    t.width = t.height = 300
    mt = w.add_material()
    mt.name, mt.texuniform, mt.texrepeat = "grid", True, [5, 5]
    mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    g = w.worldbody.add_geom()
    g.name, g.type, g.size, g.material, g.group = "checker", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05], "grid", CHECKER_GROUP
    g.pos = [0, 0, -0.0002]
    names = ["o"] + [f"d{i}" for i in range(MAX_OTHERS)]
    for i, name in enumerate(names):
        child = mujoco.MjSpec.from_file(str(ROBOT_XML))
        if i == 0:
            c = child.body("jaw_soft").add_camera()
            c.name, c.pos, c.quat, c.fovy = "eye", list(EYE_POS), list(EYE_QUAT), FOVY
        fr = w.worldbody.add_frame()
        fr.attach_body(child.body("trunk_base"), name + "_", "")
    m = w.compile()
    return m, names


_BASE = {}


def set_tints(m, names, tints):
    """match.py's team colours on each duck's light shell parts (None: as built)."""
    import mujoco
    if "rgba" not in _BASE:
        _BASE["rgba"] = m.mat_rgba.copy()
    m.mat_rgba[:] = _BASE["rgba"]
    for i, name in enumerate(names):
        tint = TEAM_RGB.get(tints[i])
        if tint is None:
            continue
        for k in range(m.nmat):
            full = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k)
            if full.startswith(name + "_") and tint_part(m.mat_rgba[k][:3], full[len(name) + 1:]):
                m.mat_rgba[k][:3] = tint


def scene(seed):
    r = np.random.default_rng(seed)
    n = 0 if r.random() < 0.15 else int(r.integers(1, MAX_OTHERS + 1))
    floor = "turf" if r.random() < 0.7 else "checker"
    tints = [[None, "blue", "yellow"][int(r.integers(3))] for _ in range(MAX_OTHERS + 1)]   # [0]: the observer
    head = dict(head_yaw=float(r.uniform(-0.3, 0.3)), head_pitch=0.3491 + float(r.choice([0.0, 0.0, 0.3, -0.15])))
    ducks, tries = [], 0
    while len(ducks) < n and tries < 200:
        tries += 1
        d = float(r.uniform(0.3, 2.5))
        b = float(r.uniform(-0.5, 0.5)) + head["head_yaw"]
        p = np.array([d * math.cos(b), d * math.sin(b)])
        if all(np.hypot(*(p - q["xy"])) > 0.25 for q in ducks):
            ducks.append(dict(xy=p, yaw=float(r.uniform(-math.pi, math.pi)), team=tints[len(ducks) + 1]))
    return dict(seed=seed, floor=floor, tints=tints, head=head, ducks=ducks)


def pose(m, d, names, sc):
    import mujoco
    d.qpos[:] = m.qpos0
    for i, name in enumerate(names):
        a = m.jnt_qposadr[m.joint(f"{name}_trunk_base_freejoint").id]
        if i == 0:
            xy, yaw = (0.0, 0.0), 0.0
        elif i - 1 < len(sc["ducks"]):
            xy, yaw = sc["ducks"][i - 1]["xy"], sc["ducks"][i - 1]["yaw"]
        else:
            xy, yaw = (50.0 + 3 * i, 50.0), 0.0       # parked far away, out of view
        d.qpos[a:a + 7] = [xy[0], xy[1], STAND_Z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for jn, v in STAND.items():
            d.qpos[m.jnt_qposadr[m.joint(f"{name}_{jn}").id]] = v
        if i == 0:
            for jn, v in sc["head"].items():
                d.qpos[m.jnt_qposadr[m.joint(f"o_{jn}").id]] = v
    mujoco.mj_forward(m, d)
    # stand on the floor: lowest sole vertex at z = 0 (approximately: shift each trunk by its lowest geom aabb)
    for i, name in enumerate(names):
        root = m.body(f"{name}_trunk_base").id
        gs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        zmin = min(d.geom_xpos[g][2] - _half_z(m, d, g) for g in gs)
        a = m.jnt_qposadr[m.joint(f"{name}_trunk_base_freejoint").id]
        d.qpos[a + 2] -= zmin
    mujoco.mj_forward(m, d)


def _half_z(m, d, g):
    R = d.geom_xmat[g].reshape(3, 3)
    return float(np.abs(R[2]) @ m.geom_aabb[g][3:])


def ground_truth(m, d, names, seg, n_others):
    """Per other duck: visible pixel count and box from the segmentation render."""
    import mujoco
    gid = seg[..., 0]
    typ = seg[..., 1]
    owner = np.full(m.ngeom + 1, -1, int)
    for i, name in enumerate(names):
        root = m.body(f"{name}_trunk_base").id
        for g in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[g]] == root:
                mat = m.geom_matid[g]
                wire = mat >= 0 and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat).endswith("owner_antenna_material")
                owner[g] = -1 if wire else i     # the antenna wires (1 px lines) do not make the box taller
    lab = np.where((typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0), owner[np.clip(gid, 0, m.ngeom)], -1)
    out = []
    h, w = seg.shape[:2]
    for i in range(1, n_others + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) == 0:
            out.append(dict(px=0, box=None, visible=False))
            continue
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        vis = len(xs) >= 5e-4 * w * h and (box[3] - box[1]) >= 0.06 * h
        out.append(dict(px=int(len(xs)), box=box, visible=bool(vis), frac=len(xs) / (w * h)))
    return out


def match(dets, gts):
    """Greedy by score: a detection takes the unmatched visible duck with the best IoU >= 0.3."""
    taken, tp, fp = set(), [], 0
    for det in sorted(dets, key=lambda x: -x["score"]):
        best, bi = 0.3, None
        for i, g in enumerate(gts):
            if g["box"] is None or i in taken:
                continue
            v = DD.iou(det["box"], g["box"])
            if v < 0.3 and _inside(det["box"], g["box"]) >= 0.7:
                v = 0.3
            if v >= best:
                best, bi = v, i
        if bi is None:
            fp += 1
        else:
            taken.add(bi)
            tp.append((bi, det))
    return tp, fp


def _inside(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    area = max(a[2] - a[0], 1e-9) * max(a[3] - a[1], 1e-9)
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0) / area


def run(n_scenes, sheet=None, thresholds=(0.1, 0.25, 0.5)):
    import mujoco
    det_model = DD.PollenDuckDetector(threads=1)
    rows = []
    t0 = time.time()
    sheets = {}
    m, names = build()
    d = mujoco.MjData(m)
    cid = m.camera("o_eye").id
    rens = {}
    for (W, H) in RES:            # one renderer per size for the whole run (OSMesa contexts do not like churn)
        rens[(W, H)] = (mujoco.Renderer(m, H, W), mujoco.Renderer(m, H, W))
        rens[(W, H)][1].enable_segmentation_rendering()
    for s in range(n_scenes):
        sc = scene(s)
        set_tints(m, names, sc["tints"])
        pose(m, d, names, sc)
        opt = mujoco.MjvOption()
        opt.geomgroup[CHECKER_GROUP] = sc["floor"] == "checker"
        opt.geomgroup[0] = sc["floor"] != "checker"
        for (W, H) in RES:
            ren, sren = rens[(W, H)]
            ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            ren.update_scene(d, camera="o_eye", scene_option=opt)
            img = ren.render().copy()
            sren.update_scene(d, camera="o_eye", scene_option=opt)
            seg = sren.render().copy()
            gts = ground_truth(m, d, names, seg, len(sc["ducks"]))
            for i, g in enumerate(gts):
                du = sc["ducks"][i]
                g.update(dist=round(float(np.hypot(*(du["xy"] - d.cam_xpos[cid][:2]))), 3), team=du["team"])
            t1 = time.time()
            raw, fit = det_model.raw(img)
            t_model = time.time() - t1
            model_dets = {th: [dict(box=b, score=sc_, team=None) for sc_, b in DD.decode(raw, fit, th)] for th in thresholds}
            t1 = time.time()
            col = DD.detect_ducks(img, d.cam_xpos[cid], d.cam_xmat[cid], FOVY, method="colour")
            t_col = time.time() - t1
            row = dict(scene=s, res=f"{W}x{H}", floor=sc["floor"], n=len(sc["ducks"]), gt=[
                dict(dist=g["dist"], team=g["team"], visible=g["visible"], px=g["px"]) for g in gts],
                max_model_score=round(float(raw.reshape(5, -1)[4].max()), 4), t_model_ms=round(1000 * t_model, 1),
                t_colour_ms=round(1000 * t_col, 2))
            for th in thresholds:
                tp, fp = match(model_dets[th], gts)
                row[f"model@{th}"] = dict(tp=[i for i, _ in tp], fp=fp)
            tp, fp = match(col, gts)
            floor_err, floor_z = [], {}
            for i, det in tp:
                if det["floor"] is not None and not det["cut"]:
                    du = sc["ducks"][i]
                    floor_err.append(round(float(np.hypot(*(det["floor"] - du["xy"]))), 3))
                    x0, y0, x1, y1 = det["box"]                     # calibration: the plane height that fits best
                    for z in FLOOR_Z_TRY:
                        q = DD.floor_point((x0 + x1) / 2, y1, W, H, d.cam_xpos[cid], d.cam_xmat[cid], FOVY, z)
                        if q is not None:
                            floor_z.setdefault(str(z), []).append(round(float(np.hypot(*(q[:2] - du["xy"]))), 3))
            row["colour"] = dict(tp=[i for i, _ in tp], fp=fp, floor_err=floor_err, floor_z=floor_z,
                                 dists=[round(det["dist"], 3) if det["dist"] else None for _, det in tp])
            rows.append(row)
            if sheet is not None and (W, H) == (640, 480) and s < 48:
                sheets.setdefault("x", []).append(_annotate(img, gts, model_dets[0.25], col))
        if s % 10 == 0:
            print(f"scene {s} / {n_scenes}  {time.time() - t0:.0f} s", flush=True)
    if sheet is not None:
        _write_sheets(sheets.get("x", []), Path(sheet))
    return rows


def _annotate(img, gts, model, col):
    import cv2
    im = img.copy()
    for g in gts:
        if g["box"] is not None:
            x0, y0, x1, y1 = map(int, g["box"])
            cv2.rectangle(im, (x0, y0), (x1, y1), (255, 255, 255) if g["visible"] else (120, 120, 120), 1)
    for k in model:
        x0, y0, x1, y1 = map(int, k["box"])
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(im, f"{k['score']:.2f}", (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    for k in col:
        x0, y0, x1, y1 = map(int, k["box"])
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 1)
    return cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)


def _write_sheets(ims, out):
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(ims), 12):
        part = ims[k:k + 12]
        while len(part) % 4:
            part.append(np.zeros_like(ims[0]))
        rows_ = [np.concatenate(part[i:i + 4], 1) for i in range(0, len(part), 4)]
        cv2.imwrite(str(out / f"sheet_{k // 12}.jpg"), cv2.cvtColor(np.concatenate(rows_, 0), cv2.COLOR_RGB2BGR))


def summarize(rows, thresholds=(0.1, 0.25, 0.5)):
    out = {}
    bins = ((0.0, 0.75), (0.75, 1.5), (1.5, 3.0))
    for res in [f"{w}x{h}" for w, h in RES]:
        rs = [r for r in rows if r["res"] == res]
        vis = [(r, i, g) for r in rs for i, g in enumerate(r["gt"]) if g["visible"]]
        empty = [r for r in rs if r["n"] == 0]
        o = dict(frames=len(rs), visible_ducks=len(vis), empty_frames=len(empty),
                 max_model_score_median=round(float(np.median([r["max_model_score"] for r in rs])), 4),
                 max_model_score_max=round(float(np.max([r["max_model_score"] for r in rs])), 4),
                 t_model_ms=round(float(np.median([r["t_model_ms"] for r in rs])), 1),
                 t_colour_ms=round(float(np.median([r["t_colour_ms"] for r in rs])), 2))
        for name in [f"model@{t}" for t in thresholds] + ["colour"]:
            det = sum(1 for r, i, g in vis if i in r[name]["tp"])
            fp = sum(r[name]["fp"] for r in rs)
            by = {}
            for lo, hi in bins:
                vv = [(r, i) for r, i, g in vis if lo <= g["dist"] < hi]
                by[f"{lo}-{hi} m"] = f"{sum(1 for r, i in vv if i in r[name]['tp'])}/{len(vv)}"
            teams = {}
            for t in (None, "blue", "yellow"):
                vv = [(r, i) for r, i, g in vis if g["team"] == t]
                teams[str(t or "untinted")] = f"{sum(1 for r, i in vv if i in r[name]['tp'])}/{len(vv)}"
            o[name] = dict(detection_rate=round(det / max(1, len(vis)), 3), detected=det, false_pos=fp,
                           fp_per_frame=round(fp / max(1, len(rs)), 3),
                           fp_in_empty_frames=sum(r[name]["fp"] for r in empty), by_distance=by, by_colour=teams)
            if name == "colour":
                tinted = [(r, i) for r, i, g in vis if g["team"] is not None]
                o[name]["detection_rate_tinted"] = round(
                    sum(1 for r, i in tinted if i in r[name]["tp"]) / max(1, len(tinted)), 3)
                o[name]["floor_err_m_median_by_plane_z"] = {
                    z: round(float(np.median([e for r in rs for e in r["colour"]["floor_z"].get(str(z), [])] or [np.nan])), 3)
                    for z in FLOOR_Z_TRY}
                fe = [e for r in rs for e in r["colour"]["floor_err"]]
                if fe:
                    o[name]["floor_err_m_median"] = round(float(np.median(fe)), 3)
                    o[name]["floor_err_m_p90"] = round(float(np.percentile(fe, 90)), 3)
        out[res] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=120)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out", default=str(HERE / "results_duck_detect_hybrid.json"))
    a = ap.parse_args()
    rows = run(a.scenes, a.sheet)
    summ = summarize(rows)
    print(json.dumps(summ, indent=1))
    Path(a.out).write_text(json.dumps(dict(summary=summ, frames=rows), indent=1))


if __name__ == "__main__":
    main()
