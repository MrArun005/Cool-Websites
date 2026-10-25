"""Find other ducks in the hybrid's head camera.

Two detectors behind one API:

  "model"   Pollen Robotics' Microduck duck detector (github.com/pollen-robotics/duck_detector, weights
            huggingface.co/pollen-robotics/microduck-duck-detector, Apache-2.0): YOLO11n fine-tuned on 200 real
            photos of Microducks taken by a Microduck's camera, one class, 320x320 letterboxed RGB in, [1, 5, 2100]
            out (planar cx, cy, w, h, score, letterbox pixels). The pre/post-processing below is a line-for-line
            port of microduck/duck-detect/src/lib.rs + onnx.rs: nearest-neighbour letterbox padded with 114 grey,
            RGB / 255 NCHW, threshold, greedy NMS at IoU 0.5, boxes mapped back to the frame.
            Weights: duck_detect.onnx (float, 10 MB) - not committed; `fetch_model()` downloads it to
            duckbody/hybrid/.cache/ (or set DUCK_DETECT_ONNX).
  "colour"  the sim-honest fallback the games use (like the ball): HSV threshold on the team colours match.py
            paints on the ducks' light shell parts (blue / yellow), dilate so a duck's parts join, connected
            components -> one box per duck. Cheap (~0.3 ms at 160x120).

Floor position: the ray through the box's bottom-centre (the feet) meets the floor z = 0 (pixel_ground as in
soccer.py; camera pose from the head joints' kinematics, trunk tilt from the IMU on the robot). Unreliable when the
bottom of the box is cut by the image edge (flag `cut`).

    from duck_detect import detect_ducks
    ducks = detect_ducks(img_rgb, cam_pos=d.cam_xpos[cid], cam_mat=d.cam_xmat[cid], fovy=49.0)
    for k in ducks: k["box"], k["score"], k["team"], k["floor"]   # floor: world (x, y), None if not on the floor
    # trunk-frame: duck_detect.to_trunk(k["floor"], d.xpos[trunk], d.xmat[trunk])

Evaluation: duck_detect_eval.py (README section "Duck detector").
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_URL = "https://huggingface.co/pollen-robotics/microduck-duck-detector/resolve/main/duck_detect.onnx"
MODEL_PATH = Path(os.environ.get("DUCK_DETECT_ONNX", HERE / ".cache" / "duck_detect.onnx"))
SIZE, PAD, STRIDE = 320, 114, 5
IOU_LIMIT = 0.5
THRESHOLD = 0.25            # float model; Pollen: tune the threshold against the deployed model
FEET_Z = 0.025               # colour boxes end at the lowest team-coloured part, not the sole: the ray meets this plane

# match.py's team colours (TEAM_RGB), as OpenCV HSV ranges (H 0-180): blue (0.13, 0.38, 0.95) -> H ~111,
# yellow (1.0, 0.82, 0.05) -> H 24 as rendered. The duck's own orange feet (H 10), bill (H 17-22), yellow-orange soles
# (H 20) and the owner's antenna holders (H 22, measured on renders by segmentation) stay just out of the yellow range; the turf (H ~62), the lines (white), the boards (dark blue, V 0.36) and the ball
# (magenta, H ~160) out of both.
TEAM_HSV = {"blue": ((100, 170, 120), (124, 255, 255)), "yellow": ((24, 170, 120), (31, 255, 255))}


# ------------------------------------------------------------------------------------------- Pollen's model
def fetch_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


def letterbox_rgb(frame, size=SIZE):
    """duck-detect letterbox_rgb: scale to fit, nearest neighbour, pad to a square with 114 grey.
    Returns (square uint8 HxWx3, (scale, pad_x, pad_y))."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    fw = min(max(int(round(width * scale)), 1), size)
    fh = min(max(int(round(height * scale)), 1), size)
    pad_x, pad_y = (size - fw) // 2, (size - fh) // 2
    out = np.full((size, size, 3), PAD, np.uint8)
    sy = (np.arange(fh) * height) // fh
    sx = (np.arange(fw) * width) // fw
    out[pad_y:pad_y + fh, pad_x:pad_x + fw] = frame[sy[:, None], sx[None, :], :3]
    return out, (scale, float(pad_x), float(pad_y))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    ov = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    area = lambda r: max(r[2] - r[0], 0.0) * max(r[3] - r[1], 0.0)   # noqa: E731
    u = area(a) + area(b) - ov
    return 0.0 if u <= 0 else ov / u


def decode(raw, fit, threshold=THRESHOLD, iou_limit=IOU_LIMIT):
    """duck-detect decode: the planar [1, 5, N] head -> boxes over threshold, greedy NMS, back to the frame."""
    raw = np.asarray(raw, np.float32).reshape(STRIDE, -1)
    scale, px, py = fit
    keep = np.nonzero(raw[4] >= threshold)[0]
    found = []
    for i in keep:
        cx, cy, w, h, s = raw[:, i]
        found.append((float(s), [(cx - w / 2 - px) / scale, (cy - h / 2 - py) / scale,
                                 (cx + w / 2 - px) / scale, (cy + h / 2 - py) / scale]))
    found.sort(key=lambda f: -f[0])
    kept = []
    for s, b in found:
        if all(iou(b, k[1]) < iou_limit for k in kept):
            kept.append((s, b))
    return kept


class PollenDuckDetector:
    def __init__(self, path=None, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path or fetch_model()), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def raw(self, img_rgb):
        sq, fit = letterbox_rgb(img_rgb)
        x = (sq.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.input: x})[0], fit

    def __call__(self, img_rgb, threshold=THRESHOLD):
        out, fit = self.raw(img_rgb)
        return [dict(box=b, score=s, team=None) for s, b in decode(out, fit, threshold)]


_MODEL = None


# ------------------------------------------------------------------------------------------- colour fallback
def detect_colour(img_rgb, teams=("blue", "yellow"), min_frac=6e-5, join=None):
    """Team-colour blobs -> one box per duck. min_frac: smallest blob as a fraction of the image (6e-5: ~1 px at
    160x120, ~18 px at 640x480); join: dilation in px so a duck's coloured parts make one blob (default ~1/80 of
    the width)."""
    import cv2
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(np.ascontiguousarray(img_rgb[..., :3]), cv2.COLOR_RGB2HSV)
    join = max(1, int(round(w / 80))) if join is None else join
    kernel = np.ones((2 * join + 1, 2 * join + 1), np.uint8)
    out = []
    for team in teams:
        lo, hi = TEAM_HSV[team]
        mask = cv2.inRange(hsv, lo, hi)
        n_px = int(np.count_nonzero(mask))
        if n_px < max(2, min_frac * w * h):
            continue
        grown = cv2.dilate(mask, kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
        for k in range(1, n):
            px = int(np.count_nonzero(mask[lab == k]))
            if px < max(2, min_frac * w * h):
                continue
            ys, xs = np.nonzero((lab == k) & (mask > 0))
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            out.append(dict(box=box, score=min(1.0, px / (4e-4 * w * h)), team=team, pixels=px))
    return _merge(out)


def _merge(dets):
    """One duck's coloured parts can come apart (legs below the shell, an arm of the shell): merge same-team boxes
    that overlap horizontally (>= 50 % of the narrower) and are less than the taller one's height apart vertically."""
    dets = list(dets)
    merged = True
    while merged:
        merged = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                if a["team"] != b["team"]:
                    continue
                A, B = a["box"], b["box"]
                ox = min(A[2], B[2]) - max(A[0], B[0])
                gap = max(A[1], B[1]) - min(A[3], B[3])
                if ox >= 0.5 * min(A[2] - A[0], B[2] - B[0]) and gap < max(A[3] - A[1], B[3] - B[1]):
                    px = a["pixels"] + b["pixels"]
                    dets[i] = dict(a, box=[min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])],
                                   pixels=px, score=max(a["score"], b["score"]))
                    del dets[j]
                    merged = True
                    break
            if merged:
                break
    return dets


# ------------------------------------------------------------------------------------------- geometry
def floor_point(u, v, width, height, cam_pos, cam_mat, fovy, z=0.0):
    """World point at height z seen at pixel (u, v) (MuJoCo camera: -z forward, +y up), or None above the horizon."""
    fy = (height / 2) / math.tan(math.radians(fovy) / 2)
    ray = np.asarray(cam_mat, float).reshape(3, 3) @ np.array([(u - width / 2) / fy, -(v - height / 2) / fy, -1.0])
    c = np.asarray(cam_pos, float)
    if ray[2] > -1e-3:
        return None
    s = (z - c[2]) / ray[2]
    return c + s * ray


def to_trunk(p, trunk_pos, trunk_mat):
    """World point -> trunk yaw frame (x forward, y left), as soccer.pixel_ground."""
    if p is None:
        return None
    R = np.asarray(trunk_mat, float).reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    g = np.asarray(p, float)[:2] - np.asarray(trunk_pos, float)[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * g[0] + s * g[1], -s * g[0] + c * g[1]])


def detect_ducks(img_rgb, cam_pos=None, cam_mat=None, fovy=49.0, method="colour", threshold=THRESHOLD,
                 teams=("blue", "yellow"), feet_z=None):
    """Ducks in a head-camera frame: list of dict(box=[x0, y0, x1, y1] px, score, team ('blue' / 'yellow' / None
    for the model), floor=world (x, y) of the feet or None, dist=camera->feet ground distance (m), cut=box touches
    the bottom edge). method: 'colour' (team colours, what the games use) or 'model' (Pollen's detector)."""
    global _MODEL
    if method == "model":
        if _MODEL is None:
            _MODEL = PollenDuckDetector()
        dets = _MODEL(img_rgb, threshold)
    else:
        dets = detect_colour(img_rgb, teams)
    h, w = img_rgb.shape[:2]
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        d["cut"] = y1 >= h - 1.5
        d["floor"] = d["dist"] = None
        if cam_pos is not None:
            z = (FEET_Z if method == "colour" else 0.0) if feet_z is None else feet_z
            p = floor_point((x0 + x1) / 2, min(y1, h - 0.5), w, h, cam_pos, cam_mat, fovy, z)
            if p is not None:
                d["floor"] = p[:2]
                d["dist"] = float(np.hypot(*(p[:2] - np.asarray(cam_pos)[:2])))
    return dets
