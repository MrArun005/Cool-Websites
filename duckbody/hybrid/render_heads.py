"""Side-by-side picture of the head sizes (README "Lighter head"): each body standing, side view and 3/4 view.

    MUJOCO_GL=osmesa /home/user/cool-websites/.venv/bin/python duckbody/hybrid/render_heads.py [OUT.png]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
HEADS = [("_noantenna", "owner head x1 (189 g)"), ("_k85", "x0.85 (159 g)"), ("_k75", "x0.75 (142 g)"),
         ("_small", "x2/3 (old variant, 189 g)")]
W, H = 420, 460


def shots(suffix):
    m = mujoco.MjModel.from_xml_path(str(HERE / f"scene_hybrid_allcollisions{suffix}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.subtree_com[1] + np.array([0.0, 0.0, 0.035])
    cam.distance = 0.56
    cam.elevation = -8
    out = []
    for az in (90, 160):                 # side (looking along +y), front three-quarter
        cam.azimuth = az
        r.update_scene(d, cam, opt)
        out.append(r.render()[:, :, ::-1].copy())
    r.close()
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "light" / "heads_side_by_side.png"
    cols = []
    for suffix, label in HEADS:
        a, b = shots(suffix)
        col = np.vstack([a, b])
        cv2.rectangle(col, (0, 0), (W, 34), (255, 255, 255), -1)
        cv2.putText(col, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cols.append(col)
        cols.append(np.full((2 * H, 4, 3), 255, np.uint8))
    img = np.hstack(cols[:-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(out, img.shape)


if __name__ == "__main__":
    main()
