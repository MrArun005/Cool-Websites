"""Side-by-side picture of the head sizes and shapes (README "Lighter head"): each body standing; front, side, 3/4.

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
HEADS = [("_noantenna", "owner head x1"), ("_k85", "owner head x0.85"), ("_k75", "owner head x0.75"),
         ("_small", "x2/3 (old)"), ("_human", "human-like head"), ("_roundduck", "round duck head")]
VIEWS = ((180, "front"), (90, "side"), (150, "3/4"))
W, H = 340, 400


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
    for az, _ in VIEWS:                  # front (camera ahead, looking back), side, front three-quarter
        cam.azimuth = az
        r.update_scene(d, cam, opt)
        out.append(r.render()[:, :, ::-1].copy())
    r.close()
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "light" / "heads_side_by_side.png"
    cols = []
    for suffix, label in HEADS:
        col = np.vstack(shots(suffix))
        cv2.rectangle(col, (0, 0), (W, 34), (255, 255, 255), -1)
        cv2.putText(col, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        cols.append(col)
        cols.append(np.full((len(VIEWS) * H, 4, 3), 255, np.uint8))
    img = np.hstack(cols[:-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(out, img.shape)


if __name__ == "__main__":
    main()
