"""Look at the retargeted clips: a contact sheet of frames per clip, and the joint angle curves.

    MUJOCO_GL=osmesa python dogmotion/preview.py --out /tmp/previews [--clips trot sit]   # from train/

Bolt is posed kinematically (qpos set, no physics) at evenly spaced moments of each clip, seen from the
side. The curves are drawn with PIL (no matplotlib needed): one panel per leg, abd/hip/knee.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "bolt.xml"


def contact_sheet(clip, n=8, size=(320, 240)):
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 1.5, -8, 90
    T = len(clip["joints"])
    tiles = []
    for t in np.linspace(0, T - 1, n).astype(int):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:19] = clip["base_pos"][t], clip["base_quat"][t], clip["joints"][t]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.25]
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((6, 6), f"t={t / 50:.2f}s", fill=(0, 0, 0))
        tiles.append(img)
    r.close()
    sheet = Image.new("RGB", (size[0] * 4, size[1] * ((n + 3) // 4)), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 4) * size[0], (i // 4) * size[1]))
    return sheet


def curves(clip, w=900, h=480):
    q = clip["joints"]
    img = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(img)
    colors = [(200, 60, 40), (40, 110, 200), (40, 160, 70)]
    ph = h // 4
    for leg, name in enumerate(["FL", "FR", "RL", "RR"]):
        y0 = leg * ph
        dr.line([(40, y0 + ph // 2), (w, y0 + ph // 2)], fill=(220, 220, 220))
        dr.text((4, y0 + 4), name, fill=(0, 0, 0))
        for j in range(3):
            v = q[:, 3 * leg + j]
            xs = 40 + np.arange(len(v)) * (w - 50) / max(1, len(v) - 1)
            ys = y0 + ph / 2 - v * (ph / 2 - 6) / 2.8   # +-2.8 rad fills the panel
            dr.line(list(zip(xs, ys)), fill=colors[j], width=2)
    dr.text((w - 200, 4), "abd red  hip blue  knee green", fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((HERE / "clips").glob("*.npz")):
        if args.clips and f.stem not in args.clips:
            continue
        clip = dict(np.load(f))
        contact_sheet(clip).save(out / f"{f.stem}_frames.png")
        curves(clip).save(out / f"{f.stem}_joints.png")
        print("wrote", out / f"{f.stem}_frames.png")


if __name__ == "__main__":
    main()
