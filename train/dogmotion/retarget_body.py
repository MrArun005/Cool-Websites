"""Retarget the dog motion capture onto a real robot of robot_bodies.BODIES: reference clips for ImitateEnv(body=...).

    python dogmotion/retarget_body.py --body go2 --bvh-dir <folder with the MANN .bvh files>   # from train/, PYTHONPATH=.
    MUJOCO_GL=osmesa python dogmotion/retarget_body.py --body go2 --check-only --png <folder>   # re-check, pictures

The same pipeline as retarget.py (Peng et al. 2020's keypoint retargeting; the same source clips, segments, smoothing,
loop closing, floor and self-penetration handling), run with the robot's body in place of Bolt's. retarget.py's
functions read the body from module constants, so this file sets those constants from the body's table entry and its
measured kinematics, and swaps in the leg solver and the kinematic model; retarget.py itself, and Bolt's clips, are
untouched. What changes (numbers for the Go2):
  * scale (step 3): the dog's standing shoulder/hip height maps onto the body's stance (Go2: paw centre 0.283 m below
    the thigh joint, the body ~0.30 m up) instead of Bolt's 0.385 m;
  * paws (step 4) are placed under the THIGH joints (Go2: 0.142 m out from the midline, the 0.0465 m abduction joint
    plus the 0.0955 m hip offset), at the abduction joints' height (Go2: level with the body centre; Bolt's hips are
    3 cm below it), the body's paw ball (Go2 0.022 m);
  * leg IK (step 5): robot_bodies.leg_ik, with the hip offset (the abduction rolls the leg plane about an axis offset
    from it) and every knee backward, the rear legs solving the same triangle as the front;
  * reach: a planted paw further from the thigh joint than the leg with the knee 0.04 rad short of its straightest
    (Go2: -0.84 rad, 0.386 m), less retarget()'s 1.5 cm, lowers the body, as Bolt's 0.385 m did;
  * folding (step 6, sit and lie): the entry's own folded legs ("sit", "lie") as the body comes down from 0.22 and
    0.12 m (Bolt's) times the stance-reach ratio above the paws (Go2: 0.16 / 0.09 m).
Clips are written next to Bolt's as dogmotion/clips/<clip_prefix><name>.npz (getup: the lie reversed, as
make_getup.py), the check as clips/<clip_prefix>check.json. The check adds paw slip: how far a planted paw (on the floor
in two frames in a row) moves along the floor between frames, which the reference should keep near zero.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import retarget as rt  # noqa: E402
import robot_bodies  # noqa: E402

PAW_SINK = 0.005   # m a paw may sink into the floor before the body is lifted (RobotKin)
BODY = None        # the robot_bodies.Body in use (use_body)


def use_body(key):
    """Point retarget.py's body constants and solvers at robot_bodies body `key`."""
    global BODY
    BODY = b = robot_bodies.get(key)
    k = b.kin
    rt.PAW_R = k["paw_r"]
    rt.HIP_Z = k["abd_z"]
    rt.HIP_X = np.array([k["abd_x"], k["abd_x"], -k["abd_x"], -k["abd_x"]])
    y = k["abd_y"] + k["hip_offset"]
    rt.HIP_Y = np.array([y, -y, y, -y])                       # where the paws go: under the thigh joints
    rt.STAND_REACH = float(k["abd_z"] - robot_bodies.paw_positions(k, b.stance)[0, 2])
    m = b.model()
    knee_max = float(m.jnt_range[m.joint(b.joints[2]).id][1]) - 0.04
    ry = np.array([[np.cos(knee_max), 0, np.sin(knee_max)], [0, 1, 0], [-np.sin(knee_max), 0, np.cos(knee_max)]])
    rt.LINK = float(np.linalg.norm([0, 0, -k["thigh"]] + ry @ k["calf_paw"])) / 2   # retarget() keeps paws within 2 LINK
    scale = rt.STAND_REACH / (2 * 0.2 * np.cos(0.2747))        # the body's stance reach over Bolt's
    rt.FOLD_FROM, rt.FOLD_FULL = 0.22 * scale, 0.12 * scale
    rt.LIE_POSE, rt.SIT_POSE = b.lie, b.sit
    rt.STANCES = {"dog": b.stance}
    rt.solve_legs = solve_legs
    rt.MODEL = None
    return b


def solve_legs(R, c, targets, lo, hi):
    """retarget.solve_legs for the robot: joint angles putting each paw centre at its world target, the roll clamped
    first and the leg solved in the plane it rolls to; also the clamped amount (rad) and how far out of reach (m)."""
    k = BODY.kin
    q, clamp, short = np.zeros(12), 0.0, 0.0
    for leg in range(4):
        left = leg in (0, 2)
        p = R.T @ (targets[leg] - c) - np.array([rt.HIP_X[leg], k["abd_y"] if left else -k["abd_y"], k["abd_z"]])
        raw, sh = robot_bodies.leg_ik(k, p, left)
        abd = float(np.clip(raw[0], lo[3 * leg], hi[3 * leg]))
        sol, sh2 = robot_bodies.leg_ik(k, p, left, abd=abd)
        raw = np.array([raw[0], sol[1], sol[2]])
        short = max(short, sh, sh2)
        cl = np.clip(raw, lo[3 * leg:3 * leg + 3], hi[3 * leg:3 * leg + 3])
        clamp = max(clamp, float(np.abs(raw - cl).max()))
        q[3 * leg:3 * leg + 3] = cl
    return q, clamp, short


class RobotKin(rt.Bolt):
    """retarget.Bolt (placing the body, floor and self-contact checks) on the robot's model; its legs and body may carry
    cylinders too, which lowest_nonfoot measures. The paws count as well once one would sink more than PAW_SINK into
    the floor: on a folded Go2 leg the paw ball (r 22 mm) sticks out below the calf (r 11-13 mm), so lifting the body
    only for the legs and body left the rear paws of the sit and lie 35 mm in the floor. Gait clips' planted paws sink
    at most 4 mm (IK misses), so they are not lifted by this."""

    def __init__(self):
        self.m = BODY.model()
        self.d = mujoco.MjData(self.m)
        jid = [self.m.joint(j).id for j in BODY.joints]
        self.lo, self.hi = self.m.jnt_range[jid, 0].copy(), self.m.jnt_range[jid, 1].copy()
        self.foot = [self.m.geom(f"{leg}_foot").id for leg in robot_bodies.LEGS]
        self.floor = self.m.geom("floor").id
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in self.m.geom_bodyid]
        self.geom_leg = np.array([["FL", "FR", "RL", "RR"].index(n[:2]) if n[:2] in ("FL", "FR", "RL", "RR") else -1
                                  for n in names])

    def lowest_nonfoot(self):
        low = super().lowest_nonfoot()
        m, d = self.m, self.d
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R, c, (r, h) = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g][:2]
            low = min(low, c[2] - abs(R[2, 2]) * h - r * np.sqrt(max(0.0, 1 - R[2, 2] ** 2)))
        paw = min(d.geom_xpos[g][2] - rt.PAW_R for g in self.foot)   # the lowest paw ball's bottom
        return min(low, paw + PAW_SINK + 0.004)                        # retarget() lifts the body while this is < 4 mm


def slip(clip):
    """Paw slip: horizontal motion (mm per 1/50 s frame) of paws planted in both frames, mean and max; and the share of
    paw-frames planted."""
    f, on = clip["foot_pos"], clip["foot_contact"].astype(bool)
    both = on[1:] & on[:-1]
    dxy = np.linalg.norm(f[1:, :, :2] - f[:-1, :, :2], axis=-1)[both]
    if not len(dxy):
        return dict(mean=0.0, max=0.0, planted_share=round(float(on.mean()), 2))
    return dict(mean=round(1000 * float(dxy.mean()), 1), max=round(1000 * float(dxy.max()), 1),
                planted_share=round(float(on.mean()), 2))


def replay_check(kin, clip):
    """Kinematic replay (qpos set frame by frame): paw slip, deepest paw and non-paw floor penetration (mm), deepest
    self penetration (mm), body height range (m)."""
    paw_pen, floor_pen, self_pen = 0.0, 0.0, 0.0
    for t in range(len(clip["joints"])):
        kin.set(clip["base_pos"][t], clip["base_quat"][t], clip["joints"][t])
        fp, sp = kin.contacts()
        floor_pen, self_pen = max(floor_pen, fp), max(self_pen, sp)
        paw_pen = max(paw_pen, max(rt.PAW_R - kin.d.geom_xpos[g][2] for g in kin.foot))
    z = clip["base_pos"][:, 2]
    return dict(paw_slip_mm=slip(clip), paw_penetration_mm=round(1000 * paw_pen, 1),
                floor_penetration_mm=round(1000 * floor_pen, 1), self_penetration_mm=round(1000 * self_pen, 1),
                body_height_m=[round(float(z.min()), 3), round(float(z.max()), 3)])


def getup(lie):
    """The getup clip: the lie clip played backwards (make_getup.py)."""
    out = {}
    for k, v in lie.items():
        if k in ("base_pos", "base_quat", "joints", "foot_pos", "foot_contact"):
            out[k] = v[::-1].copy()
        elif k in ("joint_vel", "base_linvel", "base_angvel"):
            out[k] = -v[::-1].copy()
        else:
            out[k] = v
    out["name"] = np.array("getup")
    out["source"] = np.array(str(lie.get("source", "")) + " | reversed lie clip: lie -> sit -> stand")
    return out


def pictures(kin, clips, out, n=8, size=(320, 240)):
    """A contact sheet per clip (n evenly spaced moments, seen from the side) and one larger frame of each, offscreen."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(kin.m, size[1], size[0])
    big = mujoco.Renderer(kin.m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    written = []
    for name, clip in clips.items():
        T = len(clip["joints"])
        tiles = []
        for i, t in enumerate(np.linspace(0, T - 1, n).astype(int)):
            kin.set(clip["base_pos"][t], clip["base_quat"][t], clip["joints"][t])
            mujoco.mj_forward(kin.m, kin.d)
            cam.lookat[:] = [kin.d.qpos[0], kin.d.qpos[1], 0.18]
            cam.distance, cam.elevation, cam.azimuth = 1.1, -8, 90
            r.update_scene(kin.d, cam)
            img = Image.fromarray(r.render())
            ImageDraw.Draw(img).text((6, 6), f"{name} t={t / 50:.2f}s", fill=(0, 0, 0))
            tiles.append(img)
            if i == n // 2:   # one larger, three-quarter view
                cam.distance, cam.elevation, cam.azimuth = 1.2, -15, 135
                big.update_scene(kin.d, cam)
                Image.fromarray(big.render()).save(out / f"{BODY.clip_prefix}{name}_frame.png")
                written.append(out / f"{BODY.clip_prefix}{name}_frame.png")
        sheet = Image.new("RGB", (size[0] * 4, size[1] * ((n + 3) // 4)), "white")
        for i, im in enumerate(tiles):
            sheet.paste(im, ((i % 4) * size[0], (i // 4) * size[1]))
        sheet.save(out / f"{BODY.clip_prefix}{name}_frames.png")
        written.append(out / f"{BODY.clip_prefix}{name}_frames.png")
    r.close()
    big.close()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="go2", choices=sorted(robot_bodies.BODIES), help="the robot (robot_bodies.BODIES)")
    ap.add_argument("--bvh-dir", help="folder with the MANN dog .bvh files (MotionCapture.zip)")
    ap.add_argument("--out", default=str(HERE / "clips"))
    ap.add_argument("--check-only", action="store_true", help="re-check the clips already written")
    ap.add_argument("--png", help="also save contact sheets and frames of every clip to this folder")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_body(args.body)
    kin = RobotKin()
    names = [c[0] for c in rt.CLIPS] + ["getup"]
    checks = {}
    if not args.check_only:
        assert args.bvh_dir, "--bvh-dir is needed to retarget (or --check-only)"
        cal = rt.calibrate(args.bvh_dir)
        print(f"scale per leg {np.round(cal['scale'], 3)}, planted paw keypoint height {np.round(cal['ground'], 3)} m")
        for name, file, t0, t1, loop in rt.CLIPS:
            clip, check = rt.retarget(args.bvh_dir, name, file, t0, t1, loop, cal, kin)
            np.savez_compressed(out / f"{BODY.clip_prefix}{name}.npz", **{k: np.asarray(v) for k, v in clip.items()})
            checks[name] = check
        np.savez(out / f"{BODY.clip_prefix}getup.npz", **getup(dict(np.load(out / f"{BODY.clip_prefix}lie.npz"))))
    clips = {n: dict(np.load(out / f"{BODY.clip_prefix}{n}.npz")) for n in names}
    for n in names:
        checks.setdefault(n, {})["replay"] = replay_check(kin, clips[n])
        print(n, json.dumps(checks[n]))
    report = dict(body=args.body, stance=BODY.stance.tolist(), clips=checks)
    if not args.check_only:
        report["calibration"] = {k: v.round(4).tolist() for k, v in cal.items()}
        (out / f"{BODY.clip_prefix}check.json").write_text(json.dumps(report, indent=2))
    if args.png:
        for f in pictures(kin, clips, Path(args.png)):
            print("wrote", f)


if __name__ == "__main__":
    main()
