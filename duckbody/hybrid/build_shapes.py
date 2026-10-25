"""Head SHAPE variants for the hybrid (README "Lighter head" -> "Head shapes"): a human-like robot head and a rounder
duck head, as smooth meshes for the simulation and as printable STL shells that mount like Microduck's own head.

    /home/user/cool-websites/.venv/bin/python duckbody/hybrid/build_shapes.py      # then build_hybrid.py

Both new skulls replace Microduck's top head shell + face part only. Microduck's own bottom head shell stays (it
carries the head mount: the motor support on the head_roll XL330 and the roll bearing), so the mounting interface is
Microduck's, unchanged: the skull sits on a 2 mm flange that rests on the bottom shell's rim. Inside: Microduck's head
stack as is (Pi Zero 2 W + hat, camera + M12 lens, speaker, mouth XL330 + support, head_roll XL330); the script checks
that every one of those parts is inside the skull's inner wall with clearance.

Coordinates while designing: millimetres relative to the head_roll joint at Microduck's STAND pose, world axes:
F forward (+x), L left (+y), U up (+z). The head frame (jaw_soft) is (U, L, -F) / 1000.

Writes shapes/:
  <name>_skull.stl          closed outer surface (sim visual + convex collision), head frame, metres
  print/<name>_shell.stl    printable: skull wall (WALL mm) + mounting flange, mm, upright (rim at z = 0)
  print/<name>_bill.stl, print/<name>_jaw.stl  (duck: upper bill, lower bill; human: chin) solid, mm
  shapes.json               per shape: features (eyes, bill, jaw), eye camera pose, mass split, CoM, inertia
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "shapes"
WALL = 1.6               # printed wall, mm (4 perimeters of 0.4 mm)
FLANGE = 1.5             # flange thickness, mm
LEDGE = 3.0              # flange overlap onto the bottom shell's rim, mm
CUT_U = -2.0             # skull rim plane (Microduck's top shell comes down to -4.5, its bottom shell up to +3.2)
PLA = 1.24e-3            # g / mm^3
FIXED_G = 81.0           # head parts that do not depend on the shell (README "Lighter head")
BOTTOM_SHELL_G = 12.0    # Microduck's bottom head shell (23.8 cm^3 solid, printed ~50 %)
STACK = {"elec_rpi_robot_hat_pcb", "lens", "m12_lens_holder", "motor_support", "pcb__raspberry_pi_zero_2_w",
         "seeed_bearing__configuration_default", "speaker", "xl330"}

SHAPES = {
    # superellipsoid skull: centre (F, U), semi-axes front / back / side / up / down, exponents horizontal / vertical.
    # The skull reaches 52-60 mm behind the head_roll axis and is 59 mm tall with the bottom shell: lying on its back
    # the duck levers itself up over the back of the head, as on the owner's head (47 mm behind) and Microduck's.
    "human": dict(c=(24.0, 8.0), a_f=72.0, a_b=76.0, b=50.0, c_u=34.0, c_d=36.0, ph=2.5, pv=2.8, face_u=20.0,
                  color=(0.86, 0.87, 0.9, 1), label="human-like robot head"),
    "roundduck": dict(c=(20.0, 4.0), a_f=74.0, a_b=80.0, b=53.0, c_u=40.0, c_d=34.0, ph=2.3, pv=2.6,
                      color=(1.0, 1.0, 1.0, 1), label="round duck head"),
    # first versions (README "Head shapes"): taller domes ending 39-41 mm behind the axis. They broke the get-up from
    # the back (1/10, 3/10) and the roll (5/10, 8/10), so they are kept only for that comparison.
    "human_tall": dict(c=(28.0, 12.0), a_f=68.0, a_b=67.0, b=50.0, c_u=50.0, c_d=40.0, ph=2.5, pv=2.3,
                       color=(0.86, 0.87, 0.9, 1), label="human-like robot head, first (tall) version"),
    "roundduck_tall": dict(c=(26.0, 6.0), a_f=67.0, a_b=66.0, b=53.0, c_u=42.0, c_d=36.0, ph=2.2, pv=2.1,
                           color=(1.0, 1.0, 1.0, 1), label="round duck head, first (tall) version"),
}


def sgnpow(w, e):
    return np.sign(w) * np.abs(w) ** e


class Skull:
    def __init__(self, c, a_f, a_b, b, c_u, c_d, ph, pv, **_):  # noqa: E741
        self.F0, self.U0 = c
        self.a_f, self.a_b, self.b, self.c_u, self.c_d, self.ph, self.pv = a_f, a_b, b, c_u, c_d, ph, pv

    def point(self, th, ph_):
        """th: elevation (-pi/2..pi/2), ph_: azimuth. -> (F, L, U)."""
        cth = np.cos(th)
        x = sgnpow(cth, 2 / self.pv) * sgnpow(np.cos(ph_), 2 / self.ph)
        y = sgnpow(cth, 2 / self.pv) * sgnpow(np.sin(ph_), 2 / self.ph)
        z = sgnpow(np.sin(th), 2 / self.pv)
        a = np.where(x >= 0, self.a_f, self.a_b)
        c = np.where(z >= 0, self.c_u, self.c_d)
        return np.stack([self.F0 + a * x, self.b * y, self.U0 + c * z], -1)

    def level(self, P, shrink=0.0):
        """< 1 inside. shrink (mm) evaluates against the surface moved inwards by about that much."""
        x, y, z = P[..., 0] - self.F0, P[..., 1], P[..., 2] - self.U0
        a = np.where(x >= 0, self.a_f, self.a_b) - shrink
        c = np.where(z >= 0, self.c_u, self.c_d) - shrink
        return ((np.abs(x / a) ** self.ph + np.abs(y / (self.b - shrink)) ** self.ph) ** (self.pv / self.ph)
                + np.abs(z / c) ** self.pv)

    def th_cut(self):
        z = (CUT_U - self.U0) / self.c_d
        return -np.arcsin(np.abs(z) ** (self.pv / 2)) if z < 0 else np.arcsin(z ** (self.pv / 2))

    def surface_F(self, L, U):
        """Front surface F at (L, U) (bisection along +F)."""
        lo, hi = self.F0, self.F0 + self.a_f + 1
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.level(np.array([mid, L, U])) < 1:
                lo = mid
            else:
                hi = mid
        return lo

    def surface_L(self, F, U):
        lo, hi = 0.0, self.b + 1
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.level(np.array([F, mid, U])) < 1:
                lo = mid
            else:
                hi = mid
        return lo


# ------------------------------------------------------------------------------------------------ meshes
def grid_mesh(sk, n_th=48, n_ph=128):
    """Outer surface above the rim plane: rings from the rim up + the top pole."""
    th = np.linspace(sk.th_cut(), np.pi / 2, n_th)[:-1]
    ph = np.linspace(0, 2 * np.pi, n_ph, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = sk.point(T, P).reshape(-1, 3)
    top = sk.point(np.array(np.pi / 2), np.array(0.0))
    V = np.vstack([V, top])
    F = []
    for i in range(n_th - 2):
        for j in range(n_ph):
            a, b2 = i * n_ph + j, i * n_ph + (j + 1) % n_ph
            F += [(a, b2, b2 + n_ph), (a, b2 + n_ph, a + n_ph)]
    last, pole = (n_th - 2) * n_ph, len(V) - 1
    F += [(last + j, last + (j + 1) % n_ph, pole) for j in range(n_ph)]
    return V, np.array(F), n_ph


def vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def closed_solid(V, F, n_ph):
    """Close the rim with a flat cap (fan to the rim's centroid): the sim's skull solid."""
    rim = np.arange(n_ph)
    c = V[rim].mean(0)
    V2 = np.vstack([V, c])
    ci = len(V2) - 1
    cap = [(rim[(j + 1) % n_ph], rim[j], ci) for j in range(n_ph)]
    return V2, np.vstack([F, cap])


def shell(V, F, n_ph, t):
    """Printable wall: outer surface, inner surface t mm inside (along the normals), rim strip at the plane."""
    n = vertex_normals(V, F)
    Vi = V - t * n
    Vi[:n_ph, 2] = V[:n_ph, 2]                          # keep the inner rim on the plane
    N = len(V)
    Fi = F[:, ::-1] + N
    rim = [((j + 1) % n_ph, j, N + j) for j in range(n_ph)] + \
          [((j + 1) % n_ph, N + j, N + (j + 1) % n_ph) for j in range(n_ph)]
    return np.vstack([V, Vi]), np.vstack([F, Fi, np.array(rim)]), Vi[:n_ph]


def ellipsoid(center, size, n=32, m=20):
    th = np.linspace(-np.pi / 2, np.pi / 2, m)[1:-1]
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = np.stack([np.cos(T) * np.cos(P), np.cos(T) * np.sin(P), np.sin(T)], -1).reshape(-1, 3)
    V = np.vstack([V, [0, 0, -1], [0, 0, 1]]) * np.asarray(size) + np.asarray(center)
    F = []
    for i in range(m - 3):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            F += [(a, b, b + n), (a, b + n, a + n)]
    bot, top = len(V) - 2, len(V) - 1
    F += [(bot, (j + 1) % n, j) for j in range(n)]
    F += [((m - 3) * n + j, (m - 3) * n + (j + 1) % n, top) for j in range(n)]
    return V, np.array(F)


def polar_radius(P2, c, angles):
    """Radius of the closed 2D polygon P2 (convex) along rays from c at the given angles."""
    hull = cv2.convexHull(np.asarray(P2, np.float32)).reshape(-1, 2).astype(float)
    out = []
    for a in angles:
        d = np.array([np.cos(a), np.sin(a)])
        best = 0.0
        for k in range(len(hull)):
            p, q = hull[k] - c, hull[(k + 1) % len(hull)] - c
            M = np.array([d, p - q]).T
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            s, u = np.linalg.solve(M, p)
            if s > 0 and -1e-9 <= u <= 1 + 1e-9:
                best = max(best, s)
        out.append(best)
    return np.array(out)


def flange(inner_xy, outer_xy, c, z0, t, n=128):
    """Annulus prism between two closed outlines (sampled on rays from c), z0 .. z0 + t."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ri, ro = polar_radius(inner_xy, c, ang), polar_radius(outer_xy, c, ang)
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    loops = [c + d * ri[:, None], c + d * ro[:, None]]
    V = np.vstack([np.column_stack([lp, np.full(n, z)]) for z in (z0, z0 + t) for lp in loops])
    I0, O0, I1, O1 = 0, n, 2 * n, 3 * n
    F = []
    for j in range(n):
        k = (j + 1) % n
        F += [(I0 + j, O0 + k, O0 + j), (I0 + j, I0 + k, O0 + k)]          # bottom (faces down)
        F += [(I1 + j, O1 + j, O1 + k), (I1 + j, O1 + k, I1 + k)]          # top
        F += [(O0 + j, O0 + k, O1 + k), (O0 + j, O1 + k, O1 + j)]          # outer wall
        F += [(I0 + j, I1 + k, I0 + k), (I0 + j, I1 + j, I1 + k)]          # inner wall
    return V, np.array(F), ri, ro


def write_stl(path, V, F, scale=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, float) * scale
    with open(path, "wb") as f:
        f.write(b"build_shapes.py".ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        for a, b, c in F:
            n = np.cross(V[b] - V[a], V[c] - V[a])
            n = n / (np.linalg.norm(n) or 1)
            f.write(struct.pack("<12fH", *n, *V[a], *V[b], *V[c], 0))


def mesh_props(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1) / 2
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    return area, vol, (a + b + c) / 3


def watertight(F):
    from collections import Counter
    e = Counter()
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            e[(u, v)] += 1
    return all(e[(v, u)] == 1 and n == 1 for (u, v), n in e.items())


def to_head(P):
    """(F, L, U) mm -> head frame (U, L, -F) m."""
    P = np.asarray(P, float)
    return np.stack([P[..., 2], P[..., 1], -P[..., 0]], -1) / 1000


# ------------------------------------------------------------------------------------------------ main
def microduck_parts():
    m = mujoco.MjModel.from_xml_path(str(HERE / "scene_hybrid_allcollisions_noantenna.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
    mujoco.mj_forward(m, d)
    b = m.body("jaw_soft").id
    ph = d.xpos[b]
    parts = {}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 4:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            P = (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - ph) * 1000
            parts.setdefault(m.mesh(mid).name, []).append(P)
    return {k: np.vstack(v) for k, v in parts.items()}


def build_shape(name, spec, parts, report):
    sk = Skull(**spec)
    V, F, n_ph = grid_mesh(sk)
    # 1. the head stack fits: every stack vertex inside the inner wall with >= 1 mm clearance
    stack = np.vstack([parts[k] for k in STACK])
    stack = stack[stack[:, 2] > CUT_U]                          # below the rim it is inside Microduck's bottom shell
    worst = float(sk.level(stack, shrink=WALL + 1.0).max())
    assert worst < 1.0, (name, worst)
    # 2. the rim encloses the bottom shell's rim with room for the ledge
    bs = parts["bottom_head_shell"]
    rim_pts = bs[bs[:, 2] > bs[:, 2].max() - 2.5][:, :2]
    c2 = np.array([sk.F0, 0.0])
    solid_V, solid_F = closed_solid(V, F, n_ph)
    sV, sF, inner_rim = shell(V, F, n_ph, WALL)
    ang = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    r_bs = polar_radius(rim_pts, c2, ang)
    r_in = polar_radius(inner_rim[:, :2], c2, ang)
    assert (r_in - r_bs).min() > 0.5, (name, (r_in - r_bs).min())
    ledge_xy = c2 + np.stack([np.cos(ang), np.sin(ang)], 1) * (r_bs - LEDGE)[:, None]
    over_xy = c2 + np.stack([np.cos(ang), np.sin(ang)], 1) * (r_in + 0.4)[:, None]   # 0.4 mm into the wall
    fV, fF, _, _ = flange(ledge_xy, over_xy, c2, CUT_U, FLANGE)
    assert watertight(solid_F) and watertight(sF) and watertight(fF), name

    # features (F, L, U mm), each: kind, centre, half sizes, colour, part (head | jaw), collides
    feats, eye = [], None
    if name.startswith("human"):
        Ue, Le = 20.0, 22.0
        Ff = sk.surface_F(0.0, 16.0)
        feats.append(dict(n="faceplate", kind="ellipsoid", c=(Ff - 8.0, 0.0, 15.0), s=(11.0, 36.0, spec.get("face_u", 25.0)),
                          rgba=(0.12, 0.14, 0.18, 1)))
        for sgn in (1, -1):
            Fe = sk.surface_F(sgn * Le, Ue)
            feats.append(dict(n=f"eye_{'l' if sgn > 0 else 'r'}", kind="ellipsoid", c=(max(Fe, Ff + 2.0) + 0.5, sgn * Le, Ue),
                              s=(2.0, 8.0, 8.0), rgba=(0.35, 0.85, 1.0, 1)))
        eye = (max(sk.surface_F(Le, Ue), Ff + 2.0) + 2.5, Le, Ue)        # camera behind the left eye, same height
        feats.append(dict(n="mouth_slit", kind="box", c=(Ff + 2.0, 0.0, -1.0), s=(1.5, 15.0, 1.5), rgba=(0.05, 0.05, 0.06, 1)))
        Fj = sk.surface_F(0.0, CUT_U) - 4.0
        jaw = dict(pos=(Fj, 0.0, -6.0), geoms=[dict(n="chin", kind="ellipsoid", c=(6.0, 0.0, -4.0), s=(10.0, 26.0, 8.0),
                                                     rgba=spec["color"], collide=True)])
    else:
        kb = 0.85
        Ub, Ue = 2.7, 20.0
        Fb = sk.surface_F(0.0, Ub)
        feats.append(dict(n="upper_bill", kind="ellipsoid", c=(Fb + kb * 21.0 - 6.0, 0.0, Ub),
                          s=(kb * 21.0, kb * 38.5, kb * 6.3), rgba=(1, 0.62, 0.1, 1), collide=True))
        for sgn in (1, -1):
            Fe, Ue2 = 58.0, 22.0
            Le = sk.surface_L(Fe, Ue2)
            feats.append(dict(n=f"eye_{'l' if sgn > 0 else 'r'}", kind="ellipsoid", c=(Fe, sgn * (Le - 1.5), Ue2),
                              s=(6.5, 3.5, 7.5), rgba=(0.05, 0.05, 0.06, 1)))
        eye = (sk.surface_F(0.0, Ue) + 1.0, 0.0, Ue)                     # front of the skull above the bill, as before
        Fj = Fb - 6.0
        jaw = dict(pos=(Fj, 0.0, -5.0), geoms=[dict(n="lower_bill", kind="ellipsoid", c=(kb * 21.0, 0.0, 0.0),
                                                      s=(kb * 21.0, kb * 35.0, kb * 4.2), rgba=(1, 0.62, 0.1, 1), collide=True)])

    # mass: skull wall + flange + features (PLA) + the fixed parts (Microduck's head stack) + bottom shell
    area, _, cen = mesh_props(V, F)
    skull_g = float(area.sum() * WALL * PLA)
    skull_c = (area[:, None] * cen).sum(0) / area.sum()
    _, fvol, _ = mesh_props(fV, fF)
    flange_g = float(abs(fvol) * PLA)
    feat_g, feat_m = 0.0, []
    for ft in feats + [dict(g, c=tuple(np.add(jaw["pos"], g["c"]))) for g in jaw["geoms"]]:
        s = np.array(ft["s"])
        if ft["n"] in ("upper_bill", "lower_bill", "chin"):       # printed hollow, 0.8 mm walls
            p = 1.6
            ar = 4 * np.pi * ((s[0] ** p * s[1] ** p + s[0] ** p * s[2] ** p + s[1] ** p * s[2] ** p) / 3) ** (1 / p)
            g_ = ar * 0.8 * PLA
        else:
            g_ = 0.5
        feat_g += g_
        feat_m.append((g_, np.array(ft["c"])))
    est_g = FIXED_G + BOTTOM_SHELL_G + skull_g + flange_g + feat_g

    # printable files
    write_stl(OUT / f"{name}_skull.stl", to_head(solid_V), solid_F)
    pV = np.vstack([sV, fV])
    pF = np.vstack([sF, fF + len(sV)])
    write_stl(OUT / "print" / f"{name}_shell.stl", pV - [0, 0, CUT_U], pF)
    for g in feats:
        if g.get("collide"):
            eV, eF = ellipsoid((0, 0, 0), g["s"])
            write_stl(OUT / "print" / f"{name}_bill.stl", eV, eF)
    for g in jaw["geoms"]:
        eV, eF = ellipsoid((0, 0, 0), g["s"])
        write_stl(OUT / "print" / f"{name}_jaw.stl", eV, eF)

    lo, hi = solid_V.min(0), solid_V.max(0)
    report[name] = dict(
        label=spec["label"], spec={k: v for k, v in spec.items() if k not in ("label",)},
        skull_size_mm=dict(F=round(hi[0] - lo[0], 1), L=round(hi[1] - lo[1], 1), U_above_rim=round(hi[2] - CUT_U, 1),
                           U_with_bottom_shell=round(hi[2] - bs[:, 2].min(), 1)),
        stack_fit_level=round(worst, 3), rim_margin_mm=round(float((r_in - r_bs).min()), 1),
        features=[dict(f, c=to_head(f["c"]).round(5).tolist(), s=(np.array(f["s"])[::-1] / 1000).round(5).tolist()) for f in feats],
        jaw=dict(pos=to_head(jaw["pos"]).round(5).tolist(),
                 geoms=[dict(g, c=to_head(g["c"]).round(5).tolist(), s=(np.array(g["s"])[::-1] / 1000).round(5).tolist()) for g in jaw["geoms"]]),
        eye_pos_head_frame=to_head(eye).round(5).tolist(), eye_mm=np.round(eye, 1).tolist(),
        mass_g=dict(fixed=FIXED_G, bottom_shell=BOTTOM_SHELL_G, skull_wall=round(skull_g, 1), flange=round(flange_g, 1),
                    features=round(feat_g, 1), estimate=round(est_g, 1)),
        skull_wall_centroid_head_frame=to_head(skull_c).round(5).tolist(),
        features_mass=[(round(g_, 2), to_head(c).round(5).tolist()) for g_, c in feat_m],
        print_files=[f"shapes/print/{name}_shell.stl", f"shapes/print/{name}_bill.stl" if not name.startswith("human") else None,
                     f"shapes/print/{name}_jaw.stl"],
    )
    print(name, json.dumps({k: report[name][k] for k in ("skull_size_mm", "stack_fit_level", "rim_margin_mm", "eye_mm", "mass_g")}))


def main():
    parts = microduck_parts()
    report = {}
    for name, spec in SHAPES.items():
        build_shape(name, spec, parts, report)
    (OUT / "shapes.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
