"""duck_mesh_live.json: the duck's visible meshes per body (same selection and format as page_data.mesh), made light
enough to draw 11 ducks plus their shadows on an ordinary laptop or phone while the physics runs on the same CPU.

The quadric decimator page_data uses stalls on these meshes (the big shell parts stay at 14k faces whatever the target),
so this uses vertex clustering: snap vertices to a CELL grid in the body frame, merge, drop collapsed triangles.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/live/make_live_mesh.py OUT_DIR [cell_mm=3]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "microduck_port"))
import mujoco  # noqa: E402
from harness import MicroduckSim  # noqa: E402


def cluster(v, f, cell):
    key = np.round(v / cell).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nv = np.zeros((len(uniq), 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv, minlength=len(uniq))[:, None]
    nf = inv[f]
    ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[ok]
    nf = np.unique(np.sort(nf, 1), axis=0, return_index=True)[1]
    nf = inv[f][ok][np.sort(nf)]
    used, ref = np.unique(nf, return_inverse=True)
    return nv[used], ref.reshape(-1, 3)


def main():
    out = Path(sys.argv[1])
    cell = float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.003
    m = MicroduckSim(str((HERE.parent / "scene_hybrid_allcollisions_noantenna.xml").resolve()), seed=0).m
    parts, before, after = {}, 0, 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
        if b == 0 or rgba[3] < 0.05 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_group[g] not in (0, 1, 2):
            continue
        if m.geom_group[g] == 3:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]            # body frame, then cluster
        before += len(f)
        v, f = cluster(v, f, cell)
        if len(f) == 0:
            continue
        after += len(f)
        parts.setdefault(m.body(b).name, []).append(dict(
            color="#%02x%02x%02x" % tuple(int(255 * c) for c in rgba[:3]),
            v=np.round(v * 1e4).astype(int).ravel().tolist(), f=f.ravel().tolist()))
    names = [m.body(b).name for b in range(1, m.nbody)]
    (out / "duck_mesh_live.json").write_text(json.dumps(dict(scale=1e-4, bodies=names, parts=parts), separators=(",", ":")))
    print(f"duck_mesh_live.json: {before} -> {after} triangles per duck (cell {cell * 1000:.1f} mm)")


if __name__ == "__main__":
    main()
