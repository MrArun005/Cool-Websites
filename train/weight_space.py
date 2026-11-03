"""Weight-space averaging and blending of numpy brains (params.pkl, mjx_imitate's format): no training, CPU, minutes.

    python weight_space.py average out.pkl a.pkl b.pkl c.pkl ...        # equal-weight average ("checkpoint soup")
    python weight_space.py blend out.pkl a.pkl b.pkl 0.3                 # (1 - 0.3) * a + 0.3 * b

Works for brains of one lineage (one starting brain, the same layer sizes and inputs): the checkpoints of one run (a
stochastic-weight-average-style soup, Izmailov et al. 2018), or runs fine-tuned from the same parent (model soups,
Wortsman et al. 2022). Averaged: every layer's weights and biases (policy and value), the action noise, and the input
normalizer (mean, and variance; std recomputed), since each brain's first layer expects its own normalized inputs and
the normalizers of one lineage drift slowly. Refused: brains of different bodies, sizes, slot counts or history.
Each result must still pass the exams and the vendor rehearsals before it replaces anything (deploy/PLAN.md).
"""
import pickle
import sys
from pathlib import Path

import numpy as np


def load(p):
    return pickle.loads(Path(p).read_bytes())


def combine(brains, weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    a = brains[0]
    for b in brains[1:]:
        for k in ("body", "history", "blind_vel"):
            assert b.get(k) == a.get(k), f"different {k}: {a.get(k)} vs {b.get(k)}"
        assert len(b["mean"]) == len(a["mean"]) and len(b["layers"]) == len(a["layers"]), "different inputs or layers"
        assert all(np.shape(x[0]) == np.shape(y[0]) for x, y in zip(a["layers"], b["layers"])), "different layer sizes"
    mix = lambda xs: sum(wi * np.asarray(x, np.float64) for wi, x in zip(w, xs)).astype(np.float32)   # noqa: E731
    var = [np.asarray(b.get("var", np.asarray(b["std"], np.float64) ** 2), np.float64) for b in brains]
    out = dict(a)
    out["mean"] = mix([b["mean"] for b in brains])
    out["var"] = mix(var)
    # std as the brains compute it: sqrt(var + eps) for SB3 exports, the stored std otherwise; mix the stds directly
    out["std"] = mix([b["std"] for b in brains])
    out["layers"] = [(mix([b["layers"][i][0] for b in brains]), mix([b["layers"][i][1] for b in brains])) for i in range(len(a["layers"]))]
    if all("value" in b for b in brains) and all(len(b["value"]) == len(a["value"]) for b in brains):
        out["value"] = [(mix([b["value"][i][0] for b in brains]), mix([b["value"][i][1] for b in brains])) for i in range(len(a["value"]))]
    out["log_std"] = mix([b["log_std"] for b in brains])
    out["source"] = f"weight_space: {len(brains)} brains, weights {np.round(w, 3).tolist()}"
    out["steps"] = int(max(int(b.get("steps", 0)) for b in brains))
    return out


def main():
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "average":
        paths = sys.argv[3:]
        P = combine([load(p) for p in paths], [1.0] * len(paths))
    elif cmd == "blend":
        a, b, t = sys.argv[3], sys.argv[4], float(sys.argv[5])
        P = combine([load(a), load(b)], [1 - t, t])
    else:
        sys.exit(__doc__)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(pickle.dumps(P))
    print(f"{out}: {P['source']}")


if __name__ == "__main__":
    main()
