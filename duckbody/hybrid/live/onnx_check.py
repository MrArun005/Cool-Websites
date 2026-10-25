"""Reference outputs for onnx_check.mjs: Microduck's policies through onnxruntime (Python) on 200 observations each
(random, around a standing duck's). $PY onnx_check.py OUT.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "microduck_port"))
import harness as Hm  # noqa: E402

rng = np.random.default_rng(0)
out = {}
for b in ("velstand", "ball_kick_left", "ball_kick_right", "roulade"):
    pol = Hm.Policy(b)
    X = rng.normal(0, 1, (200, 61)).astype(np.float32) * np.r_[[0.5] * 3, [0.3] * 3, [0.3] * 14, [2] * 14, [0.5] * 14, [0.4] * 13].astype(np.float32)
    X[:, 5] -= 1.0
    out[b] = dict(X=X.tolist(), Y=[pol(x).tolist() for x in X])
Path(sys.argv[1]).write_text(json.dumps(out))
