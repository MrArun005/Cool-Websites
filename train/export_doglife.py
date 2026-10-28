"""Export the brains Bolt's Day Out (gaitlab/doglife.js) runs in the browser, in the formats the pages load.

    python export_doglife.py <out data folder> [--reference /tmp/doglife_reference.json]

Writes into the folder:
    dog_gpu360.json   the GPU dog-motion brain (models/bolt-dog-imitate-gpu-360M/params.pkl, 9 clip slots) as a
                      live.js Brain: normalizer (mean, var = std^2, eps 0, clip 10) + ELU layers, base64 float32
    dog_18g.json      the canter-switching brain (run 18g at 35.5M, SB3, 8 slots), for canter and canter -> trot
    jump_16i.json     the run-up jump brain (run 16i at 35.5M, 51 inputs; its clock input is t / 200)
    dog_clips.json    all 9 reference clips (walk trot pace canter turn_left turn_right sit lie getup) + constants
--reference: Python's ImitateEnv (fixed physics) run with the GPU brain, every step's inputs and action, for
gaitlab/doglife_brain_test.mjs (the browser brain on Python's inputs must give Python's actions).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bolt_env  # noqa: E402
from bolt_env import ACTION_SCALE, FEET  # noqa: E402
from export_imitate import clips_json, parity_runs  # noqa: E402
from export_live import brain, f32, f64  # noqa: E402
from imitate_env import ALL_CLIPS, DOG, HOLD, NEXT, ImitateEnv  # noqa: E402
from mjx_imitate import MjxImitatePolicy  # noqa: E402

MODELS = HERE.parent / "models"


def gpu_brain(path):
    P = MjxImitatePolicy(path)
    std = np.asarray(P.std, dtype=np.float64)
    return dict(inputs=int(len(P.mean)), mean=f64(np.asarray(P.mean, np.float64)), var=f64(std * std), eps=0.0, clip=10.0,
                layers=[dict(w=f32(np.asarray(W).T), b=f32(b), inp=int(np.asarray(W).shape[0]), out=int(np.asarray(W).shape[1]))
                        for W, b in P.layers]), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reference")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda name, obj: (out / name).write_text(json.dumps(obj, separators=(",", ":")))
    spec, policy = gpu_brain(MODELS / "bolt-dog-imitate-gpu-360M" / "params.pkl")
    dump("dog_gpu360.json", dict(spec, source="bolt-dog-imitate-gpu-360M"))
    dump("dog_18g.json", dict(brain(MODELS / "bolt-dog-canter-switching-18g-35.5M"), source="bolt-dog-canter-switching-18g-35.5M"))
    dump("jump_16i.json", dict(brain(MODELS / "bolt-dog-jump-runup-16i-35.5M"), source="bolt-dog-jump-runup-16i-35.5M", jump_steps=200))
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS)
    constants = dict(dt=env.dt, frame_skip=env.frame_skip, dog_pose=DOG.tolist(), action_scale=ACTION_SCALE.tolist(),
                     slots=len(ALL_CLIPS), all_clips=ALL_CLIPS, hold=HOLD, next={k: list(v) for k, v in NEXT.items()},
                     feet=FEET, motor=getattr(bolt_env, "MOTOR", None))
    dump("dog_clips.json", dict(constants=constants, clips=clips_json(ALL_CLIPS)))
    if a.reference:
        runs = parity_runs(policy, ALL_CLIPS, steps=60)
        Path(a.reference).write_text(json.dumps(dict(parity=runs)))
    for f in sorted(out.glob("*.json")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
