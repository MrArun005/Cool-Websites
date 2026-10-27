"""Export the MasterMind student for the browser, in the brain format the pages load (export_live.brain: normalizer
+ ELU layers, base64 float32), plus what the page needs to drive it.

    cd train && python mastermind/export.py mastermind/out/student.pt --out mastermind/export

Writes <out>/brain.json (inputs, mean, var, eps=0, clip=10, layers) and <out>/model.json (the 84 inputs in order,
the command list, and the action mapping: ctrl = clip(dog_stance + full_range * output, lo, hi)). The reference
clips the inputs need are the imitation brain's clips.json (export_imitate.py), unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from export_live import f32, f64  # noqa: E402

OBS_LAYOUT = [("gravity_in_body_frame", 3), ("angular_velocity_x0.25", 3), ("linear_velocity_body", 3),
              ("body_height", 1), ("joint_pos_minus_dog_stance", 12), ("joint_vel_x0.1", 12), ("previous_output", 12),
              ("command_one_hot", 10), ("ref_active", 1), ("clip_phase_sin_cos", 2), ("ref_joints_next_minus_dog_stance", 12),
              ("ref_height_next", 1), ("ref_velocity_body_next", 3), ("ref_yaw_rate_next", 1), ("seconds_since_clip_end", 1),
              ("hurdle_present", 1), ("hurdle_ahead_m_clipped_-1_3", 1), ("bar_height", 1), ("sin_heading_error", 1),
              ("off_line_m", 1), ("jump_clock_s_over_2", 1), ("seconds_since_command_over_3_max_1", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("--out", default=str(HERE / "export"))
    a = ap.parse_args()
    from env import COMMANDS, OBS, Master
    from imitate_env import DOG
    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    sd = ck["state"]
    lin = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("mlp.") and k.endswith("weight")}, key=lambda s: int(s.split(".")[1]))
    std = sd["std"].numpy()
    brain = dict(inputs=OBS, mean=f64(sd["mean"].numpy()), var=f64(std ** 2), eps=0.0, clip=10.0,
                 layers=[dict(w=f32(sd[k + ".weight"].numpy()), b=f32(sd[k + ".bias"].numpy()),
                              out=int(sd[k + ".weight"].shape[0]), inp=int(sd[k + ".weight"].shape[1])) for k in lin],
                 source=str(a.student), activation="elu")
    m = Master(seed=0, randomize=False)
    assert sum(n for _, n in OBS_LAYOUT) == OBS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "brain.json").write_text(json.dumps(brain, separators=(",", ":")))
    (out / "model.json").write_text(json.dumps(dict(
        obs_size=OBS, obs_layout=OBS_LAYOUT, commands=COMMANDS, imitate_teacher=ck.get("imitate"), dagger_round=ck.get("round"),
        action="ctrl = clip(dog_stance + full_range * output, lo, hi)  (joint position targets, rad)",
        dog_stance=np.round(DOG, 5).tolist(), full_range=np.round(m.full, 5).tolist(), lo=np.round(m.lo, 5).tolist(),
        hi=np.round(m.hi, 5).tolist(), control_hz=50), indent=1))
    print(f"wrote {out}/brain.json and model.json")


if __name__ == "__main__":
    main()
