"""Short GIFs of a brain in the simulator, for the README (simulation; offscreen MuJoCo rendering).

    BOLT_BODY=go2 MUJOCO_GL=osmesa python research/render_gif.py go2 ../models/bolt-go2-damp-798M/params.pkl \
        "trot:4,sit:3" ../docs/media/go2_trot_sit.gif
    BOLT_BODY=lite3 MUJOCO_GL=osmesa python research/render_gif.py lite3 ../models/bolt-lite3-dog-61.8M/params.pkl \
        "trot:4,walk:3" ../docs/media/lite3_trot.gif

Runs the brain in ImitateEnv (the training env, physics randomization off), switching clips on the plan, with a camera
that follows the body from the side; 25 frames a second, every other control step.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imitate_env import MOTION_CLIPS, ImitateEnv  # noqa: E402
from mjx_imitate import MjxImitatePolicy  # noqa: E402


def main(body, brain, plan, out, width=480, height=300):
    p = MjxImitatePolicy(brain)
    env = ImitateEnv(body=body, randomize=False, clips=MOTION_CLIPS)
    steps = [(c.split(":")[0], float(c.split(":")[1])) for c in plan.split(",")]
    env.exam_start = (steps[0][0], 0.0)
    obs, _ = env.reset(seed=1)
    env.limit = 10 ** 6
    r = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.model.body("torso").id
    cam.distance, cam.elevation, cam.azimuth = 1.6, -12.0, 100.0
    frames, t = [], 0
    for k, (clip, secs) in enumerate(steps):
        if k:
            env._switch_to(env.clips[clip])
            env.limit = 10 ** 6
        for _ in range(int(secs / env.dt)):
            obs, _, fell, _, _ = env.step(p.act(obs))
            if t % 2 == 0:
                r.update_scene(env.data, cam)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False       # the software renderer speckles shadows
                r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                frames.append(r.render().copy())
            t += 1
            if fell:
                break
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=0.04, loop=0)
    print(f"{out}: {len(frames)} frames, {Path(out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:5])
