"""Gate for PLAN.md phase 1 steps 3-4: the robot-side controller builds the same 75 inputs and sends the same joint
targets as imitate_env.ImitateEnv does in training, for the same robot state.

    cd train && BOLT_BODY=go2 MUJOCO_GL=osmesa python deploy/test_deploy.py [../models/bolt-go2-gpu-489M/params.pkl]

The env runs the brain closed loop (no randomization, no noise) through every clip, with a clip switch and steering;
at each step the controller is given only what a robot would read (IMU quaternion and gyro, joint angles and speeds)
plus the true body velocity and height (the estimator's job on the real robot, phase 3), and must agree with the env.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from controller import ALL_CLIPS, Go2Controller  # noqa: E402
from imitate_env import ImitateEnv  # noqa: E402


def readings(env):
    """What the robot side gets: IMU quaternion (w, x, y, z) and body-frame gyro, joints; plus the two estimates."""
    d = env.data
    R = env._rotation()
    quat = np.zeros(4)   # the env's own body rotation (xmat, from the last physics substep's kinematics: up to 2 ms
    mujoco.mju_mat2Quat(quat, R.reshape(-1))   # older than qpos; far below the trained IMU noise, but exact here)
    return dict(quat=quat, gyro=d.qvel[3:6].copy(), q=d.qpos[7:19].copy(), dq=d.qvel[6:18].copy(),
                v_body=R.T @ d.qvel[0:3], height=float(d.qpos[2]))


def main():
    brain = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent / "models" / "bolt-go2-gpu-489M" / "params.pkl")
    ctl = Go2Controller(brain)
    env = ImitateEnv(randomize=False, seed=3, clips=ALL_CLIPS, body="go2")
    worst_obs = worst_tgt = 0.0
    steps = 0
    for k, name in enumerate(["walk", "trot", "canter", "turn_left", "sit", "lie", "getup", "pace"]):
        env.exam_start = (name, 0.3)
        env.reset(seed=100 + k)
        env.steer = 0.4 if name in ("trot", "canter") else 0.0
        ctl.steer = env.steer
        ctl.last_action[:] = env.last_action
        for t in range(60):
            if name == "walk" and t == 30:   # a mid-try switch: the env's rule and the controller's must agree
                ctl.start_at(env.clip.name, env.tau())
                ctl.command("trot", q=env.data.qpos[7:19])
                env._switch_to(env.clips["trot"])
                assert ctl.clip.name == "trot" and abs(ctl.tau - env.tau()) < 1e-9, (ctl.tau, env.tau())
            ctl.start_at(env.clip.name, env.tau())
            r = readings(env)
            o_env = env._obs()
            o_ctl = ctl.inputs(**r)
            worst_obs = max(worst_obs, float(np.abs(o_env - o_ctl).max()))
            a = ctl.brain.act(o_env)
            tgt = ctl.step(**r)
            env.step(a)
            worst_tgt = max(worst_tgt, float(np.abs(env.data.ctrl - tgt).max()))
            steps += 1
            if env.terminated if hasattr(env, "terminated") else False:
                break
    ok = worst_obs < 1e-6 and worst_tgt < 1e-6
    print(f"{steps} steps over 8 clips (with a walk->trot switch and steering): "
          f"inputs max diff {worst_obs:.2e}, joint targets max diff {worst_tgt:.2e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
