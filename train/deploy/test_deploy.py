"""Gate for PLAN.md phase 1 steps 3-4: the robot-side controller builds the same 75 inputs and sends the same joint
targets as imitate_env.ImitateEnv does in training, for the same robot state.

    cd train && BOLT_BODY=go2 MUJOCO_GL=osmesa python deploy/test_deploy.py [../models/bolt-go2-damp-798M/params.pkl]
    ... test_deploy.py <brain> --save     # + record deploy/golden_<brain folder>.npz (the reference below)
    ... test_deploy.py <brain> --golden   # torch-free: replay that record through the controller (the robot's machine)

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

from controller import MOTION_CLIPS, Go2Controller  # noqa: E402


def readings(env):
    """What the robot side gets: IMU quaternion (w, x, y, z) and body-frame gyro, joints; plus the two estimates."""
    d = env.data
    R = env._rotation()
    quat = np.zeros(4)   # the env's own body rotation (xmat, from the last physics substep's kinematics: up to 2 ms
    mujoco.mju_mat2Quat(quat, R.reshape(-1))   # older than qpos; far below the trained IMU noise, but exact here)
    return dict(quat=quat, gyro=d.qvel[3:6].copy(), q=d.qpos[7:19].copy(), dq=d.qvel[6:18].copy(),
                v_body=R.T @ d.qvel[0:3], height=float(d.qpos[2]))


def golden_path(brain):
    return HERE / f"golden_{Path(brain).parent.name}.npz"


def check_golden(brain):
    """Torch-free parity (the robot's machine): replay the recorded steps through the controller and compare its inputs
    and joint targets with what the training env produced (recorded by a full run of this test, --save)."""
    g = np.load(golden_path(brain))
    ctl = Go2Controller(brain)
    worst_obs = worst_tgt = 0.0
    for i in range(len(g["tau"])):
        ctl.start_at(str(g["clip"][i]), float(g["tau"][i]))
        ctl.steer, ctl.last_action[:] = float(g["steer"][i]), g["last_action"][i]
        r = {k: (float(g[k][i]) if k == "height" else g[k][i]) for k in ("quat", "gyro", "q", "dq", "v_body", "height")}
        o = ctl.inputs(**r)
        if len(o) != len(g["obs"][i]):   # recorded with fewer clip slots (9, before the jump clip): compare the old ones
            k = len(o) - len(g["obs"][i])
            o = np.concatenate([o[:48 + 9], o[48 + 9 + k:]])
        worst_obs = max(worst_obs, float(np.abs(o - g["obs"][i]).max()))
        worst_tgt = max(worst_tgt, float(np.abs(ctl.step(**r) - g["target"][i]).max()))
    sw = g["switch"]   # the walk -> trot switch rule: same clip time as the env picked
    ctl.start_at("walk", float(sw[0]))
    ctl.command("trot", q=g["switch_q"])
    worst_tgt = max(worst_tgt, abs(ctl.tau - float(sw[1])))
    ok = worst_obs < 1e-5 and worst_tgt < 1e-5
    print(f"{len(g['tau'])} recorded steps (no torch): inputs max diff {worst_obs:.2e}, joint targets max diff "
          f"{worst_tgt:.2e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def main():
    brain = next((x for x in sys.argv[1:] if not x.startswith("--")),
                 str(HERE.parent.parent / "models" / "bolt-go2-damp-798M" / "params.pkl"))
    if "--golden" in sys.argv:
        return check_golden(brain)
    from imitate_env import ImitateEnv
    ctl = Go2Controller(brain)
    env = ImitateEnv(randomize=False, seed=3, clips=MOTION_CLIPS, body="go2")
    rec = {k: [] for k in ("clip", "tau", "steer", "last_action", "quat", "gyro", "q", "dq", "v_body", "height", "obs",
                           "target")}
    switch = None
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
                switch = [env.tau(), None, env.data.qpos[7:19].copy()]
                ctl.command("trot", q=env.data.qpos[7:19])
                env._switch_to(env.clips["trot"])
                switch[1] = env.tau()
                assert ctl.clip.name == "trot" and abs(ctl.tau - env.tau()) < 1e-9, (ctl.tau, env.tau())
            ctl.start_at(env.clip.name, env.tau())
            r = readings(env)
            o_env = env._obs()
            o_ctl = ctl.inputs(**r)
            worst_obs = max(worst_obs, float(np.abs(o_env - o_ctl).max()))
            a = ctl.brain.act(o_env)
            for k, x in (("clip", ctl.clip.name), ("tau", ctl.tau), ("steer", ctl.steer),
                         ("last_action", ctl.last_action.copy()), ("obs", o_env), *r.items()):
                rec[k].append(x)
            tgt = ctl.step(**r)
            env.step(a)
            worst_tgt = max(worst_tgt, float(np.abs(env.data.ctrl - tgt).max()))
            rec["target"].append(env.data.ctrl.copy())
            steps += 1
            if env.terminated if hasattr(env, "terminated") else False:
                break
    ok = worst_obs < 1e-6 and worst_tgt < 1e-6
    print(f"{steps} steps over 8 clips (with a walk->trot switch and steering): "
          f"inputs max diff {worst_obs:.2e}, joint targets max diff {worst_tgt:.2e} -> {'PASS' if ok else 'FAIL'}")
    if ok and "--save" in sys.argv:   # the reference for the torch-free check (--golden) on the robot's machine
        np.savez(golden_path(brain), **{k: np.array(v) for k, v in rec.items()}, switch=np.array(switch[:2]),
                 switch_q=switch[2])
        print(f"saved {golden_path(brain).name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
