"""Bundle what a web page needs to run Bolt live: the robot, both brains, and a reference run.

    python export_live.py --posture ../models/bolt-dog-posture-dog-2.5M \
                          --walk ../models/bolt-dog-walk-run-8.5M -o bolt_live.json \
                          --reference bolt_live_reference.json

The page (train/gaitlab/live.js) runs the same MuJoCo physics in the browser (the official
WebAssembly build, same version as here) and the same control loop as BoltEnv with the dog
gait, turning and speed control. Each brain is stored as its weights plus its observation
normalizer, so the page needs no machine-learning library: two 256-unit layers are a few
lines of JavaScript.

--reference writes a reference run from Python (stand with the posture brain, then walk,
turn and change speed with the walk brain), so the JavaScript port can be checked step by
step against the real thing (live_check.mjs).
"""
import argparse
import base64
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

import hurdle
from bolt_env import ACTION_SCALE, LIE_POSE, HURDLE_HALF, HURDLE_MARGIN, HURDLE_XML, HURDLE_ZONE, MODEL, PAW_GAIT, STAND_POSE, STANCES, BoltEnv
from evaluate import Policy, reset_standing
from record3d import describe
from reference_gait import HIP_X, HIP_Y

# Game hazards, only in the browser bundle: an ice slab 2 m x 1.4 m, 1 cm thick, whose low friction wins
# over the paws' (higher contact priority). A mocap body, parked out of the way until a game places it.
ICE_XML = ('<body name="ice" mocap="true" pos="100 0 -1"><geom name="ice" type="box" size="1.0 0.7 0.005" '
           'friction="0.15 0.005 0.0001" priority="1" rgba="0.75 0.9 1 1"/></body>')

# The reference run: (command, steps, turn rad/s, speed m/s[, hurdle height placed 1.5 m ahead]): sit, lie,
# walk, step over a hurdle, run, turn sharply, brake, stand
SCRIPT = [("stand", 100, 0.0, 0.4), ("sit", 150, 0.0, 0.4), ("stand", 100, 0.0, 0.4), ("lie", 150, 0.0, 0.4),
          ("stand", 125, 0.0, 0.4), ("walk", 150, 0.0, 0.4), ("walk", 175, 0.0, 0.7, 0.2), ("walk", 150, 0.4, 0.7), ("walk", 150, -0.3, 1.0),
          ("walk", 100, 1.5, 0.3), ("walk", 50, 0.0, 0.2), ("stand", 100, 0.0, 0.2)]


def f32(a):
    """A float array as base64 little-endian float32 (a quarter the size of JSON numbers)."""
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def f64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f8").tobytes()).decode()


def brain(folder):
    model = PPO.load(Path(folder) / "policy.zip", device="cpu")
    with open(Path(folder) / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)  # VecNormalize: (obs - mean) / sqrt(var + eps), clipped to ±clip
    layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    assert all(isinstance(m, torch.nn.ELU) for m in model.policy.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear))
    return dict(inputs=int(model.observation_space.shape[0]),
                mean=f64(norm.obs_rms.mean), var=f64(norm.obs_rms.var), eps=float(norm.epsilon), clip=float(norm.clip_obs),
                layers=[dict(w=f32(m.weight.detach().numpy()), b=f32(m.bias.detach().numpy()),
                             out=int(m.out_features), inp=int(m.in_features)) for m in layers])


def reference(posture, walk):
    """A run in Python with no randomness, recorded step by step for checking the JavaScript port."""
    env = BoltEnv(randomize=False, max_steps=10**6, gait_generator="dog", turning=True, speed_control=True,
                  sharp_turns=walk.sharp_turns, hurdles=True)
    env.forced_command = "stand"
    reset_standing(env, 0)
    d = env.data
    start = dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist())
    steps = []
    for command, n, turn, speed, *hurdle in SCRIPT:
        env.forced_command = command
        env.set_command(command)
        env.set_turn(turn)
        env.set_speed(speed)
        if hurdle:
            env.place_hurdle(1.5, hurdle[0])
        for _ in range(n):
            obs = env._obs()
            action = (walk if command == "walk" else posture).act(obs)
            env.step(action)
            steps.append(dict(obs=obs.tolist(), action=np.asarray(action, dtype=float).tolist(), qpos=d.qpos.tolist()))
    return start, dict(start=start, script=SCRIPT, steps=steps)


def jump_reference(folder, bar=(0.55, 0.15), seed=1):
    """One jump in Python (JumpEnv, no randomness): the start state, the bar, and every step."""
    from jump_env import JumpEnv, JumpPolicy
    policy = JumpPolicy(folder)
    env = JumpEnv(randomize=False, real=policy.real)
    env.exam_bar = bar
    if policy.real:   # a real hurdle, 6 degrees askew and 5 cm off its line, so the new inputs are checked too
        env.exam_askew = (0.1, 0.05)
    obs, _ = env.reset(seed=seed)
    d = env.data
    start = dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(), bar_at=env.hurdle_at.tolist(),
                 bar_dir=env.hurdle_dir.tolist(), bar_h=env.hurdle_h, real=policy.real)
    steps = []
    for _ in range(env.max_steps):
        action = policy.act(obs)
        steps.append(dict(obs=obs.tolist(), action=np.asarray(action, dtype=float).tolist()))
        obs, _, fell, _, info = env.step(action)
        steps[-1]["qpos"] = d.qpos.tolist()
        if fell:
            break
    return dict(start=start, steps=steps, max_steps=env.max_steps, cleared=bool(info["cleared"]), touched=bool(info["touched"]),
                dropped=bool(info.get("dropped", False)))


def getup_reference(folder, goal="selfright", kind="side", seed=1):
    """One get-up in Python (GetUpEnv, no randomness): Bolt dropped on its side and settled for 1 s (the
    start state), then every step of the get-up brain's 4 s. standing: righted (belly down, still) for the
    self-righting brain, standing for a full get-up brain."""
    from getup_env import GetUpEnv, GetUpPolicy
    policy = GetUpPolicy(folder)
    env = GetUpEnv(randomize=False, goal=goal)
    env.exam_start = kind
    obs, _ = env.reset(seed=seed)
    d = env.data
    start = dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(), kind=kind, goal=goal)
    steps, info = [], {}
    for _ in range(env.max_steps):
        action = policy.act(obs)
        steps.append(dict(obs=obs.tolist(), action=np.asarray(action, dtype=float).tolist()))
        obs, _, _, _, info = env.step(action)
        steps[-1]["qpos"] = d.qpos.tolist()
    return dict(start=start, steps=steps, standing=bool(info["standing"]), stood_at=info["stood_at"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posture", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--reference", required=True, help="where to write the Python reference run")
    ap.add_argument("--jump", help="jump brain folder (jump_env.py): the bundle gets it, and the reference a Python jump")
    ap.add_argument("--competitor", action="append", default=[], metavar="NAME=FOLDER",
                    help="more walk brains for the agility competition (speed-control brains)")
    ap.add_argument("--getup", help="get-up brain folder (getup_env.py): the bundle gets it, and the reference a Python get-up")
    ap.add_argument("--getup-goal", default="selfright", choices=["selfright", "stand"],
                    help="what the get-up brain was trained for: roll onto the belly (then the posture brain stands), or stand")
    args = ap.parse_args()
    walk = Policy(args.walk)
    start, ref = reference(Policy(args.posture), walk)
    env = BoltEnv(randomize=False, gait_generator="dog", turning=True, speed_control=True, sharp_turns=walk.sharp_turns,
                  hurdles=True)
    # the old board hurdle (for the reference run), the ice, and 4 real agility hurdles (hurdle.py) for the games
    xml = hurdle.world_xml(MODEL.read_text(), n=4, extra_before_floor=HURDLE_XML + '\n    ' + ICE_XML)
    # the page draws each part at its geom's pose by id, so the ids must come from the model the page runs
    # (the ice slab, added for the games, shifted every id by one and drew Bolt's body on the floor)
    page_model = mujoco.MjModel.from_xml_string(xml)
    bundle = dict(   # the robot plus one hurdle, which the page moves to whichever bar comes next
        xml=xml,
        geoms=describe(page_model),
        constants=dict(stand_pose=STAND_POSE.tolist(), dog_pose=STANCES["dog"].tolist(), action_scale=ACTION_SCALE.tolist(),
                       residual_scale=env.RESIDUAL_SCALE, frame_skip=env.frame_skip, dt=env.dt, paw_gait=PAW_GAIT,
                       turn_gain=env.TURN_GAIN, max_turn=env.MAX_TURN, max_speed=env.MAX_SPEED, speed_ref=env.SPEED_REF,
                       sharp_turns=env.sharp_turns, sharp_turn=env.SHARP_TURN,
                       hurdle_margin=HURDLE_MARGIN, hurdle_zone=list(HURDLE_ZONE), hurdle_half=HURDLE_HALF,
                       hip_x=HIP_X.tolist(), hip_y=HIP_Y.tolist(), real_hurdle=hurdle.constants(),
                       commands=["stand", "sit", "lie", "walk"]),
        brains=dict(posture=dict(brain(args.posture), source=Path(args.posture).name),
                    walk=dict(brain(args.walk), source=Path(args.walk).name)),
        start=start,  # Bolt standing, as BoltEnv.reset leaves it (seed 0, no randomness)
    )
    if args.competitor:   # agility competitors: name -> walk brain and the gait it turns with
        bundle["competitors"] = {name: dict(brain(path), sharp_turns=Policy(path).sharp_turns, source=Path(path).name)
                                 for name, path in (c.split("=", 1) for c in args.competitor)}
    if args.jump:   # a real jump, from standing: its own brain (jump_env.py), 2 s per jump
        bundle["brains"]["jump"] = dict(brain(args.jump), source=Path(args.jump).name)
        bundle["constants"]["jump_steps"] = 100
        ref["jump"] = jump_reference(args.jump)
    if args.getup:   # getting up after a fall by itself: its own brain (getup_env.py), 4 s per try in training
        bundle["brains"]["getup"] = dict(brain(args.getup), source=Path(args.getup).name, goal=args.getup_goal)
        bundle["constants"]["getup_steps"] = 200
        # its motor range per joint: the self-righting brain uses each joint's full range around the dog stance
        # (GetUpEnv.scale), a stand brain ACTION_SCALE
        from getup_env import GetUpEnv
        bundle["constants"]["getup_scale"] = np.asarray(GetUpEnv(randomize=False, goal=args.getup_goal).scale, dtype=float).tolist()
        bundle["constants"]["lie_pose"] = LIE_POSE.tolist()   # where the legs tuck before the posture brain stands
        ref["getup"] = getup_reference(args.getup, args.getup_goal)
    Path(args.out).write_text(json.dumps(bundle, separators=(",", ":")))
    Path(args.reference).write_text(json.dumps(ref, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) and {args.reference} ({len(ref['steps'])} steps)")


if __name__ == "__main__":
    main()
