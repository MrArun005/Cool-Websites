"""Sign check for JumpEnv's reward (LESSONS §4: check the sign of every penalty).

    python duckbody/jump_check.py

Scripted through JumpEnv.step (the policy's own action interface, nominal body):
  stand        hold the home pose for 3 s
  hop-hard     crouch → push → tuck into the home pose and land on it
  hop-soft     same push and tuck, then legs reach long for the floor and yield into a crouch
  drop-hard    released 3 cm above the floor, legs held in the home pose
  drop-soft    same drop, legs reach long and yield on touchdown
The soft version of each must score higher than the hard one, and hopping higher than standing.
"""
import numpy as np

from jump_env import JumpEnv
from jump_probe import HOME, leg_action

MODEL = __import__("pathlib").Path(__file__).resolve().parent / "model" / "duck_mid.xml"
pose = lambda k, hr, ao=0.0: (-hr * k, k, -(k - hr * k) + ao)
CROUCH, PUSH, SOFT, REACH = pose(1.75, 0.46), pose(0.6, 0.6, -0.5), pose(1.75, 0.46), pose(0.9, 0.46)


def episode(env, ctrl, seconds=3.0, drop=None):
    env.reset(seed=0)
    if drop is not None:                                   # release just above the floor
        d = env.data
        d.qpos[2] += drop
        import mujoco
        mujoco.mj_forward(env.model, d)
        mujoco.mj_subtreeVel(env.model, d)
        env.com_vz = float(d.subtree_linvel[1][2])
    total, shock, peak, air, hops, landed = 0.0, 0.0, 1.0, 0, 0, None
    for t in range(int(seconds * 50)):
        a, landed = ctrl(env, t, landed)
        _, r, term, _, info = env.step(a)
        total += r
        shock += info["shock"]
        peak = max(peak, info["fw"])
        if term:
            break
    return dict(reward=round(total, 2), shock=round(shock, 2), peak_F_over_W=round(peak, 2), hops=env.hops,
                best_rise_cm=round(env.best_hop * 100, 2), up=round(env.posture()["up"], 2), fell=bool(term))


def hopper(soft):
    """One hop per second: crouch 0.3 s, push 0.14 s, tuck the legs 0.04 s (the feet leave the floor).
    Hard: stay tucked in the home pose and land on it. Soft: reach long for the floor, then yield
    into a crouch over 0.2 s after touchdown and stand back up."""
    def ctrl(env, t, landed):
        k = t % 50
        if k < 15:
            return leg_action(env, *CROUCH), None
        if k < 22:
            return leg_action(env, *PUSH), None
        if k < 24 or not soft:
            return leg_action(env, *HOME), None
        return land_softly(env, t, landed)
    return ctrl


def land_softly(env, t, landed):
    if landed is None and env.feet_on_ground().sum() > 0:
        landed = t
    if landed is None:
        return leg_action(env, *REACH), None
    f = min(1.0, (t - landed) / 10)
    target = np.array(REACH) + f * (np.array(SOFT) - np.array(REACH)) if t - landed < 20 else np.array(HOME)
    return leg_action(env, *target), landed


def dropper(soft):
    """Released 3 cm above the floor. Hard: legs held in the home pose. Soft: reach, then yield."""
    def ctrl(env, t, landed):
        if not soft:
            return leg_action(env, *HOME), None
        return land_softly(env, t, landed)
    return ctrl


def main():
    env = JumpEnv(task="jump", randomize=False, model_path=str(MODEL), practice=False)
    rows = {
        "stand": episode(env, lambda e, t, l: (leg_action(e, *HOME), None)),
        "hop-hard": episode(env, hopper(False)),
        "hop-soft": episode(env, hopper(True)),
        "drop-hard": episode(env, dropper(False), seconds=1.5, drop=0.03),
        "drop-soft": episode(env, dropper(True), seconds=1.5, drop=0.03),
    }
    for k, v in rows.items():
        print(f"{k:10s} {v}")
    ok = rows["hop-soft"]["reward"] > rows["hop-hard"]["reward"] and rows["drop-soft"]["reward"] > rows["drop-hard"]["reward"]
    print("soft landings score higher than hard ones:", ok)
    print("hopping scores higher than standing:", max(rows["hop-soft"]["reward"], rows["hop-hard"]["reward"]) > rows["stand"]["reward"])


if __name__ == "__main__":
    main()
