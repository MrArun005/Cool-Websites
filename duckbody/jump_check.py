"""Sign check for JumpEnv's reward (LESSONS §4: check the sign of every penalty).

    python duckbody/jump_check.py

Scripted through JumpEnv.step (the policy's own action interface, nominal body):
  stand        hold the home pose for 3 s
  bounce-4Hz   crouch-push-home every 0.24 s (the v1/v2 brains' frantic bounce)
  hop-hard     crouch → push → tuck into the home pose and land on it
  hop-soft     same push and tuck, then legs reach long for the floor and yield into a crouch
  drop-hard    released 3 cm above the floor, legs held in the home pose
  drop-soft    same drop, legs reach long and yield on touchdown
  drop2-*      the same from 1.5 cm (about the height a hop reaches)
Run with and without the servo back-EMF limit (JumpEnv.back_emf).
Of each pair, the one landing with the lower peak force must score higher; hopping higher than standing.
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
                best_rise_cm=round(env.best_hop * 100, 2), early=env.early_hops, up=round(env.posture()["up"], 2), fell=bool(term))


def bouncer(env, t, landed):
    """Fast bouncing, ~4 hops/s: crouch 0.06 s, push 0.1 s, home 0.08 s, again (v1/v2 brains did this)."""
    k = t % 12
    return leg_action(env, *(CROUCH if k < 3 else PUSH if k < 8 else HOME)), None


def hopper(soft, period=60):
    """One hop per 1.2 s: crouch 0.3 s, push 0.14 s, tuck the legs 0.04 s (the feet leave the floor).
    Hard: stay tucked in the home pose and land on it. Soft: reach long for the floor, then yield
    into a crouch over 0.2 s after touchdown and stand back up."""
    def ctrl(env, t, landed):
        k = t % period
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
    for back_emf in (False, True):
        print(f"--- servos {'with' if back_emf else 'without'} the back-EMF torque–speed limit")
        check(back_emf)


def check(back_emf):
    env = JumpEnv(task="jump", randomize=False, model_path=str(MODEL), practice=False)
    env.back_emf = back_emf
    rows = {
        "stand": episode(env, lambda e, t, l: (leg_action(e, *HOME), None)),
        "hop-hard": episode(env, hopper(False)),
        "hop-soft": episode(env, hopper(True)),
        "bounce-4Hz": episode(env, bouncer),
        "drop-hard": episode(env, dropper(False), seconds=1.5, drop=0.03),
        "drop-soft": episode(env, dropper(True), seconds=1.5, drop=0.03),
        "drop2-hard": episode(env, dropper(False), seconds=1.5, drop=0.015),
        "drop2-soft": episode(env, dropper(True), seconds=1.5, drop=0.015),
    }
    for k, v in rows.items():
        print(f"{k:10s} {v}")
    # The scripted "soft" landing is not always the softer one (from 1.5 cm reaching long lands harder),
    # so judge each pair by what was measured: the landing with the lower peak force must score higher.
    ok = True
    for k in ("hop", "drop", "drop2"):
        a, b = rows[k + "-soft"], rows[k + "-hard"]
        if a["fell"] or b["fell"] or a["hops"] != b["hops"] or abs(a["peak_F_over_W"] - b["peak_F_over_W"]) < 0.1:
            continue
        soft, hard = (a, b) if a["peak_F_over_W"] < b["peak_F_over_W"] else (b, a)
        ok &= soft["reward"] > hard["reward"]
    print("the softer landing of each pair scores higher:", ok)
    print("slow hops score higher than 4 Hz bouncing:", max(rows["hop-soft"]["reward"], rows["hop-hard"]["reward"]) > rows["bounce-4Hz"]["reward"])
    print("hopping scores higher than standing:", max(rows["hop-soft"]["reward"], rows["hop-hard"]["reward"]) > rows["stand"]["reward"])


if __name__ == "__main__":
    main()
