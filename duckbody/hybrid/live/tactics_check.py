"""Records world models and tactics.py's commands for the JavaScript port check (tactics_check.mjs).

A cheap kinematic stand-in for the match drives the inputs (no MuJoCo): each player moves by the twist tactics.py
commands (velstand's speed map, roughly), a kick sends the ball along the kicker's heading, the ball rolls and
slows down, phases go ready -> set -> playing with set pieces now and then, players fall and get up, opponents and
the ball estimate get noise. Only the inputs matter: both implementations see exactly the same world models.
The random numbers inside the tactics (support-run jitter, detour side) come from the same Park-Miller generator on
both sides.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/live/tactics_check.py OUT.json [seconds=120]
    node duckbody/hybrid/live/tactics_check.mjs OUT.json
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tactics as TT  # noqa: E402


class ParkMiller:
    """Deterministic stand-in for random.Random (same sequence in tactics_check.mjs)."""

    def __init__(self, seed=0):
        s = int(seed * 1000003) if isinstance(seed, float) else int(seed)
        self.s = s % 2147483646 + 1

    def random(self):
        self.s = (self.s * 16807) % 2147483647
        return self.s / 2147483647

    def uniform(self, a, b):
        return a + (b - a) * self.random()

    def choice(self, seq):
        return seq[min(len(seq) - 1, int(self.random() * len(seq)))]


TT.random.Random = ParkMiller

ROLES = ("GK", "D1", "D2", "A1", "A2")
SPOTS = {True: {"GK": (-TT.HX + 0.3, 0.0), "D1": (-1.3, 0.6), "D2": (-1.3, -0.6), "A1": (-0.22, 0.03), "A2": (-0.7, -0.75)},
         False: {"GK": (-TT.HX + 0.3, 0.0), "D1": (-1.3, 0.6), "D2": (-1.3, -0.6), "A1": (-0.6, 0.0), "A2": (-0.75, 0.75)}}


def plain(x):
    if isinstance(x, np.ndarray):
        return [float(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def cmd_dict(c):
    return dict(vx=float(c.vx), vy=float(c.vy), wz=float(c.wz), head=None if c.head is None else [None if v is None else float(v) for v in c.head],
                kick=c.kick, kind=c.kind, mate=c.mate, cam=bool(c.cam), state=c.state, role=c.role)


def main():
    out = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    rng = np.random.default_rng(7)
    teams = {"blue": 1.0, "yellow": -1.0}
    tac = {"blue": TT.TeamTactics(1), "yellow": TT.TeamTactics(2)}
    P = {}
    for tn, d in teams.items():
        for i, r in enumerate(ROLES):
            s = np.array(SPOTS[tn == "blue"][r]) * d + rng.normal(0, 0.3, 2)
            P[f"{tn[0]}{i + 1}"] = dict(team=tn, role=r, pos=s, psi=0.0 if d > 0 else math.pi, down_until=-1.0, busy_until=-1.0,
                                        active=True)
    ball, bvel = np.zeros(2), np.zeros(2)
    phase, phase_t, setpiece = "ready", 0.0, None
    ticks = []
    for k in range(int(seconds / TT.DT)):
        t = k * TT.DT
        if phase == "ready" and t - phase_t > 3.0:
            phase, phase_t = "set", t
        elif phase == "set" and t - phase_t > 1.0:
            phase, phase_t = "playing", t
            setpiece = dict(kind="kick-off", team="blue", ball=[0.0, 0.0], radius=0.375, until=t + 5.0)
        elif phase == "playing":
            if setpiece is not None and (t > setpiece["until"] or np.linalg.norm(ball - setpiece["ball"]) > 0.15):
                setpiece = None
            if abs(ball[0]) > TT.HX or abs(ball[1]) > TT.HY:          # out / goal: a throw-in-like restart
                ball = np.clip(ball, [-TT.HX + 0.1, -TT.HY + 0.05], [TT.HX - 0.1, TT.HY - 0.05])
                bvel[:] = 0
                setpiece = dict(kind="throw-in", team="yellow" if rng.random() < 0.5 else "blue", ball=ball.tolist(), radius=0.5,
                                until=t + 4.0)
            if k % 1500 == 1499:
                phase, phase_t = "goal", t
        elif phase == "goal" and t - phase_t > 2.0:
            phase, phase_t = "ready", t
        for name, p in P.items():                                     # random falls, a player taken off now and then
            if rng.random() < 0.0015:
                p["down_until"] = t + rng.uniform(0.5, 3.0)
            if rng.random() < 0.0003:
                p["active"] = not p["active"]
        wms = {}
        for tn, d in teams.items():
            players, opps = [], []
            for name, p in P.items():
                if p["team"] == tn:
                    rel = ball - p["pos"]
                    own = np.linalg.norm(rel) < 1.5
                    players.append(dict(name=name, role=p["role"], pos=p["pos"] + rng.normal(0, 0.02, 2), psi=p["psi"] + rng.normal(0, 0.02),
                                        active=p["active"], down=t < p["down_until"], busy=t < p["busy_until"],
                                        ball=(ball + rng.normal(0, 0.01, 2)) if own else None, ball_src="own" if own else "team",
                                        ball_age=0.05 if own else 1.0))
                else:
                    opps.append(dict(pos=p["pos"] + rng.normal(0, 0.06, 2), role=p["role"], active=p["active"]))
            sp = None if setpiece is None else dict(kind=setpiece["kind"], ours=setpiece["team"] == tn, ball=setpiece["ball"],
                                                      radius=setpiece["radius"])
            wms[tn] = dict(t=t, phase=phase if phase in ("playing", "ready", "set") else "other", dir=d,
                           ball=ball + rng.normal(0, 0.02, 2), setpiece=sp,
                           spots={n: (np.array(SPOTS[tn == "blue"][q["role"]]) * d).tolist() for n, q in P.items() if q["team"] == tn},
                           players=players, opps=opps)
        rec = dict(wm={tn: plain(w) for tn, w in wms.items()}, cmds={})
        for tn in teams:
            cmds = tac[tn].step(wms[tn])
            rec["cmds"][tn] = {n: cmd_dict(c) for n, c in cmds.items()}
            for n, c in cmds.items():                                 # the stand-in physics
                p = P[n]
                if t < p["down_until"] or t < p["busy_until"] or not p["active"]:
                    continue
                if c.kick:
                    p["busy_until"] = t + 0.5
                    rel = ball - p["pos"]
                    if np.linalg.norm(rel) < 0.2:
                        bvel = 1.2 * np.array([math.cos(p["psi"]), math.sin(p["psi"])]) + rng.normal(0, 0.1, 2)
                    continue
                v = 0.42 * np.clip(c.vx, -1, 1) if abs(c.vx) >= 0.3 else 0.0
                vy = 0.15 * np.clip(c.vy, -1, 1) if abs(c.vy) >= 0.3 else 0.0
                cs, sn = math.cos(p["psi"]), math.sin(p["psi"])
                p["pos"] = p["pos"] + TT.DT * np.array([cs * v - sn * vy, sn * v + cs * vy])
                p["psi"] = TT.wrap(p["psi"] + TT.DT * (c.wz if abs(c.wz) >= 0.5 else 0.0))
                if np.linalg.norm(ball - p["pos"]) < 0.08 and v > 0:  # walking into the ball pushes it
                    bvel = bvel + 0.3 * np.array([cs, sn])
        ball = ball + bvel * TT.DT
        bvel = bvel * (1 - 0.6 * TT.DT)
        ticks.append(rec)
    out.write_text(json.dumps(dict(ticks=ticks, seeds={"blue": 1, "yellow": 2})))
    kicks = sum(1 for r in ticks for tn in r["cmds"] for c in r["cmds"][tn].values() if c["kick"])
    states = sorted({c["state"] for r in ticks for tn in r["cmds"] for c in r["cmds"][tn].values()})
    print(f"{len(ticks)} ticks, {kicks} kicks; states seen: {states}")


if __name__ == "__main__":
    main()
