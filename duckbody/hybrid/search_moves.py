"""Cross-entropy searches for the hybrid's scripted moves, scored on several randomized bodies at once so a
one-body fluke can't win (fitness = mean - 0.5 * std over the bodies). Writes moves_hybrid.json.

    /home/user/cool-websites/.venv/bin/python duckbody/hybrid/search_moves.py hop [--gens 12] [--procs 3]
    /home/user/cool-websites/.venv/bin/python duckbody/hybrid/search_moves.py swag | lie_down

Bodies: harness.MicroduckSim(randomize=True), seeds 1000 + i (not the 0-9 used by test_skills_hybrid.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import moves_hybrid as mv  # noqa: E402
from harness import DEFAULT_POSE, MicroduckSim, Policy  # noqa: E402

import runtime_hybrid as rh  # noqa: E402

SCENE = str(rh.SCENE)          # the default body (HYBRID_BODY picks another)
MOVES = HERE / "moves_hybrid.json"
_SIM, _POL = {}, {}


def sim_for(seed):
    if "sim" not in _SIM:
        _SIM["sim"] = MicroduckSim(SCENE, seed=0, randomize=True)
    s = _SIM["sim"]
    s.rng = np.random.default_rng(1000 + seed)
    return s


def pol(name):
    if name not in _POL:
        _POL[name] = Policy(name)
    return _POL[name]


def floor_contacts(m, d):
    """(touching floor?, total normal force N)"""
    tot, touching, f = 0.0, False, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0:
            touching = True
            mujoco.mj_contactForce(m, d, i, f)
            tot += abs(f[0])
    return touching, tot


def upright(sim):
    return sim.tilt_deg() < 25 and sim.trunk_z() > 0.10


# ------------------------------------------------------------------ hop
HOP_SPEC = dict(hc=(0.03, 0.065), Tc=(0.2, 0.8), lean=(-0.3, 0.3), neck_c=(-0.6, 0.6), neck_p=(-0.8, 0.4),
                push_ank=(-0.2, 0.8), push_hip=(-0.4, 0.4), release=(0.0, 1.2), Tp_max=(0.08, 0.25),
                h_air=(0.0, 0.06), air_ank=(-0.4, 0.4), Ta=(0.02, 0.15), ht=(0.0, 0.05), kp=(1.0, 5.0), kd=(0.0, 0.3))
HOP_BASE = dict(hc=0.054, Tc=0.5, lean=0.0, neck_c=0.0, neck_p=-0.3, push_ank=0.22, push_hip=0.0, release=0.5,
                Tp_max=0.15, h_air=0.04, air_ank=0.0, Ta=0.06, ht=0.02, Tland=0.6, kp=2.6, kd=0.2)


def hop_trial(P, seed, record=None, after=100):
    sim = sim_for(seed)
    sim.place("standing")
    vel = pol("velstand")
    for _ in range(40):
        sim.step(vel)
    m, d = sim.m, sim.d
    W = m.body_subtreemass[1] * 9.81
    st = dict(air=0, maxair=0, fmax=0.0, landed=False, fland=0.0)

    def sub(_):
        touching, tot = floor_contacts(m, d)
        st["fmax"] = max(st["fmax"], tot / W)
        if touching:
            if st["air"] >= 4:                          # >= 20 ms flight, then the landing force
                st["landed"] = True
            st["air"] = 0
        else:
            st["air"] += 1
            st["maxair"] = max(st["maxair"], st["air"])
        if st["landed"]:
            st["fland"] = max(st["fland"], tot / W)

    hop = mv.Hop(P)
    hop.begin(None)
    tmax, k = 0.0, 0
    while not hop.done() and k < int(hop.duration / 0.02) + 1:
        sim.step(hop, on_substep=sub)
        tmax = max(tmax, sim.tilt_deg())
        if record is not None and k % 2 == 0:
            record.append(sim.frame())
        k += 1
    for t in range(after):
        sim.step(vel)
        tmax = max(tmax, sim.tilt_deg())
        if record is not None and (k + t) % 2 == 0:
            record.append(sim.frame())
    ok = upright(sim) and tmax < 45
    return dict(ok=bool(ok), flight_ms=st["maxair"] * 5, impact_bw=round(st["fmax"], 2), land_bw=round(st["fland"], 2),
                tilt_max=round(tmax, 1))


def hop_score(P, seed):
    r = hop_trial(P, seed)
    if not r["ok"]:
        return -40.0, r
    s = min(r["flight_ms"], 80) - 0.5 * max(0, r["tilt_max"] - 15) - 4 * max(0, r["impact_bw"] - 4.0)
    return s, r


# ------------------------------------------------------------------ swag sit
SWAG_SPEC = dict(roll=(0.0, 0.35), yaw=(0.0, 0.4), knee=(0.0, 1.5), hip=(-0.8, 0.4), ank=(-0.6, 0.6), rknee=(-0.5, 0.5),
                 lb=(-0.4, 0.4), ramp=(0.8, 2.0))
SWAG_BASE = dict(roll=0.25, yaw=0.3, knee=0.8, hip=-0.3, ank=0.0, rknee=0.0, lb=0.0, ramp=1.2)
SWAG_HEAD = (0.15, -0.25, 0.45, 0.25)


def swag_offsets(P):
    """Action offsets on top of alpha_sitstand (seated): legs splayed (hip roll, toes out), left knee opened with
    its hip and ankle, right knee, both hips for a lean-back."""
    o = np.zeros(14, np.float32)
    o[1] += P["roll"]; o[10] -= P["roll"]
    o[0] += P["yaw"]; o[9] -= P["yaw"]
    o[mv.L_KNEE] -= P["knee"]; o[mv.L_HIP] += P["hip"] + P["lb"]; o[mv.L_ANK] += P["ank"]
    o[mv.R_KNEE] += P["rknee"]; o[mv.R_HIP] -= P["lb"]
    return o


def swag_trial(P, seed, record=None):
    sim = sim_for(seed)
    sim.place("standing")
    vel, sp = pol("velstand"), pol("alpha_sitstand")
    head = np.array(SWAG_HEAD, np.float32)

    def step(brain, sit, h=0.0):
        sim.command[:] = 0
        sim.command[0] = sit
        sim.command[3:7] = head * h
        sim.step(brain)
        if record is not None:
            record.append(sim.frame())

    for _ in range(25):
        step(vel, 0.0)
    for _ in range(125):
        step(sp, 1.0)
    knee0 = float(sim.d.qpos[sim.qadr[mv.L_KNEE]])
    r = mv.Residual(sp, swag_offsets(P), P["ramp"])
    tmax, pitches = 0.0, []
    n = int((P["ramp"] + 6.0) / 0.02)                # hold 6 s: a slow lean-back drift must show up here
    for t in range(n):
        step(r, 1.0, min(1.0, t / 40))
        tmax = max(tmax, sim.tilt_deg())
        pitches.append(math.degrees(mv.pitch(sim.obs())))
    lean = -float(np.mean(pitches[-50:]))          # + = leaning back
    lean_max = -float(np.min(pitches))
    knee_open = knee0 - float(sim.d.qpos[sim.qadr[mv.L_KNEE]])
    still = float(np.linalg.norm(sim.d.qvel[:6]))
    seated = sim.trunk_z() < 0.09 and tmax < 40 and lean_max < 15
    ro = mv.Residual(sp, swag_offsets(P), 1.0, out=True)
    for t in range(50):
        step(ro, 1.0, 1 - t / 50)
    for _ in range(125):
        step(sp, 0.0)
    t2 = 0.0
    for _ in range(150):
        step(vel, 0.0)
        t2 = max(t2, sim.tilt_deg())
    stood = upright(sim) and t2 < 45
    return dict(seated=bool(seated), stood=bool(stood), lean_back_deg=round(lean, 1), lean_max=round(lean_max, 1),
                knee_open=round(knee_open, 2),
                tilt_max=round(tmax, 1), still=round(still, 3))


def swag_score(P, seed):
    r = swag_trial(P, seed)
    if not (r["seated"] and r["stood"]):
        return -30.0 + 10 * r["seated"], r
    return (8 * min(r["knee_open"], 1.0) + 0.3 * min(max(r["lean_back_deg"], 0), 8) + 4 * P["roll"] + 3 * P["yaw"]
            - 10 * r["still"] - 0.5 * max(0, r["lean_max"] - 8)), r


# ------------------------------------------------------------------ lie down (on the belly) and get up again
LIE_SPEC = dict(h1=(0.0, 0.06), dank1=(-0.6, 0.6), dhip1=(-0.8, 0.8), neck1=(-1.2, 0.6), hp1=(-1.0, 1.0), T1=(0.5, 2.5),
                b1=(0.0, 1.0), h2=(0.0, 0.06), dhip2=(-1.0, 1.0), dank2=(-0.6, 0.6), neck2=(-1.2, 0.6), T2=(0.5, 2.0))
LIE_BASE = dict(h1=0.05, dank1=-0.5, dhip1=0.0, neck1=0.3, hp1=0.3, T1=1.5, b1=0.0, h2=0.05, dhip2=0.3, dank2=0.0,
                neck2=-0.3, T2=1.0, squat=0.05)


def lie_keys(P):
    k1 = mv.legs(DEFAULT_POSE.copy(), P["h1"], P["dank1"], P["dhip1"])
    k1[mv.NECK] += P["neck1"]
    k1[mv.HEAD_PITCH] += P["hp1"]
    k2 = mv.legs(DEFAULT_POSE.copy(), P["h2"], P["dank2"], P["dhip2"])
    k2[mv.NECK] += P["neck2"]
    k2[mv.HEAD_PITCH] += P["hp1"]
    return [k1, k2], [P["T1"], P["T2"]], [P["b1"], 0.0]


def lie_trial(P, seed, record=None):
    sim = sim_for(seed)
    sim.place("standing")
    vel = pol("velstand")
    m, d = sim.m, sim.d
    W = m.body_subtreemass[1] * 9.81
    frames = record if record is not None else []
    st = dict(f=0.0)

    def sub(_):
        st["f"] = max(st["f"], floor_contacts(m, d)[1] / W)

    def step(brain, watch=False):
        sim.command[:] = 0
        sim.step(brain, on_substep=sub if watch else None)
        if record is not None:
            frames.append(sim.frame())

    for _ in range(25):
        step(vel)
    sq = mv.Squat(P["squat"], down=1.0)
    sq.begin(None)
    for _ in range(60):
        step(sq)
    keys, durs, bal = lie_keys(P)
    kf = mv.Keyframes(keys, durs, bal, head=False)
    kf.begin(None)
    kf.q0 = DEFAULT_POSE + sim.last_action
    a = m.sensor("imu_accel").adr[0]
    peak_g = 0.0
    for _ in range(int((sum(durs) + 1.5) / 0.02)):
        step(kf, watch=True)
        peak_g = max(peak_g, float(np.linalg.norm(d.sensordata[a:a + 3])) / 9.81)
    g = sim.gravity()
    # lying on the belly: this head is 14 cm long, so the duck rests on bill, chest and knees at ~45-60 deg
    lying = g[0] > 0.6 and sim.trunk_z() < 0.08
    still = float(np.linalg.norm(d.qvel[:6]))
    t_up = None
    for t in range(200):
        step(vel)
        if t_up is None and upright(sim):
            t_up = (t + 1) * 0.02
    stood = upright(sim)
    return dict(lying=bool(lying), stood=bool(stood), peak_bw=round(st["f"], 2), peak_g=round(peak_g, 2), still=round(still, 3),
                t_up=t_up, gx=round(float(g[0]), 2))


def lie_score(P, seed):
    r = lie_trial(P, seed)
    if not (r["lying"] and r["stood"]):
        return -30.0 + 10 * r["lying"], r
    return 20 - 3 * r["peak_bw"] - 2 * r["peak_g"] - 20 * r["still"] + 10 * min(r["gx"], 0.9), r


# ------------------------------------------------------------------ CEM
def cem(score_fn, spec, base, gens=12, pop=24, elite=6, seeds=(0, 1, 2, 3), procs=3, log=print, rng_seed=0):
    keys = list(spec)
    lo, hi = np.array([spec[k][0] for k in keys]), np.array([spec[k][1] for k in keys])
    mu = np.clip(np.array([base[k] for k in keys], float), lo, hi)
    sd = (hi - lo) / 4
    rng = np.random.default_rng(rng_seed)
    best = (-1e9, None, None)
    with Pool(procs) as pool:
        for g in range(gens):
            X = np.clip(mu + sd * rng.standard_normal((pop, len(keys))), lo, hi)
            X[0] = mu
            if best[1] is not None:
                X[1] = [best[1][k] for k in keys]
            jobs = [(dict(base, **dict(zip(keys, map(float, x)))), s) for x in X for s in seeds]
            res = pool.starmap(score_fn, jobs)
            S = np.array([[res[i * len(seeds) + j][0] for j in range(len(seeds))] for i in range(pop)])
            fit = S.mean(1) - 0.5 * S.std(1)
            order = np.argsort(-fit)
            E = X[order[:elite]]
            mu, sd = E.mean(0), np.maximum(E.std(0), (hi - lo) * 0.02)
            i0 = order[0]
            if fit[i0] > best[0]:
                best = (float(fit[i0]), dict(base, **dict(zip(keys, map(float, X[i0])))),
                        [res[i0 * len(seeds) + j][1] for j in range(len(seeds))])
            log(f"gen {g}: best {fit[i0]:.1f} (so far {best[0]:.1f}), mean {fit.mean():.1f}  {best[2]}")
    return best


def save(name, entry):
    moves = json.loads(MOVES.read_text()) if MOVES.exists() else {}
    moves[name] = entry
    MOVES.write_text(json.dumps(moves, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("move", choices=["hop", "swag", "lie_down"])
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--from-current", action="store_true", help="start from the parameters in moves_hybrid.json")
    a = ap.parse_args()
    if a.move == "hop":
        base = dict(HOP_BASE, **json.loads(MOVES.read_text())["hop"]["params"]) if a.from_current else HOP_BASE
        fit, P, detail = cem(hop_score, HOP_SPEC, base, gens=a.gens, procs=a.procs)
        P = {k: round(v, 4) for k, v in P.items()}
        save("hop", dict(params=P, search=dict(fitness=round(fit, 2), bodies=detail)))
    if a.move == "swag":
        fit, P, detail = cem(swag_score, SWAG_SPEC, SWAG_BASE, gens=a.gens, procs=a.procs)
        save("swag", dict(params={k: round(v, 4) for k, v in P.items()}, offsets=swag_offsets(P).round(4).tolist(),
                          ramp=round(P["ramp"], 3), head=list(SWAG_HEAD), search=dict(fitness=round(fit, 2), bodies=detail)))
    if a.move == "lie_down":
        fit, P, detail = cem(lie_score, LIE_SPEC, LIE_BASE, gens=a.gens, procs=a.procs)
        keys, durs, bal = lie_keys(P)
        save("lie_down", dict(keys=[k.round(4).tolist() for k in keys], durations=durs, bal=bal, head=False))
        save("lie_hold", dict(keys=[keys[-1].round(4).tolist()], durations=[0.02], head=False))
        save("lie", dict(params={k: round(v, 4) for k, v in P.items()}, search=dict(fitness=round(fit, 2), bodies=detail)))
    print("saved", a.move)


if __name__ == "__main__":
    main()
