"""Build a Kaggle CPU kernel that TESTS a robot brain at scale (no training): thousands of lock-step tries on the
maker's own MuJoCo model, every fall written down with its seed.

    python kaggle/make_sweep_kernel.py --body go2 -o /tmp/ks && kaggle kernels push -p /tmp/ks
    kaggle kernels output <user>/bolt-sweep-go2 -p out/      # sweep.json, sweep.log

Why: 48-try tests cannot see a fall that happens once in a few hundred tries, and a robot runs for hours. A free
Kaggle CPU session (4 cores, ~11 h) runs tens of thousands of tries. The kernel clones this repository's branch and
the makers' models at pinned commits (Unitree unitree_mujoco, DEEP Robotics URDF_model), then loops over the
conditions below with deploy/lockstep_vendor.py (--estimate: the robot's own inputs), 4 at a time, until the time
guard. Simulation only.
"""
import argparse
import json
from pathlib import Path

PLAN = dict(go2="trot:6,canter:8,trot:4,canter:6,walk:4,sit:5,lie:6,getup:6,trot:5",
            lite3="trot:6,canter:8,trot:4,canter:6,walk:4,sit:5,lie:6,getup:6,trot:5")
CONDITIONS = [   # name, extra lockstep_vendor flags
    ("0-6 ms", ["--delay-ms", "0", "6"]),
    ("10-19 ms", ["--delay-ms", "10", "19"]),
    ("5% missed", ["--delay-ms", "3", "6", "--skip-p", "0.05"]),
    ("10% missed", ["--delay-ms", "3", "6", "--skip-p", "0.10"]),
    ("5% missed + 10-19 ms", ["--delay-ms", "10", "19", "--skip-p", "0.05"]),
    ("left 0.8, 10-19 ms", ["--delay-ms", "10", "19", "--steer", "0.8"]),
    ("right 0.8, 10-19 ms", ["--delay-ms", "10", "19", "--steer", "-0.8"]),
    ("left 0.4, 5% missed", ["--delay-ms", "3", "6", "--skip-p", "0.05", "--steer", "0.4"]),
    ("right 0.4, 5% missed", ["--delay-ms", "3", "6", "--skip-p", "0.05", "--steer", "-0.4"]),
]

STEER_SWEEP = [(f"{side} {abs(r)}, {lo}-{hi} ms", ["--delay-ms", str(lo), str(hi), "--steer", str(r)])
               for side, sgn in (("left", 1), ("right", -1)) for r in (0.4 * sgn, 0.6 * sgn, 0.8 * sgn)
               for lo, hi in ((0, 6), (6, 10), (10, 19))][:18]
LIMITS = [   # beyond the default conditions: where does it break?
    ("15% missed", ["--delay-ms", "3", "6", "--skip-p", "0.15"]),
    ("20% missed", ["--delay-ms", "3", "6", "--skip-p", "0.20"]),
    ("10% missed + 10-19 ms", ["--delay-ms", "10", "19", "--skip-p", "0.10"]),
    ("left 0.6, 10% missed", ["--delay-ms", "3", "6", "--skip-p", "0.10", "--steer", "0.6"]),
    ("right 0.4, 10% missed", ["--delay-ms", "3", "6", "--skip-p", "0.10", "--steer", "-0.4"]),
    ("left 0.6, 5% missed + 10-19 ms", ["--delay-ms", "10", "19", "--skip-p", "0.05", "--steer", "0.6"]),
    ("right 0.4, 5% missed + 10-19 ms", ["--delay-ms", "10", "19", "--skip-p", "0.05", "--steer", "-0.4"]),
]
TERRAIN = [   # Unitree's scene with its bumps and staircase left in (lockstep_vendor --stairs); Go2 only
    ("bumps, 0-6 ms", ["--stairs", "--delay-ms", "0", "6"]),
    ("bumps, 10-19 ms", ["--stairs", "--delay-ms", "10", "19"]),
    ("bumps, 5% missed", ["--stairs", "--delay-ms", "3", "6", "--skip-p", "0.05"]),
    ("flat, 0-6 ms (control)", ["--delay-ms", "0", "6"]),
]
BARS = [(f"bars {h} cm{', ' + tag if tag else ''}", ["--bar-cm", str(h)] + extra)   # two bars across the path
        for h in (2, 3, 4, 5, 6, 8) for tag, extra in (("", ["--delay-ms", "0", "6"]), ("5% missed", ["--delay-ms", "3", "6", "--skip-p", "0.05"]))]
BARS4 = [c for c in BARS if c[0].split()[1] in ("3", "4", "5", "6")]   # the heights where brains differ (22 Nov)
MISSED = [(f"{p}% missed", ["--delay-ms", "3", "6", "--skip-p", str(p / 100)]) for p in (5, 10, 15)]
PRESETS = dict(default=CONDITIONS, steer=STEER_SWEEP, limits=LIMITS, terrain=TERRAIN, bars=BARS, bars4=BARS4, missed=MISSED)
NO_CANTER = "trot:6,trot:8,trot:4,trot:6,walk:4,sit:5,lie:6,getup:6,trot:5"

KERNEL = r'''
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
CFG = __CFG__
T0 = time.time()
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
ROOT = Path(os.environ.get("BOLT_ROOT", "/tmp/bolt"))
LOG = open(WORK / "sweep.log", "a")
def say(*a):
    line = f"[{(time.time() - T0) / 3600:5.2f} h] " + " ".join(str(x) for x in a)
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()
def sh(*cmd, **kw):
    return subprocess.run(list(cmd), check=True, **kw)
if not os.environ.get("NO_SETUP"):
    sh(sys.executable, "-m", "pip", "install", "-q", "mujoco==3.17.0", "numpy>=2")
    ROOT.mkdir(parents=True, exist_ok=True)
    for url, rev, name in CFG["repos"]:
        if not (ROOT / name).exists():
            sh("git", "clone", "-q", url, str(ROOT / name))
            sh("git", "-C", str(ROOT / name), "checkout", "-q", rev)
env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
           UNITREE_MUJOCO=os.environ.get("UNITREE_MUJOCO", str(ROOT / "unitree_mujoco")),
           LITE3_MJCF=os.environ.get("LITE3_MJCF", str(ROOT / "URDF_model" / "Lite3" / "mjcf" / "Lite3.xml")))
env.pop("MUJOCO_GL", None)
TRAIN = Path(os.environ.get("BOLT_TRAIN", str(ROOT / "bolt" / "train")))
say("body", CFG["body"], "plan", CFG["plan"], "chunk", CFG["chunk"], "hours", CFG["hours"])
res = {name: dict(tries=0, falls=0, fall_seeds=[], fall_clips=[], over_speed_steps=0, steps=0) for name, _ in CFG["conditions"]}
def piece(job):
    name, flags, seed0 = job
    cmd = [sys.executable, "deploy/lockstep_vendor.py", "--body", CFG["body"], "--estimate", "--plan", CFG["plan"],
           "--tries", str(CFG["chunk"]), "--seed0", str(seed0)] + CFG["body_flags"] + flags
    p = subprocess.run(cmd, cwd=TRAIN, env=env, capture_output=True, text=True)
    return name, seed0, p.returncode, p.stdout + p.stderr
rnd = 0
while (time.time() - T0) / 3600 < CFG["hours"]:
    jobs = [(name, flags, CFG["seed0"] + rnd * CFG["chunk"]) for name, flags in CFG["conditions"]]
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        for name, seed0, rc, out in ex.map(piece, jobs):
            m = re.search(r"(\d+)/(\d+) fell", out)
            if rc or not m:
                say("FAILED piece", name, seed0, out[-400:]); continue
            r = res[name]
            r["tries"] += int(m.group(2)); r["falls"] += int(m.group(1))
            s = re.search(r"fall seeds \[([^\]]*)\]", out)
            r["fall_seeds"] += [int(x) for x in s.group(1).split(",") if x.strip()] if s else []
            c = re.search(r"falls in (\[[^\]]*\])", out)
            r["fall_clips"] += eval(c.group(1)) if c else []
            o = re.search(r"steps over 1.2x \(run_go2.py would stop\) (\d+) of (\d+)", out)
            if o:
                r["over_speed_steps"] += int(o.group(1)); r["steps"] += int(o.group(2))
    rnd += 1
    json.dump(dict(config=CFG, hours=(time.time() - T0) / 3600, rounds=rnd, results=res), open(WORK / "sweep.json", "w"), indent=1)
    say("round", rnd, {k: f"{v['falls']}/{v['tries']}" for k, v in res.items()})
    if CFG.get("max_rounds") and rnd >= CFG["max_rounds"]:
        break
say("done")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", choices=["go2", "lite3"], required=True)
    ap.add_argument("--user", default="arunmallikarjuna")
    ap.add_argument("--slug")
    ap.add_argument("--branch", default="claude/robot-build-memory-ju76nt")
    ap.add_argument("--repo", default="https://github.com/MrArun005/Cool-Websites")
    ap.add_argument("--hours", type=float, default=10.8)
    ap.add_argument("--chunk", type=int, default=25, help="tries per piece (one piece = one condition)")
    ap.add_argument("--seed0", type=int, default=100000, help="first seed (48-try tests use 0-47)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default", help="steer: left/right 0.4-0.8 rad/s x 3 delays")
    ap.add_argument("--plan", help="the plan (default: the 50 s plan)")
    ap.add_argument("--no-canter", action="store_true", help="the plan with a trot where the canters are")
    ap.add_argument("--brains", help="compare brains on the same seeds: name=repo/relative/params.pkl,... (each condition runs "
                    "once per brain, named '<condition> [<name>]'); default: the body's current brain only")
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this many rounds (smoke tests)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    conds = PRESETS[a.preset]
    if a.brains:   # paths are relative to the repo's train/ folder (where the kernel runs lockstep_vendor)
        pairs = [b.split("=", 1) for b in a.brains.split(",")]
        for _, path in pairs:
            assert (Path(__file__).resolve().parents[1] / path).exists(), f"{path}: not in the repo (commit it first)"
        conds = [(f"{c} [{n}]", f + ["--brain", path]) for c, f in conds for n, path in pairs]
    cfg = dict(body=a.body, plan=a.plan or (NO_CANTER if a.no_canter else PLAN[a.body]), conditions=conds, hours=a.hours, chunk=a.chunk, seed0=a.seed0,
               workers=a.workers, max_rounds=a.max_rounds, body_flags=["--dt", "0.001"] if a.body == "lite3" else [],
               repos=[(a.repo, a.branch, "bolt"),
                      ("https://github.com/unitreerobotics/unitree_mujoco", "1eb6642", "unitree_mujoco"),
                      ("https://github.com/DeepRoboticsLab/URDF_model", "52352f5", "URDF_model")])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(KERNEL.replace("__CFG__", repr(cfg)))
    slug = a.slug or f"bolt-sweep-{a.body}"
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{a.user}/{slug}", title=slug, code_file="kernel.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[]), indent=1))
    print(f"kernel in {out}: push with  kaggle kernels push -p {out}")


if __name__ == "__main__":
    main()
