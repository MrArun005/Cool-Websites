"""Build a Kaggle CPU kernel that fine-tunes one Bolt brain with PPO for a whole free session (~10.5 h on 4 CPUs).
The same recipe as make_mastermind_kernel.py: a private data set holding the code and brains as one tar (named
.pack, which Kaggle leaves packed), pip installs over Kaggle's torch, everything written to /kaggle/working as it
goes (train/runs is a link into it, so every checkpoint and best/ is saved the moment it is written), and a time
guard (--hours) that ends training cleanly before Kaggle's 12 h limit.

Jobs (JOBS below):
  jump-trot  train_jump.py from the run-up brain 16i-35.5M: jump from a trot to a canter (1.2-2.4 m/s, most tries
             mid-stride), practice bars up to 75 cm; exams at 1.5 m/s and 2.0 m/s from 1.5 m, 40-70 cm.
  recover    train_getup.py from self-righting 17e-16.5M: kicks, bad landings, ice, a second kick, wider physics;
             exam = the whole recovery (self-right, tuck, stand) from those falls.

    python kaggle/make_train_kernel.py data -o /tmp/td          # code + brains from git HEAD, one tar
    kaggle datasets create -p /tmp/td                             # later: kaggle datasets version -p /tmp/td -m new
    python kaggle/make_train_kernel.py kernel --job jump-trot -o /tmp/k1
    kaggle kernels push -p /tmp/k1
    kaggle kernels status arunmallikarjuna/bolt-jump-trot
    kaggle kernels output arunmallikarjuna/bolt-jump-trot -p runs/kaggle-jump-trot

The output: runs/<name>/ (checkpoints/, best/ with best.json, train.log), fair_exam_*.txt (30 tries per test on
fresh seeds 9000+, for the best brain and the starting brain), summary.json and kernel.log.

Local smoke test of the exact kernel script, Kaggle's folders faked (tiny settings: --train-h 0.06 --every 8192
--envs 2 --n-steps 1024 --exam-episodes 2 --final-episodes 2):
    KAGGLE_INPUT=/tmp/fake/input KAGGLE_WORKING=/tmp/fake/working BOLT_ROOT=/tmp/fake/bolt NO_PIP=1 python /tmp/k1/kernel.py
"""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TAR = "bolt-train.pack"   # a tar under a name Kaggle does not unpack (it unpacks .tar AND the policy.zip files inside)
MODELS = ["bolt-dog-jump-runup-16i-35.5M", "bolt-dog-selfright-17e-16.5M", "bolt-dog-posture-getup-3M"]
PIP = ["numpy>=2", "mujoco==3.17.0", "stable-baselines3==2.9.0", "gymnasium==1.4.0"]   # torch: Kaggle's own

JOBS = {
    "jump-trot": dict(
        name="jumptrot", script="train_jump.py", start="../models/bolt-dog-jump-runup-16i-35.5M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.45", "--bar-max", "0.75",
              "--exam-speed", "1.5", "--exam-dist", "1.5", "--exam-heights", "0.40", "0.50", "0.60", "0.70",
              "--exam2", "2.0", "1.5"],
        final=["jump", "--speeds", "1.5", "2.0", "--dist", "1.5", "--heights", "0.40", "0.50", "0.60", "0.70", "--gait"]),
    "recover": dict(
        name="recover", script="train_getup.py", start="../models/bolt-dog-selfright-17e-16.5M",
        args=["--goal", "selfright", "--hard", "--full-exam", "--lr", "1e-4", "--target-kl", "0.015"],
        final=["recover", "--easy"]),
}

KERNEL = r'''# Bolt robot dog: PPO fine-tuning on a Kaggle CPU (made by train/kaggle/make_train_kernel.py)
import glob, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
T0 = time.time()
CFG = @CFG@
INP = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
ROOT = Path(os.environ.get("BOLT_ROOT", "/tmp/bolt"))
WORK.mkdir(parents=True, exist_ok=True)
log = open(WORK / "kernel.log", "a")


def say(s):
    s = f"[{(time.time() - T0) / 3600:5.2f} h] {s}"
    print(s, flush=True); log.write(s + "\n"); log.flush()


if not os.environ.get("NO_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + CFG["pip"], check=True)
import numpy, torch, mujoco, stable_baselines3  # noqa: E401
say(f"python {sys.version.split()[0]} numpy {numpy.__version__} torch {torch.__version__} mujoco {mujoco.__version__} "
    f"sb3 {stable_baselines3.__version__} cpus {os.cpu_count()}")
tars = sorted(glob.glob(str(INP / "**" / CFG["tar"]), recursive=True))
if not (ROOT / "train" / CFG["script"]).exists():
    assert tars and Path(tars[0]).is_file(), f"no {CFG['tar']} under {INP}"
    with tarfile.open(tars[0]) as t:
        t.extractall(ROOT, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
TRAIN = ROOT / "train"
(WORK / "runs").mkdir(exist_ok=True)
runs = TRAIN / "runs"
if runs.is_symlink() or not runs.exists():
    if runs.is_symlink():
        runs.unlink()
    runs.symlink_to(WORK / "runs", target_is_directory=True)   # every checkpoint lands in /kaggle/working at once
else:
    say(f"warning: {runs} is a real folder, outputs copied at the end only")
RUN = TRAIN / "runs" / CFG["name"]
RUN.mkdir(parents=True, exist_ok=True)
say(f"code in {TRAIN}, run in {RUN.resolve()}")
env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
env.pop("MUJOCO_GL", None)   # no rendering


def latest():
    cks = glob.glob(str(RUN / "checkpoints" / "policy_*_steps.zip"))
    cks = [c for c in cks if os.path.exists(c.replace("policy_", "policy_vecnormalize_", 1)[:-4] + ".pkl")]
    return max(cks, key=lambda c: int(Path(c).stem.split("_")[1])) if cks else None


rc = 0
for attempt in range(CFG["attempts"]):
    left = CFG["train_h"] - (time.time() - T0) / 3600
    if left < min(0.25, CFG["train_h"] / 4):
        say(f"{left:.2f} h of training time left: no more attempts")
        break
    ck = latest()
    cmd = [sys.executable, "-u", CFG["script"], "--name", CFG["name"], "--steps", str(CFG["steps"]),
           "--envs", str(CFG["envs"]), "--every", str(CFG["every"]), "--hours", f"{left:.3f}"] + CFG["args"]
    cmd += ["--resume", ck] if ck else ["--init-from", CFG["start"], "--exam-first"]
    say(("resume: " if ck else "start: ") + " ".join(cmd[2:]))
    with open(RUN / "train.log", "a") as tl:
        p = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True); tl.write(line); tl.flush()
            if time.time() - T0 > CFG["hard_h"] * 3600:
                say("hard time limit: killing training")
                p.kill()
                break
        rc = p.wait()
    say(f"{CFG['script']} exit code {rc}")
    if rc == 0:
        break
# fair exams on fresh seeds (9000+): the best brain and the starting brain, the same tries
best = RUN / "best"
for tag, folder in (("best", best), ("start", Path(CFG["start"]))):
    if (TRAIN / folder / "policy.zip").exists() and time.time() - T0 < CFG["hard_h"] * 3600 - 600:
        cmd = [sys.executable, "-u", "fair_exam.py"] + [CFG["final"][0], str(folder)] + CFG["final"][1:] + \
              ["--episodes", str(CFG["final_episodes"]), "--no-save"]
        say(f"fair exam of the {tag} brain: " + " ".join(cmd[2:]))
        r = subprocess.run(cmd, cwd=TRAIN, env=env, capture_output=True, text=True, timeout=max(60, CFG["hard_h"] * 3600 - (time.time() - T0) - 300))
        (WORK / f"fair_exam_{tag}.txt").write_text(r.stdout + r.stderr)
        print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
summary = dict(hours=round((time.time() - T0) / 3600, 2), exit_code=rc, latest=latest(), config=CFG)
if (best / "best.json").exists():
    summary["best"] = json.loads((best / "best.json").read_text())
(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
if not runs.is_symlink():
    shutil.copytree(RUN, WORK / "runs" / CFG["name"], dirs_exist_ok=True)
say("done: " + json.dumps(summary.get("best", {}))[:500])
sys.exit(0 if (best / "best.json").exists() else 1)
'''


def build_data(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "train", "models"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split()
    keep = []
    for f in files:
        p = Path(f)
        if (p.parent == Path("train") and p.suffix in (".py", ".xml")) or p.parent == Path("train/dogmotion/clips") or \
           (len(p.parts) == 3 and p.parts[0] == "models" and p.parts[1] in MODELS and p.suffix in (".zip", ".pkl", ".md", ".json", ".txt")):
            keep.append(f)
    missing = [m for m in MODELS if not any(f == f"models/{m}/policy.zip" for f in keep)]
    assert not missing, missing
    with tarfile.open(out / TAR, "w") as t:
        for f in keep:
            blob = subprocess.run(["git", "show", f"HEAD:{f}"], cwd=REPO, capture_output=True, check=True).stdout
            info = tarfile.TarInfo(f)
            info.size, info.mode = len(blob), 0o644
            t.addfile(info, io.BytesIO(blob))
    (out / "dataset-metadata.json").write_text(json.dumps(dict(
        id=f"{a.user}/{a.dataset}", title=a.dataset, licenses=[dict(name="CC0-1.0")]), indent=1))
    print(f"{len(keep)} files, {(out / TAR).stat().st_size / 1e6:.1f} MB in {out / TAR}")


def build_kernel(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    job = JOBS[a.job]
    args = list(job["args"])
    if a.n_steps:
        args = [x for i, x in enumerate(args) if not (x == "--n-steps" or (i and args[i - 1] == "--n-steps"))]
        args += ["--n-steps", str(a.n_steps)]
    if a.exam_episodes and job["script"] == "train_getup.py":
        args += ["--exam-episodes", str(a.exam_episodes)]
    cfg = dict(pip=PIP, tar=TAR, script=job["script"], name=job["name"], start=job["start"], args=args,
               steps=a.steps, envs=a.envs, every=a.every or 1_000_000,
               train_h=a.train_h, hard_h=a.hard_h, attempts=3, final=job["final"], final_episodes=a.final_episodes)
    (out / "kernel.py").write_text(KERNEL.replace("@CFG@", repr(cfg)))
    slug = a.slug or f"bolt-{a.job}"
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{a.user}/{slug}", title=slug, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
        dataset_sources=[f"{a.user}/{a.dataset}"], competition_sources=[], kernel_sources=[]), indent=1))
    print(f"kernel in {out}: push with  kaggle kernels push -p {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "kernel"])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--job", choices=sorted(JOBS), default="jump-trot")
    ap.add_argument("--user", default="arunmallikarjuna")
    ap.add_argument("--dataset", default="bolt-train-data")
    ap.add_argument("--slug", help="kernel name (default bolt-<job>)")
    ap.add_argument("--steps", type=int, default=400_000_000, help="a cap far beyond the session: the time guard ends it")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--every", type=int, help="steps between checkpoints and exams (default 1M)")
    ap.add_argument("--n-steps", type=int, help="override PPO steps per env per update (smoke tests)")
    ap.add_argument("--exam-episodes", type=int, default=15, help="recover: tries per kind in the checkpoint exam")
    ap.add_argument("--final-episodes", type=int, default=30, help="tries per test in the final fair exams")
    ap.add_argument("--train-h", type=float, default=10.6, help="training ends (time guard) this long after the start")
    ap.add_argument("--hard-h", type=float, default=11.6, help="kill anything still running at this time")
    a = ap.parse_args()
    build_data(a) if a.what == "data" else build_kernel(a)


if __name__ == "__main__":
    main()
