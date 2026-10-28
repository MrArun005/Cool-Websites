"""Build the Kaggle CPU kernel "bolt-mastermind": a big MasterMind DAgger run (mastermind/dagger.py) on Kaggle's
free 4-core CPU session (DAgger is CPU work: MuJoCo rollouts of the teachers and the student, a small torch MLP).

    # 1. the private data set: the code and the teacher brains the teacher set needs, as one tar (from git HEAD,
    #    so files being edited in the working tree are not picked up)
    python kaggle/make_mastermind_kernel.py data -o /tmp/md
    kaggle datasets create -p /tmp/md            # first time (private by default); later: kaggle datasets version -p /tmp/md -m new
    # 2. the kernel (the script below, no big files embedded)
    python kaggle/make_mastermind_kernel.py kernel -o /tmp/mk
    kaggle kernels push -p /tmp/mk
    kaggle kernels status arunmallikarjuna/bolt-mastermind
    kaggle kernels output arunmallikarjuna/bolt-mastermind -p runs/mastermind/kaggle   # run/: student.pt, student_r*.pt, history.json, dagger.log

The kernel runs DAgger one round per process (dagger.py --rounds k+1 --resume, which continues exactly where the
last round stopped), and before each round predicts from history.json whether it still fits in the session
(--budget-h, Kaggle stops a session at 12 h); it stops cleanly otherwise, so the output always holds the last
finished round's student. data.npz (GBs) is deleted at the end unless --keep-data.

Local smoke test of the exact kernel script, with Kaggle's folders faked:
    KAGGLE_INPUT=/tmp/fake/input KAGGLE_WORKING=/tmp/fake/working BOLT_ROOT=/tmp/fake/bolt NO_PIP=1 python /tmp/mk/kernel.py
(with the data set unpacked or the tar placed under /tmp/fake/input/bolt-mastermind-data/; make the kernel with tiny
--rounds/--steps0/--steps/--workers for that).
"""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TAR = "bolt-mastermind.pack"   # a tar under a name Kaggle does not unpack (it unpacks .tar AND the policy.zip files inside)
# teacher set v4 (mastermind/teachers.py): dog 18e-36M, clips 18g-35.5M, landing r4, jump 16i-35.5M; skills.Bolt
# also loads its default jump (16g), self-right (17e) and posture (getup-3M) brains
MODELS = ["bolt-dog-switching-18e-36M", "bolt-dog-canter-switching-18g-35.5M", "bolt-dog-imitate-pps-student-r4",
          "bolt-dog-jump-runup-16i-35.5M", "bolt-dog-jump-a1-16g-27M", "bolt-dog-selfright-17e-16.5M",
          "bolt-dog-posture-getup-3M"]
INIT = "train/mastermind/export_v3/student.pt"   # the v3 student (run b round 5), as the local v4 run
PIP = ["numpy>=2", "mujoco==3.17.0", "stable-baselines3==2.9.0", "gymnasium==1.4.0"]   # torch: Kaggle's own (numpy >= 2: the brains' stats were pickled with numpy 2)

KERNEL = r'''# Bolt robot dog: MasterMind DAgger on a Kaggle CPU (made by train/kaggle/make_mastermind_kernel.py)
import glob, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
T0 = time.time()
CFG = @CFG@
INP = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
ROOT = Path(os.environ.get("BOLT_ROOT", "/tmp/bolt"))
OUT = WORK / "run"
OUT.mkdir(parents=True, exist_ok=True)
log = open(WORK / "kernel.log", "a")


def say(s):
    s = f"[{(time.time() - T0) / 3600:5.2f} h] {s}"
    print(s, flush=True); log.write(s + "\n"); log.flush()


if not os.environ.get("NO_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + CFG["pip"], check=True)
import numpy, torch, mujoco, stable_baselines3  # noqa: E401
say(f"python {sys.version.split()[0]} numpy {numpy.__version__} torch {torch.__version__} mujoco {mujoco.__version__} "
    f"sb3 {stable_baselines3.__version__} cpus {os.cpu_count()}")
# the code and brains: the data set's tar (named .pack so that Kaggle leaves it, and the policy.zip files in it, packed)
tars = sorted(glob.glob(str(INP / "**" / CFG["tar"]), recursive=True))
if not (ROOT / "train" / "mastermind" / "dagger.py").exists():
    if tars and Path(tars[0]).is_file():
        with tarfile.open(tars[0]) as t:
            t.extractall(ROOT, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    else:
        src = [Path(p).parents[2] for p in glob.glob(str(INP / "**" / "train" / "mastermind" / "dagger.py"), recursive=True)]
        assert src, f"no {CFG['tar']} or train/mastermind/dagger.py under {INP}"
        shutil.copytree(src[0], ROOT, dirs_exist_ok=True)
TRAIN = ROOT / "train"
say(f"code in {TRAIN}")
env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
env.pop("MUJOCO_GL", None)   # no rendering
budget, hard = CFG["budget_h"] * 3600, CFG["hard_h"] * 3600


def history():
    p = OUT / "history.json"
    return json.loads(p.read_text()) if p.exists() else []


def predict(h):
    """seconds the next round should take, from the rounds so far (training time grows with the data)."""
    last = h[-1]
    later = [r for r in h if r["round"] > 0]
    coll = later[-1]["collect_s"] if later else last["collect_s"] * CFG["steps"] / CFG["steps0"]
    tr = last["train_s"] * (last["data"] + CFG["steps"]) / last["data"]
    if last["round"] == 0:
        tr *= CFG["epochs"] / CFG["epochs0"]
    return coll + tr + 120   # + start-up (teachers, data.npz) and saving


rc = 0
for rnd in range(CFG["rounds"]):
    h = history()
    if len(h) > rnd:
        continue
    if h:
        p, el = predict(h), time.time() - T0
        if el + 1.1 * p > budget:
            say(f"stop before round {rnd}: {el / 3600:.2f} h used, the round would take ~{p / 3600:.2f} h (budget {CFG['budget_h']} h)")
            break
        say(f"round {rnd}: expected ~{p / 60:.0f} min")
    cmd = [sys.executable, "-u", "mastermind/dagger.py"] + CFG["args"] + ["--rounds", str(rnd + 1), "--out", str(OUT)]
    if (OUT / "data.npz").exists():
        cmd.append("--resume")
    say(" ".join(cmd[1:]))
    p = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        if time.time() - T0 > hard:
            say("hard time limit: stopping the round")
            p.kill()
            break
    rc = p.wait()
    if rc:
        say(f"dagger.py exit code {rc}: stop")
        break
h = history()
(OUT / "summary.json").write_text(json.dumps(dict(rounds_done=len(h), hours=round((time.time() - T0) / 3600, 2),
                                                  exit_code=rc, config=CFG), indent=1))
if h and not CFG["keep_data"] and (OUT / "data.npz").exists():
    (OUT / "data.npz").unlink()
say(f"done: {len(h)} rounds" + (f", newest student {OUT / 'student.pt'}" if h else ""))
sys.exit(1 if rc and not h else 0)
'''


def build_data(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "train", "models"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split()
    keep = []
    for f in files:
        p = Path(f)
        if (p.parent == Path("train") and p.suffix in (".py", ".xml")) or \
           (p.parent == Path("train/mastermind") and p.suffix == ".py") or \
           p.parent == Path("train/dogmotion/clips") or f == INIT or \
           (len(p.parts) == 3 and p.parts[0] == "models" and p.parts[1] in MODELS):
            keep.append(f)
    missing = [m for m in MODELS if not any(f.startswith(f"models/{m}/") for f in keep)]
    assert not missing and INIT in keep, (missing, INIT)
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
    args = ["--teachers", a.teachers, "--steps0", str(a.steps0), "--steps", str(a.steps), "--workers", str(a.workers),
            "--beta-floor", str(a.beta_floor), "--hidden", a.hidden, "--epochs0", str(a.epochs0), "--epochs", str(a.epochs),
            "--seed", str(a.seed)]
    if a.init_student:
        args += ["--init-student", str(Path("..") / a.init_student)]   # relative to train/
    cfg = dict(pip=PIP, tar=TAR, rounds=a.rounds, args=args, steps0=a.steps0, steps=a.steps, epochs0=a.epochs0,
               epochs=a.epochs, budget_h=a.budget_h, hard_h=a.hard_h, keep_data=a.keep_data)
    (out / "kernel.py").write_text(KERNEL.replace("@CFG@", repr(cfg)))
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{a.user}/{a.slug}", title=a.slug, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
        dataset_sources=[f"{a.user}/{a.dataset}"], competition_sources=[], kernel_sources=[]), indent=1))
    print(f"kernel in {out}: push with  kaggle kernels push -p {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "kernel"])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--user", default="arunmallikarjuna")
    ap.add_argument("--dataset", default="bolt-mastermind-data")
    ap.add_argument("--slug", default="bolt-mastermind")
    ap.add_argument("--teachers", default="v4")
    ap.add_argument("--rounds", type=int, default=20, help="at most; the kernel stops earlier when the next round would not fit")
    ap.add_argument("--steps0", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--beta-floor", type=float, default=0.2)
    ap.add_argument("--hidden", default="1024,512,256")
    ap.add_argument("--epochs0", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init-student", default=INIT, help="'' for a fresh network")
    ap.add_argument("--budget-h", type=float, default=11.3, help="start no round that would end after this")
    ap.add_argument("--hard-h", type=float, default=11.7, help="kill a round still running at this time")
    ap.add_argument("--keep-data", action="store_true", help="keep run/data.npz in the output (GBs; needed to --resume elsewhere)")
    a = ap.parse_args()
    build_data(a) if a.what == "data" else build_kernel(a)


if __name__ == "__main__":
    main()
