"""Build a Kaggle GPU kernel for a Bolt run: one self-contained script (the robot and trainer embedded).

    python kaggle/make_kernel.py --user <kaggle username> --slug bolt-run20-gpu --steps 200000000 -o /tmp/k
    kaggle kernels push -p /tmp/k                         # starts it on a Kaggle GPU
    kaggle kernels status <user>/bolt-run20-gpu           # queued / running / complete / error
    kaggle kernels output <user>/bolt-run20-gpu -p out/   # train.log, progress.json, params.pkl, checkpoints/

The script installs MuJoCo MJX and Brax with CUDA, writes bolt.xml and mjx_run.py next to itself and runs
mjx_run.py, keeping everything under /kaggle/working (what Kaggle saves as the kernel's output).
"""
import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

TEMPLATE = '''# Bolt robot dog: {slug}, trained on a Kaggle GPU (made by train/kaggle/make_kernel.py)
import base64, os, subprocess, sys, pathlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[cuda12]", "mujoco==3.17.0", "mujoco-mjx==3.17.0", "brax==0.14.2"], check=True)
work = pathlib.Path("/kaggle/working"); work.mkdir(exist_ok=True)
FILES = {files}
for name, blob in FILES.items():
    (work / name).write_bytes(base64.b64decode(blob))
os.chdir(work)
import jax; print("devices:", jax.devices(), flush=True)
with open(work / "train.log", "w") as log:
    p = subprocess.Popen([sys.executable, "-u", "mjx_run.py", "--steps", "{steps}", "--envs", "{envs}", "--evals", "{evals}",
                          "--out", str(work / "{out}")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True); log.write(line); log.flush()
sys.exit(p.wait())
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", default="bolt-run20-gpu")
    ap.add_argument("--steps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--evals", type=int, default=40)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {n: base64.b64encode((HERE / n).read_bytes()).decode() for n in ("bolt.xml", "mjx_run.py")}
    (out / "kernel.py").write_text(TEMPLATE.format(slug=args.slug, files=json.dumps(files), steps=args.steps, envs=args.envs,
                                                   evals=args.evals, out="run"))
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{args.user}/{args.slug}", title=args.slug, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[]), indent=1))
    print(f"kernel in {out}: push with  kaggle kernels push -p {out}")


if __name__ == "__main__":
    main()
