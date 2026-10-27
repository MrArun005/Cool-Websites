"""Build a Kaggle GPU kernel for a Bolt run: one self-contained script (the robot and trainer embedded).

    python kaggle/make_kernel.py --user <kaggle username> --slug bolt-run20-gpu --steps 200000000 -o /tmp/k
    kaggle kernels push -p /tmp/k                         # starts it on a Kaggle GPU
    kaggle kernels status <user>/bolt-run20-gpu           # queued / running / complete / error
    kaggle kernels output <user>/bolt-run20-gpu -p out/   # train.log, progress.json, params.pkl, checkpoints/

    # dog imitation (mjx_imitate.py), warm-started from a CPU brain exported with `mjx_imitate.py warm`:
    python kaggle/make_kernel.py --user <user> --job imitate --slug bolt-imitate-gpu --warm warm.pkl --steps 2000000000 -o /tmp/ki

The script installs MuJoCo MJX and Brax with CUDA, writes the files it needs next to itself (bolt.xml and the trainer;
for imitate also the dog clips and the warm-start brain) and runs the trainer, keeping everything under
/kaggle/working (what Kaggle saves as the kernel's output).
"""
import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

TEMPLATE = '''# Bolt robot dog: {slug}, trained on a Kaggle GPU (made by train/kaggle/make_kernel.py)
import base64, os, subprocess, sys, pathlib
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda13-plugin", "jax-cuda13-pjrt"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[cuda12]==0.9.2", "jaxlib==0.9.2", "flax==0.12.6", "optax==0.2.8", "orbax-checkpoint==0.12.8", "mujoco==3.17.0", "mujoco-mjx==3.17.0", "brax==0.14.2"], check=True)
work = pathlib.Path("/kaggle/working"); work.mkdir(exist_ok=True)
FILES = {files}
for name, blob in FILES.items():
    (work / name).parent.mkdir(parents=True, exist_ok=True)
    (work / name).write_bytes(base64.b64decode(blob))
os.chdir(work)
import jax; print("devices:", jax.devices(), flush=True)
with open(work / "train.log", "w") as log:
    p = subprocess.Popen([sys.executable, "-u"] + {cmd}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True); log.write(line); log.flush()
sys.exit(p.wait())
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--job", choices=["run", "imitate"], default="run", help="run: mjx_run.py (run 20); imitate: mjx_imitate.py")
    ap.add_argument("--slug", default="bolt-run20-gpu")
    ap.add_argument("--warm", help="imitate: the numpy brain to start from (mjx_imitate.py warm ...)")
    ap.add_argument("--hours", type=float, default=9.5, help="imitate: stop after this long even if steps remain (Kaggle allows 12 h)")
    ap.add_argument("--steps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--evals", type=int, help="progress reports / saved brains (default: run 40, imitate 21 = every 5%%)")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = lambda p: base64.b64encode(Path(p).read_bytes()).decode()
    if args.job == "run":
        files = {n: enc(HERE / n) for n in ("bolt.xml", "mjx_run.py")}
        cmd = ["mjx_run.py", "--steps", str(args.steps), "--envs", str(args.envs), "--evals", str(args.evals or 40), "--out", "run"]
    else:
        from mjx_imitate import ALL_CLIPS   # numpy + mujoco only
        files = {n: enc(HERE / n) for n in ("bolt.xml", "mjx_imitate.py")}
        files.update({f"dogmotion/clips/{c}.npz": enc(HERE / "dogmotion" / "clips" / f"{c}.npz") for c in ALL_CLIPS})
        cmd = ["mjx_imitate.py", "train", "--steps", str(args.steps), "--envs", str(args.envs), "--evals", str(args.evals or 21),
               "--out", "run", "--hours", str(args.hours)]
        if args.warm:
            files["warm.pkl"] = enc(args.warm)
            cmd += ["--warm", "warm.pkl"]
    (out / "kernel.py").write_text(TEMPLATE.format(slug=args.slug, files=json.dumps(files), cmd=json.dumps(cmd)))
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{args.user}/{args.slug}", title=args.slug, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[]), indent=1))
    print(f"kernel in {out}: push with  kaggle kernels push -p {out}")


if __name__ == "__main__":
    main()
