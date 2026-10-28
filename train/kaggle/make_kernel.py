"""Build a Kaggle GPU kernel for a Bolt run: one self-contained script (the robot and trainer embedded).

    python kaggle/make_kernel.py --user <kaggle username> --slug bolt-run20-gpu --steps 200000000 -o /tmp/k
    kaggle kernels push -p /tmp/k                         # starts it on a Kaggle GPU
    kaggle kernels status <user>/bolt-run20-gpu           # queued / running / complete / error
    kaggle kernels output <user>/bolt-run20-gpu -p out/   # train.log, progress.json, params.pkl, checkpoints/

    # dog imitation (mjx_imitate.py), warm-started from a CPU brain exported with `mjx_imitate.py warm`:
    # (warm.pkl is too big to embed: upload it once as a private dataset with  kaggle datasets create -p <dir with warm.pkl>)
    python kaggle/make_kernel.py --user <user> --job imitate --slug bolt-imitate-gpu --warm-dataset <user>/bolt-imitate-warm \
        --steps 360000000 --hours 9.5 -o /tmp/ki        # 4096 dogs on Kaggle's 2x T4: ~11.7k steps/s, 360M in ~8.6 h
    kaggle kernels push -p /tmp/ki --accelerator NvidiaTeslaT4

    # the skill-switch fixes (steps 1-3): landing starts, posture commands while moving, transition term; motor brake on;
    # the real-robot gaps (step 3b): 0-40 ms action latency per try, legged_gym sensor noise
    python kaggle/make_kernel.py --user <user> --job imitate --slug bolt-dog-gpu3 --warm-dataset <user>/bolt-imitate-360m \
        --steps 1000000000 --hours 10.5 --landing-rate 0.2 --moving-rate 0.2 --transition-weight 0.3 \
        --latency-min-ms 0 --latency-max-ms 40 --obs-noise 1.0 \
        --evals 40 --extra "--reward-scaling 0.246 --switch-rate 0.3 --seed 3" -o /tmp/kg3
    MUJOCO_GL=osmesa python /tmp/kg3/kernel.py --local /tmp/kg3_run   # optional local smoke test: runs the trainer
                                                                      # with --steps 20000 --envs 16 --quick in /tmp/kg3_run

The script installs MuJoCo MJX and Brax with CUDA, writes the files it needs next to itself (bolt.xml and the trainer;
for imitate also the dog clips and the warm-start brain) and runs the trainer, keeping everything under
/kaggle/working (what Kaggle saves as the kernel's output).
Embedded files are lzma-compressed (Kaggle refuses scripts over ~1 MB). The imitate job always carries pose_reached.py
(moving starts) and, with --landing-rate > 0, research/results/landing_starts.npz slimmed to the arrays mjx_imitate
reads, in float32 (what MjxImitate turns them into anyway). Every mjx_imitate.py train flag of steps 1-3, the real-robot
gaps (latency, sensor noise, motor lag) and the motor brake passes through; --extra passes any other train flags verbatim.
"""
import argparse
import base64
import io
import json
import lzma
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

TEMPLATE = '''# Bolt robot dog: {slug}, trained on a Kaggle GPU (made by train/kaggle/make_kernel.py)
import base64, lzma, os, subprocess, sys, pathlib
LOCAL = len(sys.argv) > 2 and sys.argv[1] == "--local"   # local smoke test: no installs, a short CPU run in sys.argv[2]
if not LOCAL:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "jax-cuda13-plugin", "jax-cuda13-pjrt"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[cuda12]==0.9.2", "jaxlib==0.9.2", "flax==0.12.6", "optax==0.2.8", "orbax-checkpoint==0.12.8", "mujoco==3.17.0", "mujoco-mjx==3.17.0", "brax==0.14.2"], check=True)
work = pathlib.Path(sys.argv[2] if LOCAL else "/kaggle/working"); work.mkdir(parents=True, exist_ok=True)
FILES = {files}
for name, blob in FILES.items():
    (work / name).parent.mkdir(parents=True, exist_ok=True)
    (work / name).write_bytes(lzma.decompress(base64.b64decode(blob)))
for w in sorted(pathlib.Path(sys.argv[3] if LOCAL and len(sys.argv) > 3 else "/kaggle/input").rglob("warm.pkl")) if {find_warm} else []:
    (work / "warm.pkl").write_bytes(w.read_bytes()); print("warm start from", w, flush=True); break   # an attached dataset
os.chdir(work)
import jax; print("devices:", jax.devices(), flush=True)
CMD = {cmd}
if LOCAL:   # the same command, cut to a CPU smoke test (later flags win in argparse)
    CMD = CMD + ["--steps", "20000", "--envs", "16", "--quick", "--evals", "3"]
print("command:", " ".join(CMD), flush=True)
with open(work / "train.log", "w") as log:
    p = subprocess.Popen([sys.executable, "-u"] + CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True); log.write(line); log.flush()
sys.exit(p.wait())
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--job", choices=["run", "imitate"], default="run", help="run: mjx_run.py (run 20); imitate: mjx_imitate.py")
    ap.add_argument("--slug", default="bolt-run20-gpu")
    ap.add_argument("--warm", help="imitate: the numpy brain to start from (mjx_imitate.py warm ...), embedded (Kaggle refuses "
                    "scripts over ~1 MB, so a 512x256 brain must come as a dataset instead: --warm-dataset)")
    ap.add_argument("--warm-dataset", help="imitate: <user>/<dataset> holding warm.pkl (kaggle datasets create), attached to the kernel")
    ap.add_argument("--hours", type=float, default=9.5, help="imitate: stop after this long even if steps remain (Kaggle allows 12 h)")
    ap.add_argument("--steps", type=int, default=200_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--evals", type=int, help="progress reports / saved brains (default: run 40, imitate 21 = every 5%%)")
    # mjx_imitate.py train flags of the skill-switch fixes (steps 1-3) and the motor brake; None: not passed (its default)
    for flag, kw in [("--landing-rate", dict(type=float)), ("--landing-grace", dict(type=float)), ("--landing-bridge", dict(type=float)),
                     ("--landing-then", {}), ("--moving-rate", dict(type=float)), ("--moving-speed", dict(type=float, nargs=2)),
                     ("--moving-gaits", dict(nargs="+")), ("--moving-clips", dict(nargs="+")), ("--pose-time", dict(nargs="+")),
                     ("--pose-weight", dict(type=float)), ("--transition-weight", dict(type=float)), ("--transition-window", dict(type=float)),
                     ("--transition-threshold", dict(type=float)), ("--latency-min-ms", dict(type=float)), ("--latency-max-ms", dict(type=float)),
                     ("--obs-noise", dict(type=float)), ("--motor-lag-ms", dict(type=float)),
                     ("--brake-speed", dict(type=float)), ("--brake-gain", dict(type=float))]:
        ap.add_argument(flag, **kw, help="imitate: passed to mjx_imitate.py train (default: its own)")
    ap.add_argument("--motor-brake", action=argparse.BooleanOptionalAction, help="imitate: passed on (default: mjx_imitate's, on)")
    ap.add_argument("--extra", default="", help='imitate: more train flags, verbatim (e.g. "--reward-scaling 0.246 --seed 3")')
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pack = lambda b: base64.b64encode(lzma.compress(b, preset=9)).decode()
    enc = lambda p: pack(Path(p).read_bytes())
    if args.job == "run":
        files = {n: enc(HERE / n) for n in ("bolt.xml", "mjx_run.py")}
        cmd = ["mjx_run.py", "--steps", str(args.steps), "--envs", str(args.envs), "--evals", str(args.evals or 40), "--out", "run"]
    else:
        from mjx_imitate import ALL_CLIPS   # numpy + mujoco only
        files = {n: enc(HERE / n) for n in ("bolt.xml", "mjx_imitate.py", "pose_reached.py")}
        files.update({f"dogmotion/clips/{c}.npz": enc(HERE / "dogmotion" / "clips" / f"{c}.npz") for c in ALL_CLIPS})
        cmd = ["mjx_imitate.py", "train", "--steps", str(args.steps), "--envs", str(args.envs), "--evals", str(args.evals or 21),
               "--out", "run", "--hours", str(args.hours)]
        if args.warm:
            files["warm.pkl"] = enc(args.warm)
        if args.warm or args.warm_dataset:
            cmd += ["--warm", "warm.pkl"]
        for k, v in vars(args).items():   # the step 1-3 and brake flags given here
            flag = "--" + k.replace("_", "-")
            if v is None or flag not in ap._option_string_actions or k in ("user", "job", "slug", "warm", "warm_dataset", "hours", "steps",
                                                                          "envs", "evals", "out", "extra"):
                continue
            if isinstance(v, bool):
                cmd.append(flag if v else "--no-" + flag[2:])
            else:
                cmd += [flag] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
        cmd += shlex.split(args.extra)
        if args.landing_rate:   # the landing states, slimmed to what MjxImitate reads (it makes them float32 anyway)
            import numpy as np
            z = np.load(HERE / "research" / "results" / "landing_starts.npz")
            buf = io.BytesIO()
            np.savez(buf, qpos=z["qpos"].astype(np.float32), qvel=z["qvel"].astype(np.float32), clip=z["clip"], tau0=z["tau0"].astype(np.float32))
            files["research/results/landing_starts.npz"] = pack(buf.getvalue())
    (out / "kernel.py").write_text(TEMPLATE.format(slug=args.slug, files=json.dumps(files), cmd=json.dumps(cmd),
                                                   find_warm=bool(args.warm_dataset)))
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{args.user}/{args.slug}", title=args.slug, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[args.warm_dataset] if args.job == "imitate" and args.warm_dataset else [], competition_sources=[], kernel_sources=[]), indent=1))
    size = (out / "kernel.py").stat().st_size
    print(f"kernel in {out} ({size / 1e6:.2f} MB; command: {' '.join(cmd)}): push with  kaggle kernels push -p {out}"
          + (" --accelerator NvidiaTeslaT4" if args.job == "imitate" else ""))
    if size > 1_000_000:
        print("WARNING: over 1 MB, Kaggle may refuse the script", flush=True)


if __name__ == "__main__":
    main()
