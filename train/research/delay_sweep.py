"""Delay sweep (owner 30 Oct: "delay should be a dynamic value"): a brain's clean-jump rate on real hand-overs at every
control delay from 0 to 40 ms, steady and with step-to-step jitter, so no brain is judged at one delay only.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/delay_sweep.py ../models/bolt-dog-jump-delay-76M \
        --states research/results/handover_states_v8g_test.npz --out research/results/delay_sweep/76M.json

Per delay: research/handover_exam.py (fixed physics, motor brake on) at 30/40/50 cm; "jitter" rows add +-8 ms per
step and occasional spikes (bolt_env.start_delay). Reports the mean over heights and the worst delay ("floor").
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--states", required=True)
    ap.add_argument("--delays", type=float, nargs="+", default=[0, 10, 20, 30, 40])
    ap.add_argument("--heights", type=float, nargs="+", default=[0.30, 0.40, 0.50])
    ap.add_argument("--jitter-ms", type=float, default=8.0)
    ap.add_argument("--spike-p", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = [(ms, 0.0, 0.0) for ms in a.delays] + [(ms, a.jitter_ms, a.spike_p) for ms in a.delays if ms > 0]

    def one(r):
        ms, jit, sp = r
        cmd = [sys.executable, str(HERE / "handover_exam.py"), a.folder, "--states", a.states, "--motor-brake",
               "--latency-ms", str(ms), "--jitter-ms", str(jit), "--spike-p", str(sp), "--heights"] + [str(h) for h in a.heights]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE.parent).stdout
        rates = [int(m.group(1)) / int(m.group(2)) for m in re.finditer(r"cm: (\d+)/(\d+)", out)]
        return dict(delay_ms=ms, jitter_ms=jit, spike_p=sp, per_height=rates, mean=round(sum(rates) / max(1, len(rates)), 3))

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, runs))
    for r in rows:
        print(f"{r['delay_ms']:4.0f} ms {'jitter' if r['jitter_ms'] else 'steady':6s} " +
              " ".join(f"{100 * x:3.0f}" for x in r["per_height"]) + f"   mean {100 * r['mean']:.0f}%", flush=True)
    res = dict(folder=a.folder, states=a.states, heights=a.heights, rows=rows,
               floor=min(r["mean"] for r in rows), average=round(sum(r["mean"] for r in rows) / len(rows), 3),
               command="python " + " ".join(sys.argv))
    print(f"average {100 * res['average']:.0f}%, worst {100 * res['floor']:.0f}%")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
