"""Phase 2: bolt-coder improves itself on problems whose answers a program can check (expert iteration).

Each round r = 1, 2, ...:
  1 practice  the current model tries every MBPP practice problem (train + validation + prompt splits, 474 problems)
              --k times with sampling (temperature --temp)
  2 check     every try runs against the problem's asserts in a sandbox (exam.passes: fresh interpreter, empty
              temp dir, time, memory and file-size limits). Only tries that pass are kept: up to --keep different
              passing solutions per problem.
  3 learn     LoRA training from the previous round's adapter on those verified solutions, mixed with --replay
              examples of this repo's own data so it does not forget the codebase
  4 exam      coder/exam.py: HumanEval and MBPP test problems (never practiced on) + the repo held-out loss
  5 keep?     the round is kept only if HumanEval + MBPP passes do not drop and the repo held-out loss is not more
              than --loss-slack worse than the round it builds on; otherwise the loop stops and the last kept
              adapter is the result.
Every step writes its output under runs/si/round_<r>/ and is skipped if that output exists, so the whole loop can be
stopped and restarted any time (keepalive: coder/jobs/si.json).

    python coder/self_improve.py --rounds 3                      # from the plain base model
    python coder/self_improve.py --start coder/runs/c1/best --start-exam coder/runs/exam_c1.json --rounds 3
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PY = sys.executable


def log(msg, f):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f, "a") as fh:
        fh.write(line + "\n")


def practice(adapter, out, k, temp, base, limit):
    """Sample k tries per practice problem with the adapter, check them, write tries.jsonl."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import exam
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    if adapter != "base":
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    probs = [p for s in ("train", "validation", "prompt") for p in exam.mbpp_problems(s)][:limit]
    torch.manual_seed(0)
    part = out.with_suffix(".part")
    done = {json.loads(l)["task"] for l in open(part)} if part.exists() else set()
    with open(part, "a") as f:
        for i in range(0, len(probs), 4):    # 4 problems x k tries per generate call
            batch = [p for p in probs[i:i + 4] if p["task_id"] not in done]
            if not batch:
                continue
            prompts = [exam.mbpp_prompt(tok, p) for p in batch for _ in range(k)]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                g = model.generate(**enc, max_new_tokens=384, do_sample=True, temperature=temp, top_p=0.95,
                                   pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for j, p in enumerate(batch):
                tries = []
                for o in outs[j * k:(j + 1) * k]:
                    code = exam.extract_code(o).strip()
                    tries.append({"code": code, "pass": exam.passes(exam.mbpp_program(p, code), timeout=5)})
                f.write(json.dumps({"task": p["task_id"], "tries": tries}) + "\n")
                f.flush()
            print(f"  practice {min(i + 4, len(probs))}/{len(probs)}", flush=True)
    part.rename(out)


def build_data(tries_file, data_dir, keep, replay, seed):
    """Verified solutions as chat examples + replay of the repo's own training data; eval = the repo held-out set."""
    from transformers import AutoTokenizer
    import exam
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    probs = {p["task_id"]: p for s in ("train", "validation", "prompt") for p in exam.mbpp_problems(s)}
    rows, solved, tried, attempted = [], 0, 0, 0
    for line in open(tries_file):
        t = json.loads(line)
        attempted += 1
        tried += len(t["tries"])
        good = sorted({x["code"] for x in t["tries"] if x["pass"]}, key=len)[:keep]
        solved += bool(good)
        for code in good:
            rows.append({"kind": "solution", "path": f"mbpp/{t['task']}", "prompt": exam.mbpp_prompt(tok, probs[t["task"]]),
                         "completion": f"```python\n{code}\n```<|im_end|>\n"})
    n_sol = len(rows)
    repo = [json.loads(l) for l in open(HERE / "data" / "train.jsonl")]
    rows += random.Random(seed).sample(repo, min(replay, len(repo)))
    random.Random(seed).shuffle(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "train.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (data_dir / "eval.jsonl").write_text((HERE / "data" / "eval.jsonl").read_text())
    stats = {"problems": attempted, "solved": solved, "solve_rate": solved / max(1, attempted),
             "tries": tried, "solutions": n_sol, "replay": len(rows) - n_sol}
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def score(exam_file):
    e = json.loads(Path(exam_file).read_text())
    he, mb = e["humaneval"], e["mbpp"]
    return {"passes": round(he["pass@1"] * he["n"] + mb["pass@1"] * mb["n"]), "humaneval": he["pass@1"],
            "mbpp": mb["pass@1"], "repo_loss": e["heldout_loss"]["all"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="base", help="adapter to start from, or 'base' for the plain base model")
    ap.add_argument("--start-exam", default=None, help="exam.py result of the start model (run if missing)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--out", default=str(HERE / "runs" / "si"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--replay", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--loss-slack", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None, help="practice problems (smoke tests)")
    ap.add_argument("--exam-args", default="", help="extra exam.py args (smoke tests)")
    ap.add_argument("--train-args", default="", help="extra train_lora.py args (smoke tests)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = out / "loop.log"
    history_f = out / "history.json"
    if a.start_exam is None:
        a.start_exam = str(out / "round_0" / "exam.json")
    if not Path(a.start_exam).exists():
        log(f"round 0: exam of the start model {a.start}", L)
        Path(a.start_exam).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([PY, "-u", str(HERE / "exam.py"), *([] if a.start == "base" else ["--adapter", a.start]),
                        "--out", a.start_exam, "--lines", "0", *a.exam_args.split()], check=True)
    history = json.loads(history_f.read_text()) if history_f.exists() else \
        [{"round": 0, "adapter": a.start, "exam": a.start_exam, **score(a.start_exam), "kept": True}]
    log(f"start: {history[-1]}", L)

    stopped = not history[-1]["kept"]
    for r in range(len(history), 0 if stopped else a.rounds + 1):
        prev = [h for h in history if h["kept"]][-1]
        rd = out / f"round_{r}"
        rd.mkdir(exist_ok=True)
        tries = rd / "tries.jsonl"
        if not tries.exists():
            log(f"round {r}: practice from {prev['adapter']} (k={a.k}, temp={a.temp})", L)
            practice(prev["adapter"], tries, a.k, a.temp, a.base, a.limit)
        stats = build_data(tries, rd / "data", a.keep, a.replay, r)
        log(f"round {r}: solved {stats['solved']}/{stats['problems']} practice problems "
            f"({stats['solve_rate']:.1%}), {stats['solutions']} verified solutions + {stats['replay']} replay", L)
        adapter = out / f"r{r}" / "best"
        if not (out / f"r{r}" / "final").exists():
            log(f"round {r}: learn", L)
            cmd = [PY, "-u", str(HERE / "train_lora.py"), "--name", f"r{r}", "--runs", str(out), "--data",
                   str(rd / "data"), *([] if prev["adapter"] == "base" else ["--init", prev["adapter"]]), "--lr", str(a.lr), "--epochs", str(a.epochs),
                   "--eval-every", "10", "--save-every", "5", *a.train_args.split()]
            ck = sorted((out / f"r{r}" / "checkpoints").glob("step_*"), key=lambda p: int(p.name[5:]))
            if ck:
                cmd += ["--resume", str(ck[-1])]
            subprocess.run(cmd, check=True)
        if not adapter.exists():
            adapter = out / f"r{r}" / "final"
        ex = rd / "exam.json"
        if not ex.exists():
            log(f"round {r}: exam", L)
            subprocess.run([PY, "-u", str(HERE / "exam.py"), "--adapter", str(adapter), "--out", str(ex),
                            "--lines", "0", *a.exam_args.split()], check=True)
        s = score(ex)
        kept = s["passes"] >= prev["passes"] and s["repo_loss"] <= prev["repo_loss"] * (1 + a.loss_slack)
        history.append({"round": r, "adapter": str(adapter), "exam": str(ex), **s, "kept": kept, **stats})
        history_f.write_text(json.dumps(history, indent=1))
        log(f"round {r}: HumanEval {s['humaneval']:.1%}, MBPP {s['mbpp']:.1%}, repo loss {s['repo_loss']:.4f} "
            f"vs round {prev['round']}: {prev['humaneval']:.1%}, {prev['mbpp']:.1%}, {prev['repo_loss']:.4f} -> "
            f"{'KEPT' if kept else 'not better, stop'}", L)
        if not kept:
            break
    best = [h for h in history if h["kept"]][-1]
    log(f"done: best round {best['round']} adapter {best['adapter']}", L)


if __name__ == "__main__":
    main()
