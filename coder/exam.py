"""Exam for bolt-coder: the same tests for the base model and a fine-tuned adapter.

  heldout   loss per token on coder/data/eval.jsonl (files and commits never trained on), by kind
  lines     line completion on held-out .py/.js files: the model sees ~768 tokens before and ~256 after a
            removed line (fill-in-the-middle) and must write that line; exact match and similarity
  humaneval OpenAI HumanEval (164 Python problems), greedy, pass@1; generated code runs in a subprocess
            with a 10 s limit. Checks that general coding did not get worse.

    python coder/exam.py --out coder/runs/exam_base.json
    python coder/exam.py --adapter coder/runs/c1/best --out coder/runs/exam_c1.json
"""
import argparse
import difflib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_lora import BF16, evaluate, load  # noqa: E402


def generate(model, tok, prompts, max_new, bs=8):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  generated {len(outs)}/{len(prompts)}", flush=True)
    return outs


def line_cases(n, seed):
    summary = json.loads((HERE / "data" / "summary.json").read_text())
    repo = HERE.parent
    files = [p for p in summary["eval_files"] if p.endswith((".py", ".js"))]
    rng, cases = random.Random(seed), []
    texts = {p: (repo / p).read_text(errors="replace").split("\n") for p in files}
    pool = [(p, i) for p in files for i, l in enumerate(texts[p])
            if len(l.strip()) >= 12 and not l.strip().startswith(("#", "//", '"""'))]
    for p, i in rng.sample(pool, min(n, len(pool))):
        lines = texts[p]
        cases.append({"path": p, "line": i, "target": lines[i],
                      "pre": "\n".join(lines[max(0, i - 60):i]) + "\n", "suf": "\n" + "\n".join(lines[i + 1:i + 20])})
    return cases


def run_lines(model, tok, cases):
    prompts = [f"<|repo_name|>Cool-Websites\n<|file_sep|>{c['path']}\n<|fim_prefix|>{c['pre']}<|fim_suffix|>"
               f"{c['suf']}<|fim_middle|>" for c in cases]
    outs = generate(model, tok, prompts, 40)
    exact, sim = 0, 0.0
    for c, o in zip(cases, outs):
        got = o.split("\n")[0]
        exact += got.strip() == c["target"].strip()
        sim += difflib.SequenceMatcher(None, got.strip(), c["target"].strip()).ratio()
        c["got"] = got
    return {"n": len(cases), "exact": exact / len(cases), "similarity": sim / len(cases)}, cases


def passes(prog, timeout=10):
    """Run a program (model-written code + its tests) in a fresh interpreter in an empty temp dir; True if exit 0."""
    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))   # 2 GB memory
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))  # 1 MB files
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(prog)
        try:
            return subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=timeout,
                                  preexec_fn=limits).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)(```|$)", text, re.S)
    return (m.group(1) if m else text).replace("\r\n", "\n")


def mbpp_problems(split):
    import pyarrow.parquet as pq
    return pq.read_table(HERE / "data" / f"mbpp_{split}.parquet").to_pylist()


def mbpp_prompt(tok, p):
    return tok.apply_chat_template([{"role": "user", "content":
        f"{p['text']}\nYour code should pass this test:\n{p['test_list'][0]}\n"
        "Reply with the Python code in one ```python code block."}], tokenize=False, add_generation_prompt=True)


def mbpp_program(p, code):
    return f"{p['test_setup_code']}\n{code}\n\n" + "\n".join(p["test_list"]) + "\n"


def run_mbpp(model, tok, n):
    probs = mbpp_problems("test")[:n]
    outs = generate(model, tok, [mbpp_prompt(tok, p) for p in probs], 384)
    ok = [passes(mbpp_program(p, extract_code(o))) for p, o in zip(probs, outs)]
    return {"n": len(probs), "pass@1": sum(ok) / len(probs)}, [{"task": p["task_id"], "pass": k} for p, k in zip(probs, ok)]


def run_humaneval(model, tok, n):
    import pyarrow.parquet as pq
    probs = pq.read_table(HERE / "data" / "humaneval.parquet").to_pylist()[:n]
    prompts = [tok.apply_chat_template([{"role": "user", "content":
               "Complete this Python function. Reply with the whole function in one ```python code block.\n\n"
               f"```python\n{p['prompt']}```"}], tokenize=False, add_generation_prompt=True) for p in probs]
    outs = generate(model, tok, prompts, 512)
    passed, details = 0, []
    for p, o in zip(probs, outs):
        code = extract_code(o)
        if f"def {p['entry_point']}" not in code:
            code = p["prompt"] + code
        else:
            code = "\n".join(l for l in p["prompt"].split("\n") if l.startswith(("import", "from"))) + "\n" + code
        prog = f"{code}\n\n{p['test']}\n\ncheck({p['entry_point']})\n"
        ok = passes(prog)
        passed += ok
        details.append({"task": p["task_id"], "pass": ok})
    return {"n": len(probs), "pass@1": passed / len(probs)}, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lines", type=int, default=200)
    ap.add_argument("--humaneval", type=int, default=164)
    ap.add_argument("--mbpp", type=int, default=200, help="first N MBPP test problems (never practiced on)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heldout-limit", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.base)
    BF16["on"] = True
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()
    t0, res = time.time(), {"base": a.base, "adapter": a.adapter}

    raw = [json.loads(l) for l in open(HERE / "data" / "eval.jsonl")][:a.heldout_limit]
    rows = load(HERE / "data" / "eval.jsonl", tok, 1024, a.heldout_limit)
    res["heldout_loss"] = {"all": evaluate(model, rows, 4, tok.pad_token_id)}
    for k in ("file", "fim", "commit"):
        sub = [r for r, x in zip(rows, raw) if x["kind"] == k]
        res["heldout_loss"][k] = evaluate(model, sub, 4, tok.pad_token_id)
    print("heldout", res["heldout_loss"], flush=True)
    if a.lines:
        res["lines"], cases = run_lines(model, tok, line_cases(a.lines, a.seed))
        print("lines", res["lines"], flush=True)
    if a.humaneval:
        res["humaneval"], he = run_humaneval(model, tok, a.humaneval)
        print("humaneval", res["humaneval"], flush=True)
    mb = []
    if a.mbpp:
        res["mbpp"], mb = run_mbpp(model, tok, a.mbpp)
        print("mbpp", res["mbpp"], flush=True)
    res["seconds"] = round(time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    detail = Path(a.out).with_suffix(".details.json")
    detail.write_text(json.dumps({"lines": cases if a.lines else [], "humaneval": he if a.humaneval else [], "mbpp": mb}, indent=1))
    print("done:", json.dumps(res))


if __name__ == "__main__":
    main()
