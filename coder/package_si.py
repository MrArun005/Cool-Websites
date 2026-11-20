"""Package the best kept round of the self-improvement loop into models/<name>/ (model + its data + exams).

    python coder/package_si.py --name bolt-coder-si-0.5B
Safe to run while the loop is still going: it takes the last KEPT round in runs/si/history.json.
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bolt-coder-si-0.5B")
    ap.add_argument("--si", default=str(HERE / "runs" / "si"))
    a = ap.parse_args()
    si, out = Path(a.si), REPO / "models" / a.name
    hist = json.loads((si / "history.json").read_text())
    best = [h for h in hist if h["kept"]][-1]
    if best["round"] == 0:
        raise SystemExit("no round beat the base model yet")
    r = best["round"]
    if out.exists():
        shutil.rmtree(out)
    (out / "adapter").mkdir(parents=True)
    w = load_file(Path(best["adapter"]) / "adapter_model.safetensors")
    save_file({k: v.half().contiguous() for k, v in w.items()}, out / "adapter" / "adapter_model.safetensors")
    shutil.copy(Path(best["adapter"]) / "adapter_config.json", out / "adapter")
    for k in range(r + 1):
        rd = si / f"round_{k}"
        dst = out / f"round_{k}"
        dst.mkdir()
        for f in ("exam.json", "exam.details.json", "data/stats.json"):
            if (rd / f).exists():
                shutil.copy(rd / f, dst / Path(f).name)
        if (rd / "tries.jsonl").exists():   # every try, passed or not: the raw practice data
            with open(rd / "tries.jsonl", "rb") as i, gzip.open(dst / "tries.jsonl.gz", "wb") as o:
                shutil.copyfileobj(i, o)
        if (rd / "data" / "train.jsonl").exists():   # exactly what that round learned from
            with open(rd / "data" / "train.jsonl", "rb") as i, gzip.open(dst / "train.jsonl.gz", "wb") as o:
                shutil.copyfileobj(i, o)
        if (si / f"r{k}" / "train.log").exists():
            shutil.copy(si / f"r{k}" / "train.log", dst / "train.log")
    shutil.copy(si / "history.json", out / "history.json")
    shutil.copy(si / "loop.log", out / "loop.log")

    rows = "\n".join(
        f"| {h['round']} | {h['humaneval']:.1%} | {h['mbpp']:.1%} | {h['passes']} | {h['repo_loss']:.3f} | "
        f"{h.get('solved', '–')}/{h.get('problems', '–')} | {h.get('solutions', '–')} | "
        f"{'kept' if h['kept'] else 'rejected'} |" for h in hist)
    card = f"""# {a.name}

Qwen/Qwen2.5-Coder-0.5B-Instruct that improved itself on CPU, by expert iteration (coder/self_improve.py): each round
it tried 474 MBPP practice problems 4 times, a sandbox ran every try against the problem's tests, and it learned only
from tries that passed (up to 2 per problem), mixed with 300 examples of this repo's own code. A round is kept only if
HumanEval + MBPP exam passes do not drop and the repo held-out loss is not more than 2% worse.
This adapter is round {r}.

## Rounds

Exams: HumanEval (164) and the first 200 MBPP test problems, never practiced on; greedy, pass@1. Repo loss = loss per
token on {104} held-out examples from files and commits never trained on (lower is better).

| round | HumanEval | MBPP | exam passes | repo loss | practice solved | verified solutions | |
|---|---|---|---|---|---|---|---|
{rows}

Round 0 is the plain base model. Every round's tries (passed or not), training data, train log and exam are in
round_<n>/.

## Use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "models/{a.name}/adapter").merge_and_unload()
msgs = [{{"role": "user", "content": "Write a function that returns the n-th Fibonacci number."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=256)[0][ids.shape[1]:], skip_special_tokens=True))
```

For Ollama / llama.cpp: `model.save_pretrained("merged"); tok.save_pretrained("merged")`, convert with llama.cpp's
`convert_hf_to_gguf.py merged --outtype q8_0`, then `ollama create` from a Modelfile `FROM ./merged-q8_0.gguf`.

## Run it again

`python3 tools/keepalive/keepalive.py start coder/jobs/si.json` (resumes where it stopped), then
`python coder/package_si.py --name {a.name}`.
"""
    (out / "MODEL.md").write_text(card)
    print(card)


if __name__ == "__main__":
    main()
