"""Package a finished bolt-coder run into models/<name>/ (the repo rule: keep every trained model AND its data).

  adapter/        the LoRA adapter (fp16 safetensors + adapter_config.json): load it on top of the base model
  data/           train.jsonl.gz, eval.jsonl.gz and summary.json exactly as trained on
  exam_base.json, exam_<run>.json (+ .details.json)   the exam of the base model and of this adapter
  train.log       the training log
  MODEL.md        the model card with the exam table

    python coder/package.py --run c1 --name bolt-coder-0.5B
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def row(name, r):
    h, l, he = r["heldout_loss"], r.get("lines", {}), r.get("humaneval", {})
    return (f"| {name} | {h['all']:.3f} | {h['file']:.3f} | {h['fim']:.3f} | {h['commit']:.3f} | "
            f"{l.get('exact', 0):.1%} | {l.get('similarity', 0):.2f} | {he.get('pass@1', 0):.1%} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pick", default="best", help="best or final")
    a = ap.parse_args()
    run, out = HERE / "runs" / a.run, REPO / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)

    src = run / a.pick
    (out / "adapter").mkdir(exist_ok=True)
    w = load_file(src / "adapter_model.safetensors")
    save_file({k: v.half().contiguous() for k, v in w.items()}, out / "adapter" / "adapter_model.safetensors")
    shutil.copy(src / "adapter_config.json", out / "adapter" / "adapter_config.json")
    state = json.loads((src / "state.json").read_text())

    (out / "data").mkdir(exist_ok=True)
    for f in ("train.jsonl", "eval.jsonl"):
        with open(HERE / "data" / f, "rb") as i, gzip.open(out / "data" / f"{f}.gz", "wb") as o:
            shutil.copyfileobj(i, o)
    shutil.copy(HERE / "data" / "summary.json", out / "data" / "summary.json")
    for f in ("exam_base.json", "exam_base.details.json", f"exam_{a.run}.json", f"exam_{a.run}.details.json"):
        shutil.copy(HERE / "runs" / f, out / f)
    shutil.copy(run / "train.log", out / "train.log")

    base = json.loads((HERE / "runs" / "exam_base.json").read_text())
    mine = json.loads((HERE / "runs" / f"exam_{a.run}.json").read_text())
    summ = json.loads((HERE / "data" / "summary.json").read_text())
    card = f"""# {a.name}

A small coding assistant for this repository: **{base['base']}** fine-tuned with LoRA (rank 16, all attention and
MLP projections) on this repo's own code, docs and commit history. Trained on CPU (4 cores) in this repo's cloud
container, run `{a.run}`, adapter from `{a.pick}/` (optimizer step {state['step']}, held-out loss {state.get('eval_loss', state['best']):.4f}).

## Exam (coder/exam.py)

Held-out = {summ['files_eval']} files never trained on (10% of files, picked by a hash of the path) and the commits
that touch only those files. Lower loss is better. Line completion: {mine['lines']['n']} lines removed from held-out
.py/.js files, filled in from the code around them. HumanEval: {mine['humaneval']['n']} general Python problems, pass@1.

| model | held-out loss | files | fill-in-middle | commits | lines exact | lines similarity | HumanEval |
|---|---|---|---|---|---|---|---|
{row('base ' + base['base'].split('/')[-1], base)}
{row(a.name, mine)}

## Data (data/)

{summ['files_train']} files for training, {summ['files_eval']} held out; {summ['train_tokens']:,} training tokens:
{summ['train_examples']['file']} file chunks and {summ['train_examples']['fim']} fill-in-the-middle chunks in Qwen's
repo format, plus {summ['train_examples']['commit']} commits (request = commit message, answer = the diff).
Files that looked like they held a secret were left out ({len(summ['left_out_possible_secret'])}). Rebuild with
`python coder/prepare_data.py`.

## Use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{base['base']}")
model = AutoModelForCausalLM.from_pretrained("{base['base']}")
model = PeftModel.from_pretrained(model, "models/{a.name}/adapter").merge_and_unload()
```

Autocomplete prompt (fill in the middle):
`<|repo_name|>Cool-Websites\\n<|file_sep|>train/foo.py\\n<|fim_prefix|>CODE BEFORE<|fim_suffix|>CODE AFTER<|fim_middle|>`

For Ollama / llama.cpp: save the merged model (`model.save_pretrained("merged")`, `tok.save_pretrained("merged")`),
then convert it with llama.cpp's `convert_hf_to_gguf.py merged --outtype q8_0` and `ollama create` from a Modelfile
`FROM ./merged-q8_0.gguf`. (Not done here: the merged model is ~1 GB, too big for git.)

## Retrain

```bash
python coder/prepare_data.py
python coder/train_lora.py --name {a.run} --epochs 2      # or: python3 tools/keepalive/keepalive.py start coder/jobs/{a.run}.json
python coder/exam.py --out coder/runs/exam_base.json
python coder/exam.py --adapter coder/runs/{a.run}/best --out coder/runs/exam_{a.run}.json
python coder/package.py --run {a.run} --name {a.name}
```
"""
    (out / "MODEL.md").write_text(card)
    print(card)


if __name__ == "__main__":
    main()
