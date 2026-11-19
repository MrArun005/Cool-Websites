"""LoRA fine-tune of Qwen2.5-Coder on coder/data (CPU or GPU), with checkpoints, resume and a held-out loss.

Loss is taken on the completion only (the FIM middle, the whole file chunk, the commit's diff), never the prompt.
Every --save-every optimizer steps: runs/<name>/checkpoints/step_<n>/ (adapter + optimizer + step). Every
--eval-every steps: held-out loss on eval.jsonl; the lowest is kept in runs/<name>/best/. At the end:
runs/<name>/final/ and a line "done: ..." in the log.

    python coder/train_lora.py --name c1 --epochs 2                       # fresh
    python coder/train_lora.py --name c1 --epochs 2 --resume runs/c1/checkpoints/step_40
    python coder/train_lora.py --name smoke --max-steps 2 --limit 8 --eval-limit 4 --save-every 1 --eval-every 2
"""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(path, tok, max_len, limit=None):
    rows = [json.loads(l) for l in open(path)][:limit]
    out = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False).input_ids if r["prompt"] else []
        c = tok(r["completion"], add_special_tokens=False).input_ids
        ids = (p + c)[:max_len]
        labels = ([-100] * len(p) + c)[:max_len]
        if any(l != -100 for l in labels):
            out.append((ids, labels))
    return out


def batches(rows, bs, pad):
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        n = max(len(x) for x, _ in b)
        ids = torch.full((len(b), n), pad)
        lab = torch.full((len(b), n), -100)
        att = torch.zeros((len(b), n), dtype=torch.long)
        for j, (x, y) in enumerate(b):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        yield ids, lab, att


BF16 = {"on": False}   # bfloat16 autocast: ~2x faster on CPUs with AMX/avx512_bf16, same loss


def token_loss(model, ids, lab, att):
    """Sum of next-token losses and number of predicted tokens (so averages are per token, not per batch)."""
    with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=BF16["on"]):
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                             ignore_index=-100, reduction="sum")
    return loss, int((tgt != -100).sum())


@torch.no_grad()
def evaluate(model, rows, bs, pad):
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, att in batches(sorted(rows, key=lambda r: len(r[0])), bs, pad):
        l, k = token_loss(model, ids, lab, att)
        tot, n = tot + l.item(), n + k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = epochs)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init", default=None, help="start from this adapter's weights (fresh optimizer and step 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="no bfloat16 autocast")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    BF16["on"] = not a.fp32
    run = Path(a.runs) / a.name
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train = load(Path(a.data) / "train.jsonl", tok, a.max_len, a.limit)
    evals = load(Path(a.data) / "eval.jsonl", tok, a.max_len, a.eval_limit)
    per_epoch = math.ceil(len(train) / (a.bs * a.accum))
    total = a.max_steps or math.ceil(per_epoch * a.epochs)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    step, best = 0, float("inf")
    if a.resume:
        model = PeftModel.from_pretrained(model, a.resume, is_trainable=True)
        st = json.loads((Path(a.resume) / "state.json").read_text())
        step, best = st["step"], st["best"]
    elif a.init:
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    if a.resume:
        o = torch.load(Path(a.resume) / "optim.pt", weights_only=False)
        opt.load_state_dict(o["opt"])
        sched.load_state_dict(o["sched"])

    log = open(run / "train.log", "a")
    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    say(f"start: step {step}/{total}, {len(train)} train / {len(evals)} eval examples, {per_epoch} steps/epoch, "
        f"device {dev}, args {json.dumps(vars(a))}")
    if step == 0 and evals:
        base_loss = evaluate(model, evals, a.bs * 2, pad)
        best = float("inf")
        say(f"eval: step 0 loss {base_loss:.4f} (" + (f"start {a.init}" if a.init else "base model") + ")")

    def order(epoch):   # the same order for an epoch on every resume
        idx = list(range(len(train)))
        random.Random(a.seed * 1000 + epoch).shuffle(idx)
        return idx

    def save(dirname, extra=None):
        d = run / dirname
        model.save_pretrained(d)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, d / "optim.pt")
        (d / "state.json").write_text(json.dumps({"step": step, "best": best, **(extra or {})}))
        return d

    model.train()
    t0, toks = time.time(), 0
    while step < total:
        epoch, k = divmod(step, per_epoch)
        idx = order(epoch)[k * a.bs * a.accum:(k + 1) * a.bs * a.accum]
        rows = [train[i] for i in idx]
        tot, n = 0.0, sum(sum(l != -100 for l in r[1][1:]) for r in rows)
        for ids, lab, att in batches(rows, a.bs, pad):
            l, _ = token_loss(model, ids.to(dev), lab.to(dev), att.to(dev))
            (l / n).backward()
            tot += l.item()
            toks += int(att.sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        say(f"step {step}/{total} epoch {step / per_epoch:.2f} loss {tot / n:.4f} lr {sched.get_last_lr()[0]:.2e} "
            f"{toks / (time.time() - t0):.0f} tok/s")
        if step % a.eval_every == 0 or step == total:
            el = evaluate(model, evals, a.bs * 2, pad)
            say(f"eval: step {step} loss {el:.4f}")
            if el < best:
                best = el
                if (run / "best").exists():
                    shutil.rmtree(run / "best")
                save("best", {"eval_loss": el})
                say(f"best: step {step} eval loss {el:.4f}")
        if step % a.save_every == 0 or step == total:
            save(f"checkpoints/step_{step}")
            for old in sorted((run / "checkpoints").glob("step_*"), key=lambda p: int(p.name[5:]))[:-2]:
                shutil.rmtree(old)
    save("final")
    say(f"done: {step} steps, best eval loss {best:.4f}")


if __name__ == "__main__":
    main()
