"""Build the training and held-out data for bolt-coder from this repository.

Three kinds of examples, written to coder/data/{train,eval}.jsonl:
  file     repo source in Qwen2.5-Coder's repo format (<|repo_name|> / <|file_sep|>), cut into chunks of
           --chunk tokens; about half of them are turned into fill-in-the-middle examples
           (<|fim_prefix|> ... <|fim_suffix|> ... <|fim_middle|> ...) so the model learns to autocomplete.
  commit   chat example: the commit message as the request, the commit's diff as the answer (loss on the answer
           only). Commits whose diff is longer than --max-diff tokens are skipped.
Held-out split: about 10% of files (by a hash of the path, so the split never changes) and the commits that touch
only held-out files go to eval.jsonl. Files that look like they hold a secret are left out and listed.

    python coder/prepare_data.py            # writes coder/data/ and prints a summary
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
EXTS = {".py", ".js", ".ino", ".sh", ".md"}
SKIP = re.compile(r"(^|/)(vendor|node_modules|\.venv|bolt-deps)/|\.min\.js$")
SECRET = re.compile(
    r"(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_\w{40,}|sk-[A-Za-z0-9_-]{32,}|hf_[A-Za-z0-9]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(api[_-]?key|secret|token|passw(or)?d)\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{20,}['\"])", re.I)
SYSTEM = ("You are bolt-coder, a coding assistant for the Cool-Websites repository: MuJoCo + PPO training of "
          "the Bolt robot dog (train/), its browser pet (pet/), Arduino robot firmware (robot/) and tools.")


def held_out(path):
    return int(hashlib.sha1(path.encode()).hexdigest(), 16) % 10 == 0


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, errors="replace")


def source_files(max_bytes):
    out = []
    for p in git("ls-files").split("\n"):
        if not p or Path(p).suffix not in EXTS or SKIP.search(p) or p.startswith("coder/data/"):
            continue
        f = REPO / p
        if not f.is_file() or f.stat().st_size > max_bytes:
            continue
        out.append(p)
    return out


def file_examples(paths, tok, chunk, fim_rate, rng):
    ex = []
    for p in paths:
        text = (REPO / p).read_text(errors="replace")
        ids = tok(text, add_special_tokens=False).input_ids
        head = f"<|repo_name|>Cool-Websites\n<|file_sep|>{p}\n"
        for i in range(0, len(ids), chunk):
            piece = tok.decode(ids[i:i + chunk])
            if len(piece.strip()) < 200:
                continue
            if rng.random() < fim_rate and len(piece) > 400:
                # middle: a span of 1-8 whole lines somewhere in the piece
                lines = piece.split("\n")
                a = rng.randrange(1, max(2, len(lines) - 1))
                b = min(len(lines), a + rng.randint(1, 8))
                pre, mid, suf = "\n".join(lines[:a]) + "\n", "\n".join(lines[a:b]) + "\n", "\n".join(lines[b:])
                ex.append({"kind": "fim", "path": p,
                           "prompt": f"{head}<|fim_prefix|>{pre}<|fim_suffix|>{suf}<|fim_middle|>",
                           "completion": mid + "<|endoftext|>"})
            else:
                ex.append({"kind": "file", "path": p, "prompt": "", "completion": head + piece + "<|endoftext|>"})
    return ex


def commit_examples(tok, max_diff, allowed):
    ex = []
    for line in git("log", "--no-merges", "--format=%H").split():
        msg = git("log", "-1", "--format=%B", line).strip()
        files = [f for f in git("show", "--format=", "--name-only", line).split("\n") if f]
        files = [f for f in files if Path(f).suffix in EXTS and not SKIP.search(f)]
        if not files or not msg:
            continue
        diff = git("show", "--format=", "-U3", line, "--", *files)
        if SECRET.search(diff) or len(tok(diff).input_ids) > max_diff:
            continue
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Make this change in the repository and show it as a unified diff:\n\n{msg}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        split = "eval" if all(held_out(f) for f in files) else "train"
        ex.append({"kind": "commit", "split": split, "path": ",".join(files), "prompt": prompt,
                   "completion": diff.strip() + "\n<|im_end|>\n", "commit": line[:10]})
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "coder" / "data"))
    ap.add_argument("--chunk", type=int, default=896)
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--max-diff", type=int, default=896)
    ap.add_argument("--max-bytes", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    rng = random.Random(a.seed)

    paths, secret = [], []
    for p in source_files(a.max_bytes):
        (secret if SECRET.search((REPO / p).read_text(errors="replace")) else paths).append(p)
    train_p = [p for p in paths if not held_out(p)]
    eval_p = [p for p in paths if held_out(p)]
    train = file_examples(train_p, tok, a.chunk, a.fim_rate, rng)
    evals = file_examples(eval_p, tok, a.chunk, a.fim_rate, rng)
    for c in commit_examples(tok, a.max_diff, set(paths)):
        (evals if c.pop("split") == "eval" else train).append(c)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    tokens = lambda rows: sum(len(tok(r["prompt"] + r["completion"]).input_ids) for r in rows)
    kinds = lambda rows: {k: sum(r["kind"] == k for r in rows) for k in ("file", "fim", "commit")}
    summary = {"base": BASE, "files_train": len(train_p), "files_eval": len(eval_p), "eval_files": eval_p,
               "left_out_possible_secret": secret, "train_examples": kinds(train), "eval_examples": kinds(evals),
               "train_tokens": tokens(train), "eval_tokens": tokens(evals), "args": vars(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "eval_files"}, indent=1))


if __name__ == "__main__":
    main()
