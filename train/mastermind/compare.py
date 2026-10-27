"""The results table: switch success of each column (rate +- binomial standard error sqrt(p(1-p)/n) over the N tries
reached), every column the sum of one or more result files (e.g. seed sets 1000 and 5000 -> N=60).

    python mastermind/compare.py mastermind/results/student.json [more.json ...] > table.md   (v2 layout: manager 18b, 18e first)
    python mastermind/compare.py --col "old student=a.json,b.json" --col "new student=c.json,d.json" [--skip still:]
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# as REPORT.md: sit/lie count once, as "pose reached" (the plain rows only say Bolt did not fall)
EXACT_SKIP = {"walk->sit", "sit->lie"}


def rows(path):
    return {r["switch"]: r for r in json.loads(Path(path).read_text())["switches"]}


def merged(paths):
    out = OrderedDict()
    for p in paths:
        for k, r in rows(p).items():
            o = out.setdefault(k, dict(switch=k, ok=0, tries=0))
            o["ok"] += r["ok"]
            o["tries"] += r["tries"]
    return out


def cell(r):
    if not r or not r["tries"]:
        return "-"
    p, n = r["ok"] / r["tries"], r["tries"]
    return f"{r['ok']}/{n} {p:.0%} ± {round(100 * (p * (1 - p) / n) ** 0.5)}"


def table(data, skip=(), only=None):
    keys = []
    for _, d in data:
        keys += [k for k in d if k not in keys]
    keys = [k for k in keys if not any(k.startswith(s) for s in skip) and k not in EXACT_SKIP and (only is None or k in only)]
    lines = ["| switch | " + " | ".join(n for n, _ in data) + " |", "|---" * (len(data) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(cell(d.get(k)) for _, d in data) + " |")
    pooled = []
    for _, d in data:
        ok, n = sum(d[k]["ok"] for k in keys if k in d), sum(d[k]["tries"] for k in keys if k in d)
        p = ok / n if n else 0
        pooled.append(f"**{ok}/{n} {p:.0%} ± {round(100 * (p * (1 - p) / max(n, 1)) ** 0.5)}**")
    lines.append("| **pooled** | " + " | ".join(pooled) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--col", action="append", default=[], help="label=file1,file2,...")
    ap.add_argument("--skip", nargs="*", default=["still:", "shove -> fall"])
    a = ap.parse_args()
    if a.col:
        data = [(c.split("=", 1)[0], merged(c.split("=", 1)[1].split(","))) for c in a.col]
    else:
        cols = [("manager 18b", HERE / "results" / "manager_18b_n30.json"), ("manager 18e", HERE / "results" / "manager_18e_n30.json")]
        cols += [(Path(p).stem, p) for p in a.files]
        data = [(name, merged([p])) for name, p in cols]
    print(table(data, a.skip))


if __name__ == "__main__":
    main()
