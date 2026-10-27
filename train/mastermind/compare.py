"""The results table: switch success of the skill manager (skills_report.json: dog-motion brain 18b; baseline_18e.json:
the same manager with run 18e) against the MasterMind student(s). Rate +- binomial standard error sqrt(p(1-p)/n)
over the N tries reached.

    python mastermind/compare.py mastermind/results/student.json [more.json ...] > table.md
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    return {r["switch"]: r for r in json.loads(Path(path).read_text())["switches"]}


def cell(r):
    if not r or not r["tries"]:
        return "-"
    p, n = r["ok"] / r["tries"], r["tries"]
    return f"{r['ok']}/{n} ({p:.0%} ± {(p * (1 - p) / n) ** 0.5:.0%})"


def main(paths):
    cols = [("manager 18b", HERE / "results" / "manager_18b_n30.json"), ("manager 18e", HERE / "results" / "manager_18e_n30.json")]
    cols += [(Path(p).stem, p) for p in paths]
    data = [(name, rows(p)) for name, p in cols]
    keys = list(data[0][1])
    print("| switch | " + " | ".join(n for n, _ in data) + " |")
    print("|---" * (len(data) + 1) + "|")
    for k in keys:
        print(f"| {k} | " + " | ".join(cell(d.get(k)) for _, d in data) + " |")
    print("| **all switches pooled** | " + " | ".join(
        f"{sum(r['ok'] for r in d.values())}/{sum(r['tries'] for r in d.values())}" for _, d in data) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
