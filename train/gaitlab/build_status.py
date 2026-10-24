"""Write status.json for the Bolt Gait Lab page: which training jobs are running right now."""
import datetime as dt, json, os, re
from pathlib import Path
TRAIN = Path("/home/user/Cool-Websites/train")
OUT = Path(__file__).parent
TITLES = {"bolt8b": "Run 8b · Walk", "bolt9c": "Run 9c · Stand, sit, lie", "bolt10": "Run 10 · Turning (walk left and right on command)", "bolt10b": "Run 10b · Turning walk, practising starts from the real standing brain"}
runs = []
for job_file in sorted((TRAIN / "jobs").glob("*.json")):
    job = json.loads(job_file.read_text())
    state = TRAIN / "jobs" / f".keepalive-{job['name']}" / "state.json"
    if not state.exists():
        continue
    st = json.loads(state.read_text())
    if st.get("status") != "running":
        continue
    cmd = job["jobs"][0]["cmd"]
    name = cmd[cmd.index("--name") + 1]
    log = TRAIN / "runs" / name / "train.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    num = lambda k: (lambda m: float(m[-1]) if m else None)(re.findall(rf"\|\s+{k}\s+\|\s+([-\d.e+]+)", text))
    data = TRAIN / "runs" / name / "watch" / "data.json"
    cks = json.loads(data.read_text())["checkpoints"] if data.exists() else []
    runs.append(dict(title=TITLES.get(name, name), steps=int(num("total_timesteps") or 0),
                     target=int(cmd[cmd.index("--steps") + 1]), fps=num("fps"),
                     restarts=sum(j["restarts"] for j in st["jobs"]),
                     exam=cks[-1]["exam"] if cks else None, exam_steps=cks[-1]["steps"] if cks else None))
(OUT / "status.json").write_text(json.dumps(dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                 load=round(os.getloadavg()[0], 1), running=runs), indent=1))
print(f"{len(runs)} running")
