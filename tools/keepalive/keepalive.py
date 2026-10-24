#!/usr/bin/env python3
"""keepalive: keep long jobs running until they are really finished.

Give it one or more commands (a training run, a watcher, a data download...). It runs them in
the background, away from your terminal or chat, and restarts any that die before they are done.
A job can resume from its latest save instead of starting over. Only the Python standard
library is needed, so it works in any project.

    python keepalive.py start job.json     # launch in the background and return
    python keepalive.py status job.json    # what is running, restarts, last exit
    python keepalive.py stop job.json      # the only way to stop it on purpose
    python keepalive.py run job.json       # same as start, but in the foreground

job.json (paths are relative to the job file; see examples/):

    {
      "name": "bolt-walk-run7",
      "workdir": "../../train",
      "env": {"MUJOCO_GL": "osmesa"},
      "check_every": 30,
      "jobs": [
        {"name": "train",
         "cmd": ["python", "-u", "train.py", "--name", "bolt7", "--steps", "20000000"],
         "resume_cmd": ["python", "-u", "train.py", "--name", "bolt7", "--steps", "20000000",
                        "--resume", "{latest}"],
         "latest": "runs/bolt7/checkpoints/policy_*_steps.zip",
         "log": "logs/train.log",
         "done_when": {"exit_code": 0, "file": "runs/bolt7/policy.zip"}},
        {"name": "watch", "cmd": ["python", "-u", "watch.py", "..."],
         "done_when": {"log_contains": "^finished"}}
      ]
    }

A job is finished when it exits and every condition in done_when holds (exit_code, file
exists, its log matches log_contains). Any other exit is a crash: it is restarted, using
resume_cmd with {latest} set to the newest file matching "latest" when one exists. Restarts
never give up; if a job keeps crashing quickly, the wait between tries grows to max_backoff
(5 minutes) so it doesn't spin. When every job is finished, keepalive exits.
State and keepalive's own log live in <state_dir> (default: .keepalive-<name> next to job.json).
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def natural_key(path):
    """policy_10000_steps sorts after policy_9000_steps."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


class Job:
    def __init__(self, spec, workdir, state_dir):
        self.name = spec["name"]
        self.cmd = spec["cmd"]
        self.resume_cmd = spec.get("resume_cmd")
        self.latest_glob = spec.get("latest")
        self.done_when = spec.get("done_when", {"exit_code": 0})
        self.workdir = workdir
        self.log = (workdir / spec["log"]) if spec.get("log") else state_dir / f"{self.name}.log"
        self.proc, self.done, self.restarts = None, False, 0
        self.last_exit, self.started_at, self.next_try = None, None, 0.0
        self.backoff, self.log_offset = 0.0, 0

    def latest(self):
        if not self.latest_glob:
            return None
        found = sorted(glob.glob(str(self.workdir / self.latest_glob)), key=natural_key)
        return found[-1] if found else None

    def command(self):
        latest = self.latest()
        if latest and self.resume_cmd:
            return [part.replace("{latest}", latest) for part in self.resume_cmd], latest
        return list(self.cmd), None

    def start(self, env):
        cmd, latest = self.command()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log_offset = self.log.stat().st_size if self.log.exists() else 0  # earlier runs' output
        out = open(self.log, "a")
        self.proc = subprocess.Popen(cmd, cwd=self.workdir, env=env, stdout=out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        out.close()
        self.started_at = time.time()
        return cmd, latest

    def finished(self, code):
        w = self.done_when
        if "exit_code" in w and code != w["exit_code"]:
            return False
        if "file" in w and not (self.workdir / w["file"]).exists():
            return False
        if "log_contains" in w:  # only this run's output counts, not a "done" left from an earlier run
            with open(self.log, "rb") as f:
                f.seek(self.log_offset)
                text = f.read().decode(errors="ignore")
            if not re.search(w["log_contains"], text, re.MULTILINE):
                return False
        return True

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)  # the job and anything it started
            except ProcessLookupError:
                pass

    def wait_gone(self, grace=15):
        """Give a stopped job time to save and exit, then force it."""
        if not self.proc:
            return
        try:
            self.proc.wait(grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()

    def state(self):
        running = self.proc is not None and self.proc.poll() is None
        return dict(name=self.name, running=running, pid=self.proc.pid if running else None, done=self.done,
                    restarts=self.restarts, last_exit=self.last_exit, log=str(self.log))


class Keepalive:
    def __init__(self, job_file, check_every=None):
        self.job_file = Path(job_file).resolve()
        spec = json.loads(self.job_file.read_text())
        base = self.job_file.parent
        self.name = spec.get("name", self.job_file.stem)
        self.workdir = (base / spec.get("workdir", ".")).resolve()
        self.state_dir = (base / spec.get("state_dir", f".keepalive-{self.name}")).resolve()
        self.check_every = check_every or spec.get("check_every", 30)
        self.min_backoff = spec.get("min_backoff", 5)
        self.max_backoff = spec.get("max_backoff", 300)
        self.env = dict(os.environ, **{k: str(v) for k, v in spec.get("env", {}).items()})
        self.jobs = [Job(j, self.workdir, self.state_dir) for j in spec["jobs"]]

    # files that let other processes see and steer a running keepalive
    @property
    def stop_file(self):
        return self.state_dir / "STOP"

    @property
    def state_file(self):
        return self.state_dir / "state.json"

    @property
    def log_file(self):
        return self.state_dir / "keepalive.log"

    def say(self, msg):
        line = f"{now()} keepalive[{self.name}]: {msg}"
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save_state(self, status):
        state = dict(name=self.name, status=status, pid=os.getpid(), updated=now(),
                     jobs=[j.state() for j in self.jobs])
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state_file)

    def run(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)
        # Ignore hang-ups and Ctrl-C from whatever started us; only the STOP file ends a run.
        stopping = []
        try:
            for sig in (signal.SIGHUP, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
        except ValueError:
            pass  # not the main thread (embedded use): rely on the STOP file
        self.say(f"supervising {len(self.jobs)} job(s) in {self.workdir}")
        for job in self.jobs:
            self._start(job)
        while True:
            if self.stop_file.exists() or stopping:
                self.say("stop requested: stopping jobs")
                for job in self.jobs:
                    job.stop()
                for job in self.jobs:
                    job.wait_gone()
                self.save_state("stopped")
                return 1
            for job in self.jobs:
                if job.done:
                    continue
                if job.proc is None:                    # waiting out a backoff
                    if time.time() >= job.next_try:
                        self._start(job)
                    continue
                code = job.proc.poll()
                if code is None:
                    continue
                job.last_exit, job.proc = code, None
                if job.finished(code):
                    job.done = True
                    self.say(f"{job.name} finished (exit {code})")
                    continue
                ran = time.time() - job.started_at
                # crashed fast again? wait longer before the next try (but never give up)
                job.backoff = self.min_backoff if ran > 120 else min(self.max_backoff, max(self.min_backoff, job.backoff * 2))
                job.next_try = time.time() + job.backoff
                job.restarts += 1
                self.say(f"{job.name} stopped unexpectedly (exit {code} after {ran:.0f} s); "
                         f"restart #{job.restarts} in {job.backoff:.0f} s")
            self.save_state("running")
            if all(j.done for j in self.jobs):
                self.say("all jobs finished")
                self.save_state("finished")
                return 0
            time.sleep(self.check_every)

    def _start(self, job):
        cmd, latest = job.start(self.env)
        how = f"resuming from {latest}" if latest else "from the start"
        self.say(f"started {job.name} {how} (pid {job.proc.pid}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["start", "run", "status", "stop"])
    ap.add_argument("job_file")
    args = ap.parse_args()
    k = Keepalive(args.job_file)

    if args.action == "run":
        sys.exit(k.run())
    if args.action == "start":
        k.state_dir.mkdir(parents=True, exist_ok=True)
        if k.state_file.exists():
            s = json.loads(k.state_file.read_text())
            if s.get("status") == "running" and _alive(s.get("pid")):
                sys.exit(f"already running (pid {s['pid']}); see: keepalive.py status {args.job_file}")
            orphans = [j["name"] for j in s.get("jobs", []) if j.get("running") and _alive(j.get("pid"))]
            if orphans:  # keepalive itself was killed but its jobs live on: don't start duplicates
                sys.exit(f"jobs from the last run are still running unsupervised: {', '.join(orphans)}. "
                         f"Stop them (see status for pids) before starting again.")
        with open(k.state_dir / "keepalive.err", "a") as err:  # keepalive.log is written by say()
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "run", str(k.job_file)],
                                 stdout=subprocess.DEVNULL, stderr=err, stdin=subprocess.DEVNULL,
                                 start_new_session=True)  # its own session: survives the terminal/chat
        print(f"keepalive started (pid {p.pid}); log: {k.log_file}")
    elif args.action == "status":
        if not k.state_file.exists():
            sys.exit("not started yet")
        s = json.loads(k.state_file.read_text())
        alive = _alive(s["pid"])
        print(f"{s['name']}: {s['status']}{'' if alive or s['status'] != 'running' else ' (keepalive itself is gone!)'}"
              f", updated {s['updated']}")
        for j in s["jobs"]:
            what = ("done" if j["done"] else "running, pid %s" % j["pid"] if j["running"]
                    else "stopped" if s["status"] == "stopped" else "waiting to restart")
            print(f"  {j['name']:<12} {what:<22} restarts {j['restarts']:<4} last exit {j['last_exit']}  log {j['log']}")
    elif args.action == "stop":
        k.stop_file.write_text(now() + "\n")
        print(f"stop requested; keepalive stops its jobs within {k.check_every} s")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError):
        return False


if __name__ == "__main__":
    main()
