# keepalive

Keeps long jobs (model training, simulations, big downloads, a watcher next to them) running
until they are really finished, no matter what happens in the terminal or chat that started them.

- Runs each job in its own background session, so closing a terminal, a chat turn ending, or
  Ctrl-C in the launching shell doesn't stop it.
- If a job dies before it's done (crash, killed, out of memory), it's restarted. Jobs that save
  checkpoints resume from the newest one instead of starting over.
- It never gives up. If a job keeps crashing straight away, the wait between tries grows to
  5 minutes so it doesn't spin.
- It stops only when every job is finished, or when you run `stop`.
- One file, Python standard library only: copy `keepalive.py` into any project.

```
python3 keepalive.py start job.json    # launch in the background and return
python3 keepalive.py status job.json   # running / done / restarts / last exit code / log paths
python3 keepalive.py stop job.json     # the only way to stop it on purpose
python3 keepalive.py run job.json      # same as start, in the foreground (for debugging)
```

## The job file

```json
{
  "name": "my-training",
  "workdir": "../train",
  "env": {"MUJOCO_GL": "osmesa"},
  "jobs": [
    {"name": "train",
     "cmd":        ["python", "-u", "train.py", "--steps", "20000000"],
     "resume_cmd": ["python", "-u", "train.py", "--steps", "20000000", "--resume", "{latest}"],
     "latest": "runs/x/checkpoints/*.zip",
     "log": "runs/x/train.log",
     "done_when": {"exit_code": 0, "file": "runs/x/policy.zip", "log_contains": "^done:"}}
  ]
}
```

| key | meaning |
|---|---|
| `workdir` | where the jobs run, relative to the job file |
| `env` | extra environment variables for every job |
| `cmd` | the command, as a list of words |
| `resume_cmd` | used instead of `cmd` once a file matching `latest` exists; `{latest}` becomes the newest one (`save_10` counts as newer than `save_9`) |
| `log` | where the job's output goes (default: inside the state folder) |
| `done_when` | the job is finished only when it exits and **all** of these hold: `exit_code`, `file` exists, `log_contains` (a regex matched against this run's output only) |
| `check_every`, `min_backoff`, `max_backoff` | seconds between checks (30) and the restart wait range (5 to 300) |
| `state_dir` | where `state.json`, `keepalive.log` and `STOP` live (default `.keepalive-<name>` next to the job file) |

Any exit that doesn't meet `done_when` counts as a crash. So pick a marker that only a
successful run produces (a final file, a "done" line). An exit code of 0 alone isn't enough
for jobs that can quit early.

## Rules learned the hard way

- Smoke-test the job for a minute in the foreground before handing it to keepalive.
- Make the job save checkpoints and accept a `--resume` path; otherwise every restart starts over.
- Set everything the job needs in `env` (e.g. `MUJOCO_GL=osmesa` for rendering without a screen).
  A background job doesn't get your shell's settings, and it will crash-loop without them.
- Activate the project's virtualenv before `start` (or use the venv's python path in `cmd`).
- Don't edit a script while a running job is executing it. Bash reads scripts as it goes.
- keepalive can't survive the machine itself shutting down (e.g. an idle cloud container being
  reclaimed). After a restart, run `start` again and the jobs resume from their checkpoints.

Examples: `examples/bolt-walk-training.json` (the Bolt dog's walk training plus its watcher),
`examples/any-script.json`. Tests: `python3 -m unittest test_keepalive.py`.
