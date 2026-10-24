---
name: keepalive
description: Start long-running jobs (model training, simulations, watchers, big downloads) so they keep running in the background, restart after crashes, resume from checkpoints, and are never stopped by chat turns or questions. Use whenever a job will take more than a few minutes, or the user says a run must not stop.
---

# Keep long jobs running with keepalive

The tool is `tools/keepalive/keepalive.py` (stdlib only; read `tools/keepalive/README.md`).

1. **Smoke-test first.** Run the job for about a minute in the foreground. Check that it starts,
   saves a checkpoint and resumes from one.
2. **Write a job file** next to the work, or copy `tools/keepalive/examples/*.json`:
   - `cmd`, plus `resume_cmd` with `{latest}` and a `latest` glob for its checkpoints;
   - `done_when` with a marker only a successful run produces, such as a final file or a `^done:`
     line (an exit code of 0 alone is not enough);
   - `env` with everything a background process needs (e.g. `MUJOCO_GL=osmesa`).
3. **Start it with the project's venv active:** `python3 tools/keepalive/keepalive.py start <job.json>`.
   It detaches into its own session; a chat turn ending or the user asking questions won't stop it.
4. **Check on it** with `status`. Watch the job's log (or `keepalive.log` for restarts) with a
   Monitor instead of polling with sleep.
5. **Stop it only when the user asks you to stop it:** `stop <job.json>`.
   Never kill job processes by hand. `pkill -f <pattern>` also matches the shell running it.

Tell the user one limit plainly: if the machine itself shuts down (an idle cloud container is
reclaimed), nothing survives. Running `start` again resumes from the checkpoints.
