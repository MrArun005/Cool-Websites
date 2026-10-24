"""Tests for keepalive.py:  python3 -m unittest test_keepalive.py"""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from keepalive import Keepalive

# A fake training job: saves a checkpoint each run, crashes on its first two runs, finishes on the
# third. With --resume it must be handed the newest checkpoint.
FAKE_TRAIN = r'''
import sys, pathlib
ck = pathlib.Path("ckpt"); ck.mkdir(exist_ok=True)
runs = len(list(ck.glob("save_*.txt")))
resumed = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else None
with open("resumes.txt", "a") as f: f.write(f"{resumed}\n")
(ck / f"save_{runs + 1}.txt").write_text("x")
if runs + 1 < 3:
    sys.exit(1)          # crash
pathlib.Path("final.txt").write_text("done")
print("done: trained")
'''


class KeepaliveTest(unittest.TestCase):
    def make(self, jobs, **extra):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "fake_train.py").write_text(FAKE_TRAIN)
        spec = dict(name="t", check_every=0.05, min_backoff=0, jobs=jobs, **extra)
        (self.dir / "job.json").write_text(json.dumps(spec))
        return Keepalive(self.dir / "job.json")

    def test_restarts_crashes_and_resumes_from_the_latest_save(self):
        py = sys.executable
        k = self.make([dict(name="train", cmd=[py, "fake_train.py"], resume_cmd=[py, "fake_train.py", "--resume", "{latest}"],
                            latest="ckpt/save_*.txt", done_when=dict(exit_code=0, file="final.txt"))])
        self.assertEqual(k.run(), 0)
        resumes = (self.dir / "resumes.txt").read_text().split()
        self.assertEqual(resumes[0], "None")                       # first run starts fresh
        self.assertTrue(resumes[1].endswith("save_1.txt"))          # then resumes from the newest save
        self.assertTrue(resumes[2].endswith("save_2.txt"))
        state = json.loads(k.state_file.read_text())
        self.assertEqual(state["status"], "finished")
        self.assertEqual(state["jobs"][0]["restarts"], 2)

    def test_exit_zero_without_the_done_marker_is_a_crash(self):
        py = sys.executable
        # prints "finished" only on its second run; the first clean exit must not count
        script = "import pathlib,sys; p=pathlib.Path('n'); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); print('finished' if n else 'not yet')"
        k = self.make([dict(name="watch", cmd=[py, "-c", script], done_when=dict(log_contains="^finished"))])
        self.assertEqual(k.run(), 0)
        self.assertEqual((self.dir / "n").read_text(), "2")

    def test_a_done_line_from_an_earlier_run_does_not_count(self):
        k = self.make([dict(name="w", cmd=[sys.executable, "-c",
                         "import pathlib,sys; p=pathlib.Path('n'); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); print('finished' if n == 2 else 'crash'); sys.exit(0 if n == 2 else 1)"],
                        done_when=dict(log_contains="^finished"))])
        (self.dir / ".keepalive-t").mkdir()
        (self.dir / ".keepalive-t" / "w.log").write_text("finished\n")  # stale line from an old run
        self.assertEqual(k.run(), 0)
        self.assertEqual((self.dir / "n").read_text(), "3")  # crashed twice, restarted, then really finished

    def test_stop_file_stops_everything(self):
        k = self.make([dict(name="sleeper", cmd=[sys.executable, "-c", "import time; time.sleep(60)"])])
        result = []
        t = threading.Thread(target=lambda: result.append(k.run()))
        t.start()
        time.sleep(0.5)
        k.stop_file.write_text("stop")
        t.join(timeout=10)
        self.assertEqual(result, [1])
        self.assertEqual(json.loads(k.state_file.read_text())["status"], "stopped")

    def test_natural_order_picks_the_real_latest(self):
        from keepalive import natural_key
        names = ["policy_9500000_steps.zip", "policy_10000000_steps.zip", "policy_500000_steps.zip"]
        self.assertEqual(sorted(names, key=natural_key)[-1], "policy_10000000_steps.zip")


if __name__ == "__main__":
    unittest.main()
