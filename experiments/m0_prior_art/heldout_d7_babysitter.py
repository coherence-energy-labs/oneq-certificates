"""Keep the held-out d7 run alive until its stopping rule or its completion, whatever happens to the session that
started it (the preregistration: "If the run is interrupted, it resumes from its chunk index").

    python -B d7_babysitter.py FROZEN_DIR DRIVER_PID START_EPOCH [--budget-core-hours 40] [--workers 3]

Every 60 s: if the watchdog has written d7_stop_record.json, or every chunk of every arm has an index line, it exits.
If the driver is alive it does nothing. If the driver is gone before the deadline, it restarts the committed driver
(experiments/m0_prior_art/heldout_willow_d7_interleaved.py, which resumes from the index) as a detached process and
points a new copy of the committed watchdog at it, with the same START_EPOCH, so the deadline does not move. BLIND:
it counts index lines only. At most 6 restarts; every action is logged to FROZEN_DIR/d7_babysitter.log.
"""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

import psutil

PY = r"C:\Users\Josh\AppData\Local\Python\pythoncore-3.14-64\python.exe"
ARMS = [f"{p}_{b}_{r}" for b in ("X", "Z") for r, p in (("r50", "heldout_d7"), ("r10", "heldout_d7"),
                                                         ("r30", "samedev_d7"))]
DETACH = 0x00000008 | 0x00000200            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def complete(frozen: pathlib.Path) -> bool:
    for arm in ARMS:
        p = frozen / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
        seeds = {json.loads(x)["seed"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()} \
            if p.exists() else set()
        if len(seeds) < 40:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen", type=pathlib.Path)
    ap.add_argument("pid", type=int)
    ap.add_argument("start_epoch", type=float)
    ap.add_argument("--budget-core-hours", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    f = a.frozen
    deadline = a.start_epoch + a.budget_core_hours * 3600.0 / a.workers
    log = f / "d7_babysitter.log"

    def say(msg):
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")

    pid, restarts = a.pid, 0
    say(f"babysitting driver pid {pid} until the stop record, completion, or "
        f"{dt.datetime.fromtimestamp(deadline).isoformat(timespec='seconds')}")
    while True:
        if (f / "d7_stop_record.json").exists():
            say("stop record present: the stopping rule fired; done")
            return 0
        if complete(f):
            say("every chunk of every arm is indexed: the run completed; done")
            return 0
        if time.time() >= deadline:
            say("deadline passed; the watchdog owns the stop; done")
            return 0
        if not psutil.pid_exists(pid):
            if restarts >= 6:
                say("driver gone and 6 restarts used; giving up -- restart by hand")
                return 1
            restarts += 1
            out = open(f / "d7_run_interleaved.log", "a", encoding="utf-8")
            out.write(f"--- restarted by d7_babysitter.py at {dt.datetime.now().isoformat(timespec='seconds')} "
                      f"(driver pid {pid} was gone)\n")
            out.flush()
            p = subprocess.Popen([PY, "-B", "experiments/m0_prior_art/heldout_willow_d7_interleaved.py", "--workers",
                                  str(a.workers)], cwd=f, stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, creationflags=DETACH)
            time.sleep(10)
            w = subprocess.Popen([PY, "-B", str(f / "experiments" / "m0_prior_art" / "heldout_d7_watchdog.py"), str(f),
                                  str(p.pid), str(a.start_epoch)], cwd=f, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=DETACH)
            say(f"driver pid {pid} gone: restart {restarts}, new driver pid {p.pid}, new watchdog pid {w.pid}")
            pid = p.pid
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
