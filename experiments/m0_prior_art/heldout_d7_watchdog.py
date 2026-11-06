r"""Enforce the held-out d=7 stopping rule (docs/PREREG_heldout_d7_2026-11-03.md; DEVIATIONS.md, deviation 1).

    python -B experiments/m0_prior_art/heldout_d7_watchdog.py RUN_ROOT RUNNER_PID START_EPOCH \
        [--budget-core-hours 40] [--workers 3] [--every 600]

The rule: after 40 core-hours the run stops at a chunk boundary and every completed chunk is reported as a partial
sample. The runner keeps its workers busy without pause, so the core-hours consumed are workers x wall time since
START_EPOCH, the ORIGINAL start (2026-11-03 14:35:29 EST): the deadline is fixed in wall time and the amendment's
stop, its lost in-flight chunks and the idle interval before the restart all count against it. At the deadline the
runner's process tree is terminated and RUN_ROOT/d7_stop_record.json records, per arm, the completed chunks and the
chunks being computed (from the driver's dispatch markers: a marker whose worker is still alive and whose chunk has no
index line). Chunks being computed are not in the sample; heldout_d7_partial.py reports them in its worst case.

BLIND by construction: it counts index lines and seeds and never reads a chunk's status. It writes a line every
`--every` seconds to RUN_ROOT/d7_watchdog.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

ARMS = [f"{p}_{b}_{r}" for b in ("X", "Z") for r, p in (("r50", "heldout_d7"), ("r10", "heldout_d7"),
                                                         ("r30", "samedev_d7"))]
RUNNERS = ("heldout_willow_d7.py", "heldout_willow_d7_interleaved.py")


def completed(run_root: pathlib.Path) -> dict[str, list[int]]:
    """Arm -> seeds with an index line. Counts only: a line's status field is never read."""
    out = {}
    for arm in ARMS:
        p = run_root / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        out[arm] = sorted({json.loads(x)["seed"] for x in lines if x.strip()})
    return out


def in_flight(run_root: pathlib.Path, done: dict[str, list[int]], alive) -> dict[str, list[int]]:
    """Arm -> seeds whose dispatch marker names a live worker and which have no index line."""
    out: dict[str, list[int]] = {}
    d = run_root / "evidence" / "epsilon_rerun" / "heldout_d7_dispatch"
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        m = json.loads(p.read_text(encoding="utf-8"))
        if m["seed"] not in done.get(m["arm"], []) and alive(m["pid"], m["started"]):
            out.setdefault(m["arm"], []).append(m["seed"])
    return {a: sorted(v) for a, v in out.items()}


def state(run_root: pathlib.Path, alive) -> dict:
    done = completed(run_root)
    return {"completed_chunks": {a: len(v) for a, v in done.items()}, "completed_seeds": done,
            "dispatched_unfinished": in_flight(run_root, done, alive)}


def main(argv=None) -> int:
    import psutil
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=pathlib.Path)
    ap.add_argument("pid", type=int)
    ap.add_argument("start_epoch", type=float)
    ap.add_argument("--budget-core-hours", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--every", type=float, default=600.0)
    a = ap.parse_args(argv)
    log = a.run_root / "d7_watchdog.log"
    deadline = a.start_epoch + a.budget_core_hours * 3600.0 / a.workers

    def say(msg: str) -> None:
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")

    def alive(pid: int, started: float | None = None) -> bool:
        """The process is running -- and, for a marker, is the process that wrote it (Windows reuses pids)."""
        try:
            p = psutil.Process(pid)
            return p.status() != psutil.STATUS_ZOMBIE and (started is None or p.create_time() <= started + 1.0)
        except psutil.NoSuchProcess:
            return False

    try:
        proc = psutil.Process(a.pid)
        argv_ = proc.cmdline()
        cmd = " ".join(argv_)
    except psutil.NoSuchProcess:
        say(f"runner pid {a.pid} not found at start; nothing to watch")
        return 2
    if not any(pathlib.Path(x).name in RUNNERS for x in argv_):
        say(f"pid {a.pid} is not a held-out d7 runner ({cmd[:200]}); refusing")
        return 2
    say(f"watching pid {a.pid} ({cmd[:160]}); budget {a.budget_core_hours} core-hours on {a.workers} workers from "
        f"{dt.datetime.fromtimestamp(a.start_epoch).isoformat(timespec='seconds')}; deadline "
        f"{dt.datetime.fromtimestamp(deadline).isoformat(timespec='seconds')}")
    while True:
        now = time.time()
        st = state(a.run_root, alive)
        core_h = a.workers * (now - a.start_epoch) / 3600.0
        running = alive(a.pid)
        say(f"core_hours={core_h:.2f} alive={running} completed={st['completed_chunks']} "
            f"in_flight={st['dispatched_unfinished']}")
        if not running:
            say("runner exited on its own; watchdog done")
            return 0
        if now >= deadline:
            kids = [c.pid for c in proc.children(recursive=True)]
            st = state(a.run_root, alive)                  # the last moment before the stop
            r = subprocess.run(["taskkill", "/PID", str(a.pid), "/T", "/F"], capture_output=True, text=True)
            time.sleep(5)
            left = [p for p in [a.pid, *kids] if psutil.pid_exists(p)]
            after = completed(a.run_root)
            rec = {"schema": "oneq-heldout-d7-stop/2",
                   "rule": "docs/PREREG_heldout_d7_2026-11-03.md, Stopping rule, with the order of "
                           "docs/PREREG_heldout_d7_2026-11-03_DEVIATIONS.md (deviation 1)",
                   "reason": f"{a.budget_core_hours} core-hours reached ({a.workers} workers x "
                             f"{(now - a.start_epoch) / 3600.0:.2f} h wall since the original start)",
                   "start": dt.datetime.fromtimestamp(a.start_epoch).isoformat(timespec="seconds"),
                   "stopped": dt.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                   "core_hours": round(core_h, 3), "runner_pid": a.pid, "worker_pids": kids,
                   "taskkill_exit": r.returncode, "taskkill_out": (r.stdout + r.stderr).strip()[:400],
                   "processes_left": left, **st,
                   "index_unchanged_by_the_stop": after == st["completed_seeds"],
                   "note": "chunks in dispatched_unfinished were being computed at the stop and are not in the sample"}
            (a.run_root / "d7_stop_record.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8",
                                                            newline="\n")
            say(f"STOPPED by the 40 core-hour rule: taskkill exit {r.returncode}; processes left {left}; record "
                f"d7_stop_record.json")
            return 0
        time.sleep(min(a.every, max(1.0, deadline - now)))


if __name__ == "__main__":
    sys.exit(main())
