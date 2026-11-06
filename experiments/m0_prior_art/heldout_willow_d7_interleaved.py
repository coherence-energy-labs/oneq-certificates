r"""Paper 1's held-out distance-7 run in the amended chunk order (docs/PREREG_heldout_d7_2026-11-03_DEVIATIONS.md).

    python -B experiments/m0_prior_art/heldout_willow_d7_interleaved.py --workers 3       # in the frozen snapshot

DEVIATION 1 changes ONLY the order in which chunks are computed. The committed runner (heldout_willow_d7.py) runs
the six arms one after another, so its 40 core-hour stopping rule would fall inside the first arm and the
preregistered comparison could not be computed. This driver cycles through the six arms instead, each arm taking its
next uncompleted chunks in seed order: X r50, X r10, samedev X r30, Z r50, Z r10, samedev Z r30, then again.

Everything else is the committed run's own code, imported unmodified from the frozen snapshot of 2008c81f into which
this file is copied: `heldout_willow_d7` registers the six arms (names, shots, chunk size, seeds, producer method)
in `epsilon_rerun.CONFIGS`, and every chunk is computed by `epsilon_rerun.run_chunk` and indexed exactly as
`epsilon_rerun.run` indexes it (one JSON line per completed chunk in the arm's chunks.jsonl, failures in
failures.jsonl, a fresh worker per chunk, up to six passes that retry failed chunks). Chunks already completed keep
their place as their arm's first chunks. `heldout_willow_d7.py --merge` and `heldout_d7_partial.py` read the result
unchanged.

One addition, outside the chunk: when a worker starts a chunk it writes a dispatch marker
(`evidence/epsilon_rerun/heldout_d7_dispatch/<arm>__<seed>.json`: arm, seed, worker pid, start time), so the
stopping-rule watchdog (heldout_d7_watchdog.py) knows exactly which chunks were being computed when it stops the run,
without inferring it from the order. Markers carry no verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
# top level on purpose: a spawned worker re-imports this module, and this import registers the six arms there too
import heldout_willow_d7 as hw  # noqa: E402
import epsilon_rerun as er  # noqa: E402

DEVIATIONS = "docs/PREREG_heldout_d7_2026-11-03_DEVIATIONS.md"
DISPATCH = er.INDEX_DIR / "heldout_d7_dispatch"
PASSES = 6                                         # as epsilon_rerun.run


def seeds(name: str) -> list[int]:
    lo, hi = er.CONFIGS[name]["seeds"]
    return list(range(lo, hi + 1))


def order(done: dict[str, set[int]]) -> list[tuple[str, int]]:
    """The amended order: round-robin over the arms in their committed order, each arm's uncompleted seeds ascending."""
    todo = {name: [s for s in seeds(name) if s not in done.get(name, set())] for name in hw.ARMS}
    width = max((len(v) for v in todo.values()), default=0)
    return [(name, todo[name][i]) for i in range(width) for name in hw.ARMS if i < len(todo[name])]


def _job(job):
    """Mark the chunk as started (outside the chunk), then run it with the committed chunk runner, unmodified."""
    name, seed = job
    DISPATCH.mkdir(parents=True, exist_ok=True)
    p = DISPATCH / f"{name}__{seed:06d}.json"
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"arm": name, "seed": seed, "pid": os.getpid(), "started": time.time()}),
                   encoding="utf-8")
    os.replace(tmp, p)
    return name, er.run_chunk(job)


def _pin_instances() -> None:
    """What epsilon_rerun.run does before an arm's first chunk: write the graph, refusing to mix instances."""
    for name in hw.ARMS:
        cfg = er.CONFIGS[name]
        er.INDEX_DIR.joinpath(name).mkdir(parents=True, exist_ok=True)
        _c, _m, edges = er._graph(cfg)
        inst = er.instance_doc(name, edges)
        ip = er.INSTANCE_DIR / f"matching_{name}.json"
        if ip.exists() and json.loads(ip.read_text(encoding="utf-8"))["edges_sha256"] != inst["edges_sha256"]:
            raise SystemExit(f"{ip} disagrees with the rebuilt graph; refusing to mix instances")
        ip.write_text(json.dumps(inst, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
        del _c, _m, edges


def run(workers: int) -> int:
    import multiprocessing as mp
    _pin_instances()
    for attempt in range(PASSES):
        done = {name: set(er._done(name)) for name in hw.ARMS}
        jobs = order(done)
        print(f"pass {attempt + 1}: {sum(map(len, done.values()))} chunks done, {len(jobs)} to run on {workers} "
              f"workers in the amended order ({DEVIATIONS})", flush=True)
        if not jobs:
            break
        idx = {n: open(er._index_path(n), "a", encoding="utf-8", newline="\n") for n in hw.ARMS}
        fl = {n: open(er.INDEX_DIR / n / "failures.jsonl", "a", encoding="utf-8", newline="\n") for n in hw.ARMS}
        try:
            # certify_hard: a fresh process per chunk, as epsilon_rerun.run gives the full ladder
            with mp.Pool(workers, maxtasksperchild=1) as pool:
                for k, (name, r) in enumerate(pool.imap_unordered(_job, jobs), 1):
                    fh = fl[name] if r.get("failed") else idx[name]
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    if k % 50 == 0:
                        print(f"  {k}/{len(jobs)}", flush=True)
        finally:
            for fh in (*idx.values(), *fl.values()):
                fh.close()
    missing = {n: [s for s in seeds(n) if s not in er._done(n)] for n in hw.ARMS}
    for n in hw.ARMS:
        print(f"{n}: {len(seeds(n)) - len(missing[n])}/{len(seeds(n))} chunks complete")
    return 0 if not any(missing.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    return run(a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
