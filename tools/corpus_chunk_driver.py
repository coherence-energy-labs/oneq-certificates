r"""Chunk-granular resumable corpus driver: coverage completes or it says why.

Spawns corpus_chunk_runner.py as INDEPENDENT PROCESSES, P at a time; a chunk
either leaves a good artifact (skipped forever after) or gets retried on the
next round, with a backoff pause when a round's failure rate says the box is
in a commit spike rather than a bad chunk. Terminates when every chunk's
artifact exists or --rounds is exhausted -- and then MERGES the chunk
artifacts into the same schema google_corpus_gate.py writes, so the doc gate
and every downstream reader see one artifact with full accounting.

Chunk-death independence (why partial->complete merging is legitimate) is
inherited from the measured failure classes: spawn-time DLL loads and ~1 MiB
allocations die on the BOX's commit charge, never on a syndrome's content.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "m0_prior_art" / "corpus_chunk_runner.py"
CORPUS = ROOT / "data" / "google_corpus"


def merge(config: str, chunk_dir: pathlib.Path, dest: pathlib.Path,
          chunk: int, escalated: bool, wall: float, missing: list):
    import numpy as np
    docs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(chunk_dir.glob("chunk_*.json"))]
    tot = {k: sum(d[k] for d in docs) for k in
           ("shots", "nontrivial", "certified", "forged_tried",
            "forged_accepted", "battery_bogus", "seconds", "detectors_lit")}
    rungs: dict = {}
    atk: dict = {}
    atk_ok: dict = {}
    gaps: list = []
    detail: list = []
    for d in docs:
        for k, v in d["rungs"].items():
            rungs[k] = rungs.get(k, 0) + v
        for k, v in d["per_attack"].items():
            atk[k] = atk.get(k, 0) + v
        for k, v in d["per_attack_accepted"].items():
            atk_ok[k] = atk_ok.get(k, 0) + v
        gaps.extend(d["gaps"])
        detail.extend(d["accepted_detail"][:5])
    meta = json.loads((CORPUS / config / "metadata.json").read_text())
    g = np.array(gaps) if gaps else np.array([0.0])
    out = {
        "schema": "oneq-google-corpus-gate/1",
        "source": {"doi": "10.5281/zenodo.13273331",
                   "archive": "google_105Q_surface_code_d3_d5_d7.zip",
                   "paper": "Quantum error correction below the surface "
                            "code threshold",
                   "error_model":
                       "correlated_matching_decoder_with_si1000_prior",
                   "note": ("decoding weights are the experimenters' own; "
                            "the correction certified is PyMatching's, "
                            "because the corpus publishes predicted "
                            "OBSERVABLES, not the edge sets any decoder "
                            "chose")},
        "totals": {"shots": tot["shots"], "nontrivial": tot["nontrivial"],
                   "certified": tot["certified"],
                   "certified_rate":
                       tot["certified"] / max(tot["nontrivial"], 1),
                   "forgeries_attempted": tot["forged_tried"],
                   "forgeries_accepted": tot["forged_accepted"],
                   "battery_attacks_that_were_not_actually_false":
                       tot["battery_bogus"]},
        "per_attack": atk, "per_attack_accepted": atk_ok,
        "failed_chunks": [{"config": config, "reason":
                           f"chunk {lo}:{lo + chunk} unrecovered after all "
                           f"retry rounds"} for lo in missing],
        "shots_lost_to_failed_chunks": len(missing) * chunk,
        "accepted_examples": detail[:20],
        "per_config": [{
            "config": config, "distance": meta["distance"],
            "basis": meta["basis"], "rounds": meta["rounds"],
            "shots": tot["shots"], "nontrivial": tot["nontrivial"],
            "certified": tot["certified"],
            "certified_rate": tot["certified"] / max(tot["nontrivial"], 1),
            "mean_detectors_lit":
                tot["detectors_lit"] / max(tot["nontrivial"], 1),
            "rung_histogram": dict(sorted(rungs.items())),
            "forgeries_attempted": tot["forged_tried"],
            "forgeries_accepted": tot["forged_accepted"],
            "ms_per_nontrivial_shot":
                1e3 * tot["seconds"] / max(tot["nontrivial"], 1),
        }],
        "wall_seconds": wall,
        "workers": "chunk-per-process driver (see driver_note)",
        "driver_note": ("chunk-granular resumable driver "
                        "(tools/corpus_chunk_driver.py); escalation="
                        f"{escalated}; chunks retried until coverage "
                        "complete; chunk death is environmental (spawn/"
                        "allocation under commit pressure), independent of "
                        "syndrome content"),
        "uncertified_gap": {"n": int(g.size), "mean": float(g.mean()),
                            "median": float(np.median(g)),
                            "p90": float(np.percentile(g, 90)),
                            "max": float(g.max())},
        "gate": ("forgeries_accepted MUST be 0 on hardware syndromes as on "
                 "simulated ones"),
        "scope": ("the certification RATE is an empirical property of this "
                  "device, this error model and this decoder; SOUNDNESS is "
                  "a property of the checker and does not depend on any of "
                  "them"),
    }
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=12,
                    help="retry rounds over missing chunks")
    ap.add_argument("--escalate", action="store_true")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the chunk directory, so a new producer "
                         "generation sweeps fresh instead of inheriting the "
                         "previous generation's chunk artifacts")
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    chunk_dir = ROOT / "evidence" / "corpus_chunks" / (
        a.config + ("_esc" if a.escalate else "") + a.tag)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    los = list(range(0, a.shots, a.chunk))

    env = dict(os.environ)
    env["ONEQ_ESCALATE"] = "1" if a.escalate else "0"
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    for rnd in range(a.rounds):
        missing = [lo for lo in los
                   if not (chunk_dir / f"chunk_{lo:06d}.json").exists()]
        if not missing:
            break
        print(f"round {rnd}: {len(missing)} chunks to run", flush=True)
        fails = 0
        i = 0
        procs: list = []
        while i < len(missing) or procs:
            while i < len(missing) and len(procs) < a.parallel:
                lo = missing[i]
                try:
                    p = subprocess.Popen(
                        [sys.executable, "-W", "ignore", str(RUNNER),
                         "--config", a.config, "--lo", str(lo),
                         "--hi", str(min(lo + a.chunk, a.shots)),
                         "--out", str(chunk_dir / f"chunk_{lo:06d}.json")],
                        env=env, cwd=str(ROOT),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError as e:
                    # the SPAWN ITSELF can die in a commit spike (WinError
                    # 1455) -- the first launch propagated it and the whole
                    # driver fell over, which is exactly the death this
                    # driver exists to survive. A failed spawn is a failed
                    # chunk attempt: count it, breathe, let the round's
                    # retry sweep pick it up.
                    print(f"  spawn failed for chunk {lo}: {e}", flush=True)
                    fails += 1
                    i += 1
                    time.sleep(30)
                    continue
                procs.append(p)
                i += 1
            done_now = [p for p in procs if p.poll() is not None]
            for p in done_now:
                procs.remove(p)
                if p.returncode != 0:
                    fails += 1
            if not done_now:
                time.sleep(2)
        done = sum(1 for lo in los
                   if (chunk_dir / f"chunk_{lo:06d}.json").exists())
        print(f"round {rnd}: {done}/{len(los)} chunks on disk "
              f"({fails} failures this round)", flush=True)
        if fails > len(missing) // 2:
            # half the round died: the box is in a commit spike, not a bad
            # chunk -- waiting beats hammering
            print("  backoff 120s (commit spike)", flush=True)
            time.sleep(120)
    missing = [lo for lo in los
               if not (chunk_dir / f"chunk_{lo:06d}.json").exists()]
    wall = time.perf_counter() - t0
    out = merge(a.config, chunk_dir, pathlib.Path(a.out), a.chunk,
                a.escalate, wall, missing)
    t = out["totals"]
    print(f"{t['shots']:,} shots  certified {100 * t['certified_rate']:.2f}%"
          f"  forgeries {t['forgeries_attempted']:,}/"
          f"{t['forgeries_accepted']} accepted  "
          f"missing chunks {len(missing)}  wall {wall:.0f}s")
    print(f"evidence -> {a.out}")
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
