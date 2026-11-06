#!/usr/bin/env python3
"""Is NOW a good calibration window for a frozen design? Read-only (oneq.hw.window; the 25-item list, item 25).

The design's qubits and edges come from a frozen record's circuits (every qubit measured or gated, every CZ edge);
the history is every DeviceSnapshot this repository has recorded for the same backend (evidence/hw/snapshots/*.json,
every frozen record's embedded snapshot, every after_<job>.snapshot.json).

  python tools/calibration_window.py RECORD.json [--live] [--best 0.2] [--poll-min M --hours H]

Without --live it ranks the record's OWN frozen snapshot against the history (spends nothing, no network). With
--live it reads the backend's current calibration (read-only) and, with --poll-min, re-reads every M minutes for up
to H hours, printing one verdict per read and exiting 0 the first time the window ranks in the best fraction."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import window as W  # noqa: E402


def design_of(rec: dict, blob: bytes) -> tuple:
    circs = F.load(rec, blob, check_sources=False)
    qubits, edges = set(), set()
    for qc in circs.values():
        for inst in qc.data:
            qs = [qc.find_bit(b).index for b in inst.qubits]
            if inst.operation.name in ("barrier", "delay"):
                continue
            qubits.update(qs)
            if inst.operation.name == "cz":
                edges.add(tuple(sorted(qs)))
    return sorted(qubits), sorted(edges)


def history(backend: str) -> list:
    snaps = []
    for p in sorted((ROOT / "evidence").rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        cand = d.get("snapshot") if isinstance(d, dict) and isinstance(d.get("snapshot"), dict) else d
        if isinstance(cand, dict) and cand.get("name") == backend and "cz" in cand and "t1" in cand:
            try:
                snaps.append((p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),
                              S.DeviceSnapshot.from_json(cand)))
            except (S.SnapshotError, KeyError, TypeError, ValueError):
                continue
    seen, out = set(), []
    for path, s in snaps:                                  # one window per calibration, not per file that copied it
        if s.calibrated_utc not in seen:
            seen.add(s.calibrated_utc)
            out.append((path, s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--best", type=float, default=0.2)
    ap.add_argument("--poll-min", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=0.0)
    a = ap.parse_args()
    rp = pathlib.Path(a.record)
    rec = json.loads(rp.read_text(encoding="utf-8"))
    qubits, edges = design_of(rec, rp.with_suffix(".qpy").read_bytes())
    hist = history(rec["backend"])
    costs = [W.design_cost(s, qubits, edges) for _, s in hist]
    print(f"design: {len(qubits)} qubits, {len(edges)} edges; {len(hist)} recorded calibrations of {rec['backend']}")
    if not a.live:
        now = S.DeviceSnapshot.from_json(rec["snapshot"])
        print(json.dumps(W.verdict(W.design_cost(now, qubits, edges), costs, a.best), indent=1))
        return 0
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    deadline = time.time() + a.hours * 3600
    while True:
        now = S.from_backend(svc.backend(rec["backend"]))
        v = W.verdict(W.design_cost(now, qubits, edges), costs, a.best)
        print(time.strftime("%H:%M:%S"), now.calibrated_utc, json.dumps(v), flush=True)
        if v["submit_now"]:
            return 0
        if a.poll_min <= 0 or time.time() >= deadline:
            return 3
        time.sleep(a.poll_min * 60)


if __name__ == "__main__":
    raise SystemExit(main())
