r"""Milestone 0.2c: certify at scale, and let ZERO forgeries through.

The gate has two halves and only the second one can end the program.

  SCALE. Produce and check certificates for >= 10^6 shots of multi-round
  space-time decoding. This establishes that certification is an engineering
  cost, not a research demo -- if it only works on 120 shots at d=7 it is a
  figure, not a capability.

  ZERO FALSE POSITIVES. On every shot the checker is also attacked with
  forgeries: certificates that are wrong in a specific way and look right in
  the way a naive checker measures. A single acceptance would mean the checker
  can bless a wrong correction, which is the whole claim gone. This is the
  half that matters, and it is why the forgeries run on EVERY shot rather than
  a sample -- a rare false positive is exactly the kind that a sample misses
  and a deployment finds.

The attacks live in `oneq.forgeries` and are deliberately NOT restated here.
Two copies of an attack list is how one silently weakens while the other keeps
reporting "0 accepted" -- and a summary that drifts from the battery is the
same failure in prose. Read that module for what each attack does and why its
claim is false.

The battery is AUDITED on every shot before it is scored. `assert_all_false`
re-derives, independently of the checker, that each attack's claim really is
false; anything that does not falsify is counted and VOIDS the run. That guard
exists because this project has twice produced attacks that were accepted
because their claims happened to be TRUE -- equal-weight alternative optima
once, a degenerate construction that collapsed into a valid certificate the
second time. Both looked exactly like a program-ending false positive until
somebody checked the attack instead of the checker.

WHAT THIS RUN CANNOT SHOW. That the certification RATE generalises to hardware
noise; simulated circuit noise is i.i.d. in a way real devices are not. That
claim needs the Google corpus and is made separately. What it does show is
that the checker's soundness does not degrade with volume, which is a property
of the checker, not of the noise model.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder            # noqa: E402
from oneq.forgeries import assert_all_false, forgeries           # noqa: E402
from oneq.matching_cert import BOUNDARY, check                   # noqa: E402

_STATE: dict = {}


def _circuit(d: int, rounds: int, p: float):
    import stim
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def _init(d: int, rounds: int, p: float):
    import pymatching
    circ = _circuit(d, rounds, p)
    m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in m.edges()]
    _STATE.update(circ=circ, m=m, edges=edges, cd=CertifyingDecoder(edges),
                  d=d, rounds=rounds, p=p)


def _chunk(job):
    """One chunk. NEVER raises -- a pool that dies on one bad chunk discards
    every hour of work before it, and the shots it loses must be COUNTED
    rather than silently missing from the denominator."""
    try:
        return _chunk_inner(job)
    except Exception as e:                        # noqa: BLE001
        return {"failed_chunk": True, "shots": job[1],
                "reason": f"{type(e).__name__}: {e}"}


def _chunk_inner(job):
    """One worker: sample `n` shots at `seed`, certify and attack each."""
    seed, n, d, rounds, p, attack = job
    if _STATE.get("d") != d or _STATE.get("rounds") != rounds:
        _init(d, rounds, p)
    st = _STATE
    cd, m, edges = st["cd"], st["m"], st["edges"]
    det, _obs = st["circ"].compile_detector_sampler(seed=seed).sample(
        n, separate_observables=True)

    res = {"shots": n, "nontrivial": 0, "certified": 0, "rungs": Counter(),
           "forged_tried": 0, "forged_accepted": 0, "battery_bogus": 0,
           "forged_accepted_detail": [], "gaps": [], "seconds": 0.0,
           "per_attack": Counter(), "per_attack_accepted": Counter()}
    t0 = time.perf_counter()
    for s in range(n):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        res["nontrivial"] += 1
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                     for a, b in m.decode_to_edges_array(det[s]))
        r = cd.certify(fired, corr)
        if r.accepted:
            res["certified"] += 1
            res["rungs"][r.rung] += 1
        else:
            res["gaps"].append(round(r.gap, 6))
        if attack and r.best_dual:
            # EVERY shot is attacked, not just the certified ones. Building the
            # battery from `cert` conditioned the attacks on success, which
            # skipped precisely the shots where the bound was loosest.
            battery = forgeries(cd.W, fired, corr, r.best_dual, r.primal)
            # A BATTERY NOBODY AUDITS IS A GATE THAT CANNOT FAIL. Two attacks
            # in this project's history were accepted because their claims
            # happened to be TRUE, not because the checker was wrong. Confirm
            # every claim is false BEFORE scoring acceptances against it.
            res["battery_bogus"] += len(
                assert_all_false(cd.W, fired, r.primal, battery))
            for name, forged, _why in battery:
                res["forged_tried"] += 1
                res["per_attack"][name] += 1
                v = check(edges=edges, syndrome=fired, cert=forged)
                if v.accepted:
                    res["forged_accepted"] += 1
                    res["per_attack_accepted"][name] += 1
                    res["forged_accepted_detail"].append(
                        {"seed": seed, "shot": s, "attack": name})
    res["seconds"] = time.perf_counter() - t0
    res["rungs"] = dict(res["rungs"])
    res["per_attack"] = dict(res["per_attack"])
    res["per_attack_accepted"] = dict(res["per_attack_accepted"])
    # keep the gap sample bounded; the full distribution is summarised, not stored
    res["gaps"] = res["gaps"][:400]
    return res


def verdict(agg) -> tuple:
    r"""(exit code, message) from the aggregated counters. PURE.

    LIFTED OUT OF `main()` ON 2026-09-09, because the ratchet that says
    "a module publishing into evidence/ must have a test naming it" also
    says why it usually does not: the scoring is pure arithmetic over
    recorded numbers, and if it cannot be called without the cluster,
    that is the defect rather than the reason.

    THE ORDER MATTERS AND IS NOT ARBITRARY. `battery_bogus` is checked
    FIRST: it counts attacks that did not actually falsify anything, and
    a "0 forgeries accepted" headline computed against such a battery is
    measuring nothing. A run that is VOID must not be reported as a run
    that PASSED, and checking acceptance first would do exactly that
    whenever both are non-zero.
    """
    if agg.get("battery_bogus"):
        return 2, (f"GATE VOID: {agg['battery_bogus']} attacks did not actually "
                   f"falsify -- a '0 accepted' headline from this run would be "
                   f"measuring nothing")
    if agg.get("forged_accepted"):
        return 1, "GATE FAILED: a forgery was accepted"
    return 0, "no forgery accepted, and every attack in the battery was false"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=1_000_000)
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None,
                    help="space-time rounds (default 5x distance -- multi-round "
                         "is the point; a single round is not a decoding problem)")
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--no-attack", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed-base", type=int, default=1,
                    help="first chunk seed; a SUPPLEMENTAL run completing "
                         "shots lost to environmental chunk death must use "
                         "seeds disjoint from the main run's 1..n_chunks, "
                         "or it silently replays the same shots")
    a = ap.parse_args()
    rounds = a.rounds if a.rounds is not None else 5 * a.distance

    jobs = []
    remaining, seed = a.shots, a.seed_base
    while remaining > 0:
        k = min(a.chunk, remaining)
        jobs.append((seed, k, a.distance, rounds, a.p, not a.no_attack))
        remaining -= k
        seed += 1
    print(f"d={a.distance} rounds={rounds} p={a.p}  {a.shots} shots in "
          f"{len(jobs)} chunks on {a.workers} workers")

    import multiprocessing as mp
    agg = {"shots": 0, "nontrivial": 0, "certified": 0, "forged_tried": 0,
           "forged_accepted": 0, "battery_bogus": 0,
           "rungs": Counter(), "per_attack": Counter(),
           "per_attack_accepted": Counter(), "core_seconds": 0.0}
    gap_sample: list[float] = []
    detail: list[dict] = []
    failed: list[str] = []
    lost = 0
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_chunk, jobs):
            if r.get("failed_chunk"):
                failed.append(r["reason"])
                lost += r["shots"]
                done += 1
                continue
            for k in ("shots", "nontrivial", "certified", "forged_tried",
                      "forged_accepted", "battery_bogus"):
                agg[k] += r[k]
            agg["core_seconds"] += r["seconds"]
            agg["rungs"].update(r["rungs"])
            agg["per_attack"].update(r["per_attack"])
            agg["per_attack_accepted"].update(r["per_attack_accepted"])
            if len(gap_sample) < 20000:
                gap_sample.extend(r["gaps"])
            detail.extend(r["forged_accepted_detail"][:5])
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                rate = agg["shots"] / max(el, 1e-9)
                print(f"  {done:5d}/{len(jobs)} chunks  {agg['shots']:>9,} shots  "
                      f"{rate:8.0f} shots/s  certified "
                      f"{100*agg['certified']/max(agg['nontrivial'],1):5.1f}%  "
                      f"forgeries accepted {agg['forged_accepted']}", flush=True)
    wall = time.perf_counter() - t0

    g = np.array(gap_sample) if gap_sample else np.array([0.0])
    out = {
        "schema": "oneq-million-shot-gate/1",
        "distance": a.distance, "rounds": rounds, "p": a.p,
        "shots": agg["shots"], "nontrivial": agg["nontrivial"],
        "certified": agg["certified"],
        "certified_rate": agg["certified"] / max(agg["nontrivial"], 1),
        "rung_histogram": dict(sorted(agg["rungs"].items())),
        "forgeries_attempted": agg["forged_tried"],
        "forgeries_accepted": agg["forged_accepted"],
        "failed_chunks": len(failed),
        "shots_lost_to_failed_chunks": lost,
        "failed_chunk_reasons": sorted(set(failed))[:5],
        "battery_attacks_that_were_not_actually_false": agg["battery_bogus"],
        "per_attack": dict(agg["per_attack"]),
        "per_attack_accepted": dict(agg["per_attack_accepted"]),
        "accepted_examples": detail[:20],
        "wall_seconds": wall, "core_seconds": agg["core_seconds"],
        "workers": a.workers,
        "ms_per_nontrivial_shot": 1e3 * agg["core_seconds"] / max(agg["nontrivial"], 1),
        "uncertified_gap": {"n": int(g.size), "mean": float(g.mean()),
                            "median": float(np.median(g)),
                            "p90": float(np.percentile(g, 90)),
                            "max": float(g.max())},
        "gate": ("forgeries_accepted MUST be 0. A single acceptance means the "
                 "checker can bless a wrong correction, which ends the claim."),
        "scope": ("simulated circuit noise; this measures the CHECKER's "
                  "soundness at volume, not that the certification RATE "
                  "carries over to hardware noise"),
    }
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / f"million_shot_gate_d{a.distance}.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{agg['shots']:,} shots  {agg['nontrivial']:,} nontrivial  "
          f"certified {100*out['certified_rate']:.2f}%")
    print(f"forgeries: {agg['forged_tried']:,} attempted, "
          f"{agg['forged_accepted']} accepted")
    print(f"wall {wall:.0f}s on {a.workers} workers  "
          f"({out['ms_per_nontrivial_shot']:.2f} ms/shot core time)")
    print(f"evidence -> {dest}")
    code, message = verdict(agg)
    if code:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
