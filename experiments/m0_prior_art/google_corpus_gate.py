r"""Certify Google's real Willow syndromes, and let ZERO forgeries through.

The million-shot run measures the checker at volume on SIMULATED circuit
noise. This one runs the same machinery on detector data from the device
behind "Quantum error correction below the surface code threshold"
(10.5281/zenodo.13273331), decoded on the experimenters' OWN error model --
the weights are theirs, not ours to choose.

WHY BOTH RUNS EXIST. Simulated depolarizing noise is i.i.d. in a way hardware
is not: leakage, crosstalk and drift produce detector patterns no
depolarizing model generates. That difference can move the certification RATE,
which is an empirical quantity and the honest thing to measure here. It cannot
move SOUNDNESS -- a forged certificate is refused whatever produced the
syndrome, because the checker never looks at where the syndrome came from.
Reporting a rate from simulation and implying it holds on hardware is the
substitution this run exists to prevent.

THE CORRECTION BEING CERTIFIED is PyMatching's, not Google's. Their published
`obs_flips_predicted` records which logical observable each decoder thought
flipped, not the correction it chose, so there is no edge set of theirs to
certify. What this measures is therefore: on real hardware syndromes, how
often can a minimum-weight decoder's answer be PROVEN optimal, and does the
checker ever bless a wrong one.
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

CORPUS = ROOT / "data" / "google_corpus"


def load_config(d: pathlib.Path):
    import pymatching
    import stim
    dem = stim.DetectorErrorModel.from_file(str(d / "error_model.dem"))
    m = pymatching.Matching.from_detector_error_model(dem)
    meta = json.loads((d / "metadata.json").read_text())
    nb = (dem.num_detectors + 7) // 8
    # KEEP IT PACKED. Unpacking a whole configuration to bool costs
    # shots x detectors BYTES -- 168 MB for d=7 over 30 rounds -- and every
    # worker holds its own copy, so twenty workers wanted 3.4 GB before doing
    # any work. The run died with a numpy allocation failure on a 5 MB array,
    # which is what running out of memory looks like from the inside: the
    # failure lands on whatever asked next, not on what consumed it.
    raw = np.frombuffer((d / "detection_events.b8").read_bytes(),
                        dtype=np.uint8).reshape(-1, nb)
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in m.edges()]
    return m, edges, _PackedShots(raw, dem.num_detectors), meta


class _PackedShots:
    """Detection events unpacked one slice at a time, not all at once."""

    def __init__(self, raw, n_detectors):
        self.raw = raw
        self.n = n_detectors

    def __len__(self):
        return self.raw.shape[0]

    def window(self, lo, hi):
        return np.unpackbits(self.raw[lo:hi], axis=1,
                             bitorder="little")[:, :self.n].astype(bool)


_C: dict = {}

# Module scope, read in the PARENT. `_C` is the per-worker cache, populated
# only inside worker processes, so the parent that writes the artifact always
# sees it empty -- which made the tier field report BASE TIER for every run,
# escalated ones included. Caught on that field's first execution.
_ESCALATE = bool(int(os.environ.get("ONEQ_ESCALATE", "0")))
# Repair is a SEPARATE column, never blended into "certified": a certified
# receipt proves the DECODER'S answer; a repaired receipt proves a BETTER
# answer the ladder substituted. Blending them would inflate the decoder's
# apparent quality with corrections it did not produce.
_REPAIR = bool(int(os.environ.get("ONEQ_REPAIR", "0")))


def _work(job):
    """One chunk. NEVER raises: a pool that dies on one bad chunk throws away
    every hour of work that came before it.

    Four runs of this gate were lost to a single worker's allocation failure
    propagating through imap_unordered and killing the pool. A chunk that
    cannot complete should cost its own shots and nothing else -- the run
    reports how many were lost, so the coverage stays honest instead of the
    failure being either fatal or invisible.
    """
    try:
        return _work_inner(job)
    except MemoryError as e:                      # noqa: BLE001
        return job[0], {"failed_chunk": True, "reason": f"{type(e).__name__}: {e}"}
    except Exception as e:                        # noqa: BLE001
        return job[0], {"failed_chunk": True, "reason": f"{type(e).__name__}: {e}"}


def _work_inner(job):
    name, lo, hi, attack = job
    d = CORPUS / name
    if _C.get("name") != name:
        m, edges, det, meta = load_config(d)
        _C.clear()
        _C.update(name=name, m=m, edges=edges, det=det, meta=meta,
                  cd=CertifyingDecoder(edges),
                  escalate=bool(int(os.environ.get("ONEQ_ESCALATE", "0"))))
    cd, m, det, edges = _C["cd"], _C["m"], _C["det"], _C["edges"]

    r = {"shots": hi - lo, "nontrivial": 0, "certified": 0, "repaired": 0,
         "rungs": Counter(),
         "forged_tried": 0, "forged_accepted": 0, "battery_bogus": 0,
         "accepted_detail": [],
         "gaps": [], "seconds": 0.0, "per_attack": Counter(),
         "per_attack_accepted": Counter(), "detectors_lit": 0}
    window = det.window(lo, hi)
    t0 = time.perf_counter()
    for s in range(lo, hi):
        row = window[s - lo]
        fired = [int(i) for i in np.flatnonzero(row)]
        if not fired:
            continue
        r["nontrivial"] += 1
        r["detectors_lit"] += len(fired)
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                     for a, b in m.decode_to_edges_array(row))
        if _REPAIR and _C.get("escalate"):
            res = cd.certify_or_repair(fired, corr)
        elif _C.get("escalate"):
            res = cd.certify_hard(fired, corr)
        else:
            res = cd.certify(fired, corr)
        if res.accepted:
            if res.stats.get("repaired"):
                r["repaired"] += 1
            else:
                r["certified"] += 1
            r["rungs"][res.rung] += 1
        else:
            # NOT rounded. round(-2.2e-16, 6) is -0.0, and -0.0 < 0 is False,
            # so rounding here would make the sign split downstream VACUOUS --
            # every ulp-scale gap would count as zero and the negative ones
            # (dual exceeds primal, refusal correct) would vanish into the
            # same bucket as the positive ones. The rounding was the reason
            # the artifact could not tell those two cases apart.
            r["gaps"].append(float(res.gap))
        if attack and res.best_dual:
            battery = forgeries(cd.W, fired, corr, res.best_dual, res.primal)
            # A battery nobody audits is a gate that cannot fail: confirm every
            # attack's claim is FALSE before scoring acceptances against it.
            r["battery_bogus"] += len(
                assert_all_false(cd.W, fired, res.primal, battery))
            for nm, forged, _why in battery:
                r["forged_tried"] += 1
                r["per_attack"][nm] += 1
                if check(edges=edges, syndrome=fired, cert=forged).accepted:
                    r["forged_accepted"] += 1
                    r["per_attack_accepted"][nm] += 1
                    r["accepted_detail"].append({"config": name, "shot": s,
                                                 "attack": nm})
    r["seconds"] = time.perf_counter() - t0
    for k in ("rungs", "per_attack", "per_attack_accepted"):
        r[k] = dict(r[k])
    r["gaps"] = r["gaps"][:300]
    return name, r


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
    import multiprocessing as mp
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, nargs="*", default=None)
    ap.add_argument("--max-shots", type=int, default=None,
                    help="cap per configuration (default: all of them)")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--max-tasks-per-child", type=int, default=8,
                    help="restart each worker after this many chunks, to "
                         "bound solver memory retention")
    ap.add_argument("--no-attack", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    names = a.configs or sorted(p.name for p in CORPUS.iterdir()
                                if p.is_dir() and (p / "error_model.dem").exists())
    if not names:
        print(f"no configurations in {CORPUS}; run tools/fetch_google_corpus.py",
              file=sys.stderr)
        return 2

    jobs = []
    planned: dict[str, int] = {}
    for n in names:
        meta = json.loads((CORPUS / n / "metadata.json").read_text())
        total = int(meta["shots"])
        if a.max_shots:
            total = min(total, a.max_shots)
        planned[n] = total
        for lo in range(0, total, a.chunk):
            jobs.append((n, lo, min(lo + a.chunk, total), not a.no_attack))
    print(f"{len(names)} configurations, {sum(planned.values()):,} hardware shots, "
          f"{len(jobs)} chunks on {a.workers} workers", flush=True)

    per: dict[str, dict] = {}
    failed: list[dict] = []
    gaps: list[float] = []
    detail: list[dict] = []
    tot = Counter()
    atk, atk_ok = Counter(), Counter()
    t0 = time.perf_counter()
    done = 0
    # RECYCLE WORKERS. The failing allocation was 4.84 MiB with 15 GB of
    # physical memory free -- so this was never system exhaustion, it was
    # per-worker growth across thousands of shots. The refinement rung runs up
    # to twenty scipy/HiGHS solves per shot, and whatever the solver retains
    # compounds over a chunk sequence. maxtasksperchild restarts a worker after
    # a bounded number of chunks, which returns everything to the OS whether or
    # not the retention is ours to fix.
    with mp.Pool(a.workers, maxtasksperchild=a.max_tasks_per_child) as pool:
        for name, r in pool.imap_unordered(_work, jobs):
            if r.get("failed_chunk"):
                failed.append({"config": name, "reason": r["reason"]})
                done += 1
                continue
            p = per.setdefault(name, {"shots": 0, "nontrivial": 0, "certified": 0,
                                      "repaired": 0,
                                      "forged_tried": 0, "forged_accepted": 0,
                                      "battery_bogus": 0,
                                      "seconds": 0.0, "rungs": Counter(),
                                      "detectors_lit": 0})
            for k in ("shots", "nontrivial", "certified", "repaired", "forged_tried",
                      "forged_accepted", "battery_bogus", "seconds",
                      "detectors_lit"):
                p[k] += r[k]
            p["rungs"].update(r["rungs"])
            for k in ("shots", "nontrivial", "certified", "repaired", "forged_tried",
                      "forged_accepted", "battery_bogus"):
                tot[k] += r[k]
            atk.update(r["per_attack"])
            atk_ok.update(r["per_attack_accepted"])
            if len(gaps) < 20000:
                gaps.extend(r["gaps"])
            detail.extend(r["accepted_detail"][:5])
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done:5d}/{len(jobs)}  {tot['shots']:>9,} shots  "
                      f"certified {100*tot['certified']/max(tot['nontrivial'],1):5.1f}%  "
                      f"forgeries accepted {tot['forged_accepted']}", flush=True)
    wall = time.perf_counter() - t0

    rows = []
    for n in sorted(per):
        p = per[n]
        meta = json.loads((CORPUS / n / "metadata.json").read_text())
        rows.append({
            "config": n, "distance": meta["distance"], "basis": meta["basis"],
            "rounds": meta["rounds"], "shots": p["shots"],
            "nontrivial": p["nontrivial"], "certified": p["certified"],
            "certified_rate": p["certified"] / max(p["nontrivial"], 1),
            "mean_detectors_lit": p["detectors_lit"] / max(p["nontrivial"], 1),
            "rung_histogram": dict(sorted(p["rungs"].items())),
            "forgeries_attempted": p["forged_tried"],
            "forgeries_accepted": p["forged_accepted"],
            "ms_per_nontrivial_shot": 1e3 * p["seconds"] / max(p["nontrivial"], 1),
        })
        print(f"  {n:26s} d={meta['distance']} r={meta['rounds']:<4d} "
              f"certified {100*rows[-1]['certified_rate']:5.1f}%  "
              f"{rows[-1]['ms_per_nontrivial_shot']:7.1f} ms/shot  "
              f"forgeries {p['forged_accepted']}/{p['forged_tried']}")

    g = np.array(gaps) if gaps else np.array([0.0])
    out = {
        "schema": "oneq-google-corpus-gate/1",
        "source": {"doi": "10.5281/zenodo.13273331",
                   "archive": "google_105Q_surface_code_d3_d5_d7.zip",
                   "paper": "Quantum error correction below the surface code threshold",
                   "error_model": "correlated_matching_decoder_with_si1000_prior",
                   "note": ("decoding weights are the experimenters' own; the "
                            "correction certified is PyMatching's, because the "
                            "corpus publishes predicted OBSERVABLES, not the "
                            "edge sets any decoder chose")},
        "totals": {"shots": tot["shots"], "nontrivial": tot["nontrivial"],
                   "certified": tot["certified"],
                   "repaired": tot["repaired"],
                   "certified_or_repaired_rate": (tot["certified"] + tot["repaired"]) / max(tot["nontrivial"], 1),
                   "certified_rate": tot["certified"] / max(tot["nontrivial"], 1),
                   "forgeries_attempted": tot["forged_tried"],
                   "forgeries_accepted": tot["forged_accepted"],
                   "battery_attacks_that_were_not_actually_false":
                       tot["battery_bogus"]},
        "per_attack": dict(atk), "per_attack_accepted": dict(atk_ok),
        "failed_chunks": failed,
        "shots_lost_to_failed_chunks": len(failed) * a.chunk,
        "accepted_examples": detail[:20],
        "per_config": rows, "wall_seconds": wall, "workers": a.workers,
        # SPLIT BY SIGN. This summary previously reported only n/mean/median/
        # p90/max, and every value rounds to 0.0 at ulp scale -- so a gap of
        # -2.2e-16 and one of +2.2e-16 were INDISTINGUISHABLE in the artifact.
        # That is not cosmetic. A POSITIVE gap may be float rounding in the
        # dual, which the checker repairs and certifies. A NEGATIVE gap means
        # the dual EXCEEDS the primal, which for an edge-local dual is the
        # metric-closure boundary in matching_cert's docstring: y_u + y_v <=
        # w_e is not the T-join dual (that constrains PAIRS by dist(u,v)), so
        # an edge-feasible dual can exceed the true optimum. Those refusals
        # are CORRECT and must never be "closed" -- lowering such a dual to
        # meet the primal would preserve feasibility, satisfy attainment, and
        # manufacture a FALSE PROOF OF OPTIMALITY.
        #
        # I read max==0.0 off the old summary and concluded every refusal was
        # a false one. The artifact could not support that and I acted on it
        # anyway; these fields exist so the distinction is in the evidence
        # rather than in whoever is reading it.
        "uncertified_gap": {"n": int(g.size), "mean": float(g.mean()),
                            "median": float(np.median(g)),
                            "p90": float(np.percentile(g, 90)),
                            "max": float(g.max()), "min": float(g.min()),
                            "n_negative": int((g < 0).sum()),
                            "n_zero": int((g == 0).sum()),
                            "n_positive": int((g > 0).sum()),
                            "max_abs": float(np.abs(g).max()),
                            "negative_means": (
                                "dual EXCEEDS primal -- edge-local dual is not "
                                "a valid T-join bound; refusal is CORRECT")},
        # WHICH RUNGS ACTUALLY RAN. This artifact previously recorded a
        # certification rate with NO record of whether escalation was on, and
        # ONEQ_ESCALATE defaults to 0 -- so every rate here was the BASE tier
        # alone while `certify_hard` (base -> escalated -> exact) is the real
        # ladder. Measured 2026-08-05, the difference is not cosmetic:
        #
        #     d3_at_q4_5_X_r30    99.11%  base only   ->  99.90%  full ladder
        #     d5_at_q8_7_X_r13    83.89%  base only   ->  92.25%  full ladder
        #     d5_at_q8_7_X_r30    47.32%  base only   ->  67.93%  full ladder
        #
        # A rate understating the system by 20 points, produced by a default,
        # while every gate stayed green. That is structurally the SAME defect
        # as the 1e-6 tolerance the external audit killed: a switch that
        # silently weakens a claim without failing anything. So the tier is
        # now recorded beside the number, and a rate quoted without this
        # field should be treated as unattributable rather than as a result.
        # Read from the ENVIRONMENT, not from `_C`. `_C` is the per-worker
        # cache, populated inside the worker processes; in the parent that
        # writes this artifact it is empty, so `_C.get("escalate")` is always
        # falsy and the field would have reported BASE TIER for every run --
        # including escalated ones. Caught on this field's first execution,
        # which is the argument for the field.
        "certification_tier": ("full ladder (base -> escalated -> exact) via "
                               "certify_hard" if _ESCALATE
                               else "BASE TIER ONLY -- escalation and exact "
                                    "tiers did NOT run; this rate is a floor, "
                                    "not the system's capability"),
        "oneq_escalate": int(_ESCALATE),
        "oneq_repair": int(_REPAIR),
        "gate": "forgeries_accepted MUST be 0 on hardware syndromes as on simulated ones",
        "scope": ("the certification RATE is an empirical property of this "
                  "device, this error model and this decoder; SOUNDNESS is a "
                  "property of the checker and does not depend on any of them"),
    }
    dest = pathlib.Path(a.out) if a.out else ROOT / "evidence" / "google_corpus_gate.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{tot['shots']:,} hardware shots  certified "
          f"{100*out['totals']['certified_rate']:.2f}%")
    print(f"forgeries: {tot['forged_tried']:,} attempted, "
          f"{tot['forged_accepted']} accepted     wall {wall:.0f}s")
    if failed:
        print(f"WARNING: {len(failed)} chunk(s) failed and their shots are NOT "
              f"in the totals above -- coverage is {a.chunk*len(failed)} shots "
              f"short of what was requested", file=sys.stderr)
    print(f"evidence -> {dest}")
    code, message = verdict(tot)
    if code:
        print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
