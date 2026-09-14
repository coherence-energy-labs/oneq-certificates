r"""Milestone 0.4: two numbers nobody publishes.

The M0.0 sweep found this lane clean. Mutation testing has been applied to
quantum circuits and to QML models, never to QEC decoder software. The nearest
work (ScaLER, arXiv:2602.04921) injects faults into CIRCUITS and treats the
decoder as a black box; nothing tests whether the decoder itself is right.

So the question is the one an assurance argument actually needs:

    when a decoder is subtly WRONG, what notices?

Two detectors, same seeded faults:

  1. SYNDROME-CONSISTENCY -- the check every decoder already passes today, and
     what a QEC test suite effectively encodes: does the correction explain the
     observed syndrome? STRUCTURALLY BLIND to the valid-but-suboptimal class,
     because a suboptimal correction is still consistent.
  2. THE CERTIFICATE -- does a feasible cut packing attain the correction's
     weight?

WE MUTATE THE DECODER'S OUTPUT, not PyMatching's C++. That models a
wrong-answer defect exactly where it matters and is honest about what it cannot
model: a crash, a hang, a memory error, or a fault that changes which shots are
hard. A real seeded-fault campaign against the C++ would cover those too; this
covers the class the certificate is about.

THE STEALTH MUTANTS ARE THE EXPERIMENT. M1-M4 and M6 all change the
odd-degree set, so syndrome checking catches them and both detectors score
100% -- an uninformative comparison that proves nothing. M7-M9 keep the
syndrome EXACTLY satisfied and only raise the weight. Those are the mutants
the two numbers are actually about.

M5 IS A NEGATIVE CONTROL. An equal-weight alternative optimum is not a defect,
and a detector that flags it is producing false alarms. A gate that catches
everything is as useless as one that catches nothing -- and this project has
already once scored 16 alternative optima as failures and nearly condemned a
sound checker for it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder                # noqa: E402
from oneq.matching_cert import BOUNDARY, MatchingCertificate, check  # noqa: E402


def odd_set(edges_used):
    deg: dict[int, int] = {}
    for u, v in edges_used:
        for x in (int(u), int(v)):
            if x != BOUNDARY:
                deg[x] = deg.get(x, 0) + 1
    return {n for n, c in deg.items() if c % 2 == 1}


def mutants(W, matched, fired, prev):
    """Plausible decoder defects, as mutated OUTPUT."""
    mset = [tuple(e) for e in matched]
    out = []

    alt = next((k for k in W if k not in mset), None)
    if mset and alt:
        out.append(("M1_off_by_one_edge", mset[:-1] + [alt]))

    for k in mset:                       # a paired detector sent to the boundary
        if BOUNDARY not in k:
            b = (k[0], BOUNDARY)
            if b in W:
                out.append(("M2_greedy_boundary",
                            [x for x in mset if x != k] + [b]))
                break

    if mset:
        cheapest = min(mset, key=lambda k: W[k])
        out.append(("M3_dropped_edge", [x for x in mset if x != cheapest]))

    if prev:
        out.append(("M4_stale_cache", list(prev)))

    allb = [(f, BOUNDARY) for f in fired if (f, BOUNDARY) in W]
    if len(allb) == len(fired) and fired:
        if abs(sum(W[k] for k in allb) - sum(W[k] for k in mset)) < 1e-9:
            out.append(("M5_equal_weight_alternative", allb))

    if len(mset) > 2:
        out.append(("M6_truncated_path", mset[:-2] + mset[-1:]))

    # ---- THE MUTANTS THAT MATTER: syndrome-consistent, strictly heavier ----
    if alt and BOUNDARY not in alt:      # a doubled edge is a closed cycle
        out.append(("M7_added_cycle_STEALTH", mset + [alt, alt]))

    bpairs = [k for k in mset if BOUNDARY in k]
    if len(bpairs) >= 2:
        a, b = bpairs[0][0], bpairs[1][0]
        direct = (a, b) if a <= b else (b, a)
        if direct in W:
            rest = [k for k in mset if k not in (bpairs[0], bpairs[1])]
            cand = rest + [direct]
            if sum(W[k] for k in cand) > sum(W[k] for k in mset) + 1e-9:
                out.append(("M8_boundary_pair_swap_STEALTH", cand))

    for k in mset:                       # a detour through a common neighbour
        if BOUNDARY in k:
            continue
        u, v = k
        nbrs = [b if a == u else a for (a, b) in W
                if BOUNDARY not in (a, b) and u in (a, b)]
        for x in nbrs:
            if x == v:
                continue
            e1 = (u, x) if u <= x else (x, u)
            e2 = (x, v) if x <= v else (v, x)
            if e1 in W and e2 in W:
                detour = [q for q in mset if q != k] + [e1, e2]
                if sum(W[q] for q in detour) > sum(W[q] for q in mset) + 1e-9:
                    out.append(("M9_detour_STEALTH", detour))
                    break
        else:
            continue
        break
    return out


def main() -> int:
    import pymatching
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--corpus", type=str, default=None,
                    help="write the seeded-fault corpus here (released so the "
                         "two numbers can be reproduced against other detectors)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    d = a.distance
    rounds = a.rounds or d

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=a.p, before_measure_flip_probability=a.p,
        after_reset_flip_probability=a.p, before_round_data_depolarization=a.p)
    m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in m.edges()]
    cd = CertifyingDecoder(edges)
    det, _ = circ.compile_detector_sampler(seed=7).sample(
        a.shots, separate_observables=True)

    stats: dict[str, dict] = {}
    corpus: list[dict] = []
    prev = None
    skipped_no_bound = 0
    for s in range(a.shots):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        matched = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                        for u, v in m.decode_to_edges_array(det[s]))
        r = cd.certify(fired, matched)
        if not r.best_dual:
            skipped_no_bound += 1
            prev = matched
            continue
        base_w = cd.weight(matched)

        for name, mut in mutants(cd.W, matched, fired, prev):
            st = stats.setdefault(name, {"n": 0, "syndrome_caught": 0,
                                         "certificate_caught": 0,
                                         "truly_defective": 0})
            st["n"] += 1
            mw = sum(cd.W[k] for k in mut)
            same_syndrome = odd_set(mut) == set(fired)
            # DEFECTIVE means the claim "this is a minimum-weight correction
            # for this syndrome" is FALSE -- either it does not explain the
            # syndrome, or a strictly cheaper one is known to exist.
            defective = (not same_syndrome) or (mw > base_w + 1e-9)
            st["truly_defective"] += int(defective)
            if not same_syndrome:
                st["syndrome_caught"] += 1
            v = check(edges=edges, syndrome=fired,
                      cert=MatchingCertificate(matched_edges=tuple(mut),
                                               node_potentials={},
                                               blossom_duals=r.best_dual))
            if not v.accepted:
                st["certificate_caught"] += 1
            if a.corpus and len(corpus) < 4000:
                corpus.append({"shot": s, "mutant": name,
                               "syndrome": fired,
                               "correction": [list(k) for k in mut],
                               "weight": mw, "reference_weight": base_w,
                               "syndrome_consistent": bool(same_syndrome),
                               "defective": bool(defective)})
        prev = matched

    print(f"d={d} rounds={rounds}  {a.shots} shots"
          + (f"  ({skipped_no_bound} skipped: no bound produced)"
             if skipped_no_bound else ""))
    print(f"{'mutant':34s} {'n':>5} {'defective':>10} {'syndrome':>9} {'certificate':>12}")
    tot = Counter()
    for name, st in sorted(stats.items()):
        print(f"{name:34s} {st['n']:5d} {st['truly_defective']:10d} "
              f"{st['syndrome_caught']:9d} {st['certificate_caught']:12d}")
        if not name.startswith("M5"):
            tot["def"] += st["truly_defective"]
            tot["syn"] += st["syndrome_caught"]
            tot["cert"] += st["certificate_caught"]

    stealth = {k: v for k, v in stats.items() if "STEALTH" in k}
    s_def = sum(v["truly_defective"] for v in stealth.values())
    s_syn = sum(v["syndrome_caught"] for v in stealth.values())
    s_cert = sum(v["certificate_caught"] for v in stealth.values())

    print("\nTHE TWO NUMBERS (M5 control excluded)")
    print(f"  all defective mutants   syndrome {tot['syn']:5d}/{tot['def']} "
          f"({100*tot['syn']/max(tot['def'],1):5.1f}%)   "
          f"certificate {tot['cert']:5d}/{tot['def']} "
          f"({100*tot['cert']/max(tot['def'],1):5.1f}%)")
    print(f"  STEALTH class only      syndrome {s_syn:5d}/{s_def} "
          f"({100*s_syn/max(s_def,1):5.1f}%)   "
          f"certificate {s_cert:5d}/{s_def} "
          f"({100*s_cert/max(s_def,1):5.1f}%)")
    ctrl = stats.get("M5_equal_weight_alternative")
    if ctrl:
        print(f"  NEGATIVE CONTROL M5     {ctrl['certificate_caught']}/{ctrl['n']} "
              f"equal-weight alternatives flagged (0 is correct)")

    out = {"schema": "oneq-decoder-mutation-gate/2", "distance": d,
           "rounds": rounds, "shots": a.shots, "p": a.p,
           "per_mutant": stats,
           "all_defective": {"n": tot["def"], "syndrome_caught": tot["syn"],
                             "certificate_caught": tot["cert"]},
           "stealth_only": {"n": s_def, "syndrome_caught": s_syn,
                            "certificate_caught": s_cert},
           "negative_control_M5": ctrl,
           "the_two_numbers": (
               "on the STEALTH class -- corrections that explain the syndrome "
               "EXACTLY and are strictly heavier -- syndrome consistency is "
               "structurally blind, and the certificate is not. That is the "
               "whole claim, and the all-defective row is the diluted version "
               "of it: mixing in mutants that break the syndrome flatters the "
               "detector that only checks the syndrome."),
           "what_this_cannot_model": (
               "mutating decoder OUTPUT models wrong-answer defects, not "
               "crashes, hangs, memory errors, or faults that change which "
               "shots are hard. A seeded-fault campaign against PyMatching's "
               "C++ would cover those; this covers the class the certificate "
               "is about."),
           "negative_control_rationale": (
               "M5 is an equal-weight ALTERNATIVE OPTIMUM, not a defect. "
               "Minimum-weight corrections are not unique and flagging one is "
               "a false alarm. A gate that catches everything is as useless as "
               "one that catches nothing.")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / f"decoder_mutation_gate_d{d}.json")
    # BYTES, NOT TEXT: `write_text` translates a newline to
    # os.linesep on Windows, so re-running this would leave a
    # tracked LF artifact CRLF -- `git diff` empty, `git status`
    # modified. `tools/dem_cert_run.py` already uses the same
    # idiom via newline="".
    dest.write_bytes(json.dumps(out, indent=2).encode("utf-8"))
    print(f"\nevidence -> {dest}")
    if a.corpus:
        cp = pathlib.Path(a.corpus)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(json.dumps({
            "schema": "oneq-seeded-fault-corpus/1",
            "distance": d, "rounds": rounds, "p": a.p, "seed": 7,
            "circuit": "stim surface_code:rotated_memory_z",
            "released_so": ("anyone can run their own detector against the "
                            "same faults and report a third number; the two "
                            "here are only comparable if the corpus is shared"),
            "fields": {"defective": "the claim 'minimum-weight correction for "
                                    "this syndrome' is FALSE",
                       "syndrome_consistent": "the odd-degree set equals the "
                                              "syndrome; a defect can be BOTH "
                                              "consistent and defective, and "
                                              "that intersection is the point"},
            "faults": corpus}, indent=1).encode("utf-8"))
        print(f"seeded-fault corpus ({len(corpus)} faults) -> {cp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
