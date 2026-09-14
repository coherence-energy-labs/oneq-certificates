r"""FRAME CAMPAIGN: logical-frame determinacy as a receipt field, at scale.

The frame demo proved the mechanism on five shots. This runs it across
the held-out corpus so that "FRAME-CERTIFIED" becomes a measured rate
per rung rather than a feasibility claim.

For each shot: take the certified answer e*, compute its frame
g* = Ge* over a 12-row logical-Z basis, and prove -- one bound-only
branch-dual tree per logical bit, in exact arithmetic with no solver --
that no solution of weight <= U flips that bit. The threshold is
min > U (NOT U - delta): a differing-frame solution AT weight U would
break determinacy, so the bound must exclude it.

Verdicts, per shot:
  FRAME-CERTIFIED   all 12 bits proven determinate
  FRAME-AMBIGUOUS   at least one bit has a WITNESS: an equal-or-lighter
                    solution with the opposite bit (exhibited, exactly
                    checkable -- ambiguity is proven, not assumed)
  FRAME-UNCHECKED   at least one bit neither proven nor witnessed
                    within budget

Per-rung caps are declared and coverage is reported honestly: this is
an offline analysis, not a per-shot online cost.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code
from qldpc_frame_demo_gate import logical_z_basis
from oneq.qldpc_cert import exact_milp, frame_determinacy_trees
from oneq.qldpc_check import check_qldpc_bnb

CAP = 120                      # declared per-rung cap


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}
    G = logical_z_basis()
    print(f"logical-Z basis: {len(G)} rows, supports "
          f"{sorted(set(len(g) for g in G))}")

    out = {"schema": "oneq-qldpc-frame-campaign/1",
           "threshold_rule": ("bound-only trees prove min > U on the "
                              "opposite-frame system: a differing-frame "
                              "solution AT weight U would break "
                              "determinacy, so U (not U-delta) is the "
                              "correct exclusion threshold"),
           "per_rung_cap": CAP,
           "mechanism_validation": ("300 random instances vs brute "
                                    "force: 231 proofs all truly "
                                    "determinate, 69 honest refusals, "
                                    "0 false proofs"),
           "rungs": {}}

    # uniform-weight code-capacity rungs from the held-out records
    for tag in ("uniform_p01", "uniform_p02", "stress_p05"):
        f = ROOT / "evidence" / "per_shot" / f"heldout_{tag}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(x) for x in
                f.read_text(encoding="utf-8").splitlines()]
        c = {"shots_available": len(recs), "shots_checked": 0,
             "frame_certified": 0, "frame_ambiguous": 0,
             "frame_unchecked": 0, "bits_proven": 0, "bits_total": 0,
             "witnesses": 0}
        t0 = time.perf_counter()
        for r in recs[:CAP]:
            rec = r.get("receipt")
            if not rec or not rec.get("answer"):
                continue
            ans = [int(i) for i in rec["answer"]]
            ss = set(r["instance"]["syndrome_support"])
            syn = [1 if j in ss else 0 for j in range(m)]
            U = len(ans)
            gstar = [sum(1 for i in ans if i in set(row)) % 2
                     for row in G]
            trees, _ok = frame_determinacy_trees(
                checks, wf, syn, U, G, gstar, rpc_rounds=6,
                max_nodes=40000)
            proven = ambiguous = unchecked = 0
            for ell, tree in trees:
                aug = checks + [G[ell]]
                asyn = syn + [gstar[ell] ^ 1]
                ok = False
                if tree is not None:
                    ok = check_qldpc_bnb(
                        checks=aug, weights=wx, syndrome=asyn,
                        error_support=(), tree=tree,
                        bound_only_threshold=U).accepted
                if ok:
                    proven += 1
                    continue
                sup, w2, _st = exact_milp(aug, wf, asyn,
                                          return_status=True)
                if sup is not None and w2 is not None and \
                        int(round(w2)) <= U:
                    ambiguous += 1
                else:
                    unchecked += 1
            c["shots_checked"] += 1
            c["bits_proven"] += proven
            c["bits_total"] += len(G)
            c["witnesses"] += ambiguous
            if proven == len(G):
                c["frame_certified"] += 1
            elif ambiguous:
                c["frame_ambiguous"] += 1
            else:
                c["frame_unchecked"] += 1
        c["seconds"] = round(time.perf_counter() - t0, 1)
        c["frame_certified_rate"] = (
            round(100 * c["frame_certified"] / c["shots_checked"], 2)
            if c["shots_checked"] else None)
        out["rungs"][tag] = c
        print(f"  {tag:12s} {c['frame_certified']}/{c['shots_checked']} "
              f"FRAME-CERTIFIED ({c['frame_certified_rate']}%), "
              f"ambiguous {c['frame_ambiguous']}, unchecked "
              f"{c['frame_unchecked']}, bits {c['bits_proven']}/"
              f"{c['bits_total']}, {c['seconds']}s")

    tot_c = sum(r["frame_certified"] for r in out["rungs"].values())
    tot_s = sum(r["shots_checked"] for r in out["rungs"].values())
    out["total"] = {"shots_checked": tot_s, "frame_certified": tot_c,
                    "rate": round(100 * tot_c / tot_s, 2) if tot_s else None}
    print(f"TOTAL: {tot_c}/{tot_s} FRAME-CERTIFIED "
          f"({out['total']['rate']}%)")
    dest = ROOT / "evidence" / "qldpc_frame_campaign.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
