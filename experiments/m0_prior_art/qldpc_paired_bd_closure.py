r"""Close the paired-stress run's last solver-tier shots.

The cold audit's honest disclosure: the dedicated paired-stress sample
(seed 20260803, the 173-vs-0 discordance) predates the branch-dual
closer, so 20 of its 300 pipeline corrections came from the MIP tier
without certificates. This gate replays that exact run and closes them
with branch-dual trees, verified by the production checker AND the
independently implemented tree checker B.

Nothing about the paired 2x2 changes: the branch-dual closure certifies
the SAME correction the MILP proposed, so the contingency table is
untouched by construction -- this is a receipt upgrade, not a result
change, and the gate asserts the table is identical.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,
                             exact_milp, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}
    xb, xp = gf2_row_reduce(HX % 2)

    def is_logical(fault):
        if ((HZ @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(xb, xp, fault, n)

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    dec = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                       bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)

    rng = np.random.default_rng(20260803)      # the paired run's seed
    n00 = n01 = n10 = n11 = 0
    tiers = {"tier1": 0, "tier2": 0, "tierBD": 0, "tier3_left": 0}
    trees = []
    shots = 0
    while shots < 300:
        e_true = (rng.random(n) < 0.05).astype(np.uint8)
        syn = (HZ @ e_true) % 2
        if not syn.any():
            continue
        shots += 1
        e_bp = np.asarray(dec.decode(syn.astype(np.uint8))).astype(np.uint8)
        bp_wrong = is_logical((e_bp ^ e_true) % 2)
        cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                           return_primal=True, rpc_rounds=6)
        ans, tier = None, None
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
            if check_qldpc(checks=checks, weights=wf, syndrome=list(syn),
                           cert=QldpcCertificate(
                               error_support=bp_sup,
                               facet_duals=cert.facet_duals,
                               box_duals=cert.box_duals)).accepted:
                ans, tier = e_bp, "tier1"
            elif primal is not None and check_qldpc(
                    checks=checks, weights=wf, syndrome=list(syn),
                    cert=QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals)).accepted:
                rp = np.zeros(n, dtype=np.uint8)
                rp[list(primal)] = 1
                ans, tier = rp, "tier2"
        if ans is None:
            sup, wt, _st = exact_milp(checks, wf, list(syn),
                                      return_status=True)
            assert sup is not None
            U = int(round(float(wt)))
            tree, nodes = certified_bnb(checks, wf, list(syn), U,
                                        rpc_rounds=6, max_nodes=60000)
            closed = False
            if tree is not None:
                rb = check_qldpc_bnb(checks=checks, weights=wx,
                                     syndrome=list(syn),
                                     error_support=tuple(int(i)
                                                         for i in sup),
                                     tree=tree)
                closed = bool(rb.accepted)
            rp = np.zeros(n, dtype=np.uint8)
            rp[[int(i) for i in sup]] = 1
            ans = rp
            if closed:
                tier = "tierBD"
                trees.append({"shot_index": shots - 1, "nodes": nodes,
                              "answer_weight": U,
                              "syndrome_support":
                                  [int(j) for j in np.flatnonzero(syn)],
                              "candidate": [int(i) for i in sup]})
            else:
                tier = "tier3_left"
        tiers[tier] += 1
        pipe_wrong = is_logical((ans ^ e_true) % 2)
        if bp_wrong and pipe_wrong:
            n11 += 1
        elif bp_wrong:
            n10 += 1
        elif pipe_wrong:
            n01 += 1
        else:
            n00 += 1

    # the paired table must be IDENTICAL to the cited artifact
    cited = json.loads((ROOT / "evidence" / "qldpc_paired_stress.json")
                       .read_text(encoding="utf-8"))["paired_2x2"]
    now = {"both_correct": n00, "bp_correct_pipeline_wrong": n01,
           "bp_wrong_pipeline_correct": n10, "both_wrong": n11}
    same = all(cited[k] == now[k] for k in now)
    print(f"paired 2x2 replay: {now} (identical to cited: {same})")
    assert same, "closure changed the contingency table: investigate"

    out = {"schema": "oneq-qldpc-paired-bd-closure/1",
           "replayed_seed": 20260803,
           "paired_2x2": now, "table_identical_to_cited": True,
           "tiers_after_closure": tiers,
           "solver_tier_shots_closed": tiers["tierBD"],
           "solver_tier_shots_remaining": tiers["tier3_left"],
           "trees": trees,
           "note": ("the closure certifies the SAME correction the MILP "
                    "proposed, so the contingency table is unchanged by "
                    "construction -- a receipt upgrade, asserted here")}
    dest = ROOT / "evidence" / "qldpc_paired_bd_closure.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"tiers after closure: {tiers}")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
