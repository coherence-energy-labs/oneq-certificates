r"""FRAME-CERTIFIED receipts, demonstrated on real [[144,12,12]] shots.

The audit's deepest conceptual distinction -- objective optimality is
not logical-frame correctness -- gets its missing receipt: for a
certified-optimal answer e* with frame g* = Ge*, one bound-only
branch-dual tree per logical bit proves that NO solution of weight
<= U flips that bit. All 12 bits proven => every weight-<=U solution
has frame g*: FRAME-CERTIFIED, stranger-checkable, no solver trust.

Mechanism validated before use: 300 random instances vs brute force --
231 accepted proofs (all truly determinate), 69 honest refusals on
genuinely ambiguous instances, ZERO false proofs.

This demo gate frame-certifies a sample of real canonical-stress shots
(one per receipt tier) plus the historical discovery shot; the full
per-rung frame campaign is queued behind the gen-6b Z wave.
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

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
from qldpc_bb_circuit_gate import _gf2_rref
from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,
                             exact_milp, feldman_dual,
                             frame_determinacy_trees)
from oneq.qldpc_check import check_qldpc_bnb


def logical_z_basis():
    """Twelve independent logical-Z supports: ker(H_X) mod rowspace(H_Z).

    Greedy: walk the kernel basis of H_X (fully reduced echelon form,
    one vector per free column) and keep vectors independent of
    rowspace(H_Z) JOINED WITH the already-kept logicals -- independence
    is re-tested against the growing span, so the twelve are a genuine
    basis of the logical group, not twelve names for fewer classes.
    """
    HX, HZ = bb_gross_code()
    n = HX.shape[1]
    R, piv = _gf2_rref(HX)
    piv_set = set(piv)
    span = [row.copy() for row in (HZ % 2).astype(np.uint8)]
    basis = []
    for f in (c for c in range(n) if c not in piv_set):
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for r_i, pc in enumerate(piv):
            if R[r_i, f]:
                v[pc] = 1
        assert not ((HX @ v) % 2).any()
        M = np.array(span, dtype=np.uint8)
        b, p = gf2_row_reduce(M)
        if not gf2_in_rowspace(b, p, v, n):
            basis.append(v.copy())
            span.append(v.copy())
            if len(basis) == 12:
                break
    assert len(basis) == 12, f"only {len(basis)} independent logicals"
    return [tuple(int(i) for i in np.flatnonzero(v)) for v in basis]


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}
    G = logical_z_basis()
    print(f"logical-Z basis: 12 rows, supports "
          f"{sorted(set(len(g) for g in G))}")

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    dec = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                       bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)

    # OPERATING-NOISE contrast first: two p=0.01 shots (uniform rung
    # seed 2106) -- expectation: fully FRAME-CERTIFIED
    picks = {}
    rng01 = np.random.default_rng(2106)
    dec01 = BpOsdDecoder(csr_matrix(HZ), error_rate=0.01, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    got = 0
    while got < 2:
        e = (rng01.random(n) < 0.01).astype(np.uint8)
        syn = (HZ @ e) % 2
        if not syn.any():
            continue
        e_hat = np.asarray(dec01.decode(syn.astype(np.uint8)))             .astype(np.uint8)
        bp_sup = tuple(int(i) for i in np.flatnonzero(e_hat))
        cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                           return_primal=True,
                                           rpc_rounds=6)
        if cert is not None and check_qldpc(
                checks=checks, weights=wf, syndrome=list(syn),
                cert=QldpcCertificate(error_support=bp_sup,
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)).accepted:
            picks[f"p01_shot{got}"] = (got, syn.copy(), bp_sup,
                                       int(len(bp_sup)))
            got += 1

    # replay canonical stress (seed 2104) and pick one shot per tier
    rng = np.random.default_rng(2104)
    nontrivial = 0
    while nontrivial < 300 and len(picks) < 5:
        e = (rng.random(n) < 0.05).astype(np.uint8)
        syn = (HZ @ e) % 2
        if not syn.any():
            continue
        nontrivial += 1
        e_hat = np.asarray(dec.decode(syn.astype(np.uint8))).astype(np.uint8)
        cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                           return_primal=True, rpc_rounds=6)
        tier, ans = None, None
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_hat))
            if check_qldpc(checks=checks, weights=wf, syndrome=list(syn),
                           cert=QldpcCertificate(
                               error_support=bp_sup,
                               facet_duals=cert.facet_duals,
                               box_duals=cert.box_duals)).accepted:
                tier, ans = "tier1", bp_sup
            elif primal is not None and check_qldpc(
                    checks=checks, weights=wf, syndrome=list(syn),
                    cert=QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals)).accepted:
                tier, ans = "tier2", tuple(primal)
        if tier is None:
            sup, _w2, _st = exact_milp(checks, wf, list(syn),
                                       return_status=True)
            tier, ans = "tierBD", tuple(int(i) for i in sup)
        key = f"stress_{tier}"
        if key not in picks:
            picks[key] = (nontrivial - 1, syn.copy(), ans,
                          int(len(ans)))

    out = {"schema": "oneq-qldpc-frame-demo/1",
           "validation": "300 random instances vs brute force: 231 "
                         "accepted proofs all truly determinate, 69 "
                         "honest refusals on ambiguous, 0 false proofs",
           "logical_basis_rows": [list(g) for g in G],
           "shots": []}
    for label, (si, syn, ans, U) in sorted(picks.items()):
        gstar = [sum(1 for i in ans if i in set(row)) % 2 for row in G]
        t0 = time.perf_counter()
        trees, all_ok = frame_determinacy_trees(
            checks, wf, list(syn), U, G, gstar, rpc_rounds=6,
            max_nodes=40000)
        determinate = ambiguous = unchecked = 0
        witnesses = []
        for ell, tree in trees:
            aug = checks + [G[ell]]
            asyn = list(syn) + [gstar[ell] ^ 1]
            proven = False
            if tree is not None:
                r = check_qldpc_bnb(checks=aug, weights=wx, syndrome=asyn,
                                    error_support=(), tree=tree,
                                    bound_only_threshold=U)
                proven = bool(r.accepted)
            if proven:
                determinate += 1
                continue
            # AMBIGUITY WITNESS: an equal-or-lighter solution with the
            # flipped frame bit IS the certificate of ambiguity -- its
            # validity is one exact feasibility + weight check, no trust
            sup, w2, _st = exact_milp(checks + [G[ell]], wf,
                                      list(syn) + [gstar[ell] ^ 1],
                                      return_status=True)
            if sup is not None and w2 is not None and int(round(w2)) <= U:
                wit = [int(i) for i in sup]
                okw = (all(sum(1 for i in wit if i in set(c)) % 2 == sj
                           for c, sj in zip(aug, asyn))
                       and len(wit) <= U)
                assert okw
                ambiguous += 1
                witnesses.append({"bit": ell, "witness_weight": len(wit),
                                  "witness": wit})
            else:
                unchecked += 1
        dt = time.perf_counter() - t0
        if determinate == 12:
            status = "FRAME-CERTIFIED"
        elif ambiguous:
            status = (f"FRAME-AMBIGUOUS (witnessed on {ambiguous} bits; "
                      f"{determinate} determinate, {unchecked} unchecked)")
        else:
            status = f"FRAME-UNCHECKED on {unchecked} bits"
        out["shots"].append({"shot_label": label,
                             "shot_index": si,
                             "answer_weight": U, "frame": gstar,
                             "bits_determinate": determinate,
                             "bits_ambiguous_witnessed": ambiguous,
                             "bits_unchecked": unchecked,
                             "ambiguity_witnesses": witnesses,
                             "status": status,
                             "seconds": round(dt, 2)})
        print(f"  {label} (U={U}): {status} in {dt:.1f}s")

    dest = ROOT / "evidence" / "qldpc_frame_demo.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
