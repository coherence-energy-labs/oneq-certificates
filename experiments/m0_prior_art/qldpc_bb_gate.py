r"""Certify BP-OSD on the bivariate-bicycle [[144,12,12]] code.

THE TARGET NOBODY HAS A NUMBER FOR. IBM's Starling roadmap decodes this code
family with BP-OSD, and BP-OSD ships no certificate -- not even the
solver-internal kind. This experiment produces, per shot, either a PROOF that
BP-OSD's correction was minimum weight (a feasible Feldman dual meeting its
weight) or a measured optimality gap. Both are new for this code.

THE CODE, constructed rather than imported: the [[144,12,12]] gross code of
Bravyi et al. (Nature 627, 778 (2024)), l=12, m=6, A = x^3 + y + y^2,
B = y^3 + x + x^2 where x = S_l (x) I_m and y = I_l (x) S_m are commuting
shift matrices. H_X = [A | B], H_Z = [B^T | A^T], n = 2lm = 144, and every
check has weight 6 -- so the Feldman relaxation has 2^5 = 32 facets per check,
tiny. The construction is verified in-run: H_X H_Z^T = 0 over GF(2), row
weights all 6, or the experiment refuses to proceed.

NOISE MODEL, stated so the number cannot be over-read: code-capacity
independent X errors at rate p, decoded from the Z-sector syndrome s = H_Z e
with uniform weights (minimum weight = minimum flips). No measurement noise,
no circuit-level correlations -- the first certified number for this code
should be on the cleanest model, with the harder models as the ladder above
it, not smuggled in beneath it.

WHAT A GAP MEANS, sharpened by first contact: when the LP primal is INTEGRAL,
the gap is fully attributed -- the LP's own solution is a minimum-weight
correction certified by the dual beside it, so the shot is REPAIRED, not just
refused. The smoke run's very first gap was exactly this: BP-OSD returned
weight 22 against a tight bound of 6, the difference a logical operator of
weight 24 (in ker H_Z, not in ker H_X) -- a silent logical error today,
a detected-and-repaired one here. Only a FRACTIONAL primal (pseudocodeword)
leaves an unattributed residue, and that residue is measured.

Simulation privilege, used for attribution only: knowing e_true lets the run
report how many BP-OSD faults were logical (fault in ker H_Z minus ker H_X).
The certificate itself never sees e_true -- detection is syndrome-only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,  # noqa: E402
                             exact_milp, feldman_dual)


def gf2_row_reduce(M):
    """Row-reduce M over GF(2); returns (reduced rows, pivot columns)."""
    R = [int("".join(map(str, row)), 2) for row in M]
    nbits = M.shape[1]
    pivots, basis = [], []
    for row in R:
        for pc, b in zip(pivots, basis):
            if (row >> (nbits - 1 - pc)) & 1:
                row ^= b
        if row:
            pc = nbits - 1 - row.bit_length() + 1
            pivots.append(pc)
            basis.append(row)
    return basis, pivots


def gf2_in_rowspace(basis, pivots, v, nbits):
    """Is bit-vector v in the span of the reduced basis?"""
    row = int("".join(map(str, v.astype(int))), 2)
    for pc, b in zip(pivots, basis):
        if (row >> (nbits - 1 - pc)) & 1:
            row ^= b
    return row == 0


def bb_gross_code():
    """H_X, H_Z of the [[144,12,12]] gross code, self-verified."""
    l, m = 12, 6
    Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(Sl, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), Sm)
    A = (np.linalg.matrix_power(x, 3) + y + y @ y) % 2
    B = (np.linalg.matrix_power(y, 3) + x + x @ x) % 2
    HX = np.concatenate([A, B], axis=1) % 2
    HZ = np.concatenate([B.T, A.T], axis=1) % 2
    assert not ((HX @ HZ.T) % 2).any(), "H_X H_Z^T != 0: not a CSS code"
    assert set(HX.sum(1)) == {6} and set(HZ.sum(1)) == {6}, "row weight != 6"
    return HX, HZ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--rpc", type=int, default=0,
                    help="RPC cutting-plane rounds for fractional shots")
    ap.add_argument("--hetero", action="store_true",
                    help="heterogeneous per-qubit error rates (0.5x..2x p) "
                         "with log-likelihood weights, as a real DEM has -- "
                         "kills any silent dependence on uniform weights")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    _HX, HZ = bb_gross_code()
    mchk, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(mchk)]
    wrng = np.random.default_rng(a.seed + 1)
    if a.hetero:
        p_i = a.p * wrng.uniform(0.5, 2.0, n)
        weights = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
    else:
        p_i = np.full(n, float(a.p))
        weights = {i: 1.0 for i in range(n)}

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    dec = BpOsdDecoder(csr_matrix(HZ),
                       error_channel=[float(v) for v in p_i], max_iter=30,
                       bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)

    # An undetected X-fault is HARMLESS iff it lies in rowspace(H_X) -- the
    # X-stabilizer group -- and LOGICAL otherwise. Membership must be decided
    # by GF(2) elimination; the first draft tested H_X @ fault != 0, which a
    # stabilizer can also satisfy (H_X H_X^T != 0 in general), and would have
    # overcounted logicals in the headline number.
    xbasis, xpiv = gf2_row_reduce(_HX % 2)

    def is_logical(fault):
        return not gf2_in_rowspace(xbasis, xpiv, fault, n)

    rng = np.random.default_rng(a.seed)
    nontrivial = proven = repaired = 0
    consistent = 0
    bposd_faults = logical_faults = repair_logical = 0
    milp_decoded = milp_bposd_subopt = milp_bposd_logical = milp_logical = 0
    gaps = []
    t_cert = 0.0
    for _s in range(a.shots):
        e_true = rng.random(n) < p_i
        syndrome = (HZ @ e_true) % 2
        if not syndrome.any():
            continue
        nontrivial += 1
        e_hat = np.asarray(dec.decode(syndrome.astype(np.uint8))).astype(bool)
        if not ((HZ @ e_hat) % 2 == syndrome).all():
            continue                     # syndrome-inconsistent: no receipt
        consistent += 1
        support = tuple(int(i) for i in np.flatnonzero(e_hat))
        w_hat = float(sum(weights[i] for i in support))

        t0 = time.perf_counter()
        cert, bound, primal = feldman_dual(checks, weights, list(syndrome),
                                           return_primal=True,
                                           rpc_rounds=a.rpc)
        t_cert += time.perf_counter() - t0
        if cert is None:
            continue
        full = QldpcCertificate(error_support=support,
                                facet_duals=cert.facet_duals,
                                box_duals=cert.box_duals)
        r = check_qldpc(checks=checks, weights=weights,
                        syndrome=list(syndrome), cert=full)
        if r.accepted:
            proven += 1
            continue
        if "IMPOSSIBLE" in r.reason:
            print(f"IMPOSSIBLE OUTCOME: {r.reason}", file=sys.stderr)
            return 2
        if primal is not None:
            # THE REPAIR PATH -- and the repaired answer goes through the SAME
            # independent checker; a repair that cannot certify does not count.
            rep = QldpcCertificate(error_support=primal,
                                   facet_duals=cert.facet_duals,
                                   box_duals=cert.box_duals)
            rr = check_qldpc(checks=checks, weights=weights,
                             syndrome=list(syndrome), cert=rep)
            if rr.accepted:
                repaired += 1
                bposd_faults += 1        # attributed: optimum existed below w_hat
                if is_logical(e_hat ^ e_true):
                    logical_faults += 1
                e_rep = np.zeros(n, dtype=bool)
                e_rep[list(primal)] = True
                if is_logical(e_rep ^ e_true):
                    repair_logical += 1
                continue
        # DEGRADED TIER: exact MILP on the fractional residue. No stranger-
        # checkable witness -- optimality here rests on trusting HiGHS, and
        # the receipt says so -- but it decodes exactly AND attributes the
        # gap: was BP-OSD optimal-but-unprovable, or actually wrong?
        msup, mw = exact_milp(checks, weights, list(syndrome))
        if mw is not None:
            milp_decoded += 1
            if mw < w_hat - 1e-9:
                milp_bposd_subopt += 1
                e_m = np.zeros(n, dtype=bool)
                e_m[list(msup)] = True
                if is_logical(e_hat ^ e_true):
                    milp_bposd_logical += 1
                if is_logical(e_m ^ e_true):
                    milp_logical += 1
        gaps.append(w_hat - bound)

    g = np.array(gaps) if gaps else np.array([0.0])
    closed = proven + repaired
    print(f"[[144,12,12]] gross code, code-capacity X noise p={a.p}")
    print(f"  nontrivial shots        : {nontrivial}")
    print(f"  BP-OSD syndrome-valid   : {consistent}/{nontrivial}")
    print(f"  BP-OSD PROVEN min-weight: {proven}/{consistent} "
          f"({100*proven/max(consistent,1):.1f}%)")
    print(f"  REPAIRED (LP integral)  : {repaired}  -- BP-OSD suboptimal, "
          f"certified replacement shipped")
    print(f"    of which LOGICAL fault: {logical_faults} "
          f"(silent logical errors today; detected + repaired here)")
    print(f"    repair itself logical : {repair_logical} "
          f"(min-weight != truth's coset; honest cost of conformance)")
    print(f"  CERTIFIED total         : {closed}/{consistent} "
          f"({100*closed/max(consistent,1):.2f}%)")
    if gaps:
        print(f"  fractional residue      : {len(gaps)} shots, gap mean "
              f"{g.mean():.3f} max {g.max():.1f} (pseudocodewords)")
        print(f"    MILP degraded tier    : {milp_decoded}/{len(gaps)} decoded "
              f"exactly (solver-trust, labeled); BP-OSD suboptimal on "
              f"{milp_bposd_subopt} (logical on {milp_bposd_logical}), MILP "
              f"answer logical on {milp_logical}")
    print(f"  certificate cost        : {1e3*t_cert/max(consistent,1):.1f} ms/shot")

    out = {"schema": "oneq-qldpc-bb-gate/2",
           "rpc_rounds": a.rpc,
           "code": "[[144,12,12]] bivariate bicycle (gross), self-verified "
                   "CSS with row weight 6",
           "noise": (f"code-capacity independent X, heterogeneous rates "
                     f"0.5x..2x {a.p} with log-likelihood weights" if a.hetero
                     else f"code-capacity independent X at p={a.p}, "
                          f"uniform weights"),
           "shots": a.shots, "nontrivial": nontrivial,
           "bp_osd_syndrome_valid": consistent,
           "bp_osd_proven_minimum_weight": proven,
           "repaired_by_integral_lp": repaired,
           "bp_osd_faults_attributed": bposd_faults,
           "bp_osd_logical_faults_detected_and_repaired": logical_faults,
           "repairs_that_were_themselves_logical_vs_truth": repair_logical,
           "logical_membership_test": ("GF(2) rowspace(H_X) elimination -- "
                                       "NOT H_X@fault!=0, which stabilizers "
                                       "can also satisfy"),
           "certified_total": closed,
           "certified_rate": closed / max(consistent, 1),
           "fractional_residue": ({"n": len(gaps), "mean": float(g.mean()),
                                   "median": float(np.median(g)),
                                   "max": float(g.max())} if gaps else None),
           "degraded_tier_milp": ({"decoded_exactly": milp_decoded,
                                   "bp_osd_confirmed_suboptimal":
                                       milp_bposd_subopt,
                                   "bp_osd_fault_logical_on_this_tier":
                                       milp_bposd_logical,
                                   "milp_answer_logical_vs_truth":
                                       milp_logical,
                                   "trust_model": "HiGHS MIP optimality, NO "
                                       "stranger-checkable witness -- the "
                                       "labeled difference from the "
                                       "certified tier"} if gaps else None),
           "cert_ms_per_shot": 1e3 * t_cert / max(consistent, 1),
           "what_certified_means": ("a feasible dual of the Feldman "
                                    "relaxation meets the shipped answer's "
                                    "weight: LP-dual <= LP-opt <= integer-opt "
                                    "forces optimality, checkable by a "
                                    "stranger from (H, w, s) alone; repaired "
                                    "answers pass the SAME checker"),
           "what_residue_means": ("fractional LP primal (pseudocodeword): the "
                                  "only unattributed bucket; RPC cuts are the "
                                  "designed next rung for it"),
           "logical_fault_note": ("attribution to 'logical' uses e_true, a "
                                  "simulation privilege; DETECTION is "
                                  "syndrome-only and works on hardware"),
           "why_it_is_new": ("BP-OSD ships no certificate of any kind; this "
                             "is a per-shot optimality proof, or a certified "
                             "repair, on the code family IBM's Starling "
                             "roadmap decodes with it")}
    dest = pathlib.Path(a.out) if a.out else ROOT / "evidence" / "qldpc_bb_gate.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
