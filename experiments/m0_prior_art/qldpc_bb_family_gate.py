r"""THE FAMILY: exact certification across the bivariate-bicycle codes.

Everything so far lives on one code. This gate asks whether the
receipt ladder is a property of the METHOD or of [[144,12,12]], by
running the full exact pipeline (tier 0/1/2 + branch-dual closure,
integer weights, exact rational certificates, reference-checker pass)
across the published BB family.

The constructor is general -- QC(A, B) over the (l, m) torus with
monomial exponents read from the published table -- and it SELF-CHECKS:
each code's computed [[n, k]] must match the published parameters
before a single shot is decoded. A constructor that silently builds the
wrong code would otherwise produce beautiful, meaningless receipts.

Codes (Bravyi et al., Nature 627:778, Table 3):
  [[72,12,6]]    l=6,  m=6   A = x^3+y+y^2      B = y^3+x+x^2
  [[108,8,10]]   l=9,  m=6   A = x^3+y+y^2      B = y^3+x+x^2
  [[144,12,12]]  l=12, m=6   A = x^3+y+y^2      B = y^3+x+x^2   (gross)
  [[288,12,18]]  l=12, m=12  A = x^3+y^2+y^7    B = y^3+x+x^2
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import gf2_in_rowspace, gf2_row_reduce
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,
                             check_qldpc_exact, exact_milp,
                             exactify_certificate, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb

REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
REF_SHA = "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"

# (label, l, m, A monomials, B monomials) with monomials as (var, power)
FAMILY = [
    ("[[72,12,6]]", 6, 6,
     [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)],
     72, 12),
    ("[[108,8,10]]", 9, 6,
     [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)],
     108, 8),
    ("[[144,12,12]]", 12, 6,
     [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)],
     144, 12),
    ("[[288,12,18]]", 12, 12,
     [("x", 3), ("y", 2), ("y", 7)], [("y", 3), ("x", 1), ("x", 2)],
     288, 12),
]


def bb_code(l: int, m: int, Amons, Bmons):
    """QC(A,B) check matrices for the (l,m) torus, general monomials."""
    Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(Sl, np.eye(m, dtype=np.uint8)).astype(np.uint8)
    y = np.kron(np.eye(l, dtype=np.uint8), Sm).astype(np.uint8)

    def mono(var, p):
        M = np.eye(l * m, dtype=np.uint8)
        base = x if var == "x" else y
        for _ in range(p):
            M = (M @ base) % 2
        return M

    A = np.zeros((l * m, l * m), dtype=np.uint8)
    for v, p in Amons:
        A = (A + mono(v, p)) % 2
    B = np.zeros((l * m, l * m), dtype=np.uint8)
    for v, p in Bmons:
        B = (B + mono(v, p)) % 2
    HX = np.concatenate([A, B], axis=1) % 2
    HZ = np.concatenate([B.T, A.T], axis=1) % 2
    return HX, HZ


def gf2_rank(M):
    M = (M.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for rr in range(r, rows):
            if M[rr, c]:
                piv = rr
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for rr in range(rows):
            if rr != r and M[rr, c]:
                M[rr] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def main() -> int:
    got = hashlib.sha256(REF.read_bytes()).hexdigest()
    assert got == REF_SHA, "reference checker tampered"
    spec = importlib.util.spec_from_file_location("ref_checker", REF)
    ref = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = ref
    spec.loader.exec_module(ref)

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix

    out = {"schema": "oneq-qldpc-bb-family/1",
           "reference_checker_sha256": REF_SHA,
           "protocol": ("code-capacity X errors on the Z sector, "
                        "p=0.01, uniform weights, 200 nontrivial shots "
                        "per code; tier 0/1/2 + branch-dual closure; "
                        "every flat certificate through the pinned "
                        "reference checker"),
           "codes": {}}

    for (label, l, m, Am, Bm, n_pub, k_pub) in FAMILY:
        HX, HZ = bb_code(l, m, Am, Bm)
        n = HX.shape[1]
        # SELF-CHECK against the published parameters before decoding
        assert n == n_pub, f"{label}: built n={n}, published {n_pub}"
        rk = gf2_rank(HX)
        k = n - 2 * rk
        assert k == k_pub, f"{label}: computed k={k}, published {k_pub}"
        comm = (HX @ HZ.T) % 2
        assert not comm.any(), f"{label}: HX HZ^T != 0"
        print(f"{label}: constructor self-check OK "
              f"(n={n}, k={k}, rank={rk}, commuting)")

        mrows = HZ.shape[0]
        checks = [tuple(int(i) for i in np.flatnonzero(HZ[j]))
                  for j in range(mrows)]
        wint = {i: 1 for i in range(n)}
        wf = {i: 1.0 for i in range(n)}
        wx = {i: Fraction(1) for i in range(n)}
        dense = [[1 if i in set(c) else 0 for i in range(n)]
                 for c in checks]
        xb, xp = gf2_row_reduce(HX % 2)

        dec = BpOsdDecoder(csr_matrix(HZ), error_rate=0.01, max_iter=30,
                           bp_method="ms", schedule="parallel",
                           osd_method="osd_cs", osd_order=5)
        rng = np.random.default_rng(4200 + n)
        c = {"nontrivial": 0, "trivial_tier0": 0, "sampled": 0,
             "tier1": 0, "tier2": 0, "tierBD": 0, "open": 0,
             "exact_certified": 0, "ref_checked": 0, "ref_agreed": 0,
             "bp_osd_logical_faults_repaired": 0}
        t0 = time.perf_counter()
        while c["nontrivial"] < 200:
            c["sampled"] += 1
            e_true = (rng.random(n) < 0.01).astype(np.uint8)
            syn = (HZ @ e_true) % 2
            if not syn.any():
                c["trivial_tier0"] += 1
                continue
            c["nontrivial"] += 1
            e_bp = np.asarray(dec.decode(syn.astype(np.uint8))) \
                .astype(np.uint8)
            cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                               return_primal=True,
                                               rpc_rounds=6)
            ans, tier, fc = None, None, None
            if cert is not None:
                bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
                f1 = QldpcCertificate(error_support=bp_sup,
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)
                if check_qldpc(checks=checks, weights=wf,
                               syndrome=list(syn), cert=f1).accepted:
                    ans, tier, fc = bp_sup, "tier1", f1
                elif primal is not None:
                    f2 = QldpcCertificate(error_support=tuple(primal),
                                          facet_duals=cert.facet_duals,
                                          box_duals=cert.box_duals)
                    if check_qldpc(checks=checks, weights=wf,
                                   syndrome=list(syn), cert=f2).accepted:
                        ans, tier, fc = tuple(primal), "tier2", f2
            if ans is not None:
                c[tier] += 1
                ec = exactify_certificate(fc)
                rx = check_qldpc_exact(checks=checks, weights=wx,
                                       syndrome=list(syn), cert=ec)
                if rx.accepted:
                    c["exact_certified"] += 1
                    e = [0] * n
                    for i in ans:
                        e[int(i)] = 1
                    facets = []
                    for (j, F, y) in ec.facet_duals:
                        rows = (sorted(int(r) for r in j)
                                if isinstance(j, (tuple, list, frozenset))
                                else [int(j)])
                        facets.append({"rows": rows,
                                       "F": sorted(int(i) for i in F),
                                       "y": {"num": y.numerator,
                                             "den": y.denominator}})
                    rep = ref.check_certificate(
                        {"H": dense, "s": [int(b) for b in syn],
                         "weights": [{"num": 1, "den": 1}] * n,
                         "candidate": e, "facets": facets})
                    c["ref_checked"] += 1
                    c["ref_agreed"] += int(rep["objective_optimal"])
                if tier == "tier2":
                    fault = (e_bp ^ e_true).astype(np.uint8)
                    if not ((HZ @ fault) % 2).any() and \
                            not gf2_in_rowspace(xb, xp, fault, n):
                        c["bp_osd_logical_faults_repaired"] += 1
            else:
                sup, wt, _st = exact_milp(checks, wf, list(syn),
                                          return_status=True)
                if sup is None:
                    c["open"] += 1
                    continue
                U = int(round(float(wt)))
                tree, _nodes = certified_bnb(checks, wf, list(syn), U,
                                             rpc_rounds=6,
                                             max_nodes=60000)
                closed = False
                if tree is not None:
                    rb = check_qldpc_bnb(
                        checks=checks, weights=wx, syndrome=list(syn),
                        error_support=tuple(int(i) for i in sup),
                        tree=tree)
                    closed = bool(rb.accepted)
                if closed:
                    c["tierBD"] += 1
                    c["exact_certified"] += 1
                else:
                    c["open"] += 1
        c["seconds"] = round(time.perf_counter() - t0, 1)
        c["code"] = {"n": n, "k": k, "l": l, "m": m,
                     "published": f"{label} (Bravyi et al. Table 3)"}
        out["codes"][label] = c
        print(f"  {label}: {c['exact_certified']}/{c['nontrivial']} exact "
              f"(t1 {c['tier1']}, t2 {c['tier2']}, BD {c['tierBD']}, "
              f"open {c['open']}), ref {c['ref_agreed']}/"
              f"{c['ref_checked']}, {c['seconds']}s")
        assert c["ref_agreed"] == c["ref_checked"]

    dest = ROOT / "evidence" / "qldpc_bb_family.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
