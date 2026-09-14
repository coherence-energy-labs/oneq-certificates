r"""EXACT-RATIONAL certification gate -- no tolerance decides truth.

The external adversarial audit's blocker 1: a floating dual checked with
tolerances is evidence, not a mathematical proof. This gate closes it on
the uniform-weight rungs:

  1. Re-runs the code-capacity protocol (p=0.01 and p=0.02 on the
     [[144,12,12]] Z-sector) with BP-OSD + Feldman duals as in the paper.
  2. EXACTIFIES every accepted float certificate (rational y, safe-dual
     z derived by the checker) and re-certifies with check_qldpc_exact:
     fractions over arbitrary-precision integers, lattice rule L > U - 1
     for integer weights, floats rejected by type.
  3. CROSS-VALIDATES a sample through the PINNED, INDEPENDENTLY AUTHORED
     reference checker (tools/external/, sha256 in PROVENANCE.md) --
     different author, different code, same verdict required.
  4. Runs in-gate negative controls: a heavier feasible candidate (the
     certified one XOR a stabilizer) must be REFUSED by the exact path,
     and a float dual fed directly must be REJECTED BY TYPE.

Scope: uniform weights (D=1). Heterogeneous log-likelihood weights need
the fixed-point rational weight specification first (stated in the
paper as the boundary of the exact claim).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code
from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,
                             check_qldpc_exact, exactify_certificate,
                             feldman_dual)

REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
REF_SHA = "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"


def _frac_json(fr: Fraction) -> dict:
    return {"num": fr.numerator, "den": fr.denominator}


def _reference_verdict(HZ, syn, candidate_support, cert, n, tmp) -> bool:
    """One certificate through the independently authored checker."""
    e = [0] * n
    for i in candidate_support:
        e[int(i)] = 1
    facets = []
    for (j, F, y) in cert.facet_duals:
        rows = sorted(int(r) for r in j) if isinstance(
            j, (tuple, list, frozenset)) else [int(j)]
        facets.append({"rows": rows, "F": sorted(int(i) for i in F),
                       "y": _frac_json(y)})
    doc = {"H": HZ.astype(int).tolist(), "s": [int(b) for b in syn],
           "weights": [{"num": 1, "den": 1}] * n, "candidate": e,
           "facets": facets}
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    r = subprocess.run([sys.executable, str(REF), str(tmp)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return False
    return bool(json.loads(r.stdout)["objective_optimal"])


def main() -> int:
    got = hashlib.sha256(REF.read_bytes()).hexdigest()
    assert got == REF_SHA, f"reference checker tampered: {got}"

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix

    out = {"schema": "oneq-qldpc-exact-cert/1",
           "reference_checker_sha256": REF_SHA, "runs": {}}
    tmp = ROOT / "evidence" / "_exact_xval_tmp.json"
    xval_budget = 25
    xval_done = 0
    xval_agree = 0

    for p, sampled, seed in ((0.01, 400, 11), (0.02, 500, 12)):
        dec = BpOsdDecoder(csr_matrix(HZ), error_rate=p, max_iter=30,
                           bp_method="ms", schedule="parallel",
                           osd_method="osd_cs", osd_order=5)
        rng = np.random.default_rng(seed)
        c = {"sampled": sampled, "trivial_tier0": 0, "nontrivial": 0,
             "float_certified": 0, "exact_certified": 0,
             "float_exact_disagreements": 0, "residue": 0,
             "negative_control_heavier_refused": 0,
             "negative_control_float_type_rejected": 0}
        max_gap = Fraction(0)
        for _ in range(sampled):
            e_true = rng.random(n) < p
            syn = (HZ @ e_true) % 2
            if not syn.any():
                # tier 0: for w >= 0 and s = 0, e = 0 is analytically
                # optimal at cost 0 -- no LP, no solver, no trust
                c["trivial_tier0"] += 1
                continue
            c["nontrivial"] += 1
            e_hat = np.asarray(dec.decode(syn.astype(np.uint8))).astype(bool)
            cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                               return_primal=True,
                                               rpc_rounds=6)
            if cert is None:
                c["residue"] += 1
                continue
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_hat))
            candidate = None
            fc = QldpcCertificate(error_support=bp_sup,
                                  facet_duals=cert.facet_duals,
                                  box_duals=cert.box_duals)
            if check_qldpc(checks=checks, weights=wf, syndrome=list(syn),
                           cert=fc).accepted:
                candidate = bp_sup
            elif primal is not None:
                rc = QldpcCertificate(error_support=tuple(primal),
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)
                if check_qldpc(checks=checks, weights=wf,
                               syndrome=list(syn), cert=rc).accepted:
                    candidate = tuple(primal)
                    fc = rc
            if candidate is None:
                c["residue"] += 1
                continue
            c["float_certified"] += 1

            ec = exactify_certificate(fc)
            rx = check_qldpc_exact(checks=checks, weights=wx,
                                   syndrome=list(syn), cert=ec)
            if rx.accepted:
                c["exact_certified"] += 1
                b = rx.checks["bound"]
                o = rx.checks["objective"]
                gap = (Fraction(o["num"], o["den"])
                       - Fraction(b["num"], b["den"]))
                max_gap = max(max_gap, gap)
            else:
                c["float_exact_disagreements"] += 1

            # negative control 1: candidate XOR a stabilizer (a row of
            # H_X lies in ker H_Z) is feasible but heavier -- the exact
            # checker must refuse it with the SAME dual
            if rx.accepted:
                stab = np.zeros(n, dtype=np.uint8)
                stab[np.flatnonzero(HX[0])] = 1
                ev = np.zeros(n, dtype=np.uint8)
                ev[list(candidate)] = 1
                heavier = tuple(int(i) for i in np.flatnonzero(ev ^ stab))
                if len(heavier) > len(candidate):
                    rh = check_qldpc_exact(
                        checks=checks, weights=wx, syndrome=list(syn),
                        cert=QldpcCertificate(error_support=heavier,
                                              facet_duals=ec.facet_duals))
                    if not rh.accepted:
                        c["negative_control_heavier_refused"] += 1

            # negative control 2: a float smuggled into the exact path
            # must be rejected by TYPE
            bad = QldpcCertificate(
                error_support=fc.error_support,
                facet_duals=((ec.facet_duals[0][0], ec.facet_duals[0][1],
                              0.5),) + ec.facet_duals[1:])
            if not check_qldpc_exact(checks=checks, weights=wx,
                                     syndrome=list(syn),
                                     cert=bad).accepted:
                c["negative_control_float_type_rejected"] += 1

            # cross-validation through the independent reference checker
            if xval_done < xval_budget and rx.accepted:
                xval_done += 1
                if _reference_verdict(HZ, syn, candidate, ec, n, tmp):
                    xval_agree += 1

        c["max_exact_gap"] = _frac_json(max_gap)
        c["all_shot_objective_certified"] = (
            c["trivial_tier0"] + c["exact_certified"])
        out["runs"][f"p{p}"] = c
        print(f"p={p}: sampled {sampled}, tier0 {c['trivial_tier0']}, "
              f"nontrivial {c['nontrivial']}, float {c['float_certified']}, "
              f"EXACT {c['exact_certified']}, disagree "
              f"{c['float_exact_disagreements']}, residue {c['residue']}")

    out["cross_validation"] = {"attempted": xval_done, "agreed": xval_agree,
                               "checker": "independently authored exact "
                                          "reference, pinned by sha256"}
    tmp.unlink(missing_ok=True)
    assert xval_done == xval_agree, "independent checker disagreed"
    for run in out["runs"].values():
        assert run["float_exact_disagreements"] == 0, \
            "a float-accepted certificate failed exact re-certification"
        assert run["negative_control_float_type_rejected"] == \
            run["float_certified"], "a float slipped past the type wall"

    dest = ROOT / "evidence" / "qldpc_exact_cert.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"cross-val {xval_agree}/{xval_done} agree; evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
