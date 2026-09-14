r"""BP-OSD configuration sensitivity -- the referee's oldest open ask.

Every rate in both papers is measured against ONE BP-OSD
configuration (min-sum, parallel, 30 iterations, OSD-CS order 5). The
referee's round-1 objection stands until someone asks how much the
result depends on that choice. This gate sweeps the configuration and
measures three things per setting, on identical syndromes:

  tier1  how often BP-OSD's OWN answer is certified optimal
         (this is the decoder-quality axis: a better-configured
         decoder should raise it)
  exact  total exact-certified rate (tier1 + tier2 + BD)
         (this is the INSTRUMENT axis: the certification layer should
         be nearly INVARIANT to decoder configuration, because a
         refused answer gets repaired or branch-closed)
  logical BP-OSD's own retrospective logical-error count

The scientifically load-bearing prediction: tier1 moves with
configuration, exact does not. If exact moved too, the certification
rate would be a property of the decoder rather than of the problem.

Same 200 syndromes for every configuration (seed fixed, sampled once).
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
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,
                             check_qldpc_exact, exact_milp,
                             exactify_certificate, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb

CONFIGS = [
    ("baseline ms/parallel/30/osd_cs5", dict(bp_method="ms",
     schedule="parallel", max_iter=30, osd_method="osd_cs", osd_order=5)),
    ("fewer iters (10)", dict(bp_method="ms", schedule="parallel",
     max_iter=10, osd_method="osd_cs", osd_order=5)),
    ("more iters (100)", dict(bp_method="ms", schedule="parallel",
     max_iter=100, osd_method="osd_cs", osd_order=5)),
    ("serial schedule", dict(bp_method="ms", schedule="serial",
     max_iter=30, osd_method="osd_cs", osd_order=5)),
    ("product-sum BP", dict(bp_method="ps", schedule="parallel",
     max_iter=30, osd_method="osd_cs", osd_order=5)),
    ("OSD-0 (no combination sweep)", dict(bp_method="ms",
     schedule="parallel", max_iter=30, osd_method="osd0", osd_order=0)),
    ("OSD-CS order 10", dict(bp_method="ms", schedule="parallel",
     max_iter=30, osd_method="osd_cs", osd_order=10)),
]


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}
    xb, xp = gf2_row_reduce(HX % 2)

    # ONE syndrome set, reused by every configuration
    p = 0.03
    rng = np.random.default_rng(777)
    shots = []
    while len(shots) < 200:
        e = (rng.random(n) < p).astype(np.uint8)
        syn = (HZ @ e) % 2
        if syn.any():
            shots.append((syn, e))
    print(f"sensitivity sweep: {len(shots)} fixed syndromes at p={p}, "
          f"{len(CONFIGS)} configurations")

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    out = {"schema": "oneq-qldpc-bposd-sensitivity/1",
           "protocol": (f"[[144,12,12]] Z-sector, code capacity p={p}, "
                        f"200 fixed nontrivial syndromes reused across "
                        f"configurations; uniform weights"),
           "configs": {}}

    for label, cfg in CONFIGS:
        dec = BpOsdDecoder(csr_matrix(HZ), error_rate=p, **cfg)
        c = {"config": cfg, "tier1": 0, "tier2": 0, "tierBD": 0,
             "open": 0, "exact_certified": 0,
             "bp_osd_logical_errors": 0}
        t0 = time.perf_counter()
        for (syn, e_true) in shots:
            e_bp = np.asarray(dec.decode(syn.astype(np.uint8))) \
                .astype(np.uint8)
            fault = (e_bp ^ e_true).astype(np.uint8)
            if ((HZ @ fault) % 2).any() or \
                    not gf2_in_rowspace(xb, xp, fault, n):
                c["bp_osd_logical_errors"] += 1
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
                if check_qldpc_exact(checks=checks, weights=wx,
                                     syndrome=list(syn),
                                     cert=ec).accepted:
                    c["exact_certified"] += 1
            else:
                sup, wt, _st = exact_milp(checks, wf, list(syn),
                                          return_status=True)
                if sup is None:
                    c["open"] += 1
                    continue
                U = int(round(float(wt)))
                tree, _nn = certified_bnb(checks, wf, list(syn), U,
                                          rpc_rounds=6, max_nodes=60000)
                if tree is not None and check_qldpc_bnb(
                        checks=checks, weights=wx, syndrome=list(syn),
                        error_support=tuple(int(i) for i in sup),
                        tree=tree).accepted:
                    c["tierBD"] += 1
                    c["exact_certified"] += 1
                else:
                    c["open"] += 1
        c["seconds"] = round(time.perf_counter() - t0, 1)
        out["configs"][label] = c
        print(f"  {label:34s} tier1 {c['tier1']:3d}  exact "
              f"{c['exact_certified']:3d}/200  BP-OSD logicals "
              f"{c['bp_osd_logical_errors']:3d}  ({c['seconds']}s)")

    t1s = [c["tier1"] for c in out["configs"].values()]
    exs = [c["exact_certified"] for c in out["configs"].values()]
    out["summary"] = {
        "tier1_min": min(t1s), "tier1_max": max(t1s),
        "tier1_spread": max(t1s) - min(t1s),
        "exact_min": min(exs), "exact_max": max(exs),
        "exact_spread": max(exs) - min(exs),
        "finding": ("tier-1 (BP-OSD's own answer certified) moves with "
                    "configuration; the exact-certified total is the "
                    "instrument's rate and should be near-invariant")}
    print(f"\nSUMMARY: tier1 spread {out['summary']['tier1_spread']} "
          f"({min(t1s)}-{max(t1s)}), exact spread "
          f"{out['summary']['exact_spread']} ({min(exs)}-{max(exs)})")
    dest = ROOT / "evidence" / "qldpc_bposd_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
