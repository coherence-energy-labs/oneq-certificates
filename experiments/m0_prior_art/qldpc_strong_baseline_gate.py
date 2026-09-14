r"""THE DECISIVE EXPERIMENT: does the pipeline beat a STRONG BP-OSD?

The sensitivity sweep found that BP-OSD at 100 iterations, or with
product-sum BP, made ZERO logical errors on p=0.03 shots where the
papers' baseline configuration (30 iterations, OSD-CS 5) made 34. That
puts the "the certificate out-decodes BP-OSD" claim in doubt: it may be
an artifact of an under-configured baseline rather than a property of
the instrument.

Recording that as a caveat is not solving it. This gate settles it by
measurement, across the noise ladder, with the SAME syndromes handed to
every configuration:

  BASELINE   min-sum, parallel, 30 iters, OSD-CS order 5   (the papers')
  STRONG     min-sum, parallel, 100 iters, OSD-CS order 10
  STRONGEST  product-sum, parallel, 100 iters, OSD-CS order 10

For each (p, config): BP-OSD's own logical-error count, the pipeline's
logical-error count on the same shots, the paired 2x2, and the tier-1
rate (how often the decoder's own answer is already certified optimal).

Three outcomes, all publishable, decided by the data:
  A) the pipeline still beats STRONG      -> the claim survives, and
                                             now against a real baseline
  B) STRONG matches the pipeline          -> the "out-decodes" framing
                                             dies; certification
                                             COVERAGE is the claim
  C) it depends on noise                  -> publish the crossover,
                                             which is more useful than
                                             either flat claim
"""
from __future__ import annotations

import json
import math
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
                             exact_milp, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb

CONFIGS = {
    "baseline_30_osdcs5": dict(bp_method="ms", schedule="parallel",
                               max_iter=30, osd_method="osd_cs",
                               osd_order=5),
    "strong_100_osdcs10": dict(bp_method="ms", schedule="parallel",
                               max_iter=100, osd_method="osd_cs",
                               osd_order=10),
    "strongest_ps_100_osdcs10": dict(bp_method="ps", schedule="parallel",
                                     max_iter=100, osd_method="osd_cs",
                                     osd_order=10),
}
NOISE = [0.01, 0.02, 0.03, 0.05, 0.07]
SHOTS = 200


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

    out = {"schema": "oneq-qldpc-strong-baseline/1",
           "question": ("does the certified pipeline still beat BP-OSD "
                        "when BP-OSD is WELL configured, or was the "
                        "advantage an artifact of the papers' baseline?"),
           "protocol": (f"[[144,12,12]] Z-sector, {SHOTS} fixed "
                        f"nontrivial syndromes per noise level shared "
                        f"across all configurations, uniform weights"),
           "levels": {}}

    for p in NOISE:
        # ONE syndrome set per noise level, reused by every config
        rng = np.random.default_rng(int(10000 * p) + 31337)
        shots = []
        while len(shots) < SHOTS:
            e = (rng.random(n) < p).astype(np.uint8)
            syn = (HZ @ e) % 2
            if syn.any():
                shots.append((syn, e))

        # the pipeline's answer depends only on the syndrome, not on the
        # decoder configuration, EXCEPT through tier-1 (when BP-OSD's own
        # answer is certified). Compute the certified answer once.
        pipe = []
        t0 = time.perf_counter()
        for (syn, e_true) in shots:
            cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                               return_primal=True,
                                               rpc_rounds=6)
            ans = None
            if cert is not None and primal is not None:
                if check_qldpc(checks=checks, weights=wf,
                               syndrome=list(syn),
                               cert=QldpcCertificate(
                                   error_support=tuple(primal),
                                   facet_duals=cert.facet_duals,
                                   box_duals=cert.box_duals)).accepted:
                    ans = tuple(primal)
            if ans is None:
                sup, wt, _st = exact_milp(checks, wf, list(syn),
                                          return_status=True)
                ans = tuple(int(i) for i in sup) if sup is not None else ()
            rp = np.zeros(n, dtype=np.uint8)
            rp[list(ans)] = 1
            pipe.append(rp)
        pipe_err = sum(int(is_logical((pipe[k] ^ shots[k][1]) % 2))
                       for k in range(len(shots)))
        lvl = {"pipeline_logical_errors": pipe_err,
               "pipeline_seconds": round(time.perf_counter() - t0, 1),
               "configs": {}}

        for label, cfg in CONFIGS.items():
            dec = BpOsdDecoder(csr_matrix(HZ), error_rate=p, **cfg)
            bp_err = tier1 = 0
            n00 = n01 = n10 = n11 = 0
            for k, (syn, e_true) in enumerate(shots):
                e_bp = np.asarray(dec.decode(syn.astype(np.uint8))) \
                    .astype(np.uint8)
                bw = is_logical((e_bp ^ e_true) % 2)
                pw = is_logical((pipe[k] ^ e_true) % 2)
                bp_err += int(bw)
                if bw and pw:
                    n11 += 1
                elif bw:
                    n10 += 1
                elif pw:
                    n01 += 1
                else:
                    n00 += 1
                # tier-1: is the decoder's OWN answer certified optimal?
                c2, _b2, _p2 = feldman_dual(checks, wf, list(syn),
                                            return_primal=True,
                                            rpc_rounds=6)
                if c2 is not None:
                    bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
                    if check_qldpc(checks=checks, weights=wf,
                                   syndrome=list(syn),
                                   cert=QldpcCertificate(
                                       error_support=bp_sup,
                                       facet_duals=c2.facet_duals,
                                       box_duals=c2.box_duals)).accepted:
                        tier1 += 1
            nd = n10 + n01
            kmin = min(n10, n01)
            pmc = (min(1.0, 2 * sum(math.comb(nd, i)
                                    for i in range(kmin + 1)) / 2**nd)
                   if nd else 1.0)
            lvl["configs"][label] = {
                "bp_osd_logical_errors": bp_err,
                "tier1_rate": round(100 * tier1 / len(shots), 1),
                "paired_2x2": {"both_ok": n00,
                               "bp_wrong_pipe_ok": n10,
                               "bp_ok_pipe_wrong": n01,
                               "both_wrong": n11},
                "mcnemar_two_sided_p": pmc,
                "pipeline_advantage": bp_err - pipe_err}
            print(f"  p={p}  {label:26s} BP-OSD err {bp_err:3d}  "
                  f"pipeline {pipe_err:3d}  tier1 {lvl['configs'][label]['tier1_rate']:5.1f}%  "
                  f"discordant {n10}v{n01}")
        out["levels"][f"p={p}"] = lvl

    # verdict from the data, not from hope
    strong = "strongest_ps_100_osdcs10"
    adv = {p: out["levels"][p]["configs"][strong]["pipeline_advantage"]
           for p in out["levels"]}
    out["verdict"] = {
        "advantage_vs_strongest_by_noise": adv,
        "advantage_survives_anywhere": any(v > 0 for v in adv.values()),
        "crossover": ("the noise level(s) where the strongest BP-OSD "
                      "still commits logical errors the certified "
                      "pipeline does not")}
    print(f"\nadvantage vs STRONGEST by noise: {adv}")
    dest = ROOT / "evidence" / "qldpc_strong_baseline.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
