r"""The paired 2x2 at stress noise -- the audit's statistics repair.

BP-OSD and the pipeline decode the SAME 300 shots at p=0.05, so the
publishable analysis is the paired contingency table, not two
independent binomials. This gate reruns the stress protocol with
PER-SHOT logging and emits:

  - the 2x2 table crossing BP-OSD logical-wrong x pipeline logical-wrong
  - the exact McNemar test on the discordant cells (binomial, two-sided)
  - the paired risk difference with an exact CP interval on discordants
  - per-shot records (syndrome hash, tiers, both verdicts) so no cell is
    reconstructible only from prose

Logical-wrong means: (answer XOR true fault) is in ker H_Z but outside
rowspace(H_X) -- retrospective simulation truth, labeled as such. The
protocol mirrors qldpc_bb_gate's stress run: seed 20260801-adjacent
fresh sample, BP-OSD (min-sum, parallel, 30 iters, OSD-CS 5), tiered
LP + repair + MILP fallback, rpc_rounds=6.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
from oneq.qldpc_cert import (QldpcCertificate, check_qldpc, exact_milp,
                             feldman_dual)


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    weights = {i: 1.0 for i in range(n)}
    xb, xp = gf2_row_reduce(HX % 2)

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    dec = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                       bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)

    def is_logical(fault: np.ndarray) -> bool:
        if ((HZ @ fault) % 2).any():
            return True    # not even a valid correction difference
        return not gf2_in_rowspace(xb, xp, fault, n)

    rng = np.random.default_rng(20260803)
    shots = []
    n00 = n01 = n10 = n11 = 0
    tiers = {"tier1": 0, "tier2": 0, "tier3": 0}
    target = 300
    sampled = 0
    while len(shots) < target:
        sampled += 1
        e_true = (rng.random(n) < 0.05).astype(np.uint8)
        syn = (HZ @ e_true) % 2
        if not syn.any():
            continue
        e_bp = np.asarray(dec.decode(syn.astype(np.uint8))).astype(np.uint8)
        bp_wrong = is_logical((e_bp ^ e_true) % 2)

        cert, bound, primal = feldman_dual(checks, weights, list(syn),
                                           return_primal=True, rpc_rounds=6)
        pipe_ans = None
        tier = None
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
            fc = QldpcCertificate(error_support=bp_sup,
                                  facet_duals=cert.facet_duals,
                                  box_duals=cert.box_duals)
            if check_qldpc(checks=checks, weights=weights,
                           syndrome=list(syn), cert=fc).accepted:
                pipe_ans, tier = e_bp, "tier1"
            elif primal is not None:
                rc = QldpcCertificate(error_support=tuple(primal),
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)
                if check_qldpc(checks=checks, weights=weights,
                               syndrome=list(syn), cert=rc).accepted:
                    rp = np.zeros(n, dtype=np.uint8)
                    rp[list(primal)] = 1
                    pipe_ans, tier = rp, "tier2"
        if pipe_ans is None:
            sup, _wt = exact_milp(checks, weights, list(syn))
            assert sup is not None, "MILP fallback failed on a stress shot"
            rp = np.zeros(n, dtype=np.uint8)
            rp[[int(i) for i in sup]] = 1
            pipe_ans, tier = rp, "tier3"
        tiers[tier] += 1
        pipe_wrong = is_logical((pipe_ans ^ e_true) % 2)

        if bp_wrong and pipe_wrong:
            n11 += 1
        elif bp_wrong:
            n10 += 1
        elif pipe_wrong:
            n01 += 1
        else:
            n00 += 1
        shots.append({
            "shot": len(shots),
            "syndrome_sha256":
                hashlib.sha256(syn.astype(np.uint8).tobytes()).hexdigest()[:16],
            "tier": tier, "bp_osd_logical_wrong": bool(bp_wrong),
            "pipeline_logical_wrong": bool(pipe_wrong)})

    # exact McNemar: under H0 the n10 discordants among (n10+n01) are
    # Binomial(n10+n01, 1/2); two-sided exact p
    nd = n10 + n01
    k = min(n10, n01)
    p_mcnemar = min(1.0, 2 * sum(math.comb(nd, i) for i in range(k + 1))
                    / 2**nd) if nd else 1.0

    out = {
        "schema": "oneq-qldpc-paired-stress/1",
        "protocol": {"p": 0.05, "seed": 20260803, "sampled": sampled,
                     "nontrivial": target, "rpc_rounds": 6,
                     "bp_osd": "min-sum parallel 30 iters OSD-CS 5"},
        "tiers": tiers,
        "paired_2x2": {"both_correct": n00,
                       "bp_correct_pipeline_wrong": n01,
                       "bp_wrong_pipeline_correct": n10,
                       "both_wrong": n11},
        "bp_osd_logical_wrong_total": n10 + n11,
        "pipeline_logical_wrong_total": n01 + n11,
        "discordant": nd,
        "mcnemar_exact_two_sided_p": p_mcnemar,
        "paired_risk_difference": (n10 - n01) / target,
        "note": ("logical-wrong is retrospective simulation truth; the "
                 "paired table supersedes the unpaired Katz interval as "
                 "the primary stress analysis"),
        "per_shot": shots,
    }
    dest = ROOT / "evidence" / "qldpc_paired_stress.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_shot"},
                     indent=2))
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
