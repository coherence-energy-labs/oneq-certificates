r"""Quantization robustness of the fixed-point objective (audit round 4).

The certified specification is the published integer vector; this probe
measures whether the certification RESULT depends on the quantization
scale: the heterogeneous rung re-run at S = 10^3 (canonical), 10^4, and
10^5 on the SAME syndromes (same seeds as the canonical gate), comparing
per-shot receipt tiers, certified counts, and selected supports.

If tiers and supports agree across scales, the canonical result is
robust to quantization; any disagreement is reported per shot, not
averaged away.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code
from oneq.qldpc_cert import (QldpcCertificate, check_qldpc, exact_milp,
                             feldman_dual, fixed_point_weights)


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]

    rngw = np.random.default_rng(4242)          # same channel as canonical
    p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
    llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    dec = BpOsdDecoder(csr_matrix(HZ), error_channel=[float(v) for v in p_i],
                       max_iter=30, bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)

    rng = np.random.default_rng(2101)           # same shots as canonical
    syndromes = []
    bp_answers = []
    while len(syndromes) < 300:
        e = (rng.random(n) < p_i).astype(np.uint8)
        syn = (HZ @ e) % 2
        if not syn.any():
            continue
        syndromes.append(syn)
        bp_answers.append(np.asarray(dec.decode(syn.astype(np.uint8)))
                          .astype(np.uint8))

    results = {}
    for scale in (1000, 10000, 100000):
        wint, spec = fixed_point_weights(llr, scale=scale)
        wf = {i: float(wint[i]) for i in range(n)}
        tiers = []
        supports = []
        for syn, e_bp in zip(syndromes, bp_answers):
            cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                               return_primal=True,
                                               rpc_rounds=6)
            tier, sup = None, None
            if cert is not None:
                bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
                f1 = QldpcCertificate(error_support=bp_sup,
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)
                if check_qldpc(checks=checks, weights=wf,
                               syndrome=list(syn), cert=f1).accepted:
                    tier, sup = "t1", bp_sup
                elif primal is not None:
                    f2 = QldpcCertificate(error_support=tuple(primal),
                                          facet_duals=cert.facet_duals,
                                          box_duals=cert.box_duals)
                    if check_qldpc(checks=checks, weights=wf,
                                   syndrome=list(syn), cert=f2).accepted:
                        tier, sup = "t2", tuple(primal)
            if tier is None:
                s3, _w, st = exact_milp(checks, wf, list(syn),
                                        return_status=True)
                tier, sup = st["tier"][:2].replace("3A", "3A"), s3
                tier = "3A" if st["tier"].startswith("3A") else "3B"
            tiers.append(tier)
            supports.append(sup)
        results[scale] = {"tiers": tiers, "supports": supports,
                          "weights_sha256": spec["weights_sha256"],
                          "certified": sum(t in ("t1", "t2")
                                           for t in tiers)}

    base = results[1000]
    out = {"schema": "oneq-qldpc-scale-robustness/1",
           "rung": "hetero_p02 (same channel seed 4242, shots seed 2101)",
           "scales": {}, "tier_change_detail": []}
    for scale in (1000, 10000, 100000):
        r = results[scale]
        tier_diffs = sum(a != b for a, b in zip(base["tiers"], r["tiers"]))
        sup_diffs = sum(a != b for a, b in
                        zip(base["supports"], r["supports"]))
        out["scales"][str(scale)] = {
            "certified": r["certified"],
            "weights_sha256": r["weights_sha256"],
            "tier_disagreements_vs_S1000": tier_diffs,
            "support_disagreements_vs_S1000": sup_diffs}
        # every tier change explained per shot: which, old->new, and
        # whether the selected correction moved (audit round 5)
        for k, (a, b) in enumerate(zip(base["tiers"], r["tiers"])):
            if a != b:
                out["tier_change_detail"].append({
                    "scale": scale, "shot_index": k,
                    "tier_at_S1000": a, "tier_at_this_scale": b,
                    "selected_support_changed":
                        base["supports"][k] != r["supports"][k],
                    "support_size": len(r["supports"][k] or ())})
        print(f"S={scale}: certified {r['certified']}/300, "
              f"tier diffs vs 1e3: {tier_diffs}, "
              f"support diffs: {sup_diffs}")

    dest = ROOT / "evidence" / "qldpc_scale_robustness.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
