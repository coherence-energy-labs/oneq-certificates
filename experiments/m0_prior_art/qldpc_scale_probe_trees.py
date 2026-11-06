r"""The fixed-point scale probe, re-run with EXACT receipts and branch-dual closure at every scale.

    python experiments/m0_prior_art/qldpc_scale_probe_trees.py

WHY (external audit 1 of release v2026.11.03; adjudicated 2026-11-03, finding P2-06). Paper 2 says changing the
weight scale through 10^3, 10^4 and 10^5 "certified 299 / 299 / 299 shots with 0 changes in any selected correction"
(evidence/qldpc_scale_robustness.json, from qldpc_scale_probe.py). That probe ran on the DEVELOPMENT heterogeneous
rung (channel seed 4242, shots seed 2101, 300 shots), counted only the flat tiers, and left the 300th shot -- index 71
-- to the MILP's status: tier 3A at 10^3 and 3B at 10^5, a tier change the paper never printed. That shot has a
stored 25-node branch-dual tree at 10^3 (evidence/qldpc_heldout_addendum.json, dev_bd_trees, hetero_p02 #71).

THIS RUN: the same channel, the same 300 syndromes and the same BP-OSD (ldpc 2.4.1 defaults as in the probe), at
each scale S: integer weights c_i = floor(S * llr_i + 1/2); the flat route exactly as the campaign ran it (screen,
then exactify, then check_qldpc_exact on the integer weights); and every shot the flat route leaves open closed by
certified_bnb on the MILP's candidate and checked by check_qldpc_bnb on the integer weights. Every count below is an
exact checker's verdict; a shot no receipt closes is reported as unresolved, never as certified. Reported per scale:
certified shots, by route, and per shot whether the selected correction is the same as at S = 10^3.
"""
from __future__ import annotations

import hashlib
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
OUT = ROOT / "evidence" / "qldpc_scale_robustness_trees.json"
SCALES = (1000, 10000, 100000)


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    import ldpc
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    from qldpc_bb_gate import bb_gross_code
    from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc, exact_milp, exactify_certificate,
                                 feldman_dual, fixed_point_weights)
    from oneq.qldpc_check import check_qldpc_bnb, check_qldpc_exact

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    rngw = np.random.default_rng(4242)                     # the probe's channel
    p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
    llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
    dec = BpOsdDecoder(csr_matrix(HZ), error_channel=[float(v) for v in p_i], max_iter=30, bp_method="ms",
                       schedule="parallel", osd_method="osd_cs", osd_order=5)
    rng = np.random.default_rng(2101)                      # the probe's shots
    syndromes, bp = [], []
    while len(syndromes) < 300:
        e = (rng.random(n) < p_i).astype(np.uint8)
        syn = (HZ @ e) % 2
        if not syn.any():
            continue
        syndromes.append([int(b) for b in syn])
        bp.append(tuple(int(i) for i in np.flatnonzero(np.asarray(dec.decode(syn.astype(np.uint8))))))

    t0 = time.perf_counter()
    res = {}
    for S in SCALES:
        wint, spec = fixed_point_weights(llr, scale=S)
        wf = {i: float(wint[i]) for i in range(n)}
        wx = {i: Fraction(int(wint[i])) for i in range(n)}
        routes, supports, objective, nodes = [], [], [], {}
        for k, syn in enumerate(syndromes):
            cert, _bound, primal = feldman_dual(checks, wf, syn, return_primal=True, rpc_rounds=6)
            route, sup = None, None
            if cert is not None:
                for cand, name in ((bp[k], "flat_bposd"), (tuple(primal) if primal is not None else None, "flat_primal")):
                    if cand is None:
                        continue
                    fc = QldpcCertificate(error_support=cand, facet_duals=cert.facet_duals, box_duals=cert.box_duals)
                    if check_qldpc(checks=checks, weights=wf, syndrome=syn, cert=fc).accepted:
                        if check_qldpc_exact(checks=checks, weights=wx, syndrome=syn,
                                             cert=exactify_certificate(fc)).accepted:
                            route, sup = name, cand
                        break
            if route is None:
                ms, _w, _st = exact_milp(checks, wf, syn, return_status=True)
                if ms is not None:
                    U = int(sum(wint[int(i)] for i in ms))
                    tree, nn = certified_bnb(checks, wf, syn, U, rpc_rounds=6, max_nodes=60000, use_lazy=True)
                    if tree is not None and check_qldpc_bnb(checks=checks, weights=wx, syndrome=syn,
                                                            error_support=tuple(int(i) for i in ms),
                                                            tree=tree).accepted:
                        route, sup = "tree", tuple(int(i) for i in ms)
                        nodes[k] = nn
            routes.append(route or "unresolved")
            supports.append(sorted(sup) if sup is not None else None)
            objective.append(int(sum(wint[int(i)] for i in sup)) if sup is not None else None)
        res[S] = {"routes": routes, "supports": supports, "objective": objective, "tree_nodes": nodes,
                  "weights_sha256": spec["weights_sha256"]}
        print(f"S={S}: certified {sum(r != 'unresolved' for r in routes)}/300 "
              f"({ {x: routes.count(x) for x in sorted(set(routes))} }), trees on shots {sorted(nodes)}", flush=True)

    base = res[SCALES[0]]
    out = {"schema": "oneq-qldpc-scale-robustness-trees/1",
           "dataset": ("DEVELOPMENT heterogeneous rung: H_Z of the [[144,12,12]] code, p_i = 0.02 * U(0.5, 2) from "
                       "seed 4242, 300 nontrivial syndromes from seed 2101 -- the same instance and shots as "
                       "qldpc_scale_probe.py, not the held-out heterogeneous rung"),
           "decoder": "ldpc BpOsdDecoder, error_channel p_i, 30 iterations, min-sum (scaling 1.0), parallel, OSD-CS 5",
           "ldpc": ldpc.__version__, "scales": {}}
    for S in SCALES:
        r = res[S]
        out["scales"][str(S)] = {
            "certified_exact": sum(x != "unresolved" for x in r["routes"]),
            "by_route": {x: r["routes"].count(x) for x in sorted(set(r["routes"]))},
            "tree_shots": {str(k): v for k, v in sorted(r["tree_nodes"].items())},
            "route_changes_vs_1000": sum(a != b for a, b in zip(base["routes"], r["routes"])),
            "selected_correction_changes_vs_1000": sum(a != b for a, b in zip(base["supports"], r["supports"])),
            "weights_sha256": r["weights_sha256"]}
    out["route_change_detail"] = [{"scale": S, "shot_index": k, "route_at_1000": base["routes"][k],
                                   "route_here": res[S]["routes"][k],
                                   "selected_correction_changed": base["supports"][k] != res[S]["supports"][k]}
                                  for S in SCALES[1:] for k in range(300) if base["routes"][k] != res[S]["routes"][k]]
    out["all_certified_at_every_scale"] = all(out["scales"][str(S)]["certified_exact"] == 300 for S in SCALES)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    me = pathlib.Path(__file__).resolve()
    out["code_sha256"] = {me.relative_to(ROOT).as_posix(): sha(me), "src/oneq/qldpc_cert.py":
                          sha(ROOT / "src" / "oneq" / "qldpc_cert.py"), "src/oneq/qldpc_check.py":
                          sha(ROOT / "src" / "oneq" / "qldpc_check.py")}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({S: out["scales"][S] for S in out["scales"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
