r"""CLOSE THE SOLVER TIER: branch-dual trees for every residual shot.

The last impurity in the receipt architecture was tier 3: shots whose
flat LP would not close rested on TRUSTING a MIP solver's status line.
The certified branch-and-bound retires that: each residual shot gets a
tree whose every leaf carries an exact safe-dual bound (or a named
GF(2)-infeasible row combination), checkable in exact rational
arithmetic with NO solver -- validated on 162/162 flat-LP-loose random
instances against brute force before touching a named code.

This gate re-derives the exact residual shots of the canonical rungs
(same seeds as qldpc_exact_all_gate / qldpc_bb_refsched_gate), runs
certified_bnb with the solver's answer as the candidate, and checks
every tree with check_qldpc_bnb. A closed shot moves from
SOLVER-REPORTED (3A) to EXACT-CERTIFIED BY BRANCH DUALS -- the same
trust level as tier 1. An unclosed shot stays 3A, honestly.

Residues targeted (from the canonical artifacts):
  uniform_p02  1 shot     hetero_p02  1 shot
  stress_p05   17 shots   refsched    4 shots (lazy path, degree 35)
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

from qldpc_bb_gate import bb_gross_code  # noqa: E402
from qldpc_bb_refsched_gate import build_refsched_circuit  # noqa: E402
from qldpc_circuit_gate import dem_to_matrices  # noqa: E402
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb,  # noqa: E402
                             check_qldpc, exact_milp, feldman_dual,
                             feldman_dual_lazy, fixed_point_weights)
from oneq.qldpc_check import check_qldpc_bnb  # noqa: E402


def find_residues_code_capacity(HZ, checks, wint, dec, rng_seed, p_arr,
                                target):
    """Replay a canonical rung, returning its residual shots."""
    n = HZ.shape[1]
    wf = {i: float(wint[i]) for i in range(n)}
    rng = np.random.default_rng(rng_seed)
    residues = []
    nontrivial = 0
    while nontrivial < target:
        e = (rng.random(n) < p_arr).astype(np.uint8)
        syn = (HZ @ e) % 2
        if not syn.any():
            continue
        nontrivial += 1
        e_hat = np.asarray(dec.decode(syn.astype(np.uint8))).astype(np.uint8)
        cert, bound, primal = feldman_dual(checks, wf, list(syn),
                                           return_primal=True, rpc_rounds=6)
        ok = False
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_hat))
            if check_qldpc(checks=checks, weights=wf, syndrome=list(syn),
                           cert=QldpcCertificate(
                               error_support=bp_sup,
                               facet_duals=cert.facet_duals,
                               box_duals=cert.box_duals)).accepted:
                ok = True
            elif primal is not None and check_qldpc(
                    checks=checks, weights=wf, syndrome=list(syn),
                    cert=QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals)).accepted:
                ok = True
        if not ok:
            residues.append((nontrivial - 1, syn.copy()))
    return residues


def close_residues(name, checks, wint, residues, out, *, use_lazy=False,
                   max_nodes=20000):
    n = len(wint)
    wf = {i: float(wint[i]) for i in range(n)}
    wx = {i: Fraction(int(wint[i])) for i in range(n)}
    rows = []
    for (shot_i, syn) in residues:
        sup, wt, st = exact_milp(checks, wf, list(syn), time_limit=120.0,
                                 return_status=True)
        assert sup is not None, f"{name} shot {shot_i}: MILP failed"
        U = int(round(wt))
        t0 = time.perf_counter()
        tree, nodes = certified_bnb(checks, wf, list(syn), U,
                                    rpc_rounds=6, max_nodes=max_nodes,
                                    max_depth=80, use_lazy=use_lazy)
        dt = time.perf_counter() - t0
        row = {"shot_index": shot_i, "candidate_weight": U,
               "solver_status": st["tier"], "bnb_nodes": nodes,
               "bnb_seconds": round(dt, 2)}
        if tree is None:
            row["closed"] = False
        else:
            r = check_qldpc_bnb(checks=checks, weights=wx,
                                syndrome=list(syn),
                                error_support=tuple(int(i) for i in sup),
                                tree=tree)
            row["closed"] = bool(r.accepted)
            if not r.accepted:
                row["checker_reason"] = r.reason[:100]
        rows.append(row)
        print(f"  {name} shot {shot_i}: U={U} nodes={nodes} "
              f"{'CLOSED-EXACT' if row['closed'] else 'still 3A'} "
              f"({dt:.1f}s)")
    out["rungs"][name] = {
        "residues": len(rows),
        "closed_exact_by_branch_duals": sum(r["closed"] for r in rows),
        "shots": rows}


def main() -> int:
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix, lil_matrix

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks_z = [tuple(int(i) for i in np.flatnonzero(HZ[j]))
                for j in range(m)]
    out = {"schema": "oneq-qldpc-bnb-residue/1", "rungs": {},
           "validation": "162/162 flat-LP-loose random instances closed "
                         "vs brute force; heavier candidates refused"}
    wint_u = {i: 1 for i in range(n)}

    # --- uniform p=0.02 (canonical seed 2107, 1 residue) ---
    dec = BpOsdDecoder(csr_matrix(HZ), error_rate=0.02, max_iter=30,
                       bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)
    res_u = find_residues_code_capacity(
        HZ, checks_z, wint_u, dec, 2107, np.full(n, 0.02), 470)
    print(f"uniform_p02: {len(res_u)} residues replayed")
    close_residues("uniform_p02", checks_z, wint_u, res_u, out)

    # --- hetero p=0.02 (channel seed 4242, shots seed 2101, 1 residue) ---
    rngw = np.random.default_rng(4242)
    p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
    llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
    wint_h, _spec = fixed_point_weights(llr, scale=1000)
    dec_h = BpOsdDecoder(csr_matrix(HZ),
                         error_channel=[float(v) for v in p_i],
                         max_iter=30, bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    res_h = find_residues_code_capacity(
        HZ, checks_z, wint_h, dec_h, 2101, p_i, 300)
    print(f"hetero_p02: {len(res_h)} residues replayed")
    close_residues("hetero_p02", checks_z, wint_h, res_h, out)

    # --- stress p=0.05 (seed 2104, 17 residues) ---
    dec_s = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    res_s = find_residues_code_capacity(
        HZ, checks_z, wint_u, dec_s, 2104, np.full(n, 0.05), 300)
    print(f"stress_p05: {len(res_s)} residues replayed")
    close_residues("stress_p05", checks_z, wint_u, res_s, out)

    # --- reference schedule (seed 20260804, 4 residues; lazy path) ---
    circ = build_refsched_circuit(3, 0.002)
    dem = circ.detector_error_model(decompose_errors=False)
    checks_c, wllr, obs_rows, n_mech = dem_to_matrices(dem)
    import math
    wint_c = {i: int(math.floor(1000 * float(wllr[i]) + 0.5))
              for i in range(n_mech)}
    wf_c = {i: float(wint_c[i]) for i in range(n_mech)}
    Hc = lil_matrix((len(checks_c), n_mech), dtype=np.uint8)
    for j, cc in enumerate(checks_c):
        for i in cc:
            Hc[j, i] = 1
    Hc = Hc.tocsr()
    dec_c = BpOsdDecoder(Hc, error_channel=[
        float(1 / (1 + np.exp(wllr[i]))) for i in range(n_mech)],
        max_iter=40, bp_method="ms", schedule="parallel",
        osd_method="osd_cs", osd_order=5)
    sampler = dem.compile_sampler(seed=20260804)
    dets, obs, _ = sampler.sample(shots=150, return_errors=False)
    res_c = []
    idx = -1
    for si in range(dets.shape[0]):
        syn = dets[si].astype(np.uint8)
        if not syn.any():
            continue
        idx += 1
        e_hat = np.asarray(dec_c.decode(syn)).astype(np.uint8)
        cert, bound, primal = feldman_dual_lazy(checks_c, wf_c, list(syn),
                                                return_primal=True)
        ok = False
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_hat))
            if check_qldpc(checks=checks_c, weights=wf_c,
                           syndrome=list(syn),
                           cert=QldpcCertificate(
                               error_support=bp_sup,
                               facet_duals=cert.facet_duals,
                               box_duals=cert.box_duals)).accepted:
                ok = True
            elif primal is not None and check_qldpc(
                    checks=checks_c, weights=wf_c, syndrome=list(syn),
                    cert=QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals)).accepted:
                ok = True
        if not ok:
            res_c.append((idx, syn.copy()))
    print(f"bb_refsched: {len(res_c)} residues replayed")
    close_residues("bb_refsched", checks_c, wint_c, res_c, out,
                   use_lazy=True, max_nodes=60000)

    total = sum(r["residues"] for r in out["rungs"].values())
    closed = sum(r["closed_exact_by_branch_duals"]
                 for r in out["rungs"].values())
    out["total_residues"] = total
    out["total_closed_exact"] = closed
    dest = ROOT / "evidence" / "qldpc_bnb_residue.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"TOTAL: {closed}/{total} residual shots promoted to "
          f"EXACT-BY-BRANCH-DUALS; evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
