r"""Exactify the two float-provisional circuit ablations.

The last datasets in either paper still carrying floating-point
receipts: the surface-code control (d=3, r=3, p=0.005) and the custom
sequential BB schedule (r=3, p=0.002). Both get the canonical
treatment here -- published integer DEM weights, exact rational
certificates, branch-dual closure of any residue, and the pinned
reference checker on every flat certificate -- so that NO dataset
anywhere is float-provisional.

Same circuits, same seeds, same protocols as the float runs; only the
certification path changes (floats -> integer objective + exact
arithmetic). Divergence from the float rates is itself a finding and is
reported, not smoothed.
"""
from __future__ import annotations

import hashlib
import importlib.util
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

from qldpc_bb_circuit_gate import build_circuit as build_seq_circuit
from qldpc_circuit_gate import dem_to_matrices
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,
                             check_qldpc_exact, exact_milp,
                             exactify_certificate, feldman_dual_lazy)
from oneq.qldpc_check import check_qldpc_bnb

REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
REF_SHA = "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"


def run(label, circ, shots, seed, ref, dense_ref=True):
    dem = circ.detector_error_model(decompose_errors=False)
    checks, wllr, obs_rows, nm = dem_to_matrices(dem)
    wint = {i: int(math.floor(1000 * float(wllr[i]) + 0.5))
            for i in range(nm)}
    wf = {i: float(wint[i]) for i in range(nm)}
    wx = {i: Fraction(wint[i]) for i in range(nm)}
    wblob = json.dumps([wint[k] for k in sorted(wint)]).encode()

    from ldpc import BpOsdDecoder
    from scipy.sparse import lil_matrix
    H = lil_matrix((len(checks), nm), dtype=np.uint8)
    for j, cc in enumerate(checks):
        for i in cc:
            H[j, i] = 1
    H = H.tocsr()
    dec = BpOsdDecoder(H, error_channel=[
        float(1 / (1 + np.exp(wllr[i]))) for i in range(nm)],
        max_iter=40, bp_method="ms", schedule="parallel",
        osd_method="osd_cs", osd_order=5)
    sampler = dem.compile_sampler(seed=seed)
    dets, obs, _ = sampler.sample(shots=shots, return_errors=False)
    dense = None
    if dense_ref and nm <= 2600:
        dense = [[1 if i in set(c) else 0 for i in range(nm)]
                 for c in checks]

    c = {"shots": shots, "nontrivial": 0, "trivial_tier0": 0,
         "tier1": 0, "tier2": 0, "tierBD": 0, "open": 0,
         "exact_certified": 0, "ref_checked": 0, "ref_agreed": 0,
         "obs_match": 0, "obs_checked": 0,
         "detectors": len(checks), "mechanisms": nm,
         "weights_sha256": hashlib.sha256(wblob).hexdigest(),
         "weight_rule": "c_i = floor(1000*log((1-p_i)/p_i) + 1/2)"}
    t0 = time.perf_counter()
    for s in range(dets.shape[0]):
        syn = dets[s].astype(np.uint8)
        if not syn.any():
            c["trivial_tier0"] += 1
            continue
        c["nontrivial"] += 1
        e_bp = np.asarray(dec.decode(syn)).astype(np.uint8)
        bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
        cert, bound, primal = feldman_dual_lazy(checks, wf, list(syn),
                                                return_primal=True)
        ans, tier, fc = None, None, None
        if cert is not None:
            f1 = QldpcCertificate(error_support=bp_sup,
                                  facet_duals=cert.facet_duals,
                                  box_duals=cert.box_duals)
            if check_qldpc(checks=checks, weights=wf, syndrome=list(syn),
                           cert=f1).accepted:
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
                if dense is not None:
                    e = [0] * nm
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
                         "weights": [{"num": wint[i], "den": 1}
                                     for i in range(nm)],
                         "candidate": e, "facets": facets})
                    c["ref_checked"] += 1
                    c["ref_agreed"] += int(rep["objective_optimal"])
        else:
            sup, wt, _st = exact_milp(checks, wf, list(syn),
                                      time_limit=60.0, return_status=True)
            if sup is None:
                c["open"] += 1
                continue
            U = int(sum(wint[int(i)] for i in sup))
            tree, _n = certified_bnb(checks, wf, list(syn), U,
                                     rpc_rounds=6, max_nodes=60000,
                                     use_lazy=True)
            closed = False
            if tree is not None:
                rb = check_qldpc_bnb(
                    checks=checks, weights=wx, syndrome=list(syn),
                    error_support=tuple(int(i) for i in sup), tree=tree)
                closed = bool(rb.accepted)
            if closed:
                c["tierBD"] += 1
                c["exact_certified"] += 1
                ans = sup
            else:
                c["open"] += 1
                ans = sup
        if ans is not None and obs_rows:
            c["obs_checked"] += 1
            pred = sum(1 for i in ans
                       if i in set(obs_rows.get(0, ()))) % 2
            c["obs_match"] += int(int(obs[s, 0]) == pred)
    c["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  {label}: {c['exact_certified']}/{c['nontrivial']} exact "
          f"(t1 {c['tier1']}, t2 {c['tier2']}, BD {c['tierBD']}, open "
          f"{c['open']}), ref {c['ref_agreed']}/{c['ref_checked']}, obs "
          f"{c['obs_match']}/{c['obs_checked']}, {c['seconds']}s")
    return c


def main() -> int:
    import stim
    got = hashlib.sha256(REF.read_bytes()).hexdigest()
    assert got == REF_SHA
    spec = importlib.util.spec_from_file_location("ref_checker", REF)
    ref = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = ref
    spec.loader.exec_module(ref)

    out = {"schema": "oneq-qldpc-ablation-exact/1",
           "purpose": ("exactify the two float-provisional circuit "
                       "ablations so no dataset in either paper is "
                       "float-provisional"),
           "runs": {}}

    surf = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.005,
        after_reset_flip_probability=0.005,
        before_round_data_depolarization=0.005,
        before_measure_flip_probability=0.005)
    out["runs"]["surface_d3r3_p005"] = run(
        "surface control d3r3 p=0.005", surf, 200, 20260802, ref)

    seq = build_seq_circuit(3, 0.002)
    out["runs"]["bb_sequential_r3_p002"] = run(
        "BB sequential r3 p=0.002", seq, 150, 20260802, ref)

    dest = ROOT / "evidence" / "qldpc_ablation_exact.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
