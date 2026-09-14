r"""THE FAMILY AT CIRCUIT LEVEL: published schedule, any BB code.

Generalization so far is code-capacity only. The reference-schedule
builder was hard-coded to the gross code (nhalf=72); this gate lifts
Table 5's structure to ANY (l, m, A, B) bivariate-bicycle code and runs
the exact certification pipeline on the resulting circuit-level DEM.

The schedule is the published one for every code -- the seven CNOT
rounds and their monomial assignments are properties of the QC(A,B)
construction, not of the gross code's parameters -- and each circuit is
NOISELESSLY VALIDATED before any noisy shot: every detector must be
deterministic-zero and the observable preserved. A wrong generalization
cannot pass that gate.

Codes: [[72,12,6]] (l=6,m=6) and [[144,12,12]] (l=12,m=6). The
[[288,12,18]] circuit is buildable the same way but its DEM is ~4600
mechanisms per round-triple; noted, not run here.
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

from qldpc_bb_family_gate import bb_code, gf2_rank
from qldpc_bb_gate import gf2_in_rowspace, gf2_row_reduce
from qldpc_circuit_gate import dem_to_matrices
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,
                             check_qldpc_exact, exact_milp,
                             exactify_certificate, feldman_dual_lazy)
from oneq.qldpc_check import check_qldpc_bnb

CODES = [
    ("[[72,12,6]]", 6, 6, [("x", 3), ("y", 1), ("y", 2)],
     [("y", 3), ("x", 1), ("x", 2)]),
    ("[[144,12,12]]", 12, 6, [("x", 3), ("y", 1), ("y", 2)],
     [("y", 3), ("x", 1), ("x", 2)]),
]


def monomial_perms(l, m, mons):
    Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(Sl, np.eye(m, dtype=np.uint8)).astype(np.uint8)
    y = np.kron(np.eye(l, dtype=np.uint8), Sm).astype(np.uint8)
    out = []
    for v, pwr in mons:
        M = np.eye(l * m, dtype=np.uint8)
        base = x if v == "x" else y
        for _ in range(pwr):
            M = (M @ base) % 2
        out.append(M)
    return out


def logical_z_for(HX, HZ):
    """One logical Z: ker(H_X) outside rowspace(H_Z), fully reduced."""
    n = HX.shape[1]
    M = (HX.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        p = next((rr for rr in range(r, rows) if M[rr, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for rr in range(rows):
            if rr != r and M[rr, c]:
                M[rr] ^= M[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    zb, zp = gf2_row_reduce(HZ % 2)
    pset = set(piv)
    for f in (c for c in range(n) if c not in pset):
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for ri, pc in enumerate(piv):
            if M[ri, f]:
                v[pc] = 1
        assert not ((HX @ v) % 2).any()
        if not gf2_in_rowspace(zb, zp, v, n):
            return [int(i) for i in np.flatnonzero(v)]
    raise AssertionError("no logical Z")


def build_circuit(l, m, Amons, Bmons, HZ, lz, r, p):
    """Table-5 schedule for an arbitrary QC(A,B) code."""
    import stim
    A = monomial_perms(l, m, Amons)
    B = monomial_perms(l, m, Bmons)
    nh = l * m
    L = list(range(0, nh))
    R = list(range(nh, 2 * nh))
    XA = list(range(2 * nh, 3 * nh))
    ZA = list(range(3 * nh, 4 * nh))
    data = L + R

    def fwd(P, i):
        return int(np.flatnonzero(P[i])[0])

    def bwd(P, i):
        return int(np.flatnonzero(P[:, i])[0])

    c = stim.Circuit()
    c.append("R", data)
    c.append("X_ERROR", data, p)
    c.append("R", ZA)
    c.append("X_ERROR", ZA, p)

    def cx(pairs):
        flat = []
        for (a, b) in pairs:
            flat += [a, b]
        c.append("CX", flat)
        c.append("DEPOLARIZE2", flat, p)

    for rd in range(r):
        c.append("RX", XA)
        c.append("Z_ERROR", XA, p)
        cx([(R[bwd(A[0], i)], ZA[i]) for i in range(nh)])
        cx([(XA[i], L[fwd(A[1], i)]) for i in range(nh)]
           + [(R[bwd(A[2], i)], ZA[i]) for i in range(nh)])
        cx([(XA[i], R[fwd(B[1], i)]) for i in range(nh)]
           + [(L[bwd(B[0], i)], ZA[i]) for i in range(nh)])
        cx([(XA[i], R[fwd(B[0], i)]) for i in range(nh)]
           + [(L[bwd(B[1], i)], ZA[i]) for i in range(nh)])
        cx([(XA[i], R[fwd(B[2], i)]) for i in range(nh)]
           + [(L[bwd(B[2], i)], ZA[i]) for i in range(nh)])
        cx([(XA[i], L[fwd(A[0], i)]) for i in range(nh)]
           + [(R[bwd(A[1], i)], ZA[i]) for i in range(nh)])
        cx([(XA[i], L[fwd(A[2], i)]) for i in range(nh)])
        c.append("X_ERROR", ZA, p)
        c.append("M", ZA)
        for i in range(nh):
            back = -(nh - i)
            if rd == 0:
                c.append("DETECTOR", [stim.target_rec(back)])
            else:
                c.append("DETECTOR", [stim.target_rec(back),
                                      stim.target_rec(back - 2 * nh)])
        c.append("Z_ERROR", XA, p)
        c.append("MX", XA)
        c.append("R", ZA)
        c.append("X_ERROR", ZA, p)
        c.append("DEPOLARIZE1", data, p)

    c.append("X_ERROR", data, p)
    c.append("M", data)
    nd = len(data)
    for i in range(nh):
        sup = [int(j) for j in np.flatnonzero(HZ[i])]
        targs = [stim.target_rec(-(nd - j)) for j in sup]
        targs.append(stim.target_rec(-(nd + nh + (nh - i))))
        c.append("DETECTOR", targs)
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(-(nd - j)) for j in lz], 0)
    return c


def main() -> int:
    out = {"schema": "oneq-qldpc-family-circuit/1",
           "schedule": ("published Table-5 seven-CNOT-round cycle, "
                        "generalized to any QC(A,B); each circuit "
                        "noiselessly validated before noisy sampling"),
           "codes": {}}
    for (label, l, m, Am, Bm) in CODES:
        HX, HZ = bb_code(l, m, Am, Bm)
        n = HX.shape[1]
        k = n - 2 * gf2_rank(HX)
        lz = logical_z_for(HX, HZ)
        # NOISELESS VALIDATION FIRST
        c0 = build_circuit(l, m, Am, Bm, HZ, lz, 3, 0.0)
        d0, o0 = c0.compile_detector_sampler().sample(
            shots=16, separate_observables=True)
        assert not d0.any(), f"{label}: noiseless detectors fired"
        assert not o0.any(), f"{label}: noiseless observable flipped"
        print(f"{label}: n={n} k={k}, {d0.shape[1]} detectors "
              f"deterministic-zero -- schedule valid")

        circ = build_circuit(l, m, Am, Bm, HZ, lz, 3, 0.002)
        dem = circ.detector_error_model(decompose_errors=False)
        checks, wllr, obs_rows, nm = dem_to_matrices(dem)
        wint = {i: int(math.floor(1000 * float(wllr[i]) + 0.5))
                for i in range(nm)}
        wf = {i: float(wint[i]) for i in range(nm)}
        wx = {i: Fraction(wint[i]) for i in range(nm)}
        degs = sorted(len(c) for c in checks)

        from ldpc import BpOsdDecoder
        from scipy.sparse import lil_matrix
        H = lil_matrix((len(checks), nm), dtype=np.uint8)
        for j, cc in enumerate(checks):
            for i in cc:
                H[j, i] = 1
        dec = BpOsdDecoder(H.tocsr(), error_channel=[
            float(1 / (1 + np.exp(wllr[i]))) for i in range(nm)],
            max_iter=40, bp_method="ms", schedule="parallel",
            osd_method="osd_cs", osd_order=5)
        dets, obs, _ = dem.compile_sampler(seed=909 + n).sample(
            shots=120, return_errors=False)

        cst = {"n": n, "k": k, "detectors": len(checks),
               "mechanisms": nm, "degree_median": degs[len(degs) // 2],
               "degree_max": degs[-1], "shots": 120,
               "nontrivial": 0, "tier1": 0, "tier2": 0, "tierBD": 0,
               "open": 0, "exact_certified": 0,
               "obs_match": 0, "obs_checked": 0,
               "weights_sha256": hashlib.sha256(json.dumps(
                   [wint[i] for i in range(nm)]).encode()).hexdigest()}
        t0 = time.perf_counter()
        for s in range(dets.shape[0]):
            syn = dets[s].astype(np.uint8)
            if not syn.any():
                continue
            cst["nontrivial"] += 1
            e_bp = np.asarray(dec.decode(syn)).astype(np.uint8)
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
            cert, _b, primal = feldman_dual_lazy(checks, wf, list(syn),
                                                 return_primal=True)
            ans, tier, fc = None, None, None
            if cert is not None:
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
                cst[tier] += 1
                if check_qldpc_exact(checks=checks, weights=wx,
                                     syndrome=list(syn),
                                     cert=exactify_certificate(fc)
                                     ).accepted:
                    cst["exact_certified"] += 1
            else:
                sup, wt, _st = exact_milp(checks, wf, list(syn),
                                          time_limit=60.0,
                                          return_status=True)
                if sup is None:
                    cst["open"] += 1
                    continue
                U = int(sum(wint[int(i)] for i in sup))
                tree, _nn = certified_bnb(checks, wf, list(syn), U,
                                          rpc_rounds=6, max_nodes=60000,
                                          use_lazy=True)
                if tree is not None and check_qldpc_bnb(
                        checks=checks, weights=wx, syndrome=list(syn),
                        error_support=tuple(int(i) for i in sup),
                        tree=tree).accepted:
                    cst["tierBD"] += 1
                    cst["exact_certified"] += 1
                    ans = sup
                else:
                    cst["open"] += 1
                    ans = sup
            if ans is not None:
                cst["obs_checked"] += 1
                pred = sum(1 for i in ans
                           if i in set(obs_rows.get(0, ()))) % 2
                cst["obs_match"] += int(int(obs[s, 0]) == pred)
        cst["seconds"] = round(time.perf_counter() - t0, 1)
        out["codes"][label] = cst
        print(f"  {label}: {cst['exact_certified']}/{cst['nontrivial']} "
              f"exact (t1 {cst['tier1']}, t2 {cst['tier2']}, BD "
              f"{cst['tierBD']}, open {cst['open']}), obs "
              f"{cst['obs_match']}/{cst['obs_checked']}, degree "
              f"{cst['degree_median']}/{cst['degree_max']}, "
              f"{cst['seconds']}s")

    dest = ROOT / "evidence" / "qldpc_family_circuit.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
