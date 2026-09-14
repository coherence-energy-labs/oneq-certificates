r"""Certify BP-OSD on [[144,12,12]] under PHENOMENOLOGICAL noise -- with
measurement errors, the regime one honest step below circuit-level.

WHY THIS EXISTS. The code-capacity result (qldpc_bb_gate.py: 100% certified
at operating noise, 38x logical-error reduction at p=0.05) assumes perfect
syndrome extraction. Real machines measure noisily, and r rounds of noisy
measurement turn decoding into a SPACE-TIME problem. The certificate
machinery transfers UNCHANGED because the space-time system is just a bigger
sparse GF(2) code:

    detectors   D_{j,t} = s_{j,t} XOR s_{j,t-1}   (t = 1..r, s_0 = 0)
    mechanisms  e_t  = NEW data errors in round t   (n vars each)
                m_t  = measurement errors in round t (one per check)
    relation    D_{j,t} = (H e_t)_j + m_{j,t} + m_{j,t-1}

so the space-time check matrix has row degree deg(H_j) + 2 = 8: Feldman
facets 2^7 = 128 per check, enumerable, and check_qldpc/feldman_dual/
exact_milp run as-is on (checks, weights, syndrome). A final perfect round
(m_r = 0) closes the boundary, the standard convention.

Weights are log-likelihood: w = log((1-p)/p) per mechanism with its own p
(data vs measurement), so min-weight = most-likely single-fault-set under
independence, the same conformance target as everywhere else in this
program.

HONEST SCOPE: phenomenological, not circuit-level -- no correlated two-qubit
fault mechanisms, no hook errors. It is the standard intermediate regime and
is labeled as such everywhere this number is reported. Logical-fault
attribution compares the ACCUMULATED data error against truth in the final
frame, stabilizer membership decided by GF(2) rowspace elimination as
before.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce  # noqa: E402

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,  # noqa: E402
                             exact_milp, feldman_dual)


def spacetime_system(HZ, r, p_data, p_meas):
    """checks, weights, and index maps for r rounds + final perfect round."""
    m, n = HZ.shape
    checks = []
    weights = {}
    w_d = float(np.log((1 - p_data) / p_data))
    w_m = float(np.log((1 - p_meas) / p_meas))
    # variable ids: data error e_{t,i} -> t*n + i  (t = 0..r-1)
    #               meas error m_{t,j} -> r*n + t*m + j  (t = 0..r-1; round r
    #               is the final PERFECT round, so no m vars for it)
    for t in range(r):
        for i in range(n):
            weights[t * n + i] = w_d
        for j in range(m):
            weights[r * n + t * m + j] = w_m
    rows_H = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    # s_{j,t} = H(e_0+..+e_t) + m_{j,t}, so the detector difference telescopes
    # to NEW-error variables only: D_{j,t} = H e_t + m_{j,t} + m_{j,t-1}
    # (m_{-1} = 0). The final round r is measured perfectly on the same
    # accumulated error as round r-1, so D_{j,r} = m_{j,r-1} alone.
    for t in range(r + 1):          # detector rounds 0..r (r = final perfect)
        for j in range(m):
            sup = []
            if t <= r - 1:
                sup.extend(t * n + i for i in rows_H[j])
                sup.append(r * n + t * m + j)
                if t > 0:
                    sup.append(r * n + (t - 1) * m + j)
            else:
                sup.append(r * n + (r - 1) * m + j)
            checks.append(tuple(sorted(sup)))
    return checks, weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--p", type=float, default=0.01, help="data error rate")
    ap.add_argument("--pm", type=float, default=None,
                    help="measurement error rate (default: same as --p)")
    ap.add_argument("--rpc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    pm = a.pm if a.pm is not None else a.p
    r = a.rounds

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks, weights = spacetime_system(HZ, r, a.p, pm)
    xbasis, xpiv = gf2_row_reduce(HX % 2)

    from ldpc import BpOsdDecoder
    from scipy.sparse import lil_matrix

    nv = r * n + r * m
    A = lil_matrix((len(checks), nv))
    for ci, sup in enumerate(checks):
        for i in sup:
            A[ci, i] = 1
    ch = np.array([a.p] * (r * n) + [pm] * (r * m))
    dec = BpOsdDecoder(A.tocsr(), error_channel=[float(v) for v in ch],
                       max_iter=40, bp_method="ms", schedule="parallel",
                       osd_method="osd_cs", osd_order=5)

    rng = np.random.default_rng(a.seed)
    nontrivial = proven = repaired = 0
    consistent = 0
    logical_faults = repair_logical = 0
    milp_decoded = 0
    gaps = []
    t_cert = 0.0
    for _s in range(a.shots):
        e = np.zeros((r, n), dtype=np.uint8)
        me = np.zeros((r, m), dtype=np.uint8)
        for t in range(r):
            e[t] = rng.random(n) < a.p
            me[t] = rng.random(m) < pm
        # true mechanism vector and its detector syndrome
        truth = np.concatenate([e.reshape(-1), me.reshape(-1)])
        syndrome = []
        for t in range(r + 1):
            for j in range(m):
                if t <= r - 1:
                    acc = int((HZ[j] @ e[t]) % 2) ^ int(me[t, j])
                    if t > 0:
                        acc ^= int(me[t - 1, j])
                    syndrome.append(acc)
                else:
                    syndrome.append(int(me[r - 1, j]))
        syndrome = np.array(syndrome, dtype=np.uint8)
        if not syndrome.any():
            continue
        nontrivial += 1
        e_hat = np.asarray(dec.decode(syndrome)).astype(np.uint8)
        if not ((A.tocsr() @ e_hat) % 2 == syndrome).all():
            continue
        consistent += 1
        support = tuple(int(i) for i in np.flatnonzero(e_hat))
        w_hat = float(sum(weights[i] for i in support))

        t0 = time.perf_counter()
        cert, bound, primal = feldman_dual(checks, weights, list(syndrome),
                                           return_primal=True,
                                           rpc_rounds=a.rpc)
        t_cert += time.perf_counter() - t0
        if cert is None:
            gaps.append(float("nan"))
            continue
        full = QldpcCertificate(error_support=support,
                                facet_duals=cert.facet_duals,
                                box_duals=cert.box_duals)
        rr = check_qldpc(checks=checks, weights=weights,
                         syndrome=list(syndrome), cert=full)

        def final_frame(vec):
            dat = vec[:r * n].reshape(r, n)
            return (dat.sum(axis=0) % 2).astype(np.uint8)

        if rr.accepted:
            proven += 1
            continue
        if "IMPOSSIBLE" in rr.reason:
            print(f"IMPOSSIBLE: {rr.reason}", file=sys.stderr)
            return 2
        if primal is not None:
            rep_vec = np.zeros(nv, dtype=np.uint8)
            rep_vec[list(primal)] = 1
            rep = QldpcCertificate(error_support=primal,
                                   facet_duals=cert.facet_duals,
                                   box_duals=cert.box_duals)
            r2 = check_qldpc(checks=checks, weights=weights,
                             syndrome=list(syndrome), cert=rep)
            if r2.accepted:
                repaired += 1
                fault = (final_frame(e_hat) ^ final_frame(truth))
                if not gf2_in_rowspace(xbasis, xpiv, fault, n):
                    logical_faults += 1
                fault2 = (final_frame(rep_vec) ^ final_frame(truth))
                if not gf2_in_rowspace(xbasis, xpiv, fault2, n):
                    repair_logical += 1
                continue
        msup, mw = exact_milp(checks, weights, list(syndrome),
                              time_limit=60.0)
        if mw is not None:
            milp_decoded += 1
        gaps.append(w_hat - bound)

    closed = proven + repaired
    g = np.array([x for x in gaps if x == x]) if gaps else np.array([0.0])
    print(f"[[144,12,12]] PHENOMENOLOGICAL r={r} p={a.p} pm={pm} "
          f"({r * n + r * m} mechanisms, {len(checks)} space-time checks)")
    print(f"  nontrivial              : {nontrivial}")
    print(f"  BP-OSD syndrome-valid   : {consistent}/{nontrivial}")
    print(f"  BP-OSD PROVEN min-wt    : {proven}/{consistent} "
          f"({100 * proven / max(consistent, 1):.1f}%)")
    print(f"  REPAIRED (LP integral)  : {repaired}  (BP-OSD logical on "
          f"{logical_faults}; repair logical on {repair_logical})")
    print(f"  CERTIFIED total         : {closed}/{consistent} "
          f"({100 * closed / max(consistent, 1):.2f}%)")
    if len(g) and gaps:
        print(f"  residue                 : {len(gaps)} shots "
              f"(MILP decoded {milp_decoded}), gap mean {g.mean():.3f}")
    print(f"  certificate cost        : "
          f"{1e3 * t_cert / max(consistent, 1):.0f} ms/shot")

    out = {"schema": "oneq-qldpc-bb-phenom/1",
           "code": "[[144,12,12]] gross, self-verified",
           "noise": f"phenomenological: data p={a.p}, measurement pm={pm}, "
                    f"rounds={r} + final perfect round; log-likelihood "
                    f"weights; NO circuit-level correlations (labeled)",
           "mechanisms": r * n + r * m,
           "spacetime_checks": len(checks),
           "shots": a.shots, "nontrivial": nontrivial,
           "bp_osd_syndrome_valid": consistent,
           "bp_osd_proven": proven, "repaired": repaired,
           "bp_osd_logical_faults": logical_faults,
           "repair_logical": repair_logical,
           "certified_total": closed,
           "certified_rate": closed / max(consistent, 1),
           "residue": {"n": len(gaps), "milp_decoded": milp_decoded,
                       "gap_mean": float(g.mean()) if len(g) else 0.0},
           "cert_ms_per_shot": 1e3 * t_cert / max(consistent, 1)}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "qldpc_bb_phenom.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
