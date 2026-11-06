"""Decoder resource counts for the certified codes at n = 17 and n = 24, every branch (k, b).

For each (k, b): the witness decoder D_{k,b} (Kraus via codes.decoder_kraus), the support S of Bob's
state rho_{k,b}(X_iso) on Sym^{n-k} (eigen-threshold 1e-12 relative), and the cost of the two
fidelity-equivalent constructions ('restricted': all n-k+1 levels kept; 'compressed': S first mapped
to the lowest s unary levels by a Givens network). The cheaper one is taken per branch. Fidelity is
recomputed with the restricted decoders to show it is unchanged.

Aggregates: worst case over (k, b); expectation over the flag weight k ~ Bin(n, 1/2) and the observed
weight b ~ Pr(b | k) = Tr rho_{k,b} / 2 of THIS code.

Usage: python decoder_counts.py [n ...]  (default 17 24)  -> decoder_counts_n{n}.json
"""
import json
import math
import sys
import time

import numpy as np

import codes
import decoder as Dc
import dicke as Dk
from codes import sa

NS = [int(a) for a in sys.argv[1:]] or [17, 24]
for n in NS:
    t0 = time.time()
    X, decs = codes.load_code(n)
    V, Xiso, info = codes.isometric_encoder(X, n)
    F_cert_witness = codes.fidelity(X, decs, n)
    F_iso = codes.fidelity(Xiso, decs, n)
    rows = []
    F_restricted = 0.0
    for k in range(n + 1):
        m = n - k
        R = sa.bob_states(Xiso, n, k)
        Fk_res = 0.0
        for b in range(k + 1):
            D = np.asarray(decs[k][b])
            Ks = codes.decoder_kraus(D, m + 1)
            rho_sym = sa.ptrace_ref(R[b], m + 1)
            Sb, ev = Dc.support_basis(rho_sym)
            KS0 = Dc.restricted_kraus(Ks, Sb)
            I2S = np.kron(np.eye(2), Sb)
            KS = Dc.min_rank_kraus(KS0, I2S.conj().T @ R[b] @ I2S)   # same branch fidelity, rank <= s
            Dr = codes.adjoint_choi_from_kraus([K @ Sb.conj().T for K in KS], m + 1)
            Fk_res += float(np.real(np.trace(R[b] @ Dr))) / 4
            pb = float(np.real(np.trace(R[b]))) / 2
            costs = {}
            for mode in ("restricted", "compressed"):
                costs[mode] = Dc.decoder_cost(m, Ks, Sb, mode, KS=KS)
            costs["restricted_witness_rank"] = Dc.decoder_cost(m, Ks, Sb, "restricted", KS=KS0)
            best = min(("restricted", "compressed"), key=lambda md: costs[md]["total_cz"])
            rows.append({"k": k, "b": b, "m": m, "p_k": math.comb(n, k) / 2 ** n, "p_b_given_k": pb,
                         "support_dim": int(Sb.shape[1]), "witness_kraus_rank": len(Ks),
                         "restricted_kraus_rank_witness": len(KS0), "min_kraus_rank": len(KS), "best_mode": best,
                         "cost": costs[best], "costs": costs,
                         "flag_z_measurements": k})
        F_restricted += math.comb(n, k) / 2 ** n * Fk_res
        print(f"n={n} k={k:2d}: cz best per b {[r['cost']['total_cz'] for r in rows if r['k'] == k]}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    W = np.array([r["p_k"] * r["p_b_given_k"] for r in rows])
    tot = np.array([r["cost"]["total_cz"] for r in rows])
    meas = np.array([r["cost"]["x_measurements"] + r["k"] for r in rows])
    ff = np.array([r["cost"]["feedforward_1q"] for r in rows])
    qubits = np.array([r["cost"]["qubits"] for r in rows])
    summary = {
        "n": n,
        "sum_weights": float(W.sum()),
        "F_witness_float": F_cert_witness,
        "F_isometric_encoder_witness_decoders": F_iso,
        "F_isometric_encoder_restricted_decoders": F_restricted,
        "encoder_info": info,
        "decoder_cz_expected": float(W @ tot),
        "decoder_cz_worst": int(tot.max()),
        "decoder_cz_worst_branch": [rows[int(tot.argmax())]["k"], rows[int(tot.argmax())]["b"]],
        "decoder_cz_best": int(tot.min()),
        "decoder_measurements_expected": float(W @ meas),
        "decoder_measurements_worst": int(meas.max()),
        "decoder_feedforward_1q_expected": float(W @ ff),
        "decoder_qubits_max": int(qubits.max()),
        "components_expected": {key: float(W @ np.array([r["cost"][key] for r in rows]))
                                for key in ("U_dag_cz", "givens_cz", "unary_to_binary_cz", "phase_fix_cz",
                                            "dec_unitary_cz")},
        "per_k": [{"k": k, "p_k": math.comb(n, k) / 2 ** n,
                   "cz_min_over_b": int(min(r["cost"]["total_cz"] for r in rows if r["k"] == k)),
                   "cz_max_over_b": int(max(r["cost"]["total_cz"] for r in rows if r["k"] == k)),
                   "support_dims": sorted({r["support_dim"] for r in rows if r["k"] == k}),
                   "modes": sorted({r["best_mode"] for r in rows if r["k"] == k}),
                   "dec_unitary_qubits": sorted({r["cost"]["dec_unitary_qubits"] for r in rows if r["k"] == k}),
                   "U_dag_cz": Dk.dicke_unitary_cz(n - k) if n - k >= 2 else 0}
                  for k in range(n + 1)],
        "seconds": round(time.time() - t0, 1),
    }
    with open(f"decoder_counts_n{n}.json", "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_k"}, indent=1, default=str))
