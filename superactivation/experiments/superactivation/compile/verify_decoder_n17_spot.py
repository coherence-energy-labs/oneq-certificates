"""Spot-check the ACTUAL certified-code decoders at n = 17 (and 24) by exact simulation of the compiled
dynamic circuits, on every branch whose simulation fits (m = n - k <= MMAX unflagged qubits; the X-read
of m thermometer qubits multiplies branches by 2^m). For each (k, b) checked, all b, the circuit's
channel on the support S of Bob's state equals the minimum-rank restricted decoder (Choi distance), and
the compiled CZ count equals the component formula used in decoder_counts.py.

Usage: python verify_decoder_n17_spot.py [n] [MMAX]   (default 17 7)
"""
import json
import sys
import time

import numpy as np

import codes
import decoder as Dc
import dicke as Dk
import sclib as L
from codes import sa

n = int(sys.argv[1]) if len(sys.argv) > 1 else 17
MMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 7
X, decs = codes.load_code(n)
V, Xiso, _ = codes.isometric_encoder(X, n)
rows = []
t0 = time.time()
for k in range(n - MMAX, n + 1):
    m = n - k
    R = sa.bob_states(Xiso, n, k)
    for b in range(k + 1):
        Ks = codes.decoder_kraus(np.asarray(decs[k][b]), m + 1)
        Sb, _ = Dc.support_basis(sa.ptrace_ref(R[b], m + 1))
        KS0 = Dc.restricted_kraus(Ks, Sb)
        I2S = np.kron(np.eye(2), Sb)
        KS = Dc.min_rank_kraus(KS0, I2S.conj().T @ R[b] @ I2S)
        costs = {md: Dc.decoder_cost(m, Ks, Sb, md, KS=KS) for md in ("restricted", "compressed")}
        mode = min(costs, key=lambda md: costs[md]["total_cz"])
        qc, info = Dc.decoder_circuit(m, Ks, mode=mode, Sb=Sb, KS=KS)
        qcc = L.compile_cz(qc)
        vecsS = [Dk.symmetric_vector(m, Sb[:, j]) if m > 0 else np.array([1.0 + 0j]) for j in range(Sb.shape[1])]
        J = L.channel_choi_basis(qcc, vecsS, list(range(m)), [info["out_qubit"]])
        Jt = L.choi_from_kraus(KS)
        cz = L.count_ops(qcc).get("cz", 0) + L.count_ops(qcc).get("cond:cz", 0)
        rows.append({"k": k, "b": b, "m": m, "mode": mode, "choi_dist": L.choi_dist(J, Jt), "cz_compiled": cz,
                     "cz_formula": costs[mode]["total_cz"]})
    print(f"n={n} k={k}: worst Choi dist {max(r['choi_dist'] for r in rows if r['k'] == k):.1e}, "
          f"CZ compiled==formula: {all(r['cz_compiled'] == r['cz_formula'] for r in rows if r['k'] == k)} "
          f"({time.time() - t0:.0f}s)", flush=True)
ok = all(r["choi_dist"] < 1e-9 for r in rows) and all(r["cz_compiled"] == r["cz_formula"] for r in rows)
json.dump({"n": n, "MMAX": MMAX, "rows": rows, "pass": ok}, open(f"verify_decoder_spot_n{n}.json", "w"), indent=1)
print("PASS", ok, f"({len(rows)} branches)")
