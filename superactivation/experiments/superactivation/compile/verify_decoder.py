"""Deliverable 4 verification: decoder circuits against the ONE_Q module's decoders, small n.

For n in NS: a random isometric encoder V into Sym^n (complex), decoders for every (k, b) from the
module's own batched power iteration (oneq.superactivation.decode_power), then
  (1) route a (full decoder): the compiled dynamic circuit's channel on Sym^{n-k}, computed exactly
      by the branch simulator, equals the module decoder's channel (Choi, Dicke basis) -- every (k,b);
  (2) route b (support-compressed decoder): equals the module decoder on the support S of Bob's state,
      and the restricted decoder has the same branch fidelity Tr(rho D)/4 as the module decoder;
  (3) END TO END: compiled encoder circuit -> exact N~ on every qubit (all 2^n flag patterns, flagged
      qubits read in Z, b = weight) -> compiled decoder circuit on the unflagged qubits IN THEIR
      PHYSICAL POSITIONS -> entanglement fidelity with the reference, compared with the module's
      fidelity(X, decoders). This is the check that conventions (Dicke basis, Choi orientation, Kraus
      conjugation, qubit order) agree between the circuits and the certified reduction.

Usage: python verify_decoder.py [n ...]   (default 3 4 5)
"""
import itertools
import json
import math
import sys
import time

import numpy as np

import codes
import decoder as Dc
import dicke as Dk
import sclib as L
from codes import sa

TOL = 1e-9
NS = [int(a) for a in sys.argv[1:]] or [3, 4, 5]
rng = np.random.default_rng(31)
report = {"tol": TOL, "codes": []}
t0 = time.time()


def kraus_from_choi(J, din, dout):
    w, U = np.linalg.eigh((J + J.conj().T) / 2)
    Ks = []
    for lam, u in zip(w, U.T):
        if lam > 1e-13:
            Ks.append(np.sqrt(lam) * u.reshape(din, dout).T)
    return Ks


for n in NS:
    G = rng.normal(size=(n + 1, 2)) + 1j * rng.normal(size=(n + 1, 2))
    V, _ = np.linalg.qr(G)
    v = np.concatenate([V[:, 0], V[:, 1]])
    X = np.outer(v, v.conj())
    decs, F_mod, _ = sa.decode_power(X, n, iters=400)
    F_mod = sa.fidelity(X, decs, n)
    rec = {"n": n, "F_module": F_mod, "branches": []}
    MODES = ("full", "restricted", "compressed", "restricted_min", "compressed_min")
    Jdicts = {md: {} for md in MODES}
    for k in range(n + 1):
        m = n - k
        R = sa.bob_states(X, n, k)
        for b in range(k + 1):
            D = np.asarray(decs[k][b])
            Ks = codes.decoder_kraus(D, m + 1)
            Jt = L.choi_from_kraus(Ks)
            vecs = [Dk.dicke_vector(m, w) if m > 0 else np.array([1.0]) for w in range(m + 1)]
            rho_sym = sa.ptrace_ref(R[b], m + 1)
            Sb, _ = Dc.support_basis(rho_sym)
            vecsS = [Dk.symmetric_vector(m, Sb[:, j]) if m > 0 else np.array([1.0 + 0j]) for j in range(Sb.shape[1])]
            JtS = L.choi_from_kraus([K @ Sb for K in Ks])
            KS = Dc.restricted_kraus(Ks, Sb)
            I2S = np.kron(np.eye(2), Sb)
            KSmin = Dc.min_rank_kraus(KS, I2S.conj().T @ R[b] @ I2S)
            JtSmin = L.choi_from_kraus(KSmin)
            br = {"k": k, "b": b, "m": m, "s": Sb.shape[1], "rank_restricted": len(KS), "rank_min": len(KSmin)}
            for mode in MODES:
                base = mode.replace("_min", "")
                qc, info = Dc.decoder_circuit(m, Ks, mode=base, Sb=Sb, KS=KSmin if mode.endswith("_min") else None)
                qcc = L.compile_cz(qc)
                out = info["out_qubit"]
                if mode == "full":   # equals the module decoder on ALL of Sym^m
                    Jc = L.channel_choi_basis(qcc, vecs, list(range(m)), [out])
                    br["full_choi_dist"] = L.choi_dist(Jc, Jt)
                # on the support S: the module decoder (full/restricted/compressed) or its
                # fixed-fidelity minimum-rank replacement (*_min)
                JcS = L.channel_choi_basis(qcc, vecsS, list(range(m)), [out])
                br[f"{mode}_choi_dist_on_S"] = L.choi_dist(JcS, JtSmin if mode.endswith("_min") else JtS)
                # full-input channel for the end-to-end test
                Jdicts[mode][(k, b)] = L.channel_choi(qcc, list(range(m)), [out]) if m > 0 else JcS
                br[f"{mode}_cz"] = L.count_ops(qcc).get("cz", 0) + L.count_ops(qcc).get("cond:cz", 0)
                br[f"{mode}_info"] = info
                # the component formula used for n = 17, 24 must equal the compiled count
                if mode != "full":
                    cst = Dc.decoder_cost(m, Ks, Sb, base, KS=KSmin if mode.endswith("_min") else None)
                    br[f"{mode}_formula_cz"] = cst["total_cz"]
            KS = Dc.restricted_kraus(Ks, Sb)
            Dr = codes.adjoint_choi_from_kraus([K @ Sb.conj().T for K in KS], m + 1)
            br["branch_F_module"] = float(np.real(np.trace(R[b] @ D))) / 4
            br["branch_F_restricted"] = float(np.real(np.trace(R[b] @ Dr))) / 4
            rec["branches"].append(br)
        worst = max(max([r["full_choi_dist"]] + [r[f"{md}_choi_dist_on_S"] for md in MODES])
                    for r in rec["branches"] if r["k"] == k)
        fm = [(r[f"{md}_cz"], r[f"{md}_formula_cz"]) for r in rec["branches"] if r["k"] == k for md in MODES if md != "full"]
        print(f"   compiled-vs-formula CZ mismatches at k={k}: {sum(1 for a, b2 in fm if a != b2)} of {len(fm)}", flush=True)
        print(f"n={n} k={k}: worst Choi distance over modes/branches {worst:.1e} ({time.time() - t0:.0f}s)", flush=True)

    # ---- end to end
    enc, _, _ = Dk.encoder_circuit(V, "givens", "num4")
    encc = L.compile_cz(enc)
    sv = [L.statevector(encc, 0), L.statevector(encc, 1 << (n - 1))]
    Psi = np.stack([s.reshape([2] * n).transpose(list(range(n))[::-1]) for s in sv]) / math.sqrt(2)  # (R, q0..q_{n-1})
    p = sa.P_FLIP
    phi = np.zeros(4); phi[0] = phi[3] = 1 / math.sqrt(2)

    def e2e(Jdict):
        Ftot = 0.0
        for z in itertools.product((0, 1), repeat=n):
            Fl = [j for j in range(n) if z[j]]
            Un = [j for j in range(n) if not z[j]]
            k, m = len(Fl), len(Un)
            for xF in itertools.product((0, 1), repeat=k):
                b = sum(xF)
                Kd = kraus_from_choi(Jdict[(k, b)], 2 ** m if m > 0 else 1, 2)
                sig = np.zeros((4, 4), dtype=complex)
                for yF in itertools.product((0, 1), repeat=k):
                    pr = 1.0
                    for xj, yj in zip(xF, yF):
                        pr *= p if xj != yj else (1 - p)
                    idx = [slice(None)] * (n + 1)
                    for j, yj in zip(Fl, yF):
                        idx[1 + j] = yj
                    psi = Psi[tuple(idx)].reshape(2, -1)        # (R, U in ascending position order)
                    for K in Kd:
                        o = psi @ K.T                           # (R, out)
                        vec = o.reshape(-1)
                        sig += pr * np.outer(vec, vec.conj())
                Ftot += 2.0 ** (-n) * float(np.real(phi @ sig @ phi))
        return Ftot

    for mode in MODES:
        Fm = e2e(Jdicts[mode])
        rec[f"F_end_to_end_{mode}"] = Fm
        rec[f"e2e_err_{mode}"] = abs(Fm - F_mod)
    print(f"n={n}: module F = {F_mod:.12f}  end-to-end circuits |dF|: full {rec['e2e_err_full']:.1e}"
          f" restricted {rec['e2e_err_restricted']:.1e} compressed {rec['e2e_err_compressed']:.1e}"
          f" restricted_min {rec['e2e_err_restricted_min']:.1e} compressed_min {rec['e2e_err_compressed_min']:.1e}"
          f"  ({time.time() - t0:.0f}s)", flush=True)
    report["codes"].append(rec)

ok = all(max([r["full_choi_dist"]] + [r[f"{md}_choi_dist_on_S"] for md in MODES]) < TOL
         and abs(r["branch_F_module"] - r["branch_F_restricted"]) < 1e-12
         for c in report["codes"] for r in c["branches"])
ok_formula = all(r[f"{md}_cz"] == r[f"{md}_formula_cz"] for c in report["codes"] for r in c["branches"]
                 for md in MODES if md != "full")
report["pass_formula_equals_compiled"] = ok_formula
print("compiled CZ == component formula for every branch/mode:", ok_formula)
ok_e2e = all(c[f"e2e_err_{md}"] < TOL for c in report["codes"] for md in MODES)
report["pass_branches"] = ok
report["pass_end_to_end"] = ok_e2e
with open("verify_decoder.json", "w") as fh:
    json.dump(report, fh, indent=1, default=str)
print("PASS branches", ok, " PASS end-to-end", ok_e2e)
