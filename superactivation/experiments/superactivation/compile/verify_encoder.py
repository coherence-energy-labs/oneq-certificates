"""Deliverable 3: the encoder V: C^2 -> Sym^n as a circuit (unary Givens network W, then the BE Dicke
unitary U_n), verified by exact statevector comparison.

  part A  U_n alone: |unary_w> -> |D^n_w> for EVERY w with ONE common phase, n = 2..7, all gate impls
  part B  random complex isometries into Sym^n, n = 3..6 (5 draws each), raw and compiled circuits
  part C  the certified codes' isometric encoders at n = 17 (compiled, 2^17 statevector) and, with
          --n24, n = 24 (2^24 statevector, ~0.3 GB per vector)
CZ counts: compiled with optimization_level=3 to {cz, sx, rz, x}; compared to the closed forms
  U_n:  sum_l [c_i + c_ii (l-2)]  = 2(n-1)^2 for (givens, num4)
  W  :  5n - 5 (generic two-column case: n-1 one-CZ + 2 two-CZ + (n-2) four-CZ rotations)

Usage: python verify_encoder.py [--n24]
"""
import json
import sys
import time

import numpy as np

import dicke as Dk
import sclib as L
import codes

rng = np.random.default_rng(17)
out = {"partA": [], "partB": [], "partC": []}
t0 = time.time()
IMPLS = [("be", "be"), ("givens", "mux5"), ("givens", "num4")]


def phase_aligned_err(a, b):
    ov = np.vdot(a, b)
    ph = ov / abs(ov) if abs(ov) > 1e-300 else 1.0
    return float(np.max(np.abs(a * ph - b)))


# ---------------- part A ----------------
for impl in IMPLS:
    for n in range(2, 8):
        qc = L.QuantumCircuit(n)
        Dk.dicke_unitary(qc, list(range(n)), *impl)
        for stage in ("raw", "compiled"):
            q = qc if stage == "raw" else L.compile_cz(qc)
            errs, phases = [], []
            for w in range(n + 1):
                sv = L.statevector(q, Dk.unary_index(n, w))
                d = Dk.dicke_vector(n, w)
                ov = np.vdot(d, sv)
                phases.append(ov)
                errs.append(float(np.max(np.abs(sv - ov * d))) + abs(1 - abs(ov)))
            ph0 = phases[0] / abs(phases[0])
            perr = max(abs(ph / abs(ph) - ph0) for ph in phases)
            cz = L.count_ops(q).get("cz", 0) if stage == "compiled" else None
            out["partA"].append({"impl": impl, "n": n, "stage": stage, "max_err": max(errs),
                                 "common_phase_err": perr, "cz": cz,
                                 "cz_formula": Dk.dicke_unitary_cz(n, *impl)})
    print(f"part A {impl}: worst err {max(r['max_err'] + r['common_phase_err'] for r in out['partA'] if r['impl'] == impl):.2e}"
          f"  cz(n=7) compiled {out['partA'][-1]['cz']} formula {out['partA'][-1]['cz_formula']}  ({time.time() - t0:.0f}s)", flush=True)

# ---------------- part B ----------------
def check_encoder(V, impl, compiled=True):
    n = V.shape[0] - 1
    qc, ops, resid = Dk.encoder_circuit(V, *impl)
    q = L.compile_cz(qc) if compiled else qc
    targets = [sum(V[w, r] * Dk.dicke_vector(n, w) for w in range(n + 1)) for r in range(2)]
    svs = [L.statevector(q, 0), L.statevector(q, 1 << (n - 1))]
    # one global phase common to both columns
    ov = np.vdot(targets[0], svs[0]) + np.vdot(targets[1], svs[1])
    ph = ov / abs(ov)
    err = max(float(np.max(np.abs(svs[r] - ph * targets[r]))) for r in range(2))
    return err, L.count_ops(q).get("cz", 0), Dk.unary_isometry_cz(ops), resid


for impl in IMPLS:
    worst = 0.0
    for n in range(3, 7):
        for draw in range(5):
            G = rng.normal(size=(n + 1, 2)) + 1j * rng.normal(size=(n + 1, 2))
            V, _ = np.linalg.qr(G)
            for compiled in (False, True):
                err, cz, wcz, resid = check_encoder(V, impl, compiled)
                worst = max(worst, err)
                out["partB"].append({"impl": impl, "n": n, "draw": draw, "compiled": compiled, "err": err,
                                     "cz": cz if compiled else None, "W_cz_formula": wcz,
                                     "U_cz_formula": Dk.dicke_unitary_cz(n, *impl)})
    print(f"part B {impl}: worst statevector error {worst:.2e}  ({time.time() - t0:.0f}s)", flush=True)

# ---------------- part C ----------------
nlist = [17] + ([24] if "--n24" in sys.argv else [])
for n in nlist:
    X, decs = codes.load_code(n)
    V, Xiso, info = codes.isometric_encoder(X, n)
    impl = ("givens", "num4")
    qc, ops, resid = Dk.encoder_circuit(V, *impl)
    q = L.compile_cz(qc)
    cz = L.count_ops(q).get("cz", 0)
    cz_l3 = L.count_ops(L.compile_cz(qc, level=3)).get("cz", 0)
    t1 = time.time()
    targets = [Dk.symmetric_vector(n, V[:, r]) for r in range(2)]
    svs = [L.statevector(q, 0), L.statevector(q, 1 << (n - 1))]
    ov = np.vdot(targets[0], svs[0]) + np.vdot(targets[1], svs[1])
    ph = ov / abs(ov)
    err = max(float(np.max(np.abs(svs[r] - ph * targets[r]))) for r in range(2))
    rec = {"n": n, "impl": impl, "statevector_err": err, "cz_compiled": cz, "cz_level3": cz_l3,
           "W_cz": Dk.unary_isometry_cz(ops), "U_cz": Dk.dicke_unitary_cz(n, *impl),
           "depth_compiled": q.depth(), "depth_2q": q.depth(lambda i: i.operation.num_qubits == 2),
           "ops": L.count_ops(q), "encoder_info": info, "sim_seconds": round(time.time() - t1, 1)}
    out["partC"].append(rec)
    print(f"part C n={n}: err {err:.2e}  CZ {cz} (W {rec['W_cz']} + U {rec['U_cz']})  depth {rec['depth_compiled']}"
          f" 2q-depth {rec['depth_2q']}  ({time.time() - t0:.0f}s)", flush=True)
    del svs, targets

with open("verify_encoder" + ("_n24" if "--n24" in sys.argv else "") + ".json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
okA = all(r["max_err"] + r["common_phase_err"] < 1e-9 for r in out["partA"])
okB = all(r["err"] < 1e-9 for r in out["partB"])
okC = all(r["statevector_err"] < 1e-9 for r in out["partC"])
print("PASS A", okA, " PASS B", okB, " PASS C", okC)
