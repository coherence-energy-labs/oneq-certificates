"""Line-compatible (nearest-neighbour) encoder: statevector verification and heavy-hex placement.

  * random complex isometries into Sym^n, n = 3..7, and the certified n = 17 (and 24 with --n24)
    encoders: compiled circuit == sum_w V[w,r] |D_w> up to one global phase;
  * every two-qubit gate of the compiled circuit acts on |i - j| = 1 (line neighbours);
  * transpiled onto the ibm_kingston snapshot (FakeKingston, local) with the line mapped onto a
    simple path of the heavy-hex coupling graph: routed CZ count vs logical (no SWAPs expected).

Usage: python verify_encoder_lnn.py [--n24]
"""
import json
import sys
import time

import numpy as np
from qiskit import transpile

import codes
import dicke as Dk
import sclib as L

rng = np.random.default_rng(23)
out = {"small": [], "codes": []}
t0 = time.time()


def check(V, compiled=True):
    n = V.shape[0] - 1
    qc, ops, _ = Dk.encoder_circuit_lnn(V)
    q = L.compile_cz(qc) if compiled else qc
    tg = [Dk.symmetric_vector(n, V[:, r]) for r in range(2)]
    sv = [L.statevector(q, 0), L.statevector(q, 1 << (n - 1))]
    ov = np.vdot(tg[0], sv[0]) + np.vdot(tg[1], sv[1])
    ph = ov / abs(ov)
    err = max(float(np.max(np.abs(sv[r] - ph * tg[r]))) for r in range(2))
    nn = all(abs(q.find_bit(i.qubits[0]).index - q.find_bit(i.qubits[1]).index) == 1
             for i in q.data if i.operation.num_qubits == 2)
    return err, L.count_ops(q).get("cz", 0), nn, q


for n in range(3, 8):
    for draw in range(3):
        G = rng.normal(size=(n + 1, 2)) + 1j * rng.normal(size=(n + 1, 2))
        V, _ = np.linalg.qr(G)
        err, cz, nn, _ = check(V)
        out["small"].append({"n": n, "err": err, "cz": cz, "nearest_neighbour": nn,
                             "cz_formula": 5 * n - 5 + Dk.dicke_unitary_lnn_cz(n)})
print(f"small n: worst err {max(r['err'] for r in out['small']):.2e}, all nearest-neighbour "
      f"{all(r['nearest_neighbour'] for r in out['small'])}, CZ == formula "
      f"{all(r['cz'] <= r['cz_formula'] for r in out['small'])} ({time.time() - t0:.0f}s)", flush=True)


def heavy_hex_path(cmap, length):
    adj = {}
    for a, b in cmap.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = []

    class _G:
        @staticmethod
        def nodes():
            return adj.keys()
    G = _G()

    def dfs(path, seen):
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        if len(best) >= length:
            return True
        for nb in sorted(adj[path[-1]]):
            if nb not in seen:
                seen.add(nb); path.append(nb)
                if dfs(path, seen):
                    return True
                path.pop(); seen.discard(nb)
        return False
    for start in sorted(G.nodes()):
        if dfs([start], {start}):
            break
    return best[:length]


from qiskit_ibm_runtime.fake_provider import FakeKingston
backend = FakeKingston()
COUNT_ONLY = "--count-only" in sys.argv
for n in ([] if COUNT_ONLY else [17]) + ([24] if ("--n24" in sys.argv or COUNT_ONLY) else []):
    X, decs = codes.load_code(n)
    V, Xiso, _ = codes.isometric_encoder(X, n)
    if COUNT_ONLY:   # no 2^n statevector: counts, nearest-neighbour check, routing only
        qc0, _, _ = Dk.encoder_circuit_lnn(V)
        q = L.compile_cz(qc0)
        err, cz = float("nan"), L.count_ops(q).get("cz", 0)
        nn = all(abs(q.find_bit(i.qubits[0]).index - q.find_bit(i.qubits[1]).index) == 1
                 for i in q.data if i.operation.num_qubits == 2)
    else:
        err, cz, nn, q = check(V)
    path = heavy_hex_path(backend.coupling_map, n)
    qc_lnn, _, _ = Dk.encoder_circuit_lnn(V)
    tr = transpile(qc_lnn, backend=backend, initial_layout=path, optimization_level=1, seed_transpiler=3,
                   approximation_degree=1.0)
    rcz = L.count_ops(tr).get("cz", 0)
    rec = {"n": n, "statevector_err": err, "cz_logical": cz, "nearest_neighbour": nn,
           "cz_formula": 5 * n - 5 + Dk.dicke_unitary_lnn_cz(n), "heavy_hex_path": path,
           "cz_routed_on_path": rcz, "depth_routed": tr.depth(),
           "depth_2q_routed": tr.depth(lambda i: i.operation.num_qubits == 2),
           "depth_2q_logical": q.depth(lambda i: i.operation.num_qubits == 2)}
    out["codes"].append(rec)
    print(f"n={n}: LNN encoder err {err:.2e}  CZ {cz} (formula {rec['cz_formula']})  nearest-neighbour {nn}"
          f"  -> heavy-hex path routed CZ {rcz}, depth {tr.depth()} (2q {rec['depth_2q_routed']})"
          f"  ({time.time() - t0:.0f}s)", flush=True)
tag = "_n24count" if COUNT_ONLY else ("_n24" if "--n24" in sys.argv else "")
json.dump(out, open("verify_encoder_lnn" + tag + ".json", "w"), indent=1)
ok = all(r["err"] < 1e-9 and r["nearest_neighbour"] for r in out["small"]) and \
    all((COUNT_ONLY or r["statevector_err"] < 1e-9) and r["nearest_neighbour"] for r in out["codes"])
print("PASS", ok)
