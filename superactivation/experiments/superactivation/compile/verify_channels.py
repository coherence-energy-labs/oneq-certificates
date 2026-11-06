"""Deliverables 1 and 2: P, A, E~, D~ as circuits, compiled to {CZ, sx, rz, x} (+ measure, reset,
if_else), each verified against its target Choi matrix by the exact branch simulator.

  P  coherent (3 env qubits) and randomized (6 unitaries)          target: paper Kraus (eq. ppt_channel)
  A  coherent (flag + 3 env qubits)                                  target: A with |e> -> |1>|00>
  block = D~ o (P (x) A) o E~, for D~ in {paper (coherent V^dag untwist), opt (measurement-based
          untwist), coherent (no reads)} and both channel profiles     target: N~ of Prop. 1
Every circuit is checked BEFORE and AFTER compilation (the compiled circuit is what is counted).

Usage: python verify_channels.py   -> prints a table and writes verify_channels.json
"""
import json
import math
import time

import numpy as np
from qiskit import QuantumCircuit

import channels as C
import sclib as L

TOL = 1e-9
rows = []
t0 = time.time()


def body_counts(qc, path):
    """CZ count along one branch of nested if_else: path = list of booleans (True = true body)."""
    from qiskit.circuit import IfElseOp
    tot = {"cz": 0, "measure": 0, "reset": 0, "1q": 0, "ff": 0}
    it = iter(path)

    def walk(c):
        for inst in c.data:
            op = inst.operation
            if isinstance(op, IfElseOp):
                tot["ff"] += 1
                take = next(it, True)
                body = op.params[0] if take else op.params[1]
                if body is not None:
                    walk(body)
            elif op.name == "cz":
                tot["cz"] += 1
            elif op.name in ("measure", "reset"):
                tot[op.name] += 1
            elif op.name not in ("barrier",):
                tot["1q"] += 1
    walk(qc)
    return tot


def check(name, qc_fn, target, in_q, out_q, cl_out=(), weights=None, note=""):
    """qc_fn: a circuit or list of circuits (mixture with `weights`, optional per-circuit fixed
    classical output appended as diagonal block)."""
    circs = qc_fn if isinstance(qc_fn, list) else [qc_fn]
    ws = weights if weights is not None else [1.0]
    res = {}
    for stage in ("raw", "compiled"):
        J = 0
        cz = []
        for (qc, extra), w in zip(circs, ws):
            q = qc if stage == "raw" else L.compile_cz(qc, level=3)
            Jc = L.channel_choi(q, in_q, out_q, cl_out)
            if extra is not None:   # append a fixed classical flag value as a diagonal block
                d = Jc.shape[0]
                E = np.zeros((2, 2)); E[extra, extra] = 1
                Jc = np.kron(Jc, E)
                # move the flag to the end of the output ordering: already last
            J = J + w * Jc
            if stage == "compiled":
                cz.append(L.count_ops(q))
        dist = L.choi_dist(J, target)
        res[stage] = dist
        if stage == "compiled":
            res["counts"] = cz
    ok = res["raw"] < TOL and res["compiled"] < TOL
    rows.append({"name": name, "choi_trace_dist_raw": res["raw"], "choi_trace_dist_compiled": res["compiled"],
                 "pass": ok, "compiled_counts": res["counts"], "note": note})
    print(f"{name:52s} raw {res['raw']:.2e}  compiled {res['compiled']:.2e}  {'PASS' if ok else 'FAIL'}"
          f"   ({time.time() - t0:.0f}s)", flush=True)
    return res


JP = L.choi_from_kraus(L.horodecki_kraus())
JN = L.ntilde_choi()

# ---------------- P ----------------
r = check("P coherent Stinespring, gate-by-gate select phase", [(C.p_coherent_circuit(), None)], JP, [0, 1], [0, 1])
pc_gate = r["counts"][0]
r = check("P coherent Stinespring, phase-polynomial select (AAM)", [(C.p_coherent_circuit_pp(), None)], JP, [0, 1], [0, 1])
pc = r["counts"][0]
r = check("P randomized (6 unitaries, classical seed)",
          [(C.p_unitary_circuit(j), None) for j in range(6)], JP, [0, 1], [0, 1],
          weights=C.P_UNITARY_WEIGHTS)
pr = [c.get("cz", 0) for c in r["counts"]]
p_rand_expected_cz = float(np.dot(C.P_UNITARY_WEIGHTS, pr))

# alternative coherent P: generic isometry synthesis of the Stinespring (for comparison)
from qiskit.circuit.library import Isometry
Ks = L.horodecki_kraus()
Viso = np.zeros((32, 4), dtype=complex)   # out index: little-endian over (k, s, e0, e1, e2)
for e, K in enumerate(Ks):
    for col in range(4):
        # col is big-endian (k,s) in the paper; convert in/out to qiskit little-endian (s is bit 0? no: k is qubit 0)
        pass
# Build the isometry directly in qiskit little-endian: qubit 0 = k, qubit 1 = s, qubits 2..4 = env
for e, K in enumerate(Ks):
    for kin in range(2):
        for sin_ in range(2):
            colbig = kin * 2 + sin_
            colle = kin + 2 * sin_
            for kout in range(2):
                for sout in range(2):
                    rowbig = kout * 2 + sout
                    rowle = kout + 2 * sout + 4 * e
                    Viso[rowle, colle] = K[rowbig, colbig]
iso = QuantumCircuit(5)
iso.append(Isometry(Viso, 0, 0), [0, 1, 2, 3, 4])
r = check("P coherent via generic Isometry synthesis", [(iso, None)], JP, [0, 1], [0, 1],
          note="comparison only")
p_iso = r["counts"][0]

# ---------------- A ----------------
JA = C.a_target_choi()
r = check("A coherent (Toffoli move)", [(C.a_coherent_circuit("toffoli"), None)], JA, [0, 1], [2, 0, 1])
a_tof = r["counts"][0]
r = check("A coherent (relative-phase Toffoli move)", [(C.a_coherent_circuit("rccx"), None)], JA, [0, 1], [2, 0, 1])
a_rccx = r["counts"][0]

# ---------------- blocks: D~ o (P x A) o E~ = N~ ----------------
blocks = {}
for dec in ("paper", "opt", "coherent"):
    for order in (("s2_first", "s1_first") if dec == "paper" else ("s2_first",)):
        qc = C.block_coherent(decoder=dec, move_impl="rccx", order=order)
        nm = f"block COHERENT channels, D~={dec}" + (f" (U on {order})" if dec == "paper" else "")
        r = check(nm, [(qc, None)], JN, [0], [0], cl_out=(0,))
        blocks[nm] = r
        # randomized profile: mixture over (jP, erased) with the flag appended classically
        circs, ws = [], []
        for jP in range(6):
            for er in (0, 1):
                circs.append((C.block_randomized(jP, er, decoder=dec, order=order), er))
                ws.append(C.P_UNITARY_WEIGHTS[jP] * 0.5)
        nm2 = f"block RANDOMIZED channels, D~={dec}" + (f" (U on {order})" if dec == "paper" else "")
        r2 = check(nm2, circs, JN, [0], [0], weights=ws)
        blocks[nm2] = r2

# ---------------- per-branch counts of the compiled pieces ----------------
def cz_of(qc):
    return L.count_ops(L.compile_cz(qc, level=3)).get("cz", 0)

et = QuantumCircuit(4); C.e_tilde(et, 0, 1, 2, 3)
summary = {
    "E~ CZ": cz_of(et),
    "P coherent CZ (gate-by-gate select)": pc_gate.get("cz", 0),
    "P coherent CZ (phase-polynomial select, used in blocks)": pc.get("cz", 0),
    "P coherent CZ (generic Isometry)": p_iso.get("cz", 0),
    "P randomized CZ per unitary": pr,
    "P randomized expected CZ": p_rand_expected_cz,
    "A coherent CZ (Toffoli move)": a_tof.get("cz", 0),
    "A coherent CZ (rccx move)": a_rccx.get("cz", 0),
}
for dec in ("paper", "opt", "coherent"):
    q0 = QuantumCircuit(4, 2); C.d0(q0, 0, 1, 2, 3, q0.clbits[0], q0.clbits[1], dec)
    q0c = L.compile_cz(q0, level=3)
    summary[f"D^0 {dec}: compiled counts"] = L.count_ops(q0c)
    summary[f"D^0 {dec}: CZ"] = L.count_ops(q0c).get("cz", 0) + L.count_ops(q0c).get("cond:cz", 0)
summary["D^1 CZ"] = 0
print(json.dumps(summary, indent=1, default=str))
with open("verify_channels.json", "w") as fh:
    json.dump({"rows": rows, "summary": summary, "tol": TOL}, fh, indent=1, default=str)
print("all pass:", all(r["pass"] for r in rows if r["note"] != "comparison only"),
      " comparison rows pass:", all(r["pass"] for r in rows))
