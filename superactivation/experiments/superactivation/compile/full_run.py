"""Deliverable 5: whole-run circuits (one shot of the experiment, RANDOMIZED channel profile) for
sampled flag patterns, their logical counts, and heavy-hex routing on the ibm_kingston snapshot
(qiskit_ibm_runtime.fake_provider.FakeKingston: 156-qubit Heron r2 coupling map + basis; LOCAL, no
network call). Also routes the sub-blocks alone (encoder, one channel use, one decoder) to separate
routing overhead by component.

One shot = Bell pair (R, input) -> encoder -> n x [E~, P_j, (A erases?), D~] -> read flagged qubits in Z
-> decoder(k, b) on the unflagged qubits -> Bell measurement (R, out).
The controller's draws (jP per use, erasure per use) and the decoder branch (k, b) are fixed per
compiled circuit: a real run needs one compiled variant per draw (or real-time addressing), which
is recorded as NOT modeled.

Usage: python full_run.py [n] [samples] [--route]
"""
import json
import math
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

import channels as C
import codes
import decoder as Dc
import dicke as Dk
import sclib as L
from codes import sa

n = int(sys.argv[1]) if len(sys.argv) > 1 else 17
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ROUTE = "--route" in sys.argv
rng = np.random.default_rng(1000 + n)

X, decs = codes.load_code(n)
V, Xiso, _ = codes.isometric_encoder(X, n)
LNN = "--lnn" in sys.argv
if LNN:   # nearest-neighbour encoder placed on a heavy-hex path (verify_encoder_lnn.py)
    enc, ops, _ = Dk.encoder_circuit_lnn(V)
else:
    enc, ops, _ = Dk.encoder_circuit(V, "givens", "num4")


def hh_path(cmap, length):
    adj = {}
    for a_, b_ in cmap.get_edges():
        adj.setdefault(a_, set()).add(b_)
        adj.setdefault(b_, set()).add(a_)
    best = []

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
    for s0 in sorted(adj):
        if dfs([s0], {s0}):
            break
    return best[:length], adj


def make_layout(cmap, n):
    """Data qubit j on path[j]; reference R and the 3 ancillas of use j on the nearest free physical
    qubits (BFS from the data qubit). Order matches build_shot's registers: R, d[0..n-1], a[0..3n-1]."""
    path, adj = hh_path(cmap, n)
    used = set(path)

    def nearest(src):
        from collections import deque
        dq, seen = deque([src]), {src}
        while dq:
            u = dq.popleft()
            if u not in used:
                used.add(u)
                return u
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v); dq.append(v)
        raise RuntimeError("no free qubit")
    Rp = nearest(path[n - 1])
    anc = []
    for j in range(n):
        for _ in range(3):
            anc.append(nearest(path[j]))
    return [Rp] + list(path) + anc


def decoder_for(k, b):
    m = n - k
    R = sa.bob_states(Xiso, n, k)
    Ks = codes.decoder_kraus(np.asarray(decs[k][b]), m + 1)
    Sb, _ = Dc.support_basis(sa.ptrace_ref(R[b], m + 1))
    KS0 = Dc.restricted_kraus(Ks, Sb)
    I2S = np.kron(np.eye(2), Sb)
    KS = Dc.min_rank_kraus(KS0, I2S.conj().T @ R[b] @ I2S)
    costs = {md: Dc.decoder_cost(m, Ks, Sb, md, KS=KS) for md in ("restricted", "compressed")}
    best = min(costs, key=lambda md: costs[md]["total_cz"])
    qc, info = Dc.decoder_circuit(m, Ks, mode=best, Sb=Sb, KS=KS)
    return qc, info, costs[best]


def build_shot(z, jP, b):
    k = int(sum(z))
    m = n - k
    dec, dinfo, dcost = decoder_for(k, b)
    Rq = QuantumRegister(1, "R")
    Dq = QuantumRegister(n, "d")
    Aq = QuantumRegister(3 * n, "a")
    cl_use = ClassicalRegister(2 * n, "lm")
    cl_flag = ClassicalRegister(max(k, 1), "fz")
    cl_dec = ClassicalRegister(dec.num_clbits, "s")
    cl_fin = ClassicalRegister(2, "bell")
    qc = QuantumCircuit(Rq, Dq, Aq, cl_use, cl_flag, cl_dec, cl_fin, name=f"shot_n{n}_k{k}")
    qc.h(Rq[0]); qc.cx(Rq[0], Dq[n - 1])                      # reference Bell pair with the input
    qc.compose(enc, qubits=list(Dq), inplace=True)
    for j in range(n):
        blk = C.block_randomized(int(jP[j]), int(z[j]), decoder="opt")
        qc.compose(blk, qubits=[Dq[j], Aq[3 * j], Aq[3 * j + 1], Aq[3 * j + 2]],
                   clbits=[cl_use[2 * j], cl_use[2 * j + 1]], inplace=True)
    flagged = [j for j in range(n) if z[j]]
    unflagged = [j for j in range(n) if not z[j]]
    for t, j in enumerate(flagged):
        qc.measure(Dq[j], cl_flag[t])
    # decoder: data = unflagged qubits in ascending position, workspace = reset ancillas
    nwork = dec.num_qubits - m
    work = [Aq[i] for i in range(nwork)]
    qc.compose(dec, qubits=[Dq[j] for j in unflagged] + work, clbits=list(cl_dec), inplace=True)
    out = work[dinfo["out_qubit"] - m]
    qc.cx(Rq[0], out); qc.h(Rq[0])
    qc.measure(Rq[0], cl_fin[0]); qc.measure(out, cl_fin[1])
    return qc, dcost


def counts(qc):
    c = L.count_ops(qc)
    return {"cz": c.get("cz", 0) + c.get("cond:cz", 0), "measure": c.get("measure", 0) + c.get("cond:measure", 0),
            "reset": c.get("reset", 0) + c.get("cond:reset", 0), "feedforward_blocks": c.get("feedforward_blocks", 0)}


def depth2q(qc):
    return qc.depth(lambda i: i.operation.num_qubits == 2)


report = {"n": n, "samples": []}
t0 = time.time()
backend = None
if ROUTE:
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    backend = FakeKingston()

# ---- sub-blocks alone (logical and routed)
sub = {}
encc = L.compile_cz(enc)
sub["encoder"] = {"logical": counts(encc), "depth": encc.depth(), "depth_2q": depth2q(encc), "qubits": n}
blk = C.block_randomized(0, 0, decoder="opt")
blkc = L.compile_cz(blk)
sub["use_block_P1_noerase"] = {"logical": counts(blkc), "depth": blkc.depth(), "qubits": 4}
kmode = n // 2
dec, dinfo, dcost = decoder_for(kmode, kmode // 2)
decc = L.compile_cz(dec)
sub[f"decoder_k{kmode}"] = {"logical": counts(decc), "depth": decc.depth(), "depth_2q": depth2q(decc),
                            "qubits": dec.num_qubits, "formula_cz": dcost["total_cz"]}
if ROUTE:
    for name, circ in (("encoder", enc), ("use_block_P1_noerase", blk), (f"decoder_k{kmode}", dec)):
        lay = hh_path(backend.coupling_map, n)[0] if (LNN and name == "encoder") else None
        tr = transpile(circ, backend=backend, optimization_level=3, seed_transpiler=11, approximation_degree=1.0,
                       initial_layout=lay)
        sub[name]["routed_heavyhex"] = {"cz": counts(tr)["cz"], "depth": tr.depth(), "depth_2q": depth2q(tr)}
        print(f"{name}: logical CZ {sub[name]['logical']['cz']} -> heavy-hex CZ {sub[name]['routed_heavyhex']['cz']}"
              f" (depth {tr.depth()})  ({time.time() - t0:.0f}s)", flush=True)
report["subblocks"] = sub

# ---- whole shots
for sidx in range(SAMPLES):
    z = (rng.random(n) < 0.5).astype(int)
    jP = rng.choice(6, size=n, p=np.array(C.P_UNITARY_WEIGHTS) / sum(C.P_UNITARY_WEIGHTS))
    k = int(z.sum())
    R = sa.bob_states(Xiso, n, k)
    pb = np.array([max(float(np.real(np.trace(R[bb]))) / 2, 0.0) for bb in range(k + 1)])
    b = int(rng.choice(k + 1, p=pb / pb.sum()))
    qc, dcost = build_shot(z, jP, b)
    qcl = L.compile_cz(qc)
    rec = {"z": z.tolist(), "k": k, "b": b, "jP": jP.tolist(), "qubits_logical": qc.num_qubits,
           "logical": counts(qcl), "depth": qcl.depth(), "depth_2q": depth2q(qcl), "decoder_cost": dcost}
    if ROUTE:
        t1 = time.time()
        lay = make_layout(backend.coupling_map, n) if LNN else None
        tr = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=11, approximation_degree=1.0,
                       initial_layout=lay)
        rec["routed_heavyhex"] = {"cz": counts(tr)["cz"], "depth": tr.depth(), "depth_2q": depth2q(tr),
                                  "seconds": round(time.time() - t1, 1)}
    report["samples"].append(rec)
    print(f"sample {sidx}: k={k} b={b}  logical CZ {rec['logical']['cz']}  meas {rec['logical']['measure']}"
          f"  depth {rec['depth']} (2q {rec['depth_2q']})"
          + (f"  -> heavy-hex CZ {rec['routed_heavyhex']['cz']} depth {rec['routed_heavyhex']['depth']}" if ROUTE else "")
          + f"  ({time.time() - t0:.0f}s)", flush=True)

with open(f"full_run_n{n}" + ("_lnn" if LNN else "") + ".json", "w") as fh:
    json.dump(report, fh, indent=1, default=str)
