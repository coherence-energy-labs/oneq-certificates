"""The two-use superactivation protocol as hardware circuits: exact check, gate counts, and a noise budget.

P_d (Sylvester u, d = 2^m shield qubits) as RANDOMIZED Clifford(+T) circuits, verified against its Choi state:
  sector 0 (prob 1 - p, uniform over the 4^m shield Paulis W_g):
      V_g = prod_{t: g_t in {I, Z}} CZ(key, s_t) . Z_key^{#Y in g} . W_g
  sector 1 (prob p, uniform over c in {0,1}^m; from a Y-basis unitary mixing of the d Kraus operators):
      V_c = X_key . prod_t CZ(key, s_t) . P_key(sum_t +-pi/4) . (x)_t S^{(c_t)},   S^{(0)} = S, S^{(1)} = S^dag
A (50% erasure): a classical coin per use; erased copies are discarded (Bob never touches them).
Encoder: |0_L> -> |++>, |1_L> -> |--> on the two keys; per use a Bell pair (s_t, s'_t) on every shield qubit and a
CNOT key -> key copy.
Decoder, MEASURE AND CORRECT (measured_decoder.py: exact):
  kept use: CNOT key -> copy, measure the copy -> l. Per pair: CNOT(s' -> s), CH(s -> s'), then (l = 1 only)
            Ry(-pi/4) on s'; measure. The eigenvalue is (-1)^{s'}, and the key's Pauli frame is X^l Z^[product = -1].
  erased use: measure the key in Z (the Helstrom measurement).
  The 2-qubit code is decoded by Clifford gates and Pauli frames in software. The l-dependent basis is realized by a
  RANDOMIZED basis with classical selection: an unbiased estimate of the adaptive decoder, so no feed-forward.

  python hardware_design.py check        (m = 1, 2) P's decomposition = its Choi; protocol F_e = (7-6p+2p^2)/8 exactly
  python hardware_design.py count M..    heavy-hex (FakeKingston) and all-to-all CZ counts per erasure pattern
  python hardware_design.py budget M..   predicted F_e under a per-CZ error model; the error rate the 3/4 bar needs
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ppt_family as pf  # noqa: E402

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1])}


# ---------------------------------------------------------------- P_d as randomized circuits

def p_unitaries(m: int):
    """[(weight, label, matrix on key (x) shield)] with the key first and shield qubit 0 most significant"""
    d = 2 ** m
    p = 1 / (math.sqrt(d) + 1)
    out = []
    Z1 = np.diag([0.0, 1.0])
    for g in itertools.product("IXYZ", repeat=m):
        W = np.array([[1.0]])
        for s in g:
            W = np.kron(W, PAULI[s])
        C = np.kron(np.diag([1.0, 0.0]), np.eye(d)) + np.kron(Z1, _zs([s in "IZ" for s in g]))
        keyz = np.kron(PAULI["Z"] if g.count("Y") % 2 else np.eye(2), np.eye(d))
        out.append(((1 - p) / d ** 2, ("s0", g), C @ keyz @ np.kron(np.eye(2), W)))
    for c in itertools.product((0, 1), repeat=m):
        S = np.array([[1.0]])
        phase = 0.0
        for ct in c:
            S = np.kron(S, np.diag([1, 1j]) if ct == 0 else np.diag([1, -1j]))
            phase += math.pi / 4 if ct == 0 else -math.pi / 4
        C = np.kron(np.diag([1.0, 0.0]), np.eye(d)) + np.kron(Z1, np.exp(1j * phase) * _zs([True] * m))
        out.append((p / d, ("s1", c), np.kron(PAULI["X"], np.eye(d)) @ C @ np.kron(np.eye(2), S)))
    return out


def _zs(mask):
    M = np.array([[1.0]])
    for on in mask:
        M = np.kron(M, PAULI["Z"] if on else np.eye(2))
    return M


def check_p_decomposition(m: int) -> float:
    d = 2 ** m
    D = 2 * d
    J = 0
    for w, _, U in p_unitaries(m):
        vec = np.zeros(D * D, dtype=complex)
        for i in range(D):
            vec[i * D:(i + 1) * D] = U[:, i]
        J = J + w * np.outer(vec, vec.conj())
    return float(np.abs(J / D - pf.choi_state(pf.sylvester(m), 1 / (math.sqrt(d) + 1))).max())


# ---------------------------------------------------------------- the protocol, simulated exactly (small m)

class Proto:
    """qubit indices: R (reference, entangled with the logical input) = 0; use u has key k, shield s[0..m-1] (into P),
    copy c, shield copy a[0..m-1] (into A)"""

    def __init__(self, m):
        self.m = m
        self.n = 1 + 2 * (2 * m + 2)

    def base(self, u):
        return 1 + u * (2 * self.m + 2)

    def k(self, u):
        return self.base(u)

    def s(self, u, t):
        return self.base(u) + 1 + t

    def c(self, u):
        return self.base(u) + 1 + self.m

    def a(self, u, t):
        return self.base(u) + 2 + self.m + t


def build(m, z, pchoice, beta):
    """the protocol up to Bob's final Z-basis read-out (read from the statevector): R and the logical input start in
    a Bell pair, so the result is the ENTANGLEMENT fidelity directly"""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    L = Proto(m)
    qc = QuantumCircuit(L.n)
    qc.h(0)
    qc.cx(0, L.k(0))
    qc.cx(L.k(0), L.k(1))
    qc.h(L.k(0))
    qc.h(L.k(1))
    for u in range(2):
        for t in range(m):
            qc.h(L.a(u, t))
            qc.cx(L.a(u, t), L.s(u, t))
        qc.cx(L.k(u), L.c(u))
        U = pchoice[u]                                     # P on (key, s[0..m-1]); key most significant
        qc.append(UnitaryGate(U), [L.s(u, t) for t in reversed(range(m))] + [L.k(u)])
    for u in range(2):
        if z[u]:
            continue
        qc.cx(L.k(u), L.c(u))
        for t in range(m):
            qc.cx(L.a(u, t), L.s(u, t))
            qc.ch(L.s(u, t), L.a(u, t))
            if beta[u]:
                qc.ry(-math.pi / 4, L.a(u, t))
    return qc


H2 = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
CNOT2 = np.eye(4)[[0, 1, 3, 2]]
PHI = np.array([1, 0, 0, 1]) / math.sqrt(2)


def branch_fidelity(m, z, psi, beta) -> float:
    """sum over Bob's outcomes (selected by the random bases beta) of P(outcome) x F_e(outcome), for one statevector.
    Measured: copies, shield pairs of kept uses and keys of erased uses. Kept: R and kept keys. Traced: the rest."""
    L = Proto(m)
    n = L.n
    T = psi.reshape([2] * n)[tuple([slice(None)] * n)]
    T = np.moveaxis(T, list(range(n)), list(reversed(range(n))))    # axis q <-> qubit q (qiskit is little-endian)
    kept = [u for u in range(2) if not z[u]]
    meas = []
    for u in range(2):
        if z[u]:
            meas.append(L.k(u))
        else:
            meas += [L.c(u)] + [L.s(u, t) for t in range(m)] + [L.a(u, t) for t in range(m)]
    keep = [0] + [L.k(u) for u in kept]
    rest = [q for q in range(n) if q not in meas and q not in keep]
    A = np.transpose(T, meas + keep + rest).reshape(2 ** len(meas), 2 ** len(keep), 2 ** len(rest))
    rho = np.einsum("xkt,xlt->xkl", A, A.conj())                    # unnormalized conditional states on R + keys
    total = 0.0
    for x in range(2 ** len(meas)):
        pr = float(np.real(np.trace(rho[x])))
        if pr < 1e-15:
            continue
        bits = {q: (x >> (len(meas) - 1 - i)) & 1 for i, q in enumerate(meas)}
        frames, ok = [], True
        for u in kept:
            ell = bits[L.c(u)]
            if ell != beta[u]:
                ok = False
                break
            prod = 1
            for t in range(m):
                prod *= (-1) ** bits[L.a(u, t)]
            F = (PAULI["X"] if ell else np.eye(2)) @ (PAULI["Z"] if prod < 0 else np.eye(2))
            frames.append(F)
        if not ok:
            continue
        r = rho[x] / pr
        if z == (0, 0):
            U = np.kron(np.eye(2), CNOT2 @ np.kron(H2, H2) @ np.kron(frames[0], frames[1]))
            rL = _trace_last(U @ r @ U.conj().T)                     # the second key returns to |0>: trace it
        elif z in ((0, 1), (1, 0)):
            u_er = 1 if z == (0, 1) else 0
            b = bits[L.k(u_er)]
            U = np.kron(np.eye(2), H2 @ (PAULI["X"] if b else np.eye(2)) @ frames[0])
            rL = U @ r @ U.conj().T
        else:
            par = bits[L.k(0)] ^ bits[L.k(1)]
            out = np.array([1, 1 if par == 0 else -1]) / math.sqrt(2)
            rL = np.kron(r, np.outer(out, out))
        total += pr * float(np.real(PHI @ rL @ PHI))
    return total


def _trace_last(r):
    """trace out the last qubit of a 3-qubit (R, k0, k1) density matrix"""
    return np.einsum("aibj->ab", r.reshape(4, 2, 4, 2))


def simulate(m: int):
    """exact protocol entanglement fidelity: P enumerated with its weights, random bases enumerated (selection makes
    each l use its own basis), erasure patterns weighted 1/4"""
    from qiskit.quantum_info import Statevector
    Ps = p_unitaries(m)
    Fz = {}
    for z in itertools.product((0, 1), repeat=2):
        acc = 0.0
        for (w0, _, U0), (w1, _, U1) in itertools.product(Ps, Ps):
            for beta in itertools.product((0, 1), repeat=2):
                if any(z[u] and beta[u] for u in range(2)):
                    continue
                psi = Statevector(build(m, z, (U0, U1), beta)).data
                acc += w0 * w1 * branch_fidelity(m, z, psi, beta)
        Fz[z] = acc
    return sum(Fz.values()) / 4, Fz


# ---------------------------------------------------------------- hardware circuits: explicit gates, counts, budget

def apply_p(qc, L, u, choice):
    """one sampled unitary of P_d on (key, shield) as explicit gates (the decomposition check_p_decomposition proves)"""
    kind, lab = choice
    k = L.k(u)
    if kind == "s0":
        for t, g in enumerate(lab):
            if g != "I":
                getattr(qc, g.lower())(L.s(u, t))
        if lab.count("Y") % 2:
            qc.z(k)
        for t, g in enumerate(lab):
            if g in "IZ":
                qc.cz(k, L.s(u, t))
    else:
        phase = 0.0
        for t, ct in enumerate(lab):
            (qc.s if ct == 0 else qc.sdg)(L.s(u, t))
            phase += math.pi / 4 if ct == 0 else -math.pi / 4
            qc.cz(k, L.s(u, t))
        qc.p(phase, k)
        qc.x(k)


def protocol_circuit(m, z, choices, beta, basis="Z", rng=None):
    """the experiment's circuit for one erasure pattern z, sampled P unitaries, random decoder bases beta and a
    logical measurement basis. Erased uses use the equivalent cheap preparation (random key Z, random shield)."""
    from qiskit import QuantumCircuit
    L = Proto(m)
    rng = rng or np.random.default_rng(0)
    qc = QuantumCircuit(L.n - 1 + 1, L.n)
    q = lambda i: i                                          # noqa: E731
    qc.h(q(0))
    qc.cx(q(0), L.k(0))                                      # the reference R stands in for the 6 input preparations
    qc.cx(L.k(0), L.k(1))
    qc.h(L.k(0))
    qc.h(L.k(1))
    for u in range(2):
        if z[u]:
            if rng.integers(2):
                qc.z(L.k(u))
            for t in range(m):
                if rng.integers(2):
                    qc.x(L.s(u, t))
        else:
            for t in range(m):
                qc.h(L.a(u, t))
                qc.cx(L.a(u, t), L.s(u, t))
            qc.cx(L.k(u), L.c(u))
        apply_p(qc, L, u, choices[u])
    for u in range(2):
        if z[u]:
            continue
        qc.cx(L.k(u), L.c(u))
        for t in range(m):
            qc.cx(L.a(u, t), L.s(u, t))
            qc.ch(L.s(u, t), L.a(u, t))
            if beta[u]:
                qc.ry(-math.pi / 4, L.a(u, t))
    if basis == "X":
        for u in range(2):
            qc.h(L.k(u))
    qc.measure(range(L.n), range(L.n))
    return qc


def check_apply_p(m: int) -> float:
    """max over every choice of |explicit gates - verified matrix| up to a global phase (apply_p vs p_unitaries)"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    L = Proto(m)
    worst = 0.0
    for _, choice, U in p_unitaries(m):
        qc = QuantumCircuit(L.n)
        apply_p(qc, L, 0, choice)
        qs = [L.s(0, t) for t in reversed(range(m))] + [L.k(0)]
        sub = QuantumCircuit(m + 1)
        for inst in qc.data:
            sub.append(inst.operation, [qs.index(qc.find_bit(q).index) for q in inst.qubits])
        M = Operator(sub).data
        ph = np.vdot(M.ravel(), U.ravel())
        ph = ph / abs(ph)
        worst = max(worst, float(np.abs(M - ph * U).max()))
    return worst


def protocol_gates(m, z, choices, beta, rng):
    """protocol_circuit without the read-out (for statevector simulation with injected noise)"""
    qc = protocol_circuit(m, z, choices, beta, rng=rng)
    qc.remove_final_measurements(inplace=True)
    return qc


def noisy_fidelity(m, e, trajectories, seed=11):
    """Monte Carlo: P and the decoder bases sampled, a uniformly random non-identity two-qubit Pauli after every
    two-qubit gate with probability e, Bob's exact decoder on every trajectory. Returns F_e and per-pattern means."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    paulis2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]
    Fz = {}
    for z in ((0, 0), (0, 1), (1, 1)):
        vals = []
        L = Proto(m)
        for _ in range(trajectories):
            ch = (sample_choice(m, rng), sample_choice(m, rng))
            base = protocol_gates(m, z, ch, (0, 0), rng)       # bases appended after the noise, below
            qc = QuantumCircuit(base.num_qubits)
            for inst in base.data:
                qc.append(inst.operation, inst.qubits)
                if inst.operation.num_qubits == 2 and rng.random() < e:
                    pa, pb = paulis2[int(rng.integers(len(paulis2)))]
                    for g, qb in ((pa, inst.qubits[0]), (pb, inst.qubits[1])):
                        if g != "I":
                            getattr(qc, g.lower())(qb)
            # the SAME noise realization under every basis choice; selection inside branch_fidelity keeps, for each
            # outcome l, exactly the basis the adaptive decoder would use -- no discarded half, no 2^k factor
            v = 0.0
            for beta in itertools.product((0, 1), repeat=2):
                if any(z[u] and beta[u] for u in range(2)):
                    continue
                qb = qc.copy()
                for u in range(2):
                    if not z[u] and beta[u]:
                        for t in range(m):
                            qb.ry(-math.pi / 4, L.a(u, t))  # single-qubit, after the pair's last two-qubit gate
                v += branch_fidelity(m, z, Statevector(qb).data, beta)
            vals.append(v)
        Fz[z] = (float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals))))
    F = 0.25 * (Fz[(0, 0)][0] + 2 * Fz[(0, 1)][0] + Fz[(1, 1)][0])
    se = 0.25 * math.sqrt(Fz[(0, 0)][1] ** 2 + 4 * Fz[(0, 1)][1] ** 2 + Fz[(1, 1)][1] ** 2)
    return F, se, Fz


def two_qubit_gates(m, z, rng):
    qc = protocol_gates(m, z, (sample_choice(m, rng), sample_choice(m, rng)), (0, 0), rng)
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def sample_choice(m, rng):
    d = 2 ** m
    p = 1 / (math.sqrt(d) + 1)
    if rng.random() < p:
        return ("s1", tuple(int(x) for x in rng.integers(2, size=m)))
    return ("s0", tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m)))


def count(m, samples=12, seed=7):
    """mean CZ per erasure pattern: logical (all-to-all) and heavy-hex routed (FakeKingston, Sabre, level 3)"""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    rng = np.random.default_rng(seed)
    hh = FakeKingston()
    out = {}
    for z in ((0, 0), (0, 1), (1, 1)):
        a2a, routed = [], []
        for _ in range(samples):
            ch = (sample_choice(m, rng), sample_choice(m, rng))
            beta = tuple(int(x) for x in rng.integers(2, size=2))
            qc = protocol_circuit(m, z, ch, beta, rng=rng)
            n = qc.num_qubits
            full = transpile(qc, basis_gates=["cz", "rz", "sx", "x", "measure"],
                             coupling_map=CouplingMap.from_full(n), optimization_level=3, seed_transpiler=1)
            a2a.append(full.count_ops().get("cz", 0))
            hw = transpile(qc, backend=hh, optimization_level=3, seed_transpiler=1)
            routed.append(hw.count_ops().get("cz", 0))
        out[z] = (float(np.mean(a2a)), float(np.mean(routed)))
    return out


def budget(Fz_ideal, ncz, e):
    """per-branch depolarizing-equivalent model: F_z -> (1-e)^N F_z + (1-(1-e)^N)/4 (a fully depolarized logical qubit
    has entanglement fidelity 1/4); patterns weighted 1/4 (01 and 10 cost the same)"""
    tot = 0.0
    for z, F in Fz_ideal.items():
        key = z if z != (1, 0) else (0, 1)
        keep = (1 - e) ** ncz[key]
        tot += 0.25 * (keep * F + (1 - keep) / 4)
    return tot


def ideal_branches(p):
    return {(0, 0): 1.0, (0, 1): 1 - p, (1, 0): 1 - p, (1, 1): 0.5 - p * (1 - p)}


if __name__ == "__main__":
    if sys.argv[1] == "noisy":
        m, T = int(sys.argv[2]), int(sys.argv[3])
        d = 2 ** m
        p = 1 / (math.sqrt(d) + 1)
        print(f"explicit P gates vs the verified matrices, every choice: max |diff| {check_apply_p(m):.2e}", flush=True)
        rng = np.random.default_rng(3)
        ncz = {z: float(np.mean([two_qubit_gates(m, z, rng) for _ in range(40)])) for z in ((0, 0), (0, 1), (1, 1))}
        for e in (0.0, 0.01, 0.03):
            F, se, Fz = noisy_fidelity(m, e, T)
            model = budget(ideal_branches(p), ncz, e)
            print(f"m = {m}, e = {e:.3f}: noisy simulation F_e = {F:.4f} +- {se:.4f}   budget model {model:.4f}   "
                  f"(ideal {(7 - 6 * p + 2 * p * p) / 8:.4f}; mean 2q gates {ncz})", flush=True)
        sys.exit(0)
    if sys.argv[1] in ("count", "budget"):
        import json
        ms = tuple(int(x) for x in sys.argv[2].split(","))
        rows = {}
        for m in ms:
            d = 2 ** m
            p = 1 / (math.sqrt(d) + 1)
            c = count(m)
            Fid = ideal_branches(p)
            F0 = sum(Fid.values()) / 4
            row = {"m": m, "d": d, "p": p, "F_ideal": F0, "cz": {str(k): v for k, v in c.items()}}
            for which, idx in (("all_to_all", 0), ("heavy_hex", 1)):
                ncz = {k: v[idx] for k, v in c.items()}
                lo, hi = 0.0, 0.05                            # the per-CZ error at which F_e falls to 3/4
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if budget(Fid, ncz, mid) > 0.75 else (lo, mid)
                row[f"{which}_error_needed"] = lo
                row[f"{which}_F_at_0.006"] = budget(Fid, ncz, 0.006)
                row[f"{which}_F_at_0.001"] = budget(Fid, ncz, 0.001)
            rows[m] = row
            print(f"m = {m} (d = {d}): ideal F_e {F0:.4f}; CZ per pattern (all-to-all, heavy-hex): "
                  f"{ {str(k): (round(v[0], 1), round(v[1], 1)) for k, v in c.items()} }; per-CZ error for F_e = 3/4: "
                  f"all-to-all {row['all_to_all_error_needed']:.4%}, heavy-hex {row['heavy_hex_error_needed']:.4%}; "
                  f"F_e at 0.6%/CZ: {row['heavy_hex_F_at_0.006']:.4f} (hh), at 0.1%: {row['all_to_all_F_at_0.001']:.4f} "
                  f"(a2a)", flush=True)
        out = HERE.parents[2] / "evidence" / "superactivation" / "shield_family" / "hardware_budget.json"
        merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        merged.update({str(k): v for k, v in rows.items()})
        out.write_bytes((json.dumps(merged, indent=1) + "\n").encode("utf-8"))
        sys.exit(0)
    if sys.argv[1] == "check":
        for m in (1, 2, 3):
            print(f"m = {m}: P_d = randomized Clifford(+T) circuits, max |Choi diff| = {check_p_decomposition(m):.2e}")
        for m in (1,) if len(sys.argv) < 3 else tuple(int(x) for x in sys.argv[2].split(",")):
            p = 1 / (math.sqrt(2 ** m) + 1)
            F, Fz = simulate(m)
            print(f"m = {m}: PROTOCOL F_e = {F:.12f}   closed form (7-6p+2p^2)/8 = {(7 - 6 * p + 2 * p * p) / 8:.12f}   "
                  f"branches {[round(v, 10) for v in Fz.values()]}", flush=True)
        sys.exit(0)
    raise SystemExit(__doc__)
