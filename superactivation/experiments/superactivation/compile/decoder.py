"""Deliverable 4: the per-branch decoders D_{k,b}: Sym^{n-k} -> qubit, as dynamic circuits.

Pipeline for branch (k, b), m = n - k unflagged qubits (positions 1..m):
  1. U_m^dag (inverse BE Dicke unitary): Sym^m -> unary levels w = 0..m on the same m qubits.
  2. [optional] support compression: inverse unary Givens network mapping an orthonormal basis of
     the support S of Bob's state (dim s) onto unary levels 0..s-1 -> occupies positions m-s+2..m,
     positions 1..m-s+1 return to |0>.  (route 'b'; route 'a' skips it and keeps all m+1 levels)
  3. unary -> binary register B (qb = ceil(log2 L) qubits, L = levels kept): CX t_i -> b_j for
     every i divisible by 2^j (bit j of w is the parity of the thermometer bits t_{i*2^j});
     then MEASURE every thermometer qubit in X (outcome s_i) and undo the kicked-back phase
     (-1)^{sum_i s_i [w >= i]} on B with a fixed Gray-code CNOT skeleton (2^qb - 2 CZ) whose Rz
     angles are switched on by feed-forward (single-qubit conditional gates only).
  4. the decoder's Stinespring isometry on (B, out [, anc]) synthesized by qiskit's Isometry
     (column-by-column, Iten et al.); B (and anc) are discarded, `out` is the decoded qubit.
Flagged qubits are read in Z (b = Hamming weight) before this; that is k measurements.
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Isometry, UnitaryGate

import dicke as Dk
import sclib as L


def nbits(L_):
    return max(1, int(math.ceil(math.log2(L_)))) if L_ > 1 else 0


def support_basis(rho_sym: np.ndarray, rel=1e-12):
    """Orthonormal basis (columns) of the support of Tr_R rho on Sym^m, eigen-threshold `rel`."""
    w, U = np.linalg.eigh((rho_sym + rho_sym.conj().T) / 2)
    keep = w > rel * max(w.max(), 1e-300)
    return U[:, keep][:, ::-1], w


def restricted_kraus(Ks, Sb, rel=1e-13):
    """Kraus of the decoder restricted to S and expressed in the S basis (columns of Sb):
    K_S = K Sb (2 x s). Re-orthogonalized: the restricted map is TP on S because Tr_R D = I.
    Rank reduced to the minimum via the adjoint Choi in the S basis."""
    s = Sb.shape[1]
    D = np.zeros((2 * s, 2 * s), dtype=complex)
    for K in Ks:
        KS = K @ Sb
        wv = np.conj(KS).reshape(-1)
        D += np.outer(wv, wv.conj())
    w, U = np.linalg.eigh((D + D.conj().T) / 2)
    out = []
    for lam, vec in zip(w[::-1], U[:, ::-1].T):
        if lam <= rel * w[-1]:
            break
        out.append(np.conj(np.sqrt(lam) * vec.reshape(2, s)))
    return out


def min_rank_kraus(KS, rho_S, max_iter=200):
    """Exact Kraus-rank reduction at fixed fidelity (Barvinok-Pataki): with D = Y Y^dag the
    adjoint-Choi of the restricted decoder on R (x) S (s = dim S), move along D(t) = Y (I + tH) Y^dag
    with H in the null space of  {H -> Tr_R(Y H Y^dag)} (keeps Tr_R D = I_S, i.e. trace preservation)
    and {H -> Tr(rho Y H Y^dag)} (keeps the branch fidelity), until an eigenvalue of I + tH hits 0.
    Terminates with rank r, r^2 <= s^2 + 1, i.e. r <= s. Returns the new Kraus list (2 x s)."""
    s = KS[0].shape[1]
    Y = np.stack([np.conj(K).reshape(-1) for K in KS], axis=1)       # 2s x r, columns w_e
    for _ in range(max_iter):
        r = Y.shape[1]
        if r * r <= s * s + 1:
            break
        # real basis of r x r Hermitian matrices
        basis = []
        for i in range(r):
            E = np.zeros((r, r), dtype=complex); E[i, i] = 1; basis.append(E)
        for i in range(r):
            for j in range(i + 1, r):
                E = np.zeros((r, r), dtype=complex); E[i, j] = E[j, i] = 1 / math.sqrt(2); basis.append(E)
                E = np.zeros((r, r), dtype=complex); E[i, j] = 1j / math.sqrt(2); E[j, i] = -1j / math.sqrt(2); basis.append(E)
        cols = []
        for E in basis:
            Dd = Y @ E @ Y.conj().T
            T = np.einsum("rirj->ij", Dd.reshape(2, s, 2, s))
            f = np.trace(rho_S @ Dd)
            cols.append(np.concatenate([T.real.reshape(-1), T.imag.reshape(-1), [f.real]]))
        A = np.array(cols).T
        _, sv, Vh = np.linalg.svd(A)
        null = Vh[-1]
        if sv.size == len(basis) and sv[-1] > 1e-9 * max(sv[0], 1.0):
            break
        H = sum(c * E for c, E in zip(null, basis))
        w, U = np.linalg.eigh(H)
        if w[0] >= 0:
            H, w, U = -H, -w[::-1], U[:, ::-1]
        t = -1.0 / w[0]
        M = np.eye(r) + t * H
        wm, Um = np.linalg.eigh((M + M.conj().T) / 2)
        keep = wm > 1e-12 * wm.max()
        Y = Y @ (Um[:, keep] * np.sqrt(wm[keep]))
    return [np.conj(Y[:, e]).reshape(2, s) for e in range(Y.shape[1])]


def stinespring_columns(Ks, n_in_levels, q_in, n_anc):
    """Isometry matrix (2^(q_in+1+n_anc) x 2^q_in) on little-endian qubits [B (q_in), out, anc...]:
    column w (< n_in_levels) = sum_e sum_r K_e[r, w] |e_low>_B |r>_out |e_high>_anc, e = e_low + 2^q_in e_high.
    Columns >= n_in_levels are an arbitrary orthonormal completion."""
    dB = 2 ** q_in
    dim = 2 ** (q_in + 1 + n_anc)
    if len(Ks) > dB * 2 ** n_anc:
        raise ValueError(f"Kraus rank {len(Ks)} exceeds environment {dB * 2 ** n_anc}")
    V = np.zeros((dim, dB), dtype=complex)
    for e, K in enumerate(Ks):
        el, eh = e % dB, e // dB
        for r in range(2):
            idx = el + dB * r + 2 * dB * eh
            V[idx, :n_in_levels] += K[r, :n_in_levels]
    # orthonormal completion of the unused columns
    used = V[:, :n_in_levels]
    G = used.conj().T @ used
    if np.abs(G - np.eye(n_in_levels)).max() > 1e-9:
        raise ValueError(f"decoder Stinespring not isometric: {np.abs(G - np.eye(n_in_levels)).max():.2e}")
    if n_in_levels < dB:
        rng = np.random.default_rng(5)
        R = rng.normal(size=(dim, dB - n_in_levels)) + 1j * rng.normal(size=(dim, dB - n_in_levels))
        R -= used @ (used.conj().T @ R)
        Q, _ = np.linalg.qr(R)
        V[:, n_in_levels:] = Q
    return V


def phase_fix_angles(m_therm: int, qb: int):
    """theta[i][S] for the diagonal C_i = diag(exp(i pi [w >= i])) over w in [0, 2^qb):
    C_i = prod_S exp(i theta_S Z_S) up to a global phase; returns dict S(bitmask) -> list over i."""
    dB = 2 ** qb
    W = np.array([[(-1) ** bin(S & w).count("1") for w in range(dB)] for S in range(dB)])
    tab = {}
    for i in range(1, m_therm + 1):
        Th = np.array([math.pi * (w >= i) for w in range(dB)])
        coef = W @ Th / dB     # Th(w) = sum_S coef_S (-1)^{S.w}
        for S in range(1, dB):
            if abs(coef[S]) > 1e-12:
                tab.setdefault(S, []).append((i, float(coef[S])))
    return tab


def append_phase_fix(qc, Bq, s_bits, m_therm, therm_index_of_bit):
    """Gray-code skeleton: for each target t, cycle the parities of subsets {t} u T, T subset of
    {0..t-1}; at each parity apply rz(-2 theta) conditioned on each thermometer outcome s_i."""
    qb = len(Bq)
    tab = phase_fix_angles(m_therm, qb)
    nff = 0

    def rot(S, tgt):
        nonlocal nff
        for (i, th) in tab.get(S, []):
            with qc.if_test((s_bits[therm_index_of_bit[i]], 1)):
                qc.rz(-2 * th, tgt)
            nff += 1

    for t in range(qb):
        rot(1 << t, Bq[t])
        if t == 0:
            continue
        # Gray code over T subset {0..t-1}
        prev = 0
        for g in range(1, 2 ** t):
            gray = g ^ (g >> 1)
            flip = (gray ^ prev).bit_length() - 1
            qc.cx(Bq[flip], Bq[t])
            prev = gray
            rot((1 << t) | gray, Bq[t])
        # close the cycle back to T = empty
        qc.cx(Bq[(prev).bit_length() - 1], Bq[t])
    return nff


_EXACT_CACHE: dict = {}


def exact_completion(V: np.ndarray, tol: float = 1e-12, tries: int = 30):
    """A unitary completion U of V whose qiskit synthesis (compile_cz, level 1) reproduces U on the
    used columns to `tol`. MEASURED (debug_k16.py): qiskit 2.5's two-qubit KAK synthesis returned
    operator errors up to 1.6e-5 for some completions; the unused columns are free, so we re-draw the
    completion until the compiled operator is exact. Returns (U, cz, err)."""
    from qiskit.quantum_info import Operator
    key = (V.shape, V.round(13).tobytes())
    if key in _EXACT_CACHE:
        return _EXACT_CACHE[key]
    nq = int(round(math.log2(V.shape[0])))
    best = None
    for seed in range(11, 11 + tries):
        U = complete_unitary(V, seed)
        qc = QuantumCircuit(nq)
        qc.append(UnitaryGate(U), list(range(nq)))
        cc = L.compile_cz(qc)
        Uc = Operator(cc).data
        c = V.shape[1]
        ph = np.vdot(Uc[:, :c].reshape(-1), V.reshape(-1))
        ph = ph / abs(ph)
        err = float(np.abs(Uc[:, :c] * ph - V).max())
        cz = L.count_ops(cc).get("cz", 0)
        if best is None or err < best[2]:
            best = (U, cz, err)
        if err < tol:
            break
    if best[2] >= tol and nq <= 3:
        fit = fit_isometry_circuit(V, nq)
        if fit is not None and fit[2] < tol:
            best = fit
    _EXACT_CACHE[key] = best
    return best


def fit_isometry_circuit(V: np.ndarray, nq: int, max_cz: int = 6):
    """Fallback for small decoder unitaries where qiskit's KAK synthesis is inaccurate: least-squares
    fit (Levenberg-Marquardt on the amplitude residual) of a [U3 layer](CZ [U3 layer])^c circuit to the
    used columns of V, c increasing; returns (QuantumCircuit, cz, err) or None."""
    import itertools
    from scipy.optimize import least_squares
    import min_cz_search as S
    c_used = V.shape[1]
    pairs = list(itertools.combinations(range(nq), 2))
    rng = np.random.default_rng(5)
    for c in range(0, max_cz + 1):
        for placement in itertools.product(pairs, repeat=c):
            for _ in range(12):
                x0 = rng.uniform(-math.pi, math.pi, 3 * nq * (c + 1) + 1)

                def res(z):
                    U = S.build(z[:-1], nq, placement)
                    r = (U[:, :c_used] - np.exp(1j * z[-1]) * V).reshape(-1)
                    return np.concatenate([r.real, r.imag])

                sol = least_squares(res, x0, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
                err = float(np.abs(res(sol.x)).max())
                if err < 1e-12:
                    qc = QuantumCircuit(nq)
                    prm = sol.x[:-1]
                    for q in range(nq):
                        qc.u(*prm[3 * q:3 * q + 3], q)
                    off = 3 * nq
                    for (a, b) in placement:
                        qc.cz(a, b)
                        for q in range(nq):
                            qc.u(*prm[off + 3 * q: off + 3 * q + 3], q)
                        off += 3 * nq
                    return qc, c, err
    return None


def complete_unitary(V: np.ndarray, seed: int = 11) -> np.ndarray:
    """Unitary U with U[:, :c] = V (V has orthonormal columns)."""
    dim, c = V.shape
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(dim, dim - c)) + 1j * rng.normal(size=(dim, dim - c))
    R -= V @ (V.conj().T @ R)
    Q, _ = np.linalg.qr(R)
    Q -= V @ (V.conj().T @ Q)
    Q, _ = np.linalg.qr(Q)
    return np.hstack([V, Q])


def restricted_columns(KS, Sb, q_in, n_anc):
    """Stinespring columns over the LEVEL basis (m+1 levels) of a decoder that is K_S on the support
    (basis Sb) and an arbitrary isometric completion on its orthogonal complement."""
    Lk, s = Sb.shape
    VS = stinespring_columns(KS, s, q_in, n_anc)[:, :s]          # images of the S basis vectors
    # orthonormal basis of S-perp inside the level space
    Pc = np.eye(Lk) - Sb @ Sb.conj().T
    w, U = np.linalg.eigh((Pc + Pc.conj().T) / 2)
    Sperp = U[:, w > 0.5]
    dim = VS.shape[0]
    rng = np.random.default_rng(3)
    R = rng.normal(size=(dim, Sperp.shape[1])) + 1j * rng.normal(size=(dim, Sperp.shape[1]))
    R -= VS @ (VS.conj().T @ R)
    Vperp, _ = np.linalg.qr(R)
    Vlev = VS @ Sb.conj().T + Vperp @ Sperp.conj().T              # dim x Lk
    out = np.zeros((dim, 2 ** q_in), dtype=complex)
    out[:, :Lk] = Vlev
    if Lk < 2 ** q_in:
        R = rng.normal(size=(dim, 2 ** q_in - Lk)) + 1j * rng.normal(size=(dim, 2 ** q_in - Lk))
        R -= Vlev @ (Vlev.conj().T @ R)
        Q, _ = np.linalg.qr(R)
        out[:, Lk:] = Q
    return out


_QSD_CACHE: dict = {}


def synth_cz(U: np.ndarray) -> int:
    """CZ count of the QSD synthesis (qiskit default unitary synthesis, optimization_level 1)."""
    nq = int(round(math.log2(U.shape[0])))
    if nq == 1:
        return 0
    qc = QuantumCircuit(nq)
    qc.append(UnitaryGate(U), list(range(nq)))
    return L.count_ops(L.compile_cz(qc)).get("cz", 0)


def decoder_cost(m: int, Ks_levels, Sb, mode: str, impl=("givens", "num4"), synth=True, KS=None):
    """Component-wise CZ / measurement / feed-forward count of decoder_circuit(m, Ks, mode, Sb)
    without building it (the sum is checked against compiled circuits in verify_decoder.py).
    The decoder unitary itself IS built and QSD-synthesized (its count depends on the matrix only
    through its size for generic matrices, but we synthesize each one)."""
    if mode in ("restricted", "compressed"):
        Ks = KS if KS is not None else restricted_kraus(Ks_levels, Sb)
        Lk = Sb.shape[1] if mode == "compressed" else m + 1
    else:
        Ks = [np.asarray(K) for K in Ks_levels]
        Lk = m + 1
    qb = nbits(Lk)
    n_anc = 0
    while len(Ks) > 2 ** qb * 2 ** n_anc:
        n_anc += 1
    c = {"m": m, "mode": mode, "levels_kept": Lk, "qb": qb, "n_anc": n_anc, "kraus_rank": len(Ks)}
    c["U_dag_cz"] = Dk.dicke_unitary_cz(m, *impl) if m >= 2 else 0
    c["givens_cz"] = 0
    if mode == "compressed" and m >= 1:
        ops, alphas, resid = Dk.unary_isometry_ops(Sb)
        c["givens_cz"] = Dk.unary_isometry_cz(ops)
    c["unary_to_binary_cz"] = sum(1 for j in range(qb) for i in range(1, Lk) if i % (2 ** j) == 0)
    c["phase_fix_cz"] = (2 ** qb - 2) if qb >= 1 else 0
    tab = phase_fix_angles(Lk - 1, qb) if qb >= 1 else {}
    c["feedforward_1q"] = sum(len(v) for v in tab.values())
    c["x_measurements"] = Lk - 1
    c["dec_unitary_qubits"] = qb + 1 + n_anc
    if synth:
        V = restricted_columns(Ks, Sb, qb, n_anc) if mode == "restricted" else stinespring_columns(Ks, Lk, qb, n_anc)
        U, czU, errU = exact_completion(V)
        c["dec_unitary_cz"] = czU
        c["dec_unitary_synth_err"] = errU
    c["total_cz"] = (c["U_dag_cz"] + c["givens_cz"] + c["unary_to_binary_cz"] + c["phase_fix_cz"]
                     + c.get("dec_unitary_cz", 0))
    c["qubits"] = m + qb + 1 + n_anc
    return c


def decoder_circuit(m: int, Ks_levels, mode="full", Sb=None, impl=("givens", "num4"), n_anc=None, KS=None):
    """Dynamic decoder circuit on m data qubits (qubits 0..m-1 = positions 1..m) for a channel with
    Kraus Ks_levels (2 x (m+1), acting on Dicke levels).
      mode 'full'       : exactly the given decoder on all of Sym^m;
      mode 'restricted' : the decoder restricted to the support S (basis Sb, (m+1) x s), arbitrary
                          isometric completion on S-perp, no compression (keeps all m+1 levels);
      mode 'compressed' : S is first mapped onto unary levels 0..s-1 by a Givens network.
    Returns (qc, info)."""
    if mode in ("restricted", "compressed"):
        assert Sb is not None
        Ks = KS if KS is not None else restricted_kraus(Ks_levels, Sb)
        Lk = Sb.shape[1] if mode == "compressed" else m + 1
    else:
        Ks = [np.asarray(K) for K in Ks_levels]
        Lk = m + 1
    route = "b" if mode == "compressed" else "a"
    qb = nbits(Lk)
    if n_anc is None:
        n_anc = 0
        while len(Ks) > 2 ** qb * 2 ** n_anc:
            n_anc += 1
    nth = Lk - 1                       # thermometer qubits that carry the kept levels
    data = QuantumRegister(m, "d") if m > 0 else None
    Breg = QuantumRegister(qb, "B") if qb > 0 else None
    outr = QuantumRegister(1, "out")
    anc = QuantumRegister(n_anc, "anc") if n_anc > 0 else None
    regs = [r for r in (data, Breg, outr, anc) if r is not None]
    sreg = ClassicalRegister(max(nth, 1), "s")
    qc = QuantumCircuit(*regs, sreg, name=f"dec_m{m}_{route}")
    qs = list(data) if data is not None else []
    info = {"m": m, "route": route, "levels_kept": Lk, "qb": qb, "n_anc": n_anc, "kraus_rank": len(Ks)}
    # 1. U_m^dag
    if m >= 2:
        u = QuantumCircuit(m)
        Dk.dicke_unitary(u, list(range(m)), *impl)
        qc.compose(u.inverse(), qubits=qs, inplace=True)
    info["U_dag_cz"] = Dk.dicke_unitary_cz(m, *impl) if m >= 2 else 0
    # 2. support compression (inverse of the unary isometry preparing Sb from levels 0..s-1)
    info["givens_cz"] = 0
    if route == "b" and m >= 1:
        ops, alphas, resid = Dk.unary_isometry_ops(Sb)
        g = QuantumCircuit(m)
        Dk.apply_unary_isometry(g, list(range(m)), ops, alphas, Sb.shape[1])
        qc.compose(g.inverse(), qubits=qs, inplace=True)
        info["givens_cz"] = Dk.unary_isometry_cz(ops)
        info["givens_resid"] = resid
    # thermometer bit t_i (i = 1..Lk-1) lives at position m-i+1 -> qubit index m-i
    therm = {i: qs[m - i] for i in range(1, Lk)}
    # 3. unary -> binary
    cx_ub = 0
    if qb > 0:
        for j in range(qb):
            for i in range(1, Lk):
                if i % (2 ** j) == 0:
                    qc.cx(therm[i], Breg[j])
                    cx_ub += 1
        for i in range(1, Lk):
            qc.h(therm[i])
            qc.measure(therm[i], sreg[i - 1])
        nff = append_phase_fix(qc, list(Breg), sreg, Lk - 1, {i: i - 1 for i in range(1, Lk)})
    else:
        nff = 0
    info["unary_to_binary_cz"] = cx_ub
    info["phase_fix_cz"] = (2 ** qb - 2) if qb >= 1 else 0
    info["x_measurements"] = Lk - 1
    info["feedforward_1q"] = nff
    # 4. decoder Stinespring, completed to a unitary on (B, out, anc) and QSD-synthesized
    #    (QSD of the completion was measured cheaper than qiskit's column-by-column Isometry:
    #     3/4/5/6 qubits 19/95/423/1783 CZ vs 24/122/546/2300)
    if mode == "restricted":
        V = restricted_columns(Ks, Sb, qb, n_anc)
    else:
        V = stinespring_columns(Ks, Lk, qb, n_anc)
    iso_q = (list(Breg) if Breg is not None else []) + [outr[0]] + (list(anc) if anc is not None else [])
    U, _, _ = exact_completion(V)
    if isinstance(U, QuantumCircuit):     # fitted fallback circuit (see fit_isometry_circuit)
        qc.compose(U, qubits=iso_q, inplace=True)
    else:
        qc.append(UnitaryGate(U, label="Dec"), iso_q)
    info["iso_qubits"] = len(iso_q)
    info["out_qubit"] = qc.find_bit(outr[0]).index
    return qc, info
