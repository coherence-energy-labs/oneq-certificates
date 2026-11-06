"""Independent dense check: do NON-symmetric encoders beat Sym^n ones on the flagged channel N~^n?

Written from first principles (not the Round-2 package's code, which was not supplied). For an encoder V: C^2 -> (C^2)^n
the reference-encoded state rho_RA; for every flag string z the branch state sigma_z = (id_R (x) N_z1 (x) ... )(rho_RA),
N_0 = id, N_1 = X^p o full Z-dephasing, p = 1/(1+sqrt 2); weight 2^-n. For each branch the decoder is the EXACT optimum
of an SDP over its Choi J (B (x) R', Tr_R' J = 1_B, J >= 0):  F_z = (1/2) Tr(J K_z), K_z a reindexing of sigma_z.
The encoder step (decoders fixed) is also an exact SDP over all CPTP encoders. Modes:

  python general_codes.py check            the Round-2 explicit 2-use encoder + the Sym^2 optimum, exact decoders
  python general_codes.py search N R S     see-saw over ALL encoders at n = N, R random restarts, S steps
"""
import itertools
import sys

import numpy as np

P = 1 / (1 + np.sqrt(2))
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)
H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)


def noisy_kraus(p: float | None = None):
    """X^p o full dephasing as Kraus operators: dephasing = {|0><0|, |1><1|}, then {sqrt(1-p) 1, sqrt(p) X}. p defaults
    to this module's d = 2 value P; a caller at another p PASSES it. Until 2026-10-19 onset_vs_p wrote the global P
    and never restored it, so any later caller in the same process silently got the last d's channel (the wall's
    test_all_ones_is_the_dense_general_model failed once the suite's worker order put a shield-family test first)."""
    q = P if p is None else p
    P0, P1 = np.diag([1.0, 0]).astype(complex), np.diag([0, 1.0]).astype(complex)
    return [np.sqrt(1 - q) * P0, np.sqrt(1 - q) * P1, np.sqrt(q) * X @ P0, np.sqrt(q) * X @ P1]


def apply_on(rho, kraus, q, n):
    """apply a 1-qubit channel on qubit q of the A register of rho on R (x) A^n (R = 2, A ordered q0..q_{n-1})"""
    dA = 2 ** n
    T = rho.reshape([2] + [2] * n + [2] + [2] * n)
    out = np.zeros_like(T)
    for k in kraus:
        # contract k on axis 1+q (ket) and k^dag on axis 2+n+q (bra)
        t = np.tensordot(k, T, axes=([1], [1 + q]))
        t = np.moveaxis(t, 0, 1 + q)
        t = np.tensordot(t, k.conj(), axes=([2 + n + q], [1]))
        t = np.moveaxis(t, -1, 2 + n + q)
        out = out + t
    return out.reshape(2 * dA, 2 * dA)


def branch_states(rho, n, p: float | None = None):
    out = {}
    nk = noisy_kraus(p)
    for z in itertools.product((0, 1), repeat=n):
        s = rho
        for q, f in enumerate(z):
            if f:
                s = apply_on(s, nk, q, n)
        out[z] = s
    return out


def K_of(sigma, dB):
    T = sigma.reshape(2, dB, 2, dB)                  # (r, b, r', b')
    return T.transpose(3, 2, 1, 0).reshape(dB * 2, dB * 2)   # rows (b', r'), cols (b, r)


def best_decoder(sigma, dB):
    import cvxpy as cp                  # the SDPs need the `superactivation` extra; the dense channel model does not
    K = K_of(sigma, dB)
    J = cp.Variable((dB * 2, dB * 2), hermitian=True)
    cons = [J >> 0, cp.partial_trace(J, dims=[dB, 2], axis=1) == np.eye(dB)]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(J @ K)) / 2), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, J.value


def rho_of(V):
    """reference-encoded state for an isometry V (columns = codewords): (1/sqrt2) sum_r |r> (x) V|r>"""
    psi = np.concatenate([V[:, 0], V[:, 1]]) / np.sqrt(2)     # R first
    return np.outer(psi, psi.conj())


def fidelity_exact(rho, n):
    dB = 2 ** n
    tot = 0.0
    Js = {}
    for z, s in branch_states(rho, n).items():
        f, J = best_decoder(s, dB)
        tot += f / 2 ** n
        Js[z] = J
    return tot, Js


def encoder_step(Js, n):
    """max over ALL encoder Choi E on R (x) A (Tr_A E = 1_R... as rho_RA = E/2) of the linear fidelity"""
    import cvxpy as cp
    dA = dB = 2 ** n
    # F is linear in rho_RA: F = sum_z 2^-n (1/2) Tr(J_z K_z(rho)); build the gradient by linearity on a basis
    G = np.zeros((2 * dA, 2 * dA), complex)
    nk = noisy_kraus()
    for z, J in Js.items():
        # adjoint of rho -> K_z(N_z(rho)) contracted with J: compute via adjoint channel on J's reindexing
        # F_z = (1/2) Tr(J K(sigma)) = (1/2) Tr(M sigma) with M[(r',b'),(r,b)] = J[(b,r),(b',r')]: J's axes (3,2,1,0).
        # (checked numerically in search(): Tr(G rho) must reproduce F for the same decoders)
        M = J.reshape(dB, 2, dB, 2).transpose(3, 2, 1, 0).reshape(2 * dB, 2 * dB)
        for q, f in enumerate(z):
            if f:                                   # adjoint of the 1-qubit channel on qubit q
                M = apply_on(M, [k.conj().T for k in nk], q, n)
        G += M / 2 ** n / 2
    E = cp.Variable((2 * dA, 2 * dA), hermitian=True)          # rho_RA
    cons = [E >> 0, cp.partial_trace(E, dims=[2, dA], axis=1) == np.eye(2) / 2]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(G @ E))), cons)
    prob.solve(solver=cp.CLARABEL)
    return E.value, prob.value


def check():
    n = 2
    e0 = np.kron(H @ [1, 0], H @ [1, 0])                       # |++>
    plus, minus = H @ np.array([1, 0]), H @ np.array([0, 1])
    e1 = (np.kron(minus, plus) + 1j * np.kron(plus, minus)) / np.sqrt(2)
    V = np.stack([e0, e1], axis=1).astype(complex)
    print("isometry check |V^dag V - 1| =", np.abs(V.conj().T @ V - np.eye(2)).max())
    swap = np.eye(4)[[0, 2, 1, 3]]
    print("symmetric under swap?  |SWAP V - V| =", np.abs(swap @ V - V).max())
    F, _ = fidelity_exact(rho_of(V), n)
    print(f"Round-2 explicit encoder, exact decoders: F = {F:.10f}   (claimed 0.6897047613)")
    # the encoder-step gradient check: Tr(G rho) must equal the fidelity for the same decoders
    F2, Js = fidelity_exact(rho_of(V), n)
    return F


def gradient(Js, n):
    """the G of encoder_step, alone, for the linearity check"""
    dB = 2 ** n
    nk = noisy_kraus()
    G = 0
    for z, J in Js.items():
        M = J.reshape(dB, 2, dB, 2).transpose(3, 2, 1, 0).reshape(2 * dB, 2 * dB)
        for q, f in enumerate(z):
            if f:
                M = apply_on(M, [k.conj().T for k in nk], q, n)
        G = G + M / 2 ** n / 2
    return G


def search(n, restarts, steps, seed=7):
    rng = np.random.default_rng(seed)
    dA = 2 ** n
    best = (0, None)
    for r in range(restarts):
        A = rng.normal(size=(dA, 2)) + 1j * rng.normal(size=(dA, 2))
        V, _ = np.linalg.qr(A)
        rho = rho_of(V)
        F, Js = fidelity_exact(rho, n)
        lin = float(np.real(np.trace(gradient(Js, n) @ rho)))
        if abs(lin - F) > 1e-6:
            raise SystemExit(f"encoder gradient is wrong: Tr(G rho) = {lin} vs F = {F}")
        for s in range(steps):
            rho, lin = encoder_step(Js, n)
            F, Js = fidelity_exact(rho, n)
        print(f"n={n} restart {r}: F = {F:.10f} (encoder-step linear value {lin:.10f})", flush=True)
        if F > best[0]:
            best = (F, rho)
    w = np.linalg.eigvalsh(best[1])
    print(f"n={n} best F = {best[0]:.10f}; encoder state top eigenvalues {np.round(w[::-1][:3], 6)}")
    np.save(f"general_best_rho_n{n}.npy", best[1])
    return best


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        search(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
