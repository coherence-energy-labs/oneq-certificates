"""Two-group symmetric codes for N~^n: encoder output in Sym^a (x) Sym^b, n = a + b.

Generalises the exact Sym^n reduction (oneq.superactivation, R132) group by group: flag counts (k1, k2), observed
weights (b1, b2); Bob's state per branch on R (x) Sym^{a-k1} (x) Sym^{b-k2}:
  rho = sum_{a1,a2} C1[b1,a1] C2[b2,a2] (1_R (x) P^a_{k1,a1} (x) P^b_{k2,a2}) X (.)^T,   weight C(a,k1) C(b,k2) / 2^n.
Contains every Sym^n code (Sym^{a+b} is inside Sym^a (x) Sym^b) and, at a = b = 1, every 2-use code.

  python twogroup.py check                 n = 2, a = b = 1 against the dense brute force (general_codes.py)
  python twogroup.py search A B R S        see-saw (power-iteration decoders, exact encoder SDP), R restarts, S steps;
                                           final score with exact per-branch decoder SDPs
"""
import itertools
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]          # the ONE_Q repo root
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation as sa  # noqa: E402

P_FLIP = sa.P_FLIP


def sel(m, k, a1):
    """P[i, i + a1] = sqrt(C(m-k, i) / C(m, i + a1)): Dicke^m_{i+a1} -> Dicke^{m-k}_i (group of size m, k flagged)"""
    d = m - k + 1
    P = np.zeros((d, m + 1))
    for i in range(d):
        P[i, i + a1] = math.sqrt(math.comb(m - k, i) / math.comb(m, i + a1))
    return P


def coeffs(k):
    return sa.flip_transition(k, P_FLIP) * np.array([math.comb(k, a) for a in range(k + 1)])[None, :]


def branches(a, b):
    for k1 in range(a + 1):
        for k2 in range(b + 1):
            yield k1, k2, math.comb(a, k1) * math.comb(b, k2) / 2 ** (a + b)


def bob_states(X, a, b, k1, k2):
    """stack over (b1, b2) of rho on R (x) Sym^{a-k1} (x) Sym^{b-k2} (R first)"""
    C1, C2 = coeffs(k1), coeffs(k2)
    ops = {(a1, a2): np.kron(np.eye(2), np.kron(sel(a, k1, a1), sel(b, k2, a2)))
           for a1 in range(k1 + 1) for a2 in range(k2 + 1)}
    S = {key: Op @ X @ Op.T for key, Op in ops.items()}
    out = []
    for b1 in range(k1 + 1):
        for b2 in range(k2 + 1):
            out.append(sum(C1[b1, a1] * C2[b2, a2] * S[(a1, a2)] for (a1, a2) in S))
    return np.stack(out), ops


def adjoint(Ds, a, b, k1, k2):
    C1, C2 = coeffs(k1), coeffs(k2)
    m = 2 * (a + 1) * (b + 1)
    out = np.zeros((m, m), complex)
    for a1 in range(k1 + 1):
        for a2 in range(k2 + 1):
            Op = np.kron(np.eye(2), np.kron(sel(a, k1, a1), sel(b, k2, a2)))
            Dsum = 0
            idx = 0
            for b1 in range(k1 + 1):
                for b2 in range(k2 + 1):
                    Dsum = Dsum + C1[b1, a1] * C2[b2, a2] * Ds[idx]
                    idx += 1
            out += Op.T @ Dsum @ Op
    return out


def decode(X, a, b, warm=None, iters=60):
    decs, F = {}, 0.0
    for k1, k2, w in branches(a, b):
        R, _ = bob_states(X, a, b, k1, k2)
        d = (a - k1 + 1) * (b - k2 + 1)
        D = warm[(k1, k2)] if warm is not None else np.stack([np.eye(2 * d) / 2] * len(R))
        for _ in range(iters):
            D = sa.power_step(R, D, d)
        decs[(k1, k2)] = D
        F += w * float(np.real(np.einsum("bij,bji->", R, D))) / 4
    return decs, F


def exact_score(X, a, b):
    sys.path.insert(0, str(ROOT / "experiments" / "superactivation"))
    from seesaw_lane import exact_decoder
    F = 0.0
    for k1, k2, w in branches(a, b):
        R, _ = bob_states(X, a, b, k1, k2)
        d = (a - k1 + 1) * (b - k2 + 1)
        F += w * sum(float(np.real(np.trace(r @ exact_decoder(r, d)))) for r in R) / 4
    return F


def encoder_step(decs, a, b):
    import cvxpy as cp                          # the search needs the `superactivation` extra; the reduction does not
    m = (a + 1) * (b + 1)
    G = sum(w * adjoint(decs[(k1, k2)], a, b, k1, k2) / 4 for k1, k2, w in branches(a, b))
    Xv = cp.Variable((2 * m, 2 * m), hermitian=True)
    cons = [Xv >> 0, cp.partial_trace(Xv, dims=[2, m], axis=1) == np.eye(2)]
    cp.Problem(cp.Maximize(cp.real(cp.trace(sa.herm(G) @ Xv))), cons).solve(solver=cp.CLARABEL)
    Xn = sa.herm(Xv.value)
    w, v = np.linalg.eigh(Xn)
    Xn = (v * np.clip(w, 0, None)) @ v.conj().T
    m_ = m
    T = Xn.reshape(2, m_, 2, m_).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m_))
    return sa.herm(Ti @ Xn @ Ti)


def gradient(decs, a, b):
    return sa.herm(sum(w * adjoint(decs[(k1, k2)], a, b, k1, k2) / 4 for k1, k2, w in branches(a, b)))


def encoder_power(G, X, m, iters=20):
    """Reimpell power steps for max Tr(G X), X >= 0, Tr_A X = 1_R -- the encoder mirror of sa.power_step.
    No SDP: the Clarabel encoder SDP stores the PSD-cone Hessian densely, (2m(2m+1)/2)^2 doubles, which took
    a 7.7 GB peak at n = 11 (m = 42) and would need ~22 GB at n = 16. This is matrix products only."""
    for _ in range(iters):
        Y = sa.herm(G @ X @ G)
        S = Y.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
        w, v = np.linalg.eigh(sa.herm(S))
        T = np.kron((v / np.sqrt(np.maximum(w, 1e-300))) @ v.conj().T, np.eye(m))
        X = sa.herm(T @ Y @ T)
    return X


def seeded_power(a, b, eps, steps, sym_steps=300, seed=5, inner=10, dec_iters=40, save=True):
    """the seeded search with power iteration on BOTH sides (no SDP anywhere): memory stays O(m^2)"""
    sys.path.insert(0, str(ROOT / "experiments" / "superactivation"))
    import seesaw_lane as sl
    n = a + b
    m = (a + 1) * (b + 1)
    rng = np.random.default_rng(seed)
    Xs, Fs, _ = sl.seesaw(n, sa.random_encoder(n, rng, rank=1), sym_steps)
    M = np.kron(np.eye(2), sym_embedding(a, b))
    X = (1 - eps) * (M @ Xs @ M.T) + eps * random_encoder(a, b, rng)
    decs, F = decode(X, a, b, iters=200)
    print(f"(a,b)=({a},{b}) symmetric seed {Fs:.10f}; perturbed start {F:.10f}", flush=True)
    for s in range(steps):
        X = encoder_power(gradient(decs, a, b), X, m, iters=inner)
        decs, F = decode(X, a, b, warm=decs, iters=dec_iters)
        if (s + 1) % 25 == 0:
            print(f"   step {s + 1}: F = {F:.10f}  (gain over symmetric {F - Fs:+.6f})", flush=True)
            if save:
                np.save(f"twogroup_X_a{a}_b{b}_power.npy", X)
    decs, F = decode(X, a, b, warm=decs, iters=2000)                 # tight, feasible decoders
    print(f"(a,b)=({a},{b}) n={n}: F = {F:.10f} with converged power decoders; symmetric {Fs:.10f}; "
          f"gain {F - Fs:+.6f}", flush=True)
    if save:
        np.save(f"twogroup_X_a{a}_b{b}_power.npy", X)
    return F, X, decs


PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]).astype(complex)]


def encoder_exact(G, m, z0=None, tol=1e-9):
    """EXACT encoder half-step without a big SDP, through the dual:
         max Tr(G X)  s.t. X >= 0, Tr_A X = 1_R      =      min_Y Tr Y  s.t.  Y (x) 1_m >= G.
    With Y = t 1 + z.sigma, t*(z) = lambda_max(G - (z.sigma) (x) 1), so the dual is the 3-parameter convex
    minimisation of phi(z) = 2 t*(z). The primal optimum lives on the top eigenspace of G - (z*.sigma) (x) 1;
    a tiny SDP there (dimension = that eigenspace's) imposes Tr_A X = 1_R. Cost: eigendecompositions of 2m x 2m."""
    import cvxpy as cp
    from scipy.optimize import minimize
    Im = np.eye(m)

    def M(z):
        return G - np.kron(z[0] * PAULI[0] + z[1] * PAULI[1] + z[2] * PAULI[2], Im)

    scale = max(1e-12, float(np.abs(G).max()))

    def phi(z, beta):                                     # smooth upper bound (log-sum-exp) of lambda_max
        w = np.linalg.eigvalsh(M(z))
        wm = w.max()
        return wm + np.log(np.exp(beta * (w - wm)).sum()) / beta

    z = np.zeros(3) if z0 is None else np.array(z0, float)
    for beta in (1e2 / scale, 1e4 / scale, 1e6 / scale, 1e8 / scale):
        z = minimize(lambda q: phi(q, beta), z, method="BFGS", options={"gtol": 1e-12 * scale}).x
    w, v = np.linalg.eigh(M(z))
    # the exact optimum lives on the TOP eigenspace at the exact dual optimum; z is only near it, so near-degenerate
    # top eigenvalues split slightly. Solve on the span of the top K eigenvectors instead: it contains that eigenspace,
    # so the small SDP over it attains the true optimum, and it stays K x K whatever n is.
    K = min(2 * m, 16)
    top = v[:, -K:]
    r = K
    Q = cp.Variable((r, r), hermitian=True)
    cons = [Q >> 0]
    # Tr_A X = 1_R as four real linear equations on Q
    for i in range(2):
        for j in range(2):
            blk = sum(top[i * m + s, :][:, None] * top[j * m + s, :].conj()[None, :] for s in range(m))
            cons.append(cp.trace(blk.T @ Q) == (1.0 if i == j else 0.0))
    cp.Problem(cp.Maximize(cp.real(cp.trace(sa.herm(top.conj().T @ G @ top) @ Q))), cons).solve(solver=cp.CLARABEL)
    X = sa.herm(top @ Q.value @ top.conj().T)
    T = X.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, Im)
    return sa.herm(Ti @ X @ Ti), z, r


def _resample_matrix(m0, m1):
    """Dicke-profile linear resampling (as oneq.superactivation.resample_encoder), m0 -> m1 points"""
    W = np.zeros((m1, m0))
    for i, x in enumerate(np.linspace(0, 1, m1)):
        j = min(int(x * (m0 - 1)), m0 - 2)
        t = x * (m0 - 1) - j
        W[i, j] += 1 - t
        W[i, j + 1] += t
    return W


def resample_twogroup(X, a, b, a2, b2):
    K = np.kron(np.eye(2), np.kron(_resample_matrix(a + 1, a2 + 1), _resample_matrix(b + 1, b2 + 1)))
    Y = K @ X @ K.T
    m = (a2 + 1) * (b2 + 1)
    T = Y.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m))
    return sa.herm(Ti @ Y @ Ti)


SYM = {6: 0.7279760105406684, 7: 0.7314907345253038, 8: 0.7367078374908842, 9: 0.7415856671408043,
       10: 0.7446872068500315, 11: 0.7466533087592615, 12: 0.7479079597906465, 13: 0.7487123801046237,
       14: 0.7492683603781356, 15: 0.7496056047219574, 16: 0.7499205796510597, 17: 0.7501717005514995}
# ^ the authors' symmetric (isometric-ansatz) table, permqit 00c18d3d64 -- the bar each two-group code must beat


def grow(path, a, b, chain, steps, inner=10, dec_iters=40):
    """chain the two-group code upward: e.g. chain = '4,5 5,5 5,6 6,6'; each size seeded by resampling the last"""
    X = np.load(path)
    decs = None
    for tok in chain.split():
        a2, b2 = map(int, tok.split(","))
        X = resample_twogroup(X, a, b, a2, b2)
        a, b = a2, b2
        n, m = a + b, (a + 1) * (b + 1)
        decs, F = decode(X, a, b, iters=200)
        for s in range(steps):
            X = encoder_power(gradient(decs, a, b), X, m, iters=inner)
            decs, F = decode(X, a, b, warm=decs, iters=dec_iters)
            if (s + 1) % 50 == 0:
                print(f"   ({a},{b}) step {s + 1}: F = {F:.10f}", flush=True)
        decs, F = decode(X, a, b, warm=decs, iters=1000)
        np.save(f"twogroup_X_a{a}_b{b}_grown.npy", X)
        print(f"GROWN (a,b)=({a},{b}) n={n}: F = {F:.10f} | symmetric {SYM.get(n, float('nan')):.10f} | "
              f"gain {F - SYM.get(n, float('nan')):+.6f} | {'ABOVE 3/4' if F > 0.75 else 'below 3/4'}", flush=True)


def grow_exact(path, a, b, chain, steps, dec_iters=60):
    """grow(), but every encoder half-step is EXACT (encoder_exact, the 3-parameter dual) -- a true see-saw"""
    X = np.load(path)
    for tok in chain.split():
        a2, b2 = map(int, tok.split(","))
        X = resample_twogroup(X, a, b, a2, b2)
        a, b = a2, b2
        n, m = a + b, (a + 1) * (b + 1)
        decs, F = decode(X, a, b, iters=200)
        z = None
        hist = [F]
        for s in range(steps):
            X, z, _ = encoder_exact(gradient(decs, a, b), m, z0=z)
            decs, F = decode(X, a, b, warm=decs, iters=dec_iters)
            hist.append(F)
            if (s + 1) % 10 == 0:
                print(f"   ({a},{b}) exact step {s + 1}: F = {F:.10f}", flush=True)
                np.save(f"twogroup_X_a{a}_b{b}_exact.npy", X)
            if s > 20 and hist[-1] - hist[-11] < 1e-9:
                break
        decs, F = decode(X, a, b, warm=decs, iters=1000)
        np.save(f"twogroup_X_a{a}_b{b}_exact.npy", X)
        print(f"EXACT (a,b)=({a},{b}) n={n}: F = {F:.10f} | symmetric {SYM.get(n, float('nan')):.10f} | "
              f"gain {F - SYM.get(n, float('nan')):+.6f} | {'ABOVE 3/4' if F > 0.75 else 'below 3/4'}", flush=True)


def restarts_exact(a, b, restarts, steps, seed=101, dec_iters=60):
    """random-restart see-saw with EXACT encoder steps: which basin is best at this n?"""
    n, m = a + b, (a + 1) * (b + 1)
    rng = np.random.default_rng(seed)
    best = (0.0, None)
    for r in range(restarts):
        X = random_encoder(a, b, rng)
        decs, F = decode(X, a, b, iters=200)
        z, hist = None, [F]
        for s in range(steps):
            X, z, _ = encoder_exact(gradient(decs, a, b), m, z0=z)
            decs, F = decode(X, a, b, warm=decs, iters=dec_iters)
            hist.append(F)
            if s > 20 and hist[-1] - hist[-11] < 1e-9:
                break
        print(f"   ({a},{b}) restart {r}: F = {F:.10f} after {len(hist) - 1} steps "
              f"(gain {F - SYM.get(n, float('nan')):+.6f})", flush=True)
        if F > best[0]:
            best = (F, X)
            np.save(f"twogroup_X_a{a}_b{b}_restarts.npy", X)
    print(f"RESTARTS (a,b)=({a},{b}) n={n}: best F = {best[0]:.10f} | symmetric {SYM.get(n, float('nan')):.10f} | "
          f"gain {best[0] - SYM.get(n, float('nan')):+.6f} | {'ABOVE 3/4' if best[0] > 0.75 else 'below 3/4'}",
          flush=True)


def random_encoder(a, b, rng):
    m = (a + 1) * (b + 1)
    K = rng.normal(size=(2 * m, 2)) + 1j * rng.normal(size=(2 * m, 2))
    X = K @ K.conj().T
    T = X.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m))
    return sa.herm(Ti @ X @ Ti)


def search(a, b, restarts, steps, seed=11):
    rng = np.random.default_rng(seed)
    best = (0.0, None)
    for r in range(restarts):
        X = random_encoder(a, b, rng)
        decs, F = decode(X, a, b, iters=200)
        for s in range(steps):
            X = encoder_step(decs, a, b)
            decs, F = decode(X, a, b, warm=decs)
        print(f"(a,b)=({a},{b}) restart {r}: F(power decoders) = {F:.10f}", flush=True)
        if F > best[0]:
            best = (F, X)
    Fx = exact_score(best[1], a, b)
    print(f"(a,b)=({a},{b}) n={a + b}: best F = {Fx:.10f} (exact decoders)", flush=True)
    np.save(f"twogroup_X_a{a}_b{b}.npy", best[1])
    return Fx


def check():
    """a = b = 1 is the whole 2-qubit space; compare against the dense brute force on the Round-2 encoder"""
    e0 = np.array([1, 1, 1, 1]) / 2.0                                   # |++>
    plus, minus = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
    e1 = (np.kron(minus, plus) + 1j * np.kron(plus, minus)) / np.sqrt(2)
    # computational basis |q0 q1> -> Sym^1 (x) Sym^1 basis: Dicke^1_w = |w>, so the map is the identity
    V = np.stack([e0, e1], axis=1).astype(complex)
    X = np.zeros((8, 8), complex)
    for r in range(2):
        for s in range(2):
            X[r * 4:(r + 1) * 4, s * 4:(s + 1) * 4] = np.outer(V[:, r], V[:, s].conj())
    print(f"Round-2 encoder through the two-group reduction, exact decoders: F = {exact_score(X, 1, 1):.10f}"
          "   (dense brute force 0.6897047609)")


def sym_embedding(a, b):
    """M: Sym^{a+b} -> Sym^a (x) Sym^b,  |D^n_w> = sum_{i+j=w} sqrt(C(a,i) C(b,j) / C(n,w)) |D^a_i>|D^b_j>"""
    n = a + b
    M = np.zeros(((a + 1) * (b + 1), n + 1))
    for w in range(n + 1):
        for i in range(max(0, w - b), min(a, w) + 1):
            M[i * (b + 1) + (w - i), w] = math.sqrt(math.comb(a, i) * math.comb(b, w - i) / math.comb(n, w))
    return M


def seeded(a, b, eps, steps, sym_steps=300, seed=5):
    """symmetric optimum (R132 see-saw on Sym^n) embedded into Sym^a (x) Sym^b, perturbed by eps toward a random
    two-group code (a symmetric code is a fixed point of the two-group see-saw), then climbed"""
    sys.path.insert(0, str(ROOT / "experiments" / "superactivation"))
    import psutil
    import seesaw_lane as sl
    rss = lambda: f"[rss {psutil.Process().memory_info().rss / 2**30:.2f} GB]"  # noqa: E731
    n = a + b
    rng = np.random.default_rng(seed)
    Xs, Fs, _ = sl.seesaw(n, sa.random_encoder(n, rng, rank=1), sym_steps)
    print(f"after symmetric see-saw {rss()}", flush=True)
    M = np.kron(np.eye(2), sym_embedding(a, b))
    X0 = M @ Xs @ M.T
    print(f"(a,b)=({a},{b}) symmetric seed F = {Fs:.10f}; embedded through the reduction: "
          f"{decode(X0, a, b, iters=300)[1]:.10f} {rss()}", flush=True)
    X = (1 - eps) * X0 + eps * random_encoder(a, b, rng)
    decs, F = decode(X, a, b, iters=200)
    print(f"after first decode {rss()}", flush=True)
    for s in range(steps):
        X = encoder_step(decs, a, b)
        decs, F = decode(X, a, b, warm=decs)
        if s < 3:
            print(f"   step {s + 1} {rss()} F = {F:.10f}", flush=True)
        if (s + 1) % 25 == 0:
            print(f"   step {s + 1}: F = {F:.10f} {rss()}", flush=True)
    Fx = exact_score(X, a, b)
    print(f"(a,b)=({a},{b}) n={n}: seeded best F = {Fx:.10f} (exact decoders); symmetric {Fs:.10f}; "
          f"gain {Fx - Fs:+.6f}", flush=True)
    np.save(f"twogroup_X_a{a}_b{b}_seeded.npy", X)
    return Fx


def dicke(m, w):
    v = np.zeros(2 ** m)
    for bits in itertools.product((0, 1), repeat=m):
        if sum(bits) == w:
            v[int("".join(map(str, bits)), 2) if m else 0] = 1
    return v / np.linalg.norm(v)


def check_dense(a, b, seed=3):
    """a random two-group encoder, scored by the reduction AND embedded into the full 2^n space and scored by the
    dense brute force (general_codes.fidelity_exact); both with exact decoders"""
    import general_codes as G
    rng = np.random.default_rng(seed)
    X = random_encoder(a, b, rng)
    w, v = np.linalg.eigh(X)
    m = (a + 1) * (b + 1)
    E = np.zeros((2 ** (a + b), m))                     # Sym^a (x) Sym^b basis -> computational basis, group A first
    for i in range(a + 1):
        for j in range(b + 1):
            E[:, i * (b + 1) + j] = np.kron(dicke(a, i), dicke(b, j))
    big = np.kron(np.eye(2), E)
    rho = big @ (X / 2) @ big.T                          # reference-encoded state on R (x) A^n, trace 1
    Fd, _ = G.fidelity_exact(rho, a + b)
    Fr = exact_score(X, a, b)
    print(f"(a,b)=({a},{b}) random code: reduction {Fr:.10f}  dense {Fd:.10f}  |diff| {abs(Fr - Fd):.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
        check_dense(1, 2)
        check_dense(2, 2)
    elif sys.argv[1] == "restartsx":
        restarts_exact(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == "growx":
        grow_exact(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    elif sys.argv[1] == "grow":
        grow(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    elif sys.argv[1] == "power":
        seeded_power(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == "seeded":
        seeded(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
    else:
        search(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
