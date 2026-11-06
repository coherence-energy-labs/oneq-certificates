"""R170: codes for m uses of the 50% erasure channel A_D (input dimension D), sending one qubit, unassisted.

F_c(A_D^{(x)m}, 2) = max over encoders E: C^2 -> (C^D)^{(x)m} and decoders of the entanglement fidelity. Each use is
erased independently with probability 1/2 and Bob sees the flags, so the fidelity splits over Bob's set S of kept
uses (weight 2^-m):

    F = sum_S 2^-m F_S,   F_S = max_D <Phi| (id (x) D o Tr_{S^c} o E)(Phi) |Phi>
                              = (1/2) max { Tr(C rho_{R B_S}) : C >= 0, Tr_R C = 1_{B_S} }      (C = adjoint-decoder Choi)
                              = (1/2) min { Tr Z : rho_{R B_S} <= 1_R (x) Z }                  (= 2^{-H_min(R|B_S)} / 2)

Representation (PROVEN reductions, used everywhere below):
  * F is a maximum of linear functionals of the encoder's Choi matrix, so it is CONVEX in the encoder and its maximum
    over the convex set of channels sits at an extreme channel. Extreme channels with a 2-dimensional input have
    Kraus rank <= 2 (Choi 1975: {K_i^dag K_j} must be linearly independent in the 4-dimensional space of 2 x 2
    matrices). So WLOG the encoder is an isometry V: C^2 -> (C^D)^{(x)m} (x) C^e with e <= 2.
  * The code is then the pure state psi = 2^{-1/2} sum_i |i>_R (x) V|i> on R (x) B_1..B_m (x) E0, stored as a tensor.
  * Bob's branch state rho_{R B_S} = M M^dag with M = psi reshaped to (R B_S) x (B_{S^c} E0). Its decoder problem lives
    on the support of rho_{B_S}, of dimension <= min(D^|S|, 2 e D^{m - |S|}).

The see-saw below is the same minorize-maximize as R132/R140 (any decoder gives a lower bound; with the decoders fixed,
F is a convex quadratic in V, so the polar step V <- polar(grad) cannot decrease it). Pure numpy; no SDP.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def herm(Z: np.ndarray) -> np.ndarray:
    return (Z + Z.conj().T) / 2


def patterns(m: int):
    """every Bob set S as a 0/1 tuple: z[i] = 1 means use i is KEPT (Bob holds B_i)"""
    return list(itertools.product((0, 1), repeat=m))


def psi_from_isometry(V: np.ndarray, m: int, D: int, e: int) -> np.ndarray:
    """psi[r, b_1, ..., b_m, e0] = V[(b_1..b_m, e0), r] / sqrt 2   (V's column r is the image of |r>)"""
    if V.shape != (D ** m * e, 2):
        raise ValueError(f"isometry shape {V.shape} != {(D ** m * e, 2)}")
    return V.T.reshape([2] + [D] * m + [e]) / np.sqrt(2)


def isometry_error(V: np.ndarray) -> float:
    return float(np.abs(V.conj().T @ V - np.eye(2)).max())


def random_isometry(m: int, D: int, e: int, rng) -> np.ndarray:
    Z = rng.normal(size=(D ** m * e, 2)) + 1j * rng.normal(size=(D ** m * e, 2))
    Q, _ = np.linalg.qr(Z)
    return Q


def branch_matrix(psi: np.ndarray, m: int, z) -> tuple[np.ndarray, list[int]]:
    """M with rows (R, B_S) and columns (B_{S^c}, E0), rho_{R B_S} = M M^dag; also the axis permutation used"""
    bob = [1 + i for i in range(m) if z[i]]
    rest = [1 + i for i in range(m) if not z[i]] + [m + 1]
    perm = [0] + bob + rest
    T = np.transpose(psi, perm)
    rows = 2 * int(np.prod([psi.shape[a] for a in bob])) if bob else 2
    return T.reshape(rows, -1), perm


def unbranch(Mt: np.ndarray, psi_shape, perm) -> np.ndarray:
    """inverse of branch_matrix's reshape/transpose (for the encoder gradient)"""
    shp = [psi_shape[a] for a in perm]
    T = Mt.reshape(shp)
    inv = np.argsort(perm)
    return np.transpose(T, inv)


def ptrace_R(C: np.ndarray, d: int) -> np.ndarray:
    """Tr_R of an operator on R (x) B with R = C^2 first: sum_r <r| C |r>"""
    T = C.reshape(2, d, 2, d)
    return T[0, :, 0, :] + T[1, :, 1, :]


def decoder_power_step(rho: np.ndarray, C: np.ndarray, d: int) -> np.ndarray:
    """one Reimpell power step for max Tr(C rho), C >= 0, Tr_R C = 1 (oneq.superactivation.power_step, unbatched):
    C <- (1 (x) S^{-1/2}) rho C rho (1 (x) S^{-1/2}) + I_R/2 (x) (kernel projector), S = Tr_R(rho C rho)"""
    G = herm(rho @ C @ rho)
    S = herm(ptrace_R(G, d))
    w, v = np.linalg.eigh(S)
    keep = w > 1e-13 * max(w.max(), 1e-300)
    inv = np.where(keep, 1.0 / np.sqrt(np.where(keep, w, 1.0)), 0.0)
    Sih = (v * inv) @ v.conj().T
    Pk = (v * keep) @ v.conj().T
    Si = np.kron(np.eye(2), Sih)
    return herm(Si @ G @ Si + np.kron(np.eye(2), np.eye(d) - Pk) / 2)


def support_basis(M: np.ndarray, dB: int, rtol: float = 1e-12) -> np.ndarray:
    """orthonormal basis (dB x n') of the support of rho_B for rho_{RB} = M M^dag (range of [M_0, M_1])"""
    Mr = M.reshape(2, dB, -1)
    W = np.concatenate([Mr[0], Mr[1]], axis=1)
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    keep = s > rtol * max(s.max(), 1e-300)
    return U[:, keep]


class Branch:
    """one Bob set S: its (support-reduced) state, decoder, and value"""

    __slots__ = ("z", "k", "w", "dB", "Q", "rho", "C", "F")

    def __init__(self, z, w):
        self.z, self.k, self.w = z, sum(z), w
        self.dB = self.Q = self.rho = self.C = None
        self.F = 0.0


def load_branch(br: Branch, psi: np.ndarray, m: int, reduce_above: int = 32):
    M, _ = branch_matrix(psi, m, br.z)
    dB = M.shape[0] // 2
    if dB > reduce_above:
        Q = support_basis(M, dB)
        Mr = M.reshape(2, dB, -1)
        Mred = np.concatenate([Q.conj().T @ Mr[0], Q.conj().T @ Mr[1]], axis=0)
        br.Q = Q
    else:
        Mred = M
        br.Q = None
    d = Mred.shape[0] // 2
    rho = herm(Mred @ Mred.conj().T)
    if br.C is None or br.C.shape[0] != 2 * d or br.dB != d:
        br.C = np.eye(2 * d) / 2
    br.dB, br.rho = d, rho


def decode_all(branches, psi, m, iters=60, reduce_above=32):
    F = 0.0
    for br in branches:
        load_branch(br, psi, m, reduce_above)
        if br.k == 0:                                   # Bob holds nothing: exactly 1/4 (R maximally mixed)
            br.C = np.eye(2) / 2
            br.F = 0.25
        else:
            for _ in range(iters):
                br.C = decoder_power_step(br.rho, br.C, br.dB)
            br.F = 0.5 * float(np.real(np.trace(br.C @ br.rho)))
        F += br.w * br.F
    return F


def encoder_gradient(branches, psi, m):
    """Y[:, r] = sqrt 2 <r|_R sum_S w_S (C_S (x) 1) psi -- the ascent direction of F in V (decoders fixed)"""
    G = np.zeros_like(psi)
    for br in branches:
        M, perm = branch_matrix(psi, m, br.z)
        if br.Q is not None:
            dB = M.shape[0] // 2
            Mr = M.reshape(2, dB, -1)
            Mred = np.concatenate([br.Q.conj().T @ Mr[0], br.Q.conj().T @ Mr[1]], axis=0)
            CM = br.C @ Mred
            d = br.dB
            CMr = CM.reshape(2, d, -1)
            Mt = np.concatenate([br.Q @ CMr[0], br.Q @ CMr[1]], axis=0)
        else:
            Mt = br.C @ M
        G += br.w * unbranch(Mt, psi.shape, perm)
    return np.sqrt(2) * G.reshape(2, -1).T


def polar(Y: np.ndarray) -> np.ndarray:
    U, _, Wh = np.linalg.svd(Y, full_matrices=False)
    return U @ Wh


def make_branches(m: int):
    return [Branch(z, 2.0 ** -m) for z in patterns(m)]


def seesaw(m, D, e, V, steps=300, dec_iters=40, tol=1e-11, log=None, reduce_above=32):
    branches = make_branches(m)
    psi = psi_from_isometry(V, m, D, e)
    F = decode_all(branches, psi, m, iters=200, reduce_above=reduce_above)
    hist = [F]
    for s in range(steps):
        V = polar(encoder_gradient(branches, psi, m))
        psi = psi_from_isometry(V, m, D, e)
        F = decode_all(branches, psi, m, iters=dec_iters, reduce_above=reduce_above)
        hist.append(F)
        if log and (s + 1) % 50 == 0:
            log(f"   m={m} D={D} e={e} step {s + 1}: F = {F:.10f}")
        if len(hist) > 31 and hist[-1] - hist[-31] < tol:
            break
    F = decode_all(branches, psi, m, iters=1500, reduce_above=reduce_above)
    return V, branches, F, hist


def per_size(branches, m):
    """mean F_S over |S| = k, for k = 0..m"""
    out = []
    for k in range(m + 1):
        vals = [br.F for br in branches if br.k == k]
        out.append(float(np.mean(vals)))
    return out


# ------------------------------------------------------------------ analytic bounds (DERIVED; see bounds.py for proofs)

def pair_bound(m: int) -> float:
    """F_c(A_D^m, 2) <= 3/4 - 2^-(m+2) for every D: complementary Bob sets pair up, each pair <= 3/2 by
    two-extendibility, and the pair (empty, all) <= 1/4 + 1"""
    return 0.75 - 2.0 ** -(m + 2)


def eta_partner_max(eta1: float) -> float:
    """the cloning-region boundary: the largest depolarizing parameter of clone 2 given clone 1's (1 -> 2, qubits),
    eta2 = (1 - eta1 + sqrt((1 - eta1)(1 + 3 eta1))) / 2, i.e. eta1^2 + eta2^2 + eta1 eta2 = eta1 + eta2"""
    return (1 - eta1 + math.sqrt(max(0.0, (1 - eta1) * (1 + 3 * eta1)))) / 2


def fid_from_eta(eta: float) -> float:
    return (1 + 3 * eta) / 4


def eta_from_fid(F: float) -> float:
    return (4 * F - 1) / 3
