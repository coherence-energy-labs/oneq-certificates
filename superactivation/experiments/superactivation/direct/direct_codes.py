"""Codes on the JOINT channel (P (x) A)^n directly, not through Prop. 1's per-use effective channel N~.

Every code found so far (the paper's, R132-R140's) runs each use of P (x) A through the fixed pre/post-processing
(E~, D~) of Prop. 1, which squeezes the use to a qubit-input channel N~. The joint channel has d_in = 16, d_out = 20
per use (paper l. 96: "too complex" to optimize), so F_c((P (x) A)^n) >= F_c(N~^n), possibly by a lot. Nobody has
measured by how much. This model optimizes over ALL encoders R -> (C^16)^n and all decoders.

Per use: input = P-in (C^4, key (x) shield) (x) A-in (C^4). A erases with probability 1/2 and Bob sees the flag, so
the code's fidelity splits over the erasure pattern z in {0,1}^n (weight 2^-n):
  keep  (z = 0): Bob receives P-out (C^4) (x) A-out (C^4)       map rho -> (P (x) id)(rho)
  erase (z = 1): Bob receives P-out (C^4) only                 map rho -> (P (x) Tr)(rho)
F = sum_z 2^-n Tr(R_z D_z) / 4 with R_z = (id_R (x) map_z)(X), X the encoder Choi (Tr_in X = 1_R), D_z an adjoint-
decoder Choi with Tr_R D_z = 1 (the convention of oneq.superactivation; perfect identity -> F = 1).

DERIVED bound at n = 1 (live check): the erased branch sees only P's output, a PPT-binding channel, so its fidelity is
<= 1/2; the kept branch is <= 1. Hence F_c(P (x) A) <= 3/4 at n = 1, and n* >= 2. The see-saw must also land
>= N~'s 0.6464466 (N~ is a special case).

  python direct_codes.py check            sanity: identity P -> F = 1; the n = 1 bracket [0.6464, 0.75]
  python direct_codes.py search N R S     see-saw over all codes at n = N, R restarts, S steps
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "compile"))
sys.path.insert(0, str(HERE.parent / "general_codes"))
from oneq import superactivation as sa  # noqa: E402

import sclib  # noqa: E402  -- P's six Kraus operators, transcribed from the paper's supplement (R139)
import twogroup as tg  # noqa: E402  -- encoder_power

DP = 4                                   # P input/output dimension (key (x) shield)
DA = 4                                   # A input dimension; kept output dimension (erasure flag is classical)
DIN = DP * DA                            # 16 per use


def superop(Ks) -> np.ndarray:
    """S[o, o', i, i'] = sum_k K[o, i] conj(K[o', i']): rho_out[o, o'] = sum S[o, o', i, i'] rho[i, i']"""
    return np.einsum("koi,kpj->opij", np.array(Ks), np.array(Ks).conj())


def p_superop(kraus=None) -> np.ndarray:
    return superop(kraus if kraus is not None else sclib.horodecki_kraus())


def bob_state(X, n, z, SP) -> np.ndarray:
    """R_z on R (x) prod_uses (P-out (x) [A-out if kept]) for an encoder Choi X on R (x) (C^16)^n.

    X is viewed as the tensor [r, (p1 a1), ..., r', (p1' a1'), ...]. P acts on every p leg pair; a kept use leaves its
    a legs alone; an erased use traces them."""
    T = X.reshape([2] + [DP, DA] * n + [2] + [DP, DA] * n)
    off = 1 + 2 * n                                          # where the bra legs start
    for u in range(n):
        ip, ipp = 1 + 2 * u, off + 1 + 2 * u                 # p ket / bra axes of use u
        T = np.tensordot(SP, T, axes=([2, 3], [ip, ipp]))    # new axes (o, o') at the front
        T = np.moveaxis(T, [0, 1], [ip, ipp])
    # trace erased a legs (from the last use backwards so axis numbers stay valid)
    for u in reversed(range(n)):
        if z[u]:
            ia, iaa = 2 + 2 * u, off + 2 + 2 * u
            T = np.trace(T, axis1=ia, axis2=iaa)
            off -= 1
    d = int(round(np.sqrt(T.size))) // 2
    return T.reshape(2 * d, 2 * d)


def bob_adjoint(D, n, z, SP) -> np.ndarray:
    """(id_R (x) map_z^*)(D) on R (x) (C^16)^n, the dual of map_z under Tr(Y map(rho)) = Tr(map^*(Y) rho), so that
    Tr(X adjoint) = Tr(R_z D): the gradient of the fidelity in the encoder.

    Dual of an erased use's trace: insert 1_A on its a legs. Dual of P: S*[a, b, c, d] = S[d, c, b, a] (from
    Tr(Y Phi(rho)) = sum Y[o', o] S[o, o', i, i'] rho[i, i']); in Kraus form sum_k K^dag Y K. No conjugate."""
    letters = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    r, rb = next(letters), next(letters)
    pk = [next(letters) for _ in range(n)]
    pb = [next(letters) for _ in range(n)]
    ak = [next(letters) for _ in range(n)]
    ab = [next(letters) for _ in range(n)]
    dk, db, outdims = [r], [rb], []
    for u in range(n):
        dk.append(pk[u])
        db.append(pb[u])
        outdims.append(DP)
        if not z[u]:
            dk.append(ak[u])
            db.append(ab[u])
            outdims.append(DA)
    subs, ops = ["".join(dk + db)], [D.reshape([2] + outdims + [2] + outdims)]
    for u in range(n):
        if z[u]:
            subs.append(ak[u] + ab[u])
            ops.append(np.eye(DA))
    full = [r] + [x for u in range(n) for x in (pk[u], ak[u])] + [rb] + [x for u in range(n) for x in (pb[u], ab[u])]
    T = np.einsum(",".join(subs) + "->" + "".join(full), *ops)
    SPa = SP.transpose(3, 2, 1, 0)
    off = 1 + 2 * n
    for u in range(n):
        ip, ipp = 1 + 2 * u, off + 1 + 2 * u
        T = np.tensordot(SPa, T, axes=([2, 3], [ip, ipp]))
        T = np.moveaxis(T, [0, 1], [ip, ipp])
    m = DIN ** n
    return T.reshape(2 * m, 2 * m)


def patterns(n):
    return list(itertools.product((0, 1), repeat=n))


def etilde_isometry() -> np.ndarray:
    """Prop. 1's per-use encoder E~ as an isometry C^2 -> P-in (x) A-in = C^16 (R139 compile/channels.e_tilde: H and
    CNOT put a Bell pair on the two shields, CNOT copies the key into A):
        V|r> = (1/sqrt 2) sum_s |r s>_P (x) |r s>_A,   P-in index 2 key + shield (sclib's key-first order)."""
    V = np.zeros((DIN, 2), dtype=complex)
    for r in range(2):
        for s in range(2):
            V[4 * (2 * r + s) + (2 * r + s), r] = 1 / np.sqrt(2)
    return V


def lift(Xq: np.ndarray, n: int) -> np.ndarray:
    """An N~ code (encoder Choi on R (x) (C^2)^n) as a code on (P (x) A)^n: compose E~ on every use. Its direct value
    (with direct-optimal decoders) can only be >= its N~ value: D~ is one of the decoders the direct model may use."""
    W = np.eye(2)
    for _ in range(n):
        W = np.kron(W, etilde_isometry())
    return W @ Xq @ W.conj().T


def decode(X, n, SP, warm=None, iters=200):
    decs, F = {}, 0.0
    for z in patterns(n):
        R = bob_state(X, n, z, SP)[None]
        d = R.shape[1] // 2
        D = warm[z] if warm is not None else np.eye(2 * d)[None] / 2
        for _ in range(iters):
            D = sa.power_step(R, D, d)
        decs[z] = D
        F += 2.0 ** -n * float(np.real(np.einsum("bij,bji->", R, D))) / 4
    return decs, F


def gradient(decs, n, SP):
    return sa.herm(sum(2.0 ** -n * bob_adjoint(decs[z][0], n, z, SP) / 4 for z in patterns(n)))


def random_encoder(n, rng, rank=2):
    m = DIN ** n
    K = rng.normal(size=(2 * m, rank)) + 1j * rng.normal(size=(2 * m, rank))
    X = K @ K.conj().T
    T = X.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m))
    return sa.herm(Ti @ X @ Ti)


def seesaw(n, SP, X, steps, inner=20, dec_iters=60, log=None):
    m = DIN ** n
    decs, F = decode(X, n, SP, iters=300)
    hist = [F]
    for s in range(steps):
        X = tg.encoder_power(gradient(decs, n, SP), X, m, iters=inner)
        decs, F = decode(X, n, SP, warm=decs, iters=dec_iters)
        hist.append(F)
        if log and (s + 1) % 50 == 0:
            log(f"   n={n} step {s + 1}: F = {F:.10f}")
        if len(hist) > 41 and hist[-1] - hist[-41] < 1e-10:
            break
    decs, F = decode(X, n, SP, warm=decs, iters=2000)
    return X, decs, F


def check() -> int:
    ok = True
    # (1) the adjoint IS the gradient: Tr(X adjoint_z(D)) == Tr(R_z D) for random X, D
    rng = np.random.default_rng(0)
    SP = p_superop()
    for n in (1, 2):
        X = random_encoder(n, rng)
        for z in patterns(n):
            R = bob_state(X, n, z, SP)
            Dm = rng.normal(size=R.shape) + 1j * rng.normal(size=R.shape)
            Dm = Dm @ Dm.conj().T
            lhs = np.trace(X @ bob_adjoint(Dm, n, z, SP))
            rhs = np.trace(R @ Dm)
            err = abs(lhs - rhs) / abs(rhs)
            ok &= err < 1e-12
    print(f"adjoint = gradient (n = 1, 2, every pattern): {'ok' if ok else 'FAIL'}")
    # (2) trace preservation of each branch: Tr R_z = Tr X = 2
    X = random_encoder(2, rng)
    tr = [float(np.real(np.trace(bob_state(X, 2, z, SP)))) for z in patterns(2)]
    ok2 = all(abs(t - 2) < 1e-12 for t in tr)
    print(f"each branch map is trace preserving: {'ok' if ok2 else 'FAIL'} {tr}")
    # (3) P := identity makes the joint channel perfect (send through P): F -> 1
    SPid = superop([np.eye(DP, dtype=complex)])
    _, _, Fid = seesaw(1, SPid, random_encoder(1, np.random.default_rng(3)), steps=300)
    ok3 = Fid > 1 - 1e-6
    print(f"P = identity, n = 1: F = {Fid:.10f} (expect 1): {'ok' if ok3 else 'FAIL'}")
    ok = ok and ok2 and ok3
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def search(n, restarts, steps, seed=11, dec_iters=60):
    SP = p_superop()
    rng = np.random.default_rng(seed)
    best = (0.0, None)
    for r in range(restarts):
        X, decs, F = seesaw(n, SP, random_encoder(n, rng), steps, dec_iters=dec_iters,
                            log=lambda s: print(s, flush=True))
        print(f"   n={n} restart {r}: F = {F:.10f}", flush=True)
        if F > best[0]:
            best = (F, X)
            np.save(HERE / f"direct_X_n{n}.npy", X)
    ntilde = {1: 0.6464466094, 2: 0.6897047609}
    print(f"DIRECT n={n}: best F = {best[0]:.10f} | through N~ (best known) {ntilde.get(n, float('nan')):.10f} | "
          f"{'ABOVE 3/4' if best[0] > 0.75 else 'below 3/4'}", flush=True)
    return best


def seeded(n, path, steps, dec_iters=15, tag="seeded"):
    """climb from a given encoder (e.g. a lifted N~ code): the result can only be >= the seed's direct value"""
    SP = dc_sp = p_superop()
    X0 = np.load(path)
    _, F0 = decode(X0, n, SP, iters=2000)
    print(f"   seed {pathlib.Path(path).name}: direct F = {F0:.10f}", flush=True)
    X, _, F = seesaw(n, dc_sp, X0, steps, dec_iters=dec_iters, log=lambda s: print(s, flush=True))
    np.save(HERE / f"direct_X_n{n}_{tag}.npy", X)
    print(f"SEEDED n={n}: F = {F:.10f} (seed {F0:.10f}, gain {F - F0:+.6f}) | "
          f"{'ABOVE 3/4' if F > 0.75 else 'below 3/4'}", flush=True)
    return X, F


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(check())
    if sys.argv[1] == "seeded":
        seeded(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        sys.exit(0)
    if sys.argv[1] == "search":
        search(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), seed=int(sys.argv[5]) if len(sys.argv) > 5 else 11,
               dec_iters=int(sys.argv[6]) if len(sys.argv) > 6 else 60)
        sys.exit(0)
    raise SystemExit(__doc__)
