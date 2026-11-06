"""Rigorous lower bound on F_c((P (x) A)^n, 2) for a DIRECT code (direct_codes.py), with maps that are PSD BY CONSTRUCTION.

The R132/R140 certificates DEFINE the maps by rounding the float Choi matrices and then PROVE them positive definite
(exact Sylvester/Bareiss minors), which is where nearly all of their hours go. Here the certified maps are Gram
matrices of rounded factors, so they are positive semidefinite with no proof at all:
  encoder   X~ = |w~><w~|          w~ = the encoder's Choi vector rounded to the 2^-96 grid (an isometry's Choi
                                    has rank 1)
  decoder   D~_z = K~_z K~_z^dag   K~_z = a rounded factor of the float adjoint-decoder Choi D_z
Everything else is exact rational or 256-bit ball arithmetic:
  - Gershgorin excesses e_E (Tr_in X~ <= (1 + e_E) 1_R) and e_z (Tr_R D~_z <= (1 + e_z) 1) make X~/(1+e_E) and
    D~_z/(1+e_z) trace non-increasing CP maps; completing them to CPTP maps adds a non-negative term to the fidelity.
  - Bob's branch states are built from P's six Kraus operators written exactly (p = sqrt2 - 1, cos/sin pi/8 as exact
    square roots; checked against R139's sclib transcription of the supplement).
  - F = sum_z 2^-n Tr(R~_z D~_z) / (4 (1 + e_z) (1 + e_E)), R~_z = sum_K (1 (x) K) X~ (1 (x) K)^dag.

  python direct_certificate.py N ENCODER.npy OUT.json      (decoders: converged power iteration, stored next to OUT)
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from oneq import superactivation_certificate as sc  # noqa: E402

import direct_codes as dc  # noqa: E402

G = 2 ** sc.GRID_BITS


def exact_p_kraus(flint):
    """P's six Kraus operators on key (x) shield, as 4 x 4 lists of acb (paper supplement l. 371-387, via sclib)."""
    two = flint.arb(2)
    r2 = two.sqrt()
    p = r2 - 1                                            # 1/(1 + sqrt 2) = sqrt 2 - 1 exactly
    c = (two + r2).sqrt() / 2
    s = (two - r2).sqrt() / 2
    one, zero, im = flint.acb(1), flint.acb(0), flint.acb(0, 1)
    I2 = [[one, zero], [zero, one]]
    X = [[zero, one], [one, zero]]
    Y = [[zero, -im], [im, zero]]
    Z = [[one, zero], [zero, -one]]
    K0 = [[one, zero], [zero, zero]]
    K1 = [[zero, zero], [zero, one]]
    M0 = [[flint.acb(c), zero], [zero, flint.acb(s)]]
    M1 = [[flint.acb(s), zero], [zero, -flint.acb(c)]]

    def kron(A, B):
        return [[A[i // 2][j // 2] * B[i % 2][j % 2] for j in range(4)] for i in range(4)]

    def scale(a, M):
        return [[flint.acb(a) * x for x in row] for row in M]
    q2, q4 = ((1 - p) / 2).sqrt(), ((1 - p) / 4).sqrt()
    return [scale(q2, kron(I2, K0)), scale(q4, kron(I2, X)), scale(q2, kron(Z, K1)),
            scale(q4, kron(Z, Y)), scale(p.sqrt(), kron(X, M0)), scale(p.sqrt(), kron(Y, M1))]


def kraus_pattern_and_check(flint, PK):
    """NZ[j][i] = output rows where P's Kraus j has a nonzero in column i -- read from R139's float transcription,
    because comparing a ball with 0 is undecidable. Refuses unless the exact operators' midpoints match it to 1e-15."""
    import sclib
    Kf = sclib.horodecki_kraus()
    worst = 0.0
    for j in range(6):
        for o in range(4):
            for i in range(4):
                mid = complex(float(PK[j][o][i].real.mid()), float(PK[j][o][i].imag.mid()))
                worst = max(worst, abs(mid - Kf[j][o, i]))
    if worst > 1e-15:
        raise sc.CertificateRefusal(f"exact Kraus operators differ from the transcription by {worst:.2e}")
    return [[[o for o in range(4) if abs(Kf[j][o, i]) > 0] for i in range(4)] for j in range(6)]


def _round(z: np.ndarray):
    return (np.rint(z.real * G).astype(object), np.rint(z.imag * G).astype(object))


def encoder_vector(X: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((X + X.conj().T) / 2)
    if w[-2] > 1e-9 * w[-1]:
        raise sc.CertificateRefusal(f"encoder Choi is not rank 1 (second eigenvalue {w[-2]:.3e}): not an isometry")
    return v[:, -1] * np.sqrt(w[-1])


def branch_kraus_vectors(flint, wre, wim, n, z, PK, NZ):
    """u_K = (1_R (x) K) w~ for every Kraus K of branch z, as acb column lists, in bob_state's output order."""
    m_in = dc.DIN ** n
    wv = [flint.acb(flint.arb(flint.fmpq(int(a), G)), flint.arb(flint.fmpq(int(b), G))) for a, b in zip(wre, wim)]
    outs = []
    # per-use operators: kept -> (P_j (x) 1_A) as (4 x 4) on p and identity on a; erased -> P_j on p and <a| on a
    choices = []
    for u in range(n):
        choices.append([(j, None) for j in range(6)] if not z[u] else [(j, a) for j in range(6) for a in range(4)])
    out_dims = [4 * (1 if z[u] else 4) for u in range(n)]
    d_out = int(np.prod(out_dims))
    for combo in itertools.product(*choices):
        u_vec = [flint.acb(0)] * (2 * d_out)
        for r in range(2):
            for idx in range(m_in):
                if int(wre[r * m_in + idx]) == 0 and int(wim[r * m_in + idx]) == 0:
                    continue
                amp = wv[r * m_in + idx]
                digits = []
                t = idx
                for _ in range(n):
                    digits.append(t % dc.DIN)
                    t //= dc.DIN
                digits = digits[::-1]                         # use 0 is the most significant
                # each use: input (pin, ain); P Kraus column pin has entries K[o][pin]
                partial = [(0, amp)]
                for u in range(n):
                    pin, ain = divmod(digits[u], dc.DA)
                    j, a_sel = combo[u]
                    K = PK[j]
                    nxt = []
                    if z[u]:
                        if ain != a_sel:
                            partial = []
                            break
                        for o in NZ[j][pin]:
                            for (base, val) in partial:
                                nxt.append((base * 4 + o, val * K[o][pin]))
                    else:
                        for o in NZ[j][pin]:
                            for (base, val) in partial:
                                nxt.append(((base * 4 + o) * 4 + ain, val * K[o][pin]))
                    partial = nxt
                for base, val in partial:
                    u_vec[r * d_out + base] += val
        outs.append(u_vec)
    return outs, d_out


def certify(n: int, X: np.ndarray, decs: dict, log=print) -> dict:
    flint = sc._flint()
    t0 = time.time()
    PK = exact_p_kraus(flint)
    NZ = kraus_pattern_and_check(flint, PK)
    w = encoder_vector(X)
    wre, wim = _round(w)
    m_in = dc.DIN ** n
    # exact encoder normalization: T = Tr_in |w~><w~|, a 2 x 2 rational matrix
    Tre = [[flint.fmpq(0)] * 2 for _ in range(2)]
    Tim = [[flint.fmpq(0)] * 2 for _ in range(2)]
    for r in range(2):
        for s in range(2):
            acc_re, acc_im = 0, 0
            for i in range(m_in):
                ar, ai = int(wre[r * m_in + i]), int(wim[r * m_in + i])
                br, bi = int(wre[s * m_in + i]), int(wim[s * m_in + i])
                acc_re += ar * br + ai * bi                    # (a)(conj b)
                acc_im += ai * br - ar * bi
            Tre[r][s], Tim[r][s] = flint.fmpq(acc_re, G * G), flint.fmpq(acc_im, G * G)
    eE = sc.gershgorin_excess(flint, Tre, Tim)
    total = flint.arb(0)
    worst = flint.fmpq(0)
    for z in dc.patterns(n):
        D = decs[z][0] if decs[z].ndim == 3 else decs[z]
        dw, dv = np.linalg.eigh((D + D.conj().T) / 2)
        Kf = dv * np.sqrt(np.clip(dw, 0, None))
        Kre, Kim = _round(Kf)
        N = D.shape[0]
        A = flint.fmpz_mat([[int(x) for x in row] for row in Kre])
        B = flint.fmpz_mat([[int(x) for x in row] for row in Kim])
        At, Bt = A.transpose(), B.transpose()
        Dre = A * At + B * Bt                                  # Re(K K^dag) * G^2
        Dim = B * At - A * Bt                                  # Im(K K^dag) * G^2
        d = N // 2
        # Tr_R D~ (R first): sum over r of the (r, r) blocks; Gershgorin excess against 1_d
        ore = [[flint.fmpq(int(Dre[i, j]) + int(Dre[d + i, d + j]), G * G) for j in range(d)] for i in range(d)]
        oim = [[flint.fmpq(int(Dim[i, j]) + int(Dim[d + i, d + j]), G * G) for j in range(d)] for i in range(d)]
        eD = sc.gershgorin_excess(flint, ore, oim)
        worst = max(worst, eD)
        Dacb = flint.acb_mat([[flint.acb(flint.arb(flint.fmpq(int(Dre[i, j]), G * G)),
                                         flint.arb(flint.fmpq(int(Dim[i, j]), G * G))) for j in range(N)]
                              for i in range(N)])
        us, d_out = branch_kraus_vectors(flint, wre, wim, n, z, PK, NZ)
        if 2 * d_out != N:
            raise sc.CertificateRefusal(f"pattern {z}: decoder is {N} square, Bob's space is {2 * d_out}")
        U = flint.acb_mat([[us[k][i] for k in range(len(us))] for i in range(N)])
        DU = Dacb * U
        val = flint.acb(0)
        for k in range(len(us)):
            for i in range(N):
                val += us[k][i].conjugate() * DU[i, k]
        total += val.real / (4 * (1 + flint.arb(eD)))
        if log:
            log(f"  pattern {z}: {len(us)} Kraus vectors, decoder excess {float(eD):.2e} ({time.time() - t0:.0f}s)")
    F = total / flint.arb(2 ** n) / (1 + flint.arb(eE))
    lower = F.lower()
    third = flint.arb(flint.fmpq(3, 4))
    return {"n": n, "certified_lower_bound": lower.str(40), "certified_lower_bound_float": float(lower),
            "exceeds_three_quarters": bool(lower > third), "fidelity_ball": F.str(30),
            "encoder_trace_excess": float(eE), "max_decoder_trace_excess": float(worst),
            "grid_bits": sc.GRID_BITS, "precision_bits": sc.PRECISION_BITS, "seconds": round(time.time() - t0, 1),
            "psd_by_construction": True}


def main(argv) -> int:
    n, wpath, out = int(argv[0]), pathlib.Path(argv[1]), pathlib.Path(argv[2])
    wraw = wpath.read_bytes()
    X = np.load(io.BytesIO(wraw), allow_pickle=False)
    decs, Fp = dc.decode(X, n, dc.p_superop(), iters=6000)
    dpath = out.with_name(out.stem + "_decoders.npz")
    buf = io.BytesIO()
    np.savez(buf, **{"z" + "".join(map(str, z)): decs[z][0] for z in dc.patterns(n)})
    dpath.write_bytes(buf.getvalue())
    draw = dpath.read_bytes()                                  # certify the stored bytes
    with np.load(io.BytesIO(draw), allow_pickle=False) as zf:
        decs = {z: np.array(zf["z" + "".join(map(str, z))]) for z in dc.patterns(n)}
    print(f"float fidelity of the stored pair: {Fp:.12f}", flush=True)
    res = certify(n, X, decs)
    res.update({"encoder": wpath.name, "encoder_sha256": hashlib.sha256(wraw).hexdigest(),
                "decoders": dpath.name, "decoders_sha256": hashlib.sha256(draw).hexdigest(), "pair_float_fidelity": Fp})
    out.write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
