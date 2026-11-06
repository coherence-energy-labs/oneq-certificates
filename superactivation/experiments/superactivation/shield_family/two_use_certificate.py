"""Rigorous lower bound on F_c(N~_p^{(x)2}, 2) for an explicit two-use code, p = 1/(sqrt d + 1) exactly.

With R142's chain (P_d is PPT; one use of P_d (x) A_{2d} gives N~_p by per-use operations; neither channel alone
exceeds 3/4), a certified F > 3/4 here IS two-use superactivation for shield dimension d.

The certified maps are PSD BY CONSTRUCTION (as direct_certificate.py): the encoder is the rank-1 Gram |w~><w~| of the
rounded isometry; each decoder is K~ K~^dag for a rounded factor of the float decoder. Gershgorin excesses normalize
both; completing to CPTP adds a non-negative term. The flagged branch's Kraus operators of X^p o Delta,
{sqrt(1-p) |0><0|, sqrt(1-p) |1><1|, sqrt p X|0><0|, sqrt p X|1><1|}, are 256-bit balls with p = 1/(1 + sqrt d).
The two uses are symmetric, so F does not depend on which tensor factor is called the first use.

  python two_use_certificate.py D CODE.npy OUT.json     CODE: a 4 x 2 isometry (columns |0_L>, |1_L>), or 'closed'
                                                        for {|++>, |-->}
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from oneq import superactivation_certificate as sc  # noqa: E402

import onset_vs_p as ovp  # noqa: E402

G = 2 ** sc.GRID_BITS


def closed_code() -> np.ndarray:
    s = 1 / math.sqrt(2)
    plus, minus = np.array([s, s]), np.array([s, -s])
    return np.stack([np.kron(plus, plus), np.kron(minus, minus)], axis=1).astype(complex)


def _rq(x: float) -> int:
    return int(round(x * G))


def certify(V: np.ndarray, d: int, log=print) -> dict:
    if V.shape != (4, 2) or not np.allclose(V.conj().T @ V, np.eye(2), atol=1e-12):
        raise sc.CertificateRefusal("code must be a 4 x 2 isometry")
    flint = sc._flint()
    t0 = time.time()
    p = 1 / (1 + flint.arb(d).sqrt())
    q = 1 - p
    z0, one = flint.acb(0), flint.acb(1)
    sq, sp = flint.acb(q.sqrt()), flint.acb(p.sqrt())
    flag_kraus = [[[sq, z0], [z0, z0]], [[z0, z0], [z0, sq]], [[z0, z0], [sp, z0]], [[z0, sp], [z0, z0]]]
    kept_kraus = [[[one, z0], [z0, one]]]
    w = np.concatenate([V[:, 0], V[:, 1]])                     # Choi vector, R first
    wre = [_rq(x) for x in w.real]
    wim = [_rq(x) for x in w.imag]
    wv = [flint.acb(flint.arb(flint.fmpq(a, G)), flint.arb(flint.fmpq(b, G))) for a, b in zip(wre, wim)]
    # encoder normalization: T = Tr_A |w~><w~| (2 x 2, exact)
    Tre = [[flint.fmpq(sum(wre[r * 4 + i] * wre[s * 4 + i] + wim[r * 4 + i] * wim[s * 4 + i] for i in range(4)), G * G)
            for s in range(2)] for r in range(2)]
    Tim = [[flint.fmpq(sum(wim[r * 4 + i] * wre[s * 4 + i] - wre[r * 4 + i] * wim[s * 4 + i] for i in range(4)), G * G)
            for s in range(2)] for r in range(2)]
    eE = sc.gershgorin_excess(flint, Tre, Tim)
    X = np.outer(w, w.conj())
    pf = 1 / (1 + math.sqrt(d))
    decs, Ff = ovp.decode(X, 2, pf, iters=20000)
    total = flint.arb(0)
    worst = flint.fmpq(0)
    for z in itertools.product((0, 1), repeat=2):
        Dz = decs[z][0]
        dw, dv = np.linalg.eigh((Dz + Dz.conj().T) / 2)
        K = dv * np.sqrt(np.clip(dw, 0, None))
        Kre = [[_rq(x) for x in row] for row in K.real]
        Kim = [[_rq(x) for x in row] for row in K.imag]
        Dre = [[sum(Kre[i][k] * Kre[j][k] + Kim[i][k] * Kim[j][k] for k in range(8)) for j in range(8)] for i in range(8)]
        Dim = [[sum(Kim[i][k] * Kre[j][k] - Kre[i][k] * Kim[j][k] for k in range(8)) for j in range(8)] for i in range(8)]
        ore = [[flint.fmpq(Dre[i][j] + Dre[4 + i][4 + j], G * G) for j in range(4)] for i in range(4)]
        oim = [[flint.fmpq(Dim[i][j] + Dim[4 + i][4 + j], G * G) for j in range(4)] for i in range(4)]
        eD = sc.gershgorin_excess(flint, ore, oim)
        worst = max(worst, eD)
        Dacb = [[flint.acb(flint.arb(flint.fmpq(Dre[i][j], G * G)), flint.arb(flint.fmpq(Dim[i][j], G * G)))
                 for j in range(8)] for i in range(8)]
        val = flint.acb(0)
        for K0 in (flag_kraus if z[0] else kept_kraus):
            for K1 in (flag_kraus if z[1] else kept_kraus):
                u = [flint.acb(0)] * 8                           # (1_R (x) K0 (x) K1) w~
                for r in range(2):
                    for i0 in range(2):
                        for i1 in range(2):
                            amp = wv[r * 4 + i0 * 2 + i1]
                            for o0 in range(2):
                                for o1 in range(2):
                                    u[r * 4 + o0 * 2 + o1] += K0[o0][i0] * K1[o1][i1] * amp
                for i in range(8):
                    for j in range(8):
                        val += u[i].conjugate() * Dacb[i][j] * u[j]
        total += val.real / (4 * (1 + flint.arb(eD)))
    F = total / 4 / (1 + flint.arb(eE))
    lower = F.lower()
    third = flint.arb(flint.fmpq(3, 4))
    res = {"d": d, "p": float(p.mid()), "certified_lower_bound": lower.str(40),
           "certified_lower_bound_float": float(lower), "exceeds_three_quarters": bool(lower > third),
           "certified_margin_lower_bound": float(lower - third), "pair_float_fidelity": Ff,
           "encoder_trace_excess": float(eE), "max_decoder_trace_excess": float(worst),
           "grid_bits": sc.GRID_BITS, "precision_bits": sc.PRECISION_BITS, "psd_by_construction": True,
           "seconds": round(time.time() - t0, 2)}
    if log:
        log(f"d = {d}: float {Ff:.12f}, certified >= {float(lower):.12f}, exceeds 3/4: {res['exceeds_three_quarters']}")
    return res


def main(argv) -> int:
    d, src, out = int(argv[0]), argv[1], pathlib.Path(argv[2])
    if src == "closed":
        V, name, raw = closed_code(), "closed form {|++>, |-->}", None
    else:
        raw = pathlib.Path(src).read_bytes()
        V, name = np.load(io.BytesIO(raw), allow_pickle=False), pathlib.Path(src).name
    res = certify(V, d)
    res.update({"code": name, "code_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
                "code_columns": [[[float(x.real), float(x.imag)] for x in row] for row in V]})
    out.write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
