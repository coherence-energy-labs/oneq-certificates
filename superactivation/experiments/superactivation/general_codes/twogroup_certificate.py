"""Rigorous certificate for TWO-GROUP codes (encoder on R (x) Sym^a (x) Sym^b) on N~^{a+b}.

Same proof as oneq.superactivation_certificate.certify (dyadic rounding + eta shift DEFINE the maps; exact Sylvester
positivity; Gershgorin normalization with a fidelity-non-decreasing completion; 256-bit balls), applied to the
two-group reduction (twogroup.py): branch (k1, k2) weight C(a,k1) C(b,k2) / 2^n, observed weights (b1, b2),
selection P^a_{k1,a1} (x) P^b_{k2,a2}, flip coefficients c(k1,b1,a1) c(k2,b2,a2), all as rigorous balls.
Decoders: decs[(k1, k2)] = list over (b1 outer, b2 inner) of adjoint-decoder Chois on R (x) Sym^{a-k1} (x) Sym^{b-k2}.
"""
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]          # the ONE_Q repo root
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation_certificate as sc  # noqa: E402


def positive_definite_bareiss(flint, re, im) -> int:
    """Same verdict as superactivation_certificate.positive_definite (Sylvester: every leading principal minor of the
    real 2d x 2d form > 0, exactly), in ONE O(m^3) pass: Bareiss fraction-free elimination makes the k-th pivot EQUAL
    to the k-th leading principal minor (exact integer division). -1 if proved PD, else the failing minor's index."""
    m = len(re)
    order = [x for i in range(m) for x in (i, i + m)]

    def big(a, b):
        ia, ib = a % m, b % m
        if a < m and b < m:
            return re[ia][ib]
        if a < m:
            return -im[ia][ib]
        if b < m:
            return im[ia][ib]
        return re[ia][ib]
    B = [[big(order[a], order[b]) for b in range(2 * m)] for a in range(2 * m)]
    den = 1
    for row in B:
        for v in row:
            den = math.lcm(den, int(v.q))
    M = [[int(v.p) * (den // int(v.q)) for v in row] for row in B]
    N = 2 * m
    prev = 1
    for k in range(N):
        piv = M[k][k]                      # = leading principal minor of order k + 1 (scaled by den^(k+1) > 0)
        if piv <= 0:
            return k + 1
        for i in range(k + 1, N):
            Mi, Mik = M[i], M[i][k]
            for j in range(k + 1, N):
                Mi[j] = (Mi[j] * piv - Mik * M[k][j]) // prev
        prev = piv
    return -1


def positive_definite_fast(flint, re, im) -> int:
    """flint's C determinants (O(m^4) total but fast constants) on small blocks, Bareiss O(m^3) on big ones; both are
    the same exact Sylvester proof (cross-checked on 40 random blocks, half indefinite)"""
    return sc.positive_definite(flint, re, im) if len(re) <= 40 else positive_definite_bareiss(flint, re, im)


def _selection2(flint, a, b, k1, k2, a1, a2):
    da, db = a - k1 + 1, b - k2 + 1
    ncol = 2 * (a + 1) * (b + 1)
    rows = []
    for r in range(2):
        for i in range(da):
            for j in range(db):
                row = [flint.acb(0)] * ncol
                val = (flint.arb(flint.fmpq(math.comb(a - k1, i), math.comb(a, i + a1))).sqrt()
                       * flint.arb(flint.fmpq(math.comb(b - k2, j), math.comb(b, j + a2))).sqrt())
                row[r * (a + 1) * (b + 1) + (i + a1) * (b + 1) + (j + a2)] = flint.acb(val)
                rows.append(row)
    return flint.acb_mat(rows)


def certify_twogroup(a, b, X, decoders, eta=2.0 ** -34, log=None):
    flint = sc._flint()
    t0 = time.time()
    n = a + b
    m = (a + 1) * (b + 1)
    if X.shape != (2 * m, 2 * m):
        raise sc.CertificateRefusal(f"encoder must be {2 * m} square")
    reX, imX = sc.exact_block(flint, X, eta)
    bad = positive_definite_fast(flint, reX, imX)
    if bad != -1:
        raise sc.CertificateRefusal(f"encoder not proved positive definite (minor {bad})")
    eE = sc.gershgorin_excess(flint, *sc.partial_trace_exact(flint, reX, imX, 2, m, trace_first=False))
    Xb = sc._acb(flint, reX, imX)
    p = flint.arb(2).sqrt() - 1
    total = flint.arb(0)
    worst = flint.fmpq(0)
    proved = 1
    for k1 in range(a + 1):
        for k2 in range(b + 1):
            d = (a - k1 + 1) * (b - k2 + 1)
            Dk = decoders[(k1, k2)]
            if len(Dk) != (k1 + 1) * (k2 + 1):
                raise sc.CertificateRefusal(f"branch ({k1},{k2}) needs {(k1 + 1) * (k2 + 1)} decoders")
            sel = {(a1, a2): _selection2(flint, a, b, k1, k2, a1, a2)
                   for a1 in range(k1 + 1) for a2 in range(k2 + 1)}
            mid = {key: S * Xb * S.transpose() for key, S in sel.items()}
            Fk = flint.arb(0)
            idx = 0
            for b1 in range(k1 + 1):
                for b2 in range(k2 + 1):
                    D = Dk[idx]
                    idx += 1
                    if D.shape != (2 * d, 2 * d):
                        raise sc.CertificateRefusal(f"decoder ({k1},{k2},{b1},{b2}) must be {2 * d} square")
                    reD, imD = sc.exact_block(flint, D, eta)
                    bad = positive_definite_fast(flint, reD, imD)
                    if bad != -1:
                        raise sc.CertificateRefusal(f"decoder ({k1},{k2},{b1},{b2}) not proved PD (minor {bad})")
                    eD = sc.gershgorin_excess(flint, *sc.partial_trace_exact(flint, reD, imD, 2, d, trace_first=True))
                    worst = max(worst, eD)
                    Db = sc._acb(flint, reD, imD)
                    # sum_c Tr(M D) = Tr((sum_c M) D): one ball product per (b1, b2) instead of one per (a1, a2)
                    Msum = None
                    for (a1, a2), M in mid.items():
                        c = sc._coeff(flint, k1, b1, a1, p) * sc._coeff(flint, k2, b2, a2, p)
                        Msum = M * flint.acb(c) if Msum is None else Msum + M * flint.acb(c)
                    val = (Msum * Db).trace()
                    Fk += val.real / (4 * (1 + flint.arb(eD)))
                    proved += 1
            total += flint.arb(flint.fmpq(math.comb(a, k1) * math.comb(b, k2), 2 ** n)) * Fk
            if log:
                log(f"  (k1,k2)=({k1},{k2}) ({time.time() - t0:.0f}s)")
    F = total / (1 + flint.arb(eE))
    lower = F.lower()
    third = flint.arb(flint.fmpq(3, 4))
    return {"a": a, "b": b, "n": n, "certified_lower_bound": lower.str(40),
            "certified_lower_bound_float": float(lower), "exceeds_three_quarters": bool(lower > third),
            "certified_margin_lower_bound": float(lower - third), "fidelity_ball": F.str(30),
            "encoder_trace_excess": float(eE), "max_decoder_trace_excess": float(worst),
            "blocks_proved_positive_definite": proved, "grid_bits": sc.GRID_BITS,
            "precision_bits": sc.PRECISION_BITS, "eta": eta, "seconds": round(time.time() - t0, 1)}


def exact_decoders(X, a, b):
    """exact per-branch SDP decoders for a two-group encoder (keyed as certify_twogroup expects)"""
    import twogroup as tg
    sys.path.insert(0, str(ROOT / "experiments" / "superactivation"))
    from seesaw_lane import exact_decoder
    decs = {}
    for k1, k2, _ in tg.branches(a, b):
        R, _ = tg.bob_states(X, a, b, k1, k2)
        d = (a - k1 + 1) * (b - k2 + 1)
        decs[(k1, k2)] = [exact_decoder(r, d) for r in R]
    return decs


def power_decoders(X, a, b, iters=3000):
    """converged power-iteration decoders: valid CPTP (kernel-completed, full rank) -- enough for a LOWER bound;
    used where the exact per-branch SDP would need a dense multi-GB cone Hessian (n >= ~11)"""
    import numpy as np

    import twogroup as tg
    decs, F = tg.decode(X, a, b, iters=iters)
    delta = 1e-9
    out = {}
    for (k1, k2), Ds in decs.items():
        d = (a - k1 + 1) * (b - k2 + 1)
        D0 = np.kron(np.eye(2), np.eye(d)) / 2                 # completely depolarizing decoder: Tr_R D0 = 1_d
        fixed = []
        for D in Ds:
            w, v = np.linalg.eigh((D + D.conj().T) / 2)
            Dp = (v * np.clip(w, 0, None)) @ v.conj().T        # clip the power iteration's tiny negative eigenvalues
            Dm = (1 - delta) * Dp + delta * D0                 # strictly PD; costs <= delta in fidelity
            # re-impose Tr_R D = 1_d EXACTLY (clipping perturbed it): congruence by 1_R (x) S^{-1/2} keeps D PD, and
            # drives the certificate's Gershgorin excess from ~1e-5 to rounding level
            S = np.einsum("rirj->ij", Dm.reshape(2, d, 2, d))      # Tr_R (r = s); "risj" summed every R block
            sw, sv = np.linalg.eigh((S + S.conj().T) / 2)
            Si = np.kron(np.eye(2), (sv / np.sqrt(sw)) @ sv.conj().T)
            fixed.append(Si @ Dm @ Si)
        out[(k1, k2)] = fixed
    return out, F


# ---------------------------------------------------------------- binding: a certificate names the bytes it proved
#
# MEASURED 2026-10-14 (the independent adversary's A9): the first n = 15 certificate could not be regenerated. It
# certified power decoders computed in memory and never stored, under a different interpreter (a venv with its own
# numpy/BLAS/flint) than the suite's, and its witness hash was added AFTER the run by hashing the file on disk. The
# committed pipeline on the suite's interpreter regenerates a pair whose float fidelity sits 1.4e-9 BELOW the
# recorded bound: the record was the bound of a pair nobody can produce. Hence: the decoders are stored before
# certification, both inputs are hashed from the very bytes that are certified, and the record carries the code
# (git blob ids, which a test requires to be in history) and the interpreter that produced it.

CODE_FILES = ("experiments/superactivation/general_codes/twogroup.py",
              "experiments/superactivation/general_codes/twogroup_certificate.py",
              "src/oneq/superactivation.py",
              "src/oneq/superactivation_certificate.py")


def save_decoders(path, a, b, decs) -> bytes:
    """npz, no pickle: per branch (key k{k1}_{k2}) the UPPER TRIANGLE of herm(D) for each (b1 outer, b2 inner).

    Lossless for the certificate: exact_block reads exactly that triangle of herm(D) (diagonal real) and mirrors it,
    so the rebuilt matrix certifies identically -- at half the bytes (n = 16: ~22 MB instead of ~44 MB).
    """
    import io

    import numpy as np
    packed = {}
    for (k1, k2), Ds in sorted(decs.items()):
        s = Ds[0].shape[0]
        iu = np.triu_indices(s)
        packed[f"k{k1}_{k2}"] = np.stack([((D + D.conj().T) / 2)[iu] for D in Ds])
    buf = io.BytesIO()
    np.savez(buf, a=np.int64(a), b=np.int64(b), **packed)
    raw = buf.getvalue()
    pathlib.Path(path).write_bytes(raw)
    return raw


def load_decoders(raw: bytes):
    """the exactly Hermitian decoders the stored triangles define (lower = conj(upper) bitwise, diagonal real)"""
    import io

    import numpy as np
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        a, b = int(z["a"]), int(z["b"])
        decs = {}
        for k1 in range(a + 1):
            for k2 in range(b + 1):
                s = 2 * (a - k1 + 1) * (b - k2 + 1)
                iu = np.triu_indices(s)
                mats = []
                for p in np.array(z[f"k{k1}_{k2}"]):
                    M = np.zeros((s, s), dtype=complex)
                    M[iu] = p
                    mats.append(M + np.triu(M, 1).conj().T)
                decs[(k1, k2)] = mats
    return a, b, decs


def pair_fidelity(X, a, b, decs) -> float:
    """float fidelity of the (encoder, decoders) pair through the two-group reduction -- the value a certificate of
    this pair must sit just below; a test recomputes it from the stored files"""
    import numpy as np

    import twogroup as tg
    F = 0.0
    for k1, k2, w in tg.branches(a, b):
        R, _ = tg.bob_states(X, a, b, k1, k2)
        F += w * float(np.real(np.einsum("bij,bji->", R, np.stack(decs[(k1, k2)])))) / 4
    return F


def git_blob_id(raw: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def provenance() -> dict:
    import os
    import platform

    import numpy as np
    import scipy
    flint = sc._flint()
    return {"code_git_blobs": {f: git_blob_id((ROOT / f).read_bytes()) for f in CODE_FILES},
            "python": sys.version.split()[0], "executable": pathlib.Path(sys.executable).name,
            "numpy": np.__version__, "scipy": scipy.__version__, "flint": flint.__version__,
            "platform": platform.platform(),
            "thread_env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                           if k in os.environ}}


def main(argv) -> int:
    """twogroup_certificate.py a b WITNESS.npy (power | exact | DECODERS.npz) OUT.json"""
    import hashlib
    import io
    import json

    import numpy as np
    if len(argv) != 5:
        raise SystemExit(main.__doc__)
    a, b = int(argv[0]), int(argv[1])
    wpath, source, out = pathlib.Path(argv[2]), argv[3], pathlib.Path(argv[4])
    wraw = wpath.read_bytes()
    X = np.load(io.BytesIO(wraw), allow_pickle=False)                 # certify exactly the bytes hashed
    if source.endswith(".npz"):
        dpath, mode = pathlib.Path(source), "loaded"
    else:
        dpath = wpath.with_name(f"twogroup_decoders_a{a}_b{b}.npz")
        if source == "power":
            decs, Fp = power_decoders(X, a, b)
            mode = "power iteration, 3000 steps; clipped, 1e-9 depolarizing mix, Tr_R renormalized by congruence"
            print(f"float fidelity with these power decoders: {Fp:.10f}", flush=True)
        elif source == "exact":
            decs, mode = exact_decoders(X, a, b), "exact per-branch SDP"
        else:
            raise SystemExit(f"decoder source must be power, exact or a .npz, got {source!r}")
        save_decoders(dpath, a, b, decs)
    draw = dpath.read_bytes()                                          # re-read: the stored bytes are what is proved
    da, db, decs = load_decoders(draw)
    if (da, db) != (a, b):
        raise SystemExit(f"REFUSING: {dpath.name} holds decoders for ({da},{db}), not ({a},{b})")
    res = certify_twogroup(a, b, X, decs, log=lambda s: None)
    res.update({"witness": wpath.name, "witness_sha256": hashlib.sha256(wraw).hexdigest(),
                "decoders": dpath.name, "decoders_sha256": hashlib.sha256(draw).hexdigest(),
                "decoders_source": mode, "pair_float_fidelity": pair_fidelity(X, a, b, decs),
                "provenance": provenance()})
    out.write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
