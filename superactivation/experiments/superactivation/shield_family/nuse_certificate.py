"""EXACT lower bound on F_c(N~_p^{(x)n}, 2) for an explicit n-use code, p = 1/(sqrt d + 1) exactly (any n; used at
n = 3, 4, where no other certifier in the repo applies).

Why a new certifier: two_use_certificate.py is n = 2 only; superactivation_certificate.certify (Sym^n codes) and
general_codes/twogroup_certificate.py (Sym^a (x) Sym^b codes) hard-code d = 2's flip probability sqrt 2 - 1, and
kgroup.py, whose m = (1, ..., 1) case is every n-use code, has no certificate. This one generalizes
two_use_certificate.py's design to every n in the dense model (every encoder, every decoder):

  * the certified maps are PSD BY CONSTRUCTION and DEFINED by stored bytes: the encoder is the rank-1 Gram
    |w~><w~| of the code's Choi vector w = sum_r |r>_R (x) V|r> rounded to the 2^-96 grid; the decoder of flag
    pattern z is D~_z = K~_z K~_z^dag for the STORED factor K_z rounded to the same grid (decoders as in
    onset_vs_p.decode: D on R (x) A^n, R first, Tr_R D = 1, F_z = Tr(sigma_z D_z) / 4 for the Tr-2 branch state);
  * normalization: Gershgorin excesses e_E of Tr_A X~ and e_z of Tr_R D~_z (exact rationals); X~/(1+e_E) and
    D~_z/(1+e_z) are trace-non-increasing CP maps, and completing either to CPTP adds a non-negative term;
  * the channel enters EXACTLY: on the flagged uses F_z, (X^p o Delta)^{(x) F_z} expands as
        N_z(X~) = sum_{S subset F_z} p^|S| (1-p)^(|F_z|-|S|) X_S Delta_F(X~) X_S,
    and t_{z,S} = Tr(X_S Delta_F(X~) X_S D~_z) = sum_f ||K~_z^dag u_{S,f}||^2 (u_{S,f}: w~ with its S bits
    flipped, restricted to flagged bits = f) is an exact non-negative integer at scale 2^-384;
  * so F~ = (1/(1+e_E)) sum_z 2^-n / (4 (1+e_z)) sum_S p^|S| (1-p)^(k-|S|) t_{z,S} is a polynomial in p with exact
    rational coefficients, evaluated EXACTLY in Q(sqrt d) (p = (sqrt d - 1)/(d - 1)) as a + b sqrt d, and
    F~ > 3/4 is decided exactly (the sign of (a - 3/4) + b sqrt d, by squaring). No float enters the verdict:
    floats only PROPOSE the code and the decoders; their stored bytes define what is proved.

With R142's chain (P_d is PPT; one use of P_d (x) A_{2d} gives N~_p by per-use operations; neither channel alone
exceeds 3/4) a certified F~ > 3/4 IS n-use superactivation for shield dimension d.

  python nuse_certificate.py certify D N CODE.npy (power | DECODERS.npz) OUT.json
        power: decoders by onset_vs_p.decode (20000 warm power steps), factored, STORED as
        nuse_decoders_d{D}_n{N}.npz beside CODE, re-read, and the stored bytes certified
  python nuse_certificate.py verify OUT.json     re-hash both inputs, re-certify, compare exactly with the record
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
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "general_codes"))

GRID_BITS = 96
G = 2 ** GRID_BITS
THREE_QUARTERS = Fraction(3, 4)
CODE_FILES = ("experiments/superactivation/shield_family/nuse_certificate.py",
              "experiments/superactivation/shield_family/onset_vs_p.py",
              "experiments/superactivation/general_codes/general_codes.py",
              "src/oneq/superactivation.py")
IMPORTED = ["arXiv 2604.27042 Prop. 1 (generalized, R142): F_c((P_d (x) A_{2d})^n, 2) >= F_c(N~_p^n, 2)",
            "arXiv 2604.27042 Result 1: F_c(P^n, 2) <= 1/2 and F_c(A^n, 2) <= 3/4 for every n"]


class CertificateRefusal(ValueError):
    """The inputs do not define the maps (or the claim) the proof can stand on."""


# ------------------------------------------------------------------------------------------- Q(sqrt d), exactly

def _qmul(x, y, d):
    return (x[0] * y[0] + x[1] * y[1] * d, x[0] * y[1] + x[1] * y[0])


def _qpow(x, k, d):
    out = (Fraction(1), Fraction(0))
    for _ in range(k):
        out = _qmul(out, x, d)
    return out


def sign_of(a: Fraction, b: Fraction, d: int) -> int:
    """the exact sign of a + b sqrt d (d >= 0)"""
    sa, sb = (a > 0) - (a < 0), (b > 0) - (b < 0)
    if sb == 0 or d == 0:
        return sa
    if sa == 0 or sa == sb:
        return sb
    lhs, rhs = a * a, b * b * d                 # opposite signs: the larger magnitude wins
    return sa if lhs > rhs else (sb if rhs > lhs else 0)


def lower_decimal(a: Fraction, b: Fraction, d: int, digits: int = 40) -> str:
    """a decimal string <= a + b sqrt d (floor at `digits` places, sqrt d bracketed by integer isqrt)"""
    N = 10 ** (digits + 20)
    s_lo = math.isqrt(d * N * N)
    lo = a + min(b * Fraction(s_lo, N), b * Fraction(s_lo + 1, N))
    q = math.floor(lo * 10 ** digits)
    sgn = "-" if q < 0 else ""
    q = abs(q)
    return f"{sgn}{q // 10 ** digits}.{q % 10 ** digits:0{digits}d}"


# ------------------------------------------------------------------------------------------------ the certificate

def _ints(A: np.ndarray):
    if not np.all(np.isfinite(A)):
        raise CertificateRefusal("non-finite entry")
    re = np.array([[int(round(float(x) * float(G))) for x in row] for row in np.real(A)], dtype=object)
    im = np.array([[int(round(float(x) * float(G))) for x in row] for row in np.imag(A)], dtype=object)
    return re, im


def _gershgorin_excess(re, im, scale) -> Fraction:
    """exact e >= 0 with lambda_max(M) <= 1 + e for the Hermitian M = (re + i im) / scale (|z| <= |Re z| + |Im z|)"""
    m = re.shape[0]
    best = max(re[i, i] + sum(abs(re[i, j]) + abs(im[i, j]) for j in range(m) if j != i) for i in range(m))
    e = Fraction(best, scale) - 1
    return e if e > 0 else Fraction(0)


def _gram(Are, Aim):
    """(A A^dag) as exact integer (re, im) parts"""
    re = Are.dot(Are.T) + Aim.dot(Aim.T)
    im = Aim.dot(Are.T) - Are.dot(Aim.T)
    return re, im


def flag_patterns(n):
    return list(itertools.product((0, 1), repeat=n))


def zkey(z) -> str:
    return "z" + "".join(map(str, z))


def certify(V: np.ndarray, decoders: dict, d: int, n: int) -> dict:
    """V: 2^n x 2 isometry (columns |0_L>, |1_L>); decoders: {z: K_z (2^(n+1) x r)}, D_z = K_z K_z^dag"""
    if not isinstance(d, int) or d < 2 or not isinstance(n, int) or n < 1:
        raise CertificateRefusal(f"need integers d >= 2, n >= 1 (got d = {d!r}, n = {n!r})")
    m = 2 ** n
    V = np.asarray(V, dtype=complex)
    if V.shape != (m, 2) or not np.allclose(V.conj().T @ V, np.eye(2), atol=1e-10):
        raise CertificateRefusal(f"the code must be a {m} x 2 isometry")
    zs = flag_patterns(n)
    if set(decoders) != set(zs):
        raise CertificateRefusal(f"need one decoder per flag pattern ({len(zs)}), got {len(decoders)}")
    t0 = time.time()
    w = np.concatenate([V[:, 0], V[:, 1]])                          # Choi vector, R first: index r 2^n + i
    wre, wim = _ints(w[None, :])
    wre, wim = wre[0], wim[0]
    Tre, Tim = _gram(wre.reshape(2, m), wim.reshape(2, m))          # Tr_A |w~><w~| (2 x 2) at scale G^2
    eE = _gershgorin_excess(Tre, Tim, G * G)
    # qubit q of A is bit (n - 1 - q) of the A index (general_codes' reshape order, q0 most significant)
    bit = [1 << (n - 1 - q) for q in range(n)]
    p = (Fraction(-1, d - 1), Fraction(1, d - 1))                   # p = (sqrt d - 1)/(d - 1)
    one_minus_p = (Fraction(d, d - 1), Fraction(-1, d - 1))         # 1 - p = (d - sqrt d)/(d - 1)
    total = (Fraction(0), Fraction(0))
    worst = Fraction(0)
    idx_all = np.arange(2 * m)
    per_z = {}
    for z in zs:
        K = np.asarray(decoders[z], dtype=complex)
        if K.ndim != 2 or K.shape[0] != 2 * m or not 1 <= K.shape[1] <= 2 * m:
            raise CertificateRefusal(f"decoder {zkey(z)} must be a {2 * m} x r factor, got {K.shape}")
        Kre, Kim = _ints(K)
        Sre, Sim = _gram(Kre[:m], Kim[:m])                          # Tr_R (K K^dag) = sum_r K_r K_r^dag
        S2re, S2im = _gram(Kre[m:], Kim[m:])
        ez = _gershgorin_excess(Sre + S2re, Sim + S2im, G * G)
        worst = max(worst, ez)
        Fmask = sum(bit[q] for q in range(n) if z[q])
        k = sum(z)
        subs = [s for s in range(m) if s & Fmask == s]              # every subset of the flagged bits
        KdRe, KdIm = Kre.T, Kim.T                                   # K^dag = KdRe - i KdIm
        coeff = [0] * (k + 1)                                       # c_j = sum_{|S| = j} t_{z,S} at scale G^4
        for S in subs:
            ure, uim = wre[idx_all ^ S], wim[idx_all ^ S]           # X_S w~ (the flip acts on A bits only)
            t = 0
            for f in subs:
                sel = idx_all[(idx_all & Fmask) == f]
                a = KdRe[:, sel].dot(ure[sel]) + KdIm[:, sel].dot(uim[sel])
                b = KdRe[:, sel].dot(uim[sel]) - KdIm[:, sel].dot(ure[sel])
                t += int(sum(x * x for x in a)) + int(sum(x * x for x in b))
            coeff[bin(S).count("1")] += t
        val = (Fraction(0), Fraction(0))
        for j, c in enumerate(coeff):
            if c:
                term = _qmul(_qpow(p, j, d), _qpow(one_minus_p, k - j, d), d)
                cf = Fraction(c, G ** 4)
                val = (val[0] + cf * term[0], val[1] + cf * term[1])
        scale = Fraction(1, 2 ** n * 4) / (1 + ez)
        total = (total[0] + scale * val[0], total[1] + scale * val[1])
        per_z[zkey(z)] = float(scale * (val[0] + val[1] * Fraction(math.sqrt(d))))
    a, b = total[0] / (1 + eE), total[1] / (1 + eE)
    margin_sign = sign_of(a - THREE_QUARTERS, b, d)
    low = lower_decimal(a, b, d)
    return {"d": d, "n": n, "p": 1 / (1 + math.sqrt(d)),
            "certified_value_exact": {"a": str(a), "b": str(b), "value": f"a + b sqrt({d})"},
            "certified_lower_bound": low, "certified_lower_bound_float": float(Fraction(low)),
            "exceeds_three_quarters": margin_sign > 0,
            "certified_margin_lower_bound": float(Fraction(low) - THREE_QUARTERS),
            "branch_values_float": per_z, "encoder_trace_excess": float(eE), "max_decoder_trace_excess": float(worst),
            "grid_bits": GRID_BITS, "psd_by_construction": True,
            "verdict_arithmetic": "exact: Python integers and Q(sqrt d); no floating point in the verdict",
            "imported_theorems": IMPORTED, "seconds": round(time.time() - t0, 2)}


# ------------------------------------------------------------------------------------------- inputs, bytes, record

def load_code(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def save_decoders(path: pathlib.Path, d: int, n: int, decoders: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, d=np.int64(d), n=np.int64(n), **{zkey(z): np.asarray(K, dtype=complex) for z, K in decoders.items()})
    raw = buf.getvalue()
    path.write_bytes(raw)
    return raw


def load_decoders(raw: bytes):
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        d, n = int(f["d"]), int(f["n"])
        decs = {z: np.array(f[zkey(z)]) for z in flag_patterns(n) if zkey(z) in f.files}
        extra = set(f.files) - {"d", "n"} - {zkey(z) for z in flag_patterns(n)}
    if extra:
        raise CertificateRefusal(f"unexpected entries in the decoder file: {sorted(extra)}")
    return d, n, decs


def power_decoders(V: np.ndarray, d: int, n: int, iters: int = 20000) -> tuple[dict, float]:
    """float decoders by the repo's power iteration (onset_vs_p.decode), factored D = K K^dag (eigenvalues
    <= 1e-14 of the largest dropped: a deficit in Tr_R D costs nothing, the completion pays it)"""
    import onset_vs_p as ovp
    w = np.concatenate([V[:, 0], V[:, 1]])
    decs, F = ovp.decode(np.outer(w, w.conj()), n, 1 / (1 + math.sqrt(d)), iters=iters)
    out = {}
    for z, D in decs.items():
        ev, U = np.linalg.eigh((D[0] + D[0].conj().T) / 2)
        keep = ev > 1e-14 * ev.max()
        out[z] = U[:, keep] * np.sqrt(ev[keep])
    return out, F


def pair_fidelity(V: np.ndarray, decoders: dict, d: int, n: int) -> float:
    """float F of the stored (code, decoders) through the dense model: the value the certificate must sit just below"""
    import onset_vs_p as ovp
    w = np.concatenate([V[:, 0], V[:, 1]])
    p = 1 / (1 + math.sqrt(d))
    F = 0.0
    for z, R in ovp.branch_states(np.outer(w, w.conj()), n, p).items():
        K = decoders[z]
        F += 2.0 ** -n * float(np.real(np.trace(R @ (K @ K.conj().T)))) / 4
    return F


def git_blob_id(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def provenance() -> dict:
    import platform
    return {"code_git_blobs": {f: git_blob_id((ROOT / f).read_bytes()) for f in CODE_FILES},
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}


def certify_files(d: int, n: int, code_path: pathlib.Path, dec_path: pathlib.Path, source: str) -> dict:
    craw, draw = code_path.read_bytes(), dec_path.read_bytes()      # certify exactly the bytes hashed
    V = load_code(craw)
    dd, dn, decs = load_decoders(draw)
    if (dd, dn) != (d, n):
        raise CertificateRefusal(f"{dec_path.name} holds decoders for (d, n) = ({dd}, {dn}), not ({d}, {n})")
    res = certify(V, decs, d, n)
    res.update({"code": code_path.name, "code_sha256": hashlib.sha256(craw).hexdigest(),
                "code_columns": [[[float(x.real), float(x.imag)] for x in row] for row in np.asarray(V, complex)],
                "decoders": dec_path.name, "decoders_sha256": hashlib.sha256(draw).hexdigest(),
                "decoders_source": source, "pair_float_fidelity": pair_fidelity(V, decs, d, n),
                "provenance": provenance()})
    return res


def verify(record_path: pathlib.Path) -> dict:
    """re-derive a stored certificate from the bytes it names; refuse on ANY disagreement with the record"""
    rec = json.loads(record_path.read_text(encoding="utf-8"))
    code_path, dec_path = record_path.with_name(rec["code"]), record_path.with_name(rec["decoders"])
    for path, key in ((code_path, "code_sha256"), (dec_path, "decoders_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != rec[key]:
            raise CertificateRefusal(f"{path.name}: sha256 differs from the record's {key}")
    res = certify_files(int(rec["d"]), int(rec["n"]), code_path, dec_path, rec["decoders_source"])
    for key in ("certified_value_exact", "certified_lower_bound", "exceeds_three_quarters"):
        if res[key] != rec[key]:
            raise CertificateRefusal(f"{record_path.name}: recomputed {key} {res[key]!r} != recorded {rec[key]!r}")
    if not 0 <= res["pair_float_fidelity"] - res["certified_lower_bound_float"] < 1e-8:
        raise CertificateRefusal(f"{record_path.name}: the bound is not just below its own pair's float fidelity")
    return res


def main(argv) -> int:
    if argv[:1] == ["verify"] and len(argv) == 2:
        res = verify(pathlib.Path(argv[1]))
        print(f"VERIFIED {argv[1]}: d = {res['d']}, n = {res['n']}, certified >= {res['certified_lower_bound']}, "
              f"exceeds 3/4: {res['exceeds_three_quarters']}")
        return 0
    if argv[:1] != ["certify"] or len(argv) != 6:
        raise SystemExit(__doc__)
    d, n = int(argv[1]), int(argv[2])
    code_path, source, out = pathlib.Path(argv[3]), argv[4], pathlib.Path(argv[5])
    if source == "power":
        V = load_code(code_path.read_bytes())
        decs, Ff = power_decoders(V, d, n)
        dec_path = code_path.with_name(f"nuse_decoders_d{d}_n{n}.npz")
        save_decoders(dec_path, d, n, decs)
        mode = "onset_vs_p.decode power iteration, 20000 steps; factored, eigenvalues <= 1e-14 x max dropped"
        print(f"float fidelity with these power decoders: {Ff:.12f}", flush=True)
    else:
        dec_path, mode = pathlib.Path(source), "loaded"
    res = certify_files(d, n, code_path, dec_path, mode)
    out.write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(f"d = {d}, n = {n}: pair float {res['pair_float_fidelity']:.12f}, certified >= "
          f"{res['certified_lower_bound']}, exceeds 3/4: {res['exceeds_three_quarters']} ({res['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
