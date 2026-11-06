"""Dense cross-check of the group reduction behind the symmetric and two-group certificates (paper 4, F1, N6, N7).

The certificates of R132/R135 (symmetric codes, one group) and R140 (two-group codes) compute Bob's branch states in a
REDUCED representation: for an encoder supported on R (x) Sym^a (x) Sym^b, the branch with k1 (k2) flagged uses in
group A (B), summed over observed outcome strings with Hamming weights (b1, b2), is claimed to be

    rho_{b1,b2} = sum_{a1,a2} Pr(b1|a1) C(k1,a1) Pr(b2|a2) C(k2,a2) (1 (x) P^A_{a1} (x) P^B_{a2}) X (.)^T,
    P^m_{k,a}[i, i+a] = sqrt(C(m-k, i) / C(m, i+a)),

whatever positions the flags occupy inside each group. The binding tests recompute fidelities THROUGH that formula, so
they cannot catch an error in it. This script never uses the formula to build anything. It builds the dense 2^n
embedding of R (x) Sym^a (x) Sym^b with numpy alone, permutes the flagged qubits to the front, and for every flagged
string t takes the slice M_t (what the unflagged qubits and R hold given that the flagged qubits read t). Then:

  (1) IDENTITY   M_t, compressed to the Dicke bases of the unflagged parts, equals 1 (x) P^A_{|t_A|} (x) P^B_{|t_B|}
                 of the float reduction (twogroup.sel, superactivation.selection) for EVERY t;
  (2) SUPPORT    M_t lies entirely in R (x) Sym^{a-k1} (x) Sym^{b-k2} (the residual outside it is zero);
  (3) STATES     for a given encoder X, Bob's dense state per observed weights, sum_t Pr(b1, b2 | t) M_t X M_t^T, with
                 Pr(b1, b2 | t) computed for the specific string t from independent flips (a convolution over the
                 flagged qubits; no symmetry used), equals the float reduction's state (twogroup.bob_states,
                 superactivation.bob_states);
  (4) FIDELITY   the code's fidelity computed from the dense states with the STORED decoders equals the certified
                 pair's recorded floating-point fidelity;
  (5) THE CERTIFIER'S OWN SELECTIONS   the midpoints of the ball-arithmetic matrices the certificates actually use
                 (twogroup_certificate._selection2, superactivation_certificate._selection) equal M_t for EVERY t, and
                 their radii are below 1e-70;
  (6) THE CERTIFIER'S OWN COEFFICIENTS the midpoints of superactivation_certificate._coeff(k, b, a, p) (p the ball
                 sqrt 2 - 1 the certifiers use) equal the dense per-string law summed over the strings of weight a, for
                 every group size k up to 24, the largest any committed certificate uses (all 2^k strings enumerated;
                 deviation and radius per string, i.e. divided by C(k, a), radius below 1e-70 on that scale);
  (7) THE CERTIFIERS END TO END on every committed witness that re-certifies in seconds (two-group n = 4, 8;
                 symmetric n = 17, 20, 24; the two-group n = 15, 16 certificates take hours): the certifier, run now,
                 reproduces the committed bound, and the fidelity of the same stored pair sits just above it
                 (0 <= F - bound < 1e-7), F computed from the dense states where they are formed (n <= 17) and the
                 certificate's recorded float fidelity at n = 20, 24.
Flag placements: at n = 12 EVERY one of the 2^n flag patterns is enumerated (no symmetry assumed anywhere, a random
encoder and power-iteration decoders); for the certified witnesses (n = 15, 16 two-group; n = 17 symmetric) each
(k1, k2) is checked at the first positions AND at seeded random positions inside each group, and the fidelity of (4)
uses the random placement with the multiplicity C(a,k1) C(b,k2) that the permutation argument of the paper's lemma
supplies. The symmetric certificates at n = 20 and 24 are past the dense embedding (2 * 2^24 x 50 doubles is 6.7 GB):
for them (1), (2) and (5) are COUNTED over all 2^n strings, one Dicke column at a time (case_symmetric_counted: the
counts are taken from each string's own bits, never from a formula), every flagged string at both placements, and (3)
and (4) are not formed. Negative controls, each a RELATIVE change of 1e-6, must be caught:
  * one amplitude of the float reduction's selection matrix, in the identity comparator's expectation  -> by (1);
  * the same amplitude inside the module that PRODUCES the reduced states (twogroup.bob_states)          -> by (3);
  * the flip probability of the dense construction's per-string law                                     -> by (3);
  * one midpoint of the certifier's own selection matrix                                                -> by (5).

  python verify_reduction_dense.py OUT.json            all cases (measured: 12 to 15 min, peak about 2.0 GB; the
                                                       record's seconds_total and peak_memory_mb hold the committed
                                                       run's own figures, and the paper's generator refuses a record
                                                       outside this line by more than half a minute or 0.25 GB)
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import twogroup as tg  # noqa: E402  (the float reduction under test)
import twogroup_certificate as tc  # noqa: E402  (the two-group certifier under test; also the decoder reader)
from oneq import superactivation as sa  # noqa: E402  (the float one-group reduction under test)
from oneq import superactivation_certificate as sc  # noqa: E402  (the one-group certifier under test)

P = sa.P_FLIP
EV = ROOT / "evidence" / "superactivation"
CODE_FILES = ("experiments/superactivation/general_codes/verify_reduction_dense.py",
              "experiments/superactivation/general_codes/twogroup.py",
              "experiments/superactivation/general_codes/twogroup_certificate.py",
              "src/oneq/superactivation.py", "src/oneq/superactivation_certificate.py")
RADIUS_CAP = 1e-70


# ------------------------------------------------------------------ the dense construction (numpy only)

def popcount(x: np.ndarray, bits: int) -> np.ndarray:
    return sum(((x >> i) & 1) for i in range(bits))


def dicke_matrix(m: int) -> np.ndarray:
    """(2^m, m+1): column w is the normalized Dicke state D^m_w in the computational basis (big-endian)"""
    idx = np.arange(2 ** m)
    w = popcount(idx, m)
    E = np.zeros((2 ** m, m + 1))
    for wt in range(m + 1):
        E[w == wt, wt] = 1 / math.sqrt(math.comb(m, wt))
    return E


def observed_weight_law(tbits: np.ndarray, p: float) -> np.ndarray:
    """L[t, b] = Pr(observed weight b | true string t) for independent flips of probability p on each bit, computed
    by convolution over the bits of each string t (no symmetry: every string is processed on its own)"""
    nt, k = tbits.shape
    L = np.zeros((nt, k + 1))
    L[:, 0] = 1.0
    for i in range(k):
        q1 = np.where(tbits[:, i] == 1, 1 - p, p)          # probability the observed bit i is 1
        hi = i + 2                                          # after i + 1 bits only the weights 0 .. i + 1 can occur
        new = L[:, :hi] * (1 - q1)[:, None]
        new[:, 1:hi] += L[:, :hi - 1] * q1[:, None]
        L[:, :hi] = new
    return L


def bits_of(t: np.ndarray, k: int) -> np.ndarray:
    return np.stack([(t >> (k - 1 - i)) & 1 for i in range(k)], axis=1) if k else np.zeros((len(t), 0), int)


class Dense:
    """the 2^n embedding of R (x) Sym^{g_1} (x) ... (one or two groups), and its slices for a flag placement"""

    def __init__(self, groups: tuple[int, ...]):
        self.groups = groups
        self.n = sum(groups)
        E = np.ones((1, 1))
        for g in groups:
            E = np.kron(E, dicke_matrix(g))
        self.E = np.kron(np.eye(2), E)                          # (2 * 2^n, 2 * prod(g + 1)), R most significant

    def slices(self, flagged: list[int]):
        """Y[t] (rows: R (x) unflagged qubits in order) for every flagged string t, flagged qubits in the given order"""
        n, k = self.n, len(flagged)
        unflag = [q for q in range(n) if q not in flagged]
        T = self.E.reshape([2] + [2] * n + [self.E.shape[1]])
        order = [1 + q for q in flagged] + [0] + [1 + q for q in unflag] + [n + 1]
        return np.ascontiguousarray(T.transpose(order)).reshape(2 ** k, 2 * 2 ** (n - k), self.E.shape[1])


def compress_basis(sizes: tuple[int, ...]) -> np.ndarray:
    B = np.ones((1, 1))
    for g in sizes:
        B = np.kron(B, dicke_matrix(g))
    return np.kron(np.eye(2), B)


# ------------------------------------------------------------------ the certifiers' own ball arithmetic, read out

def acb_mid(M) -> tuple[np.ndarray, float]:
    """(midpoints, largest radius) of a python-flint acb_mat"""
    rows = M.tolist()
    mid = np.array([[complex(float(e.real.mid()), float(e.imag.mid())) for e in row] for row in rows])
    rad = max(max(float(e.real.rad()), float(e.imag.rad())) for row in rows for e in row)
    return mid, rad


class CertifierSelections:
    """midpoints of the matrices the certifiers multiply by (cached per (group sizes, flag counts, weights))"""

    def __init__(self, groups: tuple[int, ...]):
        self.flint = sc._flint()
        self.groups = groups
        self.cache: dict = {}
        self.max_radius = 0.0

    def __call__(self, ks: tuple[int, ...], ws: tuple[int, ...]) -> np.ndarray:
        key = (ks, ws)
        if key not in self.cache:
            if len(self.groups) == 2:
                M = tc._selection2(self.flint, self.groups[0], self.groups[1], ks[0], ks[1], ws[0], ws[1])
            else:
                M = sc._selection(self.flint, self.groups[0], ks[0], ws[0])
            mid, rad = acb_mid(M)
            self.max_radius = max(self.max_radius, rad)
            self.cache[key] = mid
        return self.cache[key]


def certifier_coefficients(kmax: int) -> dict:
    """(6): superactivation_certificate._coeff against the dense per-string law, every k <= kmax"""
    flint = sc._flint()
    p_ball = flint.arb(2).sqrt() - 1                            # the p the certifiers use
    worst, rad, rad_ps, count, strings = 0.0, 0.0, 0.0, 0, 0
    for k in range(kmax + 1):
        agg = np.zeros((k + 1, k + 1))                          # agg[a, b]: sum of Pr(b | t) over the strings t of weight a
        for lo in range(0, 2 ** k, 2 ** 16):                    # every one of the 2^k strings, in chunks
            bits = bits_of(np.arange(lo, min(2 ** k, lo + 2 ** 16)), k)
            L = observed_weight_law(bits, P)
            w = bits.sum(axis=1) if k else np.zeros(len(bits), int)
            agg += (w[None, :] == np.arange(k + 1)[:, None]).astype(float) @ L
            strings += len(bits)
        for a in range(k + 1):
            for b in range(k + 1):
                c = sc._coeff(flint, k, b, a, p_ball)
                # per string of weight a (divide by C(k, a)): a probability, so the tolerance is absolute on [0, 1]
                worst = max(worst, abs(agg[a, b] - float(c.mid())) / math.comb(k, a))
                rad = max(rad, float(c.rad()))                  # absolute: the coefficients reach C(24, 12) ~ 3e6
                rad_ps = max(rad_ps, float(c.rad()) / math.comb(k, a))   # on the scale the deviation is compared on
                count += 1
    return {"group_sizes_up_to": kmax, "coefficients_compared": count, "strings_enumerated": strings,
            "max_deviation": worst, "deviation_scale": "per string, i.e. divided by C(k, a)",
            "max_radius_absolute": rad, "max_radius": rad_ps, "radius_scale": "per string, as the deviation"}


# ------------------------------------------------------------------ one flag placement

def check_placement(dense: Dense, flagged_by_group: list[list[int]], X, red_states, expected_M, p=P,
                    expected_cert=None) -> dict:
    """identity (float twin and, if given, the certifier's own selections), support and states for one placement"""
    flagged = [q for grp in flagged_by_group for q in grp]
    ks = tuple(len(grp) for grp in flagged_by_group)
    k = len(flagged)
    Y = dense.slices(flagged)                                   # (2^k, D, cols)
    Bu = compress_basis(tuple(g - kg for g, kg in zip(dense.groups, ks)))
    M = np.matmul(Bu.T[None, :, :], Y)                          # (2^k, m', cols), real
    resid = 0.0
    for lo in range(0, 2 ** k, 256):                            # chunked: Y - Bu M, never all at once
        R = Y[lo:lo + 256] - np.matmul(Bu[None], M[lo:lo + 256])
        resid = max(resid, float(np.abs(R).max()) if R.size else 0.0)
    t = np.arange(2 ** k)
    laws, weights, shift = [], [], k
    for kg in ks:
        shift -= kg
        tg_bits = bits_of((t >> shift) & ((1 << kg) - 1), kg)
        laws.append(observed_weight_law(tg_bits, p))
        weights.append(tg_bits.sum(axis=1) if kg else np.zeros(len(t), int))
    # (1) and (5): every slice against the splitting identity, one weight class at a time (every t is compared)
    ident, ident_cert = 0.0, 0.0
    W = np.stack(weights, axis=1)
    for ws in {tuple(int(x) for x in row) for row in W}:
        mask = np.all(W == np.array(ws), axis=1)
        ident = max(ident, float(np.abs(M[mask] - expected_M(ws)[None]).max()))
        if expected_cert is not None:
            ident_cert = max(ident_cert, float(np.abs(M[mask] - expected_cert(ks, ws)[None]).max()))
    # (3) G_t = M_t X M_t^T for every t, then rho_b = sum_t Pr(b | t) G_t for every observed-weight tuple b
    G = np.matmul(np.matmul(M, X), np.swapaxes(M, 1, 2))           # (2^k, m', m')
    law = np.ones((len(t), 1))
    for L in laws:                                               # joint law, b_1 outer ... b_G inner
        law = (law[:, :, None] * L[:, None, :]).reshape(len(t), -1)
    states = np.tensordot(law, G, axes=([0], [0]))               # (number of b tuples, m', m')
    sdev = 0.0 if red_states is None else float(np.abs(states - np.asarray(red_states)).max())
    return {"identity": ident, "identity_certifier": ident_cert, "support": resid, "states": sdev, "rho": states}


def git_blob(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def peak_memory_mb() -> int | None:
    """this process's peak memory so far, MiB: the peak working set on Windows (psutil), ru_maxrss elsewhere; None if
    neither is available (the record then says so rather than guessing)"""
    try:
        import psutil
        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        if peak is not None:
            return round(peak / 2 ** 20)
    except ImportError:
        pass
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss       # KiB on Linux, bytes on macOS
    return round(r / 2 ** 20) if sys.platform == "darwin" else round(r / 2 ** 10)


# ------------------------------------------------------------------ the cases

def two_group_expected(a, b, k1, k2, sel=tg.sel):
    cache = {}

    def f(ws):
        if ws not in cache:
            cache[ws] = np.kron(np.eye(2), np.kron(sel(a, k1, ws[0]), sel(b, k2, ws[1])))
        return cache[ws]
    return f


def _worst(acc: dict, out: dict) -> None:
    for key in ("identity", "identity_certifier", "support", "states"):
        acc[key] = max(acc.get(key, 0.0), out[key])


def case_all_patterns(a: int, b: int, seed: int) -> dict:
    """every one of the 2^n flag patterns, a random encoder, power-iteration decoders; no symmetry assumed"""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    X = tg.random_encoder(a, b, rng)
    decs, F_red = tg.decode(X, a, b, iters=60)
    n = a + b
    dense = Dense((a, b))
    cert = CertifierSelections((a, b))
    red = {(k1, k2): tg.bob_states(X, a, b, k1, k2)[0] for k1 in range(a + 1) for k2 in range(b + 1)}
    worst: dict = {}
    F_dense = 0.0
    for z in range(2 ** n):
        FA = [q for q in range(a) if (z >> q) & 1]
        FB = [q for q in range(a, n) if (z >> q) & 1]
        k1, k2 = len(FA), len(FB)
        out = check_placement(dense, [FA, FB], X, red[(k1, k2)], two_group_expected(a, b, k1, k2), expected_cert=cert)
        _worst(worst, out)
        F_dense += float(np.real(np.einsum("bij,bji->", out["rho"], np.stack(decs[(k1, k2)])))) / 4 / 2 ** n
    return {"case": f"random two-group code, a = {a}, b = {b}, every flag pattern", "groups": [a, b], "n": n,
            "encoder": f"twogroup.random_encoder, numpy default_rng({seed})",
            "decoders": "twogroup.decode, 60 power-iteration steps", "flag_patterns_checked": 2 ** n,
            "max_identity_deviation": worst["identity"],
            "max_certifier_selection_deviation": worst["identity_certifier"],
            "max_certifier_selection_radius": cert.max_radius,
            "max_residual_outside_symmetric": worst["support"],
            "max_state_deviation": worst["states"], "fidelity_dense_all_patterns": F_dense,
            "fidelity_reduction": F_red, "fidelity_abs_difference": abs(F_dense - F_red),
            "seconds": round(time.time() - t0, 1)}


def placements(groups, ks, rng):
    """first positions, and seeded random positions, inside each group"""
    starts = np.cumsum((0,) + tuple(groups[:-1]))
    first = [list(range(s, s + kg)) for s, kg in zip(starts, ks)]
    rand = [sorted(int(x) for x in s + rng.choice(g, kg, replace=False)) for s, g, kg in zip(starts, groups, ks)]
    return first, rand


def case_twogroup_witness(ab: str, seed: int) -> dict:
    t0 = time.time()
    cert = json.loads((EV / "general_codes" / f"certificate_twogroup_{ab}.json").read_text(encoding="utf-8"))
    wraw = (EV / "general_codes" / cert["witness"]).read_bytes()
    draw = (EV / "general_codes" / cert["decoders"]).read_bytes()
    assert hashlib.sha256(wraw).hexdigest() == cert["witness_sha256"], "witness bytes are not the certified ones"
    assert hashlib.sha256(draw).hexdigest() == cert["decoders_sha256"], "decoder bytes are not the certified ones"
    X = np.load(io.BytesIO(wraw), allow_pickle=False)
    a, b, decs = tc.load_decoders(draw)
    n = a + b
    rng = np.random.default_rng(seed)
    dense = Dense((a, b))
    certsel = CertifierSelections((a, b))
    worst: dict = {"placement": 0.0}
    F_dense = 0.0
    for k1 in range(a + 1):
        for k2 in range(b + 1):
            red = tg.bob_states(X, a, b, k1, k2)[0]
            exp = two_group_expected(a, b, k1, k2)
            first, rand = placements((a, b), (k1, k2), rng)
            outs = []
            for pl in (first, rand):
                out = check_placement(dense, pl, X, red, exp, expected_cert=certsel)
                _worst(worst, out)
                outs.append(out["rho"])
            worst["placement"] = max(worst["placement"], float(np.abs(outs[0] - outs[1]).max()))
            w = math.comb(a, k1) * math.comb(b, k2) / 2 ** n
            F_dense += w * float(np.real(np.einsum("bij,bji->", outs[1], np.stack(decs[(k1, k2)])))) / 4
    return {"case": f"certified two-group witness {ab} (n = {n})", "groups": [a, b], "n": n,
            "certificate": f"general_codes/certificate_twogroup_{ab}.json", "witness_sha256": cert["witness_sha256"],
            "decoders_sha256": cert["decoders_sha256"],
            "placements_per_branch": "first positions and seeded random positions in each group",
            "random_seed": seed, "max_identity_deviation": worst["identity"],
            "max_certifier_selection_deviation": worst["identity_certifier"],
            "max_certifier_selection_radius": certsel.max_radius,
            "max_residual_outside_symmetric": worst["support"], "max_state_deviation": worst["states"],
            "max_placement_dependence": worst["placement"], "fidelity_dense_stored_decoders": F_dense,
            "fidelity_recorded_pair": cert["pair_float_fidelity"],
            "fidelity_abs_difference": abs(F_dense - cert["pair_float_fidelity"]),
            "seconds": round(time.time() - t0, 1)}


def case_symmetric_witness(nn: int, seed: int) -> dict:
    t0 = time.time()
    cert = json.loads((EV / f"certificate_n{nn}.json").read_text(encoding="utf-8"))
    wpath = EV / cert["witness"]
    assert hashlib.sha256(wpath.read_bytes()).hexdigest() == cert["witness_sha256"], "witness is not the certified one"
    n, X, decs = sc.load_witness(wpath)
    rng = np.random.default_rng(seed)
    dense = Dense((n,))
    certsel = CertifierSelections((n,))
    worst: dict = {"placement": 0.0}
    F_dense = 0.0
    for k in range(n + 1):
        red = sa.bob_states(X, n, k)
        outs = []
        for pl in placements((n,), (k,), rng):
            out = check_placement(dense, pl, X, red, lambda ws, k=k: sa.selection(n, k, ws[0]),
                                  expected_cert=certsel)
            _worst(worst, out)
            outs.append(out["rho"])
        worst["placement"] = max(worst["placement"], float(np.abs(outs[0] - outs[1]).max()))
        F_dense += math.comb(n, k) / 2 ** n * float(np.real(np.einsum("bij,bji->", outs[1], np.stack(decs[k])))) / 4
    return {"case": f"certified symmetric witness (n = {n})", "groups": [n], "n": n,
            "certificate": f"certificate_n{nn}.json", "witness_sha256": cert["witness_sha256"],
            "placements_per_branch": "first positions and seeded random positions", "random_seed": seed,
            "max_identity_deviation": worst["identity"],
            "max_certifier_selection_deviation": worst["identity_certifier"],
            "max_certifier_selection_radius": certsel.max_radius,
            "max_residual_outside_symmetric": worst["support"],
            "max_state_deviation": worst["states"], "max_placement_dependence": worst["placement"],
            "fidelity_dense_stored_decoders": F_dense, "fidelity_recorded_pair": cert["float_fidelity"],
            "fidelity_abs_difference": abs(F_dense - cert["float_fidelity"]), "seconds": round(time.time() - t0, 1)}


NOT_FORMED = "not formed at this n (2^n too large); the slices are counted instead"


def case_symmetric_counted(nn: int, seed: int) -> dict:
    """(1), (2) and (5) for a certified symmetric witness whose 2 * 2^n x 2(n+1) embedding is too large to form
    (n = 20, 24). D^n_w is 1/sqrt(C(n, w)) on the strings of weight w and 0 elsewhere, so its slice by a flagged string
    t, compressed to the Dicke basis of the unflagged qubits, is a COUNT over all 2^n strings:
        M_t[i, w] = #{strings of weight w whose flagged part is t and unflagged part has weight i} / sqrt(C(n,w) C(n-k,i)),
    and the residual outside Sym^{n-k} follows exactly from the same counts (in the class (t, i) of C(n-k, i) entries,
    c of them on the support, each support entry keeps 1 - c/C(n-k, i) of its value and each other entry c/C(n-k, i)).
    No formula for the counts is used: every string is assigned to its class by its own bits. R carries the identity,
    so the float selection (1) and the certifier's own (5) are compared block by block: both diagonal R blocks with
    M_t, the off-diagonal ones with zero. Every flagged string t, at the first positions and at seeded random positions.
    The states and the fidelity, (3) and (4), are not formed here."""
    t0 = time.time()
    cert = json.loads((EV / f"certificate_n{nn}.json").read_text(encoding="utf-8"))
    wpath = EV / cert["witness"]
    assert hashlib.sha256(wpath.read_bytes()).hexdigest() == cert["witness_sha256"], "witness is not the certified one"
    n = nn
    rng = np.random.default_rng(seed)
    certsel = CertifierSelections((n,))
    idx = np.arange(2 ** n, dtype=np.int64)
    bit = [((idx >> (n - 1 - q)) & 1).astype(np.int8) for q in range(n)]   # bit of qubit q in every string
    del idx
    weight = np.zeros(2 ** n, np.int8)
    for b in bit:
        weight += b
    on = [np.flatnonzero(weight == w) for w in range(n + 1)]                 # the support of D^n_w
    del weight
    worst = {"identity": 0.0, "identity_certifier": 0.0, "support": 0.0}
    strings = 0
    for k in range(n + 1):
        d = n - k + 1
        norm = np.array([1 / math.sqrt(math.comb(n - k, i)) for i in range(d)])
        t_w = popcount(np.arange(2 ** k), k) if k else np.zeros(1, np.int64)
        fl_exp = np.stack([sa.selection(n, k, a) for a in range(k + 1)])     # (k+1, 2d, 2(n+1))
        ce_exp = np.stack([certsel((k,), (a,)) for a in range(k + 1)])
        for E, key in ((fl_exp, "identity"), (ce_exp, "identity_certifier")):
            off = max(float(np.abs(E[:, :d, n + 1:]).max()), float(np.abs(E[:, d:, :n + 1]).max()))
            twin = float(np.abs(E[:, :d, :n + 1] - E[:, d:, n + 1:]).max())
            worst[key] = max(worst[key], off, twin)
        for pl in placements((n,), (k,), rng):
            fl = pl[0]
            unfl = [q for q in range(n) if q not in fl]
            cls = np.zeros(2 ** n, np.int64)                                  # t * d + (weight of the unflagged part)
            for i, q in enumerate(fl):
                cls += bit[q].astype(np.int64) << (k - 1 - i)
            cls *= d
            for q in unfl:
                cls += bit[q]
            size = np.bincount(cls, minlength=2 ** k * d).reshape(2 ** k, d)
            for w in range(n + 1):
                c = np.bincount(cls[on[w]], minlength=2 ** k * d).reshape(2 ** k, d)
                Mw = c * (norm[None, :] / math.sqrt(math.comb(n, w)))      # M_t[:, w] for every t
                worst["identity"] = max(worst["identity"], float(np.abs(Mw - fl_exp[t_w, :d, w]).max()))
                worst["identity_certifier"] = max(worst["identity_certifier"],
                                                  float(np.abs(Mw - ce_exp[t_w, :d, w]).max()))
                frac = c * norm[None, :] ** 2                                # c / C(n-k, i)
                kept = np.where(c > 0, 1 - frac, 0.0)                         # on the support
                leak = np.where((c > 0) & (size > c), frac, 0.0)              # off the support, same class
                worst["support"] = max(worst["support"],
                                       float(max(np.abs(kept).max(), leak.max())) / math.sqrt(math.comb(n, w)))
            strings += 2 ** k
    return {"case": f"certified symmetric witness (n = {n}), slices counted over all 2^n strings", "groups": [n],
            "n": n, "certificate": f"certificate_n{nn}.json", "witness_sha256": cert["witness_sha256"],
            "placements_per_branch": "first positions and seeded random positions", "random_seed": seed,
            "flagged_strings_checked": strings, "max_identity_deviation": worst["identity"],
            "max_certifier_selection_deviation": worst["identity_certifier"],
            "max_certifier_selection_radius": certsel.max_radius,
            "max_residual_outside_symmetric": worst["support"], "states_and_fidelity": NOT_FORMED,
            "seconds": round(time.time() - t0, 1)}


def end_to_end(case: dict) -> dict:
    """(7): run the certifier now on the case's committed witness; it must reproduce the committed bound, and the
    dense fidelity of the same stored pair must sit just above that bound"""
    t0 = time.time()
    rel = case["certificate"]
    rec = json.loads((EV / rel).read_text(encoding="utf-8"))
    if len(case["groups"]) == 2:
        X = np.load(EV / "general_codes" / rec["witness"], allow_pickle=False)
        a, b, decs = tc.load_decoders((EV / "general_codes" / rec["decoders"]).read_bytes())
        res = tc.certify_twogroup(a, b, X, decs)
    else:
        n, X, decs = sc.load_witness(EV / rec["witness"])
        res = sc.certify(n, X, decs)
    L = res["certified_lower_bound_float"]
    out = {"certificate": rel, "n": case["n"], "certified_now": res["certified_lower_bound"],
           "certified_committed": rec["certified_lower_bound"],
           "reproduces_committed_bound": res["certified_lower_bound"] == rec["certified_lower_bound"]}
    if case.get("states_and_fidelity") == NOT_FORMED:          # no dense states: the pair's recorded float fidelity
        F = rec["float_fidelity"]
        out.update({"fidelity_recorded": F, "recorded_minus_bound": F - L})
    else:
        F = case["fidelity_dense_stored_decoders"]
        out.update({"fidelity_dense": F, "dense_minus_bound": F - L})
    out["seconds"] = round(time.time() - t0, 1)
    return out


def negative_controls(a: int, b: int, seed: int) -> dict:
    """each comparator must catch a RELATIVE 1e-6 error (n = a + b, one branch is enough)"""
    rng = np.random.default_rng(seed)
    X = tg.random_encoder(a, b, rng)
    dense = Dense((a, b))
    k1, k2 = 2, 3
    FA, FB = list(range(k1)), list(range(a, a + k2))
    orig_sel = tg.sel

    def bad_sel(m, k, a1):
        S = orig_sel(m, k, a1)
        if (m, k, a1) == (a, k1, 1):
            S = S.copy()
            S[0, 1] *= 1 + 1e-6
        return S
    red = tg.bob_states(X, a, b, k1, k2)[0]
    good = two_group_expected(a, b, k1, k2)
    # (i) the identity comparator's expectation perturbed
    ident_bad = check_placement(dense, [FA, FB], X, red, two_group_expected(a, b, k1, k2, sel=bad_sel))["identity"]
    # (ii) the module that PRODUCES the reduced states perturbed (bob_states calls twogroup.sel)
    tg.sel = bad_sel
    try:
        red_bad = tg.bob_states(X, a, b, k1, k2)[0]
    finally:
        tg.sel = orig_sel
    state_module_bad = check_placement(dense, [FA, FB], X, red_bad, good)["states"]
    # (iii) the dense construction's flip probability perturbed
    flip_bad = check_placement(dense, [FA, FB], X, red, good, p=P * (1 + 1e-6))["states"]
    # (iv) one midpoint of the certifier's own selection matrix perturbed
    cert = CertifierSelections((a, b))

    def bad_cert(ks, ws):
        S = cert(ks, ws)
        if ws == (1, 0):
            S = S.copy()
            i, j = np.unravel_index(np.argmax(np.abs(S)), S.shape)
            S[i, j] *= 1 + 1e-6
        return S
    cert_bad = check_placement(dense, [FA, FB], X, red, good, expected_cert=bad_cert)["identity_certifier"]
    return {"relative_size": 1e-6,
            "identity_comparator_expectation_perturbed_identity_deviation": ident_bad,
            "state_producing_module_perturbed_state_deviation": state_module_bad,
            "dense_flip_probability_perturbed_state_deviation": flip_bad,
            "certifier_selection_midpoint_perturbed_deviation": cert_bad}


def main(argv) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out = pathlib.Path(argv[0])
    t0 = time.time()
    cases = [case_all_patterns(5, 7, seed=12)]
    print(json.dumps(cases[-1]), flush=True)
    for ab in ("a2_b2", "a4_b4", "a7_b8", "a8_b8"):
        cases.append(case_twogroup_witness(ab, seed=15))
        print(json.dumps(cases[-1]), flush=True)
    cases.append(case_symmetric_witness(17, seed=17))
    print(json.dumps(cases[-1]), flush=True)
    for nn in (20, 24):
        cases.append(case_symmetric_counted(nn, seed=nn))
        print(json.dumps(cases[-1]), flush=True)
    coeffs = certifier_coefficients(max(max(c["groups"]) for c in cases))
    print(json.dumps(coeffs), flush=True)
    e2e = [end_to_end(c) for c in cases if c.get("certificate") in (
        "general_codes/certificate_twogroup_a2_b2.json", "general_codes/certificate_twogroup_a4_b4.json",
        "certificate_n17.json", "certificate_n20.json", "certificate_n24.json")]
    print(json.dumps(e2e), flush=True)
    controls = negative_controls(5, 7, seed=3)
    print(json.dumps(controls), flush=True)
    tol_exact, tol_fid = 1e-12, 1e-12
    dev_keys = ("max_identity_deviation", "max_certifier_selection_deviation", "max_residual_outside_symmetric",
                "max_state_deviation")
    noise = max(max(c[k] for k in dev_keys if k in c) for c in cases)

    def case_ok(c: dict) -> bool:
        ok = (c["max_identity_deviation"] < tol_exact and c["max_residual_outside_symmetric"] < tol_exact
              and c.get("max_placement_dependence", 0) < tol_exact
              and c["max_certifier_selection_deviation"] < tol_exact and c["max_certifier_selection_radius"] < RADIUS_CAP)
        if c.get("states_and_fidelity") == NOT_FORMED:          # only the counted cases may omit (3) and (4)
            return ok
        return ok and c["max_state_deviation"] < tol_exact and c["fidelity_abs_difference"] < tol_fid

    def e2e_ok(e: dict) -> bool:
        gap = e["recorded_minus_bound"] if "recorded_minus_bound" in e else e["dense_minus_bound"]
        return e["reproduces_committed_bound"] and 0 <= gap < 1e-7
    verdict = (all(case_ok(c) for c in cases)
               and coeffs["max_deviation"] < tol_exact and coeffs["max_radius"] < RADIUS_CAP
               and len(e2e) == 5 and all(e2e_ok(e) for e in e2e)
               and all(v > 1e3 * noise for k, v in controls.items() if k != "relative_size"))
    rec = {"schema": "oneq.superactivation.reduction-dense-check/2",
           "claim": "the group reduction (Dicke splitting identity, support, branch states, fidelity), the certifiers' "
                    "own selection matrices and coefficients at every group size a committed certificate uses, and "
                    "the certifiers end to end, agree with the dense 2^n construction",
           "supersedes": "reduction_dense_check.json (schema 1: the float reduction only)",
           "tolerance_exact_quantities": tol_exact, "tolerance_fidelity": tol_fid, "radius_cap": RADIUS_CAP,
           "flip_probability": P, "cases": cases, "certifier_coefficients": coeffs, "certifier_end_to_end": e2e,
           "negative_controls": controls, "verdict": "AGREE" if verdict else "DISAGREE",
           "seconds_total": round(time.time() - t0, 1), "peak_memory_mb": peak_memory_mb(),
           "provenance": {"code_git_blobs": {f: git_blob(ROOT / f) for f in CODE_FILES},
                          "python": platform.python_version(), "numpy": np.__version__,
                          "platform": platform.platform()}}
    out.write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print("VERDICT", rec["verdict"], "in", rec["seconds_total"], "s")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
