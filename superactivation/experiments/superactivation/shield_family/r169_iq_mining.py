"""R169: the raw IQ and the twirl randomizations of R161+R158, R165 and R152, mined (the owner's items 12-15).

Every analysis reads ONLY committed evidence (flight records, frozen records, results): no network, no quota.
EXPLORATORY: none of it is a preregistered verdict. Each finding carries its evidence class in the JSON it writes.

1. COHERENT vs STOCHASTIC error (item 12).
   A twirled job runs each circuit as R randomizations, each a fresh draw of a Pauli frame around every CZ, its shots
   stored randomization-major (the span slices say so; twirl_shape() refuses anything else). A Pauli-channel error
   commutes with every frame, so all R randomizations share ONE expectation and their means scatter binomially. A
   coherent error is conjugated by each layer's frame into a random sign: its angle random-walks, each randomization
   has its own expectation, and the R means scatter MORE than binomially. For a Z-type rotation theta per layer on an
   anticommuting observable the excess, relative to the mean, is
       kappa^2 = Var_r<P> / E_r<P>^2 = cosh(d theta^2) - 1  (small angles; exact form in coherent_z_expectations)
   so it grows as d^2 -- against d for a non-unital (T1-type) frame effect and d^0 for drift between randomizations.
   homogeneity()        per-randomization means, Pearson X against the PERMUTATION null (shots exchangeable across
                        randomizations; exact mean, Monte-Carlo variance), the excess variance tau^2 and kappa^2
   coherent_z_expectations / simulate_twirled_ladder   the coherent model, and an explicit frame-by-frame
                        simulation that checks it (positive control) and gives the Pauli null
   fit_z_generators     (theta_ZI, theta_IZ, theta_ZZ) per edge from the observable-resolved kappa^2
   ladder_points / basis_chi2   R146's basis-consistency chi2 (hw_diag_run, unchanged helpers) re-weighted by the
                        measured randomization variance: does the "not a Pauli channel" chi2 survive it?
2. SOFT READOUT (item 13).
   frame()              each qubit's discriminator frame: (u, v) in pooled-sd units, |0> at (0, 0), |1> at (D, 0)
   lda_posterior        P(1 | IQ) of the job's own Gaussian LDA: sigmoid(D (u - D/2))
   readout_mixture_fit  a 1-D model of the |0>- and |1>-prepared shots: two cores, a wrong-core fraction e and a
                        uniform BRIDGE (decay during the integration window) of mass B; B T1 is the window it implies
   linear_estimator_variance / likelihood_information   per-shot variance of hard, soft and likelihood readouts
3. A |2> CLUSTER (item 14).
   mixture_em           EM over 2-D Gaussians (free or fixed) and a fixed-shape bridge; BIC; a free EXTRA component
                        over the readout template is the leakage detector, its weight against depth the per-CZ rate
4. IQ-LEVEL READOUT CROSSTALK (item 15).
   conditional_shift    qubit A's core-trimmed IQ mean given B's classified state, against the exact permutation null
                        that shuffles B within each (circuit, randomization) block
   pattern_shift        A's cluster centroid between two readout patterns, paired by randomization

  python r169_iq_mining.py [--quick]      -> evidence/superactivation/r169_iq_mining/*.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_diag_run as D146  # noqa: E402

from oneq.hw import cz_channel as CC  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "r169_iq_mining"
R161_DIR = ROOT / "evidence" / "superactivation" / "r161_spectator_cz" / "job_20261017T124413Z"
R161_JOB = "db9q3bs24mrc7384ecn0"
R165_DIR = ROOT / "evidence" / "superactivation" / "r165_dressing_bench"
R165_JOBS = {"active": "db9s4p424mrc7384gskg", "active-accum": "db9s4r8c77vs7381ug60"}
R165_RECORD = "frozen_20261017T174831Z"
R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
R152_JOB = "db9c50oc77vs73817rog"
LAYER_S = 132e-9          # R166 MEASURED: every twirled CZ layer plays as a 64 ns two-sx slot + the 68 ns CZ
NULL_DRAWS = 600
AMBIGUOUS = 0.05          # a shot is ambiguous when its LDA posterior lies in [0.05, 0.95]
CORE = 3.0                # crosstalk: a shot is in its class's core within 3 pooled sd of the class centre


class MiningRefused(ValueError):
    """data that cannot be read faithfully by an analysis"""


# ================================================================== 0. shots and randomizations
def twirl_shape(spans: list, pub_index: int, n_shots: int) -> tuple:
    """(randomizations, shots per randomization) of one PUB, read off the execution-span SLICES: exactly one span must
    cover the PUB, as one randomization-major block (shape (R, n), flattened, both slices full)"""
    found = []
    for s in spans:
        sl = (s.get("slices") or {}).get(str(pub_index))
        if sl is None:
            continue
        shape, flat, rs, ss = sl
        if not flat or len(shape) != 2 or list(rs) != [0, shape[0], None] or list(ss) != [0, shape[1], None]:
            raise MiningRefused(f"PUB {pub_index}: slice {sl} is not one randomization-major block")
        found.append((int(shape[0]), int(shape[1])))
    if len(found) != 1:
        raise MiningRefused(f"PUB {pub_index}: {len(found)} spans carry its slice; exactly one is required")
    R, n = found[0]
    if R * n != n_shots:
        raise MiningRefused(f"PUB {pub_index}: slice {R} x {n} does not cover its {n_shots} shots")
    return R, n


def unpack_bits(packed, num_bits: int) -> np.ndarray:
    """a BitArray's packed bytes (shots, bytes; big-endian, clbit 0 the low bit of the LAST byte) -> (shots, num_bits)
    booleans, clbit 0 first"""
    a = np.asarray(packed, dtype=np.uint8)
    if a.ndim != 2 or a.shape[1] * 8 < num_bits:
        raise MiningRefused(f"packed bits of shape {a.shape} cannot hold {num_bits} bits")
    return np.unpackbits(a[:, ::-1], axis=1, bitorder="little")[:, :num_bits].astype(bool)


# ================================================================== 1. overdispersion
_NULL_CACHE: dict = {}


def _null_moments(K: int, n: int, R: int, draws: int, seed: int = 169) -> tuple:
    """mean and variance of the Pearson statistic when K of the R x n shots are +1 and the shots are exchangeable
    across randomizations (multivariate hypergeometric: the permutation null)"""
    key = (K, n, R, draws)
    if key not in _NULL_CACHE:
        N = R * n
        if K in (0, N):
            _NULL_CACHE[key] = (0.0, 0.0)
        else:
            rng = np.random.default_rng([seed, K, n, R])
            k = rng.multivariate_hypergeometric([n] * R, K, size=draws)
            m = (2 * k - n) / n
            mb = (2 * K - N) / N
            X = n * ((m - mb) ** 2).sum(axis=1) / (1 - mb * mb)
            _NULL_CACHE[key] = (float(X.mean()), float(X.var(ddof=1)))
    return _NULL_CACHE[key]


def homogeneity(y, R: int, draws: int = NULL_DRAWS) -> dict:
    """y: one observable's +-1 value per shot, randomization-major. Pearson X of the R randomization means against
    the permutation null; tau2 = the per-randomization variance of the TRUE expectation beyond binomial (unbiased under
    the null); kappa2 = tau2 / mean^2; ratio = X / E0[X]"""
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N % R or N < 2 * R:
        raise MiningRefused(f"{N} shots do not split into {R} randomizations")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise MiningRefused("homogeneity reads +-1 values")
    n = N // R
    m_r = y.reshape(R, n).mean(axis=1)
    mb = float(y.mean())
    K = int(round((N + y.sum()) / 2))
    v1 = 1 - mb * mb
    if v1 <= 0:
        return dict(n=n, R=R, mean=mb, m_r=m_r.tolist(), X=0.0, E0=0.0, V0=0.0, ratio=float("nan"), tau2=0.0,
                    sigma2=0.0, kappa2=0.0, degenerate=True)
    X = float(n * ((m_r - mb) ** 2).sum() / v1)
    E0 = (R - 1) * N / (N - 1)                         # exact under the permutation null
    _, V0 = _null_moments(K, n, R, draws)
    sigma2 = v1 / n
    tau2 = sigma2 * (X / (R - 1) - N / (N - 1))
    return dict(n=n, R=R, mean=mb, m_r=m_r.tolist(), X=X, E0=E0, V0=V0, ratio=X / E0, tau2=float(tau2),
                sigma2=float(sigma2), kappa2=float(tau2 / (mb * mb)) if abs(mb) > 0 else float("nan"),
                degenerate=False)


def pool(rows: list) -> dict:
    """the pooled overdispersion of many series: sum X against sum E0, its z from the null variances; the pooled
    kappa2 = sum tau2 / sum mean^2 and its sd from each tau2's sampling variance 2 (sigma2 + tau2)^2 / (R - 1)"""
    rows = [r for r in rows if not r["degenerate"]]
    if not rows:
        return dict(series=0)
    X = sum(r["X"] for r in rows)
    E = sum(r["E0"] for r in rows)
    V = sum(r["V0"] for r in rows)
    t2 = sum(r["tau2"] for r in rows)
    m2 = sum(r["mean"] ** 2 for r in rows)
    k2 = max(t2 / m2, 0.0) if m2 > 0 else 0.0          # each series' tau2 variance at the POOLED kappa2, not its own
    vt = sum(2 * (r["sigma2"] + k2 * r["mean"] ** 2) ** 2 / (r["R"] - 1) for r in rows)
    return dict(series=len(rows), ratio=X / E, ratio_sd_null=math.sqrt(V) / E, z=(X - E) / math.sqrt(V) if V > 0
                else float("nan"), kappa2=t2 / m2 if m2 > 0 else float("nan"),
                kappa2_sd=math.sqrt(vt) / m2 if m2 > 0 else float("nan"), mean_abs=float(np.mean(
                    [abs(r["mean"]) for r in rows])))


def depth_exponent(k_lo: float, sd_lo: float, k_hi: float, sd_hi: float, d_lo: int, d_hi: int) -> dict:
    """gamma in kappa2 ~ d^gamma between two depths (coherent: ~2; non-unital frame effect: ~1; drift: ~0), with its
    delta-method sd; refused (None) unless both kappa2 are > 2 sd"""
    if not (k_lo > 2 * sd_lo and k_hi > 2 * sd_hi):
        return dict(gamma=None, reason="kappa2 not resolved at both depths")
    g = math.log(k_hi / k_lo) / math.log(d_hi / d_lo)
    sd = math.hypot(sd_hi / k_hi, sd_lo / k_lo) / math.log(d_hi / d_lo)
    return dict(gamma=g, sd=sd, coherent_prediction=math.log((d_hi ** 2 - d_hi) / (d_lo ** 2 - d_lo)) /
                math.log(d_hi / d_lo))


# ------------------------------------------------------------------ the coherent model and its check
_P1 = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
       "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.diag([1.0, -1.0]).astype(complex)}
_EIG = {"Z": np.array([1, 0], dtype=complex), "X": np.array([1, 1], dtype=complex) / math.sqrt(2),
        "Y": np.array([1, 1j], dtype=complex) / math.sqrt(2)}
_ZA = np.array([1, 1, -1, -1], dtype=float)        # |00>, |01>, |10>, |11>, qubit a the high index
_ZB = np.array([1, -1, 1, -1], dtype=float)
CZ4 = np.diag([1, 1, 1, -1]).astype(complex)


def observable_class(label: str) -> str:
    """which qubits of a two-qubit Pauli label carry X or Y: 'z' (none -- the CZ-fixed ZI, IZ, ZZ), 'a', 'b', 'ab'.
    Under Z-type coherent error, class a is hit by ZI and ZZ, class b by IZ and ZZ, class ab by ZI and IZ"""
    xa, xb = label[0] in "XY", label[1] in "XY"
    return {(False, False): "z", (True, False): "a", (False, True): "b", (True, True): "ab"}[(xa, xb)]


def _signs(n_mc: int, d: int, seed: int) -> tuple:
    rng = np.random.default_rng([seed, d])
    s = rng.choice([-1.0, 1.0], size=(n_mc, d, 2))
    return s[..., 0].sum(axis=1), s[..., 1].sum(axis=1), (s[..., 0] * s[..., 1]).sum(axis=1)


def coherent_z_expectations(basis: str, d: int, angles, n_mc: int = 4000, seed: int = 169, signs=None) -> np.ndarray:
    """(n_mc, 3): the exact per-randomization expectations of a ladder basis's three observables (a x I, I x b, a x b)
    after d twirled CZ layers, each carrying the coherent error exp(-i (tZI ZI + tIZ IZ + tZZ ZZ) / 2). Z-type errors
    commute with the CZ, so the frames only flip their signs: ZI's when the frame has X/Y on a, IZ's on b, ZZ's on
    exactly one -- the angles are random walks S_a, S_b, S_c = sum s_a s_b. Pass `signs` to reuse random numbers."""
    a, b, c = (float(x) for x in angles)
    Sa, Sb, Sc = signs if signs is not None else _signs(n_mc, d, seed)
    phase = -(a * Sa[:, None] * _ZA + b * Sb[:, None] * _ZB + c * Sc[:, None] * _ZA * _ZB) / 2
    psi = np.kron(_EIG[basis[0]], _EIG[basis[1]])[None, :] * np.exp(1j * phase)
    out = []
    for lab in CC.observables(basis):
        O = np.kron(_P1[lab[0]], _P1[lab[1]])
        out.append(np.real(np.einsum("ri,ij,rj->r", psi.conj(), O, psi)))
    return np.stack(out, axis=1)


def kappa2_coherent(basis: str, d: int, angles, n_mc: int = 4000, seed: int = 169, signs=None) -> np.ndarray:
    """the model's kappa2 of the basis's three observables"""
    e = coherent_z_expectations(basis, d, angles, n_mc, seed, signs)
    m = e.mean(axis=0)
    return e.var(axis=0) / np.where(np.abs(m) > 1e-12, m * m, np.nan)


def _pauli4(k: int) -> np.ndarray:
    return np.kron(_P1["IXYZ"[k // 4]], _P1["IXYZ"[k % 4]])


def simulate_twirled_ladder(basis: str, d: int, n_rand: int, rng, unitary=None, pauli_probs=None) -> np.ndarray:
    """(n_rand, 3): an EXPLICIT frame-by-frame simulation of d twirled CZ layers -- a uniform two-qubit Pauli P before
    each CZ, the error after it (inside the frame), P' = CZ P CZ after -- and the three observables' exact
    expectations per randomization. `unitary` (4x4) is a coherent error; `pauli_probs` ({label: p}) a Pauli channel,
    the null: it commutes with every frame, so every randomization must give the same expectations."""
    psi0 = np.kron(_EIG[basis[0]], _EIG[basis[1]])
    rho0 = np.outer(psi0, psi0.conj())
    obs = [np.kron(_P1[lab[0]], _P1[lab[1]]) for lab in CC.observables(basis)]
    paulis = [_pauli4(k) for k in range(16)]
    chan = None
    if pauli_probs:
        chan = [(p, np.kron(_P1[q[0]], _P1[q[1]])) for q, p in pauli_probs.items()]
        chan.append((1 - sum(pauli_probs.values()), np.eye(4, dtype=complex)))
    out = np.empty((n_rand, 3))
    for r in range(n_rand):
        rho = rho0.copy()
        for _ in range(d):
            P = paulis[rng.integers(16)]
            Pp = CZ4 @ P @ CZ4
            rho = CZ4 @ P @ rho @ P.conj().T @ CZ4.conj().T
            if unitary is not None:
                rho = unitary @ rho @ unitary.conj().T
            if chan is not None:
                rho = sum(p * Q @ rho @ Q.conj().T for p, Q in chan)
            rho = Pp @ rho @ Pp.conj().T
        out[r] = [np.real(np.trace(O @ rho)) for O in obs]
    return out


def sample_shots(expect, n: int, rng) -> np.ndarray:
    """+-1 shots, randomization-major: n per randomization from its expectation"""
    e = np.clip(np.asarray(expect, dtype=float), -1, 1)
    return np.concatenate([np.where(rng.random(n) < (1 + x) / 2, 1.0, -1.0) for x in e])


def fit_z_generators(points: list, n_mc: int = 4000, seed: int = 169, bound: float = 0.3) -> dict:
    """(theta_ZI, theta_IZ, theta_ZZ) per layer from observable-resolved spreads. points = [dict(basis, obs_index, d,
    T = sum of the series' tau2, m2 = [mean^2 per series], s2 = [sigma2 per series], R)]. Gaussian likelihood of T
    with the MODEL's variance, sum 2 (sigma2 + kappa2_model mean^2)^2 / (R - 1): weights from a series' own tau2
    would favour series that came out low by chance. Exact model, common random numbers; each angle's 68% interval
    from its conditional profile (delta -2 log L = 1). Sign-blind: the angles are magnitudes."""
    from scipy.optimize import minimize
    pts = [p for p in points if p["d"] > 0 and p["m2"]]
    if not pts:
        return dict(points=0)
    signs = {d: _signs(n_mc, d, seed) for d in sorted({p["d"] for p in pts})}
    keys = sorted({(p["basis"], p["d"]) for p in pts})
    m2 = [np.asarray(p["m2"]) for p in pts]
    s2 = [np.asarray(p["s2"]) for p in pts]

    def terms(theta):
        th = np.abs(theta)
        model = {k: kappa2_coherent(k[0], k[1], th, signs=signs[k[1]]) for k in keys}
        out = []
        for p, a, s in zip(pts, m2, s2):
            k2 = max(float(model[(p["basis"], p["d"])][p["obs_index"]]), 0.0)
            var = float((2 * (s + k2 * a) ** 2 / (p["R"] - 1)).sum())
            out.append(((p["T"] - k2 * a.sum()) ** 2 / var, math.log(var)))
        return out

    def nll(theta):
        return float(sum(a + b for a, b in terms(theta)))
    best = None
    for start in ([0.02, 0.02, 0.02], [0.08, 0.01, 0.01], [0.01, 0.08, 0.01], [0.01, 0.01, 0.08]):
        r = minimize(nll, start, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-5, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    th = np.minimum(np.abs(best.x), bound)
    n0 = nll(th)
    intervals = []
    for i in range(3):
        grid = np.unique(np.clip(np.concatenate([np.linspace(0, bound, 61), th[i] + np.linspace(-0.03, 0.03, 121)]),
                                 0, bound))
        prof = []
        for g in grid:
            t = th.copy()
            t[i] = g
            prof.append(nll(t))
        ok = grid[np.array(prof) <= n0 + 1.0]
        intervals.append([float(ok.min()), float(ok.max())] if ok.size else [float(th[i]), float(th[i])])
    chi2 = float(sum(a for a, _ in terms(th)))
    return dict(theta=dict(ZI=float(th[0]), IZ=float(th[1]), ZZ=float(th[2])),
                interval_68=dict(ZI=intervals[0], IZ=intervals[1], ZZ=intervals[2]), chi2=chi2, dof=len(pts) - 3,
                points=len(pts), n_mc=n_mc, sign="undetermined: kappa2 depends on theta^2 only")


# ------------------------------------------------------------------ R146's basis chi2, re-weighted
def ladder_points(bits_of: dict, cols: tuple, ro_a: dict, ro_b: dict, depths, R: int) -> list:
    """per (observable, basis, depth) of one edge: R146's readout-corrected correlator and its binomial variance
    (hw_diag_run's own helpers), and the randomization homogeneity of the RAW observable. bits_of[(basis, d)] is the
    circuit's (shots, clbits) booleans; cols the edge's two clbits"""
    i, j = cols
    sa = 1 - ro_a["p01"] - ro_a["p10"]
    sb = 1 - ro_b["p01"] - ro_b["p10"]
    out = []
    for basis in CC.BASES:
        for d in depths:
            b = bits_of[(basis, d)]
            yi, yj = 1.0 - 2.0 * b[:, i], 1.0 - 2.0 * b[:, j]
            yij = yi * yj
            tot = len(b)
            mi, mj, mij = float(yi.mean()), float(yj.mean()), float(yij.mean())
            ci, cj = D146._readout_inverse(mi, ro_a), D146._readout_inverse(mj, ro_b)
            cij = D146._pair_inverse(mij, ci, cj, ro_a, ro_b)
            for k, (P, c, v, y) in enumerate(zip(CC.observables(basis), (ci, cj, cij),
                                                 (D146._corr_var(mi, tot, sa), D146._corr_var(mj, tot, sb),
                                                  D146._corr_var(mij, tot, sa * sb)), (yi, yj, yij))):
                h = homogeneity(y, R)
                out.append(dict(P=P, basis=basis, d=d, obs_index=k, cls=observable_class(P), c=c, var=v,
                                mean=h["mean"], n=h["n"], R=R, ratio=h["ratio"], tau2=h["tau2"], sigma2=h["sigma2"],
                                kappa2=h["kappa2"], X=h["X"], E0=h["E0"], V0=h["V0"], degenerate=h["degenerate"]))
    return out


def correlator_points(bits_of: dict, cols: tuple, ro_a: dict, ro_b: dict, depths) -> list:
    """ladder_points without the spread statistics: R146's corrected correlators and binomial variances only"""
    i, j = cols
    sa = 1 - ro_a["p01"] - ro_a["p10"]
    sb = 1 - ro_b["p01"] - ro_b["p10"]
    out = []
    for basis in CC.BASES:
        for d in depths:
            b = bits_of[(basis, d)]
            yi, yj = 1.0 - 2.0 * b[:, i], 1.0 - 2.0 * b[:, j]
            tot = len(b)
            mi, mj, mij = float(yi.mean()), float(yj.mean()), float((yi * yj).mean())
            ci, cj = D146._readout_inverse(mi, ro_a), D146._readout_inverse(mj, ro_b)
            cij = D146._pair_inverse(mij, ci, cj, ro_a, ro_b)
            for k, (P, c, v) in enumerate(zip(CC.observables(basis), (ci, cj, cij),
                                              (D146._corr_var(mi, tot, sa), D146._corr_var(mj, tot, sb),
                                               D146._corr_var(mij, tot, sa * sb)))):
                out.append(dict(P=P, basis=basis, d=d, obs_index=k, c=c, var=v))
    return out


def channel_p(points: list) -> tuple:
    """R146's total CZ error p and channel from the points, by fit_cz_channels' own algebra (each series' log-slope
    with a free intercept, inverse-variance pooled per learnable quantity, the symmetric-gauge channel projected
    physical)"""
    series: dict = {}
    for p in points:
        series.setdefault((p["P"], p["basis"]), []).append((p["d"], p["c"], p["var"]))
    per_key: dict = {}
    for (P, basis), pts in series.items():
        per_key.setdefault(CC.learnable_key(P), []).append(D146._log_slope(sorted(pts)))
    g = {}
    for k, v in per_key.items():
        w = np.array([1 / max(sd, 1e-9) ** 2 for _, sd in v])
        g[k] = math.exp(2 * float((w * np.array([s for s, _ in v])).sum() / w.sum()))
    ch = CC.project_physical(CC.symmetric_channel(g))
    return 1 - ch["II"], ch


def randomization_bootstrap(bits_of: dict, cols: tuple, ro_a: dict, ro_b: dict, depths, R: int, n_boot: int,
                            seed: int = 169) -> dict:
    """p's sd INCLUDING the twirl's sampling: each circuit's R randomization blocks resampled with replacement and the
    whole estimator re-run; the variance scaled by R / (R - 1) (the bootstrap's own small-sample bias). Every
    correlation between observables sharing a randomization is kept."""
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(n_boot):
        bo = {}
        for key, b in bits_of.items():
            n = len(b) // R
            bo[key] = b.reshape(R, n, -1)[rng.integers(R, size=R)].reshape(R * n, -1)
        ps.append(channel_p(correlator_points(bo, cols, ro_a, ro_b, depths))[0])
    ps = np.array(ps)
    return dict(p=channel_p(correlator_points(bits_of, cols, ro_a, ro_b, depths))[0],
                p_sd=float(ps.std(ddof=1) * math.sqrt(R / (R - 1))), n_boot=n_boot)


def basis_chi2(points: list, phi=None) -> dict:
    """R146's estimator's consistency chi2 (fit_cz_channels: each (observable, basis) series' log-slope with a FREE
    intercept, pooled per learnable quantity) with every point's variance multiplied by phi(point) (None: 1, the
    binomial weights R146 used)"""
    series: dict = {}
    for p in points:
        f = 1.0 if phi is None else float(phi(p))
        series.setdefault((p["P"], p["basis"]), []).append((p["d"], p["c"], p["var"] * f))
    per_key: dict = {}
    for (P, basis), pts in series.items():
        per_key.setdefault(CC.learnable_key(P), []).append(D146._log_slope(sorted(pts)))
    chi2 = 0.0
    contrib = {}
    for k, v in per_key.items():
        w = np.array([1 / max(sd, 1e-9) ** 2 for _, sd in v])
        s = np.array([sl for sl, _ in v])
        m = float((w * s).sum() / w.sum())
        c = float((w * (s - m) ** 2).sum())
        chi2 += c
        contrib[str(k)] = dict(chi2=c, dof=len(v) - 1)
    from scipy.stats import chi2 as C2
    dof = len(series) - len(per_key)
    return dict(chi2=chi2, dof=dof, p=float(C2.sf(chi2, dof)), per_key=contrib)


def phi_model(theta: dict, n_mc: int = 4000, seed: int = 169):
    """phi(point) = 1 + n kappa2_model mean^2 / (1 - mean^2): the variance inflation a coherent error of these angles
    puts on a point measured over R randomizations of n shots (the full-circuit mean's variance is
    (1 - mean^2) / N + tau2 / R)"""
    th = (theta["ZI"], theta["IZ"], theta["ZZ"])
    cache: dict = {}

    def phi(p):
        if p["d"] == 0:
            return 1.0
        key = (p["basis"], p["d"])
        if key not in cache:
            cache[key] = kappa2_coherent(p["basis"], p["d"], th, n_mc, seed)
        k2 = float(cache[key][p["obs_index"]])
        v1 = max(1 - p["mean"] ** 2, 1e-6)
        return 1.0 + p["n"] * k2 * p["mean"] ** 2 / v1
    return phi


# ================================================================== 2. soft readout
def frame(disc) -> dict:
    """the discriminator frame: whitened by the pooled covariance and rotated, so |0>'s mean sits at (0, 0) and |1>'s
    at (D, 0), D the clusters' Mahalanobis separation (= the LDA d')"""
    S = np.array(disc.cov, dtype=float).reshape(2, 2)
    ev, U = np.linalg.eigh(S)
    W = U @ np.diag(ev ** -0.5) @ U.T
    mu0 = np.array([disc.mu0.real, disc.mu0.imag])
    mu1 = np.array([disc.mu1.real, disc.mu1.imag])
    e = W @ (mu1 - mu0)
    D = float(np.linalg.norm(e))
    Rot = np.array([[e[0], e[1]], [-e[1], e[0]]]) / D
    return dict(origin=mu0, M=Rot @ W, D=D)


def to_frame(fr: dict, iq) -> np.ndarray:
    z = np.asarray(iq)
    x = np.stack([z.real, z.imag], axis=-1).astype(float) - fr["origin"]
    return x @ fr["M"].T


def lda_posterior(fr: dict, uv, prior1: float = 0.5) -> np.ndarray:
    """P(1 | IQ) of the LDA model (two Gaussians of the pooled covariance): sigmoid(D (u - D/2) + logit(prior))"""
    D = fr["D"]
    t = D * (np.asarray(uv)[..., 0] - D / 2) + math.log(prior1 / (1 - prior1))
    return 1 / (1 + np.exp(-np.clip(t, -700, 700)))


def linear_estimator_variance(s0, s1, p: float) -> float:
    """per-shot variance of p_hat = (mean s - a0) / (a1 - a0) at true p, for scores s (hard: 0/1; soft: P(1|IQ))
    whose moments on prepared-0 and prepared-1 shots are those of s0 and s1"""
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    a0, a1 = s0.mean(), s1.mean()
    q0, q1 = (s0 ** 2).mean(), (s1 ** 2).mean()
    es = (1 - p) * a0 + p * a1
    var = (1 - p) * q0 + p * q1 - es * es
    return float(var / (a1 - a0) ** 2)


def effective_error(v_half: float) -> float:
    """the symmetric assignment error whose hard readout has per-shot variance v_half at p = 1/2:
    0.25 / (1 - 2 eps)^2 = v_half"""
    return float((1 - math.sqrt(0.25 / v_half)) / 2) if v_half >= 0.25 else 0.0


def _norm_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def _bridge_pdf(x, m0, m1, s):
    from scipy.special import ndtr
    return (ndtr((x - m0) / s) - ndtr((x - m1) / s)) / (m1 - m0)


def mixture_pdf(u, prm: dict, prep: int) -> np.ndarray:
    """the 1-D readout model of shots prepared in `prep`: its own core, the other core (fraction e), and a uniform
    bridge between the cores (fraction B: a state change during the integration window), noise-convolved"""
    m0, m1, s0, s1 = prm["m0"], prm["m1"], prm["s0"], prm["s1"]
    e, B = prm[f"e{prep}"], prm[f"B{prep}"]
    own, other = ((m0, s0), (m1, s1)) if prep == 0 else ((m1, s1), (m0, s0))
    return ((1 - e - B) * _norm_pdf(u, *own) + e * _norm_pdf(u, *other)
            + B * _bridge_pdf(u, m0, m1, (s0 + s1) / 2))


def readout_mixture_fit(u0, u1, D: float, fix_B1_zero: bool = False) -> dict:
    """maximum likelihood of the joint model on the frame coordinate u of |0>-prepared (u0) and |1>-prepared (u1)
    shots: shared cores (m0, s0), (m1, s1); per preparation a wrong-core fraction e and a bridge fraction B"""
    from scipy.optimize import minimize
    u0, u1 = np.asarray(u0, float), np.asarray(u1, float)

    def unpack(x):
        sig = lambda t: 1 / (1 + np.exp(-t))            # noqa: E731
        prm = dict(m0=x[0], m1=x[1], s0=math.exp(x[2]), s1=math.exp(x[3]), e0=0.5 * sig(x[4]), B0=0.5 * sig(x[5]),
                   e1=0.5 * sig(x[6]), B1=0.0 if fix_B1_zero else 0.5 * sig(x[7]))
        return prm

    def nll(x):
        prm = unpack(x)
        l0 = mixture_pdf(u0, prm, 0)
        l1 = mixture_pdf(u1, prm, 1)
        return -float(np.log(np.maximum(l0, 1e-300)).sum() + np.log(np.maximum(l1, 1e-300)).sum())
    best = None
    for b in (-4.0, -7.0):
        x0 = np.array([0.0, D, 0.0, 0.0, -5.0, -6.0, -5.0, b])
        r = minimize(nll, x0, method="L-BFGS-B", options=dict(maxiter=300, ftol=1e-12, gtol=1e-6))
        r = minimize(nll, r.x, method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-6, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    prm = unpack(best.x)
    prm["loglik"] = -float(best.fun)
    prm["n0"], prm["n1"] = int(len(u0)), int(len(u1))
    return prm


def likelihood_information(prm: dict, p: float, lo: float = -8.0, hi: float | None = None, n: int = 6000) -> float:
    """Fisher information per shot on p of the 1-D likelihood readout under the fitted model: integral of
    (f1 - f0)^2 / ((1 - p) f0 + p f1) du. At p = 1/2 the LDA posterior's linear estimator attains it for two
    equal-width Gaussians (its score 2 P1 - 1 IS the efficient score there); away from 1/2 only the likelihood does."""
    hi = prm["m1"] + 8.0 if hi is None else hi
    u = np.linspace(lo, hi, n)
    f0, f1 = mixture_pdf(u, prm, 0), mixture_pdf(u, prm, 1)
    fp = np.maximum((1 - p) * f0 + p * f1, 1e-300)
    return float(np.trapezoid((f1 - f0) ** 2 / fp, u))


def likelihood_p(u, prm: dict, iters: int = 30) -> tuple:
    """the likelihood readout's estimate of P(prepared-1 template) from shots u, by Newton on the score
    sum (f1 - f0) / ((1 - p) f0 + p f1), allowed slightly below 0 so that it is not biased at a boundary; its sd from
    the observed information"""
    from scipy.optimize import brentq
    u = np.asarray(u, float)
    f0, f1 = mixture_pdf(u, prm, 0), mixture_pdf(u, prm, 1)
    g = f1 - f0
    # the density f0 + p g must stay positive on every shot: that bounds p (just below 0 when the |0> template has
    # mass in the |1> core, e0 > 0); the score is strictly decreasing in p, so its root is bracketed
    pos, neg = g > 0, g < 0
    lo = max(-0.05, float(np.max(-f0[pos] / g[pos])) if pos.any() else -0.05)
    hi = min(1.05, float(np.min(f0[neg] / -g[neg])) if neg.any() else 1.05)
    span = hi - lo
    lo, hi = lo + 1e-9 * span, hi - 1e-9 * span

    def score(p):
        return float((g / (f0 + p * g)).sum())
    if score(lo) <= 0:
        p = lo
    elif score(hi) >= 0:
        p = hi
    else:
        p = brentq(score, lo, hi, xtol=1e-12)
    fp = f0 + p * g
    return float(p), float(1 / math.sqrt(((g / fp) ** 2).sum()))


# ================================================================== 3. mixtures, |2>, leakage
def _gauss_logpdf(X, mean, cov):
    c = np.asarray(cov, float)
    det = c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]
    inv = np.array([[c[1, 1], -c[0, 1]], [-c[1, 0], c[0, 0]]]) / det
    r = X - np.asarray(mean, float)
    q = np.einsum("ni,ij,nj->n", r, inv, r)
    return -0.5 * q - math.log(2 * math.pi) - 0.5 * math.log(det)


def _bridge_logpdf2(X, A, B, sigma):
    from scipy.special import ndtr
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.linalg.norm(B - A))
    t = (B - A) / L
    nrm = np.array([-t[1], t[0]])
    s = (X - A) @ t
    r = (X - A) @ nrm
    along = np.maximum(ndtr(s / sigma) - ndtr((s - L) / sigma), 1e-300) / L
    return np.log(along) - 0.5 * (r / sigma) ** 2 - 0.5 * math.log(2 * math.pi) - math.log(sigma)


def mixture_em(X, comps: list, weights=None, iters: int = 400, tol: float = 1e-7, reg: float = 1e-3,
               free_weights=None) -> dict:
    """EM on 2-D points X over components [dict(kind='gauss', mean, cov, free) | dict(kind='bridge', a, b, sigma)]:
    a free Gaussian's mean and covariance are re-estimated, a fixed one's are kept, a bridge's endpoints follow the
    Gaussians it names (a, b: component indices) when given as ints. Weights are free unless free_weights is False
    for a component."""
    X = np.asarray(X, float)
    comps = [dict(c) for c in comps]
    k = len(comps)
    w = np.full(k, 1 / k) if weights is None else np.asarray(weights, float).copy()
    fw = np.ones(k, bool) if free_weights is None else np.asarray(free_weights, bool)
    prev = -np.inf
    ll = -np.inf
    for it in range(iters):
        logp = np.empty((len(X), k))
        for j, c in enumerate(comps):
            if c["kind"] == "gauss":
                logp[:, j] = _gauss_logpdf(X, c["mean"], c["cov"])
            else:
                A = comps[c["a"]]["mean"] if isinstance(c["a"], int) else c["a"]
                B = comps[c["b"]]["mean"] if isinstance(c["b"], int) else c["b"]
                logp[:, j] = _bridge_logpdf2(X, A, B, c["sigma"])
        logp += np.log(np.maximum(w, 1e-300))
        mx = logp.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(axis=1))
        ll = float(lse.sum())
        resp = np.exp(logp - lse[:, None])
        nk = resp.sum(axis=0)
        if fw.all():
            w = nk / nk.sum()
        else:
            fixed_mass = w[~fw].sum()
            w[fw] = (1 - fixed_mass) * nk[fw] / max(nk[fw].sum(), 1e-300)
        for j, c in enumerate(comps):
            if c["kind"] == "gauss" and c.get("free", True) and nk[j] > 3:
                mu = (resp[:, j:j + 1] * X).sum(axis=0) / nk[j]
                r = X - mu
                cov = (resp[:, j:j + 1] * r).T @ r / nk[j] + reg * np.eye(2)
                c["mean"], c["cov"] = mu, cov
        if ll - prev < tol * max(1.0, abs(ll)):
            break
        prev = ll
    n_free = int(fw.sum()) - 1 + sum(6 - 1 for c in comps if c["kind"] == "gauss" and c.get("free", True))
    return dict(comps=comps, weights=w, loglik=ll, iters=it + 1, n_params=n_free,
                bic=-2 * ll + n_free * math.log(len(X)), resp_last=None)


def gmm_select(X, ks=(2, 3, 4), rng=None, n_init: int = 4, D: float | None = None) -> dict:
    """free full-covariance Gaussian mixtures of k components, the best of several starts each (starts include the two
    calibrated cluster centres when D is given): loglik, BIC, and every component's weight, mean and covariance"""
    rng = np.random.default_rng(169) if rng is None else rng
    X = np.asarray(X, float)
    out = {}
    for k in ks:
        best = None
        for s in range(n_init):
            if D is not None and s == 0:
                means = [np.array([0.0, 0.0]), np.array([D, 0.0])]
                far = X[np.argsort(np.minimum(np.hypot(X[:, 0], X[:, 1]), np.hypot(X[:, 0] - D, X[:, 1])))[::-1]]
                for m in range(k - 2):
                    means.append(far[m * max(1, len(far) // 200)])
            else:
                means = [X[i] for i in rng.choice(len(X), k, replace=False)]
            comps = [dict(kind="gauss", mean=m, cov=np.eye(2), free=True) for m in means[:k]]
            r = mixture_em(X, comps)
            if best is None or r["loglik"] > best["loglik"]:
                best = r
        out[k] = dict(loglik=best["loglik"], bic=best["bic"], weights=best["weights"].tolist(),
                      means=[c["mean"].tolist() for c in best["comps"]], covs=[np.asarray(c["cov"]).tolist()
                                                                              for c in best["comps"]])
    return out


def readout_template(X0, X1, D: float) -> list:
    """the two cores, each a free Gaussian fitted with a free bridge on the |0>- and |1>-prepared shots together:
    components [core0, core1, bridge(core0 -> core1, sigma 1)]"""
    X = np.concatenate([np.asarray(X0, float), np.asarray(X1, float)])
    comps = [dict(kind="gauss", mean=np.array([0.0, 0.0]), cov=np.eye(2), free=True),
             dict(kind="gauss", mean=np.array([D, 0.0]), cov=np.eye(2), free=True),
             dict(kind="bridge", a=0, b=1, sigma=1.0)]
    r = mixture_em(X, comps, weights=[0.49, 0.49, 0.02])
    comps = r["comps"]
    return [dict(kind="gauss", mean=np.asarray(comps[0]["mean"]), cov=np.asarray(comps[0]["cov"]), free=False),
            dict(kind="gauss", mean=np.asarray(comps[1]["mean"]), cov=np.asarray(comps[1]["cov"]), free=False),
            dict(kind="bridge", a=np.asarray(comps[0]["mean"]), b=np.asarray(comps[1]["mean"]), sigma=1.0)]


def extra_component_fit(X, template: list, init_mean, init_cov=None, iters: int = 400) -> dict:
    """the readout template (shapes fixed, weights free) plus one FREE Gaussian: its weight, mean and covariance,
    and the likelihood ratio against the template alone"""
    X = np.asarray(X, float)
    base = mixture_em(X, template, iters=iters)
    comps = list(template) + [dict(kind="gauss", mean=np.asarray(init_mean, float),
                                   cov=np.eye(2) * 0.5 if init_cov is None else np.asarray(init_cov, float),
                                   free=True)]
    w0 = np.append(base["weights"] * (1 - 1e-3), 1e-3)
    full = mixture_em(X, comps, weights=w0, iters=iters)
    lr = 2 * (full["loglik"] - base["loglik"])
    return dict(weight=float(full["weights"][-1]), mean=np.asarray(full["comps"][-1]["mean"]).tolist(),
                cov=np.asarray(full["comps"][-1]["cov"]).tolist(), template_weights=base["weights"].tolist(),
                lr=float(lr), n=int(len(X)))


def fixed_weights(X, comps: list, iters: int = 300) -> np.ndarray:
    """the weights of fixed components (shapes held) on X"""
    fixed = [dict(c, free=False) if c["kind"] == "gauss" else dict(c) for c in comps]
    return mixture_em(X, fixed, iters=iters)["weights"]


def depth_slope(depths, fractions, ns) -> dict:
    """weighted least-squares slope of a fraction against depth, binomial weights floored at one shot"""
    d = np.asarray(depths, float)
    f = np.asarray(fractions, float)
    n = np.asarray(ns, float)
    var = np.maximum(f * (1 - f), 1 / n) / n
    w = 1 / var
    dm = (w * d).sum() / w.sum()
    sxx = (w * (d - dm) ** 2).sum()
    b = float((w * (d - dm) * (f - (w * f).sum() / w.sum())).sum() / sxx)
    return dict(per_cz=b, sd=float(1 / math.sqrt(sxx)))


# ================================================================== 4. IQ-level crosstalk
def conditional_shift(x, b_bits, blocks) -> dict:
    """the mean of x (one coordinate of qubit A's shots, already restricted to one class core) when B reads 1 minus
    when B reads 0, combined over blocks with the efficient weights n1 n0 / n, and its z against the EXACT permutation
    null that shuffles B within each block: T = sum over B=1 shots of (x - block mean), Var0 = sum n1 n0 / n S^2"""
    x = np.asarray(x, float)
    b = np.asarray(b_bits, bool)
    _, blk = np.unique(np.asarray(blocks), return_inverse=True)
    nb = blk.max() + 1 if len(blk) else 0
    n = np.bincount(blk, minlength=nb).astype(float)
    n1 = np.bincount(blk, weights=b.astype(float), minlength=nb)
    sx = np.bincount(blk, weights=x, minlength=nb)
    sxx = np.bincount(blk, weights=x * x, minlength=nb)
    use = (n1 > 0) & (n1 < n) & (n > 1)
    mean_b = np.where(n > 0, sx / np.maximum(n, 1), 0.0)
    T = float(np.bincount(blk, weights=b * (x - mean_b[blk]), minlength=nb)[use].sum())
    S2 = np.where(n > 1, (sxx - n * mean_b ** 2) / np.maximum(n - 1, 1), 0.0)
    var0 = float((n1 * (n - n1) / np.maximum(n, 1) * S2)[use].sum())
    wsum = float((n1 * (n - n1) / np.maximum(n, 1))[use].sum())
    if wsum <= 0 or var0 <= 0:
        return dict(delta=float("nan"), sd=float("nan"), z=float("nan"), blocks=int(use.sum()), n1=int(n1[use].sum()))
    return dict(delta=T / wsum, sd=math.sqrt(var0) / wsum, z=T / math.sqrt(var0), blocks=int(use.sum()),
                n1=int(n1[use].sum()), n0=int((n - n1)[use].sum()))


def pattern_shift(xa, xb, R: int) -> dict:
    """the centroid of circuit B's shots minus circuit A's (one coordinate, core-trimmed, each randomization-major
    with R randomizations): the paired-by-randomization t statistic (it carries any drift between randomizations) and
    the shot-level z (it does not)"""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    ra = [c.mean() for c in np.array_split(xa, R)]
    rb = [c.mean() for c in np.array_split(xb, R)]
    dif = np.array(rb) - np.array(ra)
    delta = float(xb.mean() - xa.mean())
    sd_shot = math.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
    sd_pair = float(dif.std(ddof=1) / math.sqrt(R))
    return dict(delta=delta, sd_shot=sd_shot, z_shot=delta / sd_shot, sd_paired=sd_pair,
                t_paired=float(dif.mean() / sd_pair) if sd_pair > 0 else float("nan"))


def core_mask(uv, cls, D: float, radius: float = CORE) -> np.ndarray:
    """the shots within `radius` pooled sd of their class centre ((0, 0) for class 0, (D, 0) for class 1)"""
    uv = np.asarray(uv)
    cu = np.where(np.asarray(cls, bool), D, 0.0)
    return np.hypot(uv[..., 0] - cu, uv[..., 1]) < radius


# ================================================================== main: loading
def _json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _write(name: str, obj) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=_default) + "\n", encoding="utf-8", newline="\n")
    return p


def _default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def _check_memory(gb: float = 4.0, wait_s: float = 1800.0) -> float:
    """the shared box's rule: no heavy step below `gb` available. Waits (polling every 15 s) up to wait_s for the
    memory to come back, then refuses"""
    import time

    import psutil
    t0 = time.time()
    while True:
        avail = psutil.virtual_memory().available / 1e9
        if avail >= gb:
            return avail
        if time.time() - t0 > wait_s:
            raise SystemExit(f"REFUSED: {avail:.1f} GB available, below the {gb} GB floor, for {wait_s:.0f} s")
        print(f"waiting: {avail:.1f} GB available, below the {gb} GB floor", flush=True)
        time.sleep(15)


def load_r161() -> dict:
    from oneq.hw import freeze as F
    from oneq.hw import iq as IQ
    index = _json(R161_DIR / f"flight_{R161_JOB}.json")
    arrays = dict(np.load(R161_DIR / f"flight_{R161_JOB}.npz"))
    rec = _json(R161_DIR / "r161.json")
    crec = _json(R161_DIR / "job.json")
    circs = F.load(crec, (R161_DIR / "job.qpy").read_bytes(), check_sources=False)
    qbn = {n: IQ.measured_qubits(qc) for n, qc in circs.items()}
    iq = {n: IQ.iq_columns(index, arrays, n) for n in crec["circuit_order"]}
    discs = IQ.calibrate(iq, qbn, *crec["iq_calibration"])
    bits = {n: np.stack([IQ.classify(discs[q], iq[n][:, j]) for j, q in enumerate(qbn[n])], axis=1) for n in iq}
    pub = {p["name"]: p["index"] for p in index["pubs"]}
    shape = {n: twirl_shape(index["execution_spans"], pub[n], len(iq[n])) for n in iq}
    return dict(index=index, rec=rec, crec=crec, qbn=qbn, iq=iq, discs=discs, bits=bits, shape=shape,
                used=next(c["qubits"] for c in rec["circuits"] if c["name"] == "ro_all0"),
                result=_json(R161_DIR / f"result_{R161_JOB}.json"))


def load_r165() -> dict:
    from oneq.hw import freeze as F
    from oneq.hw import iq as IQ
    rec = _json(R165_DIR / f"{R165_RECORD}.json")
    circs = F.load(rec, (R165_DIR / f"{R165_RECORD}.qpy").read_bytes(), check_sources=False)
    qbn = {n: IQ.measured_qubits(qc) for n, qc in circs.items()}
    jobs = {}
    for strat, jid in R165_JOBS.items():
        index = _json(R165_DIR / f"flight_{jid}.json")
        arrays = dict(np.load(R165_DIR / f"flight_{jid}.npz"))
        iq = {n: IQ.iq_columns(index, arrays, n) for n in rec["circuit_order"]}
        discs = IQ.calibrate(iq, qbn, *rec["iq_calibration"])
        bits = {n: np.stack([IQ.classify(discs[q], iq[n][:, j]) for j, q in enumerate(qbn[n])], axis=1) for n in iq}
        pub = {p["name"]: p["index"] for p in index["pubs"]}
        shape = {n: twirl_shape(index["execution_spans"], pub[n], len(iq[n])) for n in iq}
        stored = _json(R165_DIR / f"iq_{jid}.json")["discriminators"]
        jobs[strat] = dict(job=jid, iq=iq, discs=discs, bits=bits, shape=shape, stored_discriminators=stored)
    return dict(rec=rec, qbn=qbn, jobs=jobs, used=rec["readout_qubits"])


def load_r152() -> dict:
    index = _json(R152_DIR / f"flight_{R152_JOB}.json")
    arrays = dict(np.load(R152_DIR / f"flight_{R152_JOB}.npz"))
    comp = _json(R152_DIR / "composite.json")
    spans = _json(R152_DIR / f"spans_{R152_JOB}.json")["spans"]
    parts = {t: _json(R152_DIR / f"{t}.json") for t in ("r146", "blk147")}
    result = _json(R152_DIR / f"result_{R152_JOB}.json")
    pub = {p["name"]: p for p in index["pubs"]}

    def bits(name):
        r = next(iter(pub[name]["registers"].values()))
        return unpack_bits(arrays[r["key"]], r["num_bits"])

    def shape(name):
        p = pub[name]
        return twirl_shape(spans, p["index"], next(iter(p["registers"].values()))["num_shots"])
    return dict(index=index, comp=comp, parts=parts, result=result, bits=bits, shape=shape, pub=pub)


# ================================================================== main: 1. overdispersion
def _edge_rows(bits_of, cols, ro_a, ro_b, depths, R):
    return ladder_points(bits_of, cols, ro_a, ro_b, depths, R)


def _pooled_table(points: list, keys=("d", "cls")) -> dict:
    groups: dict = {}
    for p in points:
        groups.setdefault(tuple(p[k] for k in keys), []).append(p)
    return {"|".join(str(x) for x in k): pool(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def _gen_points(points: list) -> list:
    """the fit's inputs: one point per (basis, observable, depth) -- the sum of its series' tau2, and each series'
    mean^2 and binomial sigma2"""
    groups: dict = {}
    for p in points:
        if p["degenerate"] or abs(p["mean"]) < 0.2:
            continue
        groups.setdefault((p["basis"], p["obs_index"], p["d"]), []).append(p)
    return [dict(basis=basis, obs_index=k, d=d, T=float(sum(x["tau2"] for x in v)), m2=[x["mean"] ** 2 for x in v],
                 s2=[x["sigma2"] for x in v], R=v[0]["R"]) for (basis, k, d), v in groups.items()]


def single_angle(table: dict, cls: str = "b", d: int = 48) -> dict:
    """the one-generator inversion of a class's pooled kappa2: theta = sqrt(arccosh(1 + kappa2) / d), with the
    kappa2 sd propagated (a cross-check of the full fit; exact for a single Z generator on that class)"""
    t = table.get(f"{d}|{cls}")
    if not t or not t.get("series") or not t["kappa2"] > 0:
        return dict(theta=None)
    k, sd = t["kappa2"], t["kappa2_sd"]
    th = math.sqrt(math.acosh(1 + k) / d)
    dth = sd / (math.sqrt((1 + k) ** 2 - 1) * 2 * d * th)
    return dict(theta=th, sd=dth, kappa2=k, kappa2_sd=sd)


def _depth_growth(table: dict, cls: str, depths) -> dict:
    lo, hi = [d for d in depths if d > 0][:2]
    a, b = table.get(f"{lo}|{cls}"), table.get(f"{hi}|{cls}")
    if not a or not b or not a.get("series") or not b.get("series"):
        return {}
    return depth_exponent(a["kappa2"], a["kappa2_sd"], b["kappa2"], b["kappa2_sd"], lo, hi)


def overdispersion_r161(D: dict, n_mc: int, n_boot: int = 300) -> dict:
    rec, bits, shape, used = D["rec"], D["bits"], D["shape"], D["used"]
    ro = {int(q): v for q, v in D["result"]["r161"]["readout"].items()}
    edges = [tuple(e) for e in rec["edges"]]
    tag = {"0": "s0", "1": "s1", "+": "sp", "+e": "se"}
    out: dict = dict(edges={}, controls={}, spectators={})
    recorded = D["result"]["r161"]["channels"]
    all_points: dict = {}
    for i, (a, b) in enumerate(edges):
        key = f"{a}_{b}"
        per_arm = {}
        pts_all = []
        for arm in rec["arms"]:
            bo = {(bs, d): bits[f"r161__{tag[arm]}_{bs}_d{d}"] for bs in CC.BASES for d in rec["depths"]}
            R = shape[f"r161__{tag[arm]}_ZZ_d0"][0]
            pts = _edge_rows(bo, (2 * i, 2 * i + 1), ro[a], ro[b], rec["depths"], R)
            for p in pts:
                p["arm"] = arm
            pts_all += pts
            fit = fit_z_generators(_gen_points(pts), n_mc=n_mc)
            tbl = _pooled_table(pts)
            chi = dict(binomial=basis_chi2(pts), recorded=recorded[arm][key]["chi2"])
            chi["per_point"] = basis_chi2(pts, lambda p: max(p["ratio"], 1.0) if np.isfinite(p["ratio"]) else 1.0)
            pooled_phi = {k: max(v.get("ratio", 1.0), 1.0) for k, v in tbl.items() if v.get("series")}
            chi["pooled_class"] = basis_chi2(pts, lambda p: pooled_phi.get(f"{p['d']}|{p['cls']}", 1.0))
            if "theta" in fit:
                chi["coherent_model"] = basis_chi2(pts, phi_model(fit["theta"], n_mc))
            boot = randomization_bootstrap(bo, (2 * i, 2 * i + 1), ro[a], ro[b], rec["depths"], R, n_boot)
            boot.update(p_recorded=recorded[arm][key]["p"], p_sd_recorded=recorded[arm][key]["p_sd"])
            per_arm[arm] = dict(bootstrap=boot,
                                table=tbl, growth={c: _depth_growth(tbl, c, rec["depths"]) for c in ("a", "b", "ab",
                                                                                                   "z")},
                                generators=fit, chi2=chi, single_angle={c: single_angle(tbl, c) for c in ("a", "b")},
                                fitted_channel_IZ=recorded[arm][key]["channel"]["IZ"],
                                fitted_channel_ZI=recorded[arm][key]["channel"]["ZI"],
                                fitted_p=recorded[arm][key]["p"])
        all_points[key] = pts_all
        con = {}
        for arm, base in (("1", "0"), ("+", "0"), ("+e", "0"), ("+e", "+")):
            x, y = per_arm[arm]["bootstrap"], per_arm[base]["bootstrap"]
            diff = x["p"] - y["p"]
            con[f"{arm}_vs_{base}"] = dict(diff=diff, z_recorded_sds=diff / math.hypot(x["p_sd_recorded"],
                                                                                      y["p_sd_recorded"]),
                                           z_randomization_sds=diff / math.hypot(x["p_sd"], y["p_sd"]))
        fit_all = fit_z_generators(_gen_points(pts_all), n_mc=n_mc)
        tbl_all = _pooled_table(pts_all)
        ram = D["result"]["r161"]["ramsey"]
        out.setdefault("ramsey_vs_spread", []).extend(
            predicted_vs_measured(q, cls, 2 * math.pi * ram[str(q)]["detuning_hz"], tbl_all)
            for q, cls in ((a, "a"), (b, "b")))
        out["edges"][key] = dict(arms=per_arm, contrasts=con, all_arms=dict(table=tbl_all, generators=fit_all,
                                                             single_angle={c: single_angle(tbl_all, c)
                                                                           for c in ("a", "b")},
                                                             growth={c: _depth_growth(tbl_all, c, rec["depths"])
                                                                     for c in ("a", "b", "ab", "z")}))
    # controls: circuits with no two-qubit gate (nothing to twirl): every randomization is the same circuit
    fam = {"ro": [], "xs": [], "zz": [], "r158": []}
    per_qubit_ctrl: dict = {}
    for name, b in bits.items():
        inner = name.split("__", 1)[1]
        f = "r158" if name.startswith("r158__") else inner.split("_")[0]
        if f not in fam:
            continue
        R = shape[name][0]
        for j, q in enumerate(D["qbn"][name]):
            h = homogeneity(1.0 - 2.0 * b[:, j], R)
            fam[f].append(h)
            per_qubit_ctrl.setdefault(f"{f}|{q}", []).append(h)
    out["controls"] = {f: pool(v) for f, v in fam.items()}
    out["controls_per_qubit"] = {k: pool(v) for k, v in sorted(per_qubit_ctrl.items())}
    zero = [p for pts in all_points.values() for p in pts if p["d"] == 0]
    out["controls"]["pb_d0"] = pool(zero)
    # spectators inside the ladders (twirled under active-accum: they are touched)
    sq = used[6:]
    sp: dict = {}
    for arm in rec["arms"]:
        for d in rec["depths"]:
            rows = []
            for bs in CC.BASES:
                name = f"r161__{tag[arm]}_{bs}_d{d}"
                R = shape[name][0]
                for j in range(6, len(used)):
                    rows.append(homogeneity(1.0 - 2.0 * bits[name][:, j], R))
            sp[f"{arm}|{d}"] = pool(rows)
    out["spectators"] = dict(qubits=sq, table=sp)
    return out, all_points


def ramsey_prediction(D: dict, theta: float, q: int = 149) -> dict:
    """the coherent IZ angle per layer q149's same-job Ramsey offset predicts over R166's measured 132 ns layer, against
    the angle the randomization spread measured (DERIVED vs MEASURED)"""
    ram = D["result"]["r161"]["ramsey"][str(q)]
    det = ram["detuning_hz"]
    pred = 2 * math.pi * abs(det) * LAYER_S
    return dict(qubit=q, ramsey_detuning_hz=det, ramsey_contrast=ram["contrast_w"], layer_s=LAYER_S,
                predicted_theta=pred, measured_theta=theta, ratio_measured_over_predicted=theta / pred if pred else None,
                kappa2_ratio_at_d48=(math.cosh(48 * theta ** 2) - 1) / (math.cosh(48 * pred ** 2) - 1) if pred else None,
                residual_theta_same_sign=theta - pred, residual_theta_opposite_sign=theta + pred,
                pauli_IZ_from_measured_theta=math.sin(theta / 2) ** 2)


def predicted_vs_measured(qubit: int, cls: str, offset_rad_s: float, table: dict, d: int = 48,
                          layer_s: float = LAYER_S) -> dict:
    """the coherent angle per layer a same-job Ramsey offset predicts (2 pi delta x the 132 ns layer, all of it inside
    a random frame) and the kappa2 it implies on the class where this qubit carries X or Y, against the measured
    class kappa2 and its one-generator angle. NOTHING is fitted."""
    th = abs(offset_rad_s) * layer_s
    k2p = math.cosh(d * th * th) - 1
    t = table.get(f"{d}|{cls}", {})
    sa = single_angle(table, cls, d)
    z = (t["kappa2"] - k2p) / t["kappa2_sd"] if t.get("series") and t["kappa2_sd"] > 0 else None
    return dict(qubit=qubit, cls=cls, offset_hz=offset_rad_s / (2 * math.pi), theta_predicted=th,
                kappa2_predicted=k2p, kappa2_measured=t.get("kappa2"), kappa2_sd=t.get("kappa2_sd"),
                z_measured_minus_predicted=z, theta_measured=sa.get("theta"), theta_measured_sd=sa.get("sd"))


def overdispersion_r152(P: dict, n_mc: int, n_boot: int = 300) -> dict:
    out: dict = {}
    pts_by_edge: dict = {}
    for tag, res_key in (("r146", "r146"), ("blk147", "block")):
        rec = P["parts"][tag]
        fit_rec = P["result"][res_key]["fit"]
        ro = {int(q): v for q, v in fit_rec["readout"].items()}
        rows = {}
        for r, edges in enumerate(rec["ladder_rounds"]):
            for i, (a, b) in enumerate(edges):
                bo = {(bs, d): P["bits"](f"{tag}__pb{r}_{bs}_d{d}") for bs in CC.BASES for d in D146.LADDER_DEPTHS}
                R = P["shape"](f"{tag}__pb{r}_ZZ_d0")[0]
                pts = ladder_points(bo, (2 * i, 2 * i + 1), ro[a], ro[b], D146.LADDER_DEPTHS, R)
                boot = randomization_bootstrap(bo, (2 * i, 2 * i + 1), ro[a], ro[b], D146.LADDER_DEPTHS, R, n_boot)
                boot.update(p_recorded=fit_rec["ladders"][f"{a}_{b}"]["p"],
                            p_sd_recorded=fit_rec["ladders"][f"{a}_{b}"]["p_sd"])
                fit = fit_z_generators(_gen_points(pts), n_mc=n_mc)
                tbl = _pooled_table(pts)
                chi = dict(binomial=basis_chi2(pts), recorded=fit_rec["ladders"][f"{a}_{b}"]["chi2"])
                pooled_phi = {k: max(v.get("ratio", 1.0), 1.0) for k, v in tbl.items() if v.get("series")}
                chi["pooled_class"] = basis_chi2(pts, lambda p: pooled_phi.get(f"{p['d']}|{p['cls']}", 1.0))
                if "theta" in fit:
                    chi["coherent_model"] = basis_chi2(pts, phi_model(fit["theta"], n_mc))
                rows[f"{a}_{b}"] = dict(round=r, table=tbl, generators=fit, chi2=chi, bootstrap=boot,
                                        single_angle={c: single_angle(tbl, c) for c in ("a", "b")},
                                        growth={c: _depth_growth(tbl, c, D146.LADDER_DEPTHS)
                                                for c in ("a", "b", "ab", "z")},
                                        fitted_p=fit_rec["ladders"][f"{a}_{b}"]["p"])
                pts_by_edge[f"{tag}:{a}_{b}"] = pts
                # the part's own Ramsey (X quadrature only): |offset| <= arccos(ratio / 0.98) / W, an UPPER bound on a
                # coherent offset (any decay is read as rotation), the smallest alias
                W = rec["wait_dt"] * rec["snapshot"]["dt"]
                rows[f"{a}_{b}"]["ramsey_vs_spread"] = []
                for q, cls in ((a, "a"), (b, "b")):
                    xs = fit_rec["xs"].get(str(q))
                    if xs is None:
                        continue
                    off = math.acos(max(-1.0, min(1.0, xs["ratio"] / 0.98))) / W
                    row = predicted_vs_measured(q, cls, off, tbl)
                    row.update(ramsey_ratio=xs["ratio"], wait_s=W, offset_is="upper bound (X quadrature only)")
                    rows[f"{a}_{b}"]["ramsey_vs_spread"].append(row)
        ctrl: dict = {}
        for x in rec["diag"]:
            fam = x["meta"]["family"]
            if fam not in ("ro", "xs", "sk"):
                continue
            name = f"{tag}__{x['name']}"
            b = P["bits"](name)
            R = P["shape"](name)[0]
            ctrl.setdefault(fam, []).extend(homogeneity(1.0 - 2.0 * b[:, j], R) for j in range(b.shape[1]))
        out[tag] = dict(edges=rows, controls={f: pool(v) for f, v in ctrl.items()})
    return out, pts_by_edge


def overdispersion_r165(Q: dict) -> dict:
    rec, used = Q["rec"], Q["used"]
    groups = dict(rec["design"]["groups"], edge=rec["design"]["edge"])
    out: dict = {}
    for strat, J in Q["jobs"].items():
        tbl: dict = {}
        for gname, qs in groups.items():
            for d in rec["depths"]:
                name = f"bench_d{d}"
                R = J["shape"][name][0]
                rows = [homogeneity(1.0 - 2.0 * J["bits"][name][:, used.index(q)], R) for q in qs]
                pl = pool(rows)
                if gname == "z":                         # the flip probability's relative spread, readout-corrected
                    flips = []
                    for q, h in zip(qs, rows):
                        dsc = J["discs"][q]
                        sa = 1 - dsc.err01 - dsc.err10
                        p = ((1 - h["mean"]) / 2 - dsc.err01) / sa
                        flips.append(dict(qubit=q, p_flip=p, tau_p2=h["tau2"] / 4 / sa ** 2,
                                          kappa_p2=(h["tau2"] / 4 / sa ** 2) / p ** 2 if p > 1e-4 else None))
                    pl["flips"] = flips
                    good = [f for f in flips if f["kappa_p2"] is not None]
                    if good:
                        pl["kappa_p2_pooled"] = sum(f["tau_p2"] for f in good) / sum(f["p_flip"] ** 2 for f in good)
                        pl["kappa_p2_if_flips_track_the_inverted_layer_count"] = 1 / d if d else None
                tbl[f"{gname}|{d}"] = pl
        ctrl = []
        for name in ("ro_all0", "ro_all1", "ro_end0", "ro_end1"):
            R = J["shape"][name][0]
            ctrl += [homogeneity(1.0 - 2.0 * J["bits"][name][:, j], R) for j in range(len(used))]
        out[strat] = dict(table=tbl, controls_ro=pool(ctrl))
    return out


# ================================================================== main: 2. soft readout
def _soft_qubit(disc, iq0, iq1, iq0_end, iq1_end, t1: float, R: int = 8) -> dict:
    fr = frame(disc)
    uv0, uv1 = to_frame(fr, iq0), to_frame(fr, iq1)
    uv0e, uv1e = to_frame(fr, iq0_end), to_frame(fr, iq1_end)
    D = fr["D"]
    hard = lambda uv: (uv[:, 0] > D / 2).astype(float)          # noqa: E731
    soft = lambda uv: lda_posterior(fr, uv)                     # noqa: E731
    row = dict(D=D, eps01_hard_in=float(hard(uv0).mean()), eps10_hard_in=float(1 - hard(uv1).mean()),
               eps01_hard_end=float(hard(uv0e).mean()), eps10_hard_end=float(1 - hard(uv1e).mean()),
               a0_soft_in=float(soft(uv0).mean()), one_minus_a1_soft_in=float(1 - soft(uv1).mean()),
               a0_soft_end=float(soft(uv0e).mean()), one_minus_a1_soft_end=float(1 - soft(uv1e).mean()))
    for tag_, (s0, s1) in (("hard", (hard(uv0e), hard(uv1e))), ("soft", (soft(uv0e), soft(uv1e)))):
        for p in (0.0, 0.5, 1.0):
            row[f"V_{tag_}_p{p:g}"] = linear_estimator_variance(s0, s1, p)
        row[f"eps_eff_{tag_}"] = effective_error(row[f"V_{tag_}_p0.5"])
    for p in ("0.5", "0", "1"):                      # a held-out circuit with no error at all leaves the ratio unread
        h = row[f"V_hard_p{p}"]
        row[f"soft_over_hard_variance_p{p}"] = row[f"V_soft_p{p}"] / h if h > 0 else None
    m = readout_mixture_fit(uv0[:, 0], uv1[:, 0], D)
    m0 = readout_mixture_fit(uv0[:, 0], uv1[:, 0], D, fix_B1_zero=True)
    lr = max(2 * (m["loglik"] - m0["loglik"]), 0.0)
    info = likelihood_information(m, 0.5)
    row["mixture"] = {k: v for k, v in m.items()}
    row["bridge_lr"] = lr
    row["bridge_z"] = math.sqrt(lr)
    row["bridge_window_s"] = m["B1"] * t1
    row["V_likelihood_p0.5"] = 1 / info
    row["eps_eff_likelihood"] = effective_error(1 / info)
    row["likelihood_over_hard_variance_p0.5"] = (1 / info) / row["V_hard_p0.5"]
    # the same comparison inside the fitted model (hard errors the model implies), so both sides read one dataset
    g = np.linspace(-8.0, m["m1"] + 8.0, 6000)
    e01m = float(np.trapezoid(np.where(g > D / 2, mixture_pdf(g, m, 0), 0.0), g))
    e10m = float(np.trapezoid(np.where(g <= D / 2, mixture_pdf(g, m, 1), 0.0), g))
    qh = e01m + 0.5 * (1 - e01m - e10m)
    row["model_eps01_hard"], row["model_eps10_hard"] = e01m, e10m
    row["V_hard_model_p0.5"] = qh * (1 - qh) / (1 - e01m - e10m) ** 2
    row["eps_eff_hard_model"] = effective_error(row["V_hard_model_p0.5"])
    row["likelihood_over_hard_model_variance_p0.5"] = (1 / info) / row["V_hard_model_p0.5"]
    for p in (0.02, 0.98):
        inf_p = likelihood_information(m, p)
        qp = e01m + p * (1 - e01m - e10m)
        row[f"likelihood_over_hard_model_variance_p{p:g}"] = (1 / inf_p) / (qp * (1 - qp) / (1 - e01m - e10m) ** 2)
    # the hard error split by where its shots are: a wrong core (no readout can recover it), the bridge's wrong half,
    # and the cores' Gaussian overlap
    from scipy.special import ndtr
    row["eps10_split"] = dict(wrong_core=m["e1"] * float(ndtr((D / 2 - m["m0"]) / m["s0"])),
                              bridge_wrong_half=m["B1"] * float(np.clip((D / 2 - m["m0"]) / (m["m1"] - m["m0"]), 0, 1)),
                              core_overlap=(1 - m["e1"] - m["B1"]) * float(ndtr((D / 2 - m["m1"]) / m["s1"])))
    row["eps01_split"] = dict(wrong_core=m["e0"] * float(ndtr((m["m1"] - D / 2) / m["s1"])),
                              bridge_wrong_half=m["B0"] * float(np.clip((m["m1"] - D / 2) / (m["m1"] - m["m0"]), 0, 1)),
                              core_overlap=(1 - m["e0"] - m["B0"]) * float(ndtr((m["m0"] - D / 2) / m["s0"])))
    # the |1> error per randomization block (time order within the span): a stationary readout scatters binomially
    for lab, uvx in (("ro_all1", uv1), ("ro_end1", uv1e)):
        blocks = (uvx[:, 0] <= D / 2).reshape(R, -1).mean(axis=1)
        row[f"eps10_by_randomization_{lab}"] = blocks.tolist()
    pp0, pp1 = lda_posterior(fr, uv0), lda_posterior(fr, uv1)
    row["ambiguous_fraction_prep0"] = float(np.mean((pp0 > AMBIGUOUS) & (pp0 < 1 - AMBIGUOUS)))
    row["ambiguous_fraction_prep1"] = float(np.mean((pp1 > AMBIGUOUS) & (pp1 < 1 - AMBIGUOUS)))
    mid = (uv1[:, 0] > 0.3 * D) & (uv1[:, 0] < 0.7 * D)
    core1 = np.hypot(uv1[:, 0] - D, uv1[:, 1]) < 2
    row["bridge_shots_v_sd"] = float(uv1[mid, 1].std(ddof=1)) if mid.sum() > 3 else None
    row["core1_v_sd"] = float(uv1[core1, 1].std(ddof=1))
    row["bridge_shots"] = int(mid.sum())
    return row


def soft_readout(D161: dict, Q: dict) -> dict:
    from scipy.stats import spearmanr
    snap = D161["rec"]["snapshot"]
    out: dict = dict(r161={}, r165={})
    for j, q in enumerate(D161["used"]):
        iq = D161["iq"]
        out["r161"][str(q)] = _soft_qubit(D161["discs"][q], iq["r161__ro_all0"][:, j], iq["r161__ro_all1"][:, j],
                                          iq["r161__ro_end0"][:, j], iq["r161__ro_end1"][:, j], snap["t1"][q])
    snap5 = Q["rec"]["snapshot"]
    for strat, J in Q["jobs"].items():
        rows = {}
        for j, q in enumerate(Q["used"]):
            rows[str(q)] = _soft_qubit(J["discs"][q], J["iq"]["ro_all0"][:, j], J["iq"]["ro_all1"][:, j],
                                       J["iq"]["ro_end0"][:, j], J["iq"]["ro_end1"][:, j], snap5["t1"][q])
        out["r165"][strat] = rows
    # is the |1> tail T1 decay during the read? B1 against 1/T1 across every qubit of both jobs
    pts = [(1 / snap["t1"][int(q)], r["mixture"]["B1"], r["bridge_z"]) for q, r in out["r161"].items()]
    pts += [(1 / snap5["t1"][int(q)], r["mixture"]["B1"], r["bridge_z"]) for q, r in out["r165"]["active"].items()]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    rho, pv = spearmanr(x, y)
    tau = float((x * y).sum() / (x * x).sum())
    resid = y - tau * x
    out["t1_tail_test"] = dict(qubits=len(pts), spearman_B1_vs_inverse_T1=float(rho), p=float(pv),
                               window_through_origin_s=tau, window_sd_s=float(math.sqrt(
                                   (resid ** 2).sum() / (len(x) - 1) / (x * x).sum())),
                               median_window_s=float(np.median(y / x)),
                               vendor_meas_duration_s=snap["meas_duration"][D161["used"][0]],
                               r166_measured_read_s=1.2e-6, bridge_resolved_qubits=int(sum(p[2] > 3 for p in pts)))
    out["r165_z_slope"] = z_slope_soft_vs_hard(Q)
    out["r161_assignment"] = assignment_summary(out["r161"])
    return out


def assignment_summary(rows: dict) -> dict:
    keys = ("eps_eff_hard", "eps_eff_soft", "eps_eff_likelihood", "eps_eff_hard_model", "soft_over_hard_variance_p0.5",
            "likelihood_over_hard_variance_p0.5", "likelihood_over_hard_model_variance_p0.5",
            "likelihood_over_hard_model_variance_p0.02", "likelihood_over_hard_model_variance_p0.98",
            "soft_over_hard_variance_p0", "soft_over_hard_variance_p1", "ambiguous_fraction_prep0",
            "ambiguous_fraction_prep1")
    out = {}
    for k in keys:
        v = [r[k] for r in rows.values() if r[k] is not None]
        out[k] = dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))
    return out


def z_slope_soft_vs_hard(Q: dict) -> dict:
    """R165's Z-group flip slope per layer, hard bits against LDA posteriors (each corrected by its own calibration
    from the job's start-of-job readout circuits), with shot sds and randomization sds"""
    import hw_r165_dressing_bench as R165
    rec, used = Q["rec"], Q["used"]
    out = {}
    for strat, J in Q["jobs"].items():
        per = {}
        for q in rec["design"]["groups"]["z"]:
            j = used.index(q)
            fr = frame(J["discs"][q])
            cal = {}
            for kind in ("hard", "soft"):
                s = {}
                for name in ("ro_all0", "ro_all1"):
                    uv = to_frame(fr, J["iq"][name][:, j])
                    s[name] = (uv[:, 0] > fr["D"] / 2).astype(float) if kind == "hard" else lda_posterior(fr, uv)
                cal[kind] = (s["ro_all0"].mean(), s["ro_all1"].mean())
            rows = {}
            for kind in ("hard", "soft"):
                a0, a1 = cal[kind]
                pts, pts_r = [], []
                for d in rec["depths"]:
                    uv = to_frame(fr, J["iq"][f"bench_d{d}"][:, j])
                    s = (uv[:, 0] > fr["D"] / 2).astype(float) if kind == "hard" else lda_posterior(fr, uv)
                    R = J["shape"][f"bench_d{d}"][0]
                    val = (s.mean() - a0) / (a1 - a0)
                    # R165's own floor (per_qubit): var = max(p (1 - p), 1 / n), so the hard arm reproduces its record
                    sd = math.sqrt(max(s.var(), 1 / len(s)) / len(s)) / (a1 - a0)
                    rm = s.reshape(R, -1).mean(axis=1)
                    sd_r = rm.std(ddof=1) / math.sqrt(R) / (a1 - a0)
                    pts.append((d, val, sd))
                    pts_r.append((d, val, max(sd_r, 1e-9)))
                b, sb = R165.slope(pts)
                br, sbr = R165.slope(pts_r)
                rows[kind] = dict(points=pts, slope=b, sd_shot=sb, sd_randomization=sbr, a0=a0, one_minus_a1=1 - a1)
            # the likelihood readout: the job's own 1-D readout model (cores, wrong-core, bridge) as templates
            u0 = to_frame(fr, J["iq"]["ro_all0"][:, j])[:, 0]
            u1 = to_frame(fr, J["iq"]["ro_all1"][:, j])[:, 0]
            prm = readout_mixture_fit(u0, u1, fr["D"])
            pts, pts_r = [], []
            for d in rec["depths"]:
                u = to_frame(fr, J["iq"][f"bench_d{d}"][:, j])[:, 0]
                R = J["shape"][f"bench_d{d}"][0]
                val, sd = likelihood_p(u, prm)
                rp = [likelihood_p(c, prm)[0] for c in u.reshape(R, -1)]
                pts.append((d, val, sd))
                pts_r.append((d, val, max(float(np.std(rp, ddof=1) / math.sqrt(R)), 1e-9)))
            b, sb = R165.slope(pts)
            br, sbr = R165.slope(pts_r)
            rows["likelihood"] = dict(points=pts, slope=b, sd_shot=sb, sd_randomization=sbr,
                                      model={k: prm[k] for k in ("e0", "B0", "e1", "B1", "s0", "s1", "m0", "m1")})
            per[str(q)] = rows
        for kind in ("hard", "soft", "likelihood"):
            sl = [(per[str(q)][kind]["slope"], per[str(q)][kind]["sd_shot"]) for q in rec["design"]["groups"]["z"]]
            w = np.array([1 / s ** 2 for _, s in sl])
            per[f"pooled_{kind}"] = dict(slope=float((w * np.array([b for b, _ in sl])).sum() / w.sum()),
                                         sd=float(1 / math.sqrt(w.sum())))
        zq = rec["design"]["groups"]["z"]
        for kind in ("soft", "likelihood"):
            per[f"sd_ratio_{kind}_over_hard_median"] = float(np.median(
                [per[str(q)][kind]["sd_shot"] / per[str(q)]["hard"]["sd_shot"] for q in zq]))
            per[f"sd_randomization_ratio_{kind}_over_hard_median"] = float(np.median(
                [per[str(q)][kind]["sd_randomization"] / per[str(q)]["hard"]["sd_randomization"] for q in zq]))
            per[f"slope_{kind}_minus_hard"] = per[f"pooled_{kind}"]["slope"] - per["pooled_hard"]["slope"]
        out[strat] = per
    out["recorded_result"] = _json(R165_DIR / f"result_{R165_RECORD}.json")["groups"]["z"]
    return out


# ================================================================== main: 3. |2> and leakage
def leakage_gmm(D161: dict, Q: dict, quick: bool) -> dict:
    rec, iq, used = D161["rec"], D161["iq"], D161["used"]
    tag = {"0": "s0", "1": "s1", "+": "sp", "+e": "se"}
    out: dict = dict(r161={}, r165={})
    edge_q = [q for e in rec["edges"] for q in e]
    for j, q in enumerate(used):
        fr = frame(D161["discs"][q])
        Dq = fr["D"]
        X0 = to_frame(fr, iq["r161__ro_all0"][:, j])
        X1 = to_frame(fr, iq["r161__ro_all1"][:, j])
        row = dict(D=Dq, ro_all1=_gmm_summary(gmm_select(X1, (2, 3, 4), D=Dq), Dq))
        if q in edge_q:
            template = readout_template(X0, X1, Dq)
            by_d = {}
            for d in rec["depths"]:
                X = np.concatenate([to_frame(fr, iq[f"r161__{tag[a]}_{bs}_d{d}"][:, j]) for a in rec["arms"]
                                    for bs in CC.BASES])
                by_d[d] = X
            dmax = max(rec["depths"])
            g = gmm_select(by_d[dmax], (2, 3, 4), D=Dq)
            row["deep_ladders_gmm"] = _gmm_summary(g, Dq)
            # the |2> candidate: the unsupervised 3-component fit's component that is neither core (else beyond |1>)
            cand = [(w, m, c) for w, m, c in zip(g[3]["weights"], g[3]["means"], g[3]["covs"])
                    if _component_kind(m, c, Dq) in ("off_axis", "beyond_1")]
            init_m, init_c = (cand[0][1], cand[0][2]) if cand else ([Dq + 2.5, 0.0], None)
            # the leakage detector: one free Gaussian over the readout template (cores + bridge, shapes fixed)
            ex = extra_component_fit(by_d[dmax], template, init_mean=init_m, init_cov=init_c)
            comps = template + [dict(kind="gauss", mean=np.array(ex["mean"]), cov=np.array(ex["cov"]), free=False)]
            fr_d = {d: float(fixed_weights(X, comps)[-1]) for d, X in by_d.items()}
            sl = depth_slope(list(fr_d), list(fr_d.values()), [len(by_d[d]) for d in fr_d])
            per_arm = {}
            for arm in rec["arms"]:
                w = {d: float(fixed_weights(np.concatenate([to_frame(fr, iq[f"r161__{tag[arm]}_{bs}_d{d}"][:, j])
                                                            for bs in CC.BASES]), comps)[-1]) for d in rec["depths"]}
                per_arm[arm] = dict(weight_by_depth=w, slope=depth_slope(list(w), list(w.values()),
                                                                         [7000] * len(w)))
            ro_w = {n: float(fixed_weights(to_frame(fr, iq[f"r161__{n}"][:, j]), comps)[-1])
                    for n in ("ro_all0", "ro_all1", "ro_end0", "ro_end1")}
            # what the outlier cut (D2 > 18.42 from both clusters, in this whitened frame |x|^2) catches of it
            rng = np.random.default_rng(169)
            smp = rng.multivariate_normal(ex["mean"], ex["cov"], 200000)
            from oneq.hw import iq as IQ
            caught = float(np.mean(((smp ** 2).sum(1) > IQ.D2_OUT) & (((smp - [Dq, 0]) ** 2).sum(1) > IQ.D2_OUT)))
            out_frac = {d: float(np.mean(((X ** 2).sum(1) > IQ.D2_OUT) & (((X - [Dq, 0]) ** 2).sum(1) > IQ.D2_OUT)))
                        for d, X in by_d.items()}
            row["extra_component"] = dict(
                candidate_from_unsupervised_fit=[dict(weight=w, mean=m, cov=c) for w, m, c in cand], deepest=ex,
                weight_by_depth=fr_d, slope=sl, per_arm=per_arm, readout_circuits=ro_w,
                location_u_over_D=ex["mean"][0] / Dq, location_v=ex["mean"][1],
                distance_from_core1_sd=float(math.hypot(ex["mean"][0] - Dq, ex["mean"][1])),
                fraction_the_outlier_cut_catches=caught, outlier_fraction_by_depth=out_frac,
                outlier_slope=depth_slope(list(out_frac), list(out_frac.values()), [len(by_d[d]) for d in out_frac]))
        out["r161"][str(q)] = row
    # R165: prepared-|1> readout on 23 qubits, both jobs
    for strat, J in Q["jobs"].items():
        rows = {}
        for j, q in enumerate(Q["used"]):
            if quick and j > 4:
                break
            fr = frame(J["discs"][q])
            X1 = to_frame(fr, J["iq"]["ro_all1"][:, j])
            rows[str(q)] = dict(D=fr["D"], ro_all1=_gmm_summary(gmm_select(X1, (2, 3), D=fr["D"]), fr["D"]))
        out["r165"][strat] = rows
    out["recorded_outlier_leakage_r161"] = {q: {a: v["per_cz"] for a, v in x["arms"].items()}
                                            for q, x in D161["result"]["r161"]["iq"]["leakage"].items()}
    return out


def _gmm_summary(res: dict, D: float) -> dict:
    out = {}
    for k, r in res.items():
        comps = []
        for w, m, c in zip(r["weights"], r["means"], r["covs"]):
            ev = np.linalg.eigvalsh(np.array(c))
            comps.append(dict(weight=w, u=m[0], v=m[1], u_over_D=m[0] / D, sd_major=float(math.sqrt(ev[-1])),
                              sd_minor=float(math.sqrt(max(ev[0], 0))), kind=_component_kind(m, c, D)))
        out[str(k)] = dict(loglik=r["loglik"], bic=r["bic"], components=sorted(comps, key=lambda c: -c["weight"]))
    ks = sorted(int(k) for k in out)
    out["delta_bic_vs_2"] = {str(k): out[str(k)]["bic"] - out["2"]["bic"] for k in ks if k != 2}
    out["bic_choice"] = str(min(ks, key=lambda k: out[str(k)]["bic"]))
    return out


def _component_kind(m, c, D: float) -> str:
    """where a mixture component sits in the discriminator frame: a core (within 1.5 sd of (0,0) or (D,0)), a BRIDGE
    (between the cores, on the axis, elongated along it), or OFF-AXIS / BEYOND |1> (|2>-like candidates)"""
    u, v = m
    c = np.array(c)
    ev, U = np.linalg.eigh(c)
    major = U[:, -1]
    along = abs(major[0]) > 0.8
    if math.hypot(u, v) < 1.5:
        return "core0"
    if math.hypot(u - D, v) < 1.5:
        return "core1"
    if 0 < u < D and abs(v) < 1.5 and along:
        return "bridge"
    if u >= D and abs(v) < 1.5:
        return "beyond_1"
    return "off_axis"


# ================================================================== main: 4. crosstalk
def crosstalk_r161(D161: dict) -> dict:
    used, bits, iq = D161["used"], D161["bits"], D161["iq"]
    snap = D161["rec"]["snapshot"]
    nbr = {tuple(sorted(int(x) for x in k.split("_"))) for k in snap["cz"]}
    edges = {tuple(sorted(e)) for e in D161["rec"]["edges"]}
    frames = {q: frame(D161["discs"][q]) for q in used}
    names = [n for n in iq if n.startswith("r161__")]
    uv = {n: np.stack([to_frame(frames[q], iq[n][:, j]) for j, q in enumerate(used)], axis=1) for n in names}
    out: dict = dict(pairs={})
    zs_n, zs_far = [], []
    for ia, A in enumerate(used):
        for ib, B in enumerate(used):
            if A == B:
                continue
            pair = tuple(sorted((A, B)))
            sel = [n for n in names if not (pair in edges and "_d" in n and n.split("__")[1][:2] in
                                            ("s0", "s1", "sp", "se"))]
            row = {}
            for cls in (0, 1):
                xs_u, xs_v, bb, blk = [], [], [], []
                for k, n in enumerate(sel):
                    a_cls = bits[n][:, ia]
                    m = (a_cls == bool(cls)) & core_mask(uv[n][:, ia], a_cls, frames[A]["D"])
                    R = D161["shape"][n][0]
                    rid = np.repeat(np.arange(R), len(a_cls) // R)
                    xs_u.append(uv[n][m, ia, 0])
                    xs_v.append(uv[n][m, ia, 1])
                    bb.append(bits[n][m, ib])
                    blk.append(k * 100 + rid[m])
                cu = conditional_shift(np.concatenate(xs_u), np.concatenate(bb), np.concatenate(blk))
                cv = conditional_shift(np.concatenate(xs_v), np.concatenate(bb), np.concatenate(blk))
                row[f"A{cls}"] = dict(du=cu["delta"], du_sd=cu["sd"], zu=cu["z"], dv=cv["delta"], dv_sd=cv["sd"],
                                      zv=cv["z"], blocks=cu["blocks"], n1=cu["n1"], n0=cu.get("n0"))
            row["neighbours"] = pair in nbr
            out["pairs"][f"{A}<-{B}"] = row
            for cls in (0, 1):
                for zk in ("zu", "zv"):
                    z = row[f"A{cls}"][zk]
                    if np.isfinite(z):
                        (zs_n if pair in nbr else zs_far).append(z)
    out["summary"] = _z_summary(zs_n, zs_far)
    out["patterns"] = pattern_crosstalk(D161, frames)
    return out


def _z_summary(zs_n, zs_far) -> dict:
    from scipy.stats import norm
    out = {}
    for lab, z in (("neighbours", zs_n), ("non_neighbours", zs_far)):
        z = np.array(z)
        if not len(z):
            continue
        out[lab] = dict(tests=len(z), z_sd=float(z.std(ddof=1)), max_abs_z=float(np.abs(z).max()),
                        over_3=int((np.abs(z) > 3).sum()), expected_over_3=float(len(z) * 2 * norm.sf(3)),
                        over_4=int((np.abs(z) > 4).sum()))
    return out


def pattern_crosstalk(D161: dict, frames: dict) -> dict:
    """each qubit's cluster centroid in the all-same readout circuit against the alternating one that prepares it the
    same way (half the other qubits flipped), and the job's start against its end (the drift scale)"""
    used, iq, rec = D161["used"], D161["iq"], D161["rec"]
    pat = {c["name"]: c["pattern"] for c in rec["circuits"] if c.get("family") == "ro"}
    out = {}
    for j, q in enumerate(used):
        fr = frames[q]
        row = {}
        for s in (0, 1):
            base = f"ro_all{s}"
            alt = next(n for n in ("ro_alt01", "ro_alt10") if pat[n][j] == s)
            for other, lab in ((alt, "alternating"), (f"ro_end{s}", "end_of_job")):
                xa = to_frame(fr, iq[f"r161__{base}"][:, j])
                xb = to_frame(fr, iq[f"r161__{other}"][:, j])
                ma = core_mask(xa, np.full(len(xa), bool(s)), fr["D"])
                mb = core_mask(xb, np.full(len(xb), bool(s)), fr["D"])
                R = D161["shape"][f"r161__{base}"][0]
                # the trim breaks the randomization-major split only by the trimmed shots: keep shots in place
                ua = np.where(ma, xa[:, 0], np.nan).reshape(R, -1)
                ub = np.where(mb, xb[:, 0], np.nan).reshape(R, -1)
                va = np.where(ma, xa[:, 1], np.nan).reshape(R, -1)
                vb = np.where(mb, xb[:, 1], np.nan).reshape(R, -1)
                row[f"prep{s}_{lab}"] = dict(u=_paired(ua, ub), v=_paired(va, vb),
                                             flipped_qubits=[used[k] for k in range(len(used)) if
                                                             pat[other if lab == "alternating" else base][k] != s]
                                             if lab == "alternating" else [])
        out[str(q)] = row
    return out


def _paired(a2, b2) -> dict:
    ra, rb = np.nanmean(a2, axis=1), np.nanmean(b2, axis=1)
    dif = rb - ra
    R = len(dif)
    a, b = a2[np.isfinite(a2)], b2[np.isfinite(b2)]
    delta = float(b.mean() - a.mean())
    sd_shot = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    sd_pair = float(dif.std(ddof=1) / math.sqrt(R))
    return dict(delta=delta, sd_shot=sd_shot, z_shot=delta / sd_shot, t_paired=float(dif.mean() / sd_pair))


def crosstalk_r165(Q: dict) -> dict:
    used = Q["used"]
    snap = Q["rec"]["snapshot"]
    nbr = {tuple(sorted(int(x) for x in k.split("_"))) for k in snap["cz"]}
    out = {}
    for strat, J in Q["jobs"].items():
        frames = {q: frame(J["discs"][q]) for q in used}
        names = list(J["iq"])
        uv = {n: np.stack([to_frame(frames[q], J["iq"][n][:, j]) for j, q in enumerate(used)], axis=1)
              for n in names}
        zs_n, zs_far, worst = [], [], []
        for ia, A in enumerate(used):
            for ib, B in enumerate(used):
                if A == B:
                    continue
                pair = tuple(sorted((A, B)))
                for cls in (0, 1):
                    xs, bb, blk = [], [], []
                    for k, n in enumerate(names):
                        a_cls = J["bits"][n][:, ia]
                        m = (a_cls == bool(cls)) & core_mask(uv[n][:, ia], a_cls, frames[A]["D"])
                        R = J["shape"][n][0]
                        rid = np.repeat(np.arange(R), len(a_cls) // R)
                        xs.append(uv[n][m, ia, 0])
                        bb.append(J["bits"][n][m, ib])
                        blk.append(k * 100 + rid[m])
                    c = conditional_shift(np.concatenate(xs), np.concatenate(bb), np.concatenate(blk))
                    if np.isfinite(c["z"]):
                        (zs_n if pair in nbr else zs_far).append(c["z"])
                        worst.append((abs(c["z"]), f"{A}<-{B}", cls, c["delta"], c["sd"], pair in nbr))
        worst.sort(reverse=True)
        out[strat] = dict(summary=_z_summary(zs_n, zs_far),
                          largest=[dict(pair=p, A_class=c, du=d, du_sd=s, z=z, neighbours=nb)
                                   for z, p, c, d, s, nb in worst[:8]])
    return out


# ================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer Monte-Carlo draws and fewer mixture fits")
    ap.add_argument("--only", choices=["1", "2", "3", "4"], help="run one task")
    a = ap.parse_args()
    n_mc = 1500 if a.quick else 4000
    n_boot = 60 if a.quick else 300
    print(f"memory available {_check_memory():.1f} GB")
    D161 = load_r161()
    Q = load_r165()
    written = []
    if a.only in (None, "1"):
        r161, _ = overdispersion_r161(D161, n_mc, n_boot)
        th = r161["edges"]["148_149"]["all_arms"]["generators"]["theta"]["IZ"]
        r161["q149_ramsey_vs_spread"] = ramsey_prediction(D161, th)
        r161["q149_per_arm"] = {arm: ramsey_prediction(D161, v["generators"]["theta"]["IZ"])
                                for arm, v in r161["edges"]["148_149"]["arms"].items()}
        _write("overdispersion.json", dict(schema="oneq.r169.overdispersion/1", status="partial: R161 only",
                                           r161=r161))
        _check_memory()
        P = load_r152()
        r152, _ = overdispersion_r152(P, n_mc, n_boot)
        r165 = overdispersion_r165(Q)
        written.append(_write("overdispersion.json", dict(
            schema="oneq.r169.overdispersion/1", label="EXPLORATORY -- outside every preregistered verdict",
            evidence_class="MEASURED (spread, chi2) / DERIVED (angles through the coherent model)",
            null="permutation of shots across randomizations (exact mean, Monte-Carlo variance)", n_mc=n_mc,
            layer_s=LAYER_S, r161=r161, r152=r152, r165=r165)))
    if a.only in (None, "2"):
        _check_memory()
        written.append(_write("soft_readout.json", dict(schema="oneq.r169.soft_readout/1",
                                                        label="EXPLORATORY", **soft_readout(D161, Q))))
    if a.only in (None, "3"):
        _check_memory()
        written.append(_write("leakage_gmm.json", dict(schema="oneq.r169.leakage_gmm/1", label="EXPLORATORY",
                                                       **leakage_gmm(D161, Q, a.quick))))
    if a.only in (None, "4"):
        _check_memory()
        written.append(_write("crosstalk.json", dict(schema="oneq.r169.crosstalk/1", label="EXPLORATORY",
                                                     r161=crosstalk_r161(D161), r165=crosstalk_r165(Q))))
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
