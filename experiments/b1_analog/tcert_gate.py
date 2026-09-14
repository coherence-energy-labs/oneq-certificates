r"""B1-alpha: the certified analog horizon T_cert(eps), offline, $0.

The build program's B1-alpha gate, verbatim: publish the T_cert(eps)
curve with 0 interval violations over 200 sampled-disorder
realisations, T_cert(0.1) >= 0.15 us at |LC|=3, and a demonstration
that the naive global Duhamel bound is vacuous (>1.0) where the local
instrument is not -- i.e. the instrument buys >= 10x.

THE PHYSICS. Aquila-class Rydberg chain, Hamiltonian

    H = sum_i (Omega/2) X_i - Delta sum_i n_i
        + sum_{i<j} V_ij n_i n_j,     V_ij = C6 / r_ij^6

with C6/2pi = 862690 MHz um^6 (|70S> Rb), nominal spacing a, and
GAUSSIAN POSITION DISORDER on every atom (the sampled-disorder axis).

THE BOUND. For a local observable A supported at the chain centre and
a light cone LC of radius R, let H_LC keep only terms acting entirely
inside LC. Duhamel gives

    | <A(t)>_H - <A(t)>_{H_LC} |  <=  2 t || [H - H_LC] |_{touching LC} ||

because A_LC(s) is supported in LC, so every omitted term acting only
OUTSIDE LC commutes with it and contributes nothing. Only cross terms
(i in LC, j outside) survive:

    S_local = sum_{i in LC, j not in LC} V_ij      ->  T_cert(eps) = eps / (2 S_local)

The NAIVE global Duhamel bound keeps no locality at all:

    S_global = || H - H_LC ||  <=  sum over ALL omitted terms

which includes every far-far pair and every omitted on-site term. Both
are valid upper bounds; the ratio S_global / S_local is what the
instrument buys.

VALIDATION. Exact sparse evolution of the full chain and of the
truncated chain, same initial product state, at 30 time points, over
200 disorder realisations: the measured discrepancy must never exceed
the certified interval. A single violation fails the gate.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

ROOT = pathlib.Path(__file__).resolve().parents[2]

TWO_PI = 2.0 * np.pi
C6 = TWO_PI * 862690.0          # rad/us * um^6  (|70S> Rb)
OMEGA = TWO_PI * 2.5            # rad/us, Aquila max Rabi
DELTA = TWO_PI * 1.0            # rad/us
SPACING = 6.0                   # um
DISORDER_SIGMA = 0.1            # um, per-atom Gaussian
# GEOMETRY. B1-beta names the target system: a 48-atom TRIMER ARRAY.
# That is not decoration -- on a uniform chain any light cone cuts a
# NEAREST-NEIGHBOUR bond (18.5 MHz at 6 um), so no useful certified
# horizon exists at all. The trimer array puts the cut across the wide
# INTER-trimer gap instead, where the 1/r^6 tail is weak. Both are run
# below: the chain as a negative control, the trimer array as the gate.
TRIMER = 3                      # atoms per cluster (|LC| = 3)
N_TRIMERS = 4                   # 12 atoms, dim 4096
INTRA = 6.0                     # um, inside a trimer
INTER = 18.0                    # um, gap between trimers
N = 11                          # chain length for the control (dim 2048)
CENTRE = N // 2
T_MAX = 0.30                    # us
N_TIMES = 30
N_REAL = 200


def build_ops(n):
    """Sparse X_i and n_i for an n-site chain."""
    I2 = sp.identity(2, format="csr", dtype=complex)
    X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
    NN = sp.csr_matrix(np.array([[0, 0], [0, 1]], dtype=complex))
    Xs, Ns = [], []
    for i in range(n):
        ops_x, ops_n = [], []
        for j in range(n):
            ops_x.append(X if j == i else I2)
            ops_n.append(NN if j == i else I2)
        ax, an = ops_x[0], ops_n[0]
        for j in range(1, n):
            ax = sp.kron(ax, ops_x[j], format="csr")
            an = sp.kron(an, ops_n[j], format="csr")
        Xs.append(ax)
        Ns.append(an)
    return Xs, Ns


def hamiltonian(pos, Xs, Ns, keep_sites=None):
    """H (or H_LC if keep_sites is given: only terms inside it)."""
    n = len(pos)
    dim = Xs[0].shape[0]
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(n):
        if keep_sites is not None and i not in keep_sites:
            continue
        H = H + (OMEGA / 2.0) * Xs[i] - DELTA * Ns[i]
    for i in range(n):
        for j in range(i + 1, n):
            if keep_sites is not None and (i not in keep_sites
                                           or j not in keep_sites):
                continue
            r = abs(pos[j] - pos[i])
            H = H + (C6 / r**6) * (Ns[i] @ Ns[j])
    return H


def bounds_for(pos, R):
    """(S_local, S_global) in rad/us for light-cone radius R."""
    n = len(pos)
    lc = set(range(max(0, CENTRE - R), min(n, CENTRE + R + 1)))
    s_local = 0.0
    s_global = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if i in lc and j in lc:
                continue                      # kept in H_LC
            v = C6 / abs(pos[j] - pos[i])**6
            s_global += v
            if (i in lc) != (j in lc):
                s_local += v                  # crosses the light cone
    for i in range(n):
        if i not in lc:
            s_global += OMEGA / 2.0 + abs(DELTA)   # omitted on-site
    return s_local, s_global, lc


def trimer_positions(rng):
    """Atom coordinates for the trimer array, with position disorder."""
    pos = []
    x = 0.0
    for t in range(N_TRIMERS):
        for a in range(TRIMER):
            pos.append(x + a * INTRA)
        x += (TRIMER - 1) * INTRA + INTER
    pos = np.array(pos)
    return pos + rng.normal(0.0, DISORDER_SIGMA, len(pos))


def run_trimer(rng, out):
    """The gate proper: |LC| = one trimer, cut across the wide gap."""
    n = N_TRIMERS * TRIMER
    Xs, Ns = build_ops(n)
    centre_trimer = N_TRIMERS // 2
    lc = set(range(centre_trimer * TRIMER,
                   centre_trimer * TRIMER + TRIMER))
    obs_site = centre_trimer * TRIMER + 1          # middle of the trimer
    A = Ns[obs_site]
    times = np.linspace(0.0, T_MAX, N_TIMES)
    t0 = time.perf_counter()
    viol = checked = 0
    worst = 0.0
    s_loc_s, s_glo_s = [], []
    vac = tot = 0
    for real in range(N_REAL):
        pos = trimer_positions(rng)
        s_local = s_global = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if i in lc and j in lc:
                    continue
                v = C6 / abs(pos[j] - pos[i])**6
                s_global += v
                if (i in lc) != (j in lc):
                    s_local += v
        for i in range(n):
            if i not in lc:
                s_global += OMEGA / 2.0 + abs(DELTA)
        s_loc_s.append(s_local)
        s_glo_s.append(s_global)
        H = hamiltonian(pos, Xs, Ns)
        H_lc = hamiltonian(pos, Xs, Ns, keep_sites=lc)
        psi0 = np.zeros(Xs[0].shape[0], dtype=complex)
        psi0[0] = 1.0
        full = expm_multiply(-1j * H, psi0, start=0.0, stop=T_MAX,
                             num=N_TIMES, endpoint=True)
        trunc = expm_multiply(-1j * H_lc, psi0, start=0.0, stop=T_MAX,
                              num=N_TIMES, endpoint=True)
        for k, t in enumerate(times):
            af = float(np.real(np.vdot(full[k], A @ full[k])))
            al = float(np.real(np.vdot(trunc[k], A @ trunc[k])))
            d = abs(af - al)
            cert = 2.0 * t * s_local
            naive = 2.0 * t * s_global
            checked += 1
            if d > cert + 1e-12:
                viol += 1
            if cert > 0:
                worst = max(worst, d / cert)
            if real == 0:
                tot += 1
                if naive > 1.0 and cert <= 1.0:
                    vac += 1
    sl = float(np.mean(s_loc_s))
    sg = float(np.mean(s_glo_s))
    tc = 0.1 / (2.0 * sl)
    rec = {"geometry": (f"{N_TRIMERS} trimers x {TRIMER} atoms, "
                        f"intra {INTRA} um, inter {INTER} um"),
           "light_cone": "one trimer (|LC| = 3)",
           "S_local_rad_per_us": round(sl, 4),
           "S_global_rad_per_us": round(sg, 2),
           "instrument_gain": round(sg / sl, 1),
           "T_cert_0.1_us": round(tc, 4),
           "realisations": N_REAL, "points_checked": checked,
           "interval_violations": viol,
           "worst_diff_over_bound": round(worst, 4),
           "naive_vacuous_points": f"{vac}/{tot}",
           "seconds": round(time.perf_counter() - t0, 1)}
    out["trimer_array"] = rec
    print(f"  TRIMER |LC|=3: T_cert(0.1) = {tc*1000:.0f} ns, gain "
          f"{rec['instrument_gain']}x, violations {viol}/{checked}, "
          f"worst diff/bound {worst:.3f}, naive vacuous {vac}/{tot} "
          f"({rec['seconds']}s)")
    assert viol == 0
    return rec


def main() -> int:
    rng = np.random.default_rng(4242)
    Xs, Ns = build_ops(N)
    A = Ns[CENTRE]                       # observable: centre occupation
    times = np.linspace(0.0, T_MAX, N_TIMES)

    out = {"schema": "oneq-b1-tcert/1",
           "gate": ("B1-alpha (FTQC_BUILD_PROGRAM): T_cert(eps) curve, "
                    "0 interval violations over 200 disorder "
                    "realisations, T_cert(0.1) >= 0.15 us at |LC|=3, "
                    "naive global Duhamel shown vacuous"),
           "physics": {"C6_over_2pi_MHz_um6": 862690.0,
                       "Omega_over_2pi_MHz": 2.5,
                       "Delta_over_2pi_MHz": 1.0,
                       "spacing_um": SPACING,
                       "disorder_sigma_um": DISORDER_SIGMA,
                       "sites": N, "observable": "n at chain centre",
                       "initial_state": "all ground |g...g>"},
           "radii": {}}

    for R in (1, 2, 3, 4):
        t0 = time.perf_counter()
        viol = 0
        checked = 0
        worst_ratio = 0.0
        s_loc_s, s_glo_s = [], []
        vac_points = 0
        tot_points = 0
        for real in range(N_REAL):
            pos = (np.arange(N) * SPACING
                   + rng.normal(0.0, DISORDER_SIGMA, N))
            s_local, s_global, lc = bounds_for(pos, R)
            s_loc_s.append(s_local)
            s_glo_s.append(s_global)
            # exact evolution, full vs truncated, same initial state
            H = hamiltonian(pos, Xs, Ns)
            H_lc = hamiltonian(pos, Xs, Ns, keep_sites=lc)
            psi0 = np.zeros(Xs[0].shape[0], dtype=complex)
            psi0[0] = 1.0                    # |g...g>
            full = expm_multiply(-1j * H, psi0, start=0.0, stop=T_MAX,
                                 num=N_TIMES, endpoint=True)
            trunc = expm_multiply(-1j * H_lc, psi0, start=0.0,
                                  stop=T_MAX, num=N_TIMES, endpoint=True)
            for k, t in enumerate(times):
                a_full = float(np.real(np.vdot(full[k], A @ full[k])))
                a_lc = float(np.real(np.vdot(trunc[k], A @ trunc[k])))
                diff = abs(a_full - a_lc)
                cert = 2.0 * t * s_local
                naive = 2.0 * t * s_global
                checked += 1
                if diff > cert + 1e-12:
                    viol += 1
                if cert > 0:
                    worst_ratio = max(worst_ratio, diff / cert)
                if real == 0:
                    tot_points += 1
                    if naive > 1.0 and cert <= 1.0:
                        vac_points += 1
        s_local_m = float(np.mean(s_loc_s))
        s_global_m = float(np.mean(s_glo_s))
        t_cert_01 = 0.1 / (2.0 * s_local_m)
        rec = {"light_cone_radius": R,
               "S_local_rad_per_us": round(s_local_m, 4),
               "S_global_rad_per_us": round(s_global_m, 2),
               "instrument_gain": round(s_global_m / s_local_m, 1),
               "T_cert_0.1_us": round(t_cert_01, 4),
               "realisations": N_REAL, "points_checked": checked,
               "interval_violations": viol,
               "worst_diff_over_bound": round(worst_ratio, 4),
               "naive_vacuous_points": f"{vac_points}/{tot_points}",
               "seconds": round(time.perf_counter() - t0, 1)}
        out["radii"][f"R={R}"] = rec
        print(f"  |LC|={R}: T_cert(0.1) = {t_cert_01*1000:.0f} ns, "
              f"gain {rec['instrument_gain']}x, violations {viol}/"
              f"{checked}, worst diff/bound {worst_ratio:.3f}, "
              f"naive vacuous {vac_points}/{tot_points} "
              f"({rec['seconds']}s)")
        assert viol == 0, f"R={R}: {viol} interval violations"

    print("  (uniform chain above is the NEGATIVE CONTROL: a "
          "light cone there cuts a nearest-neighbour bond, so "
          "no useful horizon exists at any radius -- a true "
          "physical statement, kept)")
    tri = run_trimer(rng, out)
    out["gate_verdict"] = {
        "zero_violations": (all(v["interval_violations"] == 0
                                for v in out["radii"].values())
                            and tri["interval_violations"] == 0),
        "T_cert_0.1_at_LC3_us": tri["T_cert_0.1_us"],
        "meets_0.15us_target": tri["T_cert_0.1_us"] >= 0.15,
        "instrument_gain_at_LC3": tri["instrument_gain"],
        "meets_10x_target": tri["instrument_gain"] >= 10.0,
        "naive_vacuous": tri["naive_vacuous_points"],
        "control": ("uniform chain: no useful horizon at any radius "
                    "(nearest-neighbour bond is cut)")}
    print(f"\nB1-alpha verdict: {json.dumps(out['gate_verdict'])}")
    dest = ROOT / "evidence" / "b1_tcert.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
