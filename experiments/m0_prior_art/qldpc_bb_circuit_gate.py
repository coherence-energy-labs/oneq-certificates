r"""THE FULL STACK: [[144,12,12]] with its OWN syndrome circuit, certified.

Every prior lane-2 artifact either assumed perfect extraction (code
capacity), modeled it phenomenologically, or ran circuit-level on a SURFACE
code. This experiment builds the bivariate-bicycle gross code's own
syndrome-extraction circuit in stim -- 144 data + 144 ancilla qubits, CNOT
schedules read off the monomial decomposition A = x^3+y+y^2, B = y^3+x+x^2
(each monomial a permutation, so each CNOT layer is a disjoint matching) --
runs it under uniform circuit-level depolarizing noise, and certifies BP-OSD
on the resulting DEM with the lazy-separation Feldman machinery.

The construction is stated exactly, because schedules matter:

  SCHEDULE: sequential, NOT the depth-7 interleaved schedule of Bravyi et
  al. (Nature 627, 778) -- X-extraction's six CNOT layers, then
  Z-extraction's six, per round. This is a VALID fault-tolerant circuit
  whose DEM is exactly what is decoded; no circuit-distance claim is made
  for it, and the certified rate applies to THIS circuit. The optimized
  schedule is a drop-in refinement.

  EXPERIMENT: memory-Z. Data initialized |0>, r noisy rounds (both sectors
  extracted, all noise present), final transversal Z measurement. Detectors
  are the Z-sector's (X-check outcomes do not affect the Z logical);
  X-extraction's noise STILL enters the DEM through its back-action on
  data, which is the point of building the full circuit.

  OBSERVABLE: one logical Z, computed by GF(2) elimination as an element of
  ker(H_X) outside rowspace(H_Z), included over the final data measurement.

  NOISE: X_ERROR(p) after resets and before measurements, DEPOLARIZE2(p)
  after every CNOT, DEPOLARIZE1(p) on data between rounds -- the standard
  uniform circuit-level depolarizing family.

Detector degrees here dwarf the surface-code case; only the lazy separator
makes the LP buildable at all (the enumerated path would need 2^(deg-1)
rows). Tiering as in qldpc_circuit_gate: lazy first, MILP on the residue.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "m0_prior_art"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from qldpc_bb_gate import bb_gross_code, gf2_row_reduce, gf2_in_rowspace  # noqa: E402
from qldpc_circuit_gate import dem_to_matrices  # noqa: E402

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,  # noqa: E402
                             exact_milp, feldman_dual_lazy)


def monomials():
    """The six permutation matrices A1..A3, B1..B3 of the gross code."""
    l, m = 12, 6
    Sl = np.roll(np.eye(l, dtype=np.uint8), 1, axis=1)
    Sm = np.roll(np.eye(m, dtype=np.uint8), 1, axis=1)
    x = np.kron(Sl, np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), Sm)
    x3 = np.linalg.matrix_power(x, 3) % 2
    y2 = (y @ y) % 2
    y3 = (y @ y2) % 2
    x2 = (x @ x) % 2
    return [x3, y, y2], [y3, x, x2]


def _gf2_rref(M):
    """Fully reduced row-echelon form over GF(2), with pivot columns."""
    M = (M.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    piv_cols = []
    r = 0
    for c in range(cols):
        pivot = None
        for rr in range(r, rows):
            if M[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        for rr in range(rows):
            if rr != r and M[rr, c]:
                M[rr] ^= M[r]
        piv_cols.append(c)
        r += 1
        if r == rows:
            break
    return M, piv_cols


def logical_z():
    """One logical Z: in ker(H_X), outside rowspace(H_Z).

    The kernel is built from the FULLY REDUCED echelon form -- the first
    version back-substituted against a non-reduced one, produced vectors
    with H_X v != 0 for every free column, and the self-check refused them
    all. The self-check stays: a 'logical' that fails H_X v = 0 or lives in
    the stabilizer group would silently score the observable against noise.
    """
    HX, HZ = bb_gross_code()
    n = HX.shape[1]
    R, piv = _gf2_rref(HX)
    piv_set = set(piv)
    zb, zpiv = gf2_row_reduce(HZ % 2)
    for f in (c for c in range(n) if c not in piv_set):
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for r_i, pc in enumerate(piv):
            if R[r_i, f]:
                v[pc] = 1
        assert not ((HX @ v) % 2).any(), "kernel construction broken"
        if not gf2_in_rowspace(zb, zpiv, v, n):
            return [int(i) for i in np.flatnonzero(v)]
    raise AssertionError("no logical Z found: elimination is broken")


def build_circuit(r: int, p: float) -> "object":
    import stim
    A_perms, B_perms = monomials()
    nhalf = 72
    L = list(range(0, 72))
    R = list(range(72, 144))
    XA = list(range(144, 216))          # X-check ancillas
    ZA = list(range(216, 288))          # Z-check ancillas
    data = L + R
    lz = logical_z()

    c = stim.Circuit()
    c.append("R", data)
    c.append("X_ERROR", data, p)

    def cnot_layer(pairs):
        flat = []
        for (a, b) in pairs:
            flat += [a, b]
        c.append("CX", flat)
        c.append("DEPOLARIZE2", flat, p)

    zmeas_offsets = []                  # measurement record offsets per round
    mcount = 0
    for rd in range(r):
        # ---- X-sector extraction (outcomes unused; noise is the point) ----
        c.append("RX", XA)
        c.append("Z_ERROR", XA, p)
        for k in range(3):              # X-check i -> L data j, A_k[i,j]=1
            P = A_perms[k]
            cnot_layer([(XA[i], L[int(np.flatnonzero(P[i])[0])])
                        for i in range(nhalf)])
        for k in range(3):
            P = B_perms[k]
            cnot_layer([(XA[i], R[int(np.flatnonzero(P[i])[0])])
                        for i in range(nhalf)])
        c.append("Z_ERROR", XA, p)
        c.append("MX", XA)
        mcount += nhalf
        # ---- Z-sector extraction ----
        c.append("R", ZA)
        c.append("X_ERROR", ZA, p)
        for k in range(3):              # L data j -> Z-check i, B_k[j,i]=1
            P = B_perms[k]
            cnot_layer([(L[j], ZA[int(np.flatnonzero(P[j])[0])])
                        for j in range(nhalf)])
        for k in range(3):
            P = A_perms[k]
            cnot_layer([(R[j], ZA[int(np.flatnonzero(P[j])[0])])
                        for j in range(nhalf)])
        c.append("X_ERROR", ZA, p)
        c.append("M", ZA)
        mcount += nhalf
        zmeas_offsets.append(mcount)    # records end at mcount
        # detectors: first round absolute, later rounds differences
        for i in range(nhalf):
            back = -(nhalf - i)
            if rd == 0:
                c.append("DETECTOR", [stim.target_rec(back)])
            else:
                prev = back - 2 * nhalf     # one full round earlier (X+Z)
                c.append("DETECTOR", [stim.target_rec(back),
                                      stim.target_rec(prev)])
        c.append("DEPOLARIZE1", data, p)

    # final transversal Z measurement of data
    c.append("X_ERROR", data, p)
    c.append("M", data)
    mcount += len(data)
    HX, HZ = bb_gross_code()
    for i in range(nhalf):
        sup = [int(j) for j in np.flatnonzero(HZ[i])]
        targs = [stim.target_rec(-(len(data) - j)) for j in sup]
        targs.append(stim.target_rec(-(len(data)) - (nhalf - i)))
        c.append("DETECTOR", targs)
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(-(len(data) - j)) for j in lz], 0)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    t0 = time.perf_counter()
    circ = build_circuit(a.rounds, a.p)
    dem = circ.detector_error_model(decompose_errors=False)
    checks, weights, obs_rows, n_mech = dem_to_matrices(dem)
    degs = sorted(len(c) for c in checks)
    print(f"[[144,12,12]] OWN CIRCUIT r={a.rounds} p={a.p}: "
          f"{len(checks)} detectors, {n_mech} mechanisms, degree median "
          f"{degs[len(degs)//2]} max {degs[-1]} "
          f"(built in {time.perf_counter()-t0:.0f}s; lazy separation only)")

    from ldpc import BpOsdDecoder
    from scipy.sparse import lil_matrix
    H = lil_matrix((len(checks), n_mech), dtype=np.uint8)
    for j, cc in enumerate(checks):
        for i in cc:
            H[j, i] = 1
    H = H.tocsr()
    dec = BpOsdDecoder(H, error_channel=[
        float(1 / (1 + np.exp(weights[i]))) for i in range(n_mech)],
        max_iter=40, bp_method="ms", schedule="parallel",
        osd_method="osd_cs", osd_order=5)

    sampler = dem.compile_sampler(seed=a.seed)
    dets, obs, _ = sampler.sample(shots=a.shots, return_errors=False)

    nontrivial = proven = repaired = 0
    consistent = milp_exact = 0
    obs_match = obs_checked = 0
    gaps = []
    t_cert = 0.0
    for s in range(dets.shape[0]):
        syndrome = dets[s].astype(np.uint8)
        if not syndrome.any():
            continue
        nontrivial += 1
        e_hat = np.asarray(dec.decode(syndrome)).astype(np.uint8)
        if not ((H @ e_hat) % 2 == syndrome).all():
            continue
        consistent += 1
        support = tuple(int(i) for i in np.flatnonzero(e_hat))
        w_hat = float(sum(weights[i] for i in support))
        t1 = time.perf_counter()
        cert, bound, primal = feldman_dual_lazy(
            checks, weights, list(syndrome), return_primal=True)
        t_cert += time.perf_counter() - t1
        chosen = None
        if cert is not None:
            r1 = check_qldpc(checks=checks, weights=weights,
                             syndrome=list(syndrome),
                             cert=QldpcCertificate(
                                 error_support=support,
                                 facet_duals=cert.facet_duals,
                                 box_duals=cert.box_duals))
            if r1.accepted:
                proven += 1
                chosen = support
            elif "IMPOSSIBLE" in r1.reason:
                print(f"IMPOSSIBLE: {r1.reason}", file=sys.stderr)
                return 2
            elif primal is not None:
                r2 = check_qldpc(checks=checks, weights=weights,
                                 syndrome=list(syndrome),
                                 cert=QldpcCertificate(
                                     error_support=primal,
                                     facet_duals=cert.facet_duals,
                                     box_duals=cert.box_duals))
                if r2.accepted:
                    repaired += 1
                    chosen = primal
        if chosen is None:
            msup, mw = exact_milp(checks, weights, list(syndrome),
                                  time_limit=60.0)
            if mw is not None:
                milp_exact += 1
                chosen = msup
            gaps.append(w_hat - (bound if cert else 0.0))
        if chosen is not None:
            obs_checked += 1
            pred = sum(1 for i in chosen
                       if i in set(obs_rows.get(0, ()))) % 2
            obs_match += int(int(obs[s, 0]) == pred)

    closed = proven + repaired
    print(f"  nontrivial              : {nontrivial}")
    print(f"  BP-OSD syndrome-valid   : {consistent}/{nontrivial}")
    print(f"  PROVEN min-weight       : {proven}/{consistent} "
          f"({100 * proven / max(consistent, 1):.1f}%)")
    print(f"  REPAIRED (LP integral)  : {repaired}")
    print(f"  CERTIFIED total         : {closed}/{consistent} "
          f"({100 * closed / max(consistent, 1):.2f}%)")
    if gaps:
        g = np.array(gaps)
        print(f"  residue                 : {len(gaps)} (MILP exact "
              f"{milp_exact}); gap mean {g.mean():.3f}")
    print(f"  observable agreement    : {obs_match}/{obs_checked}")
    print(f"  certificate cost        : "
          f"{1e3 * t_cert / max(consistent, 1):.0f} ms/shot")

    out = {"schema": "oneq-qldpc-bb-circuit-gate/1",
           "circuit": f"[[144,12,12]] gross code, OWN syndrome-extraction "
                      f"circuit (sequential schedule, stated non-optimized; "
                      f"memory-Z; both sectors extracted, Z-sector decoded), "
                      f"r={a.rounds}, uniform circuit-level p={a.p}",
           "detectors": len(checks), "mechanisms": n_mech,
           "detector_degree_median": degs[len(degs) // 2],
           "detector_degree_max": degs[-1],
           "shots": a.shots, "nontrivial": nontrivial,
           "bp_osd_syndrome_valid": consistent,
           "proven": proven, "repaired": repaired,
           "certified_total": closed,
           "certified_rate": closed / max(consistent, 1),
           "residue": len(gaps), "milp_exact_on_residue": milp_exact,
           "observable_agreement": [obs_match, obs_checked],
           "cert_ms_per_shot": 1e3 * t_cert / max(consistent, 1),
           "honest_scope": ("sequential CNOT schedule, NOT the depth-7 "
                            "interleaved one -- valid circuit, no circuit-"
                            "distance claim; min-weight conformance, not ML "
                            "over degenerate cosets; certified rate applies "
                            "to THIS circuit")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "qldpc_bb_circuit_gate.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
