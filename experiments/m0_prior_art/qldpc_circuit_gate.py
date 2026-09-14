r"""CIRCUIT-LEVEL noise, certified: the qLDPC ladder on a real stim DEM.

The last labeled gap in lane 2's noise ladder. Code-capacity and
phenomenological are measured at 100%; this experiment decodes a genuine
circuit-level detector error model -- correlated mechanisms, hook errors,
measurement noise, everything stim puts in the DEM of a rotated surface-code
memory -- and certifies BP-OSD's answers with the same machinery.

THE OBSTACLE THIS RUN EXISTS TO MEASURE: circuit-level detectors have high
degree (a detector can touch dozens of error mechanisms), and full Feldman
facet families scale 2^(deg-1). The unlock is DEGREE-CAPPED facets
(qldpc_cert.feldman_dual max_facet_size): keeping only the facets with
|F| <= k is keeping a SUBSET of the true constraints, which only enlarges
the LP's feasible region -- the bound stays SOUND, and what the cap costs is
tightness, measured here per cap. On random ensembles |F| <= 3 matched full
families exactly (40/40 soundness, identical tightness); circuit level is
where that observation earns or loses its keep.

Honest scope, before any number: min-weight conformance under log-likelihood
weights, NOT maximum-likelihood over degenerate mechanism cosets -- the same
caveat as every lane-2 artifact, sharper here because circuit-level DEMs are
exactly where degeneracy is richest. And the observable check is reported
separately: a min-weight solution can flip a different logical than the
truth even when PROVEN minimum-weight.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,  # noqa: E402
                             exact_milp, feldman_dual, feldman_dual_lazy)


def dem_to_matrices(dem):
    """H (detectors x mechanisms), observables matrix, weights from the DEM."""
    import stim
    rows: dict[int, list[int]] = {}
    obs_rows: dict[int, list[int]] = {}
    probs: list[float] = []
    col = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets = []
        obs = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        probs.append(p)
        for d in dets:
            rows.setdefault(d, []).append(col)
        for o in obs:
            obs_rows.setdefault(o, []).append(col)
        col += 1
    n_det = dem.num_detectors
    checks = [tuple(sorted(rows.get(d, ()))) for d in range(n_det)]
    weights = {i: float(np.log((1 - p) / p)) for i, p in enumerate(probs)}
    return checks, weights, obs_rows, col


def main() -> int:
    import stim

    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--shots", type=int, default=150)
    ap.add_argument("--facet-cap", type=int, default=3)
    ap.add_argument("--lazy", action="store_true",
                    help="lazy facet separation: full-family tightness at "
                         "any detector degree, no cap, no enumeration")
    ap.add_argument("--rpc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=a.distance,
        rounds=a.rounds, after_clifford_depolarization=a.p,
        after_reset_flip_probability=a.p, before_round_data_depolarization=a.p,
        before_measure_flip_probability=a.p)
    dem = circ.detector_error_model(decompose_errors=False)
    checks, weights, obs_rows, n_mech = dem_to_matrices(dem)
    degs = sorted(len(c) for c in checks)
    print(f"circuit-level d={a.distance} r={a.rounds} p={a.p}: "
          f"{len(checks)} detectors, {n_mech} mechanisms, detector degree "
          f"median {degs[len(degs)//2]} max {degs[-1]} "
          f"(full facets at max degree: 2^{degs[-1]-1}; cap |F|<={a.facet_cap})")

    from ldpc import BpOsdDecoder
    from scipy.sparse import lil_matrix
    H = lil_matrix((len(checks), n_mech), dtype=np.uint8)
    for j, c in enumerate(checks):
        for i in c:
            H[j, i] = 1
    H = H.tocsr()
    dec = BpOsdDecoder(H, error_channel=[
        float(1 / (1 + np.exp(w))) for w in
        (weights[i] for i in range(n_mech))],
        max_iter=40, bp_method="ms", schedule="parallel",
        osd_method="osd_cs", osd_order=5)

    sampler = dem.compile_sampler(seed=a.seed)
    dets, obs, _ = sampler.sample(shots=a.shots, return_errors=False)

    nontrivial = proven = repaired = 0
    consistent = 0
    milp_exact = 0
    obs_match = obs_checked = 0
    gaps = []
    t_cert = 0.0
    obs_mat = obs_rows
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

        t0 = time.perf_counter()
        if a.lazy:
            cert, bound, primal = feldman_dual_lazy(
                checks, weights, list(syndrome), return_primal=True)
        else:
            cert, bound, primal = feldman_dual(
                checks, weights, list(syndrome), return_primal=True,
                rpc_rounds=a.rpc, max_facet_size=a.facet_cap)
        t_cert += time.perf_counter() - t0
        if cert is None:
            gaps.append(float("nan"))
            continue
        chosen = None
        r = check_qldpc(checks=checks, weights=weights,
                        syndrome=list(syndrome),
                        cert=QldpcCertificate(error_support=support,
                                              facet_duals=cert.facet_duals,
                                              box_duals=cert.box_duals))
        if r.accepted:
            proven += 1
            chosen = support
        elif "IMPOSSIBLE" in r.reason:
            print(f"IMPOSSIBLE: {r.reason}", file=sys.stderr)
            return 2
        elif primal is not None:
            r2 = check_qldpc(checks=checks, weights=weights,
                             syndrome=list(syndrome),
                             cert=QldpcCertificate(error_support=primal,
                                                   facet_duals=cert.facet_duals,
                                                   box_duals=cert.box_duals))
            if r2.accepted:
                repaired += 1
                chosen = primal
        if chosen is None and a.lazy:
            # TIERED: the 5 ms lazy rung refused; pay the enumerated+RPC
            # rung only here, exactly the ladder pattern everywhere else
            t0 = time.perf_counter()
            cert2, bound2, primal2 = feldman_dual(
                checks, weights, list(syndrome), return_primal=True,
                rpc_rounds=a.rpc, max_facet_size=a.facet_cap)
            t_cert += time.perf_counter() - t0
            if cert2 is not None:
                bound = max(bound, bound2)
                for sup2, ctr in ((support, "proven"), (primal2, "repaired")):
                    if sup2 is None or chosen is not None:
                        continue
                    r3 = check_qldpc(checks=checks, weights=weights,
                                     syndrome=list(syndrome),
                                     cert=QldpcCertificate(
                                         error_support=tuple(sup2),
                                         facet_duals=cert2.facet_duals,
                                         box_duals=cert2.box_duals))
                    if r3.accepted:
                        chosen = tuple(sup2)
                        if ctr == "proven":
                            proven += 1
                        else:
                            repaired += 1
        if chosen is None:
            msup, mw = exact_milp(checks, weights, list(syndrome),
                                  time_limit=30.0)
            if mw is not None:
                milp_exact += 1
                chosen = msup
            gaps.append(w_hat - bound)
        # OBSERVABLE CHECK, reported separately from optimality: does the
        # chosen correction predict the sampled logical flip?
        if chosen is not None:
            obs_checked += 1
            pred = [sum(1 for i in chosen if i in set(obs_mat.get(o, ())))
                    % 2 for o in range(obs.shape[1])]
            if all(int(obs[s, o]) == pred[o] for o in range(obs.shape[1])):
                obs_match += 1

    closed = proven + repaired
    print(f"  nontrivial              : {nontrivial}")
    print(f"  BP-OSD syndrome-valid   : {consistent}/{nontrivial}")
    print(f"  PROVEN min-weight       : {proven}/{consistent} "
          f"({100 * proven / max(consistent, 1):.1f}%)")
    print(f"  REPAIRED (LP integral)  : {repaired}")
    print(f"  CERTIFIED total         : {closed}/{consistent} "
          f"({100 * closed / max(consistent, 1):.2f}%)")
    if gaps:
        g = np.array([x for x in gaps if x == x])
        print(f"  residue                 : {len(gaps)} (MILP exact on "
              f"{milp_exact}); gap mean {g.mean() if len(g) else 0:.3f}")
    print(f"  observable agreement    : {obs_match}/{obs_checked} "
          f"(min-weight vs sampled logical; degeneracy caveat applies)")
    print(f"  certificate cost        : "
          f"{1e3 * t_cert / max(consistent, 1):.0f} ms/shot")

    out = {"schema": "oneq-qldpc-circuit-gate/1",
           "circuit": f"stim surface_code:rotated_memory_x d={a.distance} "
                      f"r={a.rounds} circuit-level p={a.p}",
           "detectors": len(checks), "mechanisms": n_mech,
           "detector_degree_median": degs[len(degs) // 2],
           "detector_degree_max": degs[-1],
           "facet_cap": ("lazy/full-family" if a.lazy else a.facet_cap),
           "soundness_of_cap": ("capped facets are a SUBSET of the true "
                                "facet family; the LP feasible region only "
                                "grows, so the dual bound remains a valid "
                                "lower bound -- the cap costs tightness "
                                "only, and tightness is what this artifact "
                                "measures"),
           "shots": a.shots, "nontrivial": nontrivial,
           "bp_osd_syndrome_valid": consistent,
           "proven": proven, "repaired": repaired,
           "certified_total": closed,
           "certified_rate": closed / max(consistent, 1),
           "residue": len(gaps), "milp_exact_on_residue": milp_exact,
           "observable_agreement": [obs_match, obs_checked],
           "cert_ms_per_shot": 1e3 * t_cert / max(consistent, 1),
           "honest_scope": ("min-weight conformance under log-likelihood "
                            "weights, NOT maximum-likelihood over degenerate "
                            "cosets; circuit-level DEMs are where degeneracy "
                            "is richest, and the observable-agreement row "
                            "carries that caveat explicitly")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "qldpc_circuit_gate.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
