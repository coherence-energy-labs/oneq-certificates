r"""THE REFERENCE SCHEDULE: Bravyi et al. Table 5, exactified.

The audit's primary-experiment requirement, closed: the [[144,12,12]]
gross code under the PUBLISHED depth-8 syndrome cycle (7 CNOT rounds +
init/measure; Nature 627, 778, Methods Table 5 / Eq. (9)), decoded and
EXACT-CERTIFIED per shot -- the first circuit-level rung to get the
full canonical treatment (integer fixed-point DEM weights, exact
rational certificates, reference-checker pass on every certificate,
tier 3A/3B solver statuses, per-shot records, configuration binding).

THE SCHEDULE, verbatim from Table 5 (X-check i acts on q(L,A_p(i)),
q(R,B_p(i)); Z-check i on q(L,B_p^T(i)), q(R,A_p^T(i)); term
assignment follows the written order A = x^3 + y + y^2 = A1+A2+A3,
B = y^3 + x + x^2 = B1+B2+B3):

  R1: InitX q(X);  CNOT q(R,A1^T(i)) -> q(Z,i);  Idle q(L)
  R2: CNOT q(X,i) -> q(L,A2(i));  CNOT q(R,A3^T(i)) -> q(Z,i)
  R3: CNOT q(X,i) -> q(R,B2(i));  CNOT q(L,B1^T(i)) -> q(Z,i)
  R4: CNOT q(X,i) -> q(R,B1(i));  CNOT q(L,B2^T(i)) -> q(Z,i)
  R5: CNOT q(X,i) -> q(R,B3(i));  CNOT q(L,B3^T(i)) -> q(Z,i)
  R6: CNOT q(X,i) -> q(L,A1(i));  CNOT q(R,A2^T(i)) -> q(Z,i)
  R7: CNOT q(X,i) -> q(L,A3(i));  MeasZ q(Z);  Idle q(R)
  R8: MeasX q(X);  InitZ q(Z);  Idle q(L,R)

(q(Z) is initialized in R8 of the previous cycle; the first cycle
initializes it up front, the paper's footnote 2.)

EXPERIMENT: memory-Z as in qldpc_bb_circuit_gate -- data |0>, r noisy
cycles, final transversal Z measurement; detectors on the Z-check
(q(Z)) outcomes; one logical Z observable. NOISE: the same uniform
family as the sequential gate for comparability (X_ERROR after
resets/before Z-basis measurements, Z_ERROR around X-basis
init/measurement, DEPOLARIZE2 after every CNOT, DEPOLARIZE1 on data
between cycles) -- NOT the paper's exact error model (which also
faults idles); stated, not blended.

VALIDATION BEFORE NOISE: at p=0 every detector must be deterministic
zero and the observable zero over a sample -- a wrong schedule cannot
pass this, because a mis-ordered layer breaks the telescoping
transformation (I,0,0,0;0,A,B,0) -> (I,A,B,0) the paper proves.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "m0_prior_art"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from qldpc_bb_gate import bb_gross_code  # noqa: E402
from qldpc_bb_circuit_gate import logical_z, monomials  # noqa: E402
from qldpc_circuit_gate import dem_to_matrices  # noqa: E402

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,  # noqa: E402
                             check_qldpc_exact, exact_milp,
                             exactify_certificate, feldman_dual_lazy)

REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
REF_SHA = "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"


def build_refsched_circuit(r: int, p: float) -> "object":
    import stim
    A_perms, B_perms = monomials()
    nhalf = 72
    L = list(range(0, 72))
    Rr = list(range(72, 144))
    XA = list(range(144, 216))
    ZA = list(range(216, 288))
    data = L + Rr
    lz = logical_z()

    def col_of_row(P, i):               # j = P(i): P[i, j] = 1
        return int(np.flatnonzero(P[i])[0])

    def row_of_col(P, i):               # j = P^T(i): P[j, i] = 1
        return int(np.flatnonzero(P[:, i])[0])

    c = stim.Circuit()
    c.append("R", data)
    c.append("X_ERROR", data, p)
    c.append("R", ZA)                   # first-cycle InitZ (footnote 2)
    c.append("X_ERROR", ZA, p)

    def cnots(pairs):
        flat = []
        for (a, b) in pairs:
            flat += [a, b]
        c.append("CX", flat)
        c.append("DEPOLARIZE2", flat, p)

    for rd in range(r):
        # R1: InitX q(X); R-data -> Z-anc via A1^T
        c.append("RX", XA)
        c.append("Z_ERROR", XA, p)
        cnots([(Rr[row_of_col(A_perms[0], i)], ZA[i])
               for i in range(nhalf)])
        # R2: X-anc -> L via A2 ; R -> Z via A3^T
        cnots([(XA[i], L[col_of_row(A_perms[1], i)]) for i in range(nhalf)]
              + [(Rr[row_of_col(A_perms[2], i)], ZA[i])
                 for i in range(nhalf)])
        # R3: X -> R via B2 ; L -> Z via B1^T
        cnots([(XA[i], Rr[col_of_row(B_perms[1], i)]) for i in range(nhalf)]
              + [(L[row_of_col(B_perms[0], i)], ZA[i])
                 for i in range(nhalf)])
        # R4: X -> R via B1 ; L -> Z via B2^T
        cnots([(XA[i], Rr[col_of_row(B_perms[0], i)]) for i in range(nhalf)]
              + [(L[row_of_col(B_perms[1], i)], ZA[i])
                 for i in range(nhalf)])
        # R5: X -> R via B3 ; L -> Z via B3^T
        cnots([(XA[i], Rr[col_of_row(B_perms[2], i)]) for i in range(nhalf)]
              + [(L[row_of_col(B_perms[2], i)], ZA[i])
                 for i in range(nhalf)])
        # R6: X -> L via A1 ; R -> Z via A2^T
        cnots([(XA[i], L[col_of_row(A_perms[0], i)]) for i in range(nhalf)]
              + [(Rr[row_of_col(A_perms[1], i)], ZA[i])
                 for i in range(nhalf)])
        # R7: X -> L via A3 ; MeasZ q(Z)
        cnots([(XA[i], L[col_of_row(A_perms[2], i)]) for i in range(nhalf)])
        c.append("X_ERROR", ZA, p)
        c.append("M", ZA)
        for i in range(nhalf):
            back = -(nhalf - i)
            if rd == 0:
                c.append("DETECTOR", [stim.target_rec(back)])
            else:
                c.append("DETECTOR", [stim.target_rec(back),
                                      stim.target_rec(back - 2 * nhalf)])
        # R8: MeasX q(X); InitZ q(Z)
        c.append("Z_ERROR", XA, p)
        c.append("MX", XA)
        c.append("R", ZA)
        c.append("X_ERROR", ZA, p)
        c.append("DEPOLARIZE1", data, p)

    c.append("X_ERROR", data, p)
    c.append("M", data)
    _HX, HZ = bb_gross_code()
    ndata = len(data)
    for i in range(nhalf):
        sup = [int(j) for j in np.flatnonzero(HZ[i])]
        targs = [stim.target_rec(-(ndata - j)) for j in sup]
        # last cycle's ZA record i: data(144) + XA(72) before it
        targs.append(stim.target_rec(-(ndata + nhalf + (nhalf - i))))
        c.append("DETECTOR", targs)
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(-(ndata - j)) for j in lz], 0)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.002)
    ap.add_argument("--shots", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--prefix", type=str, default="",
                    help="output-name prefix (held-out runs)")
    ap.add_argument("--bd-close", action="store_true",
                    help="close solver fallbacks with branch-dual trees")
    a = ap.parse_args()
    import stim

    # ---- noiseless validation: the schedule must be deterministic ----
    circ0 = build_refsched_circuit(a.rounds, 0.0)
    dets0, obs0 = circ0.compile_detector_sampler().sample(
        shots=32, separate_observables=True)
    assert not dets0.any(), "SCHEDULE BROKEN: noiseless detectors fired"
    assert not obs0.any(), "SCHEDULE BROKEN: noiseless observable flipped"
    print(f"noiseless validation: {dets0.shape[1]} detectors all "
          f"deterministic-zero over 32 shots -- Table 5 enacted correctly")

    got = hashlib.sha256(REF.read_bytes()).hexdigest()
    assert got == REF_SHA, "reference checker tampered"
    spec = importlib.util.spec_from_file_location("ref_checker", REF)
    refmod = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = refmod
    spec.loader.exec_module(refmod)

    t0 = time.perf_counter()
    circ = build_refsched_circuit(a.rounds, a.p)
    dem = circ.detector_error_model(decompose_errors=False)
    checks, wllr, obs_rows, n_mech = dem_to_matrices(dem)
    degs = sorted(len(c) for c in checks)
    print(f"REFERENCE SCHEDULE r={a.rounds} p={a.p}: {len(checks)} "
          f"detectors, {n_mech} mechanisms, degree median "
          f"{degs[len(degs)//2]} max {degs[-1]} "
          f"({time.perf_counter()-t0:.0f}s build)")

    # fixed-point integer weights: the normative objective
    wint = {i: int(math.floor(1000 * float(wllr[i]) + 0.5))
            for i in range(n_mech)}
    wf = {i: float(wint[i]) for i in range(n_mech)}
    wx = {i: Fraction(wint[i]) for i in range(n_mech)}
    wblob = json.dumps([wint[k] for k in sorted(wint)]).encode()
    dense = [[0] * n_mech for _ in range(len(checks))]
    for j, cc in enumerate(checks):
        for i in cc:
            dense[j][int(i)] = 1

    from ldpc import BpOsdDecoder
    from scipy.sparse import lil_matrix
    H = lil_matrix((len(checks), n_mech), dtype=np.uint8)
    for j, cc in enumerate(checks):
        for i in cc:
            H[j, i] = 1
    H = H.tocsr()
    dec = BpOsdDecoder(H, error_channel=[
        float(1 / (1 + np.exp(wllr[i]))) for i in range(n_mech)],
        max_iter=40, bp_method="ms", schedule="parallel",
        osd_method="osd_cs", osd_order=5)

    sampler = dem.compile_sampler(seed=a.seed)
    dets, obs, _ = sampler.sample(shots=a.shots, return_errors=False)

    c = {"nontrivial": 0, "trivial_tier0": 0, "tier1": 0, "tier2": 0,
         "exact_certified": 0, "ref_checked": 0, "ref_agreed": 0,
         "tier3A": 0, "tier3B": 0, "residue_unsolved": 0,
         "float_exact_disagreements": 0}
    obs_match = obs_checked = 0
    recs = []
    t_lp = 0.0
    for s in range(dets.shape[0]):
        syn = dets[s].astype(np.uint8)
        if not syn.any():
            c["trivial_tier0"] += 1
            continue
        c["nontrivial"] += 1
        e_hat = np.asarray(dec.decode(syn)).astype(np.uint8)
        bp_sup = tuple(int(i) for i in np.flatnonzero(e_hat))
        t1 = time.perf_counter()
        cert, bound, primal = feldman_dual_lazy(checks, wf, list(syn),
                                                return_primal=True)
        t_lp += time.perf_counter() - t1
        chosen, tier = None, None
        fc = None
        if cert is not None:
            f1 = QldpcCertificate(error_support=bp_sup,
                                  facet_duals=cert.facet_duals,
                                  box_duals=cert.box_duals)
            if check_qldpc(checks=checks, weights=wf, syndrome=list(syn),
                           cert=f1).accepted:
                chosen, tier, fc = bp_sup, "tier1", f1
            elif primal is not None:
                f2 = QldpcCertificate(error_support=tuple(primal),
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)
                if check_qldpc(checks=checks, weights=wf,
                               syndrome=list(syn), cert=f2).accepted:
                    chosen, tier, fc = tuple(primal), "tier2", f2
        rec = {"schema": "oneq-qldpc-shot/1",
               "experiment_id": "bb_refsched_r3_p002",
               "shot_index": c["nontrivial"] - 1,
               "bposd": {"candidate": list(bp_sup),
                         "objective": int(sum(wint[i] for i in bp_sup))}}
        if chosen is not None:
            c[tier] += 1
            ec = exactify_certificate(fc)
            rx = check_qldpc_exact(checks=checks, weights=wx,
                                   syndrome=list(syn), cert=ec)
            if rx.accepted:
                c["exact_certified"] += 1
                e = [0] * n_mech
                for i in chosen:
                    e[int(i)] = 1
                facets = []
                for (j, F, y) in ec.facet_duals:
                    rows = (sorted(int(x) for x in j)
                            if isinstance(j, (tuple, list, frozenset))
                            else [int(j)])
                    facets.append({"rows": rows,
                                   "F": sorted(int(i) for i in F),
                                   "y": {"num": y.numerator,
                                         "den": y.denominator}})
                doc = {"H": dense, "s": [int(b) for b in syn],
                       "weights": [{"num": wint[i], "den": 1}
                                   for i in range(n_mech)],
                       "candidate": e, "facets": facets}
                rep = refmod.check_certificate(doc)
                c["ref_checked"] += 1
                if rep["objective_optimal"]:
                    c["ref_agreed"] += 1
            else:
                c["float_exact_disagreements"] += 1
            rec["receipt"] = {"tier": tier,
                              "exact": bool(rx.accepted),
                              "answer": [int(i) for i in chosen]}
        else:
            sup, _w, st = exact_milp(checks, wf, list(syn),
                                     time_limit=60.0, return_status=True)
            if sup is None:
                c["residue_unsolved"] += 1
                rec["fallback"] = {"status": st, "answer": None}
            else:
                chosen = sup
                closed_bd = False
                if a.bd_close:
                    from oneq.qldpc_cert import certified_bnb
                    from oneq.qldpc_check import check_qldpc_bnb
                    U = int(sum(wint[int(i)] for i in sup))
                    tree, _nn = certified_bnb(checks, wf, list(syn), U,
                                              rpc_rounds=6,
                                              max_nodes=60000,
                                              use_lazy=True)
                    if tree is not None:
                        wxb = {i: Fraction(wint[i])
                               for i in range(n_mech)}
                        rb = check_qldpc_bnb(
                            checks=checks, weights=wxb,
                            syndrome=list(syn),
                            error_support=tuple(int(i) for i in sup),
                            tree=tree)
                        closed_bd = bool(rb.accepted)
                if closed_bd:
                    c.setdefault("tierBD", 0)
                    c["tierBD"] += 1
                    c["exact_certified"] += 1
                    rec["receipt"] = {"tier": "tierBD",
                                      "answer": [int(i) for i in sup]}
                else:
                    key = ("tier3A" if st["tier"].startswith("3A")
                           else "tier3B")
                    c[key] += 1
                    rec["fallback"] = {"status": st,
                                       "answer": [int(i) for i in sup]}
        if chosen is not None:
            obs_checked += 1
            pred = sum(1 for i in chosen
                       if i in set(obs_rows.get(0, ()))) % 2
            hit = int(int(obs[s, 0]) == pred)
            obs_match += hit
            rec["retrospective_simulation"] = {"observable_match": bool(hit)}
        recs.append(rec)

    pdir = ROOT / "evidence" / "per_shot"
    pdir.mkdir(exist_ok=True)
    with open(pdir / f"{a.prefix}bb_refsched.jsonl", "w",
              encoding="utf-8") as f:
        for r0 in recs:
            f.write(json.dumps(r0) + "\n")

    ec_total = c["exact_certified"]
    print(f"  nontrivial {c['nontrivial']}  t1 {c['tier1']}  "
          f"t2 {c['tier2']}  EXACT {ec_total}  "
          f"ref {c['ref_agreed']}/{c['ref_checked']}  "
          f"3A {c['tier3A']}  3B {c['tier3B']}  "
          f"obs {obs_match}/{obs_checked}  "
          f"LP {1e3*t_lp/max(c['nontrivial'],1):.0f} ms/shot")
    assert c["float_exact_disagreements"] == 0
    assert c["ref_agreed"] == c["ref_checked"]

    out = {"schema": "oneq-qldpc-bb-refsched-gate/1",
           "circuit": ("[[144,12,12]] gross code, PUBLISHED depth-8 "
                       "reference syndrome cycle (Bravyi et al. Table 5 "
                       "/ Eq. 9; term assignment in written order), "
                       f"memory-Z, r={a.rounds}, uniform circuit-level "
                       f"p={a.p}"),
           "binding": {
               "stim_version": stim.__version__,
               "circuit_sha256": hashlib.sha256(
                   str(circ).encode()).hexdigest(),
               "dem_sha256": hashlib.sha256(str(dem).encode()).hexdigest(),
               "decompose_errors": False,
               "flatten_loops": "dem.flattened() in dem_to_matrices",
               "approximate_disjoint_errors": "not passed (default refuse)",
               "mechanism_order_sha256": hashlib.sha256(json.dumps(
                   [sorted(int(i) for i in cc)
                    for cc in checks]).encode()).hexdigest(),
               "weights_sha256": hashlib.sha256(wblob).hexdigest(),
               "weight_rule": "c_i = floor(1000*log((1-p_i)/p_i) + 1/2)"},
           "detectors": len(checks), "mechanisms": n_mech,
           "detector_degree_median": degs[len(degs) // 2],
           "detector_degree_max": degs[-1],
           "shots": a.shots, "seed": a.seed, **c,
           "observable_agreement": [obs_match, obs_checked],
           "lp_ms_per_shot": 1e3 * t_lp / max(c["nontrivial"], 1),
           "noiseless_validation": "32 shots, all detectors zero"}
    wdir = ROOT / "evidence" / "weights"
    wdir.mkdir(exist_ok=True)
    (wdir / f"{a.prefix}bb_refsched_weights.json").write_text(json.dumps(
        {"spec": {"scale": 1000,
                  "rule": "floor(1000*llr + 1/2)",
                  "weights_sha256": hashlib.sha256(wblob).hexdigest()},
         "vector": [wint[k] for k in sorted(wint)]}), encoding="utf-8")
    dest = ROOT / "evidence" / f"{a.prefix}qldpc_bb_refsched_gate.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
