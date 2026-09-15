r"""EXACTIFY EVERY NON-CIRCUIT HEADLINE RUNG -- the audit's preferred path.

External audit round 2, blocker 1: exact certification covered only the
uniform code-capacity reruns; heterogeneous and phenomenological rows
were floating-point evidence wearing the same word. This gate closes
that for every rung that does not need stim:

  hetero_p02   heterogeneous code capacity, per-qubit rates 0.5-2x 0.02,
               FIXED-POINT integer weights c_i = round(1000*LLR_i)
  phenom_p01   r=4 phenomenological, fixed-point weights (two values)
  phenom_p02   same at p=pm=0.02
  stress_p05   uniform weights, WITH tier 3A/3B solver-status split
  xsector_p02  X-sector (H_X, Z errors), uniform weights

For every shot: BP-OSD -> tiered LP -> EXACT rational certification
(integer weights => unit lattice, accept iff L > U-1) -> EVERY accepted
certificate through the independently authored reference checker
(imported in-process, pinned by sha256; a FULL pass, not a sample) ->
a per-shot record (oneq-qldpc-shot/1) written to
evidence/per_shot/<rung>.jsonl so every table cell regenerates from
records, not prose. Residues go to exact_milp with return_status: tier
3A (solver-reported optimum, zero gap) vs 3B (feasible, optimality
unknown) are counted separately and 3B is never called optimal.

The two stim rungs (surface control, BB own-circuit) are exactified in
the planned reference-schedule primary rerun; until then the paper
labels them NUMERICALLY TIGHT, not exact-certified.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
from qldpc_bb_phenom import spacetime_system
from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,
                             check_qldpc_exact, exact_milp,
                             exactify_certificate, feldman_dual,
                             fixed_point_weights)

REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
REF_SHA = "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"


def _load_reference():
    got = hashlib.sha256(REF.read_bytes()).hexdigest()
    assert got == REF_SHA, f"reference checker tampered: {got}"
    spec = importlib.util.spec_from_file_location("ref_checker", REF)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = mod    # dataclasses resolves __module__
    spec.loader.exec_module(mod)
    return mod


def _dense_H(checks, n):
    H = [[0] * n for _ in range(len(checks))]
    for j, sup in enumerate(checks):
        for i in sup:
            H[j][int(i)] = 1
    return H


def _cert_doc(dense_H, syn, wint, candidate, ec, n):
    e = [0] * n
    for i in candidate:
        e[int(i)] = 1
    facets = []
    for (j, F, y) in ec.facet_duals:
        rows = sorted(int(r) for r in j) if isinstance(
            j, (tuple, list, frozenset)) else [int(j)]
        facets.append({"rows": rows, "F": sorted(int(i) for i in F),
                       "y": {"num": y.numerator, "den": y.denominator}})
    return {"H": dense_H, "s": [int(b) for b in syn],
            "weights": [{"num": int(wint[i]), "den": 1} for i in range(n)],
            "candidate": e, "facets": facets}


def _hash_checks(checks) -> str:
    return hashlib.sha256(
        json.dumps([list(c) for c in checks]).encode()).hexdigest()[:16]


def run_rung(*, name, checks, wint, wspec, sample_syndrome, decoder,
             target, ref, is_logical=None, rpc_rounds=6,
             milp_on_residue=True, bd_close=False, use_lazy=False,
             record_prefix=""):
    n = len(wint)
    wfloat = {i: float(wint[i]) for i in range(n)}
    wexact = {i: Fraction(int(wint[i])) for i in range(n)}
    dense = _dense_H(checks, n)
    h_hash = _hash_checks(checks)
    recs = []
    c = {"nontrivial": 0, "trivial_tier0": 0, "sampled": 0,
         "tier1": 0, "tier2": 0, "exact_certified": 0,
         "float_only_disagreements": 0, "ref_checked": 0, "ref_agreed": 0,
         "tier3A": 0, "tier3B": 0, "residue_unsolved": 0,
         "weights_spec": wspec}
    t_comp = {"bposd_ns": [], "lp_ns": [], "exactify_ns": [],
              "exact_check_ns": [], "ref_check_ns": [], "milp_ns": []}
    sizes = {"max_num_bits": 0, "max_den_bits": 0, "max_facets": 0}

    while c["nontrivial"] < target:
        c["sampled"] += 1
        syn, e_true = sample_syndrome()
        if not syn.any():
            c["trivial_tier0"] += 1
            continue
        c["nontrivial"] += 1
        shot_i = c["nontrivial"] - 1

        t0 = time.perf_counter_ns()
        e_bp = np.asarray(decoder.decode(syn.astype(np.uint8))) \
            .astype(np.uint8)
        t1 = time.perf_counter_ns()
        cert, bound, primal = feldman_dual(checks, wfloat, list(syn),
                                           return_primal=True,
                                           rpc_rounds=rpc_rounds)
        t2 = time.perf_counter_ns()
        t_comp["bposd_ns"].append(t1 - t0)
        t_comp["lp_ns"].append(t2 - t1)

        rec = {"schema": "oneq-qldpc-shot/1", "experiment_id": name,
               "shot_index": shot_i,
               "instance": {"H_hash": h_hash,
                            "weights_sha256": wspec["weights_sha256"],
                            "syndrome_support":
                                [int(i) for i in np.flatnonzero(syn)]},
               "bposd": {"candidate":
                             [int(i) for i in np.flatnonzero(e_bp)],
                         "objective": int(sum(wint[int(i)] for i in
                                              np.flatnonzero(e_bp))),
                         # COMPUTED, not asserted (external audit 2026-09-14,
                         # B-06): the frozen held-out run wrote True here
                         # unconditionally; its records are checked after
                         # the fact by qldpc_bposd_syndrome_validity.py.
                         "syndrome_valid": bool(all(
                             sum(int(e_bp[i]) for i in chk) % 2 == int(syn[j])
                             for j, chk in enumerate(checks)))}}
        candidate, tier, fc = None, None, None
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
            f1 = QldpcCertificate(error_support=bp_sup,
                                  facet_duals=cert.facet_duals,
                                  box_duals=cert.box_duals)
            if check_qldpc(checks=checks, weights=wfloat,
                           syndrome=list(syn), cert=f1).accepted:
                candidate, tier, fc = bp_sup, "tier1", f1
            elif primal is not None:
                f2 = QldpcCertificate(error_support=tuple(primal),
                                      facet_duals=cert.facet_duals,
                                      box_duals=cert.box_duals)
                if check_qldpc(checks=checks, weights=wfloat,
                               syndrome=list(syn), cert=f2).accepted:
                    candidate, tier, fc = tuple(primal), "tier2", f2

        if candidate is not None:
            c[tier] += 1
            t3 = time.perf_counter_ns()
            ec = exactify_certificate(fc)
            t4 = time.perf_counter_ns()
            rx = check_qldpc_exact(checks=checks, weights=wexact,
                                   syndrome=list(syn), cert=ec)
            t5 = time.perf_counter_ns()
            t_comp["exactify_ns"].append(t4 - t3)
            t_comp["exact_check_ns"].append(t5 - t4)
            for (_j, _F, y) in ec.facet_duals:
                sizes["max_num_bits"] = max(sizes["max_num_bits"],
                                            y.numerator.bit_length())
                sizes["max_den_bits"] = max(sizes["max_den_bits"],
                                            y.denominator.bit_length())
            sizes["max_facets"] = max(sizes["max_facets"],
                                      len(ec.facet_duals))
            if rx.accepted:
                c["exact_certified"] += 1
                doc = _cert_doc(dense, syn, wint, candidate, ec, n)
                t6 = time.perf_counter_ns()
                rep = ref.check_certificate(doc)
                t7 = time.perf_counter_ns()
                t_comp["ref_check_ns"].append(t7 - t6)
                c["ref_checked"] += 1
                if rep["objective_optimal"]:
                    c["ref_agreed"] += 1
            else:
                c["float_only_disagreements"] += 1
            rec["receipt"] = {
                "tier": tier,
                "certificate_hash": hashlib.sha256(
                    json.dumps([[list(j) if isinstance(j, tuple) else j,
                                 list(F), y.numerator, y.denominator]
                                for (j, F, y) in ec.facet_duals])
                    .encode()).hexdigest()[:16],
                "exact_checker_verdict":
                    "EXACT_CERTIFIED" if rx.accepted else rx.reason[:60],
                "answer": [int(i) for i in candidate],
                "answer_objective":
                    int(sum(wint[int(i)] for i in candidate))}
        else:
            t8 = time.perf_counter_ns()
            if milp_on_residue:
                sup, wt, st = exact_milp(checks, wfloat, list(syn),
                                         return_status=True)
                t9 = time.perf_counter_ns()
                t_comp["milp_ns"].append(t9 - t8)
                if sup is None:
                    c["residue_unsolved"] += 1
                    rec["fallback"] = {"status": st, "answer": None}
                else:
                    candidate = sup
                    closed_bd = False
                    if bd_close:
                        from fractions import Fraction as _F

                        from oneq.qldpc_cert import certified_bnb
                        from oneq.qldpc_check import check_qldpc_bnb
                        U = int(round(float(wt)))
                        tree, _nn = certified_bnb(
                            checks, wfloat, list(syn), U, rpc_rounds=6,
                            max_nodes=60000, use_lazy=use_lazy)
                        if tree is not None:
                            wxx = {ii: _F(int(wint[ii]))
                                   for ii in range(len(wint))}
                            rb = check_qldpc_bnb(
                                checks=checks, weights=wxx,
                                syndrome=list(syn),
                                error_support=tuple(int(ii)
                                                    for ii in sup),
                                tree=tree)
                            closed_bd = bool(rb.accepted)
                    if closed_bd:
                        c.setdefault("tierBD", 0)
                        c["tierBD"] += 1
                        c["exact_certified"] += 1
                        rec["receipt"] = {"tier": "tierBD",
                                          "answer":
                                              [int(ii) for ii in sup]}
                    else:
                        key = ("tier3A" if st["tier"].startswith("3A")
                               else "tier3B")
                        c[key] += 1
                        rec["fallback"] = {"status": st,
                                           "answer":
                                               [int(ii) for ii in sup],
                                           "answer_objective":
                                               int(sum(wint[int(ii)]
                                                       for ii in sup))}
            else:
                c["residue_unsolved"] += 1

        if is_logical is not None and candidate is not None:
            fault_bp = np.zeros(n, dtype=np.uint8)
            fault_bp[[int(i) for i in np.flatnonzero(e_bp)]] ^= 1
            fault_bp ^= e_true
            fault_pipe = np.zeros(n, dtype=np.uint8)
            fault_pipe[[int(i) for i in candidate]] = 1
            fault_pipe ^= e_true
            rec["retrospective_simulation"] = {
                "bposd_logical_error": bool(is_logical(fault_bp)),
                "pipeline_logical_error": bool(is_logical(fault_pipe))}
        recs.append(rec)

    def _agg(v):
        if not v:
            return {}
        a = np.array(v, dtype=np.float64) / 1e6
        return {"median_ms": round(float(np.median(a)), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "max_ms": round(float(a.max()), 3)}
    c["timing"] = {k: _agg(v) for k, v in t_comp.items()}
    c["certificate_sizes"] = sizes

    pdir = ROOT / "evidence" / "per_shot"
    pdir.mkdir(exist_ok=True)
    with open(pdir / f"{record_prefix}{name}.jsonl", "w",
              encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    wdir = ROOT / "evidence" / "weights"
    wdir.mkdir(exist_ok=True)
    (wdir / f"{record_prefix}{name}_weights.json").write_text(json.dumps(
        {"spec": wspec, "vector": [int(wint[i]) for i in range(n)]}),
        encoding="utf-8")
    print(f"{name}: nontrivial {c['nontrivial']}, t1 {c['tier1']}, "
          f"t2 {c['tier2']}, EXACT {c['exact_certified']}, "
          f"ref {c['ref_agreed']}/{c['ref_checked']}, "
          f"3A {c['tier3A']}, 3B {c['tier3B']}, "
          f"disagree {c['float_only_disagreements']}")
    return c


def main() -> int:
    ref = _load_reference()
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix, lil_matrix

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks_z = [tuple(int(i) for i in np.flatnonzero(HZ[j]))
                for j in range(m)]
    checks_x = [tuple(int(i) for i in np.flatnonzero(HX[j]))
                for j in range(m)]
    xb, xp = gf2_row_reduce(HX % 2)
    zb, zp = gf2_row_reduce(HZ % 2)

    def logical_z(fault):     # for Z-sector decoding (X errors)
        if ((HZ @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(xb, xp, fault, n)

    def logical_x(fault):     # for X-sector decoding (Z errors)
        if ((HX @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(zb, zp, fault, n)

    out = {"schema": "oneq-qldpc-exact-all/1",
           "reference_checker_sha256": REF_SHA, "rungs": {}}

    wint_u = {i: 1 for i in range(n)}
    spec_u = {"scale": 1, "rounding": "uniform-unit",
              "definition": "w_i = 1 (Hamming weight)",
              "weights_sha256": hashlib.sha256(
                  json.dumps([1] * n).encode()).hexdigest()}

    # --- uniform code-capacity p=0.01 / p=0.02: the canonical rows ---
    for p, seed, tag, tgt in ((0.01, 2106, "uniform_p01", 300),
                              (0.02, 2107, "uniform_p02", 470)):
        dec_u = BpOsdDecoder(csr_matrix(HZ), error_rate=p, max_iter=30,
                             bp_method="ms", schedule="parallel",
                             osd_method="osd_cs", osd_order=5)
        rng_u = np.random.default_rng(seed)

        def sample_u(rng=rng_u, p=p):
            e = (rng.random(n) < p).astype(np.uint8)
            return (HZ @ e) % 2, e
        out["rungs"][tag] = run_rung(
            name=tag, checks=checks_z, wint=wint_u, wspec=spec_u,
            sample_syndrome=sample_u, decoder=dec_u, target=tgt, ref=ref,
            is_logical=logical_z)

    # --- hetero p=0.02, fixed-point weights ---
    rngw = np.random.default_rng(4242)
    p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
    llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
    wint_h, spec_h = fixed_point_weights(llr, scale=1000)
    dec_h = BpOsdDecoder(csr_matrix(HZ),
                         error_channel=[float(v) for v in p_i],
                         max_iter=30, bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    rng_h = np.random.default_rng(2101)

    def sample_h():
        e = (rng_h.random(n) < p_i).astype(np.uint8)
        return (HZ @ e) % 2, e
    out["rungs"]["hetero_p02"] = run_rung(
        name="hetero_p02", checks=checks_z, wint=wint_h, wspec=spec_h,
        sample_syndrome=sample_h, decoder=dec_h, target=300, ref=ref,
        is_logical=logical_z)

    # --- phenomenological p01 / p02, fixed-point ---
    for p, seed, tag in ((0.01, 2102, "phenom_p01"),
                         (0.02, 2103, "phenom_p02")):
        checks_st, w_st = spacetime_system(HZ, 4, p, p)
        nn = len(w_st)
        wint_p, spec_p = fixed_point_weights(w_st, scale=1000)
        A = lil_matrix((len(checks_st), nn))
        for j, sup in enumerate(checks_st):
            for i in sup:
                A[j, i] = 1
        ch = np.full(nn, p)
        dec_p = BpOsdDecoder(A.tocsr().astype(np.uint8),
                             error_channel=[float(v) for v in ch],
                             max_iter=30, bp_method="ms",
                             schedule="parallel", osd_method="osd_cs",
                             osd_order=5)
        rng_p = np.random.default_rng(seed)

        def sample_p(rng=rng_p, nn=nn, checks_st=checks_st, p=p):
            e = (rng.random(nn) < p).astype(np.uint8)
            syn = np.array([sum(e[i] for i in sup) % 2
                            for sup in checks_st], dtype=np.uint8)
            return syn, e
        out["rungs"][tag] = run_rung(
            name=tag, checks=checks_st, wint=wint_p, wspec=spec_p,
            sample_syndrome=sample_p, decoder=dec_p, target=150, ref=ref)

    # --- stress p=0.05 uniform, tier 3A/3B split ---
    wint_s, spec_s = wint_u, spec_u
    dec_s = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    rng_s = np.random.default_rng(2104)

    def sample_s():
        e = (rng_s.random(n) < 0.05).astype(np.uint8)
        return (HZ @ e) % 2, e
    out["rungs"]["stress_p05"] = run_rung(
        name="stress_p05", checks=checks_z, wint=wint_s, wspec=spec_s,
        sample_syndrome=sample_s, decoder=dec_s, target=300, ref=ref,
        is_logical=logical_z)

    # --- X-sector p=0.02 uniform ---
    dec_x = BpOsdDecoder(csr_matrix(HX), error_rate=0.02, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    rng_x = np.random.default_rng(2105)

    def sample_x():
        e = (rng_x.random(n) < 0.02).astype(np.uint8)
        return (HX @ e) % 2, e
    out["rungs"]["xsector_p02"] = run_rung(
        name="xsector_p02", checks=checks_x, wint=wint_s, wspec=spec_s,
        sample_syndrome=sample_x, decoder=dec_x, target=200, ref=ref,
        is_logical=logical_x)

    for tag, r in out["rungs"].items():
        assert r["float_only_disagreements"] == 0, tag
        assert r["ref_agreed"] == r["ref_checked"], \
            f"{tag}: independent checker disagreed"
        assert r["residue_unsolved"] == 0, tag

    dest = ROOT / "evidence" / "qldpc_exact_all.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # per-shot MANIFEST: bind every record file, weight vector, and the
    # checker hashes -- so "the records exist" is itself checkable
    pdir = ROOT / "evidence" / "per_shot"
    man = {"schema": "oneq-per-shot-manifest/1",
           "reference_checker_sha256": REF_SHA,
           "own_exact_checker": "src/oneq/qldpc_check.py:check_qldpc_exact",
           "rungs": {}}
    for tag, run in out["rungs"].items():
        f = pdir / f"{tag}.jsonl"
        wf = ROOT / "evidence" / "weights" / f"{tag}_weights.json"
        man["rungs"][tag] = {
            "records_file": f"evidence/per_shot/{tag}.jsonl",
            "records_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "record_count": run["nontrivial"],
            "weights_file": f"evidence/weights/{tag}_weights.json",
            "weights_sha256": run["weights_spec"]["weights_sha256"],
            "exact_certified": run["exact_certified"],
            "ref_checked": run["ref_checked"],
            "tier3A": run["tier3A"], "tier3B": run["tier3B"]}
    (pdir / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"evidence -> {dest} + per_shot/MANIFEST.json + weights/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
