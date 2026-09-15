r"""The strong-baseline experiment, rebuilt after the 2026-09-14 external audit.

WHAT THE FIRST VERSION GOT WRONG (qldpc_strong_baseline_gate.py, kept for the
record; findings B-01/B-02 in docs/EXTERNAL_AUDIT_2026-09-14.md):

  1. Its "pipeline" answer was computed ONCE per syndrome and reused for every
     BP-OSD configuration, so the pipeline's error count was constant BY
     CONSTRUCTION -- and paper 2 then presented that constancy as a theorem
     ("because its answer is the certified minimum-weight correction"). It is
     not one: tied optima can differ by a logical operator, and the real
     pipeline KEEPS BP-OSD's answer whenever it is certified (tier 1), so a
     different BP-OSD can select a different optimum with a different frame.
  2. When the certified LP path produced no answer, the reference was the MILP
     solver's support with NO receipt check -- and a MILP returning nothing
     silently became the EMPTY correction, which was still scored.

WHAT THIS VERSION MEASURES, on the SAME seeds and syndromes as the first:

  ARM "pipeline"  -- the actual receipt pipeline, run separately under EACH
      BP-OSD configuration: tier 1 keeps BP-OSD's answer if an exact receipt
      proves it optimal; tier 2 substitutes the LP primal if an exact receipt
      proves THAT optimal; otherwise a solver candidate is kept only if a
      branch-dual tree is produced AND accepted by the exact tree checker.
      No receipt -> UNRESOLVED: reported, never counted as certified.
  ARM "reference" -- a fixed reference answer per syndrome that does not
      depend on BP-OSD: the tier-2 LP primal if exactly certified, else a
      branch-dual-certified solver answer, else UNRESOLVED.

Every flat receipt is exact-rational (check_qldpc_exact after
exactify_certificate); every tree is checked by check_qldpc_bnb. Per shot the
records keep: BP-OSD support and a COMPUTED syndrome-validity bit, answers,
tiers, receipt verdicts, whether pipeline and reference are tied optima with
different supports, and each answer's retrospective logical outcome against
the sampled fault (simulation truth, not a certificate).

    python experiments/m0_prior_art/qldpc_strong_baseline_v2.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce  # noqa: E402
from qldpc_strong_baseline_gate import CONFIGS, NOISE, SHOTS  # noqa: E402

from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,  # noqa: E402
                             exact_milp, exactify_certificate, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb, check_qldpc_exact  # noqa: E402

OUT = ROOT / "evidence" / "qldpc_strong_baseline_v2.json"
RECORDS = ROOT / "evidence" / "strong_baseline_v2"
BNB_MAX_NODES = 60000


def exact_flat(checks, wf, wx, syn, support, cert):
    """Exact receipt for `support` from a float LP dual, or None."""
    fc = QldpcCertificate(error_support=tuple(support),
                          facet_duals=cert.facet_duals, box_duals=cert.box_duals)
    if not check_qldpc(checks=checks, weights=wf, syndrome=list(syn), cert=fc).accepted:
        return None
    ec = exactify_certificate(fc)
    r = check_qldpc_exact(checks=checks, weights=wx, syndrome=list(syn), cert=ec)
    return ec if r.accepted else None


def tree_receipt(checks, wf, wx, syn):
    """Solver candidate + branch-dual tree, kept ONLY if the tree checker accepts."""
    sup, wt, st = exact_milp(checks, wf, list(syn), return_status=True)
    if sup is None:
        return None, {"solver_status": st, "reason": "solver returned no candidate"}
    cand = tuple(int(i) for i in sup)
    U = int(round(float(wt)))
    tree, nodes = certified_bnb(checks, wf, list(syn), U, rpc_rounds=6,
                                max_nodes=BNB_MAX_NODES)
    if tree is None:
        return None, {"solver_status": st, "reason": f"no tree within {BNB_MAX_NODES} nodes"}
    rb = check_qldpc_bnb(checks=checks, weights=wx, syndrome=list(syn),
                         error_support=cand, tree=tree)
    if not rb.accepted:
        return None, {"solver_status": st, "reason": f"tree refused: {rb.reason[:80]}"}
    return cand, {"tree_nodes": nodes}


def main() -> int:
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}
    xb, xp = gf2_row_reduce(HX % 2)

    def is_logical(fault):
        if ((HZ @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(xb, xp, fault, n)

    def vec(support):
        v = np.zeros(n, dtype=np.uint8)
        v[list(support)] = 1
        return v

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix

    RECORDS.mkdir(exist_ok=True)
    out = {"schema": "oneq-qldpc-strong-baseline/2",
           "supersedes": "evidence/qldpc_strong_baseline.json (reference reused across "
                         "configurations; uncertified solver fallback)",
           "protocol": f"[[144,12,12]] Z-sector, uniform weights, {SHOTS} nontrivial "
                       f"syndromes per noise level from the SAME seeds as v1, shared by "
                       f"every configuration",
           "configs": CONFIGS, "levels": {}}

    for p in NOISE:
        rng = np.random.default_rng(int(10000 * p) + 31337)      # v1's seed rule
        shots = []
        while len(shots) < SHOTS:
            e = (rng.random(n) < p).astype(np.uint8)
            syn = (HZ @ e) % 2
            if syn.any():
                shots.append((syn, e))

        t0 = time.perf_counter()
        lp = [feldman_dual(checks, wf, list(s), return_primal=True, rpc_rounds=6)
              for s, _ in shots]
        # reference arm: BP-independent
        ref = []
        for (syn, e_true), (cert, _b, primal) in zip(shots, lp):
            r = {"answer": None, "tier": "unresolved", "detail": {}}
            if cert is not None and primal is not None and exact_flat(
                    checks, wf, wx, syn, primal, cert) is not None:
                r.update(answer=tuple(int(i) for i in primal), tier="tier2_lp_primal")
            else:
                cand, det = tree_receipt(checks, wf, wx, syn)
                r["detail"] = det
                if cand is not None:
                    r.update(answer=cand, tier="tierBD")
            if r["answer"] is not None:
                r["logical_error"] = bool(is_logical((vec(r["answer"]) ^ e_true) % 2))
            ref.append(r)
        level = {"reference": {
            "resolved": sum(r["answer"] is not None for r in ref),
            "unresolved": sum(r["answer"] is None for r in ref),
            "logical_errors_on_resolved": sum(bool(r.get("logical_error")) for r in ref),
            "by_tier": {t: sum(r["tier"] == t for r in ref)
                        for t in ("tier2_lp_primal", "tierBD", "unresolved")}},
            "configs": {}}

        for label, cfg in CONFIGS.items():
            dec = BpOsdDecoder(csr_matrix(HZ), error_rate=p, **cfg)
            recs = []
            for k, ((syn, e_true), (cert, _b, primal)) in enumerate(zip(shots, lp)):
                e_bp = np.asarray(dec.decode(syn.astype(np.uint8))).astype(np.uint8)
                bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
                bp_valid = bool(np.array_equal((HZ @ e_bp) % 2, syn))
                rec = {"shot": k, "bposd": {"support": list(bp_sup),
                                            "syndrome_valid": bp_valid,
                                            "weight": len(bp_sup),
                                            "logical_error": bool(is_logical((e_bp ^ e_true) % 2))}}
                answer, tier, detail = None, "unresolved", {}
                if cert is not None and bp_valid and exact_flat(
                        checks, wf, wx, syn, bp_sup, cert) is not None:
                    answer, tier = bp_sup, "tier1"
                elif ref[k]["tier"] == "tier2_lp_primal":
                    answer, tier = ref[k]["answer"], "tier2"
                elif ref[k]["tier"] == "tierBD":
                    answer, tier = ref[k]["answer"], "tierBD"
                else:
                    detail = ref[k]["detail"]
                rec["pipeline"] = {"tier": tier, "support": None if answer is None else list(answer),
                                   "detail": detail}
                if answer is not None:
                    rec["pipeline"]["logical_error"] = bool(is_logical((vec(answer) ^ e_true) % 2))
                    ra = ref[k]["answer"]
                    if ra is not None:
                        rec["pipeline"]["same_support_as_reference"] = (tuple(answer) == tuple(ra))
                        rec["pipeline"]["frame_differs_from_reference"] = bool(
                            is_logical((vec(answer) ^ vec(ra)) % 2))
                recs.append(rec)

            resolved = [r for r in recs if r["pipeline"]["support"] is not None]
            pipe_err = sum(r["pipeline"]["logical_error"] for r in resolved)
            bp_err = sum(r["bposd"]["logical_error"] for r in recs)
            n10 = sum(r["bposd"]["logical_error"] and not r["pipeline"]["logical_error"] for r in resolved)
            n01 = sum((not r["bposd"]["logical_error"]) and r["pipeline"]["logical_error"] for r in resolved)
            nd, kmin = n10 + n01, min(n10, n01)
            pmc = (min(1.0, 2 * sum(math.comb(nd, i) for i in range(kmin + 1)) / 2 ** nd)
                   if nd else 1.0)
            level["configs"][label] = {
                "bp_osd_logical_errors": bp_err,
                "bp_osd_syndrome_invalid": sum(not r["bposd"]["syndrome_valid"] for r in recs),
                "tier1": sum(r["pipeline"]["tier"] == "tier1" for r in recs),
                "tier1_rate": round(100 * sum(r["pipeline"]["tier"] == "tier1" for r in recs) / len(recs), 1),
                "tier2": sum(r["pipeline"]["tier"] == "tier2" for r in recs),
                "tierBD": sum(r["pipeline"]["tier"] == "tierBD" for r in recs),
                "unresolved": sum(r["pipeline"]["tier"] == "unresolved" for r in recs),
                "pipeline_logical_errors_on_resolved": pipe_err,
                "pipeline_vs_reference_different_optimum": sum(
                    r["pipeline"].get("same_support_as_reference") is False for r in resolved),
                "pipeline_vs_reference_frame_differs": sum(
                    bool(r["pipeline"].get("frame_differs_from_reference")) for r in resolved),
                "paired_on_resolved": {"bp_wrong_pipe_ok": n10, "bp_ok_pipe_wrong": n01},
                "mcnemar_two_sided_p": pmc}
            with open(RECORDS / f"p{p}_{label}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            c = level["configs"][label]
            print(f"  p={p} {label:26s} BP err {bp_err:3d} | pipeline err {pipe_err:3d} "
                  f"(t1 {c['tier1']} t2 {c['tier2']} BD {c['tierBD']} unres {c['unresolved']}) "
                  f"| diff-optimum {c['pipeline_vs_reference_different_optimum']} "
                  f"frame-differs {c['pipeline_vs_reference_frame_differs']}", flush=True)
        level["seconds"] = round(time.perf_counter() - t0, 1)
        out["levels"][f"p={p}"] = level
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"evidence -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
