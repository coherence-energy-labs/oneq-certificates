r"""Every quantitative macro and table in paper 2, generated from artifacts.

Same contract as make_numbers.py: the manuscript never types a measured
number, and tests/test_paper_build_tools.py compares the committed output
with a fresh generation.

    python paper/tools/make_numbers_qldpc.py

Writes paper/qldpc-certificates/generated/{numbers,tab_*}.tex.

PERCENTAGES ARE FLOORED at the last shown digit, never rounded up -- the
convention paper 2 states (Sec. on statistics) and its rates depend on.
"""

from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
OUT = ROOT / "paper" / "qldpc-certificates" / "generated"

RUNG_ORDER = ["uniform_p01", "uniform_p02", "hetero_p02", "phenom_p01",
              "phenom_p02", "xsector_p02", "stress_p05", "bb_refsched"]
RUNG_LABEL = {
    "uniform_p01": r"uniform, $p=0.01$",
    "uniform_p02": r"uniform, $p=0.02$",
    "hetero_p02": r"heterogeneous, $p=0.02$",
    "phenom_p01": r"phenomenological, $p=p_m=0.01$",
    "phenom_p02": r"phenomenological, $p=p_m=0.02$",
    "xsector_p02": r"X-sector, $p=0.02$",
    "stress_p05": r"stress, $p=0.05$",
    "bb_refsched": r"circuit, reference schedule",
}


def J(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def I(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def PF(x: float, k: int = 2) -> str:
    """Percentage, floored at k decimals."""
    v = math.floor(100 * x * 10 ** k + 1e-9) / 10 ** k
    return f"{v:.{k}f}"


def SCI(x: float, k: int = 1) -> str:
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, k) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.{k}f}\times 10^{{{e}}}}}"


def cp_lower(n_success: int, n: int) -> float:
    """One-sided 95% exact Clopper-Pearson lower bound; all-success rows only."""
    if n_success != n:
        raise ValueError("only all-success rows are tabulated")
    return 0.05 ** (1 / n)


def build() -> dict[str, str]:
    files: dict[str, str] = {}
    M: dict[str, str] = {}

    # ------------------------------------------------ held-out campaign
    ho = J("qldpc_heldout.json")
    rungs = ho["rungs"]
    if list(rungs) != RUNG_ORDER:
        raise SystemExit(f"held-out rung order changed: {list(rungs)}")
    rs = J("heldout_qldpc_bb_refsched_gate.json")
    tot_n = tot_cert = tot_flat = tot_bd = tot_ref = 0
    lines = [r"\begin{tabular}{@{}lrrrrrrrl@{}}", r"\toprule",
             r"dataset & sampled & tier 0 & $N$ & tier 1 & tier 2 & BD & certified & 95\% lower \\",
             r"\midrule"]
    for k in RUNG_ORDER:
        r = rungs[k]
        bd = r["exact_certified"] - r["tier1"] - r["tier2"]
        if k == "bb_refsched" and bd != rs["tierBD"]:
            raise SystemExit("reference-schedule tier BD disagrees between artifacts")
        if r["exact_certified"] != r["nontrivial"]:
            raise SystemExit(f"{k}: not every nontrivial shot certified; the text says all")
        tot_n += r["nontrivial"]
        tot_cert += r["exact_certified"]
        tot_flat += r["tier1"] + r["tier2"]
        tot_bd += bd
        tot_ref += r["ref_agreed"]
        lines.append(
            rf"{RUNG_LABEL[k]} & {I(r.get('sampled', r.get('shots')))} & {I(r['trivial_tier0'])} & {I(r['nontrivial'])} & "
            rf"{I(r['tier1'])} & {I(r['tier2'])} & {I(bd)} & {I(r['exact_certified'])} & "
            rf"$\ge {PF(cp_lower(r['exact_certified'], r['nontrivial']))}\%$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_heldout.tex"] = "\n".join(lines) + "\n"
    tier1 = sum(rungs[k]["tier1"] for k in RUNG_ORDER)
    tier2 = sum(rungs[k]["tier2"] for k in RUNG_ORDER)
    if tier1 + tier2 != tot_flat:
        raise SystemExit("tier split does not sum to the flat count")
    M["HoTierOne"] = I(tier1)
    M["HoTierTwo"] = I(tier2)
    M["HoTierOnePct"] = PF(tier1 / tot_n)
    M["HoTierTwoPct"] = PF(tier2 / tot_n)
    from scipy.stats import beta as _beta
    M["HoFlatLowerBound"] = PF(float(_beta.ppf(0.05, tot_flat, tot_n - tot_flat + 1)))
    man = J("receipt_manifest.json")
    M["ReplayReceipts"] = I(len(man["receipts"]))
    M["ReplayTrees"] = I(sum(v for k, v in man["expected_counts"].items() if "tree" in k))
    M["ReplayDevTrees"] = I(man["expected_counts"]["dev_tree"] + man["expected_counts"]["dev_refsched_tree"])
    M["HoNontrivial"] = I(tot_n)
    M["HoCertified"] = I(tot_cert)
    M["HoFlat"] = I(tot_flat)
    M["HoTrees"] = I(tot_bd)
    M["HoFlatPct"] = PF(tot_flat / tot_n)
    M["HoRungs"] = I(len(RUNG_ORDER))
    hf = J("qldpc_heldout_flat_certs.json")["totals"]
    if hf["attempted"] != tot_flat or hf["ref_agreed"] != tot_flat:
        raise SystemExit("shipped flat certificates disagree with the campaign's flat count")
    M["HoFlatRefAgreed"] = I(hf["ref_agreed"])
    M["HoFlatByteIdentical"] = I(hf["hash_matched"])
    M["HoFlatDifferentValid"] = I(hf["hash_differs_but_valid"])
    M["HoFlatNoHash"] = I(hf["hash_unavailable"])
    M["HoFlatDivergentPct"] = PF(hf["hash_differs_but_valid"]
                                 / (hf["hash_matched"] + hf["hash_differs_but_valid"]), 1)
    M["HoRefschedNontrivial"] = I(rs["nontrivial"])
    M["HoRefschedTierOne"] = I(rs["tier1"])
    M["HoRefschedTierTwo"] = I(rs["tier2"])
    M["HoRefschedTrees"] = I(rs["tierBD"])
    M["HoRefschedFlatPct"] = PF((rs["tier1"] + rs["tier2"]) / rs["nontrivial"])
    agreed, of = rs["observable_agreement"]            # (agreed, total)
    if of != rs["nontrivial"]:
        raise SystemExit("observable agreement is not over the nontrivial shots")
    M["HoRefschedObsAgree"] = I(agreed)
    # zero observed disagreements in N: one-sided 95% UPPER bound, ceilinged
    if agreed == of:
        M["HoRefschedObsUpperPct"] = f"{math.ceil(100 * (1 - 0.05 ** (1 / of)) * 100) / 100:.2f}"

    add = J("qldpc_heldout_addendum.json")
    t = add["stress_paired_2x2"]
    M["HoStressBothOk"] = I(t["both_correct"])
    M["HoStressBpWrongPipeOk"] = I(t["bp_wrong_pipeline_correct"])
    M["HoStressBpOkPipeWrong"] = I(t["bp_correct_pipeline_wrong"])
    M["HoStressBothWrong"] = I(t["both_wrong"])
    M["HoStressBpWrongTotal"] = I(t["bp_wrong_pipeline_correct"] + t["both_wrong"])
    M["HoStressPipeWrongTotal"] = I(t["bp_correct_pipeline_wrong"] + t["both_wrong"])
    expect_p = 2 * 0.5 ** (t["bp_wrong_pipeline_correct"] + t["bp_correct_pipeline_wrong"])
    if t["bp_correct_pipeline_wrong"] == 0 and not math.isclose(t["mcnemar_exact_two_sided_p"], expect_p, rel_tol=1e-9):
        raise SystemExit("McNemar value disagrees with 2*(1/2)^discordant")
    M["HoStressMcNemar"] = SCI(t["mcnemar_exact_two_sided_p"])
    # tree statistics over EVERY stored tree (17 held-out + 23 development), with
    # the definitions qldpc_heldout_addendum.py used, checked against its 36-tree record
    def _nodes(t):
        return 1 + (_nodes(t[2]) + _nodes(t[3]) if t[0] == "branch" else 0)

    def _depth(t):
        return 1 + (max(_depth(t[2]), _depth(t[3])) if t[0] == "branch" else 0)

    def _stats(trees):
        c = sorted(_nodes(t["tree"]) for t in trees)
        return {"count": len(c), "nodes_median": c[len(c) // 2],
                "nodes_p95": c[max(0, int(len(c) * 0.95) - 1)], "nodes_max": c[-1],
                "depth_max": max(_depth(t["tree"]) for t in trees),
                "serialized_bytes_max": max(len(json.dumps(t["tree"])) for t in trees)}

    ts_old = {k: v for k, v in add["tree_stats"].items() if k != "note"}
    if _stats(add["bd_trees"] + add["dev_bd_trees"]) != ts_old:
        raise SystemExit("tree statistics no longer reproduce the addendum's 36-tree record")
    ts = _stats(add["bd_trees"] + add["dev_bd_trees"] + J("qldpc_dev_refsched_trees.json")["trees"])
    M["TreeCount"] = I(ts["count"])
    M["TreeNodesMedian"] = I(ts["nodes_median"])
    M["TreeNodesPNinetyFive"] = I(ts["nodes_p95"])
    M["TreeNodesMax"] = I(ts["nodes_max"])
    M["TreeDepthMax"] = I(ts["depth_max"])
    M["TreeBytesMax"] = I(ts["serialized_bytes_max"])
    M["TreeXvalAgreed"] = I(add["tree_xval_checker_b"]["agreed"])
    M["TreeXvalAttempted"] = I(add["tree_xval_checker_b"]["attempted"])
    M["TreeMutantsRefused"] = I(sum(1 for v in add["tree_checker_b_mutants"].values() if v == "refused"))
    M["TreeMutants"] = I(len(add["tree_checker_b_mutants"]))

    # certificate sizes, held-out and canonical
    sizes = [rungs[k]["certificate_sizes"] for k in RUNG_ORDER if "certificate_sizes" in rungs[k]]
    M["HoMaxFacets"] = I(max(s["max_facets"] for s in sizes))
    M["HoMaxNumBits"] = I(max(s["max_num_bits"] for s in sizes))
    M["HoMaxDenBits"] = I(max(s["max_den_bits"] for s in sizes))

    # ------------------------------------------------ development (canonical) datasets
    ex = J("qldpc_exact_all.json")
    dev_cert = sum(r["exact_certified"] for r in ex["rungs"].values())
    dev_ref = sum(r.get("ref_agreed", 0) for r in ex["rungs"].values())
    M["DevExactFlat"] = I(dev_cert)
    M["DevRefAgreed"] = I(dev_ref)
    br = J("qldpc_bnb_residue.json")
    M["DevResidues"] = I(int(br["total_residues"]))
    M["DevResiduesClosed"] = I(int(br["total_closed_exact"]))
    dsz = [r["certificate_sizes"] for r in ex["rungs"].values() if "certificate_sizes" in r]
    if dsz:
        M["DevMaxFacets"] = I(max(s["max_facets"] for s in dsz))
        M["DevMaxNumBits"] = I(max(s["max_num_bits"] for s in dsz))

    # ------------------------------------------------ strong baseline (v2: receipts on every answer)
    sbd = J("qldpc_strong_baseline_v2.json")
    sb = sbd["levels"]
    order = ("baseline_30_osdcs5", "strong_100_osdcs10", "strongest_ps_100_osdcs10")
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & fixed & receipt pipeline & BP-OSD & \multicolumn{2}{c}{strongest: pipeline vs reference} \\",
             r"$p$ & reference & A / B / C & A / B / C & other optimum & other frame \\",
             r"\midrule"]
    for lvl in ("p=0.01", "p=0.02", "p=0.03", "p=0.05", "p=0.07"):
        v = sb[lvl]
        if v["reference"]["unresolved"] or any(v["configs"][c]["unresolved"] for c in order):
            raise SystemExit(f"{lvl}: an answer without a receipt; the table would mix evidence classes")
        pe = " / ".join(str(v["configs"][c]["pipeline_logical_errors_on_resolved"]) for c in order)
        be = " / ".join(str(v["configs"][c]["bp_osd_logical_errors"]) for c in order)
        s = v["configs"][order[2]]
        lines.append(rf"{lvl[2:]} & {v['reference']['logical_errors_on_resolved']} & {pe} & {be} & "
                     rf"{s['pipeline_vs_reference_different_optimum']} & {s['pipeline_vs_reference_frame_differs']} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_strong.tex"] = "\n".join(lines) + "\n"
    f5, f7 = sb["p=0.05"], sb["p=0.07"]
    strongest = order[2]
    c0 = sb["p=0.01"]["configs"][order[0]]
    M["StrongShots"] = I(c0["tier1"] + c0["tier2"] + c0["tierBD"] + c0["unresolved"])
    M["StrongFiveStrongest"] = I(f5["configs"][strongest]["bp_osd_logical_errors"])
    M["StrongFiveReference"] = I(f5["reference"]["logical_errors_on_resolved"])
    M["StrongFivePipelineStrongest"] = I(f5["configs"][strongest]["pipeline_logical_errors_on_resolved"])
    M["StrongSevenStrongest"] = I(f7["configs"][strongest]["bp_osd_logical_errors"])
    M["StrongSevenReference"] = I(f7["reference"]["logical_errors_on_resolved"])
    M["StrongSevenPipelineStrongest"] = I(f7["configs"][strongest]["pipeline_logical_errors_on_resolved"])
    M["StrongSevenOtherOptimum"] = I(f7["configs"][strongest]["pipeline_vs_reference_different_optimum"])
    M["StrongSevenOtherFrame"] = I(f7["configs"][strongest]["pipeline_vs_reference_frame_differs"])
    M["StrongFiveOtherOptimum"] = I(f5["configs"][strongest]["pipeline_vs_reference_different_optimum"])
    M["StrongFiveOtherFrame"] = I(f5["configs"][strongest]["pipeline_vs_reference_frame_differs"])
    M["StrongFiveTierOneStated"] = f"{f5['configs'][order[0]]['tier1_rate']:.1f}"
    M["StrongFiveTierOneStrongest"] = f"{f5['configs'][strongest]['tier1_rate']:.1f}"
    M["StrongTreesSeven"] = I(f7["reference"]["by_tier"]["tierBD"])
    M["StrongTreesFive"] = I(f5["reference"]["by_tier"]["tierBD"])

    # ------------------------------------------------ mechanism validations
    cb = J("qldpc_claims_backing.json")
    sep = cb["separation_bruteforce"]
    M["SepInstances"] = I(sep["instances"])
    M["SepWorstGap"] = SCI(sep["worst_gap_vs_bruteforce"])
    M["SepSuboptimal"] = I(sep["suboptimal_separations"])
    M["SepRefInstances"] = I(cb["separation_reference_agreement"]["instances"])
    M["SepRefMismatches"] = I(cb["separation_reference_agreement"]["mismatches_vs_pinned_reference"])
    ds = cb["discovery_shot"]
    M["DiscBpWeight"] = I(ds["bp_osd_weight"])
    M["DiscBound"] = I(ds["lp_bound"])
    M["DiscFaultWeight"] = I(ds["fault_weight"])
    en = J("qldpc_ensembles.json")
    M["EnsLazyCodes"] = I(en["lazy_vs_enum"]["codes"])
    M["EnsLazyWorst"] = SCI(en["lazy_vs_enum"]["worst_abs_diff"])
    M["EnsLadderCodes"] = I(en["ladder_vs_bruteforce"]["codes"])
    M["EnsBaseTight"] = I(en["ladder_vs_bruteforce"]["base_lp_tight"])
    M["EnsCutsTight"] = I(en["ladder_vs_bruteforce"]["with_cuts_tight"])
    M["EnsImpossible"] = I(en["ladder_vs_bruteforce"]["impossible_bucket"])
    M["EnsCappedTight"] = f"{en['capped_facets']['tight_capped_F3']}/{en['capped_facets']['codes']}"
    rp = cb["rpc_depth"]
    M["RpcResidueSix"] = I(rp["rounds_6"]["fractional_residue"])
    M["RpcResidueTwentyFive"] = I(rp["rounds_25"]["fractional_residue"])
    M["RpcShots"] = I(rp["rounds_6"]["nontrivial"])

    fx = J("qldpc_refsched_fault_xval.json")
    M["XvalFaults"] = I(fx["part2_fault_signatures"]["checked"])
    M["XvalMismatches"] = I(fx["part2_fault_signatures"]["mismatches"])
    M["XvalOps"] = I(fx["part1_transcription"]["ops"])
    M["XvalRecords"] = I(fx["part1_transcription"]["records"])

    cd = J("qldpc_circuit_distance.json")
    M["DistUpper"] = I(cd["upper_bound"]["value"])
    M["DistFlatBound"] = f"{cd['flat_lower_bound']['lp_bound']:.1f}"
    M["DistNodeBudget"] = I(cd["lower_bound_attempt"]["max_nodes_budget"])

    lazy, full = J("qldpc_circuit_gate_lazy.json"), J("qldpc_circuit_gate.json")
    M["TieredMs"] = f"{lazy['cert_ms_per_shot']:.2f}"
    M["EnumMs"] = f"{full['cert_ms_per_shot']:.2f}"
    M["TieredSpeedup"] = f"{full['cert_ms_per_shot'] / lazy['cert_ms_per_shot']:.1f}"
    M["SurfaceNontrivial"] = I(lazy["nontrivial"])
    M["SurfaceDegreeMax"] = I(lazy["detector_degree_max"])

    sq = J("qldpc_bb_circuit_gate.json")
    M["SeqCertified"] = I(sq["certified_total"])
    M["SeqNontrivial"] = I(sq["nontrivial"])
    M["SeqPct"] = PF(sq["certified_rate"])

    fr = J("qldpc_frame_demo.json")
    M["FrameShots"] = I(len(fr["shots"]))
    M["FrameAllCertified"] = I(sum(1 for s in fr["shots"] if s["status"] == "FRAME-CERTIFIED"))
    secs = [s["seconds"] for s in fr["shots"]]
    # a RANGE must enclose the data: floor the low end, ceiling the high end
    M["FrameSecLo"] = f"{math.floor(min(secs) * 10) / 10:.1f}"
    M["FrameSecHi"] = f"{math.ceil(max(secs) * 10) / 10:.1f}"
    import re
    mv = re.fullmatch(r"(\d+) random instances vs brute force: (\d+) accepted proofs all truly "
                      r"determinate, (\d+) honest refusals on ambiguous, (\d+) false proofs",
                      fr["validation"])
    if not mv:
        raise SystemExit("frame validation string changed shape; read its numbers again")
    M["FrameValInstances"], M["FrameValProofs"], M["FrameValRefusals"], M["FrameValFalse"] = (
        I(int(x)) for x in mv.groups())

    sr = J("qldpc_scale_robustness.json")
    scales = sr["scales"]
    M["ScaleCertified"] = " / ".join(I(scales[s]["certified"]) for s in ("1000", "10000", "100000"))
    M["ScaleSupportChanges"] = I(sum(v["support_disagreements_vs_S1000"] for v in scales.values()))
    M["ScaleTierChanges"] = I(sum(v["tier_disagreements_vs_S1000"] for v in scales.values()))

    ps = J("qldpc_paired_stress.json")["paired_2x2"]
    M["PairedRunBpWrongPipeOk"] = I(ps["bp_wrong_pipeline_correct"])
    M["PairedRunBpOkPipeWrong"] = I(ps["bp_correct_pipeline_wrong"])

    tim = {k: rungs[k]["timing"] for k in RUNG_ORDER if rungs[k].get("timing")}
    ex_med = [t["exact_check_ns"]["median_ms"] for t in tim.values()]
    lp_med = [t["lp_ns"]["median_ms"] for t in tim.values()]
    def rng(vals, k):
        f = 10 ** k
        return f"{math.floor(min(vals) * f) / f:.{k}f}", f"{math.ceil(max(vals) * f) / f:.{k}f}"
    M["TimeExactCheckMedLo"], M["TimeExactCheckMedHi"] = rng(ex_med, 2)
    M["TimeLpMedLo"], M["TimeLpMedHi"] = rng(lp_med, 1)

    # ------------------------------------------------ a checker in a second language
    js = J("qldpc_js_agreement.json")
    js_src = ROOT / js["checker"]
    if __import__("hashlib").sha256(js_src.read_bytes()).hexdigest() != js["checker_sha256"]:
        raise SystemExit("JavaScript agreement: the checker changed since the artifact was written; re-run it")
    if (js["disagreements"] or js["flat_receipts_accepted"] != js["flat_receipts"]
            or js["planted_defects_refused_for_their_reason"] != js["planted_defects"]):
        raise SystemExit("JavaScript agreement: the text says every receipt accepted and every defect refused")
    M["JsFlatAccepted"] = I(js["flat_receipts_accepted"])
    M["JsDefects"] = I(js["planted_defects"])
    M["JsDefectClasses"] = I(len([k for k in js["planted_by_class"] if k != "lattice_boundary_accept"]))

    # ------------------------------------------------ operator mutation score of the checker's tests
    mu = J("mutation_score/qldpc_check.json")

    def sha(p):   # the pins' domain: CRLF normalised to LF, the bytes git stores (tools/mutation_score.py)
        return __import__("hashlib").sha256(p.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()

    if mu.get("digest_domain") != "sha256-lf":
        raise SystemExit("mutation score: its pins are not in the sha256-lf domain; re-record or re-run it")
    if sha(ROOT / mu["target"]) != mu["target_sha256"] or any(
            sha(ROOT / t) != h for t, h in mu["tests"].items()):
        raise SystemExit("mutation score: the checker or its tests changed since the score was measured; re-run it")
    if mu["timeout"] or any(not x["equivalent"] for x in mu["survivors"]):
        raise SystemExit("mutation score: a non-equivalent mutant survives; the text says none does")
    M["MutQRun"] = I(mu["mutants_run"])
    M["MutQKilled"] = I(mu["killed"])
    M["MutQEquivalent"] = I(mu["equivalent_declared"])

    body = ["% GENERATED by paper/tools/make_numbers_qldpc.py from evidence/ -- do not edit.",
            "% Percentages are floored at the last shown digit."]
    for k in sorted(M):
        body.append(rf"\newcommand{{\{k}}}{{{M[k]}}}")
    files["numbers.tex"] = "\n".join(body) + "\n"
    return files


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in build().items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print("wrote", (OUT / name).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
