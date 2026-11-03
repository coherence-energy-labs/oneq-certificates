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
    # BP-OSD's OWN outcome on the same shots (audit F-18), derived from the records by
    # tools/bposd_circuit_outcome.py; it must describe these records and this summary
    bo = J("bposd_circuit_outcome.json")
    bc = bo["counts"]
    if bc["shots"] != rs["nontrivial"] or bc["certified_match"] != agreed:
        raise SystemExit("bposd_circuit_outcome.json does not describe the held-out circuit rung")
    if bc["by_tier"].get("tier2", 0) != rs["tier2"] or bc["by_tier"].get("tier1", 0) != rs["tier1"]:
        raise SystemExit("bposd_circuit_outcome.json tier counts disagree with the campaign summary")
    if bc["identical"] + bc["differing"] != bc["shots"]:
        raise SystemExit("bposd_circuit_outcome.json counts do not partition the shots")
    M["HoRefschedBposdObsAgree"] = I(bc["bposd_match"])
    M["HoRefschedReplaced"] = I(bc["differing"])
    M["HoRefschedReplacedSameObs"] = I(bc["differing_same_observable"])
    M["HoRefschedReplacedOtherObs"] = I(bc["differing_other_observable"])

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

    # ------------------------------------------------ strong baseline (v3: every proof retained and replayed)
    sbd = J("qldpc_strong_baseline_v3.json")
    sb = sbd["levels"]
    rp = J("strong_baseline_v3_replay.json")
    if not rp.get("passed") or rp.get("summary_sha256") != __import__("hashlib").sha256(
            (EV / "qldpc_strong_baseline_v3.json").read_bytes()).hexdigest():
        raise SystemExit("strong baseline: no passing no-producer replay of THIS summary "
                         "(python tools/replay_strong_baseline.py)")
    order = ("baseline_30_osdcs5", "strong_100_osdcs10", "strongest_ps_100_osdcs10")
    # The SCALED min-sum configurations (post hoc, 2026-11-02): A and B call ldpc's min-sum at its default scaling
    # factor 1.0, which this code punishes. They are classified by the v3 proofs under a tier-1 rule that must
    # reproduce v3's published counts, and their records must still be the bytes the summary names.
    sc = J("qldpc_strong_baseline_scaled.json")
    if any(v["published"] != v["this_rule"] for v in sc["published_rule_check"].values()):
        raise SystemExit("scaled baseline: its tier-1 rule no longer reproduces v3's published counts")
    for rel, digest in sc["files"].items():
        if __import__("hashlib").sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"scaled baseline: {rel} is not the record the summary was built from")
    if set(sc["levels"]) != set(sb) or sc["configs"]["A_s"]["ms_scaling_factor"] != sc["configs"]["B_s"]["ms_scaling_factor"]:
        raise SystemExit("scaled baseline: levels or scaling factors changed")
    scl = sc["levels"]
    # THE UNSCALED ARMS ACROSS DECLARED RATES (third referee, M1): at scaling 1 an arm's answers depend on the
    # floating-point form of the rate it is told, which exact arithmetic cannot tell apart, so A and B are reported as
    # run AND as their range over the 16 declared rates of qldpc_minsum_rate_scan.py. The scan must reproduce the
    # recorded as-run counts before its ranges mean anything.
    rs_ = J("qldpc_minsum_rate_scan.json")["datasets"]
    for lvl in ("p=0.01", "p=0.02", "p=0.03", "p=0.05", "p=0.07"):
        arms = rs_[f"strong {lvl}"]["arms"]
        for arm, rec in (("A", sb[lvl]["configs"][order[0]]), ("B", sb[lvl]["configs"][order[1]]),
                         ("A_s", scl[lvl]["A_s"]), ("B_s", scl[lvl]["B_s"])):
            if arms[arm]["as_run"]["hits"] != rec["optimum_hit"] or \
                    arms[arm]["as_run"]["failures"] != rec["bp_osd_logical_errors"]:
                raise SystemExit(f"rate scan: {arm} at {lvl} does not reproduce the recorded as-run counts")

    def ohr(lvl: str, arm: str) -> str:
        a = rs_[f"strong {lvl}"]["arms"][arm]
        return f"{a['as_run']['hits']}\\,[{a['hits_min']}--{a['hits_max']}]"

    inst = [r"\begin{tabular}{@{}lcccccccccc@{}}", r"\toprule",
            r" & \multicolumn{5}{c}{tier 1 (flat route)} & \multicolumn{5}{c}{optimum hit} \\",
            r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
            r"$p$ & A$^\dagger$ & B$^\dagger$ & C & A$_s$ & B$_s$ & A$^\dagger$ & B$^\dagger$ & C & A$_s$ & B$_s$ \\",
            r"\midrule"]
    for lvl in ("p=0.01", "p=0.02", "p=0.03", "p=0.05", "p=0.07"):
        v = sb[lvl]
        t1 = " & ".join([str(v["configs"][c]["tier1"]) for c in order] + [str(scl[lvl][c]["tier1"]) for c in ("A_s", "B_s")])
        oh = " & ".join([ohr(lvl, "A"), ohr(lvl, "B"), str(v["configs"][order[2]]["optimum_hit"])]
                        + [str(scl[lvl][c]["optimum_hit"]) for c in ("A_s", "B_s")])
        inst.append(rf"{lvl[2:]} & {t1} & {oh} \\")
    inst += [r"\bottomrule", r"\end{tabular}"]
    files["tab_instrument.tex"] = "\n".join(inst) + "\n"
    f5i, f7i = sb["p=0.05"], sb["p=0.07"]
    M["StrongFiveOptHitStated"] = f"{f5i['configs'][order[0]]['optimum_hit_rate']:.1f}"
    M["StrongFiveOptHitStrongest"] = f"{f5i['configs'][order[2]]['optimum_hit_rate']:.1f}"
    M["StrongSevenTierOneStated"] = f"{f7i['configs'][order[0]]['tier1_rate']:.1f}"
    M["StrongSevenTierOneStrongest"] = f"{f7i['configs'][order[2]]['tier1_rate']:.1f}"
    M["StrongSevenOptHitStated"] = f"{f7i['configs'][order[0]]['optimum_hit_rate']:.1f}"
    M["StrongSevenOptHitStrongest"] = f"{f7i['configs'][order[2]]['optimum_hit_rate']:.1f}"
    M["StrongSevenOptHitNotTierOne"] = I(f7i["configs"][order[2]]["optimum_hit_not_tier1"])
    if any(v["configs"][c]["tier1_not_optimum_hit"] for v in sb.values() for c in order):
        raise SystemExit("strong baseline: a tier-1 answer that is not an optimum hit; the text says none exists")
    M["StrongProofsFlat"] = I(sbd["proofs"]["flat"])
    M["StrongProofsTree"] = I(sbd["proofs"]["tree"])
    M["StrongTreeNodesMax"] = I(max(v["reference"]["trees"]["nodes_max"] or 0 for v in sb.values()))
    M["StrongTreeNodesMedianSeven"] = I(int(round(f7i["reference"]["trees"]["nodes_median"])))
    M["StrongTreeBytesMax"] = I(max(v["reference"]["trees"]["bytes_max"] or 0 for v in sb.values()))
    M["StrongReplayDecisions"] = I(rp["decisions"])
    M["StrongReplayProofChecks"] = I(rp["proof_checks"])
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & fixed & receipt pipeline & BP-OSD & \multicolumn{2}{c}{C: pipeline vs reference} \\",
             r"$p$ & reference & A$^\dagger$ / B$^\dagger$ / C & A$^\dagger$ / B$^\dagger$ / C / A$_s$ / B$_s$ & other optimum & other frame \\",
             r"\midrule"]
    for lvl in ("p=0.01", "p=0.02", "p=0.03", "p=0.05", "p=0.07"):
        v = sb[lvl]
        if v["reference"]["unresolved"] or any(v["configs"][c]["unresolved"] for c in order):
            raise SystemExit(f"{lvl}: an answer without a receipt; the table would mix evidence classes")
        pe = " / ".join(str(v["configs"][c]["pipeline_logical_errors_on_resolved"]) for c in order)
        be = " / ".join([str(v["configs"][c]["bp_osd_logical_errors"]) for c in order]
                        + [str(scl[lvl][c]["bp_osd_logical_errors"]) for c in ("A_s", "B_s")])
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
    M["ScaledAlpha"] = str(sc["configs"]["A_s"]["ms_scaling_factor"])
    M["LdpcVersion"] = sc["ldpc"]
    for tag, lvl in (("Five", "p=0.05"), ("Seven", "p=0.07")):
        for cfg in ("A", "B"):
            s = scl[lvl][f"{cfg}_s"]
            M[f"Strong{tag}OptHitScaled{cfg}"] = f"{s['optimum_hit_rate']:.1f}"
            M[f"Strong{tag}BpScaled{cfg}"] = I(s["bp_osd_logical_errors"])
            M[f"Strong{tag}TierOneScaled{cfg}"] = f"{s['tier1_rate']:.1f}"
    M["StrongFiveOptHitStatedB"] = f"{f5['configs'][order[1]]['optimum_hit_rate']:.1f}"
    M["StrongFiveBpStated"] = I(f5["configs"][order[0]]["bp_osd_logical_errors"])
    M["StrongFiveBpStatedB"] = I(f5["configs"][order[1]]["bp_osd_logical_errors"])
    M["StrongSevenBpStated"] = I(f7["configs"][order[0]]["bp_osd_logical_errors"])
    # The unscaled decoder's sensitivity to the declared rate, and the scaled settings' smoothness (post hoc).
    ss = J("qldpc_bposd_scaling_sensitivity.json")
    rb = ss["declared_rate_scan"]["p=0.05 strong_100_osdcs10"]
    M["SensRateLo"], M["SensRateHi"] = "0.03", "0.035"
    M["SensFailLo"], M["SensFailHi"] = I(rb["0.03"]), I(rb["0.035"])
    M["SensFailMin"], M["SensFailMax"] = I(min(rb.values())), I(max(rb.values()))
    if not rb["0.035"] >= 5 * rb["0.03"]:
        raise SystemExit("sensitivity: the text calls the change several-fold; the artifact no longer shows it")
    al = {float(a): v for a, v in ss["scaling_scan"]["p=0.05 strong_100_osdcs10"].items() if float(a) < 1}
    M["SensAlphas"] = ", ".join(f"{a:g}" for a in sorted(al, reverse=True))
    M["SensScaledMin"], M["SensScaledMax"] = I(min(al.values())), I(max(al.values()))
    # where 0.625 comes from (second referee, minor 8): the library's own two defaults, read from the installed package
    dfl = ss["library_default_ms_scaling_factor"]
    if dfl["BpOsdDecoder"] != 1.0 or dfl["SinterBpOsdDecoder"] != sc["configs"]["A_s"]["ms_scaling_factor"]:
        raise SystemExit("sensitivity: the text says the scaled value is the library's Sinter default and the "
                         "decoder class's default is 1; the artifact no longer shows both")
    M["LdpcClassAlpha"] = f"{dfl['BpOsdDecoder']:g}"
    # WHY the unscaled decoder depends on the declared rate (qldpc_minsum_rate_invariance.py, post hoc): exact-
    # arithmetic invariance confirmed at power-of-two prior LLRs, exact ties at scaling 1, and a tie-breaking jitter
    iv = J("qldpc_minsum_rate_invariance.json")
    cA, cB = iv["configs"]["baseline_30_osdcs5 alpha=1"], iv["configs"]["strong_100_osdcs10 alpha=1"]
    sA, sB = iv["configs"]["baseline_30_osdcs5 alpha=0.625"], iv["configs"]["strong_100_osdcs10 alpha=0.625"]
    nsh, LEVEL_TRUE = iv["shots"], iv["level"]
    if any(c["pow2_all_identical_shots"] != nsh for c in (cA, cB, sA, sB)):
        raise SystemExit("invariance: answers differ between power-of-two prior LLRs; the text says they cannot")
    if cA["jittered"]["scan_shots_with_one_answer"] != nsh or len(set(cA["jittered"]["scan_failures"].values())) != 1:
        raise SystemExit("invariance: the jittered unscaled A is not rate-independent; the text says it is")
    # what the text says: A's jittered count lies below every unscaled A count; B's bad regimes (its unscaled maximum)
    # vanish -- B's jittered maximum is under a fifth of its unscaled maximum
    if not (cA["jittered"]["scan_failures"][LEVEL_TRUE] < min(cA["scan_failures"].values())
            and 5 * max(cB["jittered"]["scan_failures"].values()) < max(cB["scan_failures"].values())):
        raise SystemExit("invariance: breaking the ties no longer removes the unscaled decoder's bad regimes")
    M["InvShots"] = I(nsh)
    M["InvOneAnswerA"], M["InvOneAnswerB"] = I(cA["scan_shots_with_one_answer"]), I(cB["scan_shots_with_one_answer"])
    M["InvFailALo"], M["InvFailAHi"] = I(min(cA["scan_failures"].values())), I(max(cA["scan_failures"].values()))
    M["InvJitterFailA"] = I(cA["jittered"]["scan_failures"][LEVEL_TRUE])
    jb = cB["jittered"]["scan_failures"].values()
    M["InvJitterFailBLo"], M["InvJitterFailBHi"] = I(min(jb)), I(max(jb))
    M["InvJitterOneAnswerB"] = I(cB["jittered"]["scan_shots_with_one_answer"])
    # the matched control (second referee, minor 3): the same declared-rate scan, scaled -- the text says it is flat
    sf = ss["declared_rate_scan_scaled_0.625"]
    for tag, lab in (("A", "baseline_30_osdcs5"), ("B", "strong_100_osdcs10")):
        vals = set(sf[f"p=0.05 {lab}"].values())
        if len(vals) != 1:
            raise SystemExit("sensitivity: the scaled decoder is not flat across declared rates; the text says it is")
        M[f"SensScaledFlat{tag}"] = I(vals.pop())
    # The held-out campaign's own syndromes, re-decoded with the scaled decoder (post hoc; optimum hit only, because the
    # held-out records keep no sampled error). The records must be the ones the campaign wrote.
    hp = J("qldpc_heldout_scaled_posthoc.json")
    if hp["decoder_scaled"]["ms_scaling_factor"] != sc["configs"]["A_s"]["ms_scaling_factor"]:
        raise SystemExit("held-out post hoc: a different scaling factor from the strong baseline's")
    for name, r in hp["rungs"].items():
        if __import__("hashlib").sha256((ROOT / r["records"]).read_bytes()).hexdigest() != r["records_sha256"]:
            raise SystemExit(f"held-out post hoc: {r['records']} changed since the analysis")
        # the replay must have reproduced the campaign exactly before its scaled numbers mean anything
        for k in ("syndrome_regenerated", "control_candidate_reproduced", "control_bposd_flag_reproduced",
                  "control_pipeline_flag_reproduced"):
            if r[k] != r["shots"]:
                raise SystemExit(f"held-out post hoc: {name} {k} {r[k]} of {r['shots']}")
    for name, tag in (("stress_p05", "Stress"), ("uniform_p02", "UniTwo"), ("xsector_p02", "XTwo")):
        r = hp["rungs"][name]
        M[f"HoPost{tag}N"] = I(r["shots"])
        M[f"HoPost{tag}Recorded"] = I(r["optimum_hit_recorded"])
        M[f"HoPost{tag}Scaled"] = I(r["optimum_hit_scaled"])
        M[f"HoPost{tag}MissRecorded"] = I(r["shots"] - r["optimum_hit_recorded"])
        M[f"HoPost{tag}MissScaled"] = I(r["shots"] - r["optimum_hit_scaled"])
        M[f"HoPost{tag}BpErrRecorded"] = I(r["bposd_logical_errors_recorded"])
        M[f"HoPost{tag}BpErrScaled"] = I(r["bposd_logical_errors_scaled"])
        M[f"HoPost{tag}PipeErr"] = I(r["pipeline_logical_errors"])
    # the datasets NOT re-decoded (second referee, minor 1): their BP-OSD answers proven heavier keep no attribution
    not_redecoded = [k for k in RUNG_ORDER if k not in hp["rungs"]]
    if sorted(not_redecoded) != sorted(["hetero_p02", "phenom_p01", "phenom_p02", "bb_refsched"]):
        raise SystemExit("held-out post hoc: the re-decoded datasets are not the ones the text names")
    M["HoNotRedecodedTierTwo"] = I(sum(rungs[k]["tier2"] for k in not_redecoded))
    st = hp["rungs"]["stress_p05"]
    M["HoPostStressPairScaledWrong"] = I(st["paired_scaled_wrong_pipeline_ok"])
    M["HoPostStressPairPipeWrong"] = I(st["paired_scaled_ok_pipeline_wrong"])
    M["HoPostStressMcNemar"] = f"{st['mcnemar_scaled_vs_pipeline_two_sided_p']:.2f}"

    # ONE-NODE TREES (second referee, M1): the producer files a shot "BD" whenever the LP primal is fractional, even
    # when its root dual alone closes; how many trees are one node, and how many optimal BP-OSD answers they certify.
    rd = J("qldpc_root_dual_classes.json")
    ho_trees = rd["trees"]["bd_trees"]
    M["HoTreesOneNode"] = I(int(ho_trees.get("1", 0)))
    if sum(ho_trees.values()) != tot_bd:
        raise SystemExit("root-dual classes: the held-out tree count differs from the campaign's")
    by_rung = rd["heldout_trees_by_rung"]
    if sum(v["trees"] for v in by_rung.values()) != tot_bd or \
            sum(v["one_node"] for v in by_rung.values()) != int(ho_trees.get("1", 0)):
        raise SystemExit("root-dual classes: the per-rung tree census disagrees with the tree histogram")
    no_branch = tot_flat + int(ho_trees.get("1", 0))
    M["HoNoBranch"] = I(no_branch)
    M["HoNoBranchPct"] = PF(no_branch / tot_n)
    # BP-OSD answers proven optimal: tier 1, plus the tree shots on which BP-OSD's own answer has the proven value
    tree_opt = sum(v["bposd_optimal"] for v in by_rung.values())
    M["HoTreeBpOptimal"] = I(tree_opt)
    M["HoBpProvenOptimal"] = I(tier1 + tree_opt)
    M["HoBpProvenOptimalPct"] = PF((tier1 + tree_opt) / tot_n)
    if by_rung["bb_refsched"]["bposd_optimal"] + rs["tier1"] + bc["differing"] != rs["nontrivial"]:
        raise SystemExit("root-dual classes: circuit tier 1 + optimal-on-tree + replaced != nontrivial")
    for tag, lvl in (("Five", "p=0.05"), ("Seven", "p=0.07")):
        c = rd["strong_baseline"][lvl]["strongest_ps_100_osdcs10"]
        if c["root_dual_accepts"] != c["on_one_node_tree"]:
            raise SystemExit("root-dual classes: a one-node tree's dual refused an optimal answer; the text says not")
        cfg_c = sb[lvl]["configs"][order[2]]
        # on an integral-primal shot the dual is tight, so every optimal answer there IS tier 1: the optimal answers
        # outside tier 1 are exactly those on tree shots -- two scripts, one count, or the explanation is wrong
        if cfg_c["optimum_hit_not_tier1"] != c["optimal_on_tree_shot"]:
            raise SystemExit("root-dual classes: optimal answers outside tier 1 are not the tree-shot answers")
        M[f"Strong{tag}OptOnTreeC"] = I(c["optimal_on_tree_shot"])
        M[f"Strong{tag}RootDualC"] = I(c["root_dual_accepts"])
        M[f"Strong{tag}BranchedC"] = I(c["optimal_on_tree_shot"] - c["root_dual_accepts"])
        M[f"Strong{tag}TierOneRuleC"] = I(cfg_c["tier1"] + c["root_dual_accepts"])
        n_c = cfg_c["tier1"] + cfg_c["tier2"] + cfg_c["tierBD"] + cfg_c["unresolved"]
        M[f"Strong{tag}TierOneRuleRateC"] = f"{100 * (cfg_c['tier1'] + c['root_dual_accepts']) / n_c:.1f}"
    # the same count for B (as run) and the scaled arms at p = 0.07 (third referee, minor 9)
    r7 = rd["strong_baseline"]["p=0.07"]
    for tag, key, rec in (("B", "strong_100_osdcs10", sb["p=0.07"]["configs"][order[1]]),
                          ("As", "A_s", scl["p=0.07"]["A_s"]), ("Bs", "B_s", scl["p=0.07"]["B_s"])):
        c = r7[key]
        if c["root_dual_accepts"] != c["on_one_node_tree"] or \
                rec["optimum_hit"] - rec["tier1"] != c["optimal_on_tree_shot"]:
            raise SystemExit(f"root-dual classes: {key} at p=0.07 -- optimal answers outside tier 1 are not the "
                             "tree-shot answers, or a one-node dual refused one")
        M[f"StrongSevenTierOneRule{tag}"] = I(rec["tier1"] + c["root_dual_accepts"])
    if M["StrongSevenTierOneRuleAs"] != M["StrongSevenTierOneRuleBs"]:
        raise SystemExit("root-dual classes: the text gives A_s and B_s one tier-1 count; they differ")
    # THE COMPARISONS THE TEXT DRAWS FROM UNSCALED ARMS, each required to hold at EVERY declared rate of the scan
    # (third referee, M1: a comparison whose sign flips across rates that exact arithmetic cannot tell apart is a
    # rounding artifact, and is refused here rather than printed). Optimum hits; "Ho" is the held-out stress rung.
    for tag, ds in (("Five", "strong p=0.05"), ("Seven", "strong p=0.07"), ("Ho", "heldout stress_p05")):
        a = rs_[ds]["arms"]
        A, B, As, Bs, Aj = a["A"], a["B"], a["A_s"], a["B_s"], a["A_j"]
        for arm in ("A_s", "B_s", "A_j"):
            if a[arm]["shots_with_one_answer"] != rs_[ds]["shots"] and arm != "B_s":
                raise SystemExit(f"rate scan {ds}: {arm} is not rate-independent; the text says it is")
        # scaling beats the unscaled A at every rate; breaking ties recovers most, not all, of the gap
        if not (A["hits_max"] < Aj["hits_min"] <= Aj["hits_max"] < As["hits_min"]):
            raise SystemExit(f"rate scan {ds}: A < A_j < A_s does not hold at every declared rate")
        mid = (A["hits_min"] + A["hits_max"]) / 2
        M[f"Scan{tag}JitterRecoveredPct"] = I(math.floor(100 * (Aj["hits_min"] - mid) / (As["hits_min"] - mid)))
        M[f"Scan{tag}ALo"], M[f"Scan{tag}AHi"] = I(A["hits_min"]), I(A["hits_max"])
        M[f"Scan{tag}BLo"], M[f"Scan{tag}BHi"] = I(B["hits_min"]), I(B["hits_max"])
        M[f"Scan{tag}As"], M[f"Scan{tag}Bs"], M[f"Scan{tag}Aj"] = I(As["hits_min"]), I(Bs["hits_min"]), I(Aj["hits_min"])
        M[f"Scan{tag}AFailLo"], M[f"Scan{tag}AFailHi"] = I(A["failures_min"]), I(A["failures_max"])
        M[f"Scan{tag}ConvA"], M[f"Scan{tag}ConvAj"] = I(A["as_run"]["converged"]), I(Aj["as_run"]["converged"])
        M[f"Scan{tag}ConvAs"] = I(As["as_run"]["converged"])
        M[f"Scan{tag}ZeroA"] = I(A["as_run"]["zero_soft_shots"])
        if Aj["as_run"]["zero_soft_shots"] or As["as_run"]["zero_soft_shots"]:
            raise SystemExit(f"rate scan {ds}: an exact-zero soft value with the ties broken; the text says none")
    if min(int(M[f"Scan{t}JitterRecoveredPct"].replace("{,}", "")) for t in ("Five", "Seven", "Ho")) < 75:
        raise SystemExit("rate scan: breaking ties no longer recovers most of A's gap; the text says it does")
    M["ScanJitterRecoveredMinPct"] = I(min(int(M[f"Scan{t}JitterRecoveredPct"]) for t in ("Five", "Seven", "Ho")))
    # the shots keeping one answer are printed from the invariance run; the scan over the same 16 rates must agree
    for arm, mac in (("A", "InvOneAnswerA"), ("B", "InvOneAnswerB")):
        if I(rs_["strong p=0.05"]["arms"][arm]["shots_with_one_answer"]) != M[mac]:
            raise SystemExit(f"rate scan p=0.05: {arm}'s one-answer count differs from the invariance run's")
    M["ScanHoAMissLo"] = I(rs_["heldout stress_p05"]["shots"] - rs_["heldout stress_p05"]["arms"]["A"]["hits_max"])
    M["ScanHoAMissHi"] = I(rs_["heldout stress_p05"]["shots"] - rs_["heldout stress_p05"]["arms"]["A"]["hits_min"])
    # B against B_s: at p = 0.07 the scaled decoder wins at every declared rate; at p <= 0.05 the unscaled B matches
    # or beats it at its best rates and falls far below at its worst, so the text draws no comparison there
    b7 = rs_["strong p=0.07"]["arms"]
    if not b7["B"]["hits_max"] < b7["B_s"]["hits_min"]:
        raise SystemExit("rate scan: B_s no longer beats B at every declared rate at p=0.07")
    for lvl in ("p=0.01", "p=0.02", "p=0.03", "p=0.05"):
        bb = rs_[f"strong {lvl}"]["arms"]
        if not bb["B"]["hits_min"] < bb["B_s"]["hits_min"] <= bb["B"]["hits_max"]:
            raise SystemExit(f"rate scan {lvl}: B against B_s is no longer undetermined; the text says it is")

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

    jt = J("qldpc_tree_js_agreement.json")
    jt_src = ROOT / jt["checker"]
    if __import__("hashlib").sha256(jt_src.read_bytes()).hexdigest() != jt["checker_sha256"]:
        raise SystemExit("JavaScript tree agreement: the checker changed since the artifact was written; re-run it")
    if (jt["disagreements"] or jt["trees_accepted"] != jt["trees"]
            or jt["planted_defects_refused_for_their_reason"] != jt["planted_defects"]):
        raise SystemExit("JavaScript tree agreement: the text says every tree accepted and every defect refused")
    M["JsTreeAccepted"] = I(jt["trees_accepted"])
    M["JsTreeDefects"] = I(jt["planted_defects"])
    M["JsTreeDefectClasses"] = I(len(jt["planted_by_class"]))

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
        _w = OUT / name
        print("wrote", _w.relative_to(ROOT) if _w.is_relative_to(ROOT) else _w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
