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


# ROUNDING RULE (final referee of paper 2, 2026-11-03). A printed number may not say more than the evidence: a point
# percentage and a LOWER bound are floored at the last shown digit (PF), an UPPER bound is ceilinged (PC), a printed
# range floors its low end and ceilings its high end, and a bound written with significant digits rounds away from
# the inside (BOUND). The upper bound 1 - 0.05^(1/149) = 1.9905% had been floored to 1.99%, below the bound it states.
def PF(x: float, k: int = 2) -> str:
    """Percentage, floored at k decimals: a point estimate or a LOWER bound."""
    v = math.floor(100 * x * 10 ** k + 1e-9) / 10 ** k
    return f"{v:.{k}f}"


def PC(x: float, k: int = 2) -> str:
    """Percentage, ceilinged at k decimals: an UPPER bound, which must never print below the value it bounds."""
    v = math.ceil(100 * x * 10 ** k - 1e-9) / 10 ** k
    return f"{v:.{k}f}"


def _sig(x: float, k: int, rounding: str) -> tuple:
    """(mantissa, exponent) of x at k significant digits under a decimal rounding mode, computed on x's exact
    binary value, so a half-way case is not decided by a float artefact."""
    from decimal import Decimal
    d = Decimal(x)
    e = d.adjusted()
    unit = Decimal(1).scaleb(1 - k)
    m = d.scaleb(-e).quantize(unit, rounding=rounding)
    if m >= 10:
        m, e = (m / 10).quantize(unit, rounding=rounding), e + 1
    return m, e


def _sig_text(m, e: int) -> str:
    if e < -3:
        return rf"\ensuremath{{{m:f}\times 10^{{{e}}}}}"
    return f"{m.scaleb(e).normalize():f}"


def BOUND(x: float, side: str, k: int = 2) -> str:
    """A stated bound at k significant digits, rounded AWAY from the inside: an upper bound ('ceil') never prints
    below its value and a lower bound ('floor') never above it; scientific notation below 10^-3."""
    from decimal import ROUND_CEILING, ROUND_FLOOR
    if side not in ("ceil", "floor"):
        raise ValueError(f"a bound rounds 'ceil' or 'floor', not {side!r}")
    if x <= 0:
        return "0"
    return _sig_text(*_sig(x, k, ROUND_CEILING if side == "ceil" else ROUND_FLOOR))


def PVAL(p: float) -> str:
    """An exact p-value as a point value: three significant digits, half-up on the exact binary value (so 0.0625
    prints 0.0625, never 0.062), '1' for 1, and two significant digits in scientific notation below 10^-3."""
    from decimal import ROUND_HALF_UP
    if p >= 1:
        return "1"
    return _sig_text(*_sig(p, 2 if p < 1e-3 else 3, ROUND_HALF_UP))


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
             r"dataset & sampled & tier 0 & $N$ & tier 1 & tier 2 & branching & certified & 95\% lower \\",
             r"\midrule"]
    # the tier columns are WHAT THE RECEIPT PROVES (qldpc_tier_by_receipt.py; Sec. III): one dual on BP-OSD's own
    # answer, one dual on a replacement, or a branching tree -- not the producer's route (adjudication P2-02)
    rt_ho = J("qldpc_tier_by_receipt.json")["heldout"]
    for k in RUNG_ORDER:
        r = rungs[k]
        bd = r["exact_certified"] - r["tier1"] - r["tier2"]
        q = rt_ho[k]
        need(q["N"] == r["nontrivial"] and q.get("tier1", 0) + q.get("tier2", 0) + q.get("branching_bp_optimal", 0)
             + q.get("branching_repair", 0) == r["exact_certified"], f"{k}: receipt tiers do not reconcile")
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
            rf"{I(q.get('tier1', 0))} & {I(q.get('tier2', 0))} & "
            rf"{I(q.get('branching_bp_optimal', 0) + q.get('branching_repair', 0))} & {I(r['exact_certified'])} & "
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
    # zero observed disagreements in N: one-sided 95% upper bound 1 - 0.05^(1/N). An UPPER bound, so CEILINGED
    # (PC): flooring it, as the first revision did, printed 1.99% for a bound of 1.9905% -- below the bound it states
    if agreed == of:
        M["HoRefschedObsUpperPct"] = PC(1 - 0.05 ** (1 / of))
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
            r" & \multicolumn{5}{c}{tier 1 (one dual)} & \multicolumn{5}{c}{optimum hit} \\",
            r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
            r"$p$ & A$^\dagger$ & B$^\dagger$ & C & A$_s$ & B$_s$ & A$^\dagger$ & B$^\dagger$ & C & A$_s$ & B$_s$ \\",
            r"\midrule"]
    for lvl in ("p=0.01", "p=0.02", "p=0.03", "p=0.05", "p=0.07"):
        v = sb[lvl]
        rtl = J("qldpc_tier_by_receipt.json")["strong_baseline"][lvl]   # tier 1 by the receipt, Sec. III
        t1 = " & ".join(str(rtl[c]["tier1"]) for c in ("A", "B", "C", "A_s", "B_s"))
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

    # ------------------------------------------------ 2026-11-03 adjudication: receipt-defined tiers, all-8 post hoc,
    #                                                  the P2/P3 mechanism equality, the scale probe with trees
    _receipt_tiers(M, files, tot_n, tier1)
    _posthoc_all8(M, files)
    _p3_equality(M)
    _scale_trees(M)
    _datasets(files)
    _prereg(M, files)
    _prereg_costs(M, files)
    _reverification(M)

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
            "% Point percentages and lower bounds are floored at the last shown digit; upper bounds are ceilinged."]
    for k in sorted(M):
        body.append(rf"\newcommand{{\{k}}}{{{M[k]}}}")
    files["numbers.tex"] = "\n".join(body) + "\n"
    return files


def need(cond: bool, msg: str) -> None:
    """A generated number is printed only if the evidence still says what the text will say about it."""
    if not cond:
        raise SystemExit(f"need: {msg}")


def _fresh(doc: dict, what: str) -> None:
    """Every file an artifact names in code_sha256 must still hash to it: stale evidence cannot be typeset."""
    import hashlib
    for rel, h in doc.get("code_sha256", {}).items():
        need(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h,
             f"{what}: {rel} changed since the artifact was produced; re-run its script")


CONFIG_WORD = {"A": "A", "B": "B", "C": "C", "A_s": "As", "B_s": "Bs"}
RUNG_WORD = {"uniform_p01": "UniOne", "uniform_p02": "UniTwo", "hetero_p02": "Hetero", "phenom_p01": "PhenOne",
             "phenom_p02": "PhenTwo", "xsector_p02": "Xsec", "stress_p05": "Stress", "bb_refsched": "Circuit"}
LEVEL_WORD = {"p=0.01": "One", "p=0.02": "Two", "p=0.03": "Three", "p=0.05": "Five", "p=0.07": "Seven"}


def _receipt_tiers(M: dict, files: dict, tot_n: int, recorded_tier1: int) -> None:
    """Tiers by what the receipt proves (experiments/m0_prior_art/qldpc_tier_by_receipt.py), Sec. III's rule."""
    d = J("qldpc_tier_by_receipt.json")
    _fresh(d, "qldpc_tier_by_receipt")
    ho, t = d["heldout"], d["heldout_totals"]
    need(list(ho) == RUNG_ORDER, "receipt tiers: held-out dataset order")
    need(t["N"] == tot_n, "receipt tiers: shot total differs from the campaign's")
    need(t["tier1"] + t["tier2"] + t["branching_bp_optimal"] + t["branching_repair"] == t["N"],
         "receipt tiers do not partition the held-out shots")
    need(t["single_dual_attaining"] + t["single_dual_lattice"] == t["tier1"] + t["tier2"],
         "every single-dual decision is attaining or lattice")
    need(t["tier1"] >= recorded_tier1, "a receipt-defined tier 1 can only add to the producer's")
    M["RtHoTierOne"] = I(t["tier1"])
    M["RtHoTierTwo"] = I(t["tier2"])
    M["RtHoSingleDual"] = I(t["tier1"] + t["tier2"])
    M["RtHoSingleDualPct"] = PF((t["tier1"] + t["tier2"]) / t["N"])
    M["RtHoSingleAttaining"] = I(t["single_dual_attaining"])
    M["RtHoSingleLattice"] = I(t["single_dual_lattice"])
    M["RtHoBranching"] = I(t["branching_bp_optimal"] + t["branching_repair"])
    M["RtHoBranchingBpOptimal"] = I(t["branching_bp_optimal"])
    M["RtHoBpOptimal"] = I(t["tier1"] + t["branching_bp_optimal"])
    M["RtHoTierOneAddedBySecThree"] = I(t["tier1"] - recorded_tier1)
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"dataset & $N$ & tier 1 & tier 2 & single dual: $L=U$ / lattice & branching: BP-OSD optimal / heavier \\",
             r"\midrule"]
    for k in RUNG_ORDER:
        r = ho[k]
        lines.append(rf"{RUNG_LABEL[k]} & {I(r['N'])} & {I(r.get('tier1', 0))} & {I(r.get('tier2', 0))} & "
                     rf"{I(r.get('single_dual_attaining', 0))} / {I(r.get('single_dual_lattice', 0))} & "
                     rf"{I(r.get('branching_bp_optimal', 0))} / {I(r.get('branching_repair', 0))} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_heldout_receipt.tex"] = "\n".join(lines) + "\n"
    sb = d["strong_baseline"]
    # compact: the levels where the producer's label and the receipt's tier differ; the others are checked equal
    rows_ = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"$p$ & config. & tier 1, producer & tier 1, receipt & optimum hit \\", r"\midrule"]
    for lvl, word in LEVEL_WORD.items():
        for cfg, cw in CONFIG_WORD.items():
            v = sb[lvl][cfg]
            need(v["tier1"] >= v["recorded_tier1"], f"{lvl} {cfg}: receipt tier 1 below the recorded one")
            need(v["optimum_hit"] == v["tier1"] + v.get("branching_bp_optimal", 0), f"{lvl} {cfg}: hit split")
            M[f"Rt{word}TierOne{cw}"] = I(v["tier1"])
            M[f"Rt{word}OptHit{cw}"] = I(v["optimum_hit"])
            if lvl in ("p=0.01", "p=0.02", "p=0.03"):
                need(v["tier1"] == v["recorded_tier1"], f"{lvl} {cfg}: the compact table says the labels agree here")
                continue
            label = cfg.replace("_s", "$_s$")
            rows_.append(rf"{lvl.split('=')[1]} & {label} & {I(v['recorded_tier1'])} & {I(v['tier1'])} & "
                         rf"{I(v['optimum_hit'])} \\")
    rows_ += [r"\bottomrule", r"\end{tabular}"]
    files["tab_instrument_receipt.tex"] = "\n".join(rows_) + "\n"


def _reverification(M: dict) -> None:
    """The newest re-verification record (qldpc_receipt_reverification.py --out), the one the current checker pins
    rest on: the shared input-domain vectors it names and the zero-multiplier census of the stored corpora."""
    import hashlib
    add = J("qldpc_heldout_addendum.json")
    rel = add["tree_checker_b_previous_pins"][-1]["reverification_record"]
    rec = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    _fresh(rec, rel)
    need(rec["code_sha256"]["tools/external/tree_checker_b.py"] == add["tree_checker_b_sha256"],
         f"{rel} does not name the tree checker the addendum pins")
    need(rec["replay"]["complete"] and rec["replay"]["receipts_passed"] == 2042, f"{rel}: replay incomplete")
    pre = rec["prereg_campaign_receipts"]
    need(all(v["both_checkers_accept"] == v["receipts"] for v in pre.values()), f"{rel}: a campaign receipt refused")
    need(rec["strong_baseline_replay"].startswith("STRONG-BASELINE REPLAY PASSED"), f"{rel}: strong baseline replay")
    vec = rec["input_domain_vectors"]
    need(hashlib.sha256((ROOT / vec["file"]).read_bytes()).hexdigest() == vec["sha256"],
         f"{rel}: the shared vector file changed since the record")
    M["DomainVectors"] = I(vec["vectors"])
    z = rec["zero_multiplier_facets_in_stored_receipts"]
    M["ZeroMultFacets"] = I(sum(v["zero_multiplier_facets"] for v in z.values()))
    M["ZeroMultReceipts"] = I(sum(v["receipts_with_one"] for v in z.values()))


def _posthoc_all8(M: dict, files: dict) -> None:
    """All eight held-out datasets re-decoded at scaling 1.0 and 0.625 (POST HOC; qldpc_heldout_scaled_posthoc_all8)."""
    d = J("qldpc_heldout_scaled_posthoc_all8.json")
    _fresh(d, "qldpc_heldout_scaled_posthoc_all8")
    need("NOT preregistered" in d["post_hoc"], "the all-8 re-decode must carry its post-hoc label")
    need(list(d["rungs"]) == RUNG_ORDER, "all-8 re-decode: dataset order")
    lines = [r"\begin{tabular}{@{}lrrrrrl@{}}", r"\toprule",
             r"dataset & $N$ & hit, 1.0 & hit, 0.625 & 1.0 only & 0.625 only & McNemar $p$ \\", r"\midrule"]
    tot = {"1.0": 0, "0.625": 0}
    for k in RUNG_ORDER:
        r = d["rungs"][k]
        need(all(r["controls"].values()), f"{k}: a control of the all-8 re-decode failed")
        need(all(r["alpha"][a]["lighter_than_receipt"] == 0 for a in ("1.0", "0.625")),
             f"{k}: a decoder answer lighter than a receipt")
        need(r["alpha"]["1.0"]["optimum_hit"] == r["recorded_optimum_hits"], f"{k}: control hits differ")
        h1, hs = r["alpha"]["1.0"]["optimum_hit"], r["alpha"]["0.625"]["optimum_hit"]
        pr = r["paired_optimum_hit"]
        need(h1 - pr["hit_at_1.0_only"] == hs - pr["hit_at_0.625_only"], f"{k}: paired counts do not reconcile")
        tot["1.0"] += h1
        tot["0.625"] += hs
        w = RUNG_WORD[k]
        M[f"Pa{w}N"] = I(r["shots"])
        M[f"Pa{w}HitUnscaled"] = I(h1)
        M[f"Pa{w}HitScaled"] = I(hs)
        M[f"Pa{w}UnscaledOnly"] = I(pr["hit_at_1.0_only"])
        M[f"Pa{w}ScaledOnly"] = I(pr["hit_at_0.625_only"])
        ptxt = PVAL(pr["exact_two_sided_mcnemar_p"])
        lines.append(rf"{RUNG_LABEL[k]} & {I(r['shots'])} & {I(h1)} & {I(hs)} & {I(pr['hit_at_1.0_only'])} & "
                     rf"{I(pr['hit_at_0.625_only'])} & {ptxt} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_posthoc_all8.tex"] = "\n".join(lines) + "\n"
    need(tot == {k: v for k, v in d["totals_per_alpha"].items()}, "all-8 totals do not reconcile")
    M["PaShots"] = I(d["shots_total"])
    M["PaHitUnscaled"] = I(tot["1.0"])
    M["PaHitScaled"] = I(tot["0.625"])
    M["PaMissUnscaled"] = I(d["shots_total"] - tot["1.0"])
    M["PaMissScaled"] = I(d["shots_total"] - tot["0.625"])
    worse = [k for k in RUNG_ORDER if d["rungs"][k]["alpha"]["0.625"]["optimum_hit"]
             < d["rungs"][k]["alpha"]["1.0"]["optimum_hit"]]
    M["PaDatasetsScaledFewer"] = I(len(worse))


def _p3_equality(M: dict) -> None:
    """Paper 2's circuit model against paper 3's IBM model at T = 3 (qldpc_refsched_p3_equality.py)."""
    d = J("qldpc_refsched_p3_equality.json")
    _fresh(d, "qldpc_refsched_p3_equality")
    for key in ("mechanisms_p2_observable", "mechanisms_13_observables"):
        need(d[key]["relation"] == "equal" and d[key]["P2_only"] == 0 and d[key]["P3_only"] == 0,
             f"P2/P3 equality: {key} is no longer equal")
    need(d["stored_model"]["sha256"] == d["stored_model"]["manifest_sha256"], "P3 stored model hash")
    need(d["stored_model"]["mechanisms_vs_p2_with_p3_observables"]["relation"] == "equal", "P3 stored model")
    need(d["detectors"]["alignment_is_a_bijection_onto_p3_z_type"], "detector alignment")
    w = d["witness"]
    need(w["zero_syndrome_and_flips_p2_observable"] and w["no_9_column_subset_does_both"], "the weight-10 witness")
    M["EqMechanisms"] = I(d["mechanisms_p2_observable"]["P2"])
    M["EqMechanismsPthree"] = I(d["mechanisms_p2_observable"]["P3"])
    M["EqWitnessWeight"] = I(w["weight"])


def _scale_trees(M: dict) -> None:
    """The fixed-point scale probe with exact receipts and tree closure (qldpc_scale_probe_trees.py)."""
    p = EV / "qldpc_scale_robustness_trees.json"
    if not p.is_file():
        raise SystemExit("need: evidence/qldpc_scale_robustness_trees.json -- run "
                         "experiments/m0_prior_art/qldpc_scale_probe_trees.py")
    d = json.loads(p.read_text(encoding="utf-8"))
    _fresh(d, "qldpc_scale_probe_trees")
    sc = d["scales"]
    need(list(sc) == ["1000", "10000", "100000"], "scale probe: scales")
    M["ScaleTreesCertified"] = " / ".join(I(sc[s]["certified_exact"]) for s in sc)
    M["ScaleTreesShots"] = I(300)
    M["ScaleTreesSupportChanges"] = I(sum(sc[s]["selected_correction_changes_vs_1000"] for s in sc))
    M["ScaleTreesRouteChanges"] = I(sum(sc[s]["route_changes_vs_1000"] for s in sc))
    M["ScaleTreesTreeShots"] = " / ".join(I(len(sc[s]["tree_shots"])) for s in sc)


DATASET_ROWS = {   # dataset -> (instance, check matrix, fault model, decoder prior, objective)
    "uniform_p01": ("bb144_z_code_capacity", r"$H_Z$", r"i.i.d.\ $X$", r"$p$", "unit"),
    "uniform_p02": ("bb144_z_code_capacity", r"$H_Z$", r"i.i.d.\ $X$", r"$p$", "unit"),
    "hetero_p02": ("bb144_z_code_capacity", r"$H_Z$", r"independent $X$, $p_i\sim 0.02\,U(\tfrac12,2)$",
                   r"$p_i$", "integer"),
    "phenom_p01": ("bb144_phenom_r4", r"spacetime $H_Z$", r"i.i.d.\ data and measurement", r"$p$", "integer"),
    "phenom_p02": ("bb144_phenom_r4", r"spacetime $H_Z$", r"i.i.d.\ data and measurement", r"$p$", "integer"),
    "xsector_p02": ("bb144_x_code_capacity", r"$H_X$", r"i.i.d.\ $Z$", r"$p$", "unit"),
    "stress_p05": ("bb144_z_code_capacity", r"$H_Z$", r"i.i.d.\ $X$", r"$p$", "unit"),
    "bb_refsched": ("bb144_refsched_r3_p002", "DEM", r"circuit-level, idles unfaulted", "DEM", "integer"),
}


def _datasets(files: dict) -> None:
    """Table I: what every dataset IS -- matrix, size, fault model, decoder prior -- read from the persisted instances."""
    lines = [r"\begin{tabular}{@{}llrlll@{}}", r"\toprule",
             r"dataset & check matrix & rows $\times$ columns & fault model & prior & objective \\", r"\midrule"]
    for k in RUNG_ORDER:
        inst, hname, fault, prior, obj = DATASET_ROWS[k]
        d = J(f"instances/{inst}.json")
        lines.append(rf"{RUNG_LABEL[k]} & {hname} & {I(len(d['checks']))} $\times$ {I(d['n_variables'])} & {fault} & "
                     rf"{prior} & {obj} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_datasets.tex"] = "\n".join(lines) + "\n"


def _prereg(M: dict, files: dict) -> None:
    """The preregistered comparison of the two scaling settings (docs/PREREG_qldpc_scaling_2026-11-03.md)."""
    import hashlib
    import importlib.util
    p = EV / "qldpc_scaling_prereg" / "summary.json"
    need(p.is_file(), "the preregistered campaign has not been run: evidence/qldpc_scaling_prereg/summary.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(
        "qldpc_scaling_prereg_campaign", ROOT / "experiments" / "m0_prior_art" / "qldpc_scaling_prereg_campaign.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    prereg = (ROOT / "docs" / "PREREG_qldpc_scaling_2026-11-03.md").read_text(encoding="utf-8")
    need(not d["smoke"] and d["seed_domain"] == run.SEED_DOMAIN, "prereg summary is a smoke run or another domain")
    for rel, h in d["code_sha256"].items():
        need(f"`{rel}` | {h} |" in prereg, f"prereg: {rel} ran under bytes the preregistration does not pin")
    need(list(d["datasets"]) == list(run.DATASETS), "prereg: datasets missing or out of the preregistered order")
    # a dataset run after a recorded deviation carries its own bytes: the checkers and producer must still be the
    # pinned ones, and the runner's must be the one the deviation record names
    devs = (ROOT / "docs" / "PREREG_qldpc_scaling_2026-11-03_DEVIATIONS.md").read_text(encoding="utf-8")
    deviated = 0
    for k, s in d["datasets"].items():
        if "code_sha256" not in s:
            continue
        need("deviation" in s, f"prereg {k}: ran under other bytes with no recorded deviation")
        deviated += 1
        for rel, h in s["code_sha256"].items():
            ok = (f"`{rel}` | {h} |" in prereg) or (rel.endswith("qldpc_scaling_prereg_campaign.py") and h in devs)
            need(ok, f"prereg {k}: {rel} ran under bytes neither the preregistration nor its deviations name")
    M["PrDeviatedDatasets"] = I(deviated)
    M["PrCommittedDatasets"] = I(len(d["datasets"]) - deviated)
    # DEVIATION 1 gave the phenomenological datasets the lazy LP; the text says exactly one of them ran after it
    lazy = [k for k, s in d["datasets"].items() if run.DATASETS[k][1] == "phenom" and "code_sha256" in s]
    need(lazy == ["phenom_p02"] and "code_sha256" not in d["datasets"]["phenom_p01"],
         f"prereg: the phenomenological datasets run on the lazy LP are {lazy}; the text says phenom_p02 alone")
    for k, s in d["datasets"].items():
        M[f"Pr{RUNG_WORD[k]}FlatReceipts"] = I(s["receipt_kinds"]["flat"])
        M[f"Pr{RUNG_WORD[k]}TreeReceipts"] = I(s["receipt_kinds"]["tree"])
        need(s["receipt_kinds"]["flat"] + s["receipt_kinds"]["tree"] == s["receipts"], f"prereg {k}: receipt kinds")
    # the preregistered split of every hit (single dual / branching tree) is printed beside the hit count, and every
    # miss is split into heavier answers and non-corrections (final referee, m1: print the preregistered analyses)
    lines = [r"\begin{tabular}{@{}lrrrrrrl@{}}", r"\toprule",
             r"dataset & $N$ & unresolved & hit (tree), 1.0 & hit (tree), 0.625 & 1.0 only & 0.625 only & "
             r"McNemar $p$ \\",
             r"\midrule"]
    sec = [r"\begin{tabular}{@{}lrrrrlr@{}}", r"\toprule",
           r"dataset & errors, 1.0 & errors, 0.625 & 1.0 only & 0.625 only & McNemar $p$ & receipt answer \\",
           r"\midrule"]
    more = fewer = same = unresolved = shots = not_corr = 0
    primary, secondary = {}, {}
    for k, s in d["datasets"].items():
        need(s["seed"] == run.seed_for(run.SEED_DOMAIN, k) and str(s["seed"]) in prereg, f"prereg {k}: seed")
        need(s["nontrivial"] == run.DATASETS[k][5], f"prereg {k}: shot count is not the preregistered one")
        rec = EV / "qldpc_scaling_prereg" / f"{k}.jsonl.gz"
        need(hashlib.sha256(rec.read_bytes()).hexdigest() == s["records_sha256"], f"prereg {k}: records altered")
        need(s["receipts"] + s["unresolved"] == s["nontrivial"], f"prereg {k}: receipts do not reconcile")
        c1, c2 = s["per_configuration"]["1.0"], s["per_configuration"]["0.625"]
        for c in (c1, c2):
            need(c["optimum_hit"] == c["tier1_single_dual"] + c["optimum_on_branching_tree"], f"prereg {k}: hits")
            need(c["optimum_hit"] + c["miss_heavier"] + c["not_a_correction"] == s["receipts"], f"prereg {k}: split")
        pr = s["paired_optimum_hit"]
        need(c1["optimum_hit"] - pr["hit_at_1.0_only"] == c2["optimum_hit"] - pr["hit_at_0.625_only"],
             f"prereg {k}: paired counts")
        more += c2["optimum_hit"] > c1["optimum_hit"]
        fewer += c2["optimum_hit"] < c1["optimum_hit"]
        same += c2["optimum_hit"] == c1["optimum_hit"]
        unresolved += s["unresolved"]
        shots += s["nontrivial"]
        w = RUNG_WORD[k]
        M[f"Pr{w}N"] = I(s["nontrivial"])
        M[f"Pr{w}HitUnscaled"] = I(c1["optimum_hit"])
        M[f"Pr{w}HitScaled"] = I(c2["optimum_hit"])
        M[f"Pr{w}UnscaledOnly"] = I(pr["hit_at_1.0_only"])
        M[f"Pr{w}ScaledOnly"] = I(pr["hit_at_0.625_only"])
        pv = pr["exact_two_sided_mcnemar_p"]
        need(abs(pv - run.mcnemar(pr["hit_at_1.0_only"], pr["hit_at_0.625_only"])) <= 1e-12 * max(pv, 1e-300),
             f"prereg {k}: the stored McNemar p is not the exact test of its own counts")
        primary[k] = (pv, c1["optimum_hit"], c2["optimum_hit"])
        ptxt = PVAL(pv)
        M[f"Pr{w}McNemar"] = ptxt
        M[f"Pr{w}TreeUnscaled"] = I(c1["optimum_on_branching_tree"])
        M[f"Pr{w}TreeScaled"] = I(c2["optimum_on_branching_tree"])
        not_corr += c1["not_a_correction"] + c2["not_a_correction"]
        lines.append(rf"{RUNG_LABEL[k]} & {I(s['nontrivial'])} & {I(s['unresolved'])} & "
                     rf"{I(c1['optimum_hit'])} ({I(c1['optimum_on_branching_tree'])}) & "
                     rf"{I(c2['optimum_hit'])} ({I(c2['optimum_on_branching_tree'])}) & "
                     rf"{I(pr['hit_at_1.0_only'])} & {I(pr['hit_at_0.625_only'])} & {ptxt} \\")
        if "logical_errors" in c1:
            le = s["paired_logical_error_secondary"]
            need(c1["logical_errors"] - le["error_at_1.0_only"] == c2["logical_errors"] - le["error_at_0.625_only"],
                 f"prereg {k}: paired logical errors do not reconcile")
            lp = le["exact_two_sided_mcnemar_p"]
            need(abs(lp - run.mcnemar(le["error_at_1.0_only"], le["error_at_0.625_only"])) <= 1e-12 * max(lp, 1e-300),
                 f"prereg {k}: the stored logical McNemar p is not the exact test of its own counts")
            secondary[k] = (lp, c1["logical_errors"], c2["logical_errors"])
            M[f"Pr{w}LogUnscaled"] = I(c1["logical_errors"])
            M[f"Pr{w}LogScaled"] = I(c2["logical_errors"])
            M[f"Pr{w}LogUnscaledOnly"] = I(le["error_at_1.0_only"])
            M[f"Pr{w}LogScaledOnly"] = I(le["error_at_0.625_only"])
            M[f"Pr{w}LogMcNemar"] = PVAL(lp)
            M[f"Pr{w}LogReceipt"] = I(s["receipt_answer_logical_errors"])
            sec.append(rf"{RUNG_LABEL[k]} & {I(c1['logical_errors'])} & {I(c2['logical_errors'])} & "
                       rf"{I(le['error_at_1.0_only'])} & {I(le['error_at_0.625_only'])} & {PVAL(lp)} & "
                       rf"{I(s['receipt_answer_logical_errors'])} \\")
        if "observable_agreement" in c1:
            M[f"Pr{w}ObsUnscaled"] = I(c1["observable_agreement"])
            M[f"Pr{w}ObsScaled"] = I(c2["observable_agreement"])
    lines += [r"\bottomrule", r"\end{tabular}"]
    sec += [r"\bottomrule", r"\end{tabular}"]
    files["tab_prereg.tex"] = "\n".join(lines) + "\n"
    files["tab_prereg_logical.tex"] = "\n".join(sec) + "\n"
    M["PrShots"] = I(shots)
    M["PrUnresolved"] = I(unresolved)
    M["PrDatasets"] = I(len(d["datasets"]))
    M["PrScaledMore"] = I(more)
    M["PrScaledFewer"] = I(fewer)
    M["PrScaledSame"] = I(same)
    M["PrNotCorrection"] = I(not_corr)
    # WHAT THE EIGHT TESTS SUPPORT (final referee, MAJOR 1). The preregistration reports them uncorrected; the text
    # adds, as a qualifier it labels as not preregistered, which of them survive a Bonferroni correction over all
    # eight (p < 0.05/8), and says of the rest only that no difference was detected at this sample size, never a
    # difference "the other way". No equivalence analysis was done (external audit 3: "indistinguishable" claimed
    # one), so neither the text nor a macro name may call them indistinguishable or equivalent.
    # The text names the datasets of each group, so the groups are pinned here.
    alpha = 0.05 / len(primary)
    sig = {k for k, (p, _, _) in primary.items() if p < alpha}
    sig_more = {k for k in sig if primary[k][2] > primary[k][1]}
    nodiff = set(primary) - sig
    need(sig_more == sig == {"uniform_p02", "xsector_p02", "stress_p05", "bb_refsched"},
         f"prereg: the significant datasets are now {sorted(sig)} (scaled more: {sorted(sig_more)}); the text names four")
    need(nodiff == {"uniform_p01", "hetero_p02", "phenom_p01", "phenom_p02"},
         "prereg: the four with no difference detected moved")
    diffs = [abs(primary[k][2] - primary[k][1]) for k in nodiff]
    need(primary["uniform_p01"][2] > primary["uniform_p01"][1]
         and all(primary[k][2] < primary[k][1] for k in ("hetero_p02", "phenom_p01", "phenom_p02")),
         "prereg: the point estimates of the four with no difference detected changed direction; the text says which")
    need(all(primary[k][1] - primary[k][2] == 2 for k in ("hetero_p02", "phenom_p01", "phenom_p02")),
         "prereg: the text says the scaled setting finds two fewer on hetero and on each phenomenological dataset")
    need(not_corr == 0, "prereg: the text says every miss is a heavier correction")
    need(all((M[f"Pr{RUNG_WORD[k]}TreeUnscaled"], M[f"Pr{RUNG_WORD[k]}TreeScaled"]) == ("0", "0")
             for k in primary if k not in ("stress_p05", "bb_refsched")),
         "prereg: the text says no hit outside stress and circuit is on a branching tree")
    M["PrSigMore"] = I(len(sig_more))
    M["PrNoDiff"] = I(len(nodiff))
    M["PrSigMaxP"] = BOUND(max(primary[k][0] for k in sig), "ceil", 1)        # 'p <=' : an upper bound
    M["PrNoDiffMinP"] = BOUND(min(primary[k][0] for k in nodiff), "floor", 2)  # 'p >=' : a lower bound
    M["PrNoDiffGapMin"] = I(min(diffs))
    M["PrNoDiffGapMax"] = I(max(diffs))
    # the secondary logical-error tests: which differences are significant at the UNCORRECTED 0.05 the text and the
    # Table V caption state, and -- so that the choice of criterion cannot be what makes them so -- the same set under
    # Bonferroni over the five; and that the scaled setting's count is the lower one on every code-capacity dataset
    lsig = {k for k, (p, _, _) in secondary.items() if p < 0.05}
    lsig_bonf = {k for k, (p, _, _) in secondary.items() if p < 0.05 / len(secondary)}
    need(lsig == lsig_bonf == {"uniform_p02", "xsector_p02", "stress_p05"}
         and set(secondary) - lsig == {"uniform_p01", "hetero_p02"},
         f"prereg: the significant logical-error differences are now {sorted(lsig)} (uncorrected) and "
         f"{sorted(lsig_bonf)} (Bonferroni); the text names three, the same under both")
    need(all(c2 < c1 for (_, c1, c2) in secondary.values()), "prereg: scaled no longer has fewer logical errors "
                                                              "on every code-capacity dataset")
    M["PrLogDatasets"] = I(len(secondary))
    M["PrLogSig"] = I(len(lsig))
    M["PrLogSigMaxP"] = BOUND(max(secondary[k][0] for k in lsig), "ceil", 1)


#: a dataset as the running text names it ("on the ... dataset")
RUNG_PLAIN = {"uniform_p01": r"uniform $p=0.01$", "uniform_p02": r"uniform $p=0.02$",
              "hetero_p02": "heterogeneous", "phenom_p01": r"phenomenological $p=0.01$",
              "phenom_p02": r"phenomenological $p=0.02$", "xsector_p02": "X-sector", "stress_p05": "stress",
              "bb_refsched": "circuit"}


def _nearest_rank(values: list) -> dict:
    """The quantile rule both cost artifacts state: the ceil(q N)-th smallest value."""
    v = sorted(values)
    out = {k: v[max(1, -(-int(q * 1000) * len(v) // 1000)) - 1] for k, q in (("median", 0.5), ("p90", 0.9),
                                                                                 ("p99", 0.99))}
    out["max"], out["n"] = v[-1], len(v)
    return out


def MS(seconds: float) -> str:
    """A latency in milliseconds, rounded half-even like the provenance gate: three decimals below 1 ms, one below
    10 ms, else the integer the gate derives from the artifact's seconds (int(round(1000 s))). Never two decimals,
    the shape the gate reads as a rate that must trace to an artifact."""
    ms = 1000 * seconds
    if ms < 1:
        return f"{ms:.3f}"
    return f"{ms:.1f}" if ms < 10 else I(int(round(ms)))


def _prereg_costs(M: dict, files: dict) -> None:
    """Costs of the preregistered campaign (external audit 3, paper A): production latency per receipt from the
    timed re-run (experiments/m0_prior_art/qldpc_scaling_prereg_timed_rerun.py), checking latency per receipt from
    the producer-free replay (tools/replay_qldpc_scaling_campaign.py), and the archive the replay reads."""
    import hashlib
    import re
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    pre = EV / "qldpc_scaling_prereg"
    summ = json.loads((pre / "summary.json").read_text(encoding="utf-8"))
    harness = ROOT / "experiments" / "m0_prior_art" / "qldpc_scaling_prereg_timed_rerun.py"
    replay_tool = ROOT / "tools" / "replay_qldpc_scaling_campaign.py"
    names = list(summ["datasets"])
    # ---- the archive the replay reads, and the replay's own verdict
    man_p = pre / "inputs" / "MANIFEST.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    pin = re.search(r'PINNED_MANIFEST_SHA256 = "([0-9a-f]{64})"', replay_tool.read_text(encoding="utf-8"))
    need(pin is not None and pin.group(1) == sha(man_p), "costs: the replay tool does not pin the archive's manifest")
    need(set(man["datasets"]) == set(names) and man["expected"]["nontrivial"] == sum(
        summ["datasets"][k]["nontrivial"] for k in names), "costs: the archive does not cover the campaign")
    for k in names:
        d = man["datasets"][k]
        need(d["records"]["sha256"] == summ["datasets"][k]["records_sha256"] == sha(ROOT / d["records"]["file"])
             and d["input"]["sha256"] == sha(ROOT / d["input"]["file"]), f"costs: archive binding of {k} is stale")
    chk = json.loads((pre / "costs" / "checking.json").read_text(encoding="utf-8"))
    need(chk["verdict"] == "COMPLETE REPLAY PASSED" and chk["manifest_sha256"] == sha(man_p)
         and chk["tool_sha256"] == sha(replay_tool), "costs: the replay report is not a complete pass of today's tool")
    for rel, h in chk["checkers_sha256"].items():
        need(sha(ROOT / rel) == h, f"costs: {rel} changed since the replay measured it")
    for k in names:
        s, got = summ["datasets"][k], chk["datasets"][k]
        for key, v in s.items():
            if key not in ("seconds", "records_sha256", "code_sha256", "deviation"):
                need(got.get(key) == v, f"costs: the replay's {k}.{key} is not summary.json's")
    need(chk["receipts_checked"] == man["expected"]["nontrivial"], "costs: the replay did not time every receipt")
    # ---- production, per dataset, from the re-run that reproduced the records
    prod, both, rows, docs = {}, {}, [], {}
    env, runs, milp, milp_bad, bnb, bnb_bad, records = [], [], 0, 0, 0, 0, 0
    for k in names:
        p = docs[k] = json.loads((pre / "costs" / f"production_{k}.json").read_text(encoding="utf-8"))
        s = summ["datasets"][k]
        need(p["schema"] == "oneq-qldpc-scaling-prereg-production-cost/1" and p["dataset"] == k, f"costs {k}: schema")
        need(p["timing_script_sha256"] == sha(harness), f"costs {k}: the timing harness changed since the run")
        code = s.get("code_sha256", summ["code_sha256"])
        need({r: v["sha256"] for r, v in p["code_of_record"].items()} == code,
             f"costs {k}: the re-run did not use the bytes the dataset ran under")
        rp = p["reproduction"]
        need(rp["records_byte_identical"] and rp["analysis_identical"] and not rp["differing_records"]
             and rp["records_sha256_rerun"] == s["records_sha256"] and rp["unresolved"] == 0
             and rp["records"] == s["nontrivial"], f"costs {k}: the re-run did not reproduce the run of record")
        records += rp["records"]
        sh = p["shots"]
        need(sh["shot_index"] == list(range(s["nontrivial"])), f"costs {k}: the timed shots are not the records")
        for i in range(len(sh["shot_index"])):
            need(abs(sh["production_wall_s"][i] - sum(sh["wall_s"][st][i] for st in p["production_stages"])) < 1e-9,
                 f"costs {k}: production is not the sum of its stages")
        need(p["production_stages"] == ["lp", "exactify", "prod_check", "milp", "bnb"], f"costs {k}: stages moved")
        prod[k] = _nearest_rank(sh["production_wall_s"])
        need(prod[k] == p["summary_s"]["production"], f"costs {k}: the stored production quantiles are not the rule's")
        ser = chk["series"][k]
        need(ser["shot_index"] == [i for i in range(s["nontrivial"])], f"costs {k}: the replay timed other shots")
        both[k] = _nearest_rank(ser["both_s"])
        need(both[k] == chk["summary_s"][k]["both"], f"costs {k}: the stored checking quantiles are not the rule's")
        ss = p["summary_s"]
        milp += ss["milp_calls"]
        milp_bad += ss["milp_not_optimal"]
        bnb += ss["bnb_calls"]
        bnb_bad += ss["bnb_without_tree"]
        need(ss["bnb_calls"] == s["receipt_kinds"]["tree"], f"costs {k}: a tree was attempted on a flat-receipt shot")
        env.append(p["environment"])
        runs.append(p["run"])
        rows.append(rf"{RUNG_LABEL[k]} & {I(s['nontrivial'])} & "
                    + " & ".join(MS(prod[k][q]) for q in ("median", "p90", "p99", "max")) + " & "
                    + " & ".join(MS(both[k][q]) for q in ("median", "p90", "p99", "max")) + r" \\")
    need(milp_bad == 0 and bnb_bad == 0, "costs: the text says every MILP returned an optimal status and every "
                                         "tree closed inside its caps")
    lim = {json.dumps(docs[k]["limits"], sort_keys=True) for k in names}
    need(len(lim) == 1, "costs: the datasets ran under different limits")
    lim = json.loads(lim.pop())
    need(lim["memory_cap"] is None and lim["dataset_wall_cap"] is None, "costs: the text says no memory or wall cap")
    need(len({e["cpu_name"] for e in env}) == 1 and len({e["logical_cpus"] for e in env}) == 1
         and env[0]["cpu_name"], "costs: the datasets ran on different machines, or the processor is unnamed")
    M["PcRecords"] = I(records)
    M["PcRerunUnresolved"] = I(0)
    M["PcMilpCalls"] = I(milp)
    M["PcBnbCalls"] = I(bnb)
    M["PcMilpLimit"] = I(int(lim["milp_time_limit_s"]))
    M["PcBnbNodes"] = I(lim["bnb_max_nodes"])
    M["PcBnbDepth"] = I(lim["bnb_max_depth"])
    M["PcTreeBTimeout"] = I(lim["tree_checker_b_timeout_s"])
    med = [prod[k]["median"] for k in names]
    M["PcProdMedLo"], M["PcProdMedHi"] = MS(min(med)), MS(max(med))
    cmed = [both[k]["median"] for k in names]
    M["PcCheckMedLo"], M["PcCheckMedHi"] = MS(min(cmed)), MS(max(cmed))
    # the split the text gives: the production checker alone, against the second checker (the reference flat checker
    # reads the check matrix as a dense array), recomputed from the series
    pc = [_nearest_rank(chk["series"][k]["production_checker_s"])["median"] for k in names]
    sc = [_nearest_rank(chk["series"][k]["second_checker_s"])["median"] for k in names]
    need(all(b > a for a, b in zip(pc, sc)), "costs: the text says the second checker is the larger share everywhere")
    M["PcProdCheckerMedLo"], M["PcProdCheckerMedHi"] = MS(min(pc)), MS(max(pc))
    worst = max(names, key=lambda k: prod[k]["max"])
    M["PcProdMaxMs"] = MS(prod[worst]["max"])
    M["PcProdMaxWhere"] = RUNG_PLAIN[worst]
    sw = docs[worst]["shots"]
    slowest = max(range(len(sw["production_wall_s"])), key=lambda i: sw["production_wall_s"][i])
    need(sw["production_wall_s"][slowest] == prod[worst]["max"], "costs: the slowest shot is not the maximum")
    M["PcProdMaxKind"] = {"flat": "flat receipt", "tree": "branch-dual tree"}[sw["receipt_kind"][slowest]]
    # where that time went: the production stage holding the largest share of the slowest shot, and the share
    stage = max(docs[worst]["production_stages"], key=lambda st: sw["wall_s"][st][slowest])
    share = sw["wall_s"][stage][slowest] / sw["production_wall_s"][slowest]
    need(share >= 0.5, "costs: the text names one stage as most of the slowest receipt's time")
    M["PcProdMaxS"] = f"{prod[worst]['max']:.1f}"
    M["PcProdMaxShare"] = str(math.floor(100 * share))
    M["PcProdMaxStage"] = {"lp": "its linear program", "bnb": "the branch-dual search over its node relaxations",
                           "milp": "the MILP", "exactify": "rationalising its dual",
                           "prod_check": "the production checker's acceptance"}[stage]
    need(sw["receipt_kind"][slowest] == "tree", "costs: the text gives the slowest receipt's tree size; it is flat")
    nodes = [t["nodes"] for t in docs[worst]["tree_receipts"] if t["shot_index"] == sw["shot_index"][slowest]]
    need(len(nodes) == 1, "costs: the slowest tree is not in the artifact's tree list")
    M["PcProdMaxNodes"] = I(nodes[0])
    cworst = max(names, key=lambda k: both[k]["max"])
    M["PcCheckMaxMs"] = MS(both[cworst]["max"])
    M["PcCheckMaxWhere"] = RUNG_PLAIN[cworst]
    # decoding both settings, per shot: its largest per-dataset median (BP-OSD's cost, not the receipt's)
    dec = [_nearest_rank([a + b for a, b in zip(docs[k]["shots"]["wall_s"]["decode_1.0"],
                                                docs[k]["shots"]["wall_s"]["decode_0.625"])])["median"] for k in names]
    M["PcDecodeMedMaxMs"] = MS(max(dec))
    M["PcCpuName"] = re.sub(r"\((R|TM)\)", "", env[0]["cpu_name"]).replace("  ", " ").strip()
    M["PcLogicalCpus"] = I(env[0]["logical_cpus"])
    cw = [r["cpu_over_wall"] for r in runs]
    need(all(0.5 < x <= 1.05 for x in cw), "costs: a run used more or less than about one core; the text says one")
    M["PcCpuWallLo"], M["PcCpuWallHi"] = f"{min(cw):.2f}", f"{max(cw):.2f}"
    box = [e["box_cpu_utilisation_during_run"] for e in env]
    M["PcBoxLo"], M["PcBoxHi"] = str(math.floor(100 * min(box))), str(math.ceil(100 * max(box)))
    peak = math.ceil(max(e["peak_working_set_bytes"] for e in env) / 2 ** 20)
    need(0 < peak < 1000, "costs: the text prints the peak working set in MB as a plain integer below 1,000")
    M["PcPeakMB"] = I(peak)
    M["PcThreadsMax"] = I(max(e["process_threads_max"] for e in env))
    lines = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{production (ms)} & \multicolumn{4}{c}{checking, both checkers (ms)} \\",
             r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
             r"dataset & $N$ & median & p90 & p99 & max & median & p90 & p99 & max \\", r"\midrule",
             *rows, r"\bottomrule", r"\end{tabular}"]
    files["tab_prereg_costs.tex"] = "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in build().items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        _w = OUT / name
        print("wrote", _w.relative_to(ROOT) if _w.is_relative_to(ROOT) else _w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
