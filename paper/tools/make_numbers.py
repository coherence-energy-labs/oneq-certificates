r"""Every quantitative macro and table in paper 1, generated from artifacts.

The manuscript never types a measured number. It writes \SimCertifiedPct,
and this script defines that macro by reading the artifact that measured it.
A transcription error therefore cannot exist; a stale number shows up as a
diff in the generated file, which tests/test_paper_build_tools.py compares
against a fresh run.

    python paper/tools/make_numbers.py

Writes paper/certifying-decoders/generated/{numbers,tab_*}.tex.
"""

from __future__ import annotations

import json
import math
import pathlib
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
OUT = ROOT / "paper" / "certifying-decoders" / "generated"


def J(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def I(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def P(x: float, k: int = 2) -> str:
    return f"{100 * x:.{k}f}"


def F(x: float, k: int = 2) -> str:
    return f"{x:.{k}f}"


def enclose(lo: float, hi: float, k: int) -> tuple[str, str]:
    """A printed range must CONTAIN the data: floor the low end, ceiling the high."""
    f = 10 ** k
    return f"{math.floor(lo * f) / f:.{k}f}", f"{math.ceil(hi * f) / f:.{k}f}"


def SCI(x: float, k: int = 1) -> str:
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, k) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.{k}f}\times 10^{{{e}}}}}"


def build() -> dict[str, str]:
    files: dict[str, str] = {}
    M: dict[str, str] = {}

    # ------------------------------------------------ simulation at scale
    ms = J("million_shot_gate_d5_merged.json")
    M["SimShots"] = I(ms["shots"])
    M["SimCertified"] = I(ms["certified"])
    M["SimCertifiedPct"] = P(ms["certified_rate"])
    M["SimUncertifiedPct"] = P(1 - ms["certified_rate"])
    M["SimForgeries"] = I(ms["forgeries_attempted"])
    M["SimForgeriesAccepted"] = I(ms["forgeries_accepted"])
    M["SimFailedChunks"] = I(ms["failed_chunks"])
    M["SimShotsLostToChunks"] = I(ms["shots_lost_to_failed_chunks"])
    ug = ms["uncertified_gap_merged"]
    M["SimGapMean"] = F(ug["nweighted_mean"])
    M["SimGapMax"] = F(ug["max_of_maxes"])
    med = sorted(ug["per_run_medians"].values())
    M["SimGapMedianLo"], M["SimGapMedianHi"] = enclose(med[0], med[-1], 2)
    dvc = J("decode_vs_certify.json")
    row = dvc["rows"][0]
    M["SimDetectors"] = I(row["nodes"])
    M["SimEdges"] = I(row["edges"])

    # ------------------------------------------------ hardware (Willow)
    h = J("google_generation_history.json")
    for g, name in (("1", "One"), ("2", "Two"), ("3", "Three"), ("4", "Four")):
        M[f"HwGen{name}Pct"] = P(h["generations"][g]["rate"])
    f = h["final_generation"]
    M["HwFinalNontrivial"] = I(f["nontrivial"])
    M["HwFinalCertified"] = I(f["certified"])
    M["HwFinalPct"] = P(f["rate"])
    M["HwForgeries"] = I(f["forgeries_attempted_gen4_gen5_gen6"])
    M["HwGenOneNontrivial"] = I(h["generations"]["1"]["nontrivial"])
    M["HwGenTwoMinConfig"] = I(h["generations"]["2"]["min_config_nontrivial"])
    M["HwGenFourForgeries"] = I(h["generations"]["4"]["forgeries_attempted"])
    g4 = h["generations"]["4"]
    M["HwGenFourNontrivial"] = I(g4["nontrivial"])
    for basis in ("X", "Z"):
        g5 = h["generations"]["5"]["per_config"][f"d7_at_q6_7_{basis}_r30"]
        g6 = h["generations"]["6"]["per_config"][f"d7_at_q6_7_{basis}_r30"]
        M[f"HwGenFive{basis}Pct"] = P(g5["rate"])
        M[f"HwGenSix{basis}Certified"] = I(g6["certified"])
        M[f"HwGenSix{basis}Nontrivial"] = I(g6["nontrivial"])
        M[f"HwGenSix{basis}Pct"] = P(g6["rate"], 3)
        c6 = J(f"corpus_gen6_d7{basis}30.json")
        pc = c6["per_config"]
        pc = list(pc.values())[0] if isinstance(pc, dict) else pc[0]
        M[f"HwGenSix{basis}ExactRung"] = I(pc["rung_histogram"].get("5", 0))
        M[f"HwGenSix{basis}ResidualGap"] = F(c6["uncertified_gap"]["max"])
        M[f"HwGenSix{basis}Forgeries"] = I(c6["totals"]["forgeries_attempted"])
    dc = J("derived_claims.json")
    M["TotalShots"] = I(dc["shots_total"])
    M["TotalForgeries"] = I(dc["forgeries_attempted_total"])
    M["HwCoreSecX"] = F(dc["gen6_core_seconds_per_shot_X"])
    M["HwCoreSecZ"] = F(dc["gen6_core_seconds_per_shot_Z"])
    M["HwWallSecX"] = F(dc["gen6_wall_seconds_per_shot_X"])
    M["HwWallSecZ"] = F(dc["gen6_wall_seconds_per_shot_Z"])

    # generation table
    configs = sorted(h["generations"]["1"]["per_config"],
                     key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_")[-1][1:]), c.split("_")[-2]))
    by = {}
    for c in configs:
        d, basis, r = int(c.split("_")[0][1:]), c.split("_")[-2], int(c.split("_")[-1][1:])
        by.setdefault((d, r), {})[basis] = c
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"configuration & gen.~1 & gen.~2 & gen.~3 & gen.~4 \\",
             r"\midrule"]
    for (d, r), bases in sorted(by.items()):
        cells = []
        for g in ("1", "2", "3", "4"):
            vals = [h["generations"][g]["per_config"][bases[b]]["rate"] for b in ("X", "Z")]
            cells.append(" / ".join(P(v) for v in vals))
        lines.append(rf"$d={d}$, {r} rounds & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("aggregate & " + " & ".join(
        P(h["generations"][g]["rate"]) for g in ("1", "2", "3", "4")) + r" \\")
    lines.append("nontrivial shots & " + " & ".join(
        I(h["generations"][g]["nontrivial"]) for g in ("1", "2", "3", "4")) + r" \\")
    lines.append("per configuration & " + " & ".join(
        f"{I(h['generations'][g]['min_config_nontrivial'])}--{I(h['generations'][g]['max_config_nontrivial'])}"
        for g in ("1", "2", "3", "4")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_generations.tex"] = "\n".join(lines) + "\n"

    # ------------------------------------------------ oracle audit
    orc = [J(f"exact_oracle_audit_d{d}.json") for d in (3, 5, 7)]
    M["OracleShots"] = I(sum(o["nontrivial"] for o in orc))
    M["OracleAgree"] = I(sum(o["decoder_optimal"] for o in orc))
    M["OracleBeaten"] = I(sum(o["decoder_beaten"] for o in orc))
    for o, name in zip(orc, ("Three", "Five", "Seven")):
        M[f"OracleShots{name}"] = I(o["nontrivial"])
        M[f"OracleRounds{name}"] = I(o["rounds"])

    # ------------------------------------------------ IBM live
    ib = J("ibm_live/d9nt2imij12s73fu9ihg.certified.json")
    M["IbmNontrivial"] = I(ib["nontrivial"])
    M["IbmCertified"] = I(ib["certified"])
    M["IbmMsPerShot"] = F(ib["cert_ms_per_shot"])
    M["IbmPreregPrefix"] = ib["prereg_sha256"][:8]
    M["IbmBackend"] = ib["backend"].replace("_", r"\_")

    # ------------------------------------------------ seeded faults
    mut = [J(f"decoder_mutation_gate_d{d}.json") for d in (3, 5, 7)]
    M["FaultsAll"] = I(sum(m["all_defective"]["n"] for m in mut))
    M["FaultsAllSyndromeCaught"] = I(sum(m["all_defective"]["syndrome_caught"] for m in mut))
    M["FaultsAllCertCaught"] = I(sum(m["all_defective"]["certificate_caught"] for m in mut))
    allsyn = sum(m["all_defective"]["syndrome_caught"] for m in mut) / sum(m["all_defective"]["n"] for m in mut)
    M["FaultsAllSyndromePct"] = P(allsyn, 1)
    M["FaultsStealth"] = I(sum(m["stealth_only"]["n"] for m in mut))
    M["FaultsStealthSyndromeCaught"] = I(sum(m["stealth_only"]["syndrome_caught"] for m in mut))
    M["FaultsStealthCertCaught"] = I(sum(m["stealth_only"]["certificate_caught"] for m in mut))
    ctrl = [m["negative_control_M5"] for m in mut if m.get("negative_control_M5")]
    M["FaultsControl"] = I(sum(c["n"] for c in ctrl))
    M["FaultsControlFlagged"] = I(sum(c["certificate_caught"] for c in ctrl))
    M["FaultCorpus"] = I(dc["seeded_fault_corpus_released"])

    # ------------------------------------------------ stale calibration
    st = {d: J(f"stale_calibration_d{d}.json") for d in (3, 5, 7)}
    tot = {}
    for d, doc in st.items():
        for r in doc["rows"]:
            t = tot.setdefault(r["drift_factor"], [0, 0, 0])
            t[0] += r["genuinely_suboptimal"]
            t[1] += r["syndrome_check_caught"]
            t[2] += r["certificate_caught"]
    swept = [k for k in sorted(tot) if k != 1.0]
    M["StaleSuboptimalTotal"] = I(sum(tot[k][0] for k in swept))
    M["StaleSyndromeCaughtTotal"] = I(sum(tot[k][1] for k in swept))
    M["StaleCertCaughtTotal"] = I(sum(tot[k][2] for k in swept))
    M["StaleControlSuboptimal"] = I(tot[1.0][0])
    for k, name in ((1.5, "OneFive"), (2.0, "Two"), (3.0, "Three")):
        M[f"StaleSuboptimal{name}"] = I(tot[k][0])
    for k, name in ((1.5, "OneFive"), (3.0, "Three")):
        ex = [next(r for r in st[d]["rows"] if r["drift_factor"] == k)["mean_excess_weight"] for d in (3, 5, 7)]
        M[f"StaleExcessLo{name}"], M[f"StaleExcessHi{name}"] = enclose(min(ex), max(ex), 2)
    # The caption says every correction was syndrome-consistent; hold it.
    for d, doc in st.items():
        for r in doc["rows"]:
            if r["syndrome_consistent"] != r["shots"]:
                raise SystemExit(f"stale calibration d={d} drift {r['drift_factor']}: "
                                 f"not every correction is syndrome-consistent")
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & & sub- & \multicolumn{2}{c}{caught by} & logical errors \\",
             r" & shots & optimal & syndrome & certificate & true / stale \\", r"\midrule"]
    for d in (3, 5, 7):
        r = next(x for x in st[d]["rows"] if x["drift_factor"] == 3.0)
        lines.append(rf"$d={d}$ & {I(r['shots'])} & "
                     rf"{I(r['genuinely_suboptimal'])} & {I(r['syndrome_check_caught'])} & "
                     rf"{I(r['certificate_caught'])} & {r['logical_errors_true_model']} / "
                     rf"{r['logical_errors_stale_model']} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_stale.tex"] = "\n".join(lines) + "\n"
    d3 = next(x for x in st[3]["rows"] if x["drift_factor"] == 3.0)
    M["StaleDThreeLogicalTrue"] = I(d3["logical_errors_true_model"])
    M["StaleDThreeSuboptimal"] = I(d3["genuinely_suboptimal"])

    # ------------------------------------------------ gap distribution
    gd = J("gap_distribution.json")
    gx = J("gap_distribution_exact_rule.json")
    if any(r["rounds"] != r["distance"] for r in gd["rows"]):
        raise SystemExit("gap distribution: the text says rounds equal distance")
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & certified & BP-OSD & \multicolumn{3}{c}{gap} \\",
             r" & shots & exact & median & p99 & max \\", r"\midrule"]
    fz = []
    for r in gd["rows"]:
        b = r["decoders"]["bp_osd"]["gap_vs_proven_optimum"]
        fz.append(b["frac_zero"])
        lines.append(rf"$d={r['distance']}$ & {I(r['mwpm_certified_exact'])} & "
                     rf"{P(b['frac_zero'], 1)}\% & {F(b['median'], 1)} & {F(b['p99'], 1)} & {F(b['max'], 1)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_gaps.tex"] = "\n".join(lines) + "\n"
    fzx = [r["decoders"]["bp_osd"]["gap_vs_proven_optimum"]["frac_zero"] for r in gx["rows"]]
    M["GapShotsPerDistance"] = I(gd["rows"][0]["shots"])
    M["GapBposdExactLo"] = P(min(fz), 0) if False else f"{math.floor(100 * min(fz))}"
    M["GapBposdExactHi"] = f"{math.floor(100 * max(fz + fzx))}"
    M["GapBposdMax"] = f"{max(r['decoders']['bp_osd']['gap_vs_proven_optimum']['max'] for r in gd['rows']):.0f}"
    bm = [r["decoders"]["belief_matching"]["gap_vs_proven_optimum"]["p99"] for r in gd["rows"]]
    bo = [r["decoders"]["bp_osd"]["gap_vs_proven_optimum"]["p99"] for r in gd["rows"]]
    M["GapBmPNinetyNine"] = " / ".join(F(x, 1) for x in bm)
    M["GapBposdPNinetyNine"] = " / ".join(F(x, 0) for x in bo)
    M["GapExactRuleCertified"] = " / ".join(I(r["mwpm_certified_exact"]) for r in gx["rows"])
    M["GapExactRuleBposdExact"] = " / ".join(P(x, 1) for x in fzx)

    # ------------------------------------------------ cost and scaling
    M["CertifyMsPerShot"] = F(dvc["certify_ms_per_shot"])
    M["CostRatioBatch"] = I(dc["cost_ratio_batch"])
    M["CostRatioPerShot"] = I(dc["cost_ratio_per_shot"])
    M["DecodeUsBatch"] = f"{dvc['rows'][0]['decode_batch_us']:.0f}"
    M["DecodeUsPerShot"] = f"{dvc['rows'][0]['decode_per_shot_us']:.0f}"
    cs = J("checker_scaling.json")
    M["CheckExponent"] = F(cs["fit_time_vs_nnz"]["exponent"], 3)
    M["CheckRSquared"] = F(cs["fit_time_vs_nnz"]["r_squared"], 3)
    M["CheckSupportExponent"] = F(cs["fit_support_vs_nnz"]["exponent"], 2)
    M["CheckWorkExponent"] = F(cs["fit_time_vs_work"]["exponent"], 2)
    gl = J("growth_vs_lp.json")["rows"]
    M["GrowthSpeedupLo"], M["GrowthSpeedupHi"] = enclose(min(r["speedup"] for r in gl), max(r["speedup"] for r in gl), 0)
    g7 = next(r for r in gl if r["distance"] == 7)
    M["GrowthDSevenCertified"] = f"{g7['certified_growth']}/{g7['nontrivial']}"
    M["LpDSevenCertified"] = f"{g7['certified_lp']}/{g7['nontrivial']}"

    # ------------------------------------------------ conformance
    cv = json.loads((ROOT / "conformance" / "cert_vectors.json").read_text(encoding="utf-8"))
    M["ConfVectors"] = I(cv["counts"]["total"])
    M["ConfRefusals"] = I(cv["counts"]["expected_refuse"])
    M["ConfVectorsMinusOne"] = I(cv["counts"]["total"] - 1)

    # ------------------------------------------------ certification semantics audit
    au = J("certification_semantics_audit.json")
    rows = {r["config"]["name"]: r for r in au["rows"]}
    label = {
        "million_shot_d5_r25": r"simulation, $d=5$, 25 rounds",
        "gap_distribution_d3": r"simulation, $d=3$, 3 rounds",
        "gap_distribution_d5": r"simulation, $d=5$, 5 rounds",
        "gap_distribution_d7": r"simulation, $d=7$, 7 rounds",
        "willow_gen4_d5_X_r13": r"Willow, $d=5$, 13 rounds, X",
        "willow_gen6_d7_X_r30": r"Willow, $d=7$, 30 rounds, X",
    }
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"configuration (code commit) & sample & accepted & largest proven & exact checker, & current producer \\",
             r" & & then & slack $\varepsilon$ & same certificate & (with escalation) \\", r"\midrule"]
    eps_all, acc_all, same_all, prod_all = [], 0, 0, 0
    for name in label:
        r = rows[name]
        eps_all.append(Fraction(r["eps_exact_max"]))
        acc_all += r["accepted_by_pinned_code"]
        same_all += r["head_exact_checker_accepts_same_certificate"]
        prod_all += r["head_producer_with_escalation_certifies"]
        lines.append(
            rf"{label[name]} (\texttt{{{r['config']['commit']}}}) & {I(r['nontrivial_sampled'])} & "
            rf"{I(r['accepted_by_pinned_code'])} & {SCI(r['eps_max'])} & "
            rf"{I(r['head_exact_checker_accepts_same_certificate'])} & "
            rf"{I(r['head_producer_with_escalation_certifies'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_semantics.tex"] = "\n".join(lines) + "\n"
    M["AuditAccepted"] = I(acc_all)
    M["AuditEpsMax"] = SCI(float(max(eps_all)))
    M["AuditExactSame"] = I(same_all)
    M["AuditExactSamePct"] = P(same_all / acc_all, 0)
    M["AuditInadmissible"] = I(sum(sum(r["inadmissible_certificates"].values()) for r in rows.values()))
    M["AuditHeadProducer"] = I(prod_all)
    rule_terms = [1e-9 * r["rule_level_bound"]["largest_dual_mass_in_sample"]
                  / r["rule_level_bound"]["min_edge_weight"] for r in rows.values()]
    M["AuditRuleTerm"] = SCI(max(rule_terms))
    orc_rows = [r["supplementary_oracle_not_exact"] for r in rows.values()
                if r.get("supplementary_oracle_not_exact")]
    if not orc_rows:
        raise SystemExit("audit artifact carries no supplementary oracle rows; "
                         "re-run it with --oracle")
    M["AuditOracleN"] = I(sum(o["n"] for o in orc_rows))
    M["AuditOracleEqual"] = I(sum(o["exactly_equal"] for o in orc_rows))
    M["AuditOracleMaxDelta"] = SCI(max(o["max_primal_minus_oracle"] for o in orc_rows))
    if min(o["min_primal_minus_oracle"] for o in orc_rows) < 0:
        raise SystemExit("the oracle found a HEAVIER matching than the decoder "
                         "somewhere: the text's reading of the oracle no longer holds")

    body = ["% GENERATED by paper/tools/make_numbers.py from evidence/ -- do not edit.",
            "% Every value below is read from an artifact at generation time."]
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
