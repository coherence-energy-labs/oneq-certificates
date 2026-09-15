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
    # the final aggregate mixes two producer generations; report each part (audit M10)
    fin = f["per_config"]
    g6c = [c for c in fin if c.startswith("d7") and c.endswith("_r30")]
    g4c = [c for c in fin if c not in g6c]
    if len(g6c) != 2 or len(g4c) != 10:
        raise SystemExit("final generation: expected ten generation-4 and two generation-6 configurations")
    for part, cs in (("GenFour", g4c), ("GenSix", g6c)):
        n = sum(fin[c]["nontrivial"] for c in cs)
        k = sum(fin[c]["certified"] for c in cs)
        M[f"HwFinal{part}Nontrivial"] = I(n)
        M[f"HwFinal{part}Certified"] = I(k)
        M[f"HwFinal{part}Pct"] = P(k / n, 3 if part == "GenSix" else 2)
    # the two shots generation 6 left open (audit B4), located and re-examined
    wr = J("willow_residual_shots.json")["shots"]
    if any(s["pymatching_minus_optimum"] != 0.0 or not s["gap_reproduced"] for s in wr.values()):
        raise SystemExit("willow residual shots: the text says both corrections are exactly optimal")
    if {s["head_exact_tier"]["exit"]["reason"] for s in wr.values()} != {"max_rounds_exhausted"}:
        raise SystemExit("willow residual shots: the text says the exact tier stopped at its round cap")
    M["HwResidualRounds"] = I(wr["X"]["head_exact_tier"]["exit"]["rounds"])
    M["HwResidualDetectorsX"] = I(wr["X"]["detectors_fired"])
    M["HwResidualDetectorsZ"] = I(wr["Z"]["detectors_fired"])
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

    # ------------------------------------------------ epsilon rerun: sound checker, every receipt retained
    PENDING = r"\textbf{[pending]}"

    def pct_floor(x, k=2):
        """A rate printed from below: 99.998 must not read 100.00."""
        return f"{math.floor(100 * x * 10 ** k) / 10 ** k:.{k}f}"

    def pct_ceil(x, k=2):
        return f"{math.ceil(100 * x * 10 ** k) / 10 ** k:.{k}f}"

    def eps_summary(name):
        p = EV / "epsilon_rerun" / name / "summary.json"
        if not p.exists():
            return None
        s = json.loads(p.read_text(encoding="utf-8"))
        lo, hi = s["spec"]["seeds"]
        if s["chunks"] != hi - lo + 1:
            return None                                   # a pilot merge is not a result
        if Fraction(s["eps_max"]) != Fraction(1, 10**6):
            raise SystemExit(f"{name}: eps_max is not 1e-6")
        if s["forgeries_accepted"] or s["battery_attacks_not_false"]:
            raise SystemExit(f"{name}: a forgery was accepted or an attack was not false")
        odd = set(s["status"]) - {"TRIVIAL", "EXACT_OPTIMAL", "EPSILON_OPTIMAL", "NOT_PROVEN"}
        if odd:
            raise SystemExit(f"{name}: producer certificates reached {sorted(odd)}; investigate before writing")
        # The abstract says every receipt replays under two verifiers: no complete
        # replay artifact, no typeset number.
        rp = EV / "epsilon_rerun" / f"replay_{name}.json"
        if not rp.exists():
            return None
        rep = json.loads(rp.read_text(encoding="utf-8"))
        res = [r for r in rep["results"] if r["config"] == name]
        inst = json.loads((EV / "instances" / f"matching_{name}.json").read_text(encoding="utf-8"))
        if (rep["kind"] != "COMPLETE" or len(res) != 1 or res[0]["receipts_rechecked"] != s["nontrivial"]
                or res[0]["instance_sha256"] != inst["edges_sha256"]
                or res[0]["battery"].get("substituted_graph_accepted", 0)):
            raise SystemExit(f"{name}: the replay artifact does not cover this run completely")
        return s

    es = eps_summary("sim_d5_r25")
    sim_keys = ("EpsSimShots", "EpsSimNontrivial", "EpsSimAccepted", "EpsSimAcceptedPct", "EpsSimExact",
                "EpsSimNotProven", "EpsSimNotProvenPct", "EpsSimLargestEps", "EpsSimForgeries",
                "EpsSimForgeryClasses", "EpsSimChunkRetries")
    if es is None:
        M.update({k: PENDING for k in sim_keys})
    else:
        if es["shots"] != ms["shots"]:
            raise SystemExit("epsilon rerun: the simulation does not cover the original seed plan")
        nt = es["nontrivial"]
        M["EpsSimShots"] = I(es["shots"])
        M["EpsSimNontrivial"] = I(nt)
        M["EpsSimAccepted"] = I(es["accepted"])
        M["EpsSimAcceptedPct"] = pct_floor(es["accepted"] / nt)
        M["EpsSimExact"] = I(es["status"].get("EXACT_OPTIMAL", 0))
        M["EpsSimNotProven"] = I(es["status"].get("NOT_PROVEN", 0))
        M["EpsSimNotProvenPct"] = pct_ceil(es["status"].get("NOT_PROVEN", 0) / nt)
        M["EpsSimLargestEps"] = SCI(es["largest_accepted_epsilon_float"])
        M["EpsSimForgeries"] = I(es["forgeries_attempted"])
        M["EpsSimForgeryClasses"] = I(len(es["forgeries_by_class"]))
        M["EpsSimChunkRetries"] = I(es["chunk_failures_retried"])

    wcfg = list(f["per_config"])
    wsum = {c: eps_summary(f"willow_{c}") for c in wcfg}
    lines = [r"\begin{tabular}{@{}llrcccrc@{}}", r"\toprule",
             r" & & & historical & \multicolumn{2}{c}{$\varepsilon$-checker} & not & largest \\",
             r"configuration & ladder & $N$ & rule (\%) & proven (\%) & exact & proven & proven $\varepsilon$ \\",
             r"\midrule"]
    order = sorted(wcfg, key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_")[-1][1:]), c.split("_")[-2]))
    for c in order:
        d, basis, r = int(c.split("_")[0][1:]), c.split("_")[-2], int(c.split("_")[-1][1:])
        ladder = "full" if c in g6c else "base"
        hist = fin[c]
        s = wsum[c]
        if s is None:
            cells = [PENDING] * 4
        else:
            if s["nontrivial"] != hist["nontrivial"]:
                raise SystemExit(f"willow_{c}: the rerun's nontrivial shots differ from the run of record's")
            cells = [pct_floor(s["accepted"] / s["nontrivial"]), I(s["status"].get("EXACT_OPTIMAL", 0)),
                     I(s["status"].get("NOT_PROVEN", 0)), SCI(s["largest_accepted_epsilon_float"])]
        lines.append(rf"$d={d}$, {r} rounds, {basis} & {ladder} & {I(hist['nontrivial'])} & "
                     rf"{pct_floor(hist['rate'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_epsilon_willow.tex"] = "\n".join(lines) + "\n"

    hw_keys = ("EpsHwNontrivial", "EpsHwAccepted", "EpsHwAcceptedPct", "EpsHwExact", "EpsHwNotProven",
               "EpsHwLargestEps", "EpsHwForgeries", "EpsHwGenFourPct", "EpsHwGenSixPct",
               "EpsForgeriesTotal", "EpsLargestEps")
    if es is None or any(v is None for v in wsum.values()):
        M.update({k: PENDING for k in hw_keys})
    else:
        def agg(cs, key):
            return sum(wsum[c][key] for c in cs)

        def status(cs, key):
            return sum(wsum[c]["status"].get(key, 0) for c in cs)

        M["EpsHwNontrivial"] = I(agg(wcfg, "nontrivial"))
        M["EpsHwAccepted"] = I(agg(wcfg, "accepted"))
        M["EpsHwAcceptedPct"] = pct_floor(agg(wcfg, "accepted") / agg(wcfg, "nontrivial"))
        M["EpsHwExact"] = I(status(wcfg, "EXACT_OPTIMAL"))
        M["EpsHwNotProven"] = I(status(wcfg, "NOT_PROVEN"))
        hw_eps = max(Fraction(wsum[c]["largest_accepted_epsilon"]) for c in wcfg)
        M["EpsHwLargestEps"] = SCI(float(hw_eps))
        M["EpsHwForgeries"] = I(agg(wcfg, "forgeries_attempted"))
        M["EpsHwGenFourPct"] = pct_floor(agg(g4c, "accepted") / agg(g4c, "nontrivial"))
        M["EpsHwGenSixPct"] = pct_floor(agg(g6c, "accepted") / agg(g6c, "nontrivial"), 3)
        M["EpsForgeriesTotal"] = I(agg(wcfg, "forgeries_attempted") + es["forgeries_attempted"])
        M["EpsLargestEps"] = SCI(float(max(hw_eps, Fraction(es["largest_accepted_epsilon"]))))

    # ------------------------------------------------ what the replays' attacks did, over every run
    names_all = ["sim_d5_r25"] + [f"willow_{c}" for c in wcfg]
    battery = {}
    for nm in names_all:
        rp = EV / "epsilon_rerun" / f"replay_{nm}.json"
        if rp.exists():
            battery[nm] = json.loads(rp.read_text(encoding="utf-8"))["results"][0]["battery"]
    bat_keys = ("EpsBoundaryRefused", "EpsBoundaryInsideAccepted", "EpsSubstitutedRefused")
    if es is None or any(v is None for v in wsum.values()) or len(battery) != len(names_all):
        M.update({k: PENDING for k in bat_keys})
    else:
        if any(b.get("substituted_graph_accepted", 0) for b in battery.values()):
            raise SystemExit("replay battery: a receipt was accepted against a substituted graph")
        if any(not b.get("boundary_attempts", 0) for b in battery.values()):
            raise SystemExit("replay battery: a run was replayed without boundary attacks")
        M["EpsBoundaryRefused"] = I(sum(b["boundary_just_outside_refused"] for b in battery.values()))
        M["EpsBoundaryInsideAccepted"] = I(sum(b.get("boundary_just_inside_accepted", 0) for b in battery.values()))
        M["EpsSubstitutedRefused"] = I(sum(b.get("substituted_graph_refused", 0) for b in battery.values()))

    # the rerun against the original runs on the same shots, configuration by configuration
    cmp_keys = ("EpsFewerConfigs", "EpsMoreConfigs")
    if any(v is None for v in wsum.values()):
        M.update({k: PENDING for k in cmp_keys})
    else:
        M["EpsFewerConfigs"] = I(sum(1 for c in wcfg if wsum[c]["accepted"] < fin[c]["certified"]))
        M["EpsMoreConfigs"] = I(sum(1 for c in wcfg if wsum[c]["accepted"] > fin[c]["certified"]))

    # ------------------------------------------------ mutation score of the epsilon checker's tests
    mu = J("mutation_score/matching_cert_epsilon.json")

    def sha(p):   # the pins' domain: CRLF normalised to LF, the bytes git stores (tools/mutation_score.py)
        return __import__("hashlib").sha256(p.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()

    if mu.get("digest_domain") != "sha256-lf":
        raise SystemExit("mutation score: its pins are not in the sha256-lf domain; re-record or re-run it")
    if sha(ROOT / mu["target"]) != mu["target_sha256"] or any(
            sha(ROOT / t) != h for t, h in mu["tests"].items()):
        raise SystemExit("mutation score: the checker or its tests changed since the score was measured; re-run it")
    if mu["timeout"] or any(not s["equivalent"] for s in mu["survivors"]):
        raise SystemExit("mutation score: a non-equivalent mutant survives; the text says none does")
    M["MutEpsRun"] = I(mu["mutants_run"])
    M["MutEpsKilled"] = I(mu["killed"])
    M["MutEpsEquivalent"] = I(mu["equivalent_declared"])

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

    # ------------------------------------------------ detection outcomes (sound epsilon checker)
    det = {d: J(f"detection_outcomes/d{d}.json") for d in (3, 5, 7)}
    for d, doc in det.items():
        if Fraction(doc["eps_max"]) != Fraction(1, 10**6):
            raise SystemExit(f"detection d={d}: eps_max is not 1e-6")
        if doc["seeded_faults"]["oracle_failures"]:
            raise SystemExit(f"detection d={d}: the exact oracle failed on some shots")

    def dsum(rows, key):
        return sum(r[key] for r in rows)

    clean = [doc["seeded_faults"]["clean_output_base_ladder"] for doc in det.values()]
    clean_full = [doc["seeded_faults"]["clean_output_full_ladder"] for doc in det.values()]
    muts = [(name, t) for doc in det.values() for name, t in doc["seeded_faults"]["per_mutant"].items()]
    stealth = [t for name, t in muts if "STEALTH" in name]
    defect = [t for name, t in muts if not name.startswith("M5")]
    ctrl = [t for name, t in muts if name.startswith("M5")]
    if dsum(defect, "missed") or dsum([t for _n, t in muts], "missed"):
        raise SystemExit("detection: a defective correction was certified")
    if dsum(stealth, "invalid") or dsum(stealth, "not_proven_on_defective"):
        raise SystemExit("detection: the text says every stealth fault is syndrome-valid and proven suboptimal")
    M["DetCleanN"] = I(dsum(clean, "n"))
    M["DetCleanFalseAlarms"] = I(dsum(clean, "false_alarms"))
    M["DetCleanFalseAlarmsFull"] = I(dsum(clean_full, "false_alarms"))
    M["DetCleanPymExcessPositive"] = I(sum(doc["seeded_faults"]["clean_output_pymatching_excess_positive"]
                                           for doc in det.values()))
    M["DetCleanPymAboveEps"] = I(sum(doc["seeded_faults"]["clean_output_pymatching_above_optimum"]
                                     for doc in det.values()))
    M["DetDefectN"] = I(dsum(defect, "defective"))
    M["DetDefectAll"] = I(dsum(defect, "n"))
    M["DetDefectInvalid"] = I(dsum(defect, "invalid"))
    M["DetDefectProven"] = I(dsum(defect, "proven_suboptimal"))
    M["DetDefectNotProven"] = I(dsum(defect, "not_proven_on_defective"))
    M["DetDefectNonDefective"] = I(dsum(defect, "n") - dsum(defect, "defective"))
    M["DetDefectNonDefectiveFlagged"] = I(dsum(defect, "false_alarms"))
    M["DetStealthN"] = I(dsum(stealth, "n"))
    M["DetStealthProven"] = I(dsum(stealth, "proven_suboptimal"))
    M["DetControlN"] = I(dsum(ctrl, "n"))
    M["DetControlFlagged"] = I(dsum(ctrl, "false_alarms"))

    srows = {(d, r["drift_factor"]): r for d, doc in det.items() for r in doc["stale_calibration"]["rows"]}
    swept = [k for k in srows if k[1] != 1.0]
    so = [srows[k]["outcomes"] for k in swept]
    if dsum(so, "invalid") or dsum(so, "missed") or dsum(so, "not_proven_on_defective"):
        raise SystemExit("stale calibration: the text says every suboptimal correction is syndrome-valid "
                         "and proven suboptimal")
    M["DetStaleSuboptimal"] = I(dsum(so, "defective"))
    M["DetStaleProven"] = I(dsum(so, "proven_suboptimal"))
    M["DetStaleCorrections"] = I(dsum(so, "n"))
    M["DetStaleFalseAlarms"] = I(dsum(so, "false_alarms"))
    ctl = [srows[k]["outcomes"] for k in srows if k[1] == 1.0]
    M["DetStaleControlSuboptimal"] = I(dsum(ctl, "defective"))
    M["DetStaleControlFalseAlarms"] = I(dsum(ctl, "false_alarms"))
    for k, name in ((1.5, "OneFive"), (2.0, "Two"), (3.0, "Three")):
        M[f"DetStaleSuboptimal{name}"] = I(sum(srows[(d, k)]["outcomes"]["defective"] for d in (3, 5, 7)))
    for k, name in ((1.5, "OneFive"), (3.0, "Three")):
        ex = [srows[(d, k)]["outcomes"]["mean_excess_of_suboptimal"] for d in (3, 5, 7)]
        M[f"DetStaleExcessLo{name}"], M[f"DetStaleExcessHi{name}"] = enclose(min(ex), max(ex), 2)
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & & proven & false & \multicolumn{3}{c}{logical errors} \\",
             r" & $N$ & subopt. & alarms & stale only & ref.\ only & both \\",
             r"\midrule"]
    for d in (3, 5, 7):
        r = srows[(d, 3.0)]
        c, o = r["logical_cells"], r["outcomes"]
        lines.append(rf"$d={d}$ & {I(o['n'])} & {I(o['proven_suboptimal'])} & {I(o['false_alarms'])} & "
                     rf"{I(c['stale_only_wrong'])} & {I(c['true_only_wrong'])} & {I(c['both_wrong'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_stale.tex"] = "\n".join(lines) + "\n"
    c3 = srows[(3, 3.0)]["logical_cells"]
    M["DetStaleDThreeStaleOnly"] = I(c3["stale_only_wrong"])
    M["DetStaleDThreeTrueOnly"] = I(c3["true_only_wrong"])
    M["DetStaleDThreeBoth"] = I(c3["both_wrong"])
    M["DetStaleDThreeProven"] = I(srows[(3, 3.0)]["outcomes"]["proven_suboptimal"])
    M["DetStaleDThreeTrueErrors"] = I(c3["true_only_wrong"] + c3["both_wrong"])
    M["DetStaleDThreeStaleErrors"] = I(c3["stale_only_wrong"] + c3["both_wrong"])
    if sum(t["defective"] for t in clean) or sum(t["defective"] for t in clean_full):
        raise SystemExit("detection: the text says every refusal of a clean output was producer slack")
    if any(n.startswith("M5") for n in det[7]["seeded_faults"]["per_mutant"]):
        raise SystemExit("detection: the text says no equal-weight alternative arose at distance 7")
    if dsum(ctrl, "certified") != dsum(ctrl, "n"):
        raise SystemExit("detection: the text says every equal-weight alternative was certified")

    # ------------------------------------------------ gap distribution
    gd = J("gap_distribution.json")
    gx = J("gap_distribution_exact_rule.json")
    for doc in (gd, gx):
        for r in doc["rows"]:
            for dname, dd in r["decoders"].items():
                for gk in ("gap_vs_certified_bound", "gap_vs_proven_optimum"):
                    gg = (dd or {}).get(gk)
                    if gg and gg.get("frac_below_zero", 0) > 0:
                        raise SystemExit(f"gap distribution: a negative {gk} for {dname}; the text says none")
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
