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

import gzip
import hashlib
import json
import math
import pathlib
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
OUT = ROOT / "paper" / "certifying-decoders" / "generated"


def J(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def judged_by_this_tree(doc: dict, what: str) -> None:
    """An artifact whose verdicts came from the exact checkers must name the bytes that decided them, and they must be
    this tree's: the exact checkers were fixed on 2026-10-18 (T'-oddness), after several artifacts were first written."""
    for p, h in doc["head"]["sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != h:
            raise SystemExit(f"{what}: judged with a {p} other than this tree's; rerun it")
    if doc["head"]["uncommitted_changes_in_src_oneq"]:
        raise SystemExit(f"{what}: judged with uncommitted checker sources")


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
    # The text says every decoding graph is simple (referee minor 15: the checker keys edges by endpoints, so a parallel
    # edge would make a correction's weight ambiguous); hold it on every persisted instance.
    for ip in sorted((EV / "instances").glob("matching_*.json")):
        ends = [tuple(sorted((u, v))) for u, v, _w in json.loads(ip.read_text(encoding="utf-8"))["edges"]]
        if len(ends) != len(set(ends)):
            raise SystemExit(f"{ip.name}: parallel edges; the text says every graph is simple")
    # the rerun's own persisted graph, not a timing artifact (referee minor 17); the cost run must be on the same graph
    inst = J("instances/matching_sim_d5_r25.json")
    sim_nodes = {u for u, v, _ in inst["edges"]} | {v for u, v, _ in inst["edges"]}
    M["SimDetectors"] = I(len([x for x in sim_nodes if x >= 0]))
    M["SimEdges"] = I(len(inst["edges"]))
    cc = J("certify_cost_d5_r25.json")
    if (cc["edges"], cc["detectors"]) != (len(inst["edges"]), len([x for x in sim_nodes if x >= 0])):
        raise SystemExit("the cost measurement did not run on the rerun's graph")

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
    # THE SAME CONFIGURATION ON FRESH SEEDS (second referee, minor 9; post hoc): seeds no run of the project used,
    # every receipt replayed. Its seeds must lie outside the main run's plan and its producer must be this tree's.
    fs = eps_summary("fresh_sim_d5_r25")
    if fs is None:
        raise SystemExit("fresh-seed simulation: no complete, replayed run")
    lo_f, hi_f = fs["spec"]["seeds"]
    # recorded seed bases of the simulation runs: the main plan 1..4,000, supplement 1 from 5,000 (its commit); the
    # other supplements' bases were not recorded, so the claim is "far from every recorded base", checked here
    if not (lo_f > 10**9 and fs["spec"]["d"] == 5 and fs["spec"]["rounds"] == 25 and fs["spec"]["p"] == 0.005
            and (es is None or lo_f > es["spec"]["seeds"][1])):
        raise SystemExit("fresh-seed simulation: not the simulation configuration, or seeds near a recorded plan")
    from scipy.stats import beta as _beta
    kf, nf = fs["accepted"], fs["nontrivial"]
    M["FreshSimN"], M["FreshSimAccepted"], M["FreshSimPct"] = I(nf), I(kf), pct_floor(kf / nf)
    M["FreshSimNotProven"] = I(fs["status"].get("NOT_PROVEN", 0))
    M["FreshSimForgeries"] = I(fs["forgeries_attempted"])
    M["FreshSimCILo"] = f"{math.floor(1e4 * float(_beta.ppf(0.025, kf, nf - kf + 1))) / 100:.2f}"
    M["FreshSimCIHi"] = f"{math.ceil(1e4 * float(_beta.ppf(0.975, kf + 1, nf - kf))) / 100:.2f}"
    if fs["forgeries_accepted"]:
        raise SystemExit("fresh-seed simulation: a forgery was accepted")

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
        # the headline mixes two producers (referee finding 3): their shot counts, so the abstract can say so
        M["EpsHwGenFourNontrivial"] = I(agg(g4c, "nontrivial"))
        M["EpsHwGenSixNontrivial"] = I(agg(g6c, "nontrivial"))
        M["EpsHwGenSixSharePct"] = f"{math.floor(100 * agg(g6c, 'nontrivial') / agg(wcfg, 'nontrivial'))}"
        M["EpsForgeriesTotal"] = I(agg(wcfg, "forgeries_attempted") + es["forgeries_attempted"])
        M["EpsLargestEps"] = SCI(float(max(hw_eps, Fraction(es["largest_accepted_epsilon"]))))

    # THE HELD-OUT SAMPLE (referee finding 4; preregistered in docs/PREREG_heldout_willow_2026-11-02.md, committed
    # before its first shot): the frozen base ladder on four 50-round configurations no generation read. The summary
    # must match the preregistration, and every configuration's receipts must have replayed completely.
    ho = J("epsilon_rerun/heldout_willow_r50.json")
    pre_cfgs = {"heldout_d3_at_q4_5_X_r50", "heldout_d3_at_q4_5_Z_r50", "heldout_d5_at_q8_7_X_r50",
                "heldout_d5_at_q8_7_Z_r50"}
    if set(ho["configurations"]) != pre_cfgs or any(r["shots"] != 5000 for r in ho["configurations"].values()):
        raise SystemExit("held-out sample: not the preregistered configurations and shot counts")
    for nm in pre_cfgs:
        rp = J(f"epsilon_rerun/replay_{nm}.json")["results"][0]
        if not rp["full"] or rp["receipts_rechecked"] != ho["configurations"][nm]["nontrivial"]:
            raise SystemExit(f"held-out sample: {nm} has no complete replay of its receipts")
    hp = ho["pooled"]
    M["HoWShotsPerConfig"] = I(5000)
    M["HoWNontrivial"] = I(hp["nontrivial"])
    M["HoWAccepted"] = I(hp["accepted"])
    M["HoWPct"] = pct_floor(hp["accepted"] / hp["nontrivial"])
    M["HoWCILo"], M["HoWCIHi"] = f"{100 * hp['cp95_two_sided'][0]:.2f}", f"{100 * hp['cp95_two_sided'][1]:.2f}"
    M["HoWNotProven"] = I(sum(r["not_proven"] for r in ho["configurations"].values()))
    M["HoWForgeries"] = I(sum(r["forgeries_attempted"] for r in ho["configurations"].values()))
    if ho["forgeries_accepted_total"]:
        raise SystemExit("held-out sample: a forgery was accepted")
    pid = J("producer_identity_willow.json")
    for p, h in pid["head"]["sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != h:
            raise SystemExit(f"producer identity: run with a {p} other than this tree's; rerun it")
    want = ({f"willow_{pl}_{b}_r{r}" for pl in ("d3_at_q4_5", "d5_at_q8_7") for b in "XZ" for r in (13, 30)}
            | pre_cfgs | {"fresh_sim_d5_r25"})
    if set(pid["configs"]) != want:
        raise SystemExit("producer identity: not the development and held-out configurations of these placements")
    pdiff = J("producer_identity_willow_diff.json")
    if pdiff["identity_head"] != pid["head"]["commit"]:
        raise SystemExit("producer identity: the shot-level diff is not of this identity run")
    # per distance, beside the development rate of the SAME placements and base ladder at 13 and 30 rounds (the
    # preregistration: "beside, not in place of, the development-sample rate, and says which is which")
    for tag, place in (("DThree", "d3_at_q4_5"), ("DFive", "d5_at_q8_7")):
        rows_d = [r for nm, r in ho["configurations"].items() if place in nm]
        n_d, k_d = sum(r["nontrivial"] for r in rows_d), sum(r["accepted"] for r in rows_d)
        M[f"HoW{tag}N"], M[f"HoW{tag}Accepted"], M[f"HoW{tag}Pct"] = I(n_d), I(k_d), pct_floor(k_d / n_d)
        M[f"HoW{tag}Exact"] = I(sum(r["exact_optimal"] for r in rows_d))
        if tag == "DThree" and k_d != n_d:
            raise SystemExit("held-out sample: the abstract says every distance-3 correction was proven")
        for word, rounds in (("Thirteen", "13"), ("Thirty", "30")):
            dev = [J(f"epsilon_rerun/willow_{place}_{b}_r{rounds}/summary.json") for b in ("X", "Z")]
            if any(d["spec"]["method"] != "certify" for d in dev):
                raise SystemExit("held-out comparison: a development configuration was not run with the base ladder")
            # the development verdicts of TODAY's producer, the one that certified the held-out sample (second
            # referee, M2): the committed histogram for every byte-identical chunk, the re-run's for the others
            today = [pdiff["development_status_today"][f"willow_{place}_{b}_r{rounds}"] for b in ("X", "Z")]
            nt = sum(sum(v for k, v in s.items() if k != "TRIVIAL") for s in today)
            acc = sum(s.get("EXACT_OPTIMAL", 0) + s.get("EPSILON_OPTIMAL", 0) for s in today)
            if nt != sum(d["nontrivial"] for d in dev):
                raise SystemExit("held-out comparison: today's development verdicts cover other shots")
            M[f"Dev{tag}{word}Pct"] = pct_floor(acc / nt)
            if tag == "DFive":
                # not proven per shot-round: the decline with rounds, normalised (second referee, M3)
                M[f"DevDFive{word}NotProvenPerRound"] = f"{1e4 * (nt - acc) / (nt * int(rounds)):.1f}"
    # THE SAME PRODUCER (second referee, M2), stated as measured: re-run with this tree's producer, every held-out
    # receipt file must come back byte for byte; the development files that do not are diffed shot by shot
    held = [c for n, c in pid["configs"].items() if c["sample"] == "held-out"]
    fresh = [c for n, c in pid["configs"].items() if c["sample"] == "fresh-seed"]
    if not fresh or any(c["identical"] != c["chunks"] for c in fresh):
        raise SystemExit("producer identity: the fresh-seed receipts were not reproduced byte for byte")
    M["IdentFreshChunks"] = I(sum(c["chunks"] for c in fresh))
    devc = [c for n, c in pid["configs"].items() if c["sample"] == "development"]
    if any(c["identical"] != c["chunks"] for c in held) or any(c["failed"] for c in pid["configs"].values()):
        raise SystemExit("producer identity: a held-out chunk was not reproduced byte for byte")
    if sorted(pdiff["chunks"]) != sorted(f"{n}:{s}" for n, c in pid["configs"].items() for s in c["differ"]):
        raise SystemExit("producer identity: the shot-level diff does not cover exactly the differing chunks")
    diffs = [d for c in pdiff["chunks"].values() for d in c["diffs"]]
    M["IdentHeldChunks"] = I(sum(c["chunks"] for c in held))
    M["IdentDevChunks"] = I(sum(c["chunks"] for c in devc))
    M["IdentDevIdentical"] = I(sum(c["identical"] for c in devc))
    M["IdentDevDiffShots"] = I(len(diffs))
    M["IdentDevStatusChanged"] = I(sum(d["status_then"] != d["status_now"] for d in diffs))
    if dc["producer_identity_development_nontrivial"] != sum(c["nontrivial"] for c in devc):
        raise SystemExit("producer identity: derived_claims.json is stale; run tools/derive_claims.py")
    M["IdentDevShots"] = I(dc["producer_identity_development_nontrivial"])
    nd5 = sum(r["nontrivial"] for nm, r in ho["configurations"].items() if "d5_at" in nm)
    np5 = sum(r["not_proven"] for nm, r in ho["configurations"].items() if "d5_at" in nm)
    M["HoWDFiveNotProvenPerRound"] = f"{1e4 * np5 / (nd5 * 50):.1f}"
    M["HoWLargestEps"] = SCI(max(r["largest_accepted_epsilon_float"] for r in ho["configurations"].values()))

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
    # re-certified under the epsilon-checker from the raw registers (third referee, M1): the same shots, the same
    # producer, every receipt decided by both verifiers, judged by this tree's bytes
    ie = J("ibm_live/d9nt2imij12s73fu9ihg.epsilon.json")
    judged_by_this_tree(ie, "IBM epsilon re-certification")
    if (ie["nontrivial"], ie["shots"]) != (ib["nontrivial"], ib["shots"]) or ie["verifiers_agree_on"] != ie["nontrivial"]:
        raise SystemExit("IBM epsilon re-certification: not the same shots, or the verifiers disagreed")
    if ie["forgeries_accepted"] or ie["battery_attacks_not_false"] or ie["accepted"] != ie["nontrivial"]:
        raise SystemExit("IBM epsilon re-certification: the text says every shot proven and no forgery accepted")
    if hashlib.sha256(gzip.decompress((ROOT / ie["receipts"]).read_bytes())).hexdigest() != ie["receipts_sha256"]:
        raise SystemExit("IBM epsilon re-certification: the receipt file is not the one recorded")
    M["IbmEpsAccepted"] = I(ie["accepted"])
    M["IbmEpsExact"] = I(ie["status"].get("EXACT_OPTIMAL", 0))
    M["IbmEpsForgeries"] = I(ie["forgeries_attempted"])

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

    # EXACT 95% CLOPPER-PEARSON INTERVALS on every small count the text quotes (referee finding 10), and the exact
    # two-sided binomial test on the stale-calibration discordant cells at drift 3.0 (all distances).
    from scipy.stats import beta as _beta

    def cp(k: int, n: int) -> tuple[float, float]:
        lo = 0.0 if k == 0 else float(_beta.ppf(0.025, k, n - k + 1))
        hi = 1.0 if k == n else float(_beta.ppf(0.975, k + 1, n - k))
        return lo, hi

    def pct2(x: float) -> str:
        """A percentage with three significant figures, so a lower bound of 0.0033% does not print as 0.00."""
        v = 100 * x
        return "0" if v == 0 else f"{v:.3g}"

    for tag, k, n in (("FaBase", dsum(clean, "false_alarms"), dsum(clean, "n")),
                      ("FaFull", dsum(clean_full, "false_alarms"), dsum(clean_full, "n")),
                      ("FaStale", dsum(so, "false_alarms"), dsum(so, "n")),
                      ("Ctrl", dsum(ctrl, "false_alarms"), dsum(ctrl, "n"))):
        lo, hi = cp(k, n)
        M[f"CI{tag}Lo"], M[f"CI{tag}Hi"] = pct2(lo), pct2(hi)
    so_d = sum(srows[(d, 3.0)]["logical_cells"]["stale_only_wrong"] for d in (3, 5, 7))
    to_d = sum(srows[(d, 3.0)]["logical_cells"]["true_only_wrong"] for d in (3, 5, 7))
    nd = so_d + to_d
    p_two = min(1.0, 2 * sum(math.comb(nd, i) for i in range(min(so_d, to_d) + 1)) / 2 ** nd) if nd else 1.0
    M["DetStaleDiscStaleOnly"], M["DetStaleDiscTrueOnly"] = I(so_d), I(to_d)
    M["DetStaleDiscP"] = f"{p_two:.2f}"
    if sum(t["defective"] for t in clean) or sum(t["defective"] for t in clean_full):
        raise SystemExit("detection: the text says every refusal of a clean output was producer slack")
    if any(n.startswith("M5") for n in det[7]["seeded_faults"]["per_mutant"]):
        raise SystemExit("detection: the text says no equal-weight alternative arose at distance 7")
    if dsum(ctrl, "certified") != dsum(ctrl, "n"):
        raise SystemExit("detection: the text says every equal-weight alternative was certified")

    # ------------------------------------------------ gap distribution
    gd = J("gap_distribution.json")
    gx = J("gap_distribution_exact_rule.json")
    judged_by_this_tree(gx, "exact-rule gap distribution")
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
    M["GapExactRuleNontrivial"] = " / ".join(I(r["nontrivial"]) for r in gx["rows"])
    M["GapExactRuleBposdExact"] = " / ".join(P(x, 1) for x in fzx)

    # ------------------------------------------------ cost and scaling
    # Re-measured 2026-11-02 with every condition recorded (referee finding 1): the old decode_vs_certify.json
    # named no noise level, shot count, commit or machine and is no longer read.
    if cc["p"] != 0.005 or cc["distance"] != 5 or cc["rounds"] != 25 or "matched" not in cc["same_shots_as"]:
        raise SystemExit("the cost run is not the d=5, 25-round, p=0.005 configuration on the rerun's shots")
    M["CostShots"] = I(cc["nontrivial"])
    M["CostP"] = F(cc["p"], 3)
    M["CertifyMsMedian"] = f"{cc['certify_ms']['median']:.0f}"
    M["CertifyMsMean"] = f"{cc['certify_ms']['mean']:.0f}"
    M["CertifyMsPNinetyNine"] = f"{cc['certify_ms']['p99']:.0f}"
    M["CertifyRefusedN"] = I(cc["certify_ms_refused"]["n"])
    M["CertifyRefusedMsMedian"] = I(dc["certify_cost_refused_median_ms_2sf"])
    M["CostRatioCheckBatch"] = I(dc["certify_cost_check_median_over_batch_2sf"])
    M["CostCpuLoadAfterPct"] = f"{max(cc['machine']['cpu_percent_before'], cc['machine']['cpu_percent_after']):.0f}"
    M["CertifyRefusedSharePct"] = f"{100 * cc['refused_share_of_certify_time']:.1f}"
    M["CheckMsMedian"] = f"{cc['check_ms']['median']:.0f}"
    M["CheckMsPNinetyNine"] = f"{cc['check_ms']['p99']:.0f}"
    M["CheckOverCertifyPct"] = f"{100 * cc['ratio_median_check_over_median_certify']:.0f}"
    M["CostRatioBatch"] = I(dc["certify_cost_ratio_batch_2sf"])
    M["CostRatioPerShot"] = I(dc["certify_cost_ratio_per_shot_2sf"])
    # both MEANS, as the sentence says, beside the mean certification cost they divide (final referee, m8)
    M["DecodeUsBatch"] = f"{cc['decode_batch_us_per_shot']:.1f}"
    M["DecodeUsPerShot"] = f"{cc['decode_per_shot_us']['mean']:.0f}"
    M["CostCpuLoadPct"] = f"{min(cc['machine']['cpu_percent_before'], cc['machine']['cpu_percent_after']):.0f}"
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
    judged_by_this_tree(au, "semantics audit")
    gd_census = J("gap_distribution.json")
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
    inadm = sum(sum(r["inadmissible_certificates"].values()) for r in rows.values())
    M["AuditInadmissible"] = I(inadm)
    # what the audit says about the original runs' acceptances (third referee, M1). NOT one pooled bound (final
    # referee, M1): the sample is not one random draw from those runs -- the gap experiment was re-run in full, the
    # other rows sample three runs. A census gives a share of exactly zero; each sampled row, with none inadmissible,
    # gives its own exact one-sided 95% upper bound, rounded UP; nothing else is bounded.
    if inadm:
        raise SystemExit("semantics audit: an inadmissible certificate; the text bounds their share from zero")
    for n_gap, r in zip((x["nontrivial"] for x in gd_census["rows"]),
                        (rows[f"gap_distribution_d{d}"] for d in (3, 5, 7))):
        if r["nontrivial_sampled"] != n_gap:
            raise SystemExit("semantics audit: the text says the gap experiment was re-run in full")
    for macro, name in (("AuditBoundSimPct", "million_shot_d5_r25"), ("AuditBoundWFiveXPct", "willow_gen4_d5_X_r13"),
                        ("AuditBoundWSevenXPct", "willow_gen6_d7_X_r30")):
        M[macro] = f"{math.ceil(1e4 * (1 - 0.05 ** (1 / rows[name]['accepted_by_pinned_code']))) / 100:.2f}"
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

    # ------------------------------- certifying against a model MEASURED from the run
    # `experiments/dual_observable/certified_measured_lane.py` hands the accept path of
    # `rep_code_cert.certify_shots` a decoding graph estimated from the FIRST HALF of each
    # run's own detectors (the two-point estimator of Spitz et al.) and judges the second
    # half. Preregistered as docs/PREREG_measured_model_program_2026-09-16.md section A;
    # every prediction A1-A4 is recorded per record in the artifact.
    cml = J("certified_measured_lane.json")
    judged_by_this_tree(cml, "certified measured-model lane")
    cm_rows = cml["runs"]
    if len(cm_rows) < 6:
        raise SystemExit("certified measured lane: the text says six records")
    cm_devices = sorted({r["backend"] for r in cm_rows})
    if len(cm_devices) != 2:
        raise SystemExit(f"certified measured lane: the text says two devices, got {cm_devices}")
    if sorted({r["distance"] for r in cm_rows}) != [3, 5]:
        raise SystemExit("certified measured lane: the text says distances 3 and 5")
    for r in cm_rows:
        # A3: the accept path and the campaign's own scorer must be the SAME decoder, or the
        # comparison below is between two different things and the sentence is void.
        if r["verdicts"]["A3_agrees_with_score"]["met"] is not True:
            raise SystemExit(f"{r['record']}: the certified accept path and the lane's scorer "
                             f"disagree; the text says they agree on every record")
        if r["models"]["measured"]["certified_rate"] < 0.99:
            raise SystemExit(f"{r['record']}: certified rate {r['models']['measured']['certified_rate']} "
                             f"is below the 0.99 the text states")
    # THE TOTALS ARE READ, NOT SUMMED HERE. The first version of this block summed the rows, so
    # the paper's own generator became the producer of its headline and the number existed in no
    # artifact -- exactly the orphan `tools/claims_provenance_gate.py` refuses, and it refused it.
    # The lane now writes `totals`; this reads them and re-derives them as a cross-check.
    cm = cml.get("totals")
    if not cm:
        raise SystemExit("certified_measured_lane.json carries no `totals` block; re-run the lane "
                         "(its producer writes them) rather than summing the rows here")
    if cm["certified_under_measured"] != sum(r["models"]["measured"]["certified"] for r in cm_rows):
        raise SystemExit("certified measured lane: the recorded totals disagree with its own rows")
    if cm["logical_errors"]["measured"] < 1:
        raise SystemExit("certified measured lane: a ratio needs a nonzero denominator")
    if cm["rounds_with_refusals"] and cm["rounds_with_refusals"] != [40]:
        raise SystemExit(f"certified measured lane: the text says every refusal is on a 40-round "
                         f"record; refusals appear at rounds {cm['rounds_with_refusals']}")
    M["CertMeasRecords"] = I(cm["records"])
    M["CertMeasDevices"] = I(len(cm["devices"]))
    M["CertMeasCertified"] = I(cm["certified_under_measured"])
    M["CertMeasNontrivial"] = I(cm["nontrivial"])
    M["CertMeasRefused"] = I(cm["refused_under_measured"])
    M["CertMeasHeldOut"] = I(cm["held_out_shots"])
    M["CertMeasDemErrors"] = I(cm["logical_errors"]["dem"])
    M["CertMeasUniformErrors"] = I(cm["logical_errors"]["uniform"])
    M["CertMeasErrors"] = I(cm["logical_errors"]["measured"])
    M["CertMeasRatio"] = I(round(cm["dem_over_measured_errors"]))
    M["CertMeasEdgesLo"] = I(cm["measured_graph_edges_min"])
    M["CertMeasEdgesHi"] = I(cm["measured_graph_edges_max"])

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
        _w = OUT / name
        print("wrote", _w.relative_to(ROOT) if _w.is_relative_to(ROOT) else _w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
