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
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
OUT = ROOT / "paper" / "certifying-decoders" / "generated"


def J(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def _sha_lf(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _committed_sha(path: str, rev: str = "HEAD") -> str | None:
    """sha256 (CRLF normalised, as git stores it) of `path` at commit `rev`, or None if the commit lacks it."""
    import subprocess
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True)
    return None if r.returncode else _sha_lf(r.stdout)


#: Set ONEQ_NUMBERS_DRAFT=1 to typeset from runs whose deciding bytes are not yet committed (a working draft only).
DRAFT = __import__("os").environ.get("ONEQ_NUMBERS_DRAFT") == "1"
DRAFT_UNCOMMITTED: list = []


def judged_by_this_tree(doc: dict, what: str) -> None:
    """An artifact whose verdicts came from the checkers must name the bytes that decided them, and they must be this
    tree's: the exact checkers were fixed on 2026-10-18 (T'-oddness), and the epsilon checker's input domain on
    2026-11-03, after several artifacts were first written.

    THE BYTES MUST ALSO BE IN A COMMIT. Before 2026-11-03 this refused any artifact whose run saw uncommitted changes
    anywhere in src/oneq -- on a box several lanes share, a run is then blocked by another lane's unrelated module, and
    a run made on bytes committed a minute later is refused for ever. The rule now is the one the claim needs: every
    judged file's bytes are this tree's, and if the run recorded them as uncommitted they must be committed NOW, at
    HEAD, byte for byte. A number is typeset only from bytes that a commit preserves."""
    for p, h in doc["head"]["sha256"].items():
        if _sha_lf((ROOT / p).read_bytes()) != h:
            raise SystemExit(f"{what}: judged with a {p} other than this tree's; rerun it")
    head = doc["head"]
    flags = ("uncommitted_changes_in_judged_files", "uncommitted_changes", "uncommitted_changes_in_src_oneq")
    if not any(k in head for k in flags):
        raise SystemExit(f"{what}: the record does not say whether its judged files were committed")
    dirty = any(head.get(k) for k in flags)
    if dirty and not _git_available():
        # a reader's extracted bundle (no repository, or a parent's): the bytes-equal-this-tree check above is the one
        # that decides there; a comparison with a commit is not possible, and the output says so (2026-11-05)
        NO_REPOSITORY.append(what)
        print(f"NOTE: {what}: no repository here, so whether its judged bytes are committed could not be checked; "
              f"they equal this tree's, byte for byte")
        return
    if dirty:
        stale = [p for p, h in head["sha256"].items() if _committed_sha(p) != h]
        if stale and DRAFT:
            DRAFT_UNCOMMITTED.append((what, stale))
        elif stale:
            raise SystemExit(f"{what}: judged with bytes no commit holds ({', '.join(stale)}); commit them, or rerun "
                             f"at a commit (ONEQ_NUMBERS_DRAFT=1 typesets a working draft)")


def replay_path(name: str) -> pathlib.Path:
    """The complete replay of a configuration's receipts: the 2026-11-03 strict-loader replay once it exists."""
    new = EV / "epsilon_rerun" / "replay_2026-11-03" / f"replay_{name}.json"
    REPLAYS_USED.append(new.exists())
    return new if new.exists() else EV / "epsilon_rerun" / f"replay_{name}.json"


REPLAYS_USED: list = []


def _git_available() -> bool:
    """Whether this tree IS a repository (its own top level, not a parent's). A reader's extracted bundle is not:
    there every byte is still pinned by sha256, but a comparison with a COMMIT's bytes is not possible, and the
    generator says so instead of refusing."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return False
    return r.returncode == 0 and pathlib.Path(r.stdout.strip()).resolve() == ROOT.resolve()


NO_REPOSITORY: list = []


def judged_at_its_commit(doc: dict, what: str) -> None:
    """A run attributed to a clean commit: every judged file's recorded bytes are that commit's."""
    head = doc["head"]
    if head.get("uncommitted_changes", head.get("uncommitted_changes_in_src_oneq")):
        raise SystemExit(f"{what}: the run was not made at a clean commit")
    if not _git_available():
        NO_REPOSITORY.append(what)
        print(f"NOTE: {what}: no repository here, so the recorded bytes could not be compared with commit "
              f"{head['commit'][:10]}; they are pinned by sha256 in the record")
        return
    for p, h in head["sha256"].items():
        if _committed_sha(p, head["commit"]) != h:
            raise SystemExit(f"{what}: {p} at {head['commit'][:10]} is not the recorded bytes")


def I(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def need(cond: bool, msg: str) -> None:
    """A generated number is printed only if the evidence still says what the text will say about it."""
    if not cond:
        raise SystemExit(f"need: {msg}")


def P(x: float, k: int = 2) -> str:
    return f"{100 * x:.{k}f}"


def PF(x: float, k: int = 2) -> str:
    """A rate printed from below, everywhere in the paper (external review 1, 2026-11-03: Table II rounded to nearest
    and Table III floored, so the same rate printed 99.99 in one and 99.98 in the other)."""
    return f"{math.floor(100 * x * 10 ** k) / 10 ** k:.{k}f}"


def F(x: float, k: int = 2) -> str:
    return f"{x:.{k}f}"


def enclose(lo: float, hi: float, k: int) -> tuple[str, str]:
    """A printed range must CONTAIN the data: floor the low end, ceiling the high."""
    f = 10 ** k
    return f"{math.floor(lo * f) / f:.{k}f}", f"{math.ceil(hi * f) / f:.{k}f}"


def SIG3(x: float, direction: str) -> str:
    """A rate as a percentage with three significant figures, rounded AWAY from the data's inside.

    An interval's low end is floored and a bound's high end ceiled, so a printed interval contains the computed one
    (as `enclose` does for fixed decimals): an upper bound of 0.72149% prints 0.722, never 0.721. Three significant
    figures so a lower end of 0.0033% does not print as 0.00. Exact decimal arithmetic on the float's shortest repr,
    so a value that IS 0.721 is not pushed to 0.722 by binary rounding."""
    from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
    if direction not in ("floor", "ceil"):
        raise ValueError(direction)
    v = Decimal(repr(100 * x))
    if v == 0:
        return "0"
    q = v.quantize(Decimal(1).scaleb(v.adjusted() - 2), rounding=ROUND_CEILING if direction == "ceil" else ROUND_FLOOR)
    if q.adjusted() > v.adjusted():                       # 9.995 ceiled to 10.0: keep three significant figures
        q = q.quantize(Decimal(1).scaleb(q.adjusted() - 2), rounding=ROUND_CEILING)
    return f"{q:f}"


def SCI(x: float, k: int = 1) -> str:
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, k) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.{k}f}\times 10^{{{e}}}}}"


def heldout_d7_macros(hd: dict) -> dict[str, str]:
    """Macros for the held-out distance-7 sample (docs/PREREG_heldout_d7_2026-11-03.md), complete or stopped.

    A round count with no completed chunk gets NO macros: a sentence that quotes it then fails to compile instead of
    typesetting a rate of nothing. Bounds are ceiled (SIG3), rates floored (PF)."""
    if hd.get("preregistration") != "docs/PREREG_heldout_d7_2026-11-03.md" or hd.get("forgeries_accepted_total"):
        raise SystemExit("held-out d=7: not the preregistered run, or a forgery was accepted")
    M: dict[str, str] = {}
    pl = hd["pooled"]
    partial = hd.get("schema") == "oneq-heldout-willow-d7-partial/1"
    if not partial and hd.get("schema") != "oneq-heldout-willow-d7/1":
        raise SystemExit(f"held-out d=7: unknown schema {hd.get('schema')}")
    if partial and hd.get("status") != "STOPPED_BY_RULE":
        raise SystemExit("held-out d=7: a partial report that was not stopped by the rule")
    for key, rr_ in (("", "r50"), ("Ten", "r10"), ("Dev", "r30")):
        n, k = pl[rr_]["nontrivial"], pl[rr_]["not_proven"]
        if not n:
            continue
        M[f"HoDSeven{key}N"] = I(n)
        M[f"HoDSeven{key}Accepted"] = I(n - k)
        M[f"HoDSeven{key}NotProven"] = I(k)
        M[f"HoDSeven{key}Pct"] = PF((n - k) / n, 3)
        M[f"HoDSeven{key}Upper"] = SIG3(pl[rr_]["not_proven_upper_95_one_sided"], "ceil")
        if partial:
            M[f"HoDSeven{key}WorstN"] = I(pl[rr_]["worst_case_nontrivial"])
            M[f"HoDSeven{key}WorstUpper"] = SIG3(pl[rr_]["worst_case_upper_95_one_sided"], "ceil")
            M[f"HoDSeven{key}Arms"] = I(len(pl[rr_]["arms_with_data"]))
            sel = [a["completed"] for a in hd["arms"].values() if a["rounds"] == rr_]
            M[f"HoDSeven{key}Exact"] = I(sum(c["exact_optimal"] for c in sel))
            M[f"HoDSeven{key}LargestEps"] = SCI(float(max(Fraction(c["largest_accepted_epsilon"]) for c in sel)))
    if partial:
        tests = {"TestP": hd["tests"]["heldout_r50_vs_same_producer_dev_r30"],
                 "TenTestP": hd["tests"]["heldout_r10_vs_same_producer_dev_r30"]}
        for key, t in tests.items():
            if t["computed"]:
                M[f"HoDSeven{key}"] = F(t["p_one_sided"], 2)
        arms = hd["arms"].values()
        M["HoDSevenChunks"] = I(sum(a["completed"]["chunks"] for a in arms))
        M["HoDSevenChunksPlanned"] = I(40 * len(hd["arms"]))
        M["HoDSevenCoreHours"] = F(hd["stop_record"]["core_hours"], 1)
        M["HoDSevenForgeries"] = I(sum(a["completed"]["forgeries_attempted"] for a in arms))
        M["HoDSevenState"] = "stopped by its preregistered 40 core-hour rule"
        need(all(a["completed"]["forgeries_accepted"] == 0 and a["completed"]["battery_attacks_not_false"] == 0
                 for a in arms), "held-out d=7: a forgery accepted or an attack not false in some arm")
        # the pooled rows ARE the sums of their arms, and the bound is the exact one-sided Clopper-Pearson bound
        from scipy.stats import beta as _b
        for rr_, row in pl.items():
            sel = [a for a in arms if a["rounds"] == rr_]
            need(row["nontrivial"] == sum(a["completed"]["nontrivial"] for a in sel) and
                 row["not_proven"] == sum(a["completed"]["not_proven"] for a in sel), f"held-out d=7: pooled {rr_}")
            if row["nontrivial"]:
                n_, k_ = row["nontrivial"], row["not_proven"]
                want = 1.0 if k_ == n_ else float(_b.ppf(0.95, k_ + 1, n_ - k_))
                need(abs(row["not_proven_upper_95_one_sided"] - want) < 1e-12, f"held-out d=7: bound {rr_}")
        unf = {a: v["dispatched_unfinished_seeds"] for a, v in hd["arms"].items()}
        M["HoDSevenUnfinishedChunks"] = I(sum(len(v) for v in unf.values()))
        M["HoDSevenUnfinishedShots"] = I(sum(v["dispatched_unfinished_nontrivial"] for v in hd["arms"].values()))
        M["HoDSevenDispatched"] = I(sum(a["completed"]["chunks"] for a in arms) + sum(len(v) for v in unf.values()))
        pre = [v["prefix"] for v in hd["arms"].values() if v["rounds"] == "r50"]
        M["HoDSevenPrefixN"] = I(sum(x["nontrivial"] for x in pre))
        M["HoDSevenPrefixNotProven"] = I(sum(x["not_proven"] for x in pre))
        rec = hd.get("recorded_development_2026_09_17")
        if rec:
            M["HoDSevenRecordedDevUpper"] = SIG3(rec["not_proven_upper_95_one_sided"], "ceil")
        # THE INTERPRETATION RULE, fixed in the preregistration before the run, applied here and nowhere else:
        #  (i)  bound on the held-out 50-round not-proven rate below 1% -> "proved at least (100 - bound)%";
        #  (ii) 50-round rate significantly higher than the same-producer development rate (p < 0.05) -> read with
        #       the 10-round bracket: bracket also significantly higher -> fitting; bracket at or below the
        #       development rate -> a configuration effect; anything else -> undecided. The preregistration defines
        #       NO reading for the non-significant case, so none is typeset: the test "does not reject".
        r50, r10, r30 = pl["r50"], pl["r10"], pl["r30"]
        t50, t10 = tests["TestP"], tests["TenTestP"]
        if r50["nontrivial"] and r50["not_proven_upper_95_one_sided"] < 0.01:
            M["HoDSevenAtLeastPct"] = PF(1 - r50["not_proven_upper_95_one_sided"], 2)
            M["HoDSevenAbstract"] = (f"at least {M['HoDSevenAtLeastPct']}\\% of {M['HoDSevenN']} at distance 7 in the "
                                     f"chunks a preregistered budget completed")
        else:
            M["HoDSevenAbstract"] = "a held-out distance-7 sample too small to bound its rate below 1\\%"
        if not (t50["computed"] and t10["computed"]):
            M["HoDSevenOutcome"] = "untestable"
        elif not t50["p_one_sided"] < 0.05:
            M["HoDSevenOutcome"] = "not_rejected"
        elif t10["p_one_sided"] < 0.05:
            M["HoDSevenOutcome"] = "fitting"
        elif r10["not_proven_rate"] <= r30["not_proven_rate"]:
            M["HoDSevenOutcome"] = "configuration"
        else:
            M["HoDSevenOutcome"] = "undecided"
        p50, p10 = M.get("HoDSevenTestP", ""), M.get("HoDSevenTenTestP", "")
        head_ = ("The preregistered one-sided test of the held-out 50-round not-proven rate against the same-producer "
                 "development rate")
        M["HoDSevenReading"] = {
            "untestable": f"{head_} could not be computed: one of the two arms completed no chunk.",
            "not_rejected": f"{head_} does not reject it (one-sided $p={p50}$; the 10-round bracket gives $p={p10}$); "
                            "the preregistration prescribes no further reading for this case.",
            "fitting": f"{head_} rejects it ($p={p50}$), and the 10-round bracket is significantly higher too "
                       f"($p={p10}$), which the preregistration reads as pointing to fitting to the development shots.",
            "configuration": f"{head_} rejects it ($p={p50}$) while the 10-round bracket is at or below the development "
                             f"rate ($p={p10}$), which the preregistration reads as what a configuration effect predicts.",
            "undecided": f"{head_} rejects it ($p={p50}$; the 10-round bracket gives $p={p10}$), a pattern the "
                         "preregistration reports as undecided between fitting and the configuration.",
        }[M["HoDSevenOutcome"]]
        M["HoDSevenOutcome"] = M["HoDSevenOutcome"].replace("_", " ")     # no underscore reaches TeX
        # the dispatch markers: every chunk the deviation's driver handed out; the committed runner's 4 completed
        # chunks were dispatched without one (final referee, 2026-11-05)
        hh = hd.get("head", {}).get("records", {})
        if hh:
            pre = sum(len(v) for v in hh["amendment_stop"]["completed_seeds"].values())
            M["HoDSevenPreDeviationChunks"] = I(pre)
            M["HoDSevenMarkers"] = I(hh["dispatch_markers"])
            need(hh["dispatch_markers"] == int(M["HoDSevenDispatched"].replace("{,}", "")) - pre,
                 "held-out d=7: the dispatch markers are not the dispatched chunks less those completed before the "
                 "deviation")
    else:
        M["HoDSevenTestP"] = F(hd["test_heldout_r50_vs_same_producer_dev_r30"]["p_one_sided"], 2)
        M["HoDSevenTenTestP"] = F(hd["test_heldout_r10_vs_same_producer_dev_r30"]["p_one_sided"], 2)
        M["HoDSevenForgeries"] = I(sum(a["forgeries_attempted"] for a in hd["arms"].values()))
        M["HoDSevenState"] = "complete"
    return M


def build() -> dict[str, str]:
    files: dict[str, str] = {}
    M: dict[str, str] = {}

    # ------------------------------------------------ simulation at scale
    ms = J("million_shot_gate_d5_merged.json")
    M["SimShots"] = I(ms["shots"])
    M["SimCertified"] = I(ms["certified"])
    M["SimCertifiedPct"] = PF(ms["certified_rate"])
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
        M[f"HwGen{name}Pct"] = PF(h["generations"][g]["rate"])
    f = h["final_generation"]
    M["HwFinalNontrivial"] = I(f["nontrivial"])
    M["HwFinalCertified"] = I(f["certified"])
    M["HwFinalPct"] = PF(f["rate"])
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
        M[f"HwFinal{part}Pct"] = PF(k / n, 3 if part == "GenSix" else 2)
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
        M[f"HwGenFive{basis}Pct"] = PF(g5["rate"])
        M[f"HwGenSix{basis}Certified"] = I(g6["certified"])
        M[f"HwGenSix{basis}Nontrivial"] = I(g6["nontrivial"])
        M[f"HwGenSix{basis}Pct"] = PF(g6["rate"], 3)
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
            cells.append(" / ".join(PF(v) for v in vals))
        lines.append(rf"$d={d}$, {r} rounds & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("aggregate & " + " & ".join(
        PF(h["generations"][g]["rate"]) for g in ("1", "2", "3", "4")) + r" \\")
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
        rp = replay_path(name)
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
        rp = json.loads(replay_path(nm).read_text(encoding="utf-8"))["results"][0]
        if not rp["full"] or rp["receipts_rechecked"] != ho["configurations"][nm]["nontrivial"]:
            raise SystemExit(f"held-out sample: {nm} has no complete replay of its receipts")
    hp = ho["pooled"]
    M["HoWShotsPerConfig"] = I(5000)
    M["HoWNontrivial"] = I(hp["nontrivial"])
    M["HoWAccepted"] = I(hp["accepted"])
    M["HoWPct"] = pct_floor(hp["accepted"] / hp["nontrivial"])
    # an interval is printed so that it CONTAINS the computed one: low end floored, high end ceiled (final referee,
    # 2026-11-03: the upper end 99.6335 printed 99.63)
    M["HoWCILo"], M["HoWCIHi"] = enclose(100 * hp['cp95_two_sided'][0], 100 * hp['cp95_two_sided'][1], 2)
    M["HoWNotProven"] = I(sum(r["not_proven"] for r in ho["configurations"].values()))
    M["HoWForgeries"] = I(sum(r["forgeries_attempted"] for r in ho["configurations"].values()))
    if ho["forgeries_accepted_total"]:
        raise SystemExit("held-out sample: a forgery was accepted")
    # the identity run is re-made whenever a module it loads changes (2026-11-03: the battery and the checker's input
    # domain); until the re-run of record exists its numbers are pending, never read from a run of other bytes
    pid_tag = "2026-11-03"
    pid_ok = (EV / f"producer_identity_willow_{pid_tag}.json").exists() and \
        (EV / f"producer_identity_willow_diff_{pid_tag}.json").exists()
    if pid_ok:
        pid = J(f"producer_identity_willow_{pid_tag}.json")
        judged_by_this_tree(pid, "producer identity")
        want = ({f"willow_{pl}_{b}_r{r}" for pl in ("d3_at_q4_5", "d5_at_q8_7") for b in "XZ" for r in (13, 30)}
                | pre_cfgs | {"fresh_sim_d5_r25"})
        if set(pid["configs"]) != want:
            raise SystemExit("producer identity: not the development and held-out configurations of these placements")
        pdiff = J(f"producer_identity_willow_diff_{pid_tag}.json")
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
            if not pid_ok:
                M[f"Dev{tag}{word}Pct"] = PENDING
                if tag == "DFive":
                    M[f"DevDFive{word}NotProvenPerRound"] = PENDING
                continue
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
    if not pid_ok:
        for k in ("IdentFreshChunks", "IdentHeldChunks", "IdentDevChunks", "IdentDevIdentical", "IdentDevDiffShots",
                  "IdentDevStatusChanged", "IdentDevShots"):
            M[k] = PENDING
        pid = {"configs": {}}
    held = [c for n, c in pid["configs"].items() if c["sample"] == "held-out"]
    fresh = [c for n, c in pid["configs"].items() if c["sample"] == "fresh-seed"]
    devc = [c for n, c in pid["configs"].items() if c["sample"] == "development"]
    if pid_ok:
        if not fresh or any(c["identical"] != c["chunks"] for c in fresh):
            raise SystemExit("producer identity: the fresh-seed receipts were not reproduced byte for byte")
        M["IdentFreshChunks"] = I(sum(c["chunks"] for c in fresh))
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
        rp = replay_path(nm)
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
    # re-run 2026-11-03 with the input-domain-fixed checker and the rebuilt battery, beside (not over) the cited record
    ie = J("ibm_live/d9nt2imij12s73fu9ihg.epsilon_2026-11-03.json")
    judged_by_this_tree(ie, "IBM epsilon re-certification")
    ie_old = J("ibm_live/d9nt2imij12s73fu9ihg.epsilon.json")
    if (ie["status"], ie["receipts_sha256"]) != (ie_old["status"], ie_old["receipts_sha256"]):
        raise SystemExit("IBM epsilon re-certification: the re-run's receipts differ from the cited record's")
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
    M["FaultsAllSyndromePct"] = PF(allsyn, 1)
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


    # ONE CONVENTION (external review 1, 2026-11-03: Sec. IV used one-sided bounds and Sec. VII two-sided endpoints):
    # an interval is the exact two-sided 95% Clopper-Pearson interval; a bound stated alone ("at most", a zero count)
    # is the exact ONE-sided 95% upper bound, 1 - 0.05^(1/n) when k = 0. And a false-alarm RATE is conditional on the
    # corrections that could raise a false alarm -- those within eps_max of optimal (external audit 2, Sec. 8).
    def cp_upper1(k: int, n: int) -> float:
        return 1.0 if k == n else float(_beta.ppf(0.95, k + 1, n - k))

    stale_ok = dsum(so, "n") - dsum(so, "defective")
    for tag, k, n in (("FaBase", dsum(clean, "false_alarms"), dsum(clean, "n") - dsum(clean, "defective")),
                      ("FaFull", dsum(clean_full, "false_alarms"), dsum(clean_full, "n") - dsum(clean_full, "defective")),
                      ("FaStale", dsum(so, "false_alarms"), stale_ok),
                      ("Ctrl", dsum(ctrl, "false_alarms"), dsum(ctrl, "n"))):
        if k:
            lo, hi = cp(k, n)
            M[f"CI{tag}Lo"], M[f"CI{tag}Hi"] = SIG3(lo, "floor"), SIG3(hi, "ceil")
        else:
            M[f"CI{tag}Lo"], M[f"CI{tag}Hi"] = "0", SIG3(cp_upper1(0, n), "ceil")
    M["DetStaleAcceptable"] = I(stale_ok)
    # the same base shots at every drift factor: which shots the false alarms fell on (stale_false_alarm_shots.py)
    sfa = J("stale_false_alarm_shots.json")
    if sfa["swept_false_alarms"] != dsum(so, "false_alarms"):
        raise SystemExit("stale calibration: the shot-level re-run does not reproduce the false-alarm count")
    M["DetStaleFalseAlarmDistinct"] = I(sfa["distinct_base_shots_among_them"])
    swept_shots = {(d, s) for d, rows in sfa["rows"].items() for f, r in rows.items() if f != "1.0"
                   for s in r["false_alarm_base_shots"]}
    ctrl_shots = {(d, s) for d, rows in sfa["rows"].items() for s in rows["1.0"]["false_alarm_base_shots"]}
    if len(swept_shots) == 1 and swept_shots == ctrl_shots:
        M["DetStaleFalseAlarmWhere"] = ("on a single base shot, the one that also raised the false alarm of the "
                                        "drift-1.0 control")
    else:
        M["DetStaleFalseAlarmWhere"] = f"on {I(len(swept_shots))} distinct base shots"
    if not (len(swept_shots) == 1 and sfa["swept_false_alarms"] >= 2):
        raise SystemExit("stale calibration: the text says the swept false alarms are one shot seen at more than one "
                         "drift factor")
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

    # ------------------------------------------------ gap distribution (preregistered, 2026-11-03)
    # Sec. VIII now comes from the preregistered run (docs/PREREG_gap_distribution_2026-11-03.md): fresh seeds,
    # selections projected onto edges mod 2, exact rationals, gap INTERVALS against the epsilon-certified reference.
    # The 2026-07-31 and 2026-11-02 artifacts used the defective scorer (external audit 2, F01); what they got wrong is
    # typeset from their re-run with stored selections (gap_distribution_original_rescore.py), never from themselves.
    gv = J("gap_distribution_v4/prereg_2026-11-03/summary.json")
    judged_at_its_commit(gv, "preregistered gap distribution")
    # the run recorded HEAD when it FINISHED; it started from 2008c81f, and the text says its judged files are
    # byte-identical at the two commits (final referee, 2026-11-03)
    def _blob(rev, path):
        return __import__("subprocess").run(["git", "rev-parse", f"{rev}:{path}"], cwd=ROOT, capture_output=True,
                                            text=True).stdout.strip()
    if not _git_available():
        NO_REPOSITORY.append("gap distribution start commit")
        print("NOTE: gap distribution: no repository here, so the judged files could not be compared between the "
              "start commit 2008c81f and the recorded one; the record pins them by sha256")
    elif any(_blob("2008c81f", p) != _blob(gv["head"]["commit"], p) or not _blob("2008c81f", p)
             for p in gv["head"]["sha256"]):
        raise SystemExit("gap distribution: a judged file differs between the start commit and the recorded one")
    M["GapRunCommit"] = gv["head"]["commit"][:8]
    if gv["argv"] != ["--seed", "4108100", "--shots", "2000", "--tag", "prereg_2026-11-03"]:
        raise SystemExit("gap distribution: not the preregistered command")
    if [r["seed"] for r in gv["rows"]] != [4108103, 4108105, 4108107] or any(
            r["shots"] != 2000 or r["rounds"] != r["distance"] or r["p"] != 0.005 for r in gv["rows"]):
        raise SystemExit("gap distribution: not the preregistered seeds, shots, rounds or noise")
    rj = J("gap_distribution_v4/prereg_2026-11-03/rejudged.json")
    judged_by_this_tree(rj, "gap distribution re-judged by this tree's checker")
    if not rj["all_identical"] or rj["run"]["commit"] != gv["head"]["commit"]:
        raise SystemExit("gap distribution: this tree's checker does not reproduce the run's reference verdicts")
    arms = (("bposd_merged", r"BP-OSD, merged, $s=0.625$"), ("bposd_component", r"BP-OSD, components, $s=1$"),
            ("belief_matching", r"belief-matching"))
    for row in gv["rows"]:
        st = row["bposd_settings"]
        if (st["bposd_merged"]["ms_scaling_factor"], st["bposd_component"]["ms_scaling_factor"]) != ("0.625", "1.0"):
            raise SystemExit("gap distribution: the BP-OSD arms did not run with the preregistered scaling")
        for a, _lbl in arms:
            x = row["decoders"][a]
            if x["syndrome_inconsistent"] or x["epsilon_reference"]["undecided_by_eps"]:
                raise SystemExit(f"gap distribution: {a} inconsistent or undecided shots; the text says none")
    lines = [r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{proven within $\varepsilon_{\max}$ of optimal} & proven miss &"
             r" \multicolumn{3}{c}{excess (upper end)} \\",
             r"$d$ & decoder & $N$ & \% & 95\% interval & beyond $\varepsilon_{\max}$ & median & p99 & max \\",
             r"\midrule"]
    for row in gv["rows"]:
        for i, (a, lbl) in enumerate(arms):
            e = row["decoders"][a]["epsilon_reference"]
            lo, hi = e["clopper_pearson_95"]["proven_eps_optimal"]
            lines.append(rf"{'$' + str(row['distance']) + '$' if i == 0 else ''} & {lbl} & {I(e['n'])} & "
                         rf"{PF(e['eps_optimal_rate'])} & {PF(lo)}--{math.ceil(1e4 * hi) / 100:.2f} & "
                         rf"{I(e['proven_miss_beyond_eps'])} & {F(e['gap_hi']['median'], 2)} & "
                         rf"{F(e['gap_hi']['p99'], 2)} & {F(e['gap_hi']['max'], 2)} \\")
        if row is not gv["rows"][-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files["tab_gaps.tex"] = "\n".join(lines) + "\n"

    def per_d(f):
        return " / ".join(f(r) for r in gv["rows"])
    M["GapShotsPerDistance"] = I(2000)
    M["GapSeeds"] = "4108103, 4108105 and 4108107"
    M["GapNontrivial"] = per_d(lambda r: I(r["nontrivial"]))
    M["GapEpsRef"] = per_d(lambda r: I(r["reference"]["epsilon_accepted"]))
    M["GapExactRef"] = per_d(lambda r: I(r["reference"]["exact_checker_accepted"]))
    M["GapMergeRelDiff"] = SCI(max(r["merged_weight_max_relative_difference"] for r in gv["rows"]))
    M["GapRejudged"] = I(sum(r["identical"] for r in rj["rows"]))
    for a, tag in (("bposd_merged", "Merged"), ("bposd_component", "Component"), ("belief_matching", "Bm")):
        M[f"Gap{tag}EpsOptPct"] = per_d(lambda r: PF(r["decoders"][a]["epsilon_reference"]["eps_optimal_rate"]))
        M[f"Gap{tag}PNinetyNine"] = per_d(lambda r: F(r["decoders"][a]["epsilon_reference"]["gap_hi"]["p99"], 1))
        M[f"Gap{tag}Max"] = F(max(r["decoders"][a]["epsilon_reference"]["gap_hi"]["max"] for r in gv["rows"]), 1)
        M[f"Gap{tag}ExactSubsetHitPct"] = per_d(lambda r: PF(r["decoders"][a]["exact_reference"]["hit_rate"]))
    M["GapComponentCancelled"] = per_d(lambda r: I(r["decoders"]["bposd_component"]["shots_with_cancelled_copies"]))
    if any(r["decoders"][a]["shots_with_cancelled_copies"] for r in gv["rows"]
           for a in ("bposd_merged", "belief_matching")):
        raise SystemExit("gap distribution: the text says only the component arm selects parallel copies")

    def er(r, a):
        return r["decoders"][a]["epsilon_reference"]
    # the comparative sentences of Sec. VIII, each held to the data at EVERY distance (final referee, MAJOR 2: "the
    # input model matters more than the decoder" was typeset unguarded and was false at d=7)
    for r in gv["rows"]:
        if not (er(r, "bposd_merged")["eps_optimal_rate"] > er(r, "belief_matching")["eps_optimal_rate"]
                and er(r, "bposd_merged")["eps_optimal_rate"] > er(r, "bposd_component")["eps_optimal_rate"]):
            raise SystemExit(f"gap distribution d={r['distance']}: the text says merged BP-OSD lands on the minimum "
                             f"more often than belief-matching and than component BP-OSD at every distance")
        if not er(r, "bposd_component")["gap_hi"]["p99"] > er(r, "bposd_merged")["gap_hi"]["p99"]:
            raise SystemExit(f"gap distribution d={r['distance']}: the text says the component arm's 99th-percentile "
                             f"excess is larger at every distance")
    lf = J("gap_distribution_v4/prereg_2026-11-03/logical_failures.json")
    if lf["run"] != gv["head"] or [x["distance"] for x in lf["rows"]] != [r["distance"] for r in gv["rows"]] or \
            [x["nontrivial"] for x in lf["rows"]] != [r["nontrivial"] for r in gv["rows"]]:
        raise SystemExit("gap logical failures: not scored on the preregistered run's shots")
    runner_rel = "experiments/m0_prior_art/gap_distribution.py"
    if _git_available():
        ran = __import__("subprocess").run(["git", "show", f"{gv['head']['commit']}:{runner_rel}"], cwd=ROOT,
                                           capture_output=True, text=True).stdout
        where = "the runner at the run's commit"
    else:
        # a reader's extracted bundle: the run's commit cannot be opened, so the SHIPPED runner is read, and the
        # output says whether its bytes are the run's (the record pins them by sha256)
        ran = (ROOT / runner_rel).read_text(encoding="utf-8")
        same = _sha_lf((ROOT / runner_rel).read_bytes()) == gv["head"]["sha256"].get(runner_rel)
        NO_REPOSITORY.append("gap distribution runner at its commit")
        print(f"NOTE: gap distribution: no repository here, so the runner at commit {gv['head']['commit'][:10]} could "
              f"not be opened; belief-matching's setting is read from the shipped runner, whose bytes "
              f"{'are' if same else 'are NOT'} the run's (sha256 in the record)")
        where = "the shipped runner"
    if "BeliefMatching(dem, max_bp_iters=20)" not in ran:
        raise SystemExit(f"gap distribution: the text states belief-matching ran with max_bp_iters=20; {where} does "
                         "not say so")
    for a, tag in (("pymatching", "Pm"), ("bposd_merged", "Merged"), ("bposd_component", "Component"),
                   ("belief_matching", "Bm")):
        M[f"GapLogical{tag}"] = " / ".join(I(x["logical_failures"][a]) for x in lf["rows"])
    # how wrong the published section was, from the re-run of both published artifacts with selections kept
    orig = J("gap_distribution_v4/original_rescored.json")["results"]
    tv, f3 = orig["table_v_2026-07-31"]["rows"], orig["fig3_2026-11-02"]["rows"]
    if not all(r["reproduces_stored_artifact"] for x in (tv, f3) for r in x.values()):
        raise SystemExit("gap rescore: the re-run does not reproduce a published artifact")
    ds = ("3", "5", "7")
    M["GapOrigRefTableV"] = " / ".join(I(tv[d]["reference_shots"]) for d in ds)
    M["GapOrigOverTableV"] = " / ".join(I(tv[d]["overcounted_shots"]) for d in ds)
    M["GapOrigOverTableVTotal"] = I(sum(tv[d]["overcounted_shots"] for d in ds))
    M["GapOrigRefTableVTotal"] = I(sum(tv[d]["reference_shots"] for d in ds))
    M["GapOrigOverFigTotal"] = I(sum(f3[d]["overcounted_shots"] for d in ds))
    M["GapOrigRefFigTotal"] = I(sum(f3[d]["reference_shots"] for d in ds))
    M["GapOrigTrueZeroTableVTotal"] = I(sum(tv[d]["overcounted_true_gap_exactly_zero"] for d in ds))
    if any(tv[d]["overcounted_bp_converged"] != tv[d]["overcounted_shots"] for d in ds):
        raise SystemExit("gap rescore: the text says every overcounted shot was a converged BP hard decision")
    M["GapOrigOvercountLo"], M["GapOrigOvercountHi"] = enclose(min(tv[d]["overcount_min"] for d in ds),
                                                                max(tv[d]["overcount_max"] for d in ds), 2)
    # two decimals, enclosing (final referee, 2026-11-03: one decimal ceiled printed 146.4 for 146.309, which the
    # preregistration's text, truncating, gives as 146.3)
    M["GapOrigMaxCopies"] = I(max(tv[d]["max_copies_of_one_edge"] for d in ds))
    M["GapOrigPublishedMax"] = F(max(tv[d]["published"]["max"] for d in ds), 0)
    M["GapOrigRescoredMax"] = F(max(tv[d]["rescored"]["max"] for d in ds), 1)
    M["GapOrigPublishedPNinetyNine"] = " / ".join(F(tv[d]["published"]["p99"], 0) for d in ds)
    M["GapOrigRescoredPNinetyNine"] = " / ".join(F(tv[d]["rescored"]["p99"], 1) for d in ds)
    M["GapOrigPublishedExactPct"] = " / ".join(PF(tv[d]["published"]["frac_zero"], 1) for d in ds)
    M["GapOrigRescoredExactPct"] = " / ".join(PF(tv[d]["rescored"]["exact_zero_rate"], 1) for d in ds)
    M["GapOrigTolTiny"] = I(sum(tv[d]["rescored"]["tiny_nonzero_below_1e9"] for d in ds))
    M["GapOrigNegExact"] = I(sum(tv[d]["rescored"]["negative_exact"] for d in ds))

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
    M["CostRatioCheckBatch"] = I(dc["certify_cost_check_mean_over_batch_2sf"])   # mean over mean, as the ratio beside it
    M["CheckMsMean"] = f"{cc['check_ms']['mean']:.1f}"
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

    def two_sig(x: float) -> int:
        return int(round(x, -max(0, len(str(int(x))) - 2)))
    # every printed cost ratio must follow from the printed means it divides (final referee, 2026-11-03)
    for ratio, num, den, scale in (("CostRatioCheckBatch", "CheckMsMean", "DecodeUsBatch", 1000),
                                   ("CostRatioBatch", "CertifyMsMean", "DecodeUsBatch", 1000),
                                   ("CostRatioPerShot", "CertifyMsMean", "DecodeUsPerShot", 1000)):
        if I(two_sig(float(M[num]) * scale / float(M[den]))) != M[ratio]:
            raise SystemExit(f"cost: {ratio} = {M[ratio]} does not follow from the printed {num} / {den}")
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
    M["AuditExactSamePct"] = PF(same_all / acc_all, 0)
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

    # ------------------------------- the same section's receipts, KEPT (external review 1, 2026-11-03)
    cmr = J("certified_measured_receipts/summary.json")
    judged_by_this_tree(cmr, "measured-model receipts")
    t = cmr["totals"]
    if (t["nontrivial"], t["exact_checker_accepted"]) != (cm["nontrivial"], cm["certified_under_measured"]):
        raise SystemExit("measured-model receipts: the receipt run does not reproduce the lane's counts")
    if t["forgeries_accepted"] or any(r["battery_attacks_not_false"] for r in cmr["runs"]) or \
            t["verifiers_agree_on"] != t["nontrivial"]:
        raise SystemExit("measured-model receipts: a forgery accepted, an attack not false, or the verifiers split")
    for r in cmr["runs"]:
        body_r = gzip.decompress((ROOT / r["receipts"]).read_bytes())
        if hashlib.sha256(body_r).hexdigest() != r["receipts_sha256"]:
            raise SystemExit(f"measured-model receipts: {r['receipts']} is not the file recorded")
    M["CertMeasEpsAccepted"] = I(t["epsilon_accepted"])
    M["CertMeasForgeries"] = I(t["forgeries_attempted"])
    def derived(key: str, value: int) -> str:
        """A sum no single artifact states: typeset only if tools/derive_claims.py recorded the same value, so the
        printed number traces to evidence/derived_claims.json (claims_provenance_gate)."""
        if dc.get(key) != value:
            raise SystemExit(f"derived_claims.json has {key} = {dc.get(key)}, the artifacts give {value}: "
                             f"rerun tools/derive_claims.py")
        return I(value)

    M["CertMeasBoundForgeries"] = derived("p1_cert_measured_forgeries_refused_by_bound",
                                          sum(r["forgeries_by_class"].get(c, 0) for r in cmr["runs"]
                                              for c in ("A_suboptimal", "D_spurious_cycle",
                                                        "G_dual_inflated_to_meet_a_worse_primal")))

    # ------------------------------- the forgery battery: which check refused each class
    if es is not None:
        fb = es["forgeries_by_class"]
        M["EpsSimForgeriesBound"] = I(fb.get("A_suboptimal", 0))
        M["EpsSimForgeriesDoubled"] = I(sum(fb.get(c, 0) for c in ("D_spurious_cycle",
                                                                      "G_dual_inflated_to_meet_a_worse_primal",
                                                                      "H_negative_dual_hides_inflation")))
        M["EpsSimForgeriesSyndrome"] = I(sum(fb.get(c, 0) for c in ("E_dropped_edge", "F_duplicated_edge",
                                                                       "I_mismatched_pair")))
        M["EpsSimForgeriesNonFinite"] = I(sum(fb.get(c, 0) for c in ("N_nan_dual", "N_inf_dual")))
    else:
        for k in ("EpsSimForgeriesBound", "EpsSimForgeriesDoubled", "EpsSimForgeriesSyndrome",
                  "EpsSimForgeriesNonFinite"):
            M[k] = PENDING
    # the rebuilt battery over the strict replay of every released run (replay_epsilon_receipts.py --forgeries)
    fbat, par, have_par = Counter(), Counter(), []
    for nm in ["sim_d5_r25", "fresh_sim_d5_r25"] + [f"willow_{c}" for c in wcfg] + sorted(pre_cfgs):
        rp = EV / "epsilon_rerun" / "replay_2026-11-03" / f"replay_{nm}.json"
        if rp.exists():
            fb = json.loads(rp.read_text(encoding="utf-8"))["results"][0].get("forgery_battery", {})
            fbat.update(fb.get("refusals_by_class_and_verdict", {}))
            par.update(fb.get("parity_adversary", {}))
            # replayed with the CALIBRATED adversary judged by the parity-blind copies (2026-11-03, final referee)
            have_par.append("accepted_by_both_parity_blind_copies" in fb.get("parity_adversary", {}))
        else:
            have_par.append(False)
    # every configuration replayed WITH the parity adversary, or the replay numbers are pending
    if fbat and all(REPLAYS_USED) and all(have_par):
        bound = sum(v for k, v in fbat.items() if k.endswith(":NOT_PROVEN"))
        M["ReplayForgeries"] = derived("p1_strict_replay_forgeries", sum(fbat.values()))
        M["ReplayForgeriesBound"] = derived("p1_strict_replay_forgeries_refused_by_bound", bound)
        # the T'-oddness adversary, part of the replay battery since 2026-11-03 (final referee, minor 5)
        if set(k for k in fbat if k.startswith("P_parity")) != {"P_parity_not_T_prime_odd:INVALID_INPUT"} or \
                fbat["P_parity_not_T_prime_odd:INVALID_INPUT"] != par.get("attempted", 0):
            raise SystemExit("replay battery: the parity adversary is absent or was refused other than as invalid input")
        M["ReplayParity"] = derived("p1_strict_replay_parity_forgeries", par["attempted"])
        # REAL = accepted by the parity-blind copies of BOTH verifiers, RUN on each forgery (final referee,
        # 2026-11-03: the old counter was an inequality with no upper end and counted forgeries those copies refuse)
        if par.get("no_scaling_fools_a_parity_blind_checker", 0):
            raise SystemExit("replay battery: a parity forgery that no scaling makes acceptable to a parity-blind "
                             "checker; the text says every one was")
        M["ReplayParityFool"] = derived("p1_strict_replay_parity_accepted_by_blind_copies",
                                        par.get("accepted_by_both_parity_blind_copies", 0))
        if not (par.get("accepted_by_both_parity_blind_copies") == par.get("accepted_by_parity_blind_check_epsilon")
                == par.get("accepted_by_parity_blind_verify_b") == par["attempted"]):
            raise SystemExit("replay battery: the text says every parity forgery is accepted by both parity-blind copies")
        if par.get("no_disjoint_cycle", 0) or par["attempted"] != fbat.get("D_spurious_cycle:NOT_PROVEN", -1):
            raise SystemExit("replay battery: the text says the parity adversary attacked every attacked receipt")
        for c in ("D_spurious_cycle", "G_dual_inflated_to_meet_a_worse_primal"):
            if fbat.get(f"{c}:INVALID_INPUT"):
                raise SystemExit(f"replay battery: {c} was refused as malformed; the text says it reaches the bound")
        M["ReplayStrict"] = "every"
    else:
        for k in ("ReplayForgeries", "ReplayForgeriesBound", "ReplayStrict", "ReplayParity", "ReplayParityFool"):
            M[k] = PENDING

    # ------------------------------- the JavaScript epsilon-checker and the differential fuzzer
    fz0 = J("epsilon_checker_fuzz.json")
    fz1 = J("epsilon_checker_fuzz_after_fix.json")
    if fz1["valid_domain_disagreements"] or fz1["disagreeing_classes"] or fz0["valid_domain_disagreements"]:
        raise SystemExit("fuzz: the text says valid inputs never split the implementations and the fix closed every "
                         "malformed class")
    M["FuzzCases"] = I(fz1["cases"])
    M["FuzzValidCases"] = I(fz1["valid_domain_cases"])
    M["FuzzClassesBefore"] = I(len(fz0["disagreeing_classes"]))
    M["FuzzDisagreementsBefore"] = derived("p1_epsilon_fuzz_disagreements_before_fix",
                                           sum(v.get("disagree", 0) for v in fz0["per_class"].values()))
    rr = fz1["real_receipts_js_vs_recorded"]
    if any(x["identical_status_epsilon_bound"] != x["receipts"] for x in rr.values()):
        raise SystemExit("fuzz: the JavaScript checker split from the recorded verdict on a real receipt")
    M["JsReplayReceipts"] = derived("p1_js_epsilon_replayed_receipts",
                                    sum(x["receipts"] for x in rr.values()) + t["nontrivial"])
    M["JsReplayConfigs"] = I(len(rr))
    ev = json.loads((ROOT / "conformance" / "epsilon_vectors.json").read_text(encoding="utf-8"))
    M["EpsVectors"] = I(len(ev["vectors"]))
    M["EpsVectorClasses"] = I(len(ev["classes"]))

    # ------------------------------- R177 reproduced: PyMatching's own dual (pymatching_native_dual.py)
    nd = J("pymatching_native_dual.json")["configs"]

    def ndv(cfg, arm, key):
        return nd[cfg][arm][key]
    for cfg, tag in (("sim_d5_r25", "Sim"), ("willow_d5_at_q8_7_X_r30", "WFive"),
                     ("willow_d7_at_q6_7_X_r30", "WSeven")):
        g = nd[cfg]["grid"]
        r = nd[cfg]["raw"]
        if g["status"].get("EXACT_OPTIMAL", 0) != g["nontrivial"] or g["correction_equals_plain_pymatching"] != \
                g["nontrivial"] or r["correction_equals_plain_pymatching"] != r["nontrivial"]:
            raise SystemExit(f"native dual {cfg}: not every rounded-graph certificate exact, or a correction changed")
        M[f"Nd{tag}N"] = I(g["nontrivial"])
        M[f"Nd{tag}RawEpsMedian"] = SCI(r["eps_median"])
        M[f"Nd{tag}RawEpsMax"] = SCI(r["eps_max_seen"])
        M[f"Nd{tag}RawProven"] = I(r["status"].get("EXACT_OPTIMAL", 0) + r["status"].get("EPSILON_OPTIMAL", 0))
        M[f"Nd{tag}DecodeUs"] = F(r["decode_us_median"], 0)
        M[f"Nd{tag}DualUs"] = F(r["decode_with_duals_us_median"], 0)
        M[f"Nd{tag}ConvertUs"] = F(r["convert_us_median"], 0)
        M[f"Nd{tag}CheckMs"] = F(r["check_ms_median"], 1)
        M[f"Nd{tag}GridBits"] = I(g["grid_bits"])
    M["NdRawEpsMaxAll"] = SCI(max(nd[c]["raw"]["eps_max_seen"] for c in nd))
    M["NdRawProvenTotal"] = I(sum(nd[c]["raw"]["status"].get("EXACT_OPTIMAL", 0)
                                  + nd[c]["raw"]["status"].get("EPSILON_OPTIMAL", 0) for c in nd))
    # The GRID arm's dual judged on the published doubles is its own measurement (final referee, MAJOR 1): it is not
    # the RAW arm's, and the two must never be typeset under one name again.
    for cfg, tag in (("sim_d5_r25", "Sim"), ("willow_d5_at_q8_7_X_r30", "WFive"), ("willow_d7_at_q6_7_X_r30", "WSeven")):
        M[f"Nd{tag}GridOnRawEpsMedian"] = SCI(nd[cfg]["grid"]["same_dual_on_raw_graph"]["eps_median"])
    M["NdGridOnRawEpsMaxAll"] = SCI(max(nd[c]["grid"]["same_dual_on_raw_graph"]["eps_max_seen"] for c in nd))
    M["NdGridOnRawProvenTotal"] = I(sum(nd[c]["grid"]["same_dual_on_raw_graph"]["status"].get(s, 0) for c in nd
                                        for s in ("EXACT_OPTIMAL", "EPSILON_OPTIMAL")))
    # ... and the two measurements the first record did not make (pymatching_native_dual_arms.py): the RAW dual checked
    # exactly against PyMatching's own integer weights, and the corrections of the two arms against the certified ones
    na = J("pymatching_native_dual_arms.json")["configs"]
    if set(na) != set(nd):
        raise SystemExit("native dual: the two records do not cover the same configurations")
    for cfg, r in na.items():
        n = r["nontrivial"]
        if n != nd[cfg]["raw"]["nontrivial"] or r["normalising_constant_equals_2c"] != n:
            raise SystemExit(f"native dual {cfg}: the decoder's scale is not 2c on every shot, or the shots differ")
        if r["raw_dual_on_pymatching_integer_weights"] != {"EXACT_OPTIMAL": n}:
            raise SystemExit(f"native dual {cfg}: the text says the dual is exact on PyMatching's integers on every shot")
        c = r["corrections"]
        if not (c["grid_eq_raw"] == c["raw_eq_released"] == c["grid_eq_released"] == c["released_compared"] == n):
            raise SystemExit(f"native dual {cfg}: the text says both arms return the certified correction on every shot")
        if not (c.get("control_reoriented_still_equal") == c.get("control_dropped_edge_caught") == n):
            raise SystemExit(f"native dual {cfg}: the comparison's positive controls did not run or did not hold")
    M["NdIntShots"] = I(sum(r["nontrivial"] for r in na.values()))
    for cfg, tag in (("sim_d5_r25", "Sim"), ("willow_d7_at_q6_7_X_r30", "WSeven")):
        cu = na[cfg]["cost_per_call_us"]
        M[f"Nd{tag}PlainUs"] = F(cu["plain_decode_median"], 0)
        M[f"Nd{tag}DualConvUs"] = F(cu["with_duals_plus_conversion_median"], 0)
    # per-call medians of (decode with duals + region-to-cut conversion) over a plain decode, as the arms record
    # measured them for the INTEGER claim: the conversion there builds the cut sets in PyMatching's own units. The first
    # record's conversion also rescaled every value into binary64 weight units (pymatching_native_dual.py, the
    # Fraction division per set), which the integer claim does not need; its ratio is stated beside, as what it timed
    # (final referee, 2026-11-03: the two were presented as run-to-run variation)
    now = [r["cost_per_call_us"]["ratio_of_medians"] for r in na.values()]
    before = [(nd[c]["raw"]["decode_with_duals_us_median"] + nd[c]["raw"]["convert_us_median"])
              / nd[c]["raw"]["decode_us_median"] for c in nd]
    M["NdCallRatioLo"], M["NdCallRatioHi"] = enclose(min(now), max(now), 1)
    M["NdRescaleRatioLo"], M["NdRescaleRatioHi"] = enclose(min(before), max(before), 1)

    # ------------------------------- the held-out distance-7 sample (docs/PREREG_heldout_d7_2026-11-03.md)
    # Two artifacts can carry it: the complete run (heldout_willow_d7.py --merge) or, when the preregistered 40
    # core-hour rule stopped it, the partial report (heldout_d7_partial.py). With neither, the run is still going and
    # the release is held: HoDSevenState stays pending, so a release build refuses.
    full_p, part_p = EV / "epsilon_rerun" / "heldout_d7.json", EV / "epsilon_rerun" / "heldout_d7_partial.json"
    if full_p.exists() or part_p.exists():
        hd = J("epsilon_rerun/heldout_d7.json") if full_p.exists() else J("epsilon_rerun/heldout_d7_partial.json")
        M.update(heldout_d7_macros(hd))
        if hd.get("schema") == "oneq-heldout-willow-d7-partial/1":
            # who produced the sample: the frozen producer's modules at the preregistration's commit, the deviation's
            # driver and watchdog at theirs, and the report at a commit (or this draft)
            h = hd["head"]
            judged_at_its_commit({"head": {"commit": h["producer"]["commit"], "sha256": h["producer"]["sha256"],
                                           "uncommitted_changes": False}}, "held-out d=7 producer")
            judged_at_its_commit({"head": {"commit": h["deviation"]["commit"], "sha256": h["deviation"]["sha256"],
                                           "uncommitted_changes": False}}, "held-out d=7 deviation driver")
            judged_by_this_tree({"head": h["report"]}, "held-out d=7 partial report")
            need(h.get("copied_from", {}).get("index_receipts_and_instances_identical") is True,
                 "held-out d=7: the sample here is not verified identical to the run's tree")
            rec_dir = ROOT / h["records"]["dir"]
            for nm, sha_ in h["records"]["sha256"].items():
                need((rec_dir / nm).is_file() and _sha_lf((rec_dir / nm).read_bytes()) == sha_,
                     f"held-out d=7: run record {nm} missing or changed")
            stop = json.loads((rec_dir / "d7_stop_record.json").read_text(encoding="utf-8"))
            need(stop == hd["stop_record"], "held-out d=7: the report's stop record is not the one on disk")
            need(stop["core_hours"] == 40.0 and stop["stopped"] == "2026-11-04T03:55:29"
                 and stop["start"] == "2026-11-03T14:35:29" and stop["index_unchanged_by_the_stop"] is True,
                 "held-out d=7: the stop is not the preregistered 40 core-hours from the original start")
            need(set(hd["arms"]) == {"heldout_d7_X_r50", "heldout_d7_X_r10", "samedev_d7_X_r30", "heldout_d7_Z_r50",
                                     "heldout_d7_Z_r10", "samedev_d7_Z_r30"}
                 and all(v["completed"]["chunks"] > 0 for v in hd["arms"].values()),
                 "held-out d=7: the text says every arm completed chunks")
            # every completed receipt replayed through both verifiers with the battery (partial-plan mode)
            rp = J("epsilon_rerun/replay_heldout_d7_2026-11-05.json")
            need(rp["kind"].startswith("PARTIAL-PLAN") and rp["battery_per_chunk"] >= 1,
                 "held-out d=7: the replay is not the partial-plan replay with the battery")
            res = {r["config"]: r for r in rp["results"]}
            need(set(res) == set(hd["arms"]), "held-out d=7: the replay does not cover every arm")
            rr_f = Counter()
            for arm, v in hd["arms"].items():
                r = res[arm]
                inst = json.loads((EV / "instances" / f"matching_{arm}.json").read_text(encoding="utf-8"))
                need(r.get("partial") is True and r["seeds"] == v["completed_seeds"]
                     and r["receipts_rechecked"] == v["completed"]["nontrivial"] == r["nontrivial"]
                     and r["accepted"] == v["completed"]["accepted"]
                     and r["status"].get("NOT_PROVEN", 0) == v["completed"]["not_proven"]
                     and r["instance_sha256"] == inst["edges_sha256"]
                     and Fraction(r["eps_max"]) == Fraction(1, 10**6)
                     and not r["battery"].get("substituted_graph_accepted", 0),
                     f"held-out d=7: the replay of {arm} does not reproduce its completed chunks")
                fb = r["forgery_battery"]
                need(all(k.endswith((":INVALID_INPUT", ":NOT_PROVEN")) for k in fb["refusals_by_class_and_verdict"])
                     and fb["parity_adversary"].get("accepted_by_both_parity_blind_copies", 0)
                     == fb["parity_adversary"].get("attempted", -1),
                     f"held-out d=7: a forgery accepted, or the parity adversary not real, in the replay of {arm}")
                rr_f.update(fb["refusals_by_class_and_verdict"])
            M["HoDSevenReplayReceipts"] = I(sum(r["receipts_rechecked"] for r in res.values()))
            M["HoDSevenReplayForgeries"] = I(sum(rr_f.values()))
            need(isinstance(rp.get("head"), dict) and "sha256" in rp["head"],
                 "held-out d=7: the replay record carries no head")
            if isinstance(rp.get("head"), dict) and "sha256" in rp["head"]:
                judged_by_this_tree({"head": rp["head"]}, "held-out d=7 replay")
            # the per-arm table: each view with its own sample size
            lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
                     r"arm & rounds & chunks & $N$ & proven & not proven & bound (\%) & prefix & worst (\%) \\",
                     r"\midrule"]
            # the six arms in the deviation's cycle order, then the pooled rows
            for arm in ("heldout_d7_X_r50", "heldout_d7_X_r10", "samedev_d7_X_r30", "heldout_d7_Z_r50",
                        "heldout_d7_Z_r10", "samedev_d7_Z_r30"):
                v, c = hd["arms"][arm], hd["arms"][arm]["completed"]
                label = ("held-out X" if arm.startswith("heldout_d7_X") else "held-out Z" if arm.startswith("heldout_d7_Z")
                         else "development X" if arm.endswith("X_r30") else "development Z")
                lines.append(rf"{label} & {v['rounds'][1:]} & {c['chunks']}/40 & {I(c['nontrivial'])} & "
                             rf"{I(c['accepted'])} & {I(c['not_proven'])} & "
                             rf"{SIG3(c['not_proven_upper_95_one_sided'], 'ceil')} & {v['prefix']['chunks']} & "
                             rf"{SIG3(v['dispatched_worst_case']['upper_95_one_sided'], 'ceil')} \\")
            lines.append(r"\midrule")
            for rr_, label in (("r50", "held-out, pooled"), ("r10", "held-out bracket, pooled"),
                               ("r30", "development, pooled")):
                row = hd["pooled"][rr_]
                ch = sum(hd["arms"][a]["completed"]["chunks"] for a in row["arms_with_data"])
                lines.append(rf"{label} & {rr_[1:]} & {ch}/80 & {I(row['nontrivial'])} & "
                             rf"{I(row['nontrivial'] - row['not_proven'])} & {I(row['not_proven'])} & "
                             rf"{SIG3(row['not_proven_upper_95_one_sided'], 'ceil')} & -- & "
                             rf"{SIG3(row['worst_case_upper_95_one_sided'], 'ceil')} \\")
            lines += [r"\bottomrule", r"\end{tabular}"]
            files["tab_heldout_d7.tex"] = "\n".join(lines) + "\n"
    else:
        M["HoDSevenState"] = PENDING

    # every strict-replay record the numbers above were read from names the tool and verifiers that wrote it, and
    # those are this tree's bytes (final referee, 2026-11-05: the 18 records carried no head)
    for rp_ in sorted((EV / "epsilon_rerun" / "replay_2026-11-03").glob("replay_*.json")):
        head_ = json.loads(rp_.read_text(encoding="utf-8")).get("head")
        need(isinstance(head_, dict) and head_.get("tool") == "tools/replay_epsilon_receipts.py"
             and "sha256" in head_, f"strict replay {rp_.name}: no head naming the tool that wrote it")
        if isinstance(head_, dict) and "sha256" in head_:
            judged_by_this_tree({"head": head_}, f"strict replay {rp_.name}")

    if DRAFT_UNCOMMITTED:
        print("DRAFT: typeset from bytes no commit holds yet:", DRAFT_UNCOMMITTED)

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
