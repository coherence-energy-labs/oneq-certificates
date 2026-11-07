r"""Every quantitative macro and table in paper 3 (exact circuit-level distances), generated from evidence.

Same contract as make_numbers.py and make_numbers_qldpc.py: the manuscript never types a number. It writes
\IbmD, and this script defines \IbmD by reading the committed artifact that established it. It REFUSES, with
the reason, when an input is missing or two inputs disagree.

    python paper/tools/make_numbers_distance.py

Writes paper/circuit-distance/generated/{numbers,tab_summary,tab_morph,tab_costs,tab_recert}.tex.

INPUTS (all committed):
  evidence/circuit_distance_exact/results.json           the lane's claims, as collect_results.py assembles them
  evidence/circuit_distance_exact/records/*.jsonl        the run records those claims rest on
  evidence/circuit_distance_exact/lean_r198_receipt.json the Lean receipt (sources, theorems, axioms)
  evidence/circuit_distance_exact/r205_cdp5/             two shipped CDP5 proofs, their models and the manifest
  evidence/circuit_distance_exact/r206_performance.json  later re-certification of the IBM T = 2 models
  evidence/circuit_distance_exact/r206_schedule_discovery/registry.json
  experiments/circuit_distance_exact/morph2026_targets.json   the polynomials of the 2026 targets
A few facts the lane records only in prose (the generic-solver stalls, the brute-force oracle, the Lean
kernel's cost, the exposure of the Lean findings, the literature's values, figure numbers and arXiv identifiers)
are read from RESULTS.md, FOLD_PROOF.md, lean/CertR198/SPEC.md and docs/RESEARCH_R205_CDP5.md by strict patterns.
A pattern that stops matching is a refusal, never a default: the sentence changed, so its numbers must be read
again. Each macro in numbers.tex is preceded by a comment naming the file it was read from.

BINDINGS checked before any number is used:
  * results.json equals what collect_results.py computes from records/ today (the lane test's own check), so the
    summary cannot drift from its records;
  * the Lean receipt certifies the current sources (lean_receipt.check(): no stale source, no sorryAx, no
    non-standard axiom);
  * every distance is DERIVED here from its records (certified weight w, witness weight w + 1) and the claim
    strings of results.json must state that same value;
  * every certificate a claim names was VERIFIED on the very model (sha256) its search ran on;
  * the two shipped CDP5 proofs and their models hash to the manifest, and those models are the certified ones.

ROUNDING, one rule, which the paper states (Sec. IX; external audit 4, X6): times, sizes and ratios computed here are
rounded to the nearest value at the digit shown, except that a printed MAXIMUM ("up to", "at most") is rounded up
(up()) and a printed RANGE encloses the data, its low end rounded down and its high end up (enclose(), down()).
Counts, weights and distances are exact. The few figures read from the lane's prose (the Lean kernel's cost) are
as those documents state them.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence" / "circuit_distance_exact"
REC = EV / "records"
LANE = ROOT / "experiments" / "circuit_distance_exact"
OUT = ROOT / "paper" / "circuit-distance" / "generated"
STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
# The values Pan's released research record reports for the two morphing circuits of 2407.16336 (exhaustive search,
# fold premise checked by computation, no certificate): research/audits/2026-10-08-E-019.md Sec. 2.1 (the [[72,12,6]]
# end-cycle circuits, 6/6) and 2026-10-08-E-017.md Sec. 8 (the [[144,12,12]] end-cycle circuit, 12), at commit
# 5ae848ab of github.com/GavinNull/gross-code-distance-12 (blobs eab30a71, 25afd539; also in release v1.0). Read
# 2026-11-03; docs/PAPER3_LITERATURE_SEARCH.md.
PAN_E019_MORPH72 = 6
PAN_E017_MORPH144 = 12
# IBM's depth-8 gross cycle in Pan's records on his main branch after v2.0: E-037 (commit 34ffcb14, 2026-10-25 21:39
# UTC = 2026-10-26 05:39 in the commit's own UTC+8, the date in the record's file name), 88
# VeriPB certificates of "no undetectable logical of weight <= 9" for R = 1..12; E-043 (commit 18e47dae, 2026-10-31),
# d_circ = 10 for every R (bulk criterion with two uncertified solvers, direct search R <= 3, persistence). Read
# 2026-11-03 at main HEAD 41c8f978; docs/PAPER3_LITERATURE_SEARCH.md.
PAN_E043_IBM = 10
# The three-colour cycle (Fig. 12 of 2603.05481) in Pan's released manuscript (release v2.0, release/arxiv/main.tex
# lines 260-265, read 2026-11-03): d_circ = 11 for every R under SD6, from the reduction of Strikis, Browne and
# Beverland (Prop. 3, Cor. 1) and d_ext = 11 by two exhaustive searches. Table I's prior-value column names it
# (external audit 4, 2.2); it must equal ours. docs/PAPER3_LITERATURE_SEARCH.md.
PAN_V2_TC = 11
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


# ------------------------------------------------------------------ reading and refusing

def refuse(why: str):
    raise SystemExit(f"REFUSED: {why}")


def need(ok, why: str) -> None:
    if not ok:
        refuse(why)


def rel(p: pathlib.Path) -> str:
    """A path for a message: repo-relative when it is inside the repository, else as given."""
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def J(path: pathlib.Path) -> dict:
    need(path.exists(), f"{rel(path)} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def JL(name: str) -> list[dict]:
    p = REC / name
    need(p.exists(), f"record {name} is missing")
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def TXT(rel: str) -> str:
    p = ROOT / rel
    need(p.exists(), f"{rel} is missing")
    return p.read_text(encoding="utf-8")


def RX(pattern: str, text: str, what: str) -> re.Match:
    m = re.search(pattern, text, re.S)
    need(m is not None, f"{what}: the source sentence changed shape; read its numbers again")
    return m


def one(values, what: str):
    """The single value a collection takes; refuse if it takes several or none."""
    s = set(values)
    need(len(s) == 1, f"{what}: expected one value, found {sorted(s, key=str)}")
    return s.pop()


def sha256(p: pathlib.Path) -> str:
    need(p.exists(), f"{rel(p)} is missing")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def word(w: str, what: str) -> int:
    need(w.lower() in NUMBER_WORDS, f"{what}: {w!r} is not a number word this script reads")
    return NUMBER_WORDS[w.lower()]


# ------------------------------------------------------------------ formatting

def I(n) -> str:
    """An exact count, LaTeX thousands separator."""
    if isinstance(n, float):
        need(n == int(n), f"{n} is not an integer count")
    return f"{int(n):,}".replace(",", "{,}")


def R0(x: float) -> str:
    """Rounded half up to an integer, grouped (Python's round() would send 2924.5 to 2924; the lane's documents,
    like a reader, round it to 2925)."""
    return I(math.floor(x + 0.5))


def D1(x: float) -> str:
    return f"{x:.1f}"


def SCI(x: float, k: int = 1) -> str:
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, k) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.{k}f}\times 10^{{{e}}}}}"


def enclose(lo: float, hi: float, k: int) -> tuple[str, str]:
    """A printed range must CONTAIN the data: floor the low end, ceiling the high end."""
    f = 10 ** k
    return f"{math.floor(lo * f + 1e-9) / f:.{k}f}", f"{math.ceil(hi * f - 1e-9) / f:.{k}f}"


def up(x: float, k: int) -> str:
    """A maximum ("up to", "at most") rounded UP at k decimals, so the printed bound still holds."""
    return enclose(x, x, k)[1]


def down(x: float, k: int) -> str:
    """The low end of a printed range rounded DOWN at k decimals."""
    return enclose(x, x, k)[0]


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}"


def MB(nbytes: int) -> str:
    return R0(nbytes / 1e6)


def KB(nbytes: int) -> str:
    return R0(nbytes / 1e3)


def code_params(s: str, what: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"\[\[(\d+),(\d+),(\d+)\]\]", s)
    need(m is not None, f"{what}: {s!r} is not a code parameter triple")
    return tuple(int(x) for x in m.groups())


def code_tex(n: int, k: int, d: int) -> str:
    return f"[[{n},{k},{d}]]"


def monomial(a: int, b: int = 0) -> str:
    def part(v, e):
        return "" if e == 0 else (v if e == 1 else f"{v}^{{{e}}}")
    return (part("x", a) + part("y", b)) or "1"


def trange(ts, what: str) -> tuple[int, int]:
    """(lo, hi) of a set of round counts that must be a contiguous run."""
    ts = sorted(set(ts))
    need(ts and ts == list(range(ts[0], ts[-1] + 1)), f"{what}: the rounds {ts} are not a contiguous run")
    return ts[0], ts[-1]


# ------------------------------------------------------------------ the bindings

def lane_modules():
    if str(LANE) not in sys.path:
        sys.path.insert(0, str(LANE))
    import collect_results as cr  # noqa: E402  (the lane's own assembler)
    import lean_receipt as lr  # noqa: E402
    return cr, lr


def check_results_current(results: dict) -> None:
    """results.json must be exactly what collect_results.py computes from the records now."""
    cr, _ = lane_modules()
    fresh = {"ibm_gross": cr.ibm_claim, "three_colour": cr.three_colour_claim,
             "three_colour_every_T": cr.three_colour_every_T_claim, "bb72": cr.bb72_claim,
             "morph72": cr.morph72_claim, "morph144": cr.morph144_claim, "morph2026": cr.morph2026_claim,
             "trusted_base": cr.dem_claim}
    need(set(results) - {"lane"} == set(fresh), "results.json carries claims this generator does not know")
    for key, fn in fresh.items():
        need(results[key] == fn(), f"results.json[{key}] is not what collect_results.py computes from the records "
                                   f"today; re-run collect_results.py and read the claims again")


def check_lean_receipt() -> dict:
    _, lr = lane_modules()
    bad = lr.check()
    need(not bad, "the Lean receipt does not certify the current sources: " + "; ".join(bad[:3]))
    return J(EV / "lean_r198_receipt.json")


# ------------------------------------------------------------------ the generator

def build() -> dict[str, str]:
    M: dict[str, tuple[str, str]] = {}

    def put(name: str, value, source: str) -> None:
        need(re.fullmatch(r"[A-Za-z]+", name) is not None, f"macro name {name} must be letters only")
        need(name not in M, f"macro {name} defined twice")
        # a source is a repository-relative name: an absolute path is this machine's, and a copy regenerates another
        need(isinstance(source, str) and re.match(r"[A-Za-z]:|/", source) is None,
             f"macro {name}'s source is not a repository-relative name: {source!r}")
        M[name] = (str(value), source)

    res = J(EV / "results.json")
    check_results_current(res)
    receipt = check_lean_receipt()
    results_md = TXT("experiments/circuit_distance_exact/RESULTS.md")
    fold_md = TXT("experiments/circuit_distance_exact/FOLD_PROOF.md")
    spec_md = TXT("lean/CertR198/SPEC.md")
    r205_md = TXT("docs/RESEARCH_R205_CDP5.md")
    ccheck_c = TXT("experiments/circuit_distance_exact/ccheck.c")
    RJ = "evidence/circuit_distance_exact/results.json"

    # -------------------------------------------- the literature, as the lane's documents state it
    q = RX(r"the literature had only UPPER bounds: <= (\d+) for IBM's depth-(\d+) cycle \(BP-OSD\s+search\), and "
           r"(\d+) conjectured by Strikis, Browne & Beverland \(arXiv (\d{4}\.\d{5})\)", results_md,
           "RESULTS.md, the question")
    put("LitIbmUpper", q.group(1), "RESULTS.md: the question")
    put("LitTcConj", q.group(3), "RESULTS.md: the question")
    put("ArxSBB", q.group(4), "RESULTS.md: the question")
    lit_depth = int(q.group(2))
    put("SBBFigCycle", RX(r"3-colour cycle of 2603\.05481 \(Fig\. (\d+)\)", results_md, "RESULTS.md, 3-colour row")
        .group(1), "RESULTS.md: results table")
    figs = RX(r"rebuilt from Figs (\d+)-(\d+)", results_md, "RESULTS.md, the 3-colour certificates")
    put("SBBFigLo", figs.group(1), "RESULTS.md: the 24 time-shift certificates")
    put("SBBFigHi", figs.group(2), "RESULTS.md: the 24 time-shift certificates")
    six = RX(r"its hook residuals reproduce all (\w+) patterns of Fig\. (\d+)", results_md, "RESULTS.md, Fig. 11")
    put("SBBResiduals", word(six.group(1), "RESULTS.md residual patterns"), "RESULTS.md: the circuits are published")
    put("SBBFigResid", six.group(2), "RESULTS.md: the circuits are the published ones")
    put("SBBFigGeom", RX(r"matched to Fig\. (\d+)'s long-edge directions", results_md, "RESULTS.md, geometry").group(1),
        "RESULTS.md: the circuits are the published ones")
    st = RX(r"## Shaw & Terhal (\d{4}) \(arXiv (\d{4}\.\d{5})\)", results_md, "RESULTS.md, the 2026 section")
    put("STbYear", st.group(1), "RESULTS.md: the 2026 section heading")
    put("ArxSTb", st.group(2), "RESULTS.md: the 2026 section heading")
    put("STbAppendix", RX(r'"we did not check the circuit-level distance" \(its App\. ([A-Z]\.\d+)\)', results_md,
                          "RESULTS.md, the 2026 quotation").group(1), "RESULTS.md: the 2026 section")
    tdef = RX(r"searches its homomorphism-based circuits \(Def\. (\d+)\), each fixed by one contraction tree "
              r"\(Sec\. (\d+\.\d+)\)", results_md, "RESULTS.md, Def. and contraction trees")
    put("STbDef", tdef.group(1), "RESULTS.md: the 2026 section")
    put("STbTreeSec", tdef.group(2), "RESULTS.md: the 2026 section (contraction trees)")
    put("STbProp", RX(r"arXiv 2604\.09797, Prop\. (\d+)", results_md, "RESULTS.md, rule F").group(1),
        "RESULTS.md: rule F")
    put("STbFig", RX(r"a degree-4 morphing point in the paper's Fig\. (\d+), read off its k = \d+ panel", results_md,
                     "RESULTS.md, Fig. 25").group(1), "RESULTS.md: the 2026 table")
    sta = RX(r"Morphing circuit of Shaw & Terhal \((\d{4}\.\d{5})\)", results_md, "RESULTS.md, morphing row")
    put("ArxSTa", sta.group(1), "RESULTS.md: results table")
    need(f"morphing circuit of {sta.group(1)}" in res["morph72"]["claim"], "the morphing source differs between "
         "RESULTS.md and results.json")
    need(f"{st.group(2)}'s Abelian-2BGA" in res["morph2026"]["claim"], "the 2026 source differs between RESULTS.md "
         "and results.json")
    put("ArxWJH", RX(r"Webster, Jacob and Higgott, \*Distance-Finding Algorithms for Quantum Codes and Circuits\* "
                     r"\(arXiv:(\d{4}\.\d{5})\)", r205_md, "R205 prior art").group(1),
        "docs/RESEARCH_R205_CDP5.md: prior art")
    reg = J(EV / "r206_schedule_discovery" / "registry.json")
    put("ArxAlpha", RX(r"AlphaSyndrome, ASPLOS 2026, arXiv:(\d{4}\.\d{5})", reg["prior_art"], "R206 prior art").group(1),
        "r206_schedule_discovery/registry.json: prior_art")

    # -------------------------------------------- codes
    m72 = RX(r"with the (\[\[\d+,\d+,\d+\]\]) end-cycle code \(mid-cycle (\[\[\d+,\d+,\d+\]\]), homomorphism x\)",
             res["morph72"]["claim"], "morph72 claim")
    m144 = RX(r"with the (\[\[\d+,\d+,\d+\]\]) end-cycle code \(mid-cycle (\[\[\d+,\d+,\d+\]\])\)",
              res["morph144"]["claim"], "morph144 claim")
    gross = code_params(m144.group(1), "morph144 end-cycle code")
    small = code_params(m72.group(1), "morph72 end-cycle code")
    big_mid = code_params(m144.group(2), "morph144 mid-cycle code")
    need(code_params(m72.group(2), "morph72 mid-cycle code") == gross, "morph72's mid-cycle code is not the gross code")
    need(RX(r"(\[\[\d+,\d+,\d+\]\]) IBM-style cycle", res["bb72"]["claim"], "bb72 claim").group(1) == m72.group(1),
         "the IBM-style small code differs from the morphing end-cycle code")
    need(int(RX(r"end-cycle code's distance \((\d+), the authors' integer program\)", res["morph144"]["claim"],
                "morph144 code distance").group(1)) == gross[2],
         "the gross code's distance in the morph144 claim is not its third parameter")
    put("GrossCode", code_tex(*gross), RJ + ": morph144 claim")
    put("GrossD", gross[2], RJ + ": morph144 claim")
    put("SmallCode", code_tex(*small), RJ + ": morph72 / bb72 claims")
    # Exact values published for SMALLER BB circuits (the referee's finding of 2026-10-29). The bib entries are tied to
    # the arXiv ids RESULTS.md quotes, so a sentence and its citation cannot drift apart.
    pr = RX(r"Aghababaie Beni, Higgott & Shutty \(arXiv (\d{4}\.\d{5}), App\. ([A-Z])\) verified with stim's\s+"
            r"shortest_error_sat_problem and a MaxSAT Evaluation (\d{4}) solver that the circuit distance of the "
            r"(\[\[\d+,\d+,\d+\]\]) code is (\d+)\s+and of the (\[\[\d+,\d+,\d+\]\]) code is (\d+); Webster, Jacob & "
            r"Higgott \(arXiv (\d{4}\.\d{5}), Sec\. ([\d.]+)\) report that Gurobi\s+completed within (\d+) hours only "
            r"for the first member of their BB circuit family", results_md, "RESULTS.md, exact values for smaller circuits")
    bib = TXT("paper/refs.bib")
    for key, arx in (("beni2025tesseract", pr.group(1)), ("webster2026distance", pr.group(8))):
        # a pinned version (2603.22532v2, the latest one the literature log re-read) is the same work
        need(re.search(r"@\w+\{" + key + r",[^@]*eprint = \{" + re.escape(arx) + r"(v\d+)?\}", bib) is not None,
             f"refs.bib's {key} is not arXiv {arx}, the source RESULTS.md quotes")
    need(code_params(pr.group(4), "Tesseract's first code") == small, "Tesseract's exact value is not for the small code")
    put("PriorSmallD", pr.group(5), "RESULTS.md: exact values for smaller BB circuits (arXiv " + pr.group(1) + ")")
    put("PriorMidCode", code_tex(*code_params(pr.group(6), "Tesseract's second code")), "RESULTS.md: ditto")
    put("PriorMidD", pr.group(7), "RESULTS.md: ditto")
    put("PriorMaxSatYear", pr.group(3), "RESULTS.md: ditto")
    put("PriorTessApp", pr.group(2), "RESULTS.md: ditto")
    put("PriorMipHours", pr.group(10), "RESULTS.md: ditto (arXiv " + pr.group(8) + ")")
    put("PriorWebsterSec", pr.group(9), "RESULTS.md: ditto")

    # SD6 (R229, 2026-11-02): the lane's fault placement gives the SAME mechanism sets as SD6 on the compared circuits,
    # each comparison with a control that must change the set. The paper says "every circuit and number of rounds
    # compared", so the record must cover all four circuits, both bases, T = 1..Tmax, and be equal everywhere.
    sd = JL("sd6_equivalence.jsonl")
    tmax = max(r["rounds"] for r in sd)
    need({(r["circuit"], r["basis"], r["rounds"]) for r in sd}
         == {(c, b, t) for c in ("ibm", "3c", "morph72", "morph144") for b in "ZX" for t in range(1, tmax + 1)}
         and len(sd) == 8 * tmax, "the SD6 record does not cover the four circuits, both bases, T = 1..Tmax")
    need(all(r["full_equal"] and r["proj_equal"] and r["control_differs"] for r in sd),
         "an SD6 comparison differs, or its control did not change the set")
    put("SdSixRows", len(sd), "records/sd6_equivalence.jsonl")
    put("SdSixTmax", tmax, "records/sd6_equivalence.jsonl")
    # The supplement's IBM corollary must rest on the shipped re-certification, not on the campaign certificates the
    # paper says it does not use (the second referee's finding 2, 2026-11-02).
    wp = TXT("experiments/circuit_distance_exact/WINDOW_PROOF.md")
    ibm_cor = RX(r"- \*\*IBM cycle, memory-Z and memory-X: d_circ = 10 for every T >= 2\.\*\* Lower bound: (.*?)\n- \*\*3-colour",
                 wp, "WINDOW_PROOF.md, Corollary 3 (IBM)")
    need("ibm_recert_20261026" in ibm_cor.group(1) and "Lemma W(iii)" in ibm_cor.group(1),
         "WINDOW_PROOF.md's IBM corollary does not rest on the shipped re-certification through Lemma W(iii)")
    # ... and so must every other corollary the paper's every-T claims use (second referee of 2026-11-02, M3: a fix to
    # the main text that never reaches a supplement recurs; only a gate stops it)
    tc_cor = RX(r"- \*\*3-colour cycle, memory-Z and memory-X: d_circ = 11 for every T >= 2\.\*\* Lower bound: (.*?)"
                r"Upper bound", wp, "WINDOW_PROOF.md, Corollary 3 (3-colour)")
    need("recert_20261102" in tc_cor.group(1) and "Lemma W(iii)" in tc_cor.group(1),
         "WINDOW_PROOF.md's 3-colour corollary does not rest on the re-certification through Lemma W(iii)")
    need("recert_20261102" in RX(r"- Corollary 3: IBM.*?(3-colour.*?)\n- ", wp, "WINDOW_PROOF.md summary").group(1),
         "WINDOW_PROOF.md's summary of the 3-colour corollary does not cite the re-certification")
    fp = TXT("experiments/circuit_distance_exact/FOLD_PROOF.md")
    for circ in ("morph144", "morph72"):
        cor = RX(rf"- \*\*{circ}, both bases\.\*\*(.*?)(?=\n- \*\*|\n\n)", fp, f"FOLD_PROOF.md, Corollary 3 ({circ})")
        need("recert_20261102" in cor.group(1), f"FOLD_PROOF.md's {circ} corollary does not cite the re-certification")
    need("irreducible" in RX(r"- \*\*morph72, both bases\.\*\*(.*?)(?=\n\n)", fp, "FOLD_PROOF.md morph72").group(1),
         "FOLD_PROOF.md's morph72 corollary must say the certificate excludes IRREDUCIBLE solutions")
    put("BigMidCode", code_tex(*big_mid), RJ + ": morph144 claim")

    # -------------------------------------------- the committed projected model (IBM, T = 2, memory-Z)
    man = J(EV / "r205_cdp5" / "manifest.json")
    obs = {o["model_file"]: o for o in man["observations"]}
    need(set(obs) == {"ibm_l12_Z2_model.txt", "threecolour_l12_X2_model.txt"}, "the CDP5 manifest changed")
    for f, o in obs.items():
        need(sha256(EV / "r205_cdp5" / f) == o["model_sha256"], f"{f} does not hash to the manifest")
        need(sha256(EV / "r205_cdp5" / o["proof_file"]) == o["proof_sha256"], f"{o['proof_file']} does not hash to "
                                                                              "the manifest")
        need(o["checker_exit_code"] == 0, f"{o['proof_file']} was not VERIFIED")
    ibm_proofs = JL("ibm_proofs_w9.jsonl")
    ibm_cell = {(r["basis"], r["T"]): r for r in ibm_proofs}
    need(len(ibm_cell) == len(ibm_proofs), "the IBM certificate record repeats a cell")
    oz, otc = obs["ibm_l12_Z2_model.txt"], obs["threecolour_l12_X2_model.txt"]
    need(oz["model_sha256"] == ibm_cell[("Z", 2)]["dem_sha256"],
         "the shipped IBM model is not the model of the certified (Z, T = 2) cell")
    tok = (EV / "r205_cdp5" / "ibm_l12_Z2_model.txt").read_text(encoding="utf-8").split()
    nd, nm, no, norb, gl, gm = (int(x) for x in tok[:6])
    i, mdeg, inc = 6, [], [0] * nd
    for _ in range(nm):
        k = int(tok[i])
        ds = [int(x) for x in tok[i + 1:i + 1 + k]]
        i += 3 + k
        need(len(set(ds)) == k, "a mechanism of the shipped model repeats a detector")
        mdeg.append(k)
        for d in ds:
            inc[d] += 1
    need(no == gross[1], "the shipped model's observable count is not k")
    sweep_ibm = {(r["basis"], r["T"]): r for r in JL("ibm_sweep_w9.jsonl")}
    hz = RX(r"ND (\d+) NM (\d+) orbits (\d+)", sweep_ibm[("Z", 2)]["head"], "IBM sweep head")
    need((int(hz.group(1)), int(hz.group(2)), int(hz.group(3))) == (nd, nm, norb),
         "the IBM sweep and the shipped model disagree on the T = 2 model's size")
    put("IbmProjDet", I(nd), "r205_cdp5/ibm_l12_Z2_model.txt header (= ibm_sweep_w9 T = 2)")
    put("IbmProjMech", I(nm), "r205_cdp5/ibm_l12_Z2_model.txt header (= ibm_sweep_w9 T = 2)")
    put("GrossL", gl, "r205_cdp5/ibm_l12_Z2_model.txt header")
    put("GrossM", gm, "r205_cdp5/ibm_l12_Z2_model.txt header")
    put("ProjDetDegLo", min(inc), "computed from r205_cdp5/ibm_l12_Z2_model.txt")
    put("ProjDetDegHi", max(inc), "computed from r205_cdp5/ibm_l12_Z2_model.txt")
    put("ProjMechDegHi", max(mdeg), "computed from r205_cdp5/ibm_l12_Z2_model.txt")

    dem = JL("dem_independent.jsonl")
    dem_cell = {(r["circuit"], r["l"], r["basis"], r["rounds"]): r for r in dem}
    full_z, full_x = dem_cell[("ibm", gl, "Z", 2)]["mechanisms"], dem_cell[("ibm", gl, "X", 2)]["mechanisms"]
    put("IbmFullMechZ", I(full_z), "records/dem_independent.jsonl (ibm, Z, T = 2)")
    put("IbmFullMechX", I(full_x), "records/dem_independent.jsonl (ibm, X, T = 2)")

    # generic solvers, as RESULTS.md records them (prose only; no raw record exists)
    gs = RX(r"Each detector is a parity over ~(\d+) mechanisms \(gross code, (\d+) rounds: ([\d,]+) incidences, "
            r"([\d,]+) mechanisms\)", results_md, "RESULTS.md, why generic solvers stall")
    need(int(gs.group(4).replace(",", "")) == full_z and int(gs.group(2)) == 2,
         "RESULTS.md's generic-solver model is not the IBM T = 2 memory-Z model")
    full_inc = int(gs.group(3).replace(",", ""))
    wu = {(r["circuit"], r["basis"]): r for r in JL("window_uniform_run.jsonl")}
    need(set(wu) == {(c, b) for c in ("ibm", "3c") for b in "ZX"}, "the window-theorem runs changed")
    full_det = one((t["n_detectors"] for t in wu[("ibm", "Z")]["TPL"] if t["T"] == 2), "full IBM T = 2 detectors")
    need(round(full_inc / full_det) == int(gs.group(1)), "RESULTS.md's ~mechanisms per detector is not "
                                                          "incidences / detectors of the full model")
    # The incidence count itself is recorded only in RESULTS.md's prose, not in any evidence artifact, so it is
    # used here only to check the per-detector figure against the recorded detector count, never typeset.
    put("FullDetDeg", gs.group(1), "RESULTS.md: why generic solvers stall (checked: incidences / detectors)")
    put("GenericRounds", gs.group(2), "RESULTS.md: why generic solvers stall")
    # The Glucose / RC2 runs are WITHDRAWN (no raw record, short budgets, and the RC2 instance was solved exactly in
    # arXiv 2503.10988): the paper must not typeset them, and RESULTS.md must keep saying so.
    need("WITHDRAWN from paper 3" in results_md, "RESULTS.md no longer marks the generic-solver runs as withdrawn")

    # -------------------------------------------- IBM depth-8 cycle
    ib = res["ibm_gross"]
    ibm_w = one((r["W"] for r in ibm_proofs), "IBM certified weight")
    need(all(r["checker_exit"] == 0 and r["checker"].startswith(
        f"VERIFIED: no undetectable logical error of weight <= {ibm_w} in this DEM") for r in ibm_proofs),
        "an IBM certificate is not VERIFIED")
    need(ib["proofs"] == [{k: r[k] for k in ("basis", "T", "dem_sha256", "proof_sha256", "proof_bytes", "checker_s")}
                          for r in ibm_proofs], "results.json's IBM proofs are not the records")
    ibm_d = one((w["weight"] for w in ib["witnesses"]), "IBM witness weight")
    need(ibm_d == ibm_w + 1 and len(ib["witnesses"]) == 2 and {w["basis"] for w in ib["witnesses"]} == {"Z", "X"}
         and all(w["verified"] and w["minus_one_fails"] for w in ib["witnesses"]),
         "the IBM witnesses are not two verified minimal sets at the certified weight + 1")
    cl = RX(r"^IBM depth-(\d+) gross cycle: d_circ = (\d+) for every T >= (\d+), memory-Z and memory-X$", ib["claim"],
            "IBM claim")
    need(int(cl.group(1)) == lit_depth and int(cl.group(2)) == ibm_d, "the IBM claim states another value")
    ibm_tmin = int(cl.group(3))
    need(ib["evidence_class"].startswith(f"PROVEN for every T >= {ibm_tmin} from MACHINE-CHECKED premises"),
         "the IBM every-T claim is not PROVEN from machine-checked premises")
    ibm_t = {b: trange((t for (bb, t) in ibm_cell if bb == b), f"IBM {b} certificates") for b in "ZX"}
    ibm_t_lo, ibm_t_hi = one(ibm_t.values(), "IBM certified rounds per basis")
    iw = [wu[("ibm", b)]["WITNESS"][0] for b in "ZX"]
    need(all(w["mode"] == "end" and w["holds"] for w in iw) and one((w["every_T_from"] for w in iw), "IBM witness from")
         == ibm_tmin, "the IBM witness's every-T placement changed")
    ibm_dir = one((trange((int(t) for t in w["direct"]), "IBM witness direct") for w in iw), "IBM witness direct")
    put("IbmDepth", lit_depth, "RESULTS.md / results.json: IBM claim")
    put("IbmD", ibm_d, RJ + ": ibm_gross witnesses (= certified weight + 1)")
    put("IbmW", ibm_w, "records/ibm_proofs_w9.jsonl")
    put("IbmTmin", ibm_tmin, RJ + ": ibm_gross claim")
    put("IbmCerts", len(ibm_proofs), "records/ibm_proofs_w9.jsonl")
    put("IbmCertTlo", ibm_t_lo, "records/ibm_proofs_w9.jsonl")
    put("IbmCertThi", ibm_t_hi, "records/ibm_proofs_w9.jsonl")
    put("IbmLoss", gross[2] - ibm_d, "code distance minus d_circ")
    put("IbmWitDirectLo", ibm_dir[0], "records/window_uniform_run.jsonl (ibm WITNESS direct)")
    put("IbmWitDirectHi", ibm_dir[1], "records/window_uniform_run.jsonl (ibm WITNESS direct)")
    lo, hi = enclose(min(r["proof_bytes"] for r in ibm_proofs) / 1e6, max(r["proof_bytes"] for r in ibm_proofs) / 1e6, 1)
    put("IbmProofMBLo", lo, "records/ibm_proofs_w9.jsonl")
    put("IbmProofMBHi", hi, "records/ibm_proofs_w9.jsonl")
    lo, hi = enclose(min(r["checker_s"] for r in ibm_proofs), max(r["checker_s"] for r in ibm_proofs), 1)
    put("IbmCheckSLo", lo, "records/ibm_proofs_w9.jsonl")
    put("IbmCheckSHi", hi, "records/ibm_proofs_w9.jsonl")
    bpe = {round(r["proof_bytes"] / int(RX(r"\((\d+) trace entries", r["prover"], "IBM prover").group(1)))
           for r in ibm_proofs}
    put("IbmBytesPerEntry", one(bpe, "IBM proof bytes per trace entry"), "records/ibm_proofs_w9.jsonl prover lines")
    gz = RX(r"symmetry group of (\d+) translations, (\d+) orbits", ibm_cell[("Z", 2)]["checker"], "IBM group")
    need(int(gz.group(1)) == gl * gm and int(gz.group(2)) == norb, "the IBM group is not all translations")
    put("IbmGroup", gz.group(1), "records/ibm_proofs_w9.jsonl checker line")
    put("IbmFir", one((wu[("ibm", b)][h]["mem"]["minimal"] for b in "ZX" for h in ("FIR", "END")), "IBM FIR/END"),
        "records/window_uniform_run.jsonl (ibm FIR / END minimal)")
    # The PRIMARY IBM evidence since the referee's finding of 2026-10-29: the rule-S re-certification by the committed
    # prover and the Lean-modelled checker, every proof and model shipped. Each shipped byte is re-hashed here.
    RD = EV / "ibm_recert_20261026"
    rc = J(RD / "manifest.json")
    obs = rc["observations"]
    need(rc["verdict"] == f"{len(obs)}/{len(obs)} VERIFIED" and len(obs) == len(ibm_proofs),
         "the IBM re-certification is not complete, or covers a different count than the campaign")
    need({(o["basis"], o["rounds"]) for o in obs} == {(b, t) for b in "ZX" for t in range(ibm_t_lo, ibm_t_hi + 1)},
         "the IBM re-certification does not cover every (basis, T) the campaign did")
    for o in obs:
        need(o["w"] == ibm_w and o["proof_magic"] == man["proof_format"]
             and o["checker_verdict"].startswith(f"VERIFIED: no irreducible undetectable logical error of weight <= "
                                                 f"{ibm_w} whose earliest round is <= 1"),
             f"IBM re-certification {o['basis']}{o['rounds']}: not a rule-S verdict at the certified weight")
        for f, h in ((o["model"], o["model_sha256"]), (o["proof"], o["proof_sha256"])):
            need((RD / f).is_file() and sha256(RD / f) == h, f"IBM re-certification: {f} is not the recorded bytes")
        need((RD / o["proof"]).stat().st_size == o["proof_bytes"], f"IBM re-certification: {o['proof']} size")
    mv = RX(r"\*\*Modelled version\.\*\* `experiments/circuit_distance_exact/ccheck\.c` at `[0-9a-f]+` \(\d+ lines,\s+"
            r"sha256 `([0-9a-f]{12})\.\.\.([0-9a-f]{4})`\)", spec_md, "SPEC.md, the modelled checker")
    # THE CHECKER ON DISK, in both line-ending forms (2026-11-02: SPEC.md had pinned a Windows checkout's CRLF bytes,
    # which no clean clone holds). SPEC pins the bytes git stores (LF); the IBM manifest recorded the CRLF form of the
    # same text. Both are tied to the file itself here, so the check holds in any checkout and fails on any edit.
    ccheck_lf = hashlib.sha256(ccheck_c.encode("utf-8")).hexdigest()
    ccheck_crlf = hashlib.sha256(ccheck_c.replace("\n", "\r\n").encode("utf-8")).hexdigest()
    need(ccheck_lf.startswith(mv.group(1)) and ccheck_lf.endswith(mv.group(2)),
         "ccheck.c (as git stores it) is not the version SPEC.md modelled")
    need(rc["checker"]["sha256"] == ccheck_crlf
         and rc["checker"]["source"] == "experiments/circuit_distance_exact/ccheck.c",
         "the IBM re-certification's checker is not the committed ccheck.c")
    prove_c = TXT("experiments/circuit_distance_exact/cdist_prove.c")
    need(rc["prover"]["sha256"] == hashlib.sha256(prove_c.replace("\n", "\r\n").encode("utf-8")).hexdigest()
         and rc["prover"]["source"] == "experiments/circuit_distance_exact/cdist_prove.c",
         "the IBM re-certification's prover is not the committed cdist_prove.c")
    # ... and each shipped model must BE the cycle's export (ibm_recert_models.py re-exports every cell).
    rex = JL("ibm_recert_reexport.jsonl")
    need(sorted((r["basis"], r["rounds"], r["shipped_sha256"]) for r in rex)
         == sorted((o["basis"], o["rounds"], o["model_sha256"]) for o in obs)
         and all(r["equal"] and r["reexport_sha256"] == r["shipped_sha256"] for r in rex),
         "a shipped IBM re-certification model is not byte-identical to a fresh export of the cycle")
    put("IbmRecertCerts", len(obs), "ibm_recert_20261026/manifest.json")
    put("IbmRecertSame", sum(bool(o["same_model_bytes_as_campaign"]) for o in obs), "ibm_recert_20261026/manifest.json")
    lo, hi = enclose(min(o["proof_bytes"] for o in obs) / 1e3, max(o["proof_bytes"] for o in obs) / 1e3, 1)
    put("IbmRecertKBLo", lo, "ibm_recert_20261026/manifest.json")
    put("IbmRecertKBHi", hi, "ibm_recert_20261026/manifest.json")
    lo, hi = enclose(min(o["checker_s"] for o in obs), max(o["checker_s"] for o in obs), 1)
    put("IbmRecertCheckSLo", lo, "ibm_recert_20261026/manifest.json")
    put("IbmRecertCheckSHi", hi, "ibm_recert_20261026/manifest.json")

    # -------------------------------------------- every OTHER certificate, re-proved and re-checked (2026-11-02)
    # recert_campaign.py ran every recorded certificate cell through the committed prover and checker;
    # recert_package.py packaged it. Re-derived here from the records themselves, not from the campaign's code.
    RR = EV / "recert_20261102"
    rm = J(RR / "manifest.json")
    src = "recert_20261102/manifest.json"
    prove_lf = hashlib.sha256(prove_c.encode("utf-8")).hexdigest()
    need(rm["checker"]["sha256_git"] == ccheck_lf and rm["prover"]["sha256_git"] == prove_lf,
         "the re-certification campaign did not run the committed prover and checker")
    want = collections.Counter()
    for fam, name, kinds in (("morph72", "morph72_shift_certificates_w5.jsonl", None),
                             ("morph144", "morph144_certificates_w11.jsonl", None),
                             ("3c", "3c_shift_certificates_w10.jsonl", None),
                             ("3c_t2", "3c_T2_certificates_w10.jsonl", None),
                             ("morph2026", "morph2026_campaign.jsonl", ("cert", "shift_cert"))):
        for r in JL(name):
            if kinds is None or r.get("kind") in kinds:
                kind = r.get("kind") or ("cert" if fam in ("morph144", "3c_t2") else "shift_cert")
                want[(fam, kind, r["basis"], r["rounds"], r["w"], r.get("maxround", -1), r["model_sha256"])] += 1
    cells = rm["cells"]
    old_rows = JL("recert_old_prover.jsonl")
    need(collections.Counter((c["family"], c["kind"], c["basis"], c["rounds"], c["w"], c["maxround"],
                              c["model_sha256"]) for c in cells) == want,
         "the re-certification does not cover exactly the recorded certificate cells, once each")
    for c in cells:
        tag = f"re-certification {c['family']} {c['circuit']} {c['basis']}{c['rounds']}"
        need(c["verified"] and c["model_equal"] and c["proof_magic"] == "CDP5",
             f"{tag}: not verified, not the recorded model, or not a CDP5 proof")
        # the verdict must be the claim this cell is counted for: its own weight, and for an anchored certificate its
        # own round scope (second referee of 2026-11-02, minor 4)
        scope = (f"whose earliest round is <= {c['maxround']} in this DEM" if c["maxround"] >= 0 else "in this DEM")
        need(re.match(rf"VERIFIED: no (irreducible )?undetectable logical error of weight <= {c['w']} {scope}",
                      c["checker"]) is not None, f"{tag}: the verdict is not the claim this cell is counted for")
        need((RR / c["model"]).is_file() and sha256(RR / c["model"]) == c["model_sha256"],
             f"{tag}: the shipped model is not the recorded bytes")
        # a proof may differ from its record ONLY when an earlier prover, compiled from a named commit, reproduced the
        # recorded proof bit for bit -- checked against the reproduction records themselves, not the manifest's word
        if not c["proof_equal"]:
            rb = c["record_reproduced_by"]
            need(rb is not None and any(
                o["reproduces_record"] and o["commit"] == rb["commit"] and o["recorded_proof_sha256"]
                == c["recorded_proof_sha256"] and (o["family"], o["basis"], o["rounds"], o["w"])
                == (c["family"], c["basis"], c["rounds"], c["w"]) for o in old_rows),
                f"{tag}: differs from its record, and no earlier prover's reproduction of the record is recorded")
        if c["shipped"]:
            need((RR / c["proof"]).is_file() and sha256(RR / c["proof"]) == c["proof_sha256"],
                 f"{tag}: the shipped proof is not the recorded bytes")
    # DETERMINISM from the runs that could fail: proofs with trace bits, regenerated at another thread count
    det = rm["determinism"]
    det_rows = JL("recert_determinism.jsonl")
    det_ok = [d for d in det_rows if d["can_test_determinism"]]
    need(det_ok and all(d["identical"] and d["threads"] != rm["campaign_threads"] for d in det_rows)
         and det["cells_with_trace_bits"] == det["identical"] == len(det_ok),
         "the determinism runs do not show proofs with trace bits regenerated identically at another thread count")
    eq_bits = [c for c in cells if c["proof_equal"] and c["trace_bits"] > 0]
    put("RecertDetCells", len(det_ok), "records/recert_determinism.jsonl")
    put("RecertDetMaxMB", up(max(d["proof_bytes"] for d in det_ok) / 1e6, 0), "records/recert_determinism.jsonl")
    put("RecertDetThreads", " and ".join(str(t) for t in sorted({d["threads"] for d in det_ok})),
        "records/recert_determinism.jsonl")
    # every proof that does not ship has been produced twice, bit for bit: by the record's run (proof_equal) or by a
    # determinism run at another thread count -- what "regenerate" rests on, observed rather than argued
    # keyed on the whole cell, not (family, basis, rounds): two cells of one family that share a basis and a round
    # count (a cert and a shift certificate, or two morph2026 circuits) must not vouch for each other
    def cell_key(c):
        return (c["family"], c["kind"], c["circuit"], c["basis"], c["rounds"], c["w"], c["maxround"])

    twice = {cell_key(d) for d in det_ok if d["identical"]}
    need(all(c["proof_equal"] or cell_key(c) in twice for c in cells if not c["shipped"]),
         "a proof that does not ship has been produced only once; the text says each was produced twice")
    # THE TABLE OF THE RE-CERTIFICATION (external review 1, finding P3-11): one row per proof set, every count
    # recomputed here from the cells, the determinism rows and the reproduction rows; nothing typed
    old_ok = {(o["family"], o["kind"], o["circuit"], o["basis"], o["rounds"], o["w"], o["maxround"])
              for o in old_rows if o["reproduces_record"]}
    groups = [("morph2026", "2026 circuits (Table~\\ref{tab:morph})", lambda c: c["family"] == "morph2026"),
              ("morph72", r"morphing $[[72,12,6]]$, shift", lambda c: c["family"] == "morph72"),
              ("3c=", r"three-colour, shift (a)", lambda c: c["family"] == "3c" and c["proof_equal"]),
              ("3c!", r"three-colour, shift (b)", lambda c: c["family"] == "3c" and not c["proof_equal"]),
              ("3c_t2", r"three-colour, $T=2$", lambda c: c["family"] == "3c_t2"),
              ("morph144", r"morphing $[[144,12,12]]$", lambda c: c["family"] == "morph144")]
    rec_rows, tot = [], collections.Counter()
    covered = set()
    for _, label, sel in groups:
        cs = [c for c in cells if sel(c)]
        need(cs, f"re-certification table: the group {label!r} is empty")
        covered.update(id(c) for c in cs)
        n = collections.Counter(
            count=len(cs), shipped=sum(c["shipped"] for c in cs), equal=sum(bool(c["proof_equal"]) for c in cs),
            old=sum(cell_key(c) in old_ok for c in cs), det=sum(cell_key(c) in twice for c in cs),
            twice=sum(bool(c["proof_equal"]) or cell_key(c) in twice for c in cs))
        need(n["equal"] + n["old"] == n["count"], f"re-certification table, {label}: a proof neither equals its "
                                                  f"record nor has its record reproduced by the earlier prover")
        need(n["shipped"] in (0, n["count"]), f"re-certification table, {label}: a group ships only partly")
        need(n["shipped"] == n["count"] or n["twice"] == n["count"],
             f"re-certification table, {label}: an unshipped proof was produced only once")
        tot.update(n)
        rec_rows.append(rf"{label} & {n['count']} & {n['shipped']} & {n['count']} & {n['equal']} & {n['old']} & "
                        rf"{n['det']} \\")
    need(len(covered) == len(cells), "re-certification table: a cell belongs to no row")
    need(tot["count"] == len(cells) and tot["shipped"] == sum(c["shipped"] for c in cells)
         and tot["det"] == len(det_ok), "re-certification table: the totals do not reconcile with the manifest")
    unshipped_twice = sum(1 for c in cells if not c["shipped"] and (c["proof_equal"] or cell_key(c) in twice))
    need(unshipped_twice == len([c for c in cells if not c["shipped"]]),
         "re-certification table: not every unshipped proof was produced twice")
    # the sets behind each count of App. A, named (final referee of 2026-11-03, N4: two different 30s and two
    # different 252s); each sum must reconcile with the total it decomposes
    def n_(sel):
        return sum(1 for c in cells if sel(c))
    parts = dict(
        RecertEqTwentySix=n_(lambda c: c["proof_equal"] and c["family"] == "morph2026"),
        RecertEqTc=n_(lambda c: c["proof_equal"] and c["family"] == "3c"),
        RecertDiffSmall=n_(lambda c: not c["proof_equal"] and c["family"] == "morph72"),
        RecertDiffTc=n_(lambda c: not c["proof_equal"] and c["family"] == "3c"),
        RecertDiffTcTwo=n_(lambda c: not c["proof_equal"] and c["family"] == "3c_t2"),
        RecertDiffBig=n_(lambda c: not c["proof_equal"] and c["family"] == "morph144"),
        RecertRegenTc=n_(lambda c: not c["shipped"] and c["family"] == "3c"),
        RecertOverlap=n_(lambda c: not c["shipped"] and not c["proof_equal"]))
    need(parts["RecertEqTwentySix"] + parts["RecertEqTc"] == sum(bool(c["proof_equal"]) for c in cells)
         and n_(lambda c: c["proof_equal"] and c["family"] not in ("morph2026", "3c")) == 0,
         "the byte-identical proofs are not exactly the 2026 and three-colour ones the text names")
    need(sum(parts[k] for k in ("RecertDiffSmall", "RecertDiffTc", "RecertDiffTcTwo", "RecertDiffBig"))
         == sum(not c["proof_equal"] for c in cells), "the differing proofs are not the four sets the text names")
    need(n_(lambda c: c["shipped"]) == parts["RecertEqTwentySix"] + parts["RecertDiffSmall"]
         and all(c["shipped"] == (c["family"] in ("morph2026", "morph72")) for c in cells),
         "the shipped proofs are not exactly the 2026 and [[72,12,6]] ones the text names")
    need(n_(lambda c: not c["shipped"]) == parts["RecertRegenTc"] + parts["RecertDiffTcTwo"] + parts["RecertDiffBig"]
         and parts["RecertOverlap"] == parts["RecertDiffTc"] + parts["RecertDiffTcTwo"] + parts["RecertDiffBig"],
         "the unshipped proofs are not the three-colour, T = 2 and [[144,12,12]] sets the text names")
    for k, v in parts.items():
        put(k, v, src)
    put("RecertTwiceByRecord", sum(1 for c in cells if not c["shipped"] and c["proof_equal"]), src)
    put("RecertTwiceByRerun", sum(1 for c in cells if not c["shipped"] and not c["proof_equal"]
                                  and cell_key(c) in twice), src)
    rec_rows.append(rf"total & {tot['count']} & {tot['shipped']} & {tot['count']} & {tot['equal']} & {tot['old']} & "
                    rf"{tot['det']} \\")
    # IBM's cycle: its own campaign (rule S, every proof shipped) and the original rule-W certificates, which carry no
    # result and which the committed prover cannot regenerate (another trace format; checker not preserved)
    need(all(o.get("proof_magic") == "CDP5" and o["checker_verdict"].startswith("VERIFIED") for o in obs),
         "an IBM re-certification proof is not a verified CDP5 proof")
    rec_rows.append(r"\midrule")
    rec_rows.append(rf"IBM, rule S (own campaign) & {len(obs)} & {len(obs)} & {len(obs)} & -- & -- & -- \\")
    rec_rows.append(rf"IBM, rule W (original) & {len(ibm_proofs)} & 0 & -- & -- & -- & -- \\")
    tab_recert = "\n".join([
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"proof set & proofs & shipped & regen. & $=$ record & old prover & 2nd run \\", r"\midrule",
        *rec_rows, r"\bottomrule", r"\end{tabular}"]) + "\n"
    put("RecertCampaignThreads", rm["campaign_threads"], src)
    put("RecertProofEqualBits", len(eq_bits), src)
    put("RecertProofEqualMaxMB", up(max(c["proof_bytes"] for c in eq_bits) / 1e6, 0), src)
    put("RecertZeroBits", sum(c["trace_bits"] == 0 for c in cells), src)
    put("RecertMorphCells", sum(c["family"].startswith("morph") for c in cells), src)
    put("RecertMorphZeroBits", sum(c["family"].startswith("morph") and c["trace_bits"] == 0 for c in cells), src)
    tc_cells = [c for c in cells if c["family"] == "3c"]
    tc_rec = JL("3c_shift_certificates_w10.jsonl")
    need(sorted((c["basis"], c["rounds"], c["w"], c["model_sha256"]) for c in tc_cells)
         == sorted((r["basis"], r["rounds"], r["w"], r["model_sha256"]) for r in tc_rec),
         "the re-certified three-colour cells are not the time-shift certificates the paper counts")
    regen = [c for c in cells if not c["shipped"]]
    put("RecertCells", len(cells), src)
    put("RecertProofEqual", sum(bool(c["proof_equal"]) for c in cells), src)
    put("RecertProofFormat", sum(not c["proof_equal"] for c in cells), src)
    put("RecertShipped", sum(c["shipped"] for c in cells), src)
    put("RecertRegen", len(regen), src)
    # nearest, not ceiling: a ceiling overstated both (0.73 GB printed 0.8; 32.2 h printed 33; final referee)
    put("RecertRegenGB", f"{sum(c['proof_bytes'] for c in regen) / 1e9:.2f}", src)
    put("RecertTcHours", f"{sum(c['prover_s'] + c['checker_s'] for c in tc_cells) / 3600:.0f}", src)

    # THE LARGE PROOFS AS RELEASE ASSETS (2026-11-06, external audit 3, Paper C): v2026.11.05 attached them as two
    # archives while the text said they ship only as models and hashes. What the text says about them is read from the
    # record recert_assets.py writes from the archives, held here to the manifest cell by cell.
    asrc = "recert_20261102/release_assets.json"
    ar = J(RR / "release_assets.json")
    lf = lambda p: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()  # noqa: E731
    need(ar["manifest_sha256_lf"] == lf(RR / "manifest.json"), "the release-asset record was written against another manifest")
    want = {f"recert_20261102/proofs/{c['family']}/{c['kind']}/proof_{c['circuit']}_{c['basis']}{c['rounds']}_w{c['w']}.bin": c
            for c in regen}
    got = {m["path"]: (m["bytes"], m["sha256"]) for a in ar["archives"] for m in a["members"]}
    need(len(got) == sum(len(a["members"]) for a in ar["archives"]) and set(got) == set(want)
         and all(got[p] == (c["proof_bytes"], c["proof_sha256"]) for p, c in want.items()),
         "the release-asset archives are not exactly the manifest's unshipped proofs")
    arch = {a["name"]: a for a in ar["archives"]}
    need(sorted(arch) == ["oneq-recert-proofs-3c.tar.gz", "oneq-recert-proofs-morph144.tar.gz"]
         and arch["oneq-recert-proofs-3c.tar.gz"]["families"] == ["3c", "3c_t2"]
         and arch["oneq-recert-proofs-morph144.tar.gz"]["families"] == ["morph144"],
         "the release-asset archives are not the two the text names")
    named = set(re.findall(r"oneq-recert-proofs-[a-z0-9]+\.tar\.gz", TXT("paper/circuit-distance/main.tex")))
    need(named == set(arch), f"paper 3 names the archives {sorted(named)}, the record {sorted(arch)}")
    put("RecertAssetTcN", len(arch["oneq-recert-proofs-3c.tar.gz"]["members"]), asrc)
    put("RecertAssetMorphN", len(arch["oneq-recert-proofs-morph144.tar.gz"]["members"]), asrc)
    put("RecertAssetMB", f"{ar['totals']['archive_bytes'] / 1e6:.0f}", asrc)
    # checking cost, from the campaign's own checker runs at its thread count: the large proofs take minutes to
    # hours each, the shipped ones about a second at most
    big = [c["checker_s"] for c in regen]
    put("RecertBigCheckMinMin", down(min(big) / 60, 1), src)    # a printed range, widened outward
    put("RecertBigCheckMaxH", up(max(big) / 3600, 1), src)
    put("RecertBigCheckHours", f"{sum(big) / 3600:.1f}", src)
    put("RecertSmallCheckMaxS", up(max(c["checker_s"] for c in cells if c["shipped"]), 1), src)   # "at most"
    # and a reader's route actually run: proofs downloaded from the release, verified, extracted and checked again
    runs = [J(p) for p in sorted((RR / "asset_checks").glob("*.json"))]
    rows = [r for run in runs for r in run["rows"]]
    need(rows and all(r["verified"] for r in rows) and len({r["proof"] for r in rows}) == len(rows)
         and all(r["proof"] in want for r in rows), "no release-asset proof was checked, or one was not VERIFIED")
    need(all(run["checker_source_sha256_lf"] == lf(ROOT / "experiments" / "circuit_distance_exact" / "ccheck.c")
             and run["manifest_sha256_lf"] == ar["manifest_sha256_lf"] for run in runs),
         "a release-asset check ran another checker or against another manifest")
    put("RecertAssetChecked", len(rows), "recert_20261102/asset_checks/")
    put("RecertAssetUnchecked", len(regen) - len(rows), "recert_20261102/asset_checks/ against the manifest")
    need(all(c["verified"] for c in regen), "a release-asset proof was not VERIFIED in the re-certification")
    differ = [c for c in cells if not c["proof_equal"]]
    magics = sorted({c["record_reproduced_by"]["magic"] for c in differ})
    commits = sorted({c["record_reproduced_by"]["commit"] for c in differ})
    need(all(m != "CDP5" for m in magics), "a reproduced earlier proof is CDP5; the text says an earlier language")
    put("RecertOldMagic", ", ".join(m for m in magics), "records/recert_old_prover.jsonl")
    put("RecertOldCommits", ", ".join(rf"\texttt{{{c[:8]}}}" for c in commits), "records/recert_old_prover.jsonl")
    # Those commits are of the development repository, whose history is not public (the public repository's is
    # squashed; external audit 4, R1 / X3). What is public is the source that ran, shipped as old_provers/, and it must
    # be the bytes every reproduction row pins (prover_source_sha256_git), read as git stores them (LF).
    need(len(commits) == 1, "the text names one development commit; the reproductions name another number")
    olds = []
    for c in commits:
        pins = {o["prover_source_sha256_git"] for o in old_rows if o["commit"] == c}
        # its own name: this loop once reused `src`, the re-certification manifest's name that the memory range below
        # cites, and wrote this file's ABSOLUTE path into numbers.tex (found by the v2026.11.07.1 trial rehearsal)
        prover_src = LANE / "old_provers" / f"cdist_prove_{c}.c"
        need(len(pins) == 1 and prover_src.is_file()
             and hashlib.sha256(prover_src.read_bytes().replace(b"\r\n", b"\n")).hexdigest() in pins,
             f"the shipped prover of commit {c} is missing or is not the source its reproduction rows pin")
        olds.append(prover_src.relative_to(ROOT).as_posix())
    put("RecertOldSource", ", ".join(r"\texttt{" + p.replace("_", r"\_").replace("/", r"/\allowbreak ") + "}"
                                     for p in olds), "experiments/circuit_distance_exact/old_provers/ "
                                                     "(sha256 = records/recert_old_prover.jsonl)")
    # later re-certification of both T = 2 models in CDP5
    perf = J(EV / "r206_performance.json")
    pc = {c["name"]: c for c in perf["cases"]}
    need(pc["ibm_Z_weight9"]["model_sha256"] == ibm_cell[("Z", 2)]["dem_sha256"]
         and pc["ibm_X_weight9"]["model_sha256"] == ibm_cell[("X", 2)]["dem_sha256"]
         and pc["ibm_Z_weight9"]["weight_limit"] == ibm_w == pc["ibm_X_weight9"]["weight_limit"]
         and oz["fault_weight_bound"] == ibm_w,
         "the CDP5 re-certifications are not of the certified IBM T = 2 models at the certified weight")
    need(pc["ibm_Z_weight9"]["proof_sha256_identical_for_all_versions"] == oz["proof_sha256"],
         "the R206 IBM memory-Z proof is not the shipped one")
    put("CdpFiveName", man["proof_format"], "r205_cdp5/manifest.json")
    put("CdpIbmZKB", KB(oz["cdp5_proof_bytes"]), "r205_cdp5/manifest.json")
    put("CdpIbmBaseKB", KB(oz["baseline_proof_bytes"]), "r205_cdp5/manifest.json")
    put("CdpIbmRatio", D1(oz["baseline_proof_bytes"] / oz["cdp5_proof_bytes"]), "r205_cdp5/manifest.json")
    put("CdpTcRatio", D1(otc["baseline_proof_bytes"] / otc["cdp5_proof_bytes"]), "r205_cdp5/manifest.json")
    put("CdpTcW", otc["fault_weight_bound"], "r205_cdp5/manifest.json")
    put("StimVersion", man["environment"]["stim"], "r205_cdp5/manifest.json")
    put("GccVersion", RX(r"GCC (\d+\.\d+\.\d+)", man["environment"]["compiler"], "compiler").group(1),
        "r205_cdp5/manifest.json")

    # -------------------------------------------- the three-colour cycle
    tc, tce = res["three_colour"], res["three_colour_every_T"]
    t2 = {r["basis"]: r for r in JL("3c_T2_certificates_w10.jsonl")}
    tc_w = one((r["w"] for r in t2.values()), "3-colour T = 2 certified weight")
    need(set(t2) == {"Z", "X"} and tc["evidence_class"] == "MACHINE-CHECKED" and all(
        tc["bases"][b]["certified"] for b in "ZX"), "the 3-colour T = 2 certificates are not both VERIFIED")
    tc_d = one((w["weight"] for w in tc["witnesses"]), "3-colour witness weight")
    need(tc_d == tc_w + 1 and {w["basis"] for w in tc["witnesses"]} == {"Z", "X"} and all(
        w["verified"] and w["minus_one_fails"] for w in tc["witnesses"]), "3-colour witnesses changed")
    cl = RX(r"^3-colour cycle of (\d{4}\.\d{5}): d_circ = (\d+) for every T >= (\d+), memory-Z and memory-X$",
            tce["claim"], "3-colour every-T claim")
    need(cl.group(1) == q.group(4) and int(cl.group(2)) == tc_d, "the 3-colour claim states another value or source")
    tc_tmin = int(cl.group(3))
    need(tce["evidence_class"].startswith(f"PROVEN for every T >= {tc_tmin} from MACHINE-CHECKED premises"),
         "the 3-colour every-T claim is not PROVEN from machine-checked premises")
    need(int(q.group(3)) == tc_d, "the conjectured value is not the certified one; the text says they agree")
    put("TcD", tc_d, RJ + ": three_colour witnesses (= certified weight + 1)")
    put("TcW", tc_w, "records/3c_T2_certificates_w10.jsonl")
    put("TcTmin", tc_tmin, RJ + ": three_colour_every_T claim")
    put("TcLoss", gross[2] - tc_d, "code distance minus d_circ")
    for b in "ZX":
        v = tc["bases"][b]
        put(f"TcSearchNodes{b}", SCI(v["search_nodes"]), RJ + f": three_colour bases {b}")
        put(f"TcSearchS{b}", R0(v["search_seconds"]), RJ + f": three_colour bases {b}")
        c = v["certificate"]
        need(c["model_sha256"] == t2[b]["model_sha256"], f"3-colour {b}: the certificate is on another model")
        put(f"TcCertMB{b}", MB(c["proof_bytes"]), RJ + f": three_colour bases {b} certificate")
        put(f"TcCheckS{b}", R0(c["checker_s"]), RJ + f": three_colour bases {b} certificate")
    tc_group, tc_orb = one(((r["group"], r["orbits"]) for r in t2.values()), "3-colour T = 2 group and orbits")
    need(tc_group * tc_orb == nm, "the 3-colour T = 2 orbits do not partition the mechanisms freely")
    need(one(((r["mechanisms"], r["detectors"]) for r in t2.values()), "3-colour T = 2 size") == (nm, nd),
         "the 3-colour T = 2 projected models are not the size of the IBM one")
    put("TcGroup", tc_group, "records/3c_T2_certificates_w10.jsonl")
    tc_full = one((r["full_mechanisms"] for r in t2.values()), "3-colour full model size")
    need(dem_cell[("3c", gl, "Z", 2)]["mechanisms"] == dem_cell[("3c", gl, "X", 2)]["mechanisms"] == tc_full,
         "the 3-colour full model sizes disagree")
    put("TcFullMech", I(tc_full), "records/3c_T2_certificates_w10.jsonl")
    # the 24 time-shift certificates, re-read from the records: VERIFIED, at tc_w, on the very swept models
    sh, sw = JL("3c_shift_certificates_w10.jsonl"), JL("3c_sweep_shift_w10.jsonl")
    swm = {(r["basis"], r["rounds"]): r["model_sha256"] for r in sw}
    tc_s = {b: trange((r["rounds"] for r in sh if r["basis"] == b), f"3-colour {b} shift certificates") for b in "ZX"}
    tc_s_lo, tc_s_hi = one(tc_s.values(), "3-colour shift rounds per basis")
    need(len(sh) == 2 * (tc_s_hi - tc_s_lo + 1) and set(swm) == {(r["basis"], r["rounds"]) for r in sh},
         "the 3-colour time-shift certificates do not cover the swept cells exactly once")
    # Two verdict wordings exist: the checker says "irreducible" since the Lean pass's finding F2 (part 14); cells
    # checked before then print the same verified fact without the word (collect_results._shift_verified).
    new_v = f"VERIFIED: no irreducible undetectable logical error of weight <= {tc_w} whose earliest round is <= 1"
    old_v = f"VERIFIED: no undetectable logical error of weight <= {tc_w} whose earliest round is <= 1"
    need(all(r["checker_exit"] == 0 and r["w"] == tc_w and r["model_sha256"] == swm[(r["basis"], r["rounds"])]
             and r["group"] == tc_group and (r["checker"].startswith(new_v) or r["checker"].startswith(old_v))
             for r in sh), "a 3-colour time-shift certificate is not VERIFIED on its swept model")
    put("TcShiftCerts", len(sh), "records/3c_shift_certificates_w10.jsonl")
    put("TcShiftPreFix", sum(r["checker"].startswith(old_v) for r in sh),
        "records/3c_shift_certificates_w10.jsonl (verdicts without the word irreducible)")
    put("TcShiftTlo", tc_s_lo, "records/3c_shift_certificates_w10.jsonl")
    put("TcShiftThi", tc_s_hi, "records/3c_shift_certificates_w10.jsonl")
    put("TcShiftHours", hours(sum(r["prover_s"] + r["checker_s"] for r in sh)), "records/3c_shift_certificates_w10.jsonl")
    # the proofs the committed checker verified are the re-certification's CDP5 ones (second referee of 2026-11-02,
    # minor 1): their sizes, not those of the original records, some of which were CDP3
    lo, hi = enclose(min(c["proof_bytes"] for c in tc_cells) / 1e6, max(c["proof_bytes"] for c in tc_cells) / 1e6, 1)
    put("TcShiftMBLo", lo, src)
    put("TcShiftMBHi", hi, src)
    put("TcShiftMechLo", I(min(r["mechanisms"] for r in sh)), "records/3c_shift_certificates_w10.jsonl")
    put("TcShiftMechHi", I(max(r["mechanisms"] for r in sh)), "records/3c_shift_certificates_w10.jsonl")
    anc = [int(RX(r"(\d+) anchors", r["checker"], "3c anchors").group(1)) for r in sh]
    put("TcAnchorsLo", min(anc), "records/3c_shift_certificates_w10.jsonl checker lines")
    put("TcAnchorsHi", max(anc), "records/3c_shift_certificates_w10.jsonl checker lines")
    # the memory-X proofs were all written in one format; their trace length stops changing with T
    bits_x = {r["rounds"]: int(RX(r"\((\d+) trace bits", r["prover"], "3c prover").group(1)) for r in sh
              if r["basis"] == "X"}
    plateau = min(t for t in bits_x if len({bits_x[u] for u in bits_x if u >= t}) == 1)
    need(plateau < tc_s_hi, "the memory-X time-shift proofs never stop growing with T; the text says they do")
    put("TcPlateauFrom", plateau, "records/3c_shift_certificates_w10.jsonl (memory-X trace bits)")
    longest = max(sh, key=lambda r: r["prover_s"] + r["checker_s"])
    put("TcLongestH", hours(longest["prover_s"] + longest["checker_s"]), "records/3c_shift_certificates_w10.jsonl")
    put("TcLongestT", longest["rounds"], "records/3c_shift_certificates_w10.jsonl")
    put("TcLongestB", longest["basis"], "records/3c_shift_certificates_w10.jsonl")
    emb = [r for r in JL("3c_witness_embedding.jsonl") if r["full_verified"] and r["weight"] == tc_d]
    tc_e = {b: trange((r["T"] for r in emb if r["basis"] == b), f"3-colour {b} embeddings") for b in "ZX"}
    tc_e_lo, tc_e_hi = one(tc_e.values(), "3-colour embedding rounds per basis")
    need(tc_e_lo == tc_tmin, "the 3-colour witnesses are not verified from the first claimed round")
    put("TcEmbedThi", tc_e_hi, "records/3c_witness_embedding.jsonl")
    tw3 = [wu[("3c", b)]["WITNESS"][0] for b in "ZX"]
    need(all(w["mode"] == "start" and w["holds"] for w in tw3), "the 3-colour witness's every-T placement changed")
    put("TcWitEveryFrom", one((w["every_T_from"] for w in tw3), "3c witness from"),
        "records/window_uniform_run.jsonl (3c WITNESS)")
    tc_dir = one((trange((int(t) for t in w["direct"]), "3c witness direct") for w in tw3), "3c witness direct")
    need(tc_dir[0] == tc_tmin, "the 3-colour witness is not checked directly from the first claimed round")
    put("TcWitDirectHi", tc_dir[1], "records/window_uniform_run.jsonl (3c WITNESS direct)")
    put("TcFir", one((wu[("3c", b)][h]["mem"]["minimal"] for b in "ZX" for h in ("FIR", "END")), "3c FIR/END"),
        "records/window_uniform_run.jsonl (3c FIR / END minimal)")
    # The window formula's start index j_s and threshold T_0 = j_s + U, U = max(h, e) + 1, per model (external audit 4,
    # 2.3 and 2.16: the paper printed T_0 = 1 + U, the projected value, and called the full model's start "a later
    # block" for both circuits). The identities' h, e are those of each model's row set (memory-type rows projected,
    # all rows full). The witnesses' every-T placement must be the full model's T_0: an IBM end-type witness lies in
    # tau_T(End) for every T >= T_0, and a 3-colour start-type one, placed at T_0, lies wholly in the start part S
    # (a mechanism in an interior translate tau_i(I), i >= j_s, would need T >= j_s + 1 + U = T_0 + 1).
    T0 = {}
    for kind in ("ibm", "3c"):
        for model, rows in (("projected", "mem"), ("full", "all")):
            fs = [wu[(kind, b)]["FORMULA"][model] for b in "ZX"]
            for b, f in zip("ZX", fs):
                hh, ee = wu[(kind, b)]["FIR"][rows]["minimal"], wu[(kind, b)]["END"][rows]["minimal"]
                need(f["holds"] and (f["h"], f["e"]) == (hh, ee) and f["U"] == max(hh, ee) + 1
                     and f["T0"] == f["j_s"] + f["U"], f"{kind} {b} {model}: the formula's T_0 is not j_s + U")
            T0[(kind, model)] = one(((f["j_s"], f["T0"]) for f in fs), f"{kind} {model} (j_s, T_0) per basis")
        # the text gives one h = e per circuit "for both models": the two row sets' minimal h and e must agree
        one((wu[(kind, b)][hy][rows]["minimal"] for b in "ZX" for hy in ("FIR", "END") for rows in ("mem", "all")),
            f"{kind}: FIR / END minimal over both bases and both row sets")
    need(T0[("ibm", "projected")][0] == T0[("3c", "projected")][0] == T0[("ibm", "full")][0] == 1
         and T0[("3c", "full")][0] == 2, "the start indices changed: the text says 1, except 2 for the full "
                                         "three-colour model")
    need(T0[("ibm", "projected")][1] == T0[("ibm", "full")][1] == ibm_tmin == one((w["every_T_from"] for w in iw),
                                                                                  "IBM witness from"),
         "IBM's thresholds and its witness placement are not all the claimed first T")
    need(T0[("3c", "full")][1] == one((w["every_T_from"] for w in tw3), "3c witness from"),
         "the 3-colour witnesses are not placed at the full model's threshold (so not wholly in its start part)")
    put("IbmTzero", T0[("ibm", "full")][1], "records/window_uniform_run.jsonl (ibm FORMULA T0, both models)")
    put("TcTzeroProj", T0[("3c", "projected")][1], "records/window_uniform_run.jsonl (3c FORMULA projected T0)")
    put("TcTzeroFull", T0[("3c", "full")][1], "records/window_uniform_run.jsonl (3c FORMULA full T0)")
    put("TcJsFull", T0[("3c", "full")][0], "records/window_uniform_run.jsonl (3c FORMULA full j_s)")

    # -------------------------------------------- the window theorem's machine side
    need(all(not any(r["CONTROLS"].values()) for r in wu.values()), "a window-theorem control passed")
    put("WinControls", sum(len(r["CONTROLS"]) for r in wu.values()), "records/window_uniform_run.jsonl")
    stim_t = sorted({int(t) for r in wu.values() for t in r["FORMULA"]["projected"]["vs_stim"]})
    need(all(v["equal"] for r in wu.values() for v in r["FORMULA"]["projected"]["vs_stim"].values()),
         "the closed form disagrees with stim somewhere")
    need(len(stim_t) == 3, "the closed form was compared with stim at other than three T")
    for nm_, t in zip(("WinStimTa", "WinStimTb", "WinStimTc"), stim_t):
        put(nm_, t, "records/window_uniform_run.jsonl FORMULA vs_stim")
    need(all(all(t["agrees"] for t in r["TPL"]) for r in wu.values()), "a template comparison failed")
    tpl = {t["T"] for r in wu.values() for t in r["TPL"]}
    put("WinTplCount", len(tpl), "records/window_uniform_run.jsonl TPL")
    put("WinTplTmax", max(tpl), "records/window_uniform_run.jsonl TPL")
    rev = JL("window_review_formula.jsonl")
    cmp_ = [r for r in rev if r["step"] == "formula_vs_stim"]
    need(cmp_ and all(r["all_equal"] for r in cmp_), "the window audit found a difference")
    need(all(r["all_fail_as_required"] for r in rev if r["step"] == "controls_must_be_false"),
         "a window-audit control passed")
    put("WinAuditCmp", sum(len(r["per_T"]) for r in cmp_), "records/window_review_formula.jsonl")
    put("WinAuditTmax", max(int(t) for r in cmp_ for t in r["per_T"]), "records/window_review_formula.jsonl")

    # -------------------------------------------- morphing circuits (2024)
    mo72, mo144 = res["morph72"], res["morph144"]
    c72 = JL("morph72_shift_certificates_w5.jsonl")
    sw72 = {(r["basis"], r["rounds"]): r["model_sha256"] for r in JL("morph72_sweep_shift_w5.jsonl")}
    m72_w = one((r["w"] for r in c72), "morph72 certified weight")
    need(all(r["checker_exit"] == 0 and r["checker"].startswith("VERIFIED") and
             r["model_sha256"] == sw72[(r["basis"], r["rounds"])] for r in c72) and
         {(r["basis"], r["rounds"]) for r in c72} == set(sw72), "the morph72 time-shift certificates changed")
    m72_t = one((trange((r["rounds"] for r in c72 if r["basis"] == b), f"morph72 {b}") for b in "ZX"), "morph72 rounds")
    m72w = JL("morph72_witnesses.jsonl")
    m72_d = one((w["weight"] for w in m72w), "morph72 witness weight")
    need(m72_d == m72_w + 1 and all(w["verified"] and w["minus_one_fails"] and w["survives_metachecks"] for w in m72w)
         and {w["basis"] for w in m72w} == {"Z", "X"}, "the morph72 witnesses changed")
    cl = RX(r"d_circ = (\d+) for every T >= (\d+), memory-Z and memory-X -- the authors' BP-OSD bound \(<= (\d+)\) is "
            r"exact$", mo72["claim"], "morph72 claim")
    need(int(cl.group(1)) == m72_d == small[2], "the morph72 claim states another value, or the circuit loses "
                                                 "distance; the text says it loses none")
    morph_tmin = int(cl.group(2))
    need(int(cl.group(3)) == m72_d, "the authors' bound is not tight; the text says it is")
    need("PROVEN for every T >= 1 from MACHINE-CHECKED premises" in mo72["evidence_class"],
         "the morph72 every-T claim is not PROVEN from machine-checked premises")
    m72_e = [r for r in JL("morph72_witness_embedding.jsonl") if r["full_verified"] and r["weight"] == m72_d]
    m72_e_t = one((trange((r["T"] for r in m72_e if r["basis"] == b), f"morph72 {b} embeddings") for b in "ZX"),
                  "morph72 embeddings")
    put("MorphSmallD", m72_d, "records/morph72_witnesses.jsonl (= certified weight + 1)")
    put("MorphSmallW", m72_w, "records/morph72_shift_certificates_w5.jsonl")
    put("LitMorphSmall", cl.group(3), RJ + ": morph72 claim")
    put("MorphTmin", morph_tmin, RJ + ": morph72 claim")
    put("MorphSmallCerts", len(c72), "records/morph72_shift_certificates_w5.jsonl")
    put("MorphSmallTlo", m72_t[0], "records/morph72_shift_certificates_w5.jsonl")
    put("MorphSmallThi", m72_t[1], "records/morph72_shift_certificates_w5.jsonl")
    put("MorphSmallEmbLo", m72_e_t[0], "records/morph72_witness_embedding.jsonl")
    put("MorphSmallEmbHi", m72_e_t[1], "records/morph72_witness_embedding.jsonl")
    put("MorphSmallGroup", one((r["group"] for r in c72), "morph72 group"), "records/morph72_shift_certificates_w5.jsonl")
    folds = JL("fold_lemma.jsonl")
    need(all(r["holds"] for r in folds), "a per-T fold row does not hold")
    fold_t = one((trange((r["T"] for r in folds if (r["circuit"], r["basis"]) == cb), f"fold rows {cb}")
                  for cb in {(r["circuit"], r["basis"]) for r in folds}), "fold rows per circuit and basis")
    put("FoldRows", len(folds), "records/fold_lemma.jsonl")
    put("FoldRowsThi", fold_t[1], "records/fold_lemma.jsonl")
    c144 = {(r["basis"], r["rounds"]): r for r in JL("morph144_certificates_w11.jsonl")}
    m144_w = one((r["w"] for r in c144.values()), "morph144 certified weight")
    need(all(r["checker_exit"] == 0 and r["checker"].startswith(
        f"VERIFIED: no undetectable logical error of weight <= {m144_w} in this DEM") for r in c144.values()),
        "a morph144 certificate is not VERIFIED")
    m144_ct = one((tuple(sorted(t for (bb, t) in c144 if bb == b)) for b in "ZX"), "morph144 certified rounds")
    m144_d = one((mo144["bases"][b]["witness"]["weight"] for b in "ZX"), "morph144 witness weight")
    need(m144_d == m144_w + 1 and all(mo144["bases"][b]["witness"]["verified"] and
                                      mo144["bases"][b]["witness"]["minus_one_fails"] and
                                      mo144["bases"][b]["witness"]["survives_metachecks"] for b in "ZX"),
         "the morph144 witnesses changed")
    cl = RX(r"memory-X: d_circ = (\d+) at T = 2, = (\d+) for every T >= (\d+); memory-Z: d_circ = (\d+) at T = 2, = "
            r"(\d+) for every T >= (\d+)", mo144["claim"], "morph144 claim")
    need({int(cl.group(i)) for i in (1, 2, 4, 5)} == {m144_d} and {int(cl.group(i)) for i in (3, 6)} == {morph_tmin},
         "the morph144 claim states another value")
    need(m144_d == gross[2], "the morph144 circuit loses distance; the text says it loses none")
    lit144 = RX(r"\(their BP-OSD circuit bound: <= (\d+)\)", mo144["claim"], "morph144 literature")
    need(int(lit144.group(1)) == m144_d, "the authors' morph144 bound is not tight; the text says it is")
    need(mo144["evidence_class"].startswith(f"PROVEN for every T >= {morph_tmin} from MACHINE-CHECKED premises"),
         "the morph144 every-T claim is not PROVEN from machine-checked premises")
    m144_tmax = one((mo144["every_T"][b]["tmax"] for b in "ZX"), "morph144 per-T route")
    need(all(mo144["every_T"][b]["certified"] and mo144["every_T"][b]["d"] == m144_d for b in "ZX"),
         "the morph144 per-T route is not certified")
    need(fold_t[1] == m144_tmax, "the per-T fold rows and the morph144 route reach different T")
    for b in "ZX":
        need(c144[(b, 1)]["model_sha256"] == mo144["every_T"][b]["model_sha256_T1"], f"morph144 {b}: the T = 1 "
                                                                                       "certificate is on another model")
        w = mo144["bases"][b]["witness"]
        put(f"MorphWitS{b}", R0(float(RX(r"cdist w=\d+, ([\d.]+) s", w["found_in"], "morph144 witness").group(1))),
            RJ + f": morph144 bases {b} witness")
    emb144 = [r for r in JL("morph144_witness_embedding.jsonl") if r["full_verified"] and r["weight"] == m144_d]
    e144 = one((trange((r["T"] for r in emb144 if r["basis"] == b), f"morph144 {b} embeddings") for b in "ZX"),
               "morph144 embeddings")
    need(e144 == (1, m144_tmax), "the morph144 witnesses are not verified in every T of the per-T route")
    put("MorphBigD", m144_d, RJ + ": morph144 witnesses (= certified weight + 1)")
    put("MorphBigW", m144_w, "records/morph144_certificates_w11.jsonl")
    put("LitMorphBig", lit144.group(1), RJ + ": morph144 claim")
    put("MorphBigTmax", m144_tmax, RJ + ": morph144 every_T")
    put("MorphBigCertTa", m144_ct[0], "records/morph144_certificates_w11.jsonl")
    put("MorphBigCertTb", m144_ct[-1], "records/morph144_certificates_w11.jsonl")
    put("MorphBigGroup", one((r["group"] for r in c144.values()), "morph144 group"),
        "records/morph144_certificates_w11.jsonl")
    gap = one((w["metacheck_gap"] for w in m72w + JL("morph144_witnesses.jsonl")), "meta-check gap")
    put("MetaChecks", gap, "records/morph72_witnesses.jsonl, morph144_witnesses.jsonl (metacheck_gap)")
    sched = JL("morph_schedule_vs_authors.jsonl")
    need(all(r["gates_identical"] and r["measurements_identical"] and r["contraction_rounds_identical"] for r in sched)
         and {r["kind"] for r in sched} == {"morph72", "morph144"}, "the schedule comparison changed")
    authors = one((r["authors"] for r in sched), "authors' builder revision")
    repo, commit = RX(r"^(github\.com/[\w.-]+/[\w.-]+)@([0-9a-f]{40})$", authors, "authors' repo").groups()
    put("MorphRepo", repo.replace("/", r"/\allowbreak ").replace("_", r"\_"), "records/morph_schedule_vs_authors.jsonl")
    put("MorphCommit", commit[:8], "records/morph_schedule_vs_authors.jsonl")
    put("MorphHoms", len({r["hom"] for r in sched}), "records/morph_schedule_vs_authors.jsonl")

    # -------------------------------------------- the fold theorem's machine side
    fu = JL("fold_uniform_hom_x.jsonl")
    fctl = one((len(r["controls"]) for r in fu), "fold-theorem control count")
    need(not any(v for r in fu for v in r["controls"].values()), "a fold-theorem control passed")
    put("FoldControls", fctl, "records/fold_uniform_hom_x.jsonl")
    put("FoldRuns", len(fu), "records/fold_uniform_hom_x.jsonl")
    need(all(t["agrees"] for r in fu for t in r["template"] + r["fresh"]), "a fold template comparison failed")
    tt = {t["T"] for r in fu for t in r["template"] + r["fresh"]}
    put("FoldTplCount", len(tt), "records/fold_uniform_hom_x.jsonl")
    put("FoldTplTmax", max(tt), "records/fold_uniform_hom_x.jsonl")
    oth = JL("fold_uniform_hom_y.jsonl") + JL("fold_uniform_hom_xy.jsonl")
    need(not any(v for r in oth for v in r["controls"].values()) and all(
        all(h["holds"] for h in r["hypotheses"].values()) for r in oth), "the other homomorphisms' fold runs changed")
    put("FoldOtherControls", one((len(r["controls"]) for r in oth), "other-hom control count"),
        "records/fold_uniform_hom_y.jsonl, fold_uniform_hom_xy.jsonl")
    comp = RX(r"0 mismatches over ([\d,]+) components", fold_md, "FOLD_PROOF.md, components")
    put("FoldComponents", SCI(int(comp.group(1).replace(",", ""))), "FOLD_PROOF.md section 5")
    cor2 = JL("fold_review_cor2.jsonl")
    ctl2 = [r for r in cor2 if "control_flipped_window_equal" in r]
    need(cor2 and all(r["equal"] for r in cor2) and ctl2 and not any(
        r["control_flipped_window_equal"] or r["control_no_exp_noise_last_round_equal"] for r in ctl2),
        "the fold audit changed: a point disagrees with stim, or a control agrees")
    put("FoldAuditCtlPoints", len(ctl2), "records/fold_review_cor2.jsonl (rows carrying both controls)")
    put("FoldAuditPoints", len({(r["kind"], r["basis"], r["hom"], r["T"]) for r in cor2}), "records/fold_review_cor2.jsonl")
    small_rows = JL("fold_review_b_n18.jsonl") + JL("fold_review_b_n36.jsonl")
    need(all(len(set(r["d_proj"].values())) == 1 for r in small_rows), "a small audit code's distance varies with T")
    need(any(len(set(r["ctl_corr_reset_odd"].values())) > 1 for r in JL("fold_review_b_n18ctl.jsonl")),
         "the audit's sensitivity control never saw a T-dependence")
    put("FoldAuditCodes", len({r["instance"] for r in small_rows}), "records/fold_review_b_n18.jsonl, _n36.jsonl")
    twr = JL("fold_uniform_witness_twin_T32.jsonl")
    tw = [r for r in twr if "control" not in r]
    twc = [r for r in twr if "control" in r]
    need(tw and {r["basis"] for r in tw} == {"Z", "X"} and all(
        r["weight"] == m144_d and r["projected_undetectable_logical_at_T"] and
        r["full_dem_pure_lift_undetectable_logical_at_T"] for r in tw), "the twin witnesses changed")
    need(twc and not any(r["projected_undetectable_logical_at_T"] or r["full_dem_pure_lift_undetectable_logical_at_T"]
                         for r in twc), "the twin-witness control (one mechanism dropped) does not fail")
    put("FoldTwinT", one((r["T"] for r in tw), "twin T"), "records/fold_uniform_witness_twin_T32.jsonl")
    # The ten 2026 standard schedules (Table II) are built by a second builder, morph2026.morphing_cycle_from, which
    # FOLD_PROOF.md Appendix C reads; fold_uniform.py checks the theorem's hypotheses on each one's own circuit and
    # identifies that circuit with the template (external audit 4, finding 2.15: the record had T = 1-4, 9, 16 only).
    # Every standard-schedule target, both bases, one contiguous run of T from 1, every comparison agreeing, every
    # hypothesis holding, every control failing; and the count FOLD_PROOF.md section 5 quotes is the record's.
    tg26 = J(LANE / "morph2026_targets.json")
    std26 = sorted(k for k, t in tg26.items() if t["kind"] == "std")
    f26 = JL("morph2026_fold_uniform.jsonl")
    need(sorted((r["circuit"], r["basis"]) for r in f26) == sorted((c, b) for c in std26 for b in "ZX")
         and all(r["hom"] == "x" for r in f26), "the 2026 fold runs do not cover every standard target and basis once")
    t26 = one((tuple(sorted(t["T"] for t in r["template"] + r["fresh"])) for r in f26), "2026 template T per run")
    need(t26 == tuple(range(1, len(t26) + 1)), "the 2026 template T are not every T from 1 up")
    need(all(t["agrees"] and t["symptom_formula_mismatches"] == 0 for r in f26 for t in r["template"] + r["fresh"]),
         "a 2026 template comparison failed")
    need(all(all(h["holds"] for h in r["hypotheses"].values()) and r["hypotheses"]["PAL"]["root_layer_invariance"]
             and all(v["holds"] for v in r["derived"].values()) and r["info"]["holds"] for r in f26),
         "a 2026 fold hypothesis fails")
    c26 = one((len(r["controls"]) for r in f26), "2026 fold control count")
    need(c26 >= 18 and not any(v for r in f26 for v in r["controls"].values()), "a 2026 fold control passed")
    comp26 = sum(t["components"] for r in f26 for t in r["template"] + r["fresh"])
    need(int(RX(r"no mismatch in ([\d,]+) components", fold_md, "FOLD_PROOF.md, 2026 components").group(1)
             .replace(",", "")) == comp26, "FOLD_PROOF.md's 2026 component count is not the record's")
    lay26 = {r["code"]: r["layers"] for r in JL("morph2026_campaign.jsonl") if r["kind"] == "info"}
    need(set(std26) <= set(lay26), "a standard 2026 target has no campaign info row")
    put("MorphLayers", one((lay26[c] for c in std26), "layers of the standard 2026 schedules"),
        "records/morph2026_campaign.jsonl (info rows: layers; fold_uniform.py requires this L in H1)")
    put("MtwoTplTmax", t26[-1], "records/morph2026_fold_uniform.jsonl")
    put("MtwoTplComponents", SCI(comp26), "records/morph2026_fold_uniform.jsonl")
    put("MtwoFoldControls", c26, "records/morph2026_fold_uniform.jsonl")

    # -------------------------------------------- trusted base
    tb = res["trusted_base"]
    need(all(r["equal"] and r["only_independent"] == 0 and r["only_stim"] == 0 for r in dem) and len(dem) == tb["models"],
         "the independent DEM derivation is not equal on every model")
    put("DemModels", tb["models"], RJ + ": trusted_base")
    put("DemMechLo", I(tb["mechanisms_min"]), RJ + ": trusted_base")
    put("DemMechHi", I(tb["mechanisms_max"]), RJ + ": trusted_base")
    # The 2026 circuits' models (Table II and the kernel-checked ones), derived independently since 2026-10-29. The
    # record must cover EXACTLY the models the campaign ran, by the driver's own enumeration of the campaign record.
    cr_, _lr = lane_modules()
    import dem_independent_morph2026 as d26  # noqa: E402  (the lane's driver; on sys.path via lane_modules)
    dem26 = JL("dem_independent_morph2026.jsonl")
    need({(r["target"], r["basis"], r["rounds"]) for r in dem26} == set(d26.campaign_models())
         and len(dem26) == len(set(d26.campaign_models())),
         "the independent DEM record for the 2026 circuits does not cover exactly the campaign's models")
    need(all(r["equal"] and r["only_independent"] == 0 and r["only_stim"] == 0 for r in dem26),
         "the independent DEM derivation differs from stim's on a 2026 model")
    put("DemTwentySixModels", len(dem26), "records/dem_independent_morph2026.jsonl")
    put("DemTwentySixMechLo", I(min(r["mechanisms"] for r in dem26)), "records/dem_independent_morph2026.jsonl")
    put("DemTwentySixMechHi", I(max(r["mechanisms"] for r in dem26)), "records/dem_independent_morph2026.jsonl")

    # -------------------------------------------- the [[72,12,6]] IBM-style cycle
    bb = res["bb72"]["record"]
    need(bb["found_weight"] == bb["none_up_to"] + 1 and bb["lift"]["verified"] and bb["lift"]["minus_one_fails"],
         "bb72 record changed")
    # The schedule gate's baseline already sits at the code distance, so "none improved it" is not a finding: the
    # paper must say none COULD (the referee's finding 14). Held here so the sentence cannot outlive the fact.
    need(bb["found_weight"] == small[2], "the IBM-style small circuit no longer sits at its code distance")
    need(bb["found_weight"] == int(M["PriorSmallD"][0]), "our [[72,12,6]] value disagrees with the published exact one")
    put("BbD", bb["found_weight"], RJ + ": bb72")
    put("BbRounds", bb["rounds"], RJ + ": bb72")

    # -------------------------------------------- the 2026 circuits
    tg = res["morph2026"]["targets"]
    specs = J(LANE / "morph2026_targets.json")
    vsa = {r["target"]: r for r in JL("morph2026_vs_authors.jsonl")}
    need(any(not r["gates_identical"] for k, r in vsa.items() if k.endswith("must FAIL)")),
         "the authors-builder comparison has no failing control")
    need(set(specs) == set(tg), "the 2026 target list differs between the campaign and its specification")
    rows, losses, kernel, tree_tmax, groups = [], {}, {}, set(), {}
    for name in sorted(tg, key=lambda n: (n[0] != "c", n)):
        t, sp = tg[name], specs[name]
        z, x = t["bases"]["Z"], t["bases"]["X"]
        need(z and x and z["d_circ"] == x["d_circ"], f"{name}: the two bases differ; the table prints one value")
        need(z["loses"] == x["loses"] == t["end_cycle_d"] - z["d_circ"], f"{name}: loss is not d_end - d_circ")
        if t["schedule"] == "standard":
            need(sp["kind"] == "std" and sp["m"] == 1 and vsa[name]["A"] == sp["A"] and vsa[name]["B"] == sp["B"]
                 and vsa[name]["gates_identical"] and t["matches_authors_builder"],
                 f"{name}: the specification is not the one compared with the authors' builder")
            groups.setdefault(t["n_end"], set()).add(sp["l"])
            A = ", ".join(monomial(a, b) for a, b in sp["A"])
            B = ", ".join(monomial(a, b) for a, b in sp["B"])
            need(z["every_T"].startswith("PROVEN") and x["every_T"].startswith("PROVEN"),
                 f"{name}: a standard schedule without the fold theorem")
            every = r"every $T$"
        else:
            need(sp["kind"] == "tree", f"{name}: unknown schedule kind")
            groups.setdefault(t["n_end"], set()).add(sp["N"])
            A = ", ".join(monomial(a) for a in sp["A"])
            B = ", ".join(monomial(a) for a in sp["B"])
            need(x["every_T"] == z["every_T"], f"{name}: every-T classes differ between bases")
            m_ = RX(r"^MACHINE-CHECKED for T <= (\d+), DERIVED beyond", z["every_T"], f"{name} every-T")
            tree_tmax.add(int(m_.group(1)))
            every = rf"MC, $T\le {m_.group(1)}$"
        for b, v in (("Z", z), ("X", x)):
            if v["kernel_checked_T1"]:
                need(v["kernel_checked_T1"] in receipt["theorems"]
                     and set(receipt["theorems"][v["kernel_checked_T1"]]) <= STANDARD_AXIOMS,
                     f"{name} {b}: its kernel theorem is not in the receipt on the standard axioms")
                kernel[(name, b)] = (v["kernel_checked_T1"], v["d_circ"])
        nk = sum((name, b) in kernel for b in "ZX")
        need(nk in (0, 2), f"{name}: kernel-checked in one basis only; the table prints both or neither")
        losses[name] = z["loses"]
        rows.append(rf"\texttt{{{name}}} & ${code_tex(t['n_end'], t['k'], t['end_cycle_d'])}$ & {t['degree']} & "
                    rf"{t['schedule']} & $({A})$ & $({B})$ & {z['d_circ']} / {x['d_circ']} & {z['loses']} & "
                    rf"{every} & {'Z, X' if nk else '--'} \\")
    std = [n for n in tg if tg[n]["schedule"] == "standard"]
    trees = [n for n in tg if tg[n]["schedule"] == "tree"]
    need(len(std) + len(trees) == len(tg) and trees, "the 2026 targets are not standard or tree schedules")
    # The tree circuits print identically in every column (same A, B, case); only their contraction trees differ, so
    # the caption lists each tree (the referee's finding 7), read from the specification the builder ran.
    need(len({(tuple(specs[n]["A"]), tuple(specs[n]["B"]), specs[n]["case"]) for n in trees}) == 1,
         "the tree circuits no longer share A, B and case, so the caption's shared-polynomial sentence is false")
    need(len({json.dumps(specs[n]["sched"]) for n in trees}) == len(trees), "two tree circuits share a schedule")
    nsup = len(specs[trees[0]]["A"]) + len(specs[trees[0]]["B"])
    need(all(0 <= v < nsup for n in trees for layer in specs[n]["sched"] for pair in layer for v in pair),
         "a tree schedule names a vertex outside the support")
    put("MtwoTreeSupport", nsup - 1, "morph2026_targets.json (support vertices 0..w-1)")
    put("MtwoTreeALast", len(specs[trees[0]]["A"]) - 1, "morph2026_targets.json")
    put("MtwoTreeBFirst", len(specs[trees[0]]["A"]), "morph2026_targets.json")
    put("MtwoTreeSchedules", "; ".join(
        rf"\texttt{{{n}}}: $" + r"\mid ".join(", ".join(rf"{p}{{\to}}{c}" for p, c in layer)
                                              for layer in specs[n]["sched"]) + "$" for n in trees),
        "morph2026_targets.json: sched of the tree targets")
    put("MtwoTargets", len(tg), RJ + ": morph2026 targets")
    put("MtwoMaxLoss", max(losses.values()), RJ + ": morph2026 targets")
    put("MtwoTreeTmax", one(tree_tmax, "tree every-T bound"), RJ + ": morph2026 tree targets every_T")
    need(all(len(v) == 1 for v in groups.values()), "a 2026 end-cycle family spans several groups")
    by_n = {}
    for n, t in tg.items():
        by_n.setdefault(t["n_end"], set()).add((t["k"], t["end_cycle_d"]))
    need(all(len(v) == 1 for v in by_n.values()) and len(by_n) == 3, "the 2026 end-cycle codes changed")
    n30, n42, n48 = sorted(by_n)
    d48 = [tg[n]["bases"]["Z"]["d_circ"] for n in tg if tg[n]["n_end"] == n48]
    put("MtwoBestFortyEight", max(d48), RJ + ": morph2026 targets (the 48-qubit family)")
    put("MtwoFortyEightBestCount", d48.count(max(d48)), RJ + ": morph2026 targets (the 48-qubit family)")
    put("MtwoFortyEightCount", len(d48), RJ + ": morph2026 targets (the 48-qubit family)")
    for nn, tag in ((n30, "Thirty"), (n42, "FortyTwo"), (n48, "FortyEight")):
        k_, d_ = by_n[nn].pop()
        put(f"MtwoCode{tag}", code_tex(nn, k_, d_), RJ + ": morph2026 targets")
        put(f"MtwoGroup{tag}", groups[nn].pop(), "experiments/circuit_distance_exact/morph2026_targets.json")
        if nn == n48:
            put("MtwoEndDFortyEight", d_, RJ + ": morph2026 targets")
    keep42 = [n for n in tg if tg[n]["n_end"] == n42 and losses[n] == 0]
    need(len(keep42) == 1, "the 42-qubit circuit that keeps its distance is no longer unique")
    put("MtwoKeeper", keep42[0], RJ + ": morph2026 targets")
    put("MtwoFortyTwoDegree", tg[keep42[0]]["degree"], RJ + ": morph2026 targets")
    put("MtwoKeeperD", tg[keep42[0]]["bases"]["Z"]["d_circ"], RJ + ": morph2026 targets")
    put("MtwoFortyTwoMaxLoss", max(losses[n] for n in tg if tg[n]["n_end"] == n42 and n != keep42[0]),
        RJ + ": morph2026 targets (the 42-qubit family)")
    # "the CNOT order ALONE changes the distance by up to" is a claim about pairs that differ ONLY in order -- same
    # end-cycle code, same group, same polynomial terms -- not the largest loss against the code (the final referee,
    # 2026-11-03: the two numbers agreed only through one pair)
    def terms(n):
        return (tg[n]["n_end"], specs[n]["l"], frozenset(map(tuple, specs[n]["A"])), frozenset(map(tuple, specs[n]["B"])))
    pairs = [(a, b) for a in std for b in std if a < b and terms(a) == terms(b)
             and (specs[a]["A"], specs[a]["B"]) != (specs[b]["A"], specs[b]["B"])]
    need(pairs, "no two standard-schedule circuits differ only in CNOT order; the text says the order alone matters")
    put("MtwoOrderMax", max(abs(tg[a]["bases"][q]["d_circ"] - tg[b]["bases"][q]["d_circ"])
                            for a, b in pairs for q in "ZX"),
        RJ + ": morph2026 targets (pairs differing only in CNOT order)")
    put("MtwoOrderPairs", len(pairs), RJ + ": morph2026 targets (pairs differing only in CNOT order)")
    put("MtwoTreeDegree", one((tg[n]["degree"] for n in trees), "tree degree"), RJ + ": morph2026 targets")
    need(all(losses[n] == 0 for n in trees), "a tree circuit loses distance; the text says none does")
    s42 = next(r for r in JL("morph2026_codes_N42.jsonl") if r.get("summary"))
    s30 = next(r for r in JL("morph2026_codes_N30.jsonl") if r.get("summary"))
    need(s42["N"] == n42 and s30["N"] == n30, "the code-search summaries are for other groups")
    need(s42["circuits_exact"] == len([n for n in tg if tg[n]["n_end"] == n42]), "the 42-qubit search and the "
                                                                                  "campaign disagree")
    need(s30["circuits_hit"] == 0, "the standard schedule now reaches the 30-qubit code; the text says it does not")
    put("MtwoFortyTwoCircuits", s42["circuits_exact"], "records/morph2026_codes_N42.jsonl summary")
    camp = JL("morph2026_campaign.jsonl")
    kk = sorted({n for n, _ in kernel})
    need(len(kk) == 2 and len(kernel) == 4, "the kernel-checked targets changed")
    for tag, n in (("A", kk[0]), ("B", kk[1])):
        put(f"Kernel{tag}", n, RJ + ": morph2026 kernel_checked_T1")
        put(f"KernelD{tag}", one((kernel[(n, b)][1] for b in "ZX"), f"{n} kernel distance"), RJ + ": morph2026")
        mech = one((r["mechanisms"] for r in camp if r["kind"] == "search" and r.get("code") == n and r["rounds"] == 1),
                   f"{n} 1-round model size")
        put(f"KernelMech{tag}", mech, "records/morph2026_campaign.jsonl (T = 1 search)")
    tab_morph = "\n".join([
        r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
        r"circuit & end-cycle code & degree & schedule & $A$ & $B$ & $d_{\mathrm{circ}}$ (Z / X) & loses & "
        r"rounds & kernel ($T=1$) \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"

    # -------------------------------------------- Lean
    th = receipt["theorems"]
    need(receipt["build_exit"] == 0 and all(set(a) <= STANDARD_AXIOMS for a in th.values()),
         "a theorem in the receipt rests on a non-standard axiom")
    for t in ("CertR198.cert_sound", "CertR198.cert_sound_scoped", "CertR198.Exec.cert2_sound_exec",
              "CertR198.Symmetry.anchoring", "CertR198.DEM.bound5_sound", "CertR198.Valid.obs_eq_zero_of_syn_zero"):
        need(t in th, f"the receipt lacks {t}, which the text names")
    need(receipt["axiom_check_lines"] == len(th), "the receipt's axiom check covers a different theorem count")
    put("LeanTheorems", len(th), "lean_r198_receipt.json")
    nker = sum(1 for t in th if ".Kernel." in t)
    need(nker == len(kernel), "the receipt and results.json disagree on the kernel theorems")
    put("LeanKernel", nker, "lean_r198_receipt.json")
    lv = RX(r"^leanprover/lean(\d+):(v\d+\.\d+\.\d+)$", receipt["lean"], "Lean toolchain")
    put("LeanName", f"Lean~{lv.group(1)}", "lean_r198_receipt.json")
    put("LeanVersion", lv.group(2), "lean_r198_receipt.json")
    ml = RX(r"Toolchain `leanprover/lean4:(v[\d.]+)`, mathlib `(v[\d.]+)`", spec_md, "SPEC.md toolchain")
    need(ml.group(1) == lv.group(2), "SPEC.md and the receipt name different Lean versions")
    put("MathlibVersion", ml.group(2), "lean/CertR198/SPEC.md")
    put("LeanProject", receipt["project"], "lean_r198_receipt.json")
    kt = RX(r"Kernel time (\d+)-(\d+) s per model at <= ([\d.]+) GB resident", results_md, "RESULTS.md, kernel cost")
    put("KernelSLo", kt.group(1), "RESULTS.md: part 18")
    put("KernelSHi", kt.group(2), "RESULTS.md: part 18")
    put("KernelGB", kt.group(3), "RESULTS.md: part 18")
    ex = RX(r"the (\d+) model files then in the campaign directories had\s+[\d,]+ mechanism lines and (\d+) repeated "
            r"detectors, so F1 and F5 never affected a recorded result", spec_md, "SPEC.md, exposure")
    need(ex.group(2) == "0", "SPEC.md now records a repeated detector in a campaign model")
    put("ExposureModels", ex.group(1), "lean/CertR198/SPEC.md: exposure")
    orc = RX(r"on ([\d,]+) queries over (\d+) random models \(breadth-first search.*?no prune ever cut a completion",
             results_md, "RESULTS.md, oracle")
    put("OracleQueries", I(int(orc.group(1).replace(",", ""))), "RESULTS.md: independent review of R205/R206")
    put("OracleModels", orc.group(2), "RESULTS.md: independent review of R205/R206")
    mut = RX(r"(\w+) one-token mutants, one per rule .*?each\s+produce violations \((\d+) to ([\d,]+)\)", results_md,
             "RESULTS.md, mutants")
    put("OracleMutants", word(mut.group(1), "RESULTS.md mutant count"), "RESULTS.md: independent review of R205/R206")
    put("OracleViolLo", mut.group(2), "RESULTS.md: independent review of R205/R206")
    put("OracleViolHi", I(int(mut.group(3).replace(",", ""))), "RESULTS.md: independent review of R205/R206")
    put("CoverScale", RX(r"weight\[d\] = \(unsigned short\)\((\d+) / max_cover\)", ccheck_c, "ccheck.c cover").group(1),
        "experiments/circuit_distance_exact/ccheck.c")

    # -------------------------------------------- the schedule gate (R206)
    recs = reg["records"]
    verdicts = [r["verdict"] for r in recs if r["name"] != "baseline"]
    base = next(r for r in recs if r["name"] == "baseline")
    need(reg["sample_size"] == reg["side_matched_proposals"] and reg["candidate_count_including_baseline"] == len(recs)
         and reg["valid_two_basis_schedules"] == len(verdicts), "the R206 registry's counts do not reconcile")
    need(all(b["status"] == "PROJECTED_EXACT_AND_FULL_UPPER_WITNESS" and b["full_witness_verified"]
             for r in recs for b in r["basis"]), "an R206 schedule lacks a certified distance and a verified witness")
    need(one((b["fault_weight"] for b in base["basis"]), "R206 baseline distance") == small[2],
         "the R206 baseline is not the small IBM-style cycle at its distance")
    same = [r for r in recs if r["verdict"] == "SAME_PROJECTED_DISTANCE__LER_NOT_MEASURED"]
    worse = [r for r in recs if r["verdict"] == "REJECTED_DISTANCE_REGRESSION"]
    need(len(same) + len(worse) == len(verdicts), "an R206 verdict is neither same nor regressed; the text says none "
                                                  "improved")
    need(all(one((b["fault_weight"] for b in r["basis"]), "R206 same") == small[2] for r in same),
         "an R206 'same' schedule is not at the baseline distance")
    sr = RX(r"^(\[\[\d+,\d+,\d+\]\]) IBM-style BB, (\w+) memory rounds", reg["input_model"], "R206 input model")
    need(sr.group(1) == code_tex(*small), "the R206 schedule search is not on the small IBM-style code")
    put("SchedRounds", word(sr.group(2), "R206 rounds"), "r206_schedule_discovery/registry.json: input_model")
    put("SchedProposals", reg["sample_size"], "r206_schedule_discovery/registry.json")
    put("SchedValid", reg["valid_two_basis_schedules"], "r206_schedule_discovery/registry.json")
    put("SchedSame", len(same), "r206_schedule_discovery/registry.json")
    put("SchedRegressed", len(worse), "r206_schedule_discovery/registry.json")
    put("SchedRegressedD", one((b["fault_weight"] for r in worse for b in r["basis"]), "R206 regressed distance"),
        "r206_schedule_discovery/registry.json")

    # -------------------------------------------- the morphing end-cycle code is not the gross code (audit 2, F09)
    # Read off the emitted circuit, not the builder's algebra: Stim accepts a final detector only if it is
    # deterministic, so the memory-type product over its data readouts is an end-cycle stabilizer. Every element of
    # the gross code's stabilizer group has even weight (weight-6 generators that commute); qubit permutations and
    # local Cliffords preserve Pauli weight.
    import bb_circuits as bcm  # noqa: E402  (on sys.path via lane_modules)
    import numpy as np  # noqa: E402
    gx, gz = bcm.BB(12, 6).check_matrices()
    gw = {int(w) for w in np.concatenate([gx.sum(1), gz.sum(1)])}
    need(gw == {6}, "the gross code's generators are not all of weight 6")
    end_w = set()
    for b in "ZX":
        circ = bcm.morphing_cycle("morph144", b, 1)
        circ.detector_error_model(decompose_errors=False)
        first_data, k_ = circ.num_measurements - 144, 0
        for ins in circ.flattened():
            if ins.name in ("M", "MX"):
                k_ += len(ins.targets_copy())
            elif ins.name == "DETECTOR":
                nd = sum(1 for t in ins.targets_copy() if k_ + t.value >= first_data)
                if nd:
                    end_w.add(nd)
    need(len(end_w) == 1 and next(iter(end_w)) % 2 == 1,
         "the morph144 end-cycle stabilizers read off the circuit are not all of one odd weight")
    put("MorphEndStabW", next(iter(end_w)), "bb_circuits.morphing_cycle('morph144', b, 1): final detectors")
    put("GrossStabW", next(iter(gw)), "bb_circuits.BB(12, 6) check matrices")

    # -------------------------------------------- our circuits against Pan's own (audit 1/2, finding P3-09)
    pan = JL("pan_crosscheck.jsonl")
    ph, prow = pan[0], [r for r in pan[1:] if r.get("kind") != "control"]
    pctl = [r for r in pan[1:] if r.get("kind") == "control"]
    need(ph.get("kind") == "header" and ph["all_equal"] and ph["controls_all_fail"],
         "the Pan cross-check is not all-equal with every control failing")
    need(ph["ours"]["bb_circuits"] == sha256(LANE / "bb_circuits.py")
         and ph["ours"]["export_model"] == sha256(LANE / "export_model.py")
         and ph["ours"]["this_script"] == sha256(LANE / "pan_crosscheck.py"),
         "the Pan cross-check was run on other builder, exporter or script bytes; re-run pan_crosscheck.py")
    need(ph["pan_release"]["tag_commit"].startswith("5ae848ab"), "the Pan cross-check is not pinned to v2.0")
    kinds_ = {(r["circuit"], r["basis"], r["T"]) for r in prow}
    need(kinds_ == {(c, b, t) for c in ("ibm", "3c", "morph72", "morph144") for b in "ZX" for t in ph["rounds"]}
         and all(r["equal"] and r["observable_map_rank"] == 12 and r["ours_mechanisms"] == r["export_model_mechanisms"]
                 for r in prow), "the Pan cross-check does not cover every circuit, basis and round count equal")
    need({(r["circuit"], r["basis"]) for r in pctl} == {(c, b) for c in ("ibm", "3c", "morph72", "morph144") for b in "ZX"}
         and all(r["fails_as_required"] for r in pctl), "a Pan cross-check control is missing or did not fail")
    put("PanXRows", len(prow), "records/pan_crosscheck.jsonl")
    put("PanXControls", len(pctl), "records/pan_crosscheck.jsonl")
    put("PanXTmin", min(ph["rounds"]), "records/pan_crosscheck.jsonl")
    put("PanXTmax", max(ph["rounds"]), "records/pan_crosscheck.jsonl")

    # -------------------------------------------- tables
    need(int(m72_d) == PAN_E019_MORPH72 and int(m144_d) == PAN_E017_MORPH144 and int(ibm_d) == PAN_E043_IBM
         and int(tc_d) == PAN_V2_TC,
         "our distances differ from the values Pan's records and manuscript report; the text says they agree")
    G = code_tex(*gross)
    tab_summary = "\n".join([
        r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
        # two-line headers for the two widest heads, so the prior-value column has room for two values
        r"circuit & code (mid-cycle) & prior value & $d_{\mathrm{circ}}$ (Z / X) & "
        r"\begin{tabular}[b]{@{}c@{}}lower bound\\ certified\end{tabular} & "
        r"\begin{tabular}[b]{@{}c@{}}witness\\ verified\end{tabular} & holds for \\", r"\midrule",
        # Pan's records on his main branch after v2.0 (final referee of 2026-11-03, N1; literature search of
        # 2026-11-07): certificates for R <= 12 (E-037), every R under SD6 (E-043) and in his re-implementation of
        # IBM's fault model (E-045)
        rf"IBM depth-{lit_depth}~\cite{{bravyi2024high}} & ${G}$ & $\le {q.group(1)}$; "
        rf"{PAN_E043_IBM}~\cite{{pan2026e037,pan2026e043,pan2026e045}} & {ibm_d} / {ibm_d} & "
        rf"$T={ibm_t_lo},\dots,{ibm_t_hi}$ & $T={ibm_dir[0]},\dots,{ibm_dir[1]}$ & $T\ge {ibm_tmin}$ \\",
        # the source's conjecture, then the value of Pan's released manuscript for every number of rounds
        rf"three-colour & ${G}$ & {q.group(3)} (conj.); {PAN_V2_TC}~\cite{{pan2026interleaving}} & {tc_d} / {tc_d} & "
        rf"$T={tc_s_lo},\dots,{tc_s_hi}$ & $T={tc_e_lo},\dots,{tc_e_hi}$ & $T\ge {tc_tmin}$ \\",
        # prior value: the source paper's bound, then the value Pan's released research record reports (by exhaustive
        # search and fold checks, no certificate; external audit 2, F11). Pan's values are typed from his records
        # E-019 Sec. 2.1 and E-017 Sec. 8 (read 2026-11-03), and must equal ours: a disagreement is a refusal.
        rf"morphing & ${code_tex(*small)}$ (${G}$) & $\le {M['LitMorphSmall'][0]}$; "
        rf"{PAN_E019_MORPH72}~\cite{{pan2026e019}} & {m72_d} / {m72_d} & "
        rf"$T={m72_t[0]},\dots,{m72_t[1]}$ & $T={m72_e_t[0]},\dots,{m72_e_t[1]}$ & $T\ge {morph_tmin}$ \\",
        rf"morphing & ${G}$ (${code_tex(*big_mid)}$) & $\le {lit144.group(1)}$; "
        rf"{PAN_E017_MORPH144}~\cite{{pan2026e017}} & {m144_d} / {m144_d} & "
        rf"$T={', '.join(str(t) for t in m144_ct)}$ (fold: $T\le {m144_tmax}$) & $T={e144[0]},\dots,{e144[1]}$ & "
        rf"$T\ge {morph_tmin}$ \\",
        rf"IBM-style & ${code_tex(*small)}$ & {M['PriorSmallD'][0]} (exact)~\cite{{beni2025tesseract}} & "
        rf"{bb['found_weight']} / -- & $T={bb['rounds']}$ & "
        rf"$T={bb['rounds']}$ & -- \\",
        r"\bottomrule", r"\end{tabular}"]) + "\n"
    cost_rows = []
    for b in "ZX":
        r = ibm_cell[(b, 2)]
        cost_rows.append((rf"IBM, $T=2$, {b}", I(nm), ibm_w, f"{D1(r['proof_bytes'] / 1e6)} MB", D1(r["prover_s"]),
                          D1(r["checker_s"])))
    for b, key, nbytes in (("Z", "ibm_Z_weight9", oz["cdp5_proof_bytes"]),
                           ("X", "ibm_X_weight9", pc["ibm_X_weight9"]["proof_bytes"])):
        cost_rows.append((rf"IBM, $T=2$, {b}, {man['proof_format']}", I(nm), ibm_w, f"{KB(nbytes)} KB",
                          D1(pc[key]["prover"]["r206_median_s"]), D1(pc[key]["checker"]["r206_median_s"])))
    # three-colour T = 2, shift and morphing rows: the re-certification's CDP5 proofs by the committed prover and
    # checker (one format, one campaign), joined to the records for the model sizes by the model's hash
    rec_cell = {(c["family"], c["basis"], c["rounds"], c["model_sha256"]): c for c in cells}
    for b in "ZX":
        c = rec_cell[("3c_t2", b, 2, t2[b]["model_sha256"])]
        cost_rows.append((rf"three-colour, $T=2$, {b}", I(t2[b]["mechanisms"]), tc_w, f"{MB(c['proof_bytes'])} MB",
                          R0(c["prover_s"]), R0(c["checker_s"])))
    for b in "ZX":
        rs = [r for r in sh if r["basis"] == b]
        cs = [rec_cell[("3c", b, r["rounds"], r["model_sha256"])] for r in rs]
        lo_, hi_ = enclose(min(c["proof_bytes"] for c in cs) / 1e6, max(c["proof_bytes"] for c in cs) / 1e6, 1)
        cost_rows.append((rf"three-colour, shift, {b} ($\times {len(rs)}$)",
                          f"{I(min(r['mechanisms'] for r in rs))}--{I(max(r['mechanisms'] for r in rs))}", tc_w,
                          f"{lo_}--{hi_} MB", f"{hours(sum(c['prover_s'] for c in cs))} h",
                          f"{hours(sum(c['checker_s'] for c in cs))} h"))
    for t in m144_ct:
        for b in "ZX":
            r = c144[(b, t)]
            c = rec_cell[("morph144", b, t, r["model_sha256"])]
            cost_rows.append((rf"morphing ${G}$, $T={t}$, {b}", I(r["mechanisms"]), m144_w,
                              f"{MB(c['proof_bytes'])} MB", R0(c["prover_s"]), R0(c["checker_s"])))
    tab_costs = "\n".join([
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"model & mechanisms & $w$ & proof & prove (s) & check (s) \\", r"\midrule",
        *[rf"{a} & {b} & {c} & {d} & {e} & {f} \\" for a, b, c, d, e, f in cost_rows],
        r"\bottomrule", r"\end{tabular}"]) + "\n"

    body = ["% GENERATED by paper/tools/make_numbers_distance.py from committed evidence -- do not edit.",
            "% Each macro is preceded by the file it was read from. Ranges enclose the data."]
    for k in sorted(M):
        v, src = M[k]
        body.append(f"% {src}")
        body.append(rf"\newcommand{{\{k}}}{{{v}}}")
    return {"numbers.tex": "\n".join(body) + "\n", "tab_summary.tex": tab_summary, "tab_morph.tex": tab_morph,
            "tab_costs.tex": tab_costs, "tab_recert.tex": tab_recert}


def main() -> int:
    files = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print("wrote", (OUT / name).relative_to(ROOT).as_posix() if (OUT / name).is_relative_to(ROOT) else OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
