r"""R211: the Helios run's statistical validity theorem, machine-checked in Lean -- producer, mutant drill, Lean-free gate.

    python r203_lean_receipt.py --write-statement            # regenerate Statement.lean from the committed package
    python r203_lean_receipt.py --build C:/lq/BettingR203    # PRODUCER: copy lean/BettingR203 to a SHORT path, lake
                                                            # build, print every theorem's axioms, write the receipt
    python r203_lean_receipt.py --mutants C:/lq/BettingR203  # after --build: Lean must refuse every MUTANTS edit
    python r203_lean_receipt.py                             # GATE (default): needs no Lean

WHAT IS PROVED (lean/BettingR203, see its SPEC.md), in the Lean 4 kernel against Mathlib v4.34.0, standard axioms only:
  - `ville_mixture`: for ANY score process W adapted to any filtration, in [0, Wmax], whose conditional mean given the
    past is a predictable mu <= Wmax, and any finite mixture of nonnegative constant bets safe against Wmax,
    P(some look t <= T has K_t >= 1/delta while t m <= sum_{i<t} mu_{i+1}) <= delta. The event ranges over every
    t <= T: every stopping rule, look schedule and R209's futility stop (which only stops without claiming).
  - `clippedMean_spec` / `ville_any_device`: the conditional expectation clipped into [0, Wmax] is such a mu, so the
    hypotheses are on the score process and the bets only.
  - `package_valid_exact_threshold`: for the frozen package -- m, both scores, the bets, equal weights and delta as
    the exact values of the doubles the producer computes with -- with the null at the design's EXACT F_e = 3/4
    deficit 63/1156 (`m_le_exact`: the producer's double m lies below it, so its test is conservative).
  - `Statement.frozen_run_valid`: THE BINDING. Statement.lean is GENERATED from the committed package with every
    number written as a literal, and Lean elaborates it from `package_valid_exact_threshold`, so the kernel checks
    that the proved theorem is this statement about these numbers. The gate regenerates the file and refuses any
    difference; `--mutants` shows Lean refusing the edits that would otherwise decouple them (lean_mutants.json).

R225 (2026-10-26): the package is read by rule 'betting_ad', the fixed mixture of R203's F_e wealth and the
antidegradability witness w*'s (ion_betting.verdict_ad). The gate binds a SECOND statement to it:
  - `ville_family` (Family.lean): `ville_mixture` with each component's own score process, threshold and bound;
  - `PackageAD.package_ad_valid_exact_threshold`: the six components' doubles, with the null at BOTH exact thresholds
    (63/1156 for F_e; w* . lam(ideal) - hbar for the witness, `mW_le_exact`);
  - `StatementAD.frozen_run_valid_ad`: generated like Statement.lean, every number a literal; four more mutants on
    PackageAD.lean must be refused by it. R211's statement stays, bound to the same package's F_e constants.
  The float decision against the reals is r225_float_guard.py's (a derived bound under a 1e-9 guard), not
  r203_float_boundary.py's enumeration.

WHAT IS NOT PROVED (named in SPEC.md and the prereg):
  - that the score's conditional mean is the implemented design channel's deficit sum_g c_g e_g(i): iid labels
    drawn independently of the past and BEHAVIOURAL non-anticipation (the device-time premise; DERIVED);
  - the producer's double-precision decision against the reals: `r203_float_boundary.py` checks it on every
    reachable state of the frozen package (MEASURED, with a derived error bound), not Lean;
  - one submission per batch (ion_sequential.verdict enforces it from the job list);
  - the Lean kernel and Mathlib.

THE GATE TRUSTS THE RECEIPT. It runs no Lean (the wall cannot), so a PASS means: a receipt exists whose source digests
are the committed sources (Statement.lean included), which says the build exited 0 with standard axioms for exactly
AxiomCheck's theorems, under a recorded Lean and a clean mathlib at the manifest's revision. Anyone with Lean re-derives
it with --build. The gate REFUSES when:
  - a source changed since the receipt, or Statement.lean is not the one generated from the committed package;
  - the theorem lists differ or a headline theorem is missing;
  - any theorem rests on an axiom beyond propext / Classical.choice / Quot.sound (sorryAx named separately);
  - the build did not exit 0, or the recorded toolchain is not the pinned one;
  - the mutant drill is missing, stale, or a mutant was not refused by the statement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEAN = ROOT / "lean" / "BettingR203"
STATEMENT = "BettingR203/Statement.lean"
STATEMENT_AD = "BettingR203/StatementAD.lean"          # R225: the package read by rule 'betting_ad'
AD_RULE = "betting_ad"
RECEIPT = ROOT / "evidence" / "superactivation" / "r203_betting" / "lean_receipt.json"
MUTANT_RECEIPT = ROOT / "evidence" / "superactivation" / "r203_betting" / "lean_mutants.json"
PACKAGE = ROOT / "evidence" / "superactivation" / "ion_run_m8_r224"   # R224 re-freeze (both erasure orders)
STANDARD = {"propext", "Classical.choice", "Quot.sound"}
HEADLINE = ("BettingR203.ville_mixture", "BettingR203.ville_any_device", "BettingR203.Package.package_valid_any_device",
            "BettingR203.Package.package_valid_exact_threshold", "BettingR203.Statement.frozen_run_valid")
HEADLINE_AD = ("BettingR203.ville_family", "BettingR203.ville_family_any_device",
               "BettingR203.PackageAD.package_ad_valid_exact_threshold", "BettingR203.StatementAD.frozen_run_valid_ad")
SHARED_PACKAGES = pathlib.Path(r"C:\Users\Josh\lq\OneqCert\.lake\packages")   # mathlib v4.34.0, already built
BELOW_NORMAL = 0x00004000   # Windows priority class; lake's lean children inherit it
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import ion_betting as B  # noqa: E402
from roster_digest import roster_digest  # noqa: E402  (the repo's one checkout-invariant source digest)


class ReceiptRefused(ValueError):
    pass


def sources(root: pathlib.Path = LEAN) -> dict[str, str]:
    """repo path (as lean/BettingR203/...) -> digest of its text with CRLF read as LF. What Lean checked does not
    depend on line endings, and the repo-relative key lets tools/hash_pin_audit.py replay each digest against
    history wherever the build copy was made."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and not rel.startswith(".lake/") and (p.suffix == ".lean" or p.name in
                                                              ("lakefile.toml", "lean-toolchain", "lake-manifest.json")):
            out[f"lean/BettingR203/{rel}"] = roster_digest(p)
    return out


def declared(root: pathlib.Path = LEAN) -> list[str]:
    return re.findall(r"^#print axioms (\S+)", (root / "BettingR203" / "AxiomCheck.lean").read_text(encoding="utf-8"),
                      re.M)


def parse_axioms(text: str) -> dict[str, list[str]]:
    """Lean prints "'X' depends on axioms: [a, b]" or "'X' does not depend on any axioms" per #print axioms."""
    out = {}
    for m in re.finditer(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text):
        out[m.group(1)] = sorted(a.strip() for a in m.group(2).replace("\n", " ").split(",") if a.strip())
    for m in re.finditer(r"'([^']+)' does not depend on any axioms", text):
        out[m.group(1)] = []
    return out


def _frac(x) -> list[int]:
    f = Fraction(x)
    return [f.numerator, f.denominator]


def exact_threshold(design: dict) -> Fraction:
    """the deficit at which the design channel has F_e = bar_F_e, EXACTLY: f_ideal = (2 f_e(p) + 1) / 3 with
    f_e(p) = (7 - 6p + 2p^2) / 8 and p = 1 / (sqrt(2^m) + 1) (hw_two_use), rational when m is even"""
    m = int(design["m"])
    root = math.isqrt(2 ** m)
    if root * root != 2 ** m:
        raise ReceiptRefused(f"shield size m = {m} is odd: p_flip = 1/(sqrt(2^m) + 1) is irrational")
    p = Fraction(1, root + 1)
    f_ideal = (2 * (7 - 6 * p + 2 * p * p) / 8 + 1) / 3
    if abs(float(f_ideal) - design["f_ideal"]) > 4e-16:
        raise ReceiptRefused(f"the design's f_ideal {design['f_ideal']!r} is not the closed form's {f_ideal}")
    return f_ideal - (Fraction(design["bar_F_e"]) + Fraction(1, 2)) / Fraction(3, 2)


def package_constants(package: pathlib.Path = PACKAGE) -> dict:
    """the exact rationals of the doubles ion_betting computes the frozen package's test with, and the exact
    threshold; refused when the double m is ABOVE the exact one (the test would not be conservative)"""
    design = json.loads((package / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((package / "sequential.json").read_text(encoding="utf-8"))
    if seq.get("rule") not in ("betting", AD_RULE):
        raise ReceiptRefused(f"the package's rule is {seq.get('rule')!r}, not the betting test the theorem is about")
    scores = B.scores(design, seq["pi_rare"])
    m = B.threshold(design)["m"]
    m_exact = exact_threshold(design)
    if Fraction(m) > m_exact:
        raise ReceiptRefused(f"the producer's m {m!r} lies ABOVE the exact threshold {m_exact}: not conservative")
    out = dict(m=_frac(m), m_exact=_frac(m_exact), sRare=_frac(scores[B.RARE]), sCommon=_frac(scores[B.COMMON]),
               bet=[_frac(b) for b in B.check_bets(design, seq["pi_rare"], seq["bets"])], δ=_frac(seq["claim_delta"]))
    if seq["rule"] == AD_RULE:
        out["witness"] = witness_constants(design, seq)
    return out


def witness_constants(design: dict, seq: dict) -> dict:
    """R225: the witness components' doubles (ion_betting.ad_components) and the witness's EXACT threshold
    w . lam(ideal) - hbar; refused when the double m_w is ABOVE it, or the components are not three F_e and three
    witness bets"""
    import ad_witness as A
    comps = B.ad_components(design, seq)
    if [c["test"] for c in comps] != ["F_e"] * 3 + ["witness"] * 3:
        raise ReceiptRefused(f"the mixture's components {[c['test'] for c in comps]} are not three F_e and three "
                             "witness bets, the six PackageAD.lean is about")
    wit = comps[3:]
    w = [Fraction(x) for x in seq["witness"]["w"]]
    m_exact = sum(a * b for a, b in zip(w, A.lam_ideal(int(design["m"])))) - Fraction(seq["witness"]["hbar"])
    if Fraction(wit[0]["m"]) > m_exact:
        raise ReceiptRefused(f"the witness's m {wit[0]['m']!r} lies ABOVE its exact threshold {m_exact}")
    k = wit[0]["kappa"]
    return dict(mW=_frac(wit[0]["m"]), mW_exact=_frac(m_exact), kRX=_frac(k["ee_x/high|X"]),
                kKX=_frac(k["kept/high|X"]), kKY=_frac(k["kept/high|Y"]), kKZ=_frac(k["kept/high|Z"]),
                wbet=[_frac(c["bet"]) for c in wit])


def _lit(c: list[int]) -> str:
    return f"({c[0]} / {c[1]} : ℝ)"


def statement_text(c: dict) -> str:
    """Statement.lean for the package constants `c`: the run's statement, every number a literal"""
    bets = ", ".join(_lit(b) for b in c["bet"])
    return f"""import BettingR203.Package

/-!
# The frozen run's validity, with its numbers written out

GENERATED by `experiments/superactivation/shield_family/r203_lean_receipt.py --write-statement` from
`evidence/superactivation/ion_run_m8_r224` (design.json, sequential.json). Do not edit: the gate regenerates this
file from the committed package and refuses any difference.

Every number is a literal: the exact value of the double the producer computes with, and the design's exact
`F_e = 3/4` deficit. Lean elaborates `frozen_run_valid` from `Package.package_valid_exact_threshold`, so the kernel
checks that the proved theorem is exactly this statement about exactly these numbers.
-/

open MeasureTheory Finset

namespace BettingR203.Statement

theorem frozen_run_valid {{Ω : Type*}} {{mΩ : MeasurableSpace Ω}} {{P : Measure Ω}} [IsProbabilityMeasure P]
    {{𝓕 : Filtration ℕ mΩ}} (W : ℕ → Ω → ℝ) (T : ℕ)
    (hW : ∀ i, StronglyMeasurable[𝓕 (i + 1)] (W (i + 1)))
    (hval : ∀ i ω, W (i + 1) ω = 0 ∨ W (i + 1) ω = {_lit(c["sRare"])} ∨
      W (i + 1) ω = {_lit(c["sCommon"])}) :
    P {{ω | ∃ t ≤ T, 1 / {_lit(c["δ"])} ≤
        mixture univ (fun _ => (1 : ℝ) / {len(c["bet"])}) ![{bets}]
          {_lit(c["m"])} W t ω ∧
        (t : ℝ) * {_lit(c["m_exact"])} ≤ ∑ i ∈ range t, clippedMean P 𝓕 W {_lit(c["sRare"])} (i + 1) ω}} ≤
      ENNReal.ofReal {_lit(c["δ"])} :=
  Package.package_valid_exact_threshold W T hW hval

end BettingR203.Statement
"""


def statement_ad_text(c: dict) -> str:
    """StatementAD.lean for the package constants `c` under rule 'betting_ad': the run's R225 statement, every number
    a literal -- the six components (F_e's three bets, the witness's three), both score sets, both exact thresholds"""
    w = c["witness"]
    fe_bets = ", ".join(_lit(b) for b in c["bet"])
    w_bets = ", ".join(_lit(b) for b in w["wbet"])
    return f"""import BettingR203.PackageAD

/-!
# The frozen run's validity under rule `betting_ad` (R225), with its numbers written out

GENERATED by `experiments/superactivation/shield_family/r203_lean_receipt.py --write-statement` from
`evidence/superactivation/ion_run_m8_r224` (design.json, sequential.json). Do not edit: the gate regenerates this
file from the committed package and refuses any difference.

Every number is a literal: the exact value of the double the producer computes with, and the two exact thresholds
-- the design's `F_e = 3/4` deficit and the witness's `w* · λ(ideal) - h̄`. Component `(false, l)` is `F_e` with bet
`l`, `(true, l)` the witness with bet `l`; `WF` and `WW` are their score processes. Lean elaborates
`frozen_run_valid_ad` from `PackageAD.package_ad_valid_exact_threshold`, so the kernel checks that the proved theorem
is exactly this statement about exactly these numbers.
-/

open MeasureTheory Finset

namespace BettingR203.StatementAD

theorem frozen_run_valid_ad {{Ω : Type*}} {{mΩ : MeasurableSpace Ω}} {{P : Measure Ω}} [IsProbabilityMeasure P]
    {{𝓕 : Filtration ℕ mΩ}} (WF WW : ℕ → Ω → ℝ) (T : ℕ)
    (hWF : ∀ i, StronglyMeasurable[𝓕 (i + 1)] (WF (i + 1)))
    (hWW : ∀ i, StronglyMeasurable[𝓕 (i + 1)] (WW (i + 1)))
    (hvalF : ∀ i ω, WF (i + 1) ω = 0 ∨ WF (i + 1) ω = {_lit(c["sRare"])} ∨
      WF (i + 1) ω = {_lit(c["sCommon"])})
    (hvalW : ∀ i ω, WW (i + 1) ω = 0 ∨ WW (i + 1) ω = {_lit(w["kRX"])} ∨
      WW (i + 1) ω = {_lit(w["kKX"])} ∨ WW (i + 1) ω = {_lit(w["kKY"])} ∨
      WW (i + 1) ω = {_lit(w["kKZ"])}) :
    P {{ω | ∃ t ≤ T, 1 / {_lit(c["δ"])} ≤
        family univ (fun _ => (1 : ℝ) / 6)
          (fun j : Bool × Fin 3 => cond j.1 (![{w_bets}] j.2) (![{fe_bets}] j.2))
          (fun j : Bool × Fin 3 => cond j.1 {_lit(w["mW"])} {_lit(c["m"])})
          (fun j : Bool × Fin 3 => cond j.1 WW WF) t ω ∧
        (t : ℝ) * {_lit(c["m_exact"])} ≤ ∑ i ∈ range t, clippedMean P 𝓕 WF {_lit(c["sRare"])} (i + 1) ω ∧
        (t : ℝ) * {_lit(w["mW_exact"])} ≤ ∑ i ∈ range t, clippedMean P 𝓕 WW {_lit(w["kRX"])} (i + 1) ω}} ≤
      ENNReal.ofReal {_lit(c["δ"])} :=
  PackageAD.package_ad_valid_exact_threshold WF WW T hWF hWW hvalF hvalW

end BettingR203.StatementAD
"""


def _read_lf(p: pathlib.Path) -> str:
    return p.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _segment(text: str, start: str) -> tuple[int, int]:
    """[a, b) of the declaration that begins at `start` and ends where the next doc comment or the namespace's end
    begins"""
    a = text.find(start)
    if a < 0 or text.count(start) != 1:
        raise ReceiptRefused(f"mutant anchor {start!r} is not unique in the package file")
    ends = [x for x in (text.find("\n/--", a), text.find("\nend ", a)) if x >= 0]
    return a, (min(ends) if ends else len(text))


def _in_theorem(text: str, theorem: str, old: str, new: str) -> str:
    a, b = _segment(text, f"theorem {theorem}")
    seg = text[a:b]
    if seg.count(old) != 1:
        raise ReceiptRefused(f"mutant edit {old!r} does not occur exactly once in {theorem}")
    return text[:a] + seg.replace(old, new) + text[b:]


def _in_threshold_theorem(text: str, old: str, new: str) -> str:
    return _in_theorem(text, "package_valid_exact_threshold", old, new)


def _mut_delta(text: str) -> str:
    """the adversary's witness: the frozen delta commented out and redefined as 1/2 below it"""
    hits = re.findall(r"^noncomputable def δ : ℝ := .*$", text, re.M)
    if len(hits) != 1:
        raise ReceiptRefused("Package.lean does not define delta exactly once")
    return text.replace(hits[0], f"/- {hits[0]} -/\nnoncomputable def δ : ℝ := 1 / 2")


def _mut_rhs(text: str) -> str:
    """the adversary's second witness, on the theorem the statement consumes: its bound weakened to 1"""
    a, b = _segment(text, "theorem package_valid_exact_threshold")
    seg = text[a:b]
    k = seg.find("≤ ENNReal.ofReal δ := by")
    if k < 0:
        raise ReceiptRefused("package_valid_exact_threshold's conclusion is not '≤ ENNReal.ofReal δ := by ...'")
    return text[:a] + seg[:k] + "≤ 1 := prob_le_one\n" + text[b:]


PACKAGE_LEAN, PACKAGE_AD_LEAN = "BettingR203/Package.lean", "BettingR203/PackageAD.lean"
MUTANTS = {
    "delta_redefined": ("the frozen delta commented out and redefined as 1/2 (adversary witness F1)", _mut_delta),
    "rhs_weakened": ("package_valid_exact_threshold's bound weakened from ENNReal.ofReal delta to 1 (adversary "
                     "witness F2)", _mut_rhs),
    "threshold_moved": ("the null's threshold 63/1156 moved to 1, an almost empty null",
                        lambda t: _in_threshold_theorem(_in_threshold_theorem(t, "(63 / 1156)", "(1 : ℝ)"),
                                                        "m_le_exact", "(by norm_num [m] : m ≤ 1)")),
    "hypothesis_added": ("an extra hypothesis T = 0 added to package_valid_exact_threshold",
                         lambda t: _in_threshold_theorem(
                             t, "W (i + 1) ω = sCommon) :", "W (i + 1) ω = sCommon) (hT : T = 0) :")),
}


def _mut_ad_rhs(text: str) -> str:
    """R225: package_ad_valid_exact_threshold's bound weakened to 1 (F2's shape, on the mixture's theorem)"""
    a, b = _segment(text, "theorem package_ad_valid_exact_threshold")
    seg = text[a:b]
    k = seg.find("ENNReal.ofReal Package.δ := by")
    if k < 0 or seg.count("ENNReal.ofReal Package.δ := by") != 1:
        raise ReceiptRefused("package_ad_valid_exact_threshold's conclusion is not '≤ ENNReal.ofReal Package.δ := by'")
    return text[:a] + seg[:k] + "1 := prob_le_one\n" + text[b:]


def _mut_ad_score(text: str) -> str:
    """R225: the witness's kept/X score commented out and redefined as 1 (still between its neighbours, so
    PackageAD.lean compiles and only the statement can refuse it)"""
    hits = re.findall(r"^noncomputable def kKX : ℝ := .*$", text, re.M)
    if len(hits) != 1:
        raise ReceiptRefused("PackageAD.lean does not define kKX exactly once")
    return text.replace(hits[0], f"/- {hits[0]} -/\nnoncomputable def kKX : ℝ := 1")


def _ad_threshold_literal(text: str) -> str:
    hits = re.findall(r"theorem mW_le_exact : mW ≤ (\d+ / \d+) :=", text)
    if len(hits) != 1:
        raise ReceiptRefused("PackageAD.lean does not state mW_le_exact exactly once")
    return f"({hits[0]})"


MUTANTS_AD = {
    "ad_rhs_weakened": ("package_ad_valid_exact_threshold's bound weakened from ENNReal.ofReal delta to 1",
                        _mut_ad_rhs),
    "ad_threshold_moved": ("the witness's exact threshold moved to 1, an almost empty null",
                           lambda t: _in_theorem(_in_theorem(t, "package_ad_valid_exact_threshold",
                                                             _ad_threshold_literal(t), "(1 : ℝ)"),
                                                 "package_ad_valid_exact_threshold", "mW_le_exact",
                                                 "(by norm_num [mW] : mW ≤ 1)")),
    "ad_hypothesis_added": ("an extra hypothesis T = 0 added to package_ad_valid_exact_threshold",
                            lambda t: _in_theorem(t, "package_ad_valid_exact_threshold", "WW (i + 1) ω = kKZ) :",
                                                  "WW (i + 1) ω = kKZ) (hT : T = 0) :")),
    "ad_score_changed": ("the witness's kept/X score kKX redefined as 1", _mut_ad_score),
}


def mutants_for(consts: dict) -> dict:
    """{name: (what, file, edit, statements that may refuse it)}: R211's four on Package.lean (either statement,
    both import it), and under rule 'betting_ad' R225's four on PackageAD.lean (StatementAD.lean)"""
    out = {k: (what, PACKAGE_LEAN, edit, (STATEMENT, STATEMENT_AD)) for k, (what, edit) in MUTANTS.items()}
    if "witness" in consts:
        out.update({k: (what, PACKAGE_AD_LEAN, edit, (STATEMENT_AD,)) for k, (what, edit) in MUTANTS_AD.items()})
    return out


def _manifest_rev(root: pathlib.Path = LEAN) -> str:
    man = json.loads((root / "lake-manifest.json").read_text(encoding="utf-8"))
    return next(p["rev"] for p in man["packages"] if p["name"] == "mathlib")


def _toolchain(root: pathlib.Path = LEAN) -> str:
    return (root / "lean-toolchain").read_text(encoding="utf-8").strip()


def check_mutants(path: pathlib.Path = MUTANT_RECEIPT, root: pathlib.Path = LEAN,
                  package: pathlib.Path = PACKAGE) -> list[str]:
    if not path.exists():
        return ["no mutant drill: run --mutants after --build"]
    r = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    try:
        table = mutants_for(package_constants(package))
    except (ReceiptRefused, B.BettingRefused, OSError, KeyError, ValueError) as e:
        return [f"mutant drill: no package constants ({e})"]
    if r.get("base_sources") != sources(root):
        bad.append("the mutant drill ran on other sources: re-run --mutants")
    if sorted(r.get("mutants", {})) != sorted(table):
        bad.append(f"the drill's mutants {sorted(r.get('mutants', {}))} are not {sorted(table)}")
    for name, (_, file, edit, refusers) in table.items():
        got = r.get("mutants", {}).get(name)
        if not got:
            continue
        try:
            want = edit(_read_lf(root / file))
        except (ReceiptRefused, OSError) as e:
            bad.append(f"mutant {name}: {e}")
            continue
        if got.get("file") != file or got.get("source") != want:
            bad.append(f"mutant {name}: the drill built a different {file} than this edit gives")
        if got.get("build_exit") == 0:
            bad.append(f"mutant {name} BUILT: the statement did not refuse it")
        elif not str(got.get("first_error", "")).startswith(refusers):
            bad.append(f"mutant {name} failed outside the statement ({got.get('first_error')!r}): it must compile "
                       f"and be refused by {' or '.join(refusers)}")
    return bad


def statements(consts: dict) -> dict:
    """{statement file: its generated text} for the package constants: R211's always, R225's under 'betting_ad'"""
    out = {STATEMENT: statement_text(consts)}
    if "witness" in consts:
        out[STATEMENT_AD] = statement_ad_text(consts)
    return out


def check(receipt_path: pathlib.Path = RECEIPT, root: pathlib.Path = LEAN, package: pathlib.Path = PACKAGE,
          mutants_path: pathlib.Path = MUTANT_RECEIPT) -> list:
    """[] when the receipt certifies the current sources and the statement is the committed package's"""
    if not receipt_path.exists():
        return ["no receipt: run --build"]
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    bad = []
    if r.get("build_exit") != 0:
        bad.append(f"the build exited {r.get('build_exit')}")
    now = sources(root)
    for rel in sorted(set(now) | set(r.get("sources", {}))):
        if now.get(rel) != r.get("sources", {}).get(rel):
            bad.append(f"STALE: {rel} differs from the proved source")
    headline = HEADLINE
    try:
        consts = package_constants(package)
        for file, text in statements(consts).items():
            if not (root / file).exists() or _read_lf(root / file) != text:
                bad.append(f"STATEMENT: {file} is not the statement generated from the committed package "
                           "(run --write-statement, then --build)")
        if r.get("constants") != consts:
            bad.append("the receipt was built for other package constants")
        if "witness" in consts:
            headline = HEADLINE + HEADLINE_AD
    except (ReceiptRefused, B.BettingRefused, OSError, KeyError, ValueError) as e:
        bad.append(f"constants: {e}")
    want = declared(root)
    missing = [h for h in headline if h not in want]
    if missing:
        bad.append(f"AxiomCheck.lean does not name the headline theorems {missing}")
    if sorted(want) != sorted(r.get("theorems", {})):
        bad.append("the receipt's theorem list differs from AxiomCheck.lean's")
    for name, axioms in r.get("theorems", {}).items():
        if "sorryAx" in axioms:
            bad.append(f"{name} rests on sorryAx (a hole in the proof)")
        extra = sorted(set(axioms) - STANDARD - {"sorryAx"})
        if extra:
            bad.append(f"{name} rests on non-standard axioms {extra}")
    version = _toolchain(root).split(":v")[-1]
    if version not in str(r.get("lean_version", "")):
        bad.append(f"the recorded Lean {r.get('lean_version')!r} is not the toolchain's {version}")
    if r.get("mathlib_rev") != _manifest_rev(root) or r.get("mathlib_clean") is not True:
        bad.append(f"mathlib was {r.get('mathlib_rev')!r} (clean: {r.get('mathlib_clean')}), not a clean checkout "
                   f"of the manifest's {_manifest_rev(root)}")
    return bad + check_mutants(mutants_path, root, package)


def _lake(where: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["lake", *args], cwd=where, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL,
                          creationflags=BELOW_NORMAL if os.name == "nt" else 0)


def _fresh_copy(where: pathlib.Path) -> None:
    if where.exists():
        junction = where / ".lake" / "packages"
        if junction.exists():   # remove the link itself, never what it points to
            subprocess.run(["cmd", "/c", "rmdir", str(junction)], check=True, capture_output=True,
                           stdin=subprocess.DEVNULL)
        shutil.rmtree(where)
    shutil.copytree(LEAN, where, ignore=shutil.ignore_patterns(".lake"))
    if SHARED_PACKAGES.exists():   # reuse the built mathlib; a fresh machine runs `lake exe cache get` instead
        (where / ".lake").mkdir()
        subprocess.run(["cmd", "/c", "mklink", "/J", str(where / ".lake" / "packages"), str(SHARED_PACKAGES)],
                       check=True, capture_output=True, stdin=subprocess.DEVNULL)


def build(where: pathlib.Path) -> int:
    consts = package_constants()
    for file, text in statements(consts).items():
        if not (LEAN / file).exists() or _read_lf(LEAN / file) != text:
            raise ReceiptRefused(f"{file} is not generated from the committed package: run --write-statement")
    _fresh_copy(where)
    b = _lake(where, "build")
    print((b.stdout[-800:] + b.stderr[-800:]).encode("ascii", "replace").decode("ascii"))   # lake prints U+2714
    a = _lake(where, "env", "lean", "BettingR203/AxiomCheck.lean")
    v = _lake(where, "env", "lean", "--version")
    mathlib = where / ".lake" / "packages" / "mathlib"
    git = ("git", "-C", str(mathlib))
    rev = subprocess.run((*git, "rev-parse", "HEAD"), capture_output=True, text=True, stdin=subprocess.DEVNULL)
    dirty = subprocess.run((*git, "status", "--porcelain", "--untracked-files=no"), capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
    theorems = parse_axioms(a.stdout + a.stderr)
    receipt = dict(project="lean/BettingR203", build_exit=b.returncode or a.returncode, sources=sources(where),
                   theorems=theorems, constants=consts, lean_version=v.stdout.strip(),
                   mathlib_rev=rev.stdout.strip(), mathlib_clean=(rev.returncode == 0 and dirty.returncode == 0
                                                                  and dirty.stdout.strip() == ""),
                   axiom_check_lines=len(theorems))
    RECEIPT.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"receipt: {len(theorems)} theorems, build exit {receipt['build_exit']} -> {RECEIPT}")
    bad = [f for f in check() if not f.startswith(("no mutant drill", "the mutant drill", "the drill's mutants",
                                                    "mutant "))]
    for f in bad:
        print("REFUSED:", f)
    return 0 if not bad else 1


def mutants(where: pathlib.Path) -> int:
    """each mutant edit applied to its file in the BUILT copy: the package file must still compile and a statement
    must refuse it. Each file is restored after its mutant, and the copy is rebuilt at the end."""
    table = mutants_for(package_constants())
    base = {f: _read_lf(LEAN / f) for f in {file for _, file, _, _ in table.values()}}
    for f, text in base.items():
        if not (where / f).exists() or _read_lf(where / f) != text:
            raise ReceiptRefused(f"{where} is not a build of the current sources: run --build first")
    out = {}
    try:
        for name, (what, file, edit, _) in table.items():
            text = edit(base[file])
            (where / file).write_text(text, encoding="utf-8", newline="\n")
            b = _lake(where, "build")
            (where / file).write_text(base[file], encoding="utf-8", newline="\n")
            log = b.stdout + b.stderr
            err = re.search(r"error: (\S+?\.lean):(\d+):(\d+): ([^\n]*)", log)
            out[name] = dict(what=what, file=file, source=text, build_exit=b.returncode,
                             first_error=(f"{err.group(1)}:{err.group(2)}:{err.group(3)}: {err.group(4)}" if err else ""))
            print(f"{name}: exit {b.returncode}; {out[name]['first_error'] or 'NO ERROR'}")
    finally:
        for f, text in base.items():
            (where / f).write_text(text, encoding="utf-8", newline="\n")
        restored = _lake(where, "build")
        print(f"restored: build exit {restored.returncode}")
    MUTANT_RECEIPT.write_text(json.dumps(dict(base_sources=sources(LEAN), mutants=out), indent=1, ensure_ascii=False)
                              + "\n", encoding="utf-8")
    bad = check_mutants()
    for f in bad:
        print("REFUSED:", f)
    return 0 if not bad else 1


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):   # Lean's messages carry Unicode; a cp1252 console must not crash the run
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", type=pathlib.Path, help="a SHORT path to build a copy of lean/BettingR203 in")
    ap.add_argument("--mutants", type=pathlib.Path, help="the built copy to run the mutant drill in")
    ap.add_argument("--write-statement", action="store_true", help="regenerate Statement.lean from the package")
    args = ap.parse_args(argv)
    if args.write_statement:
        for file, text in statements(package_constants()).items():
            (LEAN / file).write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {LEAN / file}")
        return 0
    if args.build:
        return build(args.build)
    if args.mutants:
        return mutants(args.mutants)
    bad = check()
    for f in bad:
        print("REFUSED:", f)
    if not bad:
        r = json.loads(RECEIPT.read_text(encoding="utf-8"))
        n_mut = len(mutants_for(package_constants()))
        print(f"PASS: {len(r['theorems'])} theorems, standard axioms only, the statements are the committed package's, "
              f"{n_mut} mutants refused by them")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
