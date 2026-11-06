r"""Every quantitative macro and table in paper 4 (superactivation in two channel uses), generated from evidence.

Same contract as make_numbers_distance.py: the manuscript never types a result. It writes \TgFifteenF, and this
script defines \TgFifteenF by reading the committed artifact that established it. It REFUSES, with the reason, when
an input is missing, when two inputs disagree, or when the evidence no longer supports a sentence the text makes.

    python paper/tools/make_numbers_superactivation.py

Writes paper/superactivation/generated/{numbers,tab_claims,tab_dtwo,tab_onset}.tex.

INPUTS (all committed):
  evidence/superactivation/general_codes/certificate_twogroup_a{7_b8,8_b8}.json   d = 2, n = 15 and 16 (and the
        witness and decoder bytes they name); twogroup_n14_exact.log, twogroup_grow_15_16.log (the n = 14 search)
  evidence/superactivation/certificate_n{17,20,24}.json, witness_n*.npz, curve_n17_n24.json, encoder_n*_seesaw.npy
  evidence/superactivation/erasure_sensitivity.json, stored_maps_entrywise.json, ppt_relaxation_probe.json
  evidence/superactivation/direct/certificate_direct_n1.json (+ encoder and decoders)
  evidence/superactivation/shield_family/certificate_two_use_*.json, two_use_code_*.npy, rational_recovery_d*.json,
        certificate_nuse_d*_n*.json (+ codes and decoders), phase_diagram.json
  evidence/superactivation/general_codes/reduction_dense_check_v2.json   the group reduction, the certifiers' own
        ball-arithmetic selections and coefficients, and the certifiers end to end, against the dense 2^n
        construction (verify_reduction_dense.py; findings F1, N6, N7), bound to the bytes of the code that ran it;
        it supersedes reduction_dense_check.json, which is kept and no longer read
  evidence/superactivation/shield_family/p2_kraus_comparison_v2.json      the d = 2 member against Smith and Yard's
        channel as arXiv:2604.27042 states it, Kraus operators AND Choi-state formula, in floating point and exactly
        (findings F10, N3); supersedes p2_kraus_comparison.json
  evidence/superactivation/ppt_relaxation_fullspace.json   Proposition 11 on the FULL output space, exactly: the
        feasible point and the dual certificate, n <= 3 at six p, and the joint channel P_2 (x) A_4 at n = 1 in
        Q(sqrt 2) (findings N2, W1, W2); it supersedes ppt_relaxation_exact.json, which is kept and no longer read
  evidence/superactivation/hw_run_m5/{frozen_20261015T195926Z,result_db8jpre94fic73etigpg}.json   (the outlook)
  evidence/superactivation/ion_run_m6/design_n2000/{design,pass_rates}.json, ion_run_m8_r224/{design,sequential}.json
  tools/build_certificate_bundle.py   its CONTENTS decide which release carries this lane (\LaneRelease)
The literature's values are read by strict patterns: the published onset and the n = 17 fidelity from the literature
record's account of arXiv 2604.27042 and its ancillary data (docs/PAPER4_LITERATURE_SEARCH.md, which ships with the
lane), and the authors' public symmetric table from the copy the
two-group lane compares against (experiments/superactivation/general_codes/twogroup.py). A pattern that stops matching
is a refusal: the sentence changed, so its numbers must be read again. Each macro in numbers.tex is preceded by a
comment naming the file it was read from.

RE-DERIVED HERE, in seconds, never trusted from a record:
  * every n-use certificate (nuse_certificate.verify: re-hashes the code and decoder bytes and re-proves the bound
    exactly in Q(sqrt d); the record must agree to the last digit);
  * the n = 2 bounds at d = 20, 21, 22 by the producer-free exact-rational checker
    (tools/verify_r142_rational_witness.py), which must agree with the ball-arithmetic records;
  * the closed form F = (7 - 6p + 2p^2)/8 at p = 1/(sqrt d + 1), its threshold (3 - sqrt 7)/2 and the least integer d
    above it; family A's F_A* and its threshold (49 - sqrt 97)/2; the one-use optimum 1 - sqrt 2/4 of N~ at d = 2;
    each checked against the certificates that claim them.
The two-group and symmetric certificates (hours of exact arithmetic) are NOT re-run: each is bound by sha256 to the
witness and decoder bytes it proved, and tests/test_superactivation_twogroup.py and test_superactivation_certificate.py
recompute the stored pairs.

ROUNDING. A certified lower bound is printed TRUNCATED (it may only fall), a maximum residual is printed rounded UP,
and a floating-point (uncertified) value is rounded half up and refused if the rounding moves it across 3/4.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal, getcontext
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence" / "superactivation"
SF = EV / "shield_family"
GC = EV / "general_codes"
OUT = ROOT / "paper" / "superactivation" / "generated"
# The published onset and its fidelity are read from the literature record that ships with the lane. They were read
# from docs/RESEARCH_LEDGER.md R132 until 2026-11-03, which no release carries: the private research ledger of every
# lane, which a reader regenerating this paper from the release could not have held.
LITERATURE = "docs/PAPER4_LITERATURE_SEARCH.md"
TWOGROUP_PY = "experiments/superactivation/general_codes/twogroup.py"
getcontext().prec = 80
TQ = Decimal(3) / 4                                   # the threshold of Result 1 (epsilon = 1/4)


# ------------------------------------------------------------------ reading and refusing

def refuse(why: str):
    raise SystemExit(f"REFUSED: {why}")


def need(ok, why: str) -> None:
    if not ok:
        refuse(why)


def rel(p: pathlib.Path) -> str:
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def J(path: pathlib.Path) -> dict:
    need(path.exists(), f"{rel(path)} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def TXT(relpath: str) -> str:
    p = ROOT / relpath
    need(p.exists(), f"{relpath} is missing")
    return p.read_text(encoding="utf-8")


def RX(pattern: str, text: str, what: str) -> re.Match:
    m = re.search(pattern, text, re.S)
    need(m is not None, f"{what}: the source sentence changed shape; read its numbers again")
    return m


def sha256(p: pathlib.Path) -> str:
    need(p.exists(), f"{rel(p)} is missing")
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ the release that carries this lane

#: What "carries paper 4's lane" means: every directory of the lane, this manuscript and this generator.
LANE_ROOTS = ("experiments/superactivation/", "evidence/superactivation/")
LANE_FILES = ("paper/superactivation/main.tex", "paper/tools/make_numbers_superactivation.py")


def release_builder_contents() -> list[tuple[str, str]]:
    """What the release builder declares paper 4's bundle (oneq-superactivation) ships, read by importing it.

    The superactivation bundle, not the decoder bundle: since 2026-11-03 paper 4 travels in its own signed bundle in
    the same release tag, and the decoder bundle carries none of it."""
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import build_certificate_bundle as bcb  # noqa: E402
    return list(bcb.bundle_spec("superactivation")["contents"])


def lane_release(contents) -> tuple[str, str]:
    r"""(\LaneRelease's body, its source comment), DERIVED from what the release builder declares.

    The tag is never typed here. When the builder ships the whole lane, the release that carries it is the release
    being minted, which paper/release.tex names (\ReleaseTag, set at mint); when it ships none of it, no release
    carries it and the macro is [pending], which `build.py --release` refuses. A builder that ships PART of the lane is
    a refusal: a release that carries half the evidence carries none of the claims.

    Until 2026-11-03 this macro was typed as [pending] behind a guard that fired when a minted copy appeared under
    release/, so the tag would have had to be typed in by hand at the next mint.
    """
    shipped = [rel_ for rel_, _ in contents]
    roots = {r: any(s.startswith(r) for s in shipped) for r in LANE_ROOTS}
    files = {f: f in shipped for f in LANE_FILES}
    have = {**roots, **files}
    if all(have.values()):
        return r"\ReleaseTag", ("tools/build_certificate_bundle.py: the oneq-superactivation bundle declares the whole lane (" + ", ".join(LANE_ROOTS)
                                + "), so it is in the release paper/release.tex names")
    if not any(have.values()):
        return r"\textbf{[pending]}", "tools/build_certificate_bundle.py: the oneq-superactivation bundle declares no part of the lane"
    refuse(f"the release builder declares part of paper 4's lane and not the rest: "
           f"{sorted(k for k, v in have.items() if not v)} missing")


def bound_to(record: dict, folder: pathlib.Path, pairs, what: str) -> None:
    """every (file field, hash field) the record names hashes to what the record says it proved"""
    for fkey, hkey in pairs:
        need(sha256(folder / record[fkey]) == record[hkey],
             f"{what}: {record[fkey]} does not hash to the {hkey} its certificate proved")


# ------------------------------------------------------------------ exact numbers and their printing

def D(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    if isinstance(x, float):
        return Decimal(repr(x))
    return Decimal(str(x))


def fixed(x, k: int, mode) -> str:
    return str(D(x).quantize(Decimal(1).scaleb(-k), rounding=mode))


def lo(x, k: int) -> str:
    """a lower bound, truncated: the printed digits never claim more than was proved"""
    return fixed(x, k, ROUND_FLOOR)


def hi(x, k: int) -> str:
    return fixed(x, k, ROUND_CEILING)


def rd(x, k: int) -> str:
    return fixed(x, k, ROUND_HALF_UP)


def measured(x, k: int, what: str) -> str:
    """an uncertified floating-point value, rounded; refused if the rounding would move it across 3/4"""
    s = rd(x, k)
    need(D(s) != TQ and (D(s) > TQ) == (D(x) > TQ), f"{what}: {x} rounds to {s}, across or onto 3/4")
    return s


def sci(x, k: int, mode) -> str:
    x = D(x)
    need(x > 0, f"scientific notation of a non-positive value {x}")
    e = x.adjusted()
    m = x.scaleb(-e).quantize(Decimal(1).scaleb(-k), rounding=mode)
    if m >= 10:
        m, e = (m / 10).quantize(Decimal(1).scaleb(-k), rounding=mode), e + 1
    return rf"\ensuremath{{{m}\times 10^{{{e}}}}}"


def ball_lower(s: str, what: str) -> Decimal:
    """the lower end of a python-flint ball '[mid +/- rad]'"""
    m = re.fullmatch(r"\[(\d+\.\d+) \+/- (\d+(?:\.\d+)?e-\d+)\]", s)
    need(m is not None, f"{what}: {s!r} is not a ball")
    return Decimal(m.group(1)) - Decimal(m.group(2))


def p_of(d: int) -> Decimal:
    """the family's flip probability, p = 1/(sqrt d + 1)"""
    return 1 / (Decimal(d).sqrt() + 1)


def closed_form(p: Decimal) -> Decimal:
    return (7 - 6 * p + 2 * p * p) / 8


def family_a(d: int) -> Decimal:
    p = p_of(d)
    return (9 - 2 * p + (Decimal(d).sqrt() + (3 - 2 * p) ** 2).sqrt()) / 16


def hours(seconds: float) -> str:
    return rd(Decimal(repr(seconds)) / 3600, 1)


def grouped(n: int) -> str:
    """an exact count with LaTeX's thousands separator"""
    return f"{int(n):,}".replace(",", "{,}")


def lane_modules():
    for p in (ROOT / "src", ROOT / "experiments" / "superactivation" / "shield_family"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import nuse_certificate as nc  # noqa: E402  (the exact n-use certifier, re-run on every record)
    spec = importlib.util.spec_from_file_location("r142_rational", ROOT / "tools" / "verify_r142_rational_witness.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    return nc, rv


# ------------------------------------------------------------------ the generator

def build() -> dict[str, str]:
    M: dict[str, tuple[str, str]] = {}

    def put(name: str, value, source: str) -> None:
        need(re.fullmatch(r"[A-Za-z]+", name) is not None, f"macro name {name} must be letters only")
        need(name not in M, f"macro {name} defined twice")
        M[name] = (str(value), source)

    nc, rv = lane_modules()
    lit = TXT(LITERATURE)

    # -------------------------------------------- the literature: Parentin, Bergh, Datta, Wilde (arXiv 2604.27042)
    pub = RX(r"arXiv:2604\.27042 \(Parentin, Bergh, Datta, Wilde; v2 2026-05-04\): n\* <= (\d+) for the Smith",
             lit, "the literature record's account of the published onset")
    anc = RX(r"the ancillary `fidelity_verified\.npy` holds\s+(0\.\d+)", lit,
             "the literature record's reading of the published code's ancillary fidelity")
    pub_n, pub_f = int(pub.group(1)), D(anc.group(1))
    need(pub_f > TQ and D(rd(pub_f, 6)) > TQ, "the published code is recorded below 3/4, or its rounding is")
    put("PubN", pub_n, f"{LITERATURE}: arXiv 2604.27042's n* <= {pub_n}")
    put("PubF", rd(pub_f, 6), f"{LITERATURE}: the published n = {pub_n} code's fidelity in arXiv 2604.27042's "
                              f"ancillary data (fidelity_verified.npy, {anc.group(1)}), rounded half up")
    # the paper's text prints it to five digits, "0.75013 > 3/4" (onset_superactivation.tex l.159, l.171;
    # supplemental.tex l.2351 of the v2 source); the macro is that rounding, held to the ancillary value
    put("PubFPrinted", rd(pub_f, 5), "arXiv 2604.27042v2 text, three places: the five-digit rounding of PubF")
    tg_src = TXT(TWOGROUP_PY)
    sym_tab = RX(r"SYM = \{(.*?)\}\n# \^ the authors' symmetric \(isometric-ansatz\) table, permqit ([0-9a-f]{10})",
                 tg_src, "the authors' symmetric table in twogroup.py")
    auth = {int(k): D(v) for k, v in re.findall(r"(\d+): (0\.\d+)", sym_tab.group(1))}
    need(set(range(14, 18)) <= set(auth), "the authors' symmetric table no longer covers n = 14..17")
    need(all(auth[n] < TQ for n in auth if n < pub_n) and auth[pub_n] > TQ,
         "the authors' symmetric table does not first cross 3/4 at the published n")
    put("PermqitCommit", r"\texttt{" + sym_tab.group(2) + "}", f"{TWOGROUP_PY}: the SYM table's provenance comment")

    # -------------------------------------------- certificate machinery constants (every certificate shares them)
    sym = {n: J(EV / f"certificate_n{n}.json") for n in (17, 20, 24)}
    tg = {ab: J(GC / f"certificate_twogroup_{ab}.json") for ab in ("a7_b8", "a8_b8")}
    for what, c in [*(("n=%d" % n, c) for n, c in sym.items()), *tg.items()]:
        need(c["grid_bits"] == 96 and c["precision_bits"] == 256 and c["eta"] == 2.0 ** -34,
             f"certificate {what}: grid, precision or eta differ from the method the text describes")
    put("GridBits", 96, "certificate_n17.json, _n20, _n24, certificate_twogroup_*.json: grid_bits")
    put("PrecBits", 256, "the same certificates: precision_bits")
    put("EtaExp", 34, "the same certificates: eta = 2^-34")

    # -------------------------------------------- d = 2: the two-group codes at n = 15 and 16 (R140)
    tgr = {}
    for ab, c in tg.items():
        bound_to(c, GC, (("witness", "witness_sha256"), ("decoders", "decoders_sha256")), f"two-group {ab}")
        low = ball_lower(c["certified_lower_bound"], f"two-group {ab}")
        need(c["n"] == c["a"] + c["b"], f"two-group {ab}: n is not a + b")
        need(c["exceeds_three_quarters"] is True and low > TQ, f"two-group {ab}: the certificate no longer exceeds 3/4")
        need(0 <= D(c["pair_float_fidelity"]) - low < D("1e-8"),
             f"two-group {ab}: the bound is not just below its own pair (a bound above it certifies another pair)")
        tgr[c["n"]] = (c, low)
    need(sorted(tgr) == [15, 16], "the two-group certificates are not at n = 15 and 16")
    # the text: "so the crossing holds at two consecutive n and the margin grows" (referee mutation test, 2026-11-03)
    need(tgr[16][1] - TQ > tgr[15][1] - TQ > 0,
         "the n = 16 certified margin no longer exceeds the n = 15 one: the text says the margin grows")
    for n, word in ((15, "Fifteen"), (16, "Sixteen")):
        c, low = tgr[n]
        src = f"general_codes/certificate_twogroup_a{c['a']}_b{c['b']}.json"
        put(f"Tg{word}N", n, src)
        put(f"Tg{word}A", c["a"], src)
        put(f"Tg{word}B", c["b"], src)
        put(f"Tg{word}F", lo(low, 10), src + ": certified_lower_bound (lower end of the ball, truncated)")
        put(f"Tg{word}Margin", sci(low - TQ, 1, ROUND_FLOOR), src + ": certified bound minus 3/4, truncated")
        put(f"Tg{word}Blocks", grouped(c["blocks_proved_positive_definite"]), src)
        put(f"Tg{word}Hours", hours(c["seconds"]), src + ": seconds")
    need(tgr[15][0]["n"] < pub_n, "the certified two-group code is not below the published n")
    n14 = RX(r"EXACT \(a,b\)=\((\d+),(\d+)\) n=(\d+): F = (0\.\d+) \| symmetric (0\.\d+) \| gain \+[0-9.]+ \| below 3/4",
             (GC / "twogroup_n14_exact.log").read_text(encoding="utf-8"), "twogroup_n14_exact.log")
    need(int(n14.group(3)) == 14 and D(n14.group(4)) < TQ, "the n = 14 two-group search is not the recorded miss")
    put("TgFourteenN", 14, "general_codes/twogroup_n14_exact.log")
    put("TgFourteenA", n14.group(1), "general_codes/twogroup_n14_exact.log")
    put("TgFourteenB", n14.group(2), "general_codes/twogroup_n14_exact.log")
    put("TgFourteenF", measured(n14.group(4), 7, "n = 14"), "general_codes/twogroup_n14_exact.log (float)")
    grow = (GC / "twogroup_grow_15_16.log").read_text(encoding="utf-8")
    for n, (c, low) in tgr.items():
        g = RX(rf"GROWN \(a,b\)=\({c['a']},{c['b']}\) n={n}: F = (0\.\d+) \| symmetric (0\.\d+)", grow,
               f"twogroup_grow_15_16.log n = {n}")
        need(abs(D(g.group(2)) - auth[n]) < D("1e-9"), f"the n = {n} search compared against another symmetric value")

    # -------------------------------------------- d = 2: the symmetric curve n = 17..24 and its certificates (R132, R135)
    curve = J(EV / "curve_n17_n24.json")
    cn = [r["n"] for r in curve]
    need(cn == list(range(17, 25)), "the curve is not n = 17..24")
    need(all(curve[i + 1]["F_exact_decoders"] > curve[i]["F_exact_decoders"] for i in range(len(curve) - 1)),
         "the curve no longer rises with n")
    for r in curve:
        need(sha256(EV / r["encoder"]) == r["encoder_sha256"], f"curve n = {r['n']}: the encoder bytes changed")
        need(r["F_exact_decoders"] > 0.75, f"curve n = {r['n']} is below 3/4")
    curve_f = {r["n"]: r["F_exact_decoders"] for r in curve}
    symlow = {}
    for n, c in sym.items():
        need(sha256(EV / c["witness"]) == c["witness_sha256"], f"certificate_n{n}: the witness bytes changed")
        low = ball_lower(c["certified_lower_bound"], f"certificate_n{n}")
        need(c["exceeds_three_quarters"] is True and low > TQ, f"certificate_n{n} no longer exceeds 3/4")
        need(c["float_fidelity"] == curve_f[n], f"certificate_n{n} certifies another code than the curve's")
        need(0 <= D(c["float_fidelity"]) - low < D("1e-8"), f"certificate_n{n}: the bound is not just below its code")
        need(any("Prop. 1" in t for t in c["imported_theorems"]) and any("Result 1" in t for t in c["imported_theorems"]),
             f"certificate_n{n} no longer records the two imported theorems")
        symlow[n] = low
    need(symlow[24] - TQ > D("1e-3"), "the n = 24 margin is no longer above 10^-3, as the text says")
    put("SymSeventeenF", lo(symlow[17], 10), "certificate_n17.json")
    put("SymTwentyF", lo(symlow[20], 10), "certificate_n20.json")
    put("CurveLo", cn[0], "curve_n17_n24.json")
    put("CurveHi", cn[-1], "curve_n17_n24.json")
    put("SymMidN", 20, "certificate_n20.json")
    need(sorted(sym) == [cn[0], 20, cn[-1]], "the symmetric certificates are not at the curve's ends and n = 20")
    put("SymCertN", 24, "certificate_n24.json")
    put("SymCertF", lo(symlow[24], 10), "certificate_n24.json")
    put("SymCertMargin", sci(symlow[24] - TQ, 1, ROUND_FLOOR), "certificate_n24.json, truncated")
    put("SymCertBlocks", grouped(sym[24]["blocks_proved_positive_definite"]), "certificate_n24.json")
    need(curve_f[17] > float(auth[17]), "our n = 17 code no longer exceeds the authors' public symmetric value")
    # Table II's caption: the authors' table's n = 17 value is a later code of theirs, better than the published one
    need(auth[pub_n] - pub_f > 0, "the authors' later n = 17 value is no longer above the published code's")
    put("AuthLaterGain", sci(auth[pub_n] - pub_f, 1, ROUND_HALF_UP),
        "twogroup.py's permqit table at n = 17 minus PubF: the later code's gain over the published one")

    # -------------------------------------------- the group reduction, checked against the dense 2^n construction (F1)
    rdc = J(GC / "reduction_dense_check_v2.json")
    need(rdc["schema"] == "oneq.superactivation.reduction-dense-check/2" and rdc["verdict"] == "AGREE",
         "reduction_dense_check_v2.json does not record agreement")
    for f, blob in rdc["provenance"]["code_git_blobs"].items():
        raw = (ROOT / f).read_bytes()
        need(hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob,
             f"reduction_dense_check_v2.json was produced by other bytes of {f} than the tree holds; re-run it")
    cases = rdc["cases"]
    allp = [c for c in cases if c.get("flag_patterns_checked") == 2 ** c["n"]]
    need(len(allp) == 1 and len(allp[0]["groups"]) == 2 and allp[0]["groups"][0] != allp[0]["groups"][1],
         "the dense check has no every-pattern case with unequal groups")
    for ab, (c, _) in ((f"a{tgr[n][0]['a']}_b{tgr[n][0]['b']}", tgr[n]) for n in (15, 16)):
        hit = [x for x in cases if x.get("witness_sha256") == c["witness_sha256"]]
        need(len(hit) == 1 and hit[0]["decoders_sha256"] == c["decoders_sha256"]
             and hit[0]["fidelity_recorded_pair"] == c["pair_float_fidelity"],
             f"the dense check did not run on the certified two-group witness {ab} and its decoders")
    need(any(x.get("witness_sha256") == sym[17]["witness_sha256"] and "fidelity_abs_difference" in x for x in cases),
         "the dense check did not run on the certified symmetric n = 17 witness")
    # the symmetric certificates past the dense embedding: slices counted, states not formed -- only there
    counted = [x for x in cases if x.get("states_and_fidelity", "").startswith("not formed")]
    need(sorted(x["n"] for x in counted) == [20, 24]
         and all(x["witness_sha256"] == sym[x["n"]]["witness_sha256"] for x in counted)
         and all(x["flagged_strings_checked"] == 2 * (2 ** (x["n"] + 1) - 1) for x in counted),
         "the dense check did not count every flagged string, at both placements, for the n = 20 and 24 certificates")
    formed = [x for x in cases if x not in counted]
    need(all("max_state_deviation" in x and "fidelity_abs_difference" in x for x in formed),
         "a dense-check case outside n = 20, 24 skipped the branch states or the fidelity")
    devs = [x[k] for x in cases for k in ("max_identity_deviation", "max_certifier_selection_deviation",
                                          "max_residual_outside_symmetric", "max_state_deviation",
                                          "max_placement_dependence") if k in x]
    fdiffs = [x["fidelity_abs_difference"] for x in formed]
    need(all("max_certifier_selection_deviation" in x for x in cases),
         "a dense-check case did not compare the certifier's own selection matrices")
    need(max(devs) < 1e-12 and max(fdiffs) < 1e-12, "the dense check's deviations exceed its tolerance")
    coef = rdc["certifier_coefficients"]
    need(coef["max_deviation"] < 1e-12 and coef["group_sizes_up_to"] >= max(max(x["groups"]) for x in cases)
         and coef["group_sizes_up_to"] >= max(sym) and coef.get("radius_scale") == "per string, as the deviation",
         "the certifier's coefficients were not checked at every group size, or disagree with the dense law")
    radius = max([x["max_certifier_selection_radius"] for x in cases] + [coef["max_radius"]])
    need(radius < 1e-70, "a certifier ball is wider than the check assumes")
    e2e = rdc["certifier_end_to_end"]
    e2e_ns = sorted(e["n"] for e in e2e)

    def e2e_gap(e):                     # dense fidelity where the states are formed, the recorded one at n = 20, 24
        return e["recorded_minus_bound"] if e["n"] in (20, 24) else e["dense_minus_bound"]
    need(len(e2e) == 5 and all(e["reproduces_committed_bound"] and 0 <= e2e_gap(e) < 1e-7 for e in e2e),
         "a certifier, re-run, did not reproduce its committed bound, or its bound is not just below its pair")
    for e in e2e:
        committed = J(EV / e["certificate"])["certified_lower_bound"]
        need(e["certified_committed"] == committed == e["certified_now"],
             f"{e['certificate']}: the end-to-end record does not match the committed certificate")
    ctl = rdc["negative_controls"]
    ctl_keys = ("identity_comparator_expectation_perturbed_identity_deviation",
                "state_producing_module_perturbed_state_deviation",
                "dense_flip_probability_perturbed_state_deviation",
                "certifier_selection_midpoint_perturbed_deviation")
    need(ctl["relative_size"] == 1e-6 and all(ctl[k] > 1e3 * max(devs) for k in ctl_keys),
         "the dense check's negative controls are not caught far above its noise")
    src = "general_codes/reduction_dense_check_v2.json"
    put("RedN", allp[0]["n"], src)
    put("RedA", allp[0]["groups"][0], src)
    put("RedB", allp[0]["groups"][1], src)
    put("RedPatterns", grouped(allp[0]["flag_patterns_checked"]), src)
    put("RedDevMax", sci(max(devs), 1, ROUND_CEILING), src + ": largest deviation over every case, rounded up")
    put("RedFidMax", sci(max(fdiffs), 1, ROUND_CEILING), src + ": largest fidelity difference, rounded up")
    put("RedCoeffDev", sci(coef["max_deviation"], 1, ROUND_CEILING), src + ": certifier coefficients, per string")
    put("RedRadius", sci(radius, 1, ROUND_CEILING), src + ": largest certifier ball radius, rounded up")
    need(e2e_ns == [4, 8, 17, 20, 24], "the end-to-end runs are not at n = 4, 8 (two-group) and 17, 20, 24")
    put("RedEtoEFour", e2e_ns[0], src + ": certifier_end_to_end")
    put("RedEtoEEight", e2e_ns[1], src + ": certifier_end_to_end")
    put("RedEtoEMax", sci(max(e2e_gap(e) for e in e2e), 1, ROUND_CEILING),
        src + ": largest fidelity of the stored pair minus its re-certified bound, rounded up")
    put("RedCtlExpect", sci(ctl[ctl_keys[0]], 1, ROUND_FLOOR), src + ": negative controls")
    put("RedCtlModule", sci(ctl[ctl_keys[1]], 1, ROUND_FLOOR), src + ": negative controls")
    put("RedCtlFlip", sci(ctl[ctl_keys[2]], 1, ROUND_FLOOR), src + ": negative controls")
    put("RedCtlCert", sci(ctl[ctl_keys[3]], 1, ROUND_FLOOR), src + ": negative controls")
    put("RedMinutes", rd(D(rdc["seconds_total"]) / 60, 0), src + ": seconds_total")
    # the script's usage line quotes a runtime; it must be the runtime the committed record measured (referee nit)
    dense_doc = (ROOT / "experiments/superactivation/general_codes/verify_reduction_dense.py").read_text(encoding="utf-8")
    quoted = re.findall(r"measured: (\d+) to (\d+) min, peak about (\d+(?:\.\d+)?) GB", dense_doc)
    need(len(quoted) == 1, "verify_reduction_dense.py's usage line no longer quotes its measured runtime and memory")
    t_lo, t_hi, gb = int(quoted[0][0]), int(quoted[0][1]), float(quoted[0][2])
    need(t_lo - 0.5 <= rdc["seconds_total"] / 60 <= t_hi + 0.5,
         f"verify_reduction_dense.py quotes {t_lo} to {t_hi} min but its committed record took {rdc['seconds_total']} s")
    need(isinstance(rdc.get("peak_memory_mb"), int) and abs(rdc["peak_memory_mb"] / 1024 - gb) <= 0.25,
         f"verify_reduction_dense.py quotes a peak of about {gb} GB; the record says {rdc.get('peak_memory_mb')} MiB")

    # -------------------------------------------- R136: the published maps entry by entry, and the erasure tolerance
    sm = J(EV / "stored_maps_entrywise.json")
    need(sm["blocks_compared"] == 330 and sm["best_convention"]["max_abs_residual"] < sm["tolerance"],
         "stored_maps_entrywise.json: the published maps no longer agree with the composition")
    others = sm["other_conventions_max_abs_residual"]
    need(min(others) > 1e-3, "stored_maps_entrywise.json: another basis order also agrees, so the check is dark")
    put("StoredBlocks", sm["blocks_compared"], "stored_maps_entrywise.json")
    put("StoredResid", sci(sm["best_convention"]["max_abs_residual"], 1, ROUND_CEILING),
        "stored_maps_entrywise.json: max_abs_residual, rounded up")
    put("StoredOther", sci(min(others), 1, ROUND_FLOOR), "stored_maps_entrywise.json: the best other convention")
    er = J(EV / "erasure_sensitivity.json")["codes"]
    need(sorted(er) == ["17", "20", "24"], "erasure_sensitivity.json covers other codes")
    slopes = []
    for n, word in (("17", "Seventeen"), ("20", "Twenty"), ("24", "TwentyFour")):
        e = er[n]
        need(e["F_at_half"] == curve_f[int(n)], f"erasure n = {n}: not the certified code")
        need(0.5 < e["q_where_F_reaches_three_quarters"] < 0.51, f"erasure n = {n}: the tolerance is not above 1/2")
        put(f"ErasQ{word}", lo(e["q_where_F_reaches_three_quarters"], 6),
            f"erasure_sensitivity.json codes[{n}], truncated")
        slopes.append(-e["dF_dq_at_half"])
    need(all(0.45 < s < 0.55 for s in slopes), "the erasure slopes are no longer about -1/2")
    put("ErasSlopeLo", lo(min(slopes), 3), "erasure_sensitivity.json: |dF/dq| at q = 1/2, range enclosed")
    put("ErasSlopeHi", hi(max(slopes), 3), "erasure_sensitivity.json: |dF/dq| at q = 1/2, range enclosed")

    # -------------------------------------------- R137 and Proposition 11: the PPT-assisted relaxation's value
    pp = J(EV / "ppt_relaxation_probe.json")
    need(abs(pp["sanity"]["identity"] - 1) < 1e-6 and abs(pp["sanity"]["entanglement_breaking"] - 0.5) < 1e-6,
         "ppt_relaxation_probe.json: the sanity cases fail")
    probe_rows = {r["n"]: r for r in pp["rows"]}
    # the full output space, exactly (adversary finding W1): the feasible point AND the dual certificate, no blocks
    psrc = "ppt_relaxation_fullspace.json"
    px = J(EV / psrc)
    need(px["schema"] == "oneq.superactivation.ppt-relaxation-fullspace/1" and px["verdict"] == "EXACT"
         and len(px["calibration"]) >= 7 and all(px["calibration"].values()),
         f"{psrc}: not exact, or its PSD tester is not calibrated")
    for f, blob in px["provenance"]["code_git_blobs"].items():
        raw = (ROOT / f).read_bytes()
        need(hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob,
             f"{psrc} was produced by other bytes of {f}; re-run it")
    need(sorted((r["n"], r["channel"].split()[0]) for r in px["rows"])
         == [(n, c) for n in (1, 2, 3) for c in ("50%", "flagged")],
         f"{psrc}: the rows are not n = 1, 2, 3 for both channels")
    n_ps = set()
    for r in px["rows"]:
        want = str(Fraction(1) - Fraction(1, 2 ** (r["n"] + 1)))            # Proposition 11's value
        pr, du = r["primal"], r["dual_at_p"]
        per_use = 8 if r["channel"].startswith("flagged") else 6            # (input 2) x (output 4 or 3: e is 1-dim)
        need(r["block_decomposition_assumed"] is False and pr["full_dimension"] == per_use ** r["n"]
             and pr["feasible"] and pr["trace_rho"] == "1" and set(pr["value_at_p"].values()) == {want}
             and set(du) == set(pr["value_at_p"])
             and all(d["valid"] and d["t"] == want and not d["J_couples_all_erased_sector_to_another"]
                     for d in du.values()),
             f"{psrc} n = {r['n']} ({r['channel']}): the feasible point and the dual do not both give {want} at every p")
        need(all(0 <= Fraction(p) <= 1 for p in du), f"{psrc}: a p outside [0, 1]")
        n_ps.add(len(du))
    need(n_ps == {6}, f"{psrc}: not six values of p for every row")
    jt = px["joint"]
    need(jt["n"] == 1 and jt["block_decomposition_assumed"] is False and jt["primal"]["full_dimension"] == 320
         and jt["primal"]["feasible"] and set(jt["primal"]["value_at_p"].values()) == {"3/4"}
         and jt["dual"]["valid"] and jt["dual"]["t"] == "3/4" and jt["field"] == "Q(sqrt 2)"
         and jt["P_2"]["max_abs_difference_vs_ppt_family_choi_state"] < 1e-14
         and all(v is True for k, v in jt["P_2"].items() if k != "max_abs_difference_vs_ppt_family_choi_state"),
         f"{psrc}: the joint channel P_2 (x) A_4 is not exactly 3/4 at one use, or P_2 is not the family's member")
    ctl = px["controls"]
    names = " | ".join(c["control"] for c in ctl)
    need(len(ctl) >= 13 and not any(c["passed_but_must_fail"] for c in ctl)
         and all(s in names for s in ("kept block without its 1/2", "all-erased block without its 1/2",
                                      "kept block scaled 1/2 + 1/64", "all-erased block scaled 1/2 + 1/64",
                                      "dual with t - 1/64", "erased branch is the identity, not PPT",
                                      "joint dual with P_2 at p1 = 1/2, not PPT")),
         f"{psrc}: a control passed, or a required control is missing")
    pdevs = []
    for r in px["rows"]:
        if r["n"] in (1, 2):
            col = "F_PPT_flagged" if r["channel"].startswith("flagged") else "F_PPT_erasure"
            need(r["probe_measured"] == probe_rows[r["n"]][col], "the exact check compared another probe record")
            pdevs.append(abs(r["probe_minus_exact"]))
    need(len(pdevs) == 4 and max(pdevs) < 1e-6, "the probe's measured optimum does not agree with Proposition 11")
    val = {r["n"]: r["primal"]["value_at_p"]["0"] for r in px["rows"]}
    put("PptOne", val[1], psrc + ": rows[n=1] (Proposition 11), both channels, every p")
    put("PptTwo", val[2], psrc + ": rows[n=2] (Proposition 11), both channels, every p")
    put("PptProbeDev", sci(max(pdevs), 1, ROUND_CEILING), psrc + ": probe_minus_exact, rounded up")
    put("PptNmax", max(val), psrc + ": rows")
    put("PptPs", n_ps.pop(), psrc + ": the rational values of p of every row")
    put("PptJointDim", jt["primal"]["full_dimension"], psrc + ": joint")
    put("PptControls", len(ctl), psrc + ": controls, every one fails")

    # -------------------------------------------- R141: one direct use of P (x) A beats every N~ code (d = 2)
    dc = J(EV / "direct" / "certificate_direct_n1.json")
    bound_to(dc, EV / "direct", (("encoder", "encoder_sha256"), ("decoders", "decoders_sha256")), "direct n = 1")
    dlow = ball_lower(dc["certified_lower_bound"], "direct n = 1")
    nt1 = 1 - Decimal(2).sqrt() / 4                    # F_c(N~, 2) = 1/2 + (1 - p)/4 at p = sqrt 2 - 1, exactly
    need(dc["n"] == 1 and dlow > nt1, "the one-use direct code no longer beats the exact N~ optimum")
    need(dlow < TQ, "the one-use direct code exceeds 3/4, which the one-use lemma forbids: an input is wrong")
    put("DirectF", lo(dlow, 10), "direct/certificate_direct_n1.json")
    put("NtOneOpt", lo(nt1, 10), "computed: 1/2 + (1 - p)/4 at p = sqrt 2 - 1, i.e. 1 - sqrt 2/4")
    put("DirectGain", lo(dlow - nt1, 4), "direct/certificate_direct_n1.json minus the exact optimum, truncated")

    # -------------------------------------------- the family's d = 2 member IS Smith and Yard's channel (F10)
    kc = J(SF / "p2_kraus_comparison_v2.json")
    ksrc = "shield_family/p2_kraus_comparison_v2.json"
    need(kc["schema"] == "oneq.superactivation.p2-kraus-comparison/2" and kc["verdict"] == "IDENTICAL"
         and kc["source_sha256"] == sm["source_sha256"],
         f"{ksrc}: not identical, or not read from the pinned arXiv source")
    ex = kc["exact_symbolic"]
    pairs = ("family_vs_kraus_S2.20-S2.27", "family_vs_choi_formula_S2.4-S2.17", "kraus_vs_choi_formula")
    need(ex.get("entries") == 256 and all(ex.get(k) == 256 for k in pairs),
         "the exact comparison did not prove every entry equal for the Kraus operators AND the Choi-state formula")
    kdiff = max(kc["max_abs_differences"].values())
    kctl = kc["controls_must_differ"]
    typo_key = "exact_check_with_the_M1_sign_typo_family_vs_kraus_entries_equal"
    swap_key = "exact_check_with_swapped_formula_weights_family_vs_choi_formula_entries_equal"
    need(kdiff < 1e-14 and kctl["smith_yard_original_M1_sign_typo_vs_family_d2"] > 1e-3
         and kctl["family_at_0.97p_vs_kraus"] > 1e-3 and kctl.get(typo_key, 256) < 256
         and kctl.get(swap_key, 256) < 256,
         f"{ksrc}: a difference is too large or a control is not caught")
    for f, blob in kc["provenance"]["code_git_blobs"].items():
        raw = (ROOT / f).read_bytes()
        need(hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob,
             f"{ksrc} was produced by other bytes of {f}; re-run it")
    put("KrausDiff", sci(kdiff, 1, ROUND_CEILING), ksrc + ", rounded up")
    put("KrausEntries", ex["entries"], ksrc + ": exact_symbolic")
    put("KrausTypoDiff", rd(kctl["smith_yard_original_M1_sign_typo_vs_family_d2"], 3), ksrc + ": controls")
    put("KrausTypoUnequal", 256 - kctl[typo_key], ksrc + ": entries the exact check leaves unequal under the typo")
    put("KrausSwapUnequal", 256 - kctl[swap_key], ksrc + ": entries unequal with the formula's weights exchanged")

    # -------------------------------------------- the shield family: two uses (R142, R186)
    def two_use(name: str) -> tuple[dict, Decimal]:
        c = J(SF / name)
        d = c["d"]
        need(abs(c["p"] - float(p_of(d))) < 1e-15, f"{name}: p is not 1/(sqrt d + 1)")
        need(c["psd_by_construction"] is True, f"{name}: not the PSD-by-construction certifier")
        if c.get("code_sha256"):
            need(sha256(SF / c["code"]) == c["code_sha256"], f"{name}: the code bytes changed")
        low = ball_lower(c["certified_lower_bound"], name)
        need(c["exceeds_three_quarters"] == (low > TQ), f"{name}: the verdict disagrees with its bound")
        need(0 <= D(c["pair_float_fidelity"]) - low < D("1e-9"), f"{name}: the bound is not just below its pair")
        return c, low

    c19, l19 = two_use("certificate_two_use_d19.json")
    c20, l20 = two_use("certificate_two_use_d20.json")
    c21, l21 = two_use("certificate_two_use_d21.json")
    c22, l22 = two_use("certificate_two_use_d22_closed.json")
    c64, l64 = two_use("certificate_two_use_d64_closed.json")
    ca20, la20 = two_use("certificate_two_use_famA_d20.json")
    ca32, la32 = two_use("certificate_two_use_famA_d32.json")
    need(l19 < TQ and l20 > TQ and l21 > TQ and l22 > TQ and l64 > TQ, "a two-use verdict changed")
    # the closed form: recomputed exactly and held against its certificates
    for c, low in ((c22, l22), (c64, l64)):
        f = closed_form(p_of(c["d"]))
        need(low <= f and f - low < D("1e-12"), f"d = {c['d']}: the closed-form certificate is not the closed form")
    p9 = Fraction(1, 9)
    need((7 - 6 * p9 + 2 * p9 * p9) / 8 == Fraction(515, 648), "the closed form at p = 1/9 is not 515/648")
    need(p_of(64) == D(p9) and c64["d"] == 64, "p(64) is not 1/9")
    pstar = (3 - Decimal(7).sqrt()) / 2
    dstar = 11 + 4 * Decimal(7).sqrt()
    need(abs(closed_form(pstar) - TQ) < D("1e-60"), "(3 - sqrt 7)/2 is not where the closed form meets 3/4")
    vals = [closed_form(p_of(d)) for d in range(2, 400)]
    need(all(vals[i + 1] > vals[i] for i in range(len(vals) - 1)), "the closed form is not increasing in d")
    closed_dmin = next(d for d in range(2, 400) if closed_form(p_of(d)) > TQ)
    need(closed_dmin == 22 and dstar < 22 < dstar + 1, "the least d above the closed-form threshold is not 22")
    need(c22["d"] == closed_dmin, "the closed-form certificate is not at the least d it covers")
    # family A (R186 part 1): F_A* = [9 - 2p + sqrt(sqrt d + (3 - 2p)^2)]/16, above 3/4 iff d > (49 - sqrt 97)/2
    for c, low in ((ca20, la20), (ca32, la32)):
        fa = family_a(c["d"])
        need(low <= fa and fa - low < D("1e-9"), f"family A at d = {c['d']}: the certificate is not F_A*")
    da_star = (49 - Decimal(97).sqrt()) / 2
    famA_dmin = next(d for d in range(2, 400) if family_a(d) > TQ)
    need(famA_dmin == 20 and da_star < 20, "family A's least crossing d is not 20")
    need(family_a(32) > closed_form(p_of(32)), "family A no longer beats the closed form at d = 32")
    # the producer-free exact-rational checker, re-run on the d = 20, 21, 22 witnesses
    rat = {}
    for c, low in ((c20, l20), (c21, l21), (c22, l22)):
        d = c["d"]
        res = rv.prove(c, J(SF / f"rational_recovery_d{d}.json"))
        need(res["crosses_three_quarters_PROVEN_on_effective_channel"] is True and res["strict_margin_sign"] == 1,
             f"the exact-rational checker no longer proves d = {d}")
        r = D(res["lower_bound_floor_15_decimals"])
        need(abs(r - low) < D("1e-12"), f"d = {d}: the exact-rational bound and the ball bound disagree")
        rat[d] = r
    # the exact n-use certifier, re-run on every committed record
    nuse = {}
    for rec in sorted(SF.glob("certificate_nuse_d*_n*.json")):
        res = nc.verify(rec)
        nuse[(int(res["d"]), int(res["n"]))] = res
    need(len(nuse) >= 16, "fewer n-use certificates than the onset table rests on")
    nlow = {k: D(v["certified_lower_bound"]) for k, v in nuse.items()}
    for k, v in nuse.items():
        need((nlow[k] > TQ) == v["exceeds_three_quarters"], f"n-use {k}: the verdict disagrees with its bound")
        need(abs(v["p"] - float(p_of(k[0]))) < 1e-15, f"n-use {k}: p is not 1/(sqrt d + 1)")
    need(nuse[(8, 3)]["exceeds_three_quarters"] is False, "the dark case d = 8, n = 3 is no longer below 3/4")
    # two uses for every d >= TwoUseDmin: certificates below the closed form's reach, none at d - 1
    cross2 = ({d for (d, n), v in nuse.items() if n == 2 and v["exceeds_three_quarters"]}
              | {c["d"] for c, low in ((c19, l19), (c20, l20), (c21, l21)) if low > TQ})
    two_dmin = min(cross2)
    need(two_dmin == 20, "the least certified two-use crossing is not d = 20")
    need(all(d in cross2 for d in range(two_dmin, closed_dmin)), "a d between the certificates and the closed form "
                                                                   "has no two-use certificate")
    need(c19["d"] == two_dmin - 1 and l19 < TQ and D(c19["pair_float_fidelity"]) < TQ,
         "the best two-use code at d - 1 is no longer recorded below 3/4")
    put("TwoUseDmin", two_dmin, "certificate_nuse_d20_n2.json, certificate_two_use_d20.json (least crossing d)")
    put("TwoUseDnext", two_dmin + 1, "certificate_nuse_d21_n2.json, certificate_two_use_d21.json")
    put("TwoUseDbelow", c19["d"], "shield_family/certificate_two_use_d19.json (the best two-use code below 3/4)")
    put("ClosedSixtyFourD", c64["d"], "shield_family/certificate_two_use_d64_closed.json")
    put("FamAD", ca32["d"], "shield_family/certificate_two_use_famA_d32.json")
    put("ClosedDmin", closed_dmin, "computed: least integer d with (7 - 6p + 2p^2)/8 > 3/4 at p = 1/(sqrt d + 1)")
    put("ClosedPstar", lo(pstar, 4), "computed: (3 - sqrt 7)/2, truncated")
    put("ClosedDstar", rd(dstar, 1), "computed: 11 + 4 sqrt 7")
    put("FamADstar", rd(da_star, 1), "computed: (49 - sqrt 97)/2")
    put("FamAPstar", lo((Decimal(97).sqrt() - 1) / 48, 4), "computed: (sqrt 97 - 1)/48, truncated")
    put("FamADopt", rd(21 + 12 * Decimal(3).sqrt(), 1), "computed: 21 + 12 sqrt 3 (family A is the closed form beyond)")
    put("TwoNineteen", measured(c19["pair_float_fidelity"], 5, "d = 19"), "shield_family/certificate_two_use_d19.json")
    put("TwoTwentyBall", lo(l20, 10), "shield_family/certificate_two_use_d20.json")
    put("TwoTwentyOneBall", lo(l21, 10), "shield_family/certificate_two_use_d21.json")
    need(lo(rat[20], 10) == lo(l20, 10) and lo(rat[21], 10) == lo(l21, 10) and lo(rat[22], 10) == lo(l22, 10),
         "the exact-rational checker does not reproduce the printed digits")
    # the text: "At d = 20 a code with two entangling gates does better" than family A's certified d = 20 code
    need(nlow[(20, 2)] > la20, "the d = 20 n-use code no longer beats family A at d = 20")
    put("NuseTwenty", lo(nlow[(20, 2)], 10), "shield_family/certificate_nuse_d20_n2.json (re-proved here)")
    put("NuseTwentyOne", lo(nlow[(21, 2)], 10), "shield_family/certificate_nuse_d21_n2.json (re-proved here)")
    put("NuseTwentyMargin", sci(nlow[(20, 2)] - TQ, 1, ROUND_FLOOR), "certificate_nuse_d20_n2.json, truncated")
    put("ClosedTwentyTwo", lo(l22, 10), "shield_family/certificate_two_use_d22_closed.json")
    put("ClosedTwentyTwoMargin", sci(closed_form(p_of(22)) - TQ, 1, ROUND_FLOOR), "computed: closed form at d = 22")
    put("ClosedSixtyFourFrac", "515/648", "computed: (7 - 6p + 2p^2)/8 at p = 1/9; certificate_two_use_d64_closed.json")
    put("ClosedSixtyFourMargin", lo(D(Fraction(515, 648)) - TQ, 3), "computed: 515/648 - 3/4, truncated")
    put("FamATwenty", lo(la20, 10), "shield_family/certificate_two_use_famA_d20.json")
    put("FamAThirtyTwo", lo(la32, 10), "shield_family/certificate_two_use_famA_d32.json")
    put("ClosedThirtyTwo", rd(closed_form(p_of(32)), 7), "computed: the closed form at d = 32")
    put("FamAGain", sci(family_a(32) - closed_form(p_of(32)), 1, ROUND_HALF_UP), "computed: F_A* - closed form, d = 32")

    # -------------------------------------------- the onset table (R186 part 4, which corrects R142's)
    pd = J(SF / "phase_diagram.json")
    need(sorted(int(k) for k in pd) == list(range(2, 22)), "phase_diagram.json does not cover d = 2..21")
    certified_ds, spreads, seeds, restarts = [], [], set(), set()
    for dk, row in pd.items():
        d = int(dk)
        need(abs(row["p"] - float(p_of(d))) < 1e-15, f"phase diagram d = {d}: p is not 1/(sqrt d + 1)")
        F = {int(n): v for n, v in row["F"].items()}
        on = row.get("onset_upper_bound")
        crossing = [n for n in sorted(F) if F[n] > 0.75]
        if on is None:
            need(not crossing, f"phase diagram d = {d}: a crossing with no onset")
        else:
            need(crossing and crossing[0] == on, f"phase diagram d = {d}: the onset is not the first crossing")
        if row.get("onset_class") == "CERTIFIED":
            need((d, on) in nuse and nuse[(d, on)]["exceeds_three_quarters"], f"d = {d}: CERTIFIED with no certificate")
            need(row["onset_certified_lower_bound"] == nuse[(d, on)]["certified_lower_bound_float"],
                 f"d = {d}: the table quotes another bound than its certificate")
            need(row["entries"][str(on)]["class"] == "CERTIFIED", f"d = {d}: entry class disagrees")
            certified_ds.append(d)
        for e in row["entries"].values():
            if "seed_spread" in e:
                spreads.append((e["seed_spread"], d))
                seeds.add(len(e["seeds"]))
                restarts.add(e["restarts_per_seed"])
    for (d, n), v in nuse.items():
        if v["exceeds_three_quarters"]:
            need(pd[str(d)]["onset_upper_bound"] <= n, f"certified crossing d = {d}, n = {n} missing from the table")
    need(sorted(certified_ds) == list(range(7, 22)), "the certified onsets are not exactly d = 7..21")
    by_onset = {}
    for d in certified_ds:
        by_onset.setdefault(pd[str(d)]["onset_upper_bound"], []).append(d)
    need(sorted(by_onset) == [2, 3, 4] and by_onset[4] == [7, 8] and by_onset[3] == list(range(9, 20))
         and by_onset[2] == [20, 21], "the certified onsets no longer group as 4 (d = 7, 8), 3 (d = 9..19), 2 (d = 20, 21)")
    put("OnsetFourLo", min(by_onset[4]), "shield_family/phase_diagram.json: certified onsets equal to 4")
    put("OnsetFourHi", max(by_onset[4]), "shield_family/phase_diagram.json")
    put("OnsetThreeLo", min(by_onset[3]), "shield_family/phase_diagram.json: certified onsets equal to 3")
    put("OnsetThreeHi", max(by_onset[3]), "shield_family/phase_diagram.json")
    need(pd["9"]["onset_upper_bound"] == 3 and pd["2"]["onset_upper_bound"] == pub_n
         and pd["2"]["family"] == "symmetric", "the table's d = 2 or d = 9 row changed")
    need(pd["5"]["onset_upper_bound"] is None, "d = 5 is no longer undetermined")
    need(len(seeds) == 1 and len(restarts) == 1, "the general searches do not share one seed and restart count")
    put("OnsetCertLo", min(certified_ds), "shield_family/phase_diagram.json: rows with onset_class CERTIFIED")
    put("OnsetCertHi", max(certified_ds), "shield_family/phase_diagram.json")
    put("SearchSeeds", seeds.pop(), "shield_family/phase_diagram.json: entries[].seeds")
    put("SearchRestarts", restarts.pop(), "shield_family/phase_diagram.json: entries[].restarts_per_seed")
    smax, sd = max(spreads)
    put("SeedSpreadMax", sci(smax, 1, ROUND_CEILING), "shield_family/phase_diagram.json: max seed_spread, rounded up")
    put("SeedSpreadD", sd, "shield_family/phase_diagram.json: where the spread is largest")
    need(nlow[(8, 3)] < TQ, "the dark case's certified lower bound is no longer below 3/4")
    put("DarkF", lo(nlow[(8, 3)], 5), "certificate_nuse_d8_n3.json: certified lower bound (re-proved), truncated")
    put("DarkD", 8, "certificate_nuse_d8_n3.json: the code certified below 3/4")
    put("DarkN", 3, "certificate_nuse_d8_n3.json")
    put("TableDLo", min(int(k) for k in pd), "shield_family/phase_diagram.json: the rows of Table III")
    put("TableDHi", max(int(k) for k in pd), "shield_family/phase_diagram.json")
    # the text: "the thinnest margin among the n-use certificates of the table is at d = 9, n = 3"
    thin = min(certified_ds, key=lambda d: pd[str(d)]["onset_certified_lower_bound"])
    need(thin == 9 and pd["9"]["onset_upper_bound"] == 3 and nlow[(9, 3)] > TQ
         and float(nlow[(9, 3)]) == pd["9"]["onset_certified_lower_bound"],
         "the thinnest certified onset margin of Table III is no longer d = 9, n = 3")
    put("ThinD", thin, "shield_family/phase_diagram.json: the certified onset with the least margin")
    put("ThinN", pd[str(thin)]["onset_upper_bound"], "shield_family/phase_diagram.json: its n")
    put("NineF", lo(nlow[(9, 3)], 10), "certificate_nuse_d9_n3.json (re-proved here)")
    put("NineMargin", sci(nlow[(9, 3)] - TQ, 1, ROUND_FLOOR), "certificate_nuse_d9_n3.json, truncated")
    # the text: "For d <= 6 no code with n <= 4 that our search found crosses" (and the best values it quotes)
    small = {d: {int(n): v for n, v in pd[str(d)]["F"].items() if int(n) <= 4} for d in range(2, 7)}
    need(all(sorted(F) == [2, 3, 4] and all(v < 0.75 for v in F.values()) for F in small.values()),   # n = 1: Lemma 6
         "a code with n <= 4 at some d <= 6 now crosses 3/4, or an entry is missing")
    put("FiveBest", measured(pd["5"]["F"]["4"], 4, "d = 5"), "shield_family/phase_diagram.json d = 5, n = 4 (float)")
    put("SixFour", measured(pd["6"]["F"]["4"], 4, "d = 6"), "shield_family/phase_diagram.json d = 6, n = 4 (float)")

    def onset_row(d: int) -> str:
        row = pd[str(d)]
        F = {int(n): v for n, v in row["F"].items()}
        p = rd(p_of(d), 4)
        if d == 2:                                      # the two-group certificate (R140) replaces the symmetric 17
            c, low = tgr[15]
            need(int(n14.group(3)) == c["n"] - 1, "the d = 2 row's n - 1 search is not at n - 1")
            return rf"{d} & {p} & {c['n']} & {lo(low, 7)} & cert. & {measured(n14.group(4), 7, 'n14')} \\"
        on = row.get("onset_upper_bound")
        if on is None:
            best = max(F)
            return rf"{d} & {p} & -- & -- & -- & {measured(F[best], 7, f'd={d}')} ($n={best}$) \\"
        if row.get("onset_class") == "CERTIFIED":
            val, cls = lo(nlow[(d, on)], 7), "cert."
        else:
            val, cls = measured(F[on], 7, f"d={d}"), "float"
        if on - 1 == 1:
            prev = r"$\le 3/4^{\ddagger}$"
        elif (d, on - 1) in nuse:
            prev = measured(nuse[(d, on - 1)]["pair_float_fidelity"], 7, f"d={d}") + r"$^{\dagger}$"
        else:
            prev = measured(F[on - 1], 7, f"d={d} n-1")
        return rf"{d} & {p} & {on} & {val} & {cls} & {prev} \\"

    tab_onset = "\n".join([
        r"\begin{tabular}{@{}rcrlcl@{}}", r"\toprule",
        r"$d$ & $p$ & $n^*\le$ & $F$ at $n$ & class & best at $n-1$ \\", r"\midrule",
        *[onset_row(d) for d in range(2, 22)],
        rf"$\ge {closed_dmin}$ & $\le {rd(p_of(closed_dmin), 4)}$ & 2 & Eq.~(\ref{{eq:closed}}) & proven & "
        r"$\le 3/4^{\ddagger}$ \\",
        r"\bottomrule", r"\end{tabular}"]) + "\n"

    # -------------------------------------------- the d = 2 table
    def dtwo_row(n: int) -> str:
        sym_auth = rd(auth[n], 7) if n in auth else "--"
        if n == 14:
            return rf"{n} & ${n14.group(1)}{{+}}{n14.group(2)}$ & {sym_auth} & {measured(n14.group(4), 7, 'n14')} & -- \\"
        if n in tgr:
            c, low = tgr[n]
            return (rf"{n} & ${c['a']}{{+}}{c['b']}$ & {sym_auth} & "
                    rf"{measured(c['pair_float_fidelity'], 10, f'tg{n}')} & {lo(low, 10)} \\")
        cert = lo(symlow[n], 10) if n in symlow else "--"
        return rf"{n} & sym. & {sym_auth} & {measured(curve_f[n], 10, f'sym{n}')} & {cert} \\"

    tab_dtwo = "\n".join([
        r"\begin{tabular}{@{}rcccc@{}}", r"\toprule",
        r"$n$ & code & authors' & $F$ (float) & certified $F\ge$ \\", r"\midrule",
        *[dtwo_row(n) for n in range(14, 25)],
        r"\bottomrule", r"\end{tabular}"]) + "\n"

    # -------------------------------------------- the claims and their evidence classes (Table I)
    tab_claims = "\n".join([
        r"\begin{tabular}{@{}p{0.60\textwidth}p{0.25\textwidth}l@{}}", r"\toprule",
        r"statement & class & where \\", r"\midrule",
        r"The group reduction behind the symmetric and two-group certificates is exact & proven; also checked "
        r"against the dense construction & Lemma~\ref{lem:groups} \\",
        rf"Smith--Yard pair ($d=2$): a code with $n={tgr[15][0]['n']}$ uses has $F\ge{lo(tgr[15][1], 10)}$, so "
        rf"$n^*\le {tgr[15][0]['n']}$ (published: $n^*\le {pub_n}$) & certified; the steps from it to $n^*$ "
        r"proven & Thm.~\ref{thm:dtwo} \\",
        r"$P_d$, the construction of Ref.~\cite{horodecki2008low}, is PPT if and only if $p_1=1/(\sqrt d+1)$, for "
        r"every complex Hadamard twist (``if'' for a symmetric twist: Ref.~\cite{horodecki2008low}; ``only if'' for "
        r"the Hadamard twist: its Obs.~2, and for every twist: here); "
        r"$P_2$ is Smith and Yard's channel & proven; $P_2$ by exact computation & Thm.~\ref{thm:ppt} \\",
        r"One joint use of $P_d$ and $A_{2d}$ simulates $\nt_p$, $p=1/(\sqrt d+1)$ (Smith and Yard's coupling; "
        r"at $d=2$, Prop.~1 of Ref.~\cite{parentin2026onset}) & proven & Lemma~\ref{lem:reduction} \\",
        r"Neither $P_d$ nor $A_{2d}$ alone exceeds $3/4$ at any number of uses & proven (standard argument) & "
        r"Lemma~\ref{lem:alone} \\",
        r"One use of any PPT channel with a 50\% erasure channel: $F_c\le 3/4$ & proven & Lemma~\ref{lem:oneuse} \\",
        rf"Two uses of $P_d\otimes A_{{2d}}$: $F_c>3/4$ for every $d\ge {two_dmin}$ & certified ($d={two_dmin},"
        rf"{two_dmin + 1}$); proven ($d\ge {closed_dmin}$) & Thm.~\ref{{thm:two}} \\",
        rf"At threshold $3/4$, $n^*=2$ exactly for every $d\ge {two_dmin}$ & follows from the rows above & "
        r"Cor.~\ref{cor:exact} \\",
        rf"$n^*\le 4$ for $d={by_onset[4][0]},{by_onset[4][1]}$ and $n^*\le 3$ for "
        rf"$d={by_onset[3][0]},\dots,{by_onset[3][-1]}$ & certified upper bounds & Table~\ref{{tab:onset}} \\",
        rf"One use of the joint channel ($d=2$): a code with $F\ge {lo(dlow, 10)}$ beats every code on the "
        rf"reduced channel ($\max={lo(nt1, 10)}$) & certified and proven & Thm.~\ref{{thm:direct}} \\",
        r"A PPT-assisted relaxation cannot certify a lower bound on $n^*$ above $2$: at $n$ uses its value is exactly "
        r"$1-2^{-(n+1)}$, for $\nt_p$, for $A$ and for every pair $P_d\otimes A_{2d}$ & proven; feasible point and dual "
        r"certificate also checked exactly on the full output space for $\nt_p$ and $A$ ($n\le\PptNmax$) and for "
        r"$P_2\otimes A_4$ ($n=1$) & Prop.~\ref{prop:ppt} \\",
        r"\bottomrule", r"\end{tabular}"]) + "\n"

    # -------------------------------------------- the hardware outlook (not a result of this paper)
    fz = J(EV / "hw_run_m5" / "frozen_20261015T195926Z.json")
    hr = J(EV / "hw_run_m5" / "result_db8jpre94fic73etigpg.json")
    need(hr["job"]["qpy_sha256"] == fz["qpy_sha256"], "the R145 result is not the frozen run's")
    ms = hr["analysis"]["measured"]
    need(ms["F_e"] < 0.75 and ms["upper99_F_e"] < 0.75 and hr["analysis"]["verdicts"]["superactivation_on_this_device"]
         is False, "the device run is no longer recorded below 3/4")
    put("HwD", fz["d"], "hw_run_m5/frozen_20261015T195926Z.json")
    put("HwFe", rd(ms["F_e"], 3), "hw_run_m5/result_db8jpre94fic73etigpg.json: analysis.measured.F_e")
    put("HwFeSd", rd(ms["sd_F_e"], 3), "hw_run_m5/result_db8jpre94fic73etigpg.json: analysis.measured.sd_F_e")
    put("HwPred", rd(hr["analysis"]["predicted"]["F_e"], 3), "hw_run_m5/result_db8jpre94fic73etigpg.json")
    over = D(hr["analysis"]["predicted"]["F_e"]) - D(ms["F_e"])
    nsig = over / D(ms["sd_F_e"])
    need(over > 0 and nsig > 3, "the model no longer over-predicts the device run by several standard errors")
    put("HwOver", rd(over, 3), "hw_run_m5/result_db8jpre94fic73etigpg.json: predicted minus measured F_e")
    put("HwOverSigma", rd(nsig, 1), "hw_run_m5/result_db8jpre94fic73etigpg.json: that difference over sd_F_e")
    # the text: "On IBM heavy-hex devices our calibrated model predicts that no shield size we modelled reaches 3/4"
    rq = J(EV / "r186_faithful_requirements" / "r186_summary.json")
    hh = [r for r in rq["runs"].values() if r["device"].startswith("kingston_hh") or r["device"] == "spec:kingston_spec"]
    need(rq["bar"] == 0.75 and len(hh) >= 8 and {r["m"] for r in hh} == {5, 6} and all(r["F_e"] < 0.75 for r in hh),
         "the calibrated heavy-hex model now reaches 3/4 at some modelled shield size, or no longer covers m = 5, 6")
    pr = J(EV / "ion_run_m6" / "design_n2000" / "pass_rates.json")
    faithful = [dev["arms"][arm]["F_e_projected"] for dev in pr for arm in ("barriers", "exec")]
    mlo, mhi = D(min(faithful)) - TQ, D(max(faithful)) - TQ
    need(0 < mlo <= mhi < over, "the projected ion margins are no longer positive and below the device run's miss")
    put("IonMarginLo", lo(mlo, 3), "ion_run_m6/design_n2000/pass_rates.json: channel-faithful arms, truncated")
    put("IonMarginHi", hi(mhi, 3), "ion_run_m6/design_n2000/pass_rates.json: channel-faithful arms, rounded up")
    old = J(EV / "ion_run_m6" / "design_n2000" / "design.json")
    new = J(EV / "ion_run_m8_r224" / "design.json")
    seq = J(EV / "ion_run_m8_r224" / "sequential.json")
    branches_old = {u["branch"] for u in old["units"]}
    branches_new = {u["branch"] for u in new["units"]}
    need("erased_kept" not in branches_old and "kept_erased" in branches_old,
         "the first ion package is no longer the folded one the text describes")
    need({"erased_kept", "kept_erased"} <= branches_new and new["erasure_orders"] == "both"
         and new["erasure_marginals"] == {"use0": 0.5, "use1": 0.5, "both": 0.25},
         "the re-frozen ion package does not sample both one-erased orders")
    need(seq["rule"] == "betting_ad" and seq["units"] == len(new["units"]), "the re-frozen package's rule changed")
    put("IonOldUnits", grouped(len(old["units"])), "ion_run_m6/design_n2000/design.json: units")
    put("IonOldD", old["d"], "ion_run_m6/design_n2000/design.json")
    put("IonUnits", grouped(len(new["units"])), "ion_run_m8_r224/design.json: units")
    put("IonD", new["d"], "ion_run_m8_r224/design.json")

    # -------------------------------------------- the release that carries this lane
    put("LaneRelease", *lane_release(release_builder_contents()))

    body = ["% GENERATED by paper/tools/make_numbers_superactivation.py from committed evidence -- do not edit.",
            "% Each macro is preceded by the file it was read from. Certified lower bounds are truncated."]
    for k in sorted(M):
        v, src = M[k]
        body.append(f"% {src}")
        body.append(rf"\newcommand{{\{k}}}{{{v}}}")
    return {"numbers.tex": "\n".join(body) + "\n", "tab_claims.tex": tab_claims, "tab_dtwo.tex": tab_dtwo,
            "tab_onset.tex": tab_onset}


def main() -> int:
    files = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print("wrote", (OUT / name).relative_to(ROOT).as_posix() if (OUT / name).is_relative_to(ROOT) else OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
