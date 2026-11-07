"""Paper 2's generator rounds every printed bound AWAY from the inside (final referee of paper 2, 2026-11-03).

The circuit dataset's observable bound, 1 - 0.05^(1/149) = 1.9905%, is an UPPER bound and was printed floored, 1.99%:
below the bound the sentence states. Exact McNemar p-values were printed with '%.2g', so 0.0625 came out '0.062'.
These tests hold the rule (PF floors a point percentage or a lower bound, PC ceilings an upper bound, BOUND rounds a
significant-digit bound away from the inside, PVAL prints a p-value with enough digits) and the generated numbers
that depend on it.
"""
from __future__ import annotations

import math
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "tools"))

import make_numbers_qldpc as G  # noqa: E402

NUMBERS = ROOT / "paper" / "qldpc-certificates" / "generated" / "numbers.tex"


def _value(text: str) -> float:
    """The number a generated string prints, scientific notation included."""
    m = re.fullmatch(r"\\ensuremath\{([0-9.]+)\\times 10\^\{(-?\d+)\}\}", text)
    return float(m.group(1)) * 10 ** int(m.group(2)) if m else float(text)


def _macros() -> dict:
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", NUMBERS.read_text(encoding="utf-8")))


def test_an_upper_bound_percentage_is_ceilinged_and_a_lower_one_floored():
    b = 1 - 0.05 ** (1 / 149)
    assert G.PC(b) == "2.00" and G.PF(b) == "1.99"
    assert G.PC(0.02) == "2.00" == G.PF(0.02)                       # an exact value is not pushed out
    for x in (0.0001234, 0.019905, 0.5, 0.99999, 0.123456):
        assert float(G.PC(x)) >= 100 * x >= float(G.PF(x))


@pytest.mark.parametrize("x", [4.6219383875722997e-10, 0.21875, 0.0099, 9.96e-5, 0.5, 0.7265625, 3.3e-170, 0.123])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_significant_digit_bound_never_crosses_the_value_it_bounds(x, k):
    assert _value(G.BOUND(x, "ceil", k)) >= x * (1 - 1e-15)
    assert _value(G.BOUND(x, "floor", k)) <= x * (1 + 1e-15)


def test_bound_needs_a_direction():
    with pytest.raises(ValueError):
        G.BOUND(0.5, "nearest")


def test_p_values_print_enough_digits():
    assert G.PVAL(0.0625) == "0.0625"               # was '0.062' under '%.2g'
    assert G.PVAL(0.21875) == "0.219" and G.PVAL(0.6875) == "0.688" and G.PVAL(0.5) == "0.5"
    assert G.PVAL(1.0) == "1" and G.PVAL(0.625) == "0.625"
    assert G.PVAL(4.6219383875722997e-10) == r"\ensuremath{4.6\times 10^{-10}}"
    assert G.PVAL(0.001) == "0.001" and G.PVAL(0.00099) == r"\ensuremath{9.9\times 10^{-4}}"


def test_the_generated_bounds_enclose_their_values():
    """The typeset numbers themselves: every bound the text writes as <= or >= holds for the value it bounds."""
    M = _macros()
    assert float(M["HoRefschedObsUpperPct"]) >= 100 * (1 - 0.05 ** (1 / 149))
    import json
    d = json.loads((ROOT / "evidence" / "qldpc_scaling_prereg" / "summary.json").read_text(encoding="utf-8"))
    p = {k: s["paired_optimum_hit"]["exact_two_sided_mcnemar_p"] for k, s in d["datasets"].items()}
    sig = [v for v in p.values() if v < 0.05 / len(p)]
    rest = [v for v in p.values() if v >= 0.05 / len(p)]
    assert len(sig) == int(M["PrSigMore"]) and len(rest) == int(M["PrNoDiff"])
    assert _value(M["PrSigMaxP"]) >= max(sig) and _value(M["PrNoDiffMinP"]) <= min(rest)
    assert math.isclose(_value(M["PrUniOneMcNemar"]), p["uniform_p01"], rel_tol=5e-3)


def _ms(text: str) -> float:
    return float(text.replace("{,}", ""))


@pytest.mark.parametrize("s", [0.000311234, 0.0009996, 0.00249, 0.0049, 0.009996, 0.1274826, 0.6078, 1.2345678])
def test_a_latency_bound_never_lies_inside_the_value_it_bounds(s):
    """MS_BOUND (external audit 4, X6): a range's low end is floored and its high end, or an 'at most', ceilinged,
    in MS's own digits; a ceiling that reaches the next magnitude prints in that magnitude's digits."""
    assert _ms(G.MS_BOUND(s, "floor")) <= 1000 * s * (1 + 1e-12)
    assert _ms(G.MS_BOUND(s, "ceil")) >= 1000 * s * (1 - 1e-12)
    assert _ms(G.MS_BOUND(s, "floor")) <= _ms(G.MS(s)) <= _ms(G.MS_BOUND(s, "ceil"))


def test_a_latency_ceiling_prints_in_the_magnitude_it_reaches():
    assert G.MS_BOUND(0.0009996, "ceil") == "1.0" and G.MS_BOUND(0.009996, "ceil") == "10"
    assert G.MS_BOUND(0.0000311, "floor") == "0.031" and G.MS_BOUND(0.00498, "ceil") == "5.0"
    with pytest.raises(ValueError):
        G.MS_BOUND(0.001, "nearest")


def test_the_generated_ranges_and_upper_bounds_of_section_vi_enclose_their_values():
    """The typeset bounds the X6 pass found rounded to nearest: 'agreed to within' (printed 4.4e-16 for 4.44e-16),
    the CPU-over-wall range, the production checker's median range and decoding's 'at most'."""
    import json
    M = _macros()
    ev = ROOT / "evidence"
    worst = json.loads((ev / "qldpc_ensembles.json").read_text(encoding="utf-8"))["lazy_vs_enum"]["worst_abs_diff"]
    assert _value(M["EnsLazyWorst"]) >= worst
    costs = ev / "qldpc_scaling_prereg" / "costs"
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(costs.glob("production_*.json"))]
    assert len(docs) == 8
    cw = [d["run"]["cpu_over_wall"] for d in docs]
    assert float(M["PcCpuWallLo"]) <= min(cw) and max(cw) <= float(M["PcCpuWallHi"])
    dec = [G._nearest_rank([a + b for a, b in zip(d["shots"]["wall_s"]["decode_1.0"],
                                                  d["shots"]["wall_s"]["decode_0.625"])])["median"] for d in docs]
    assert 1000 * max(dec) <= _ms(M["PcDecodeMedMaxMs"])
    chk = json.loads((costs / "checking.json").read_text(encoding="utf-8"))
    pc = [G._nearest_rank(chk["series"][d["dataset"]]["production_checker_s"])["median"] for d in docs]
    assert _ms(M["PcProdCheckerMedLo"]) <= 1000 * min(pc) and 1000 * max(pc) <= _ms(M["PcProdCheckerMedHi"])
