"""The hardware analysis for PREREG_superactivation_hw_floor_2026-10-12 recovers known noise.

Counts are generated exactly (expected values, large shot count) from a model in which the
Bell pair suffers depolarizing replacement delta0 and each CZ adds eps; readout is perfect. The
analysis must return delta0, eps and the flagged-channel fidelity of that model -- and must FAIL
to (P2 false) when the inserted Paulis are mislabeled, so the check can both pass and fail.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hw_flagged_channel", ROOT / "experiments" / "superactivation" / "hw_flagged_channel.py")
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)

SHOTS = 10 ** 8


def pauli_bell_corr(sigma: str):
    """<XX>, <YY>, <ZZ> of (1 x sigma)|Phi+>."""
    return {"I": (1, -1, 1), "X": (1, 1, -1), "Y": (-1, -1, -1), "Z": (-1, 1, 1)}[sigma]


def counts_for(corr: float) -> dict:
    """Two-qubit counts with parity correlator `corr`, split evenly within each parity."""
    pe = (1 + corr) / 2
    a = round(SHOTS * pe / 2)
    b = round(SHOTS * (1 - pe) / 2)
    return {"00": a, "11": a, "01": b, "10": b}


def synthetic(delta0: float, eps: float, swap_xy: bool = False) -> dict:
    out = {}
    lam0 = 1 - delta0                       # depolarizing shrinks every correlator by (1 - delta)
    for s in hw.PAULIS:
        lab = {"X": "Y", "Y": "X"}.get(s, s) if swap_xy else s
        xx, yy, zz = pauli_bell_corr(lab)
        for basis, c in zip(hw.BASES, (zz, xx, yy)):
            out[f"pauli_{s}_{basis}"] = counts_for(lam0 * c)
    for m in hw.DEPTHS:
        lam = lam0 * (1 - eps) ** m
        # F = (1 + 3 lam)/4 for depolarized |Phi+>; that is the model the fit linearises
        xx, yy, zz = pauli_bell_corr("I")
        for basis, c in zip(hw.BASES, (zz, xx, yy)):
            out[f"depth_{m}_{basis}"] = counts_for(lam * c)
    for c in hw.CALS:
        out[f"cal_{c}"] = {c[::-1]: SHOTS}          # perfect readout; keys are 'c1c0'
    return out


def test_the_analysis_recovers_the_injected_floor_and_flagged_channel():
    res = hw.analyse(synthetic(0.012, 0.004))["raw"]
    assert res["delta0"] == pytest.approx(0.012, abs=2e-5)
    model = (1 - 0.012) * hw.F_IDEAL + 0.012 / 4
    assert res["F_flagged_channel"] == pytest.approx(model, abs=2e-5)
    assert res["P2_within_2se"] is True
    assert res["F_per_pauli"]["X"]["F"] == pytest.approx(0.012 / 4, abs=2e-5)


def test_the_per_cz_fit_is_close_to_the_injected_error():
    res = hw.analyse(synthetic(0.012, 0.004))["raw"]
    # the fit is on lam = (4F-1)/3, exactly exponential in m under the injected model
    assert res["eps_CZ"] == pytest.approx(0.004, rel=1e-3)
    assert res["infeasibility_factor_R"] == pytest.approx(16 * res["eps_CZ"] / hw.DELTA_BUDGET)


def test_a_mislabeled_pauli_insertion_is_caught_by_P2():
    res = hw.analyse(synthetic(0.012, 0.004, swap_xy=True))["raw"]
    # swapping X and Y changes nothing (equal weights) -- the check must not be fooled by a
    # symmetric relabel into a false alarm
    assert res["P2_within_2se"] is True
    bad = synthetic(0.012, 0.004)
    bad["pauli_Z_XX"], bad["pauli_I_XX"] = bad["pauli_I_XX"], bad["pauli_Z_XX"]
    assert hw.analyse(bad)["raw"]["P2_within_2se"] is False


def test_the_budget_constant_is_the_certified_margin_over_F_minus_a_quarter():
    assert hw.DELTA_BUDGET == pytest.approx(1.38020e-4 / (0.750138 - 0.25), rel=1e-5)
    assert hw.F_IDEAL == pytest.approx(0.5 + (1 - (math.sqrt(2) - 1)) / 4)
