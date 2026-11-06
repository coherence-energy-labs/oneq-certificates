"""Independent, explicit recovery-map cross-check for R142's d=20,21,22 claims."""
import copy
import importlib.util
import json
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
path = ROOT / "tools" / "verify_r142_two_use_independent.py"
spec = importlib.util.spec_from_file_location("r142_independent", path)
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)
CERTS = ROOT / "evidence" / "superactivation" / "shield_family"
FILES = {
    20: "certificate_two_use_d20.json",
    21: "certificate_two_use_d21.json",
    22: "certificate_two_use_d22_closed.json",
}


def source(d):
    return json.loads((CERTS / FILES[d]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("d,expected", [
    (20, 0.7508847632706279),
    (21, 0.7526654055459749),
    (22, 0.7509200542015164),
])
def test_independent_small_matrix_recoveries_reproduce_certified_numbers(d, expected):
    res = independent.verify(source(d))
    assert res["achievable_fidelity_double_precision"] == pytest.approx(expected, abs=1e-12)
    assert abs(res["difference_from_recorded"]) < 1e-12
    assert res["crosses_three_quarters_numerically"] is True
    assert 0 <= res["encoder_gram_deviation_before_polar"] < 1e-12
    branch = res["branch_fidelities"]
    assert 0.0 < branch["both_flagged"] <= 0.5
    assert branch["both_kept"] == 1
    assert 0 <= branch["first_flagged"] <= 1
    assert 0 <= branch["second_flagged"] <= 1
    assert res["status"].endswith("NOT_INTERVAL_PROOF")
    for family in ("first_flagged", "second_flagged"):
        for dec in res["recovery_witnesses"][family]:
            assert sum(q*q for q in dec["su2_quaternion"]) == pytest.approx(1, abs=1e-12)


def test_analytic_closed_code_at_d22():
    d = 22
    p = 1 / (1 + math.sqrt(d))
    F = (7 - 6*p + 2*p*p) / 8
    assert independent.verify(source(d))["achievable_fidelity_double_precision"] == pytest.approx(F, abs=1e-12)


@pytest.mark.parametrize("attack", ["encoder_not_isometry", "wrong_p", "wrong_fidelity",
                                   "false_crossing", "wrong_margin", "nan_encoder"])
def test_adversarial_certificate_changes_refused(attack):
    data = copy.deepcopy(source(20))
    if attack == "encoder_not_isometry":
        for row in data["code_columns"]:
            row[1] = row[0][:]
    elif attack == "wrong_p":
        data["p"] = 0.5
    elif attack == "wrong_fidelity":
        data["certified_lower_bound_float"] = 0.99
    elif attack == "false_crossing":
        data["exceeds_three_quarters"] = False
    elif attack == "wrong_margin":
        data["certified_margin_lower_bound"] = 0.1
    elif attack == "nan_encoder":
        data["code_columns"][0][0][0] = float("nan")
    with pytest.raises(independent.WitnessRefused):
        independent.verify(data)
