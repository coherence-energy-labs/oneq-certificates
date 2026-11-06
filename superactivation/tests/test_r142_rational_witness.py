"""Producer-free, exact-arithmetic proof gates for R142's explicit d20/d21/d22 codes."""
import copy
import importlib.util
import json
import pathlib
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "r142_exact", ROOT / "tools" / "verify_r142_rational_witness.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
EV = ROOT / "evidence" / "superactivation" / "shield_family"


def objects(d):
    name = f"certificate_two_use_d{d}.json" if d != 22 else "certificate_two_use_d22_closed.json"
    cert = json.loads((EV / name).read_text(encoding="utf-8"))
    witness = json.loads((EV / f"rational_recovery_d{d}.json").read_text(encoding="utf-8"))
    return cert, witness


@pytest.mark.parametrize("d,lower", [
    (20, "0.750884763270629"),
    (21, "0.752665405545975"),
    (22, "0.750920054201517"),
])
def test_independent_exact_rational_lower_bound_crosses_three_quarters(d, lower):
    res = verifier.prove(*objects(d))
    assert res["dimension"] == d
    assert res["lower_bound_floor_15_decimals"] == lower
    assert res["crosses_three_quarters_PROVEN_on_effective_channel"] is True
    assert res["strict_margin_sign"] == 1
    assert res["scope"].startswith("conditional effective flagged channel")
    assert 0 < Fraction(res["p_interval_width_exact"]) < Fraction(1, 10**30)


@pytest.mark.parametrize("attack", [
    "wrong_sha", "wrong_schema", "wrong_d", "zero_SU2", "zero_output",
    "tampered_encoder", "noninteger_quaternion", "wrong_grid",
])
def test_exact_proof_checker_refuses_broken_certificate_or_physical_map(attack):
    cert, w = objects(20)
    w = copy.deepcopy(w)
    cert = copy.deepcopy(cert)
    if attack == "wrong_sha":
        w["encoder_sha256"] = "f"*64
    elif attack == "wrong_schema":
        w["schema"] = "oneq.r142.unknown"
    elif attack == "wrong_d":
        w["d"] = 21
    elif attack == "zero_SU2":
        w["first_flagged"][0] = [0, 0, 0, 0]
    elif attack == "zero_output":
        w["both_flagged"][0] = [[0, 0], [0, 0]]
    elif attack == "tampered_encoder":
        cert["code_columns"][0][0][0] += 0.01
    elif attack == "noninteger_quaternion":
        w["first_flagged"][0][0] = 1.0
    elif attack == "wrong_grid":
        w["grid_scale"] = 0
    with pytest.raises(verifier.ProofRefused):
        verifier.prove(cert, w)
