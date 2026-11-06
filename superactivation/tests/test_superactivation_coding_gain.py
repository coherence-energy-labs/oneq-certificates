"""The n=1 vs n=2 hardware coding-gain runs: re-derived from their committed raw counts.

v1 (chain 149-148-147) FAILED its predictions -- the gain was negative -- and v2 (chain 148-149-150)
passed them; both verdicts are held here from the counts, not from the stored summaries, so an edit
to the analysis that changed either would fail. The noise-free circuit simulation is the one test
that needs qiskit (the `qpu` extra, absent from CI's dev install, as for the other hardware lanes).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "evidence" / "superactivation"
spec = importlib.util.spec_from_file_location(
    "hw_coding_gain", ROOT / "experiments" / "superactivation" / "hw_coding_gain.py")
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)


def _result(pattern):
    return json.loads(sorted(EV.glob(pattern))[-1].read_text(encoding="utf-8"))


def test_v2_reproduces_from_its_counts_and_shows_a_coding_gain():
    rec = _result("coding_gain_v2_*.result.json")
    a = hw.analyse(rec["counts"])
    assert a["gain"] == pytest.approx(rec["analysis"]["gain"], abs=1e-12)
    assert rec["job"]["layout"] == [148, 149, 150]
    assert a["gain"] > 0.02                                   # measured 0.0261
    assert abs(a["n2"]["F10"] - a["n2"]["F01"]) < 0.02          # the mirror branches agree
    assert a["n2"]["F"] > hw.F1_IDEAL                          # noisy n=2 beats NOISELESS n=1
    assert all(rec["analysis"]["predictions"].values())


def test_v1_reproduces_from_its_counts_and_stays_a_recorded_failure():
    rec = _result("coding_gain_d*.result.json")
    a = hw.analyse(rec["counts"])
    assert rec["job"]["layout"] == [149, 148, 147]
    assert a["gain"] < 0                                       # measured -0.0081
    # the asymmetry that located the failure on qubit 147
    assert a["n2"]["F01"] - a["n2"]["F10"] > 0.05


def test_the_bootstrap_artifact_agrees_with_the_point_estimates():
    b = json.loads((EV / "coding_gain_bootstrap.json").read_text(encoding="utf-8"))
    v2 = _result("coding_gain_v2_*.result.json")
    assert b["v2"]["gain"] == pytest.approx(v2["analysis"]["gain"], abs=1e-12)
    lo, hi = b["v2"]["gain_95"]
    assert 0 < lo < b["v2"]["gain"] < hi
    assert b["v1"]["gain_95"][1] < 0
    assert b["v2"]["z"] > 10


def test_the_circuits_reproduce_the_ideal_fidelities_noise_free():
    pytest.importorskip("qiskit")   # not in the [dev] extra CI installs
    res = hw.analyse(hw.simulate_counts())
    assert res["n1"]["F"] == pytest.approx(hw.F1_IDEAL, abs=1e-9)
    assert res["n2"]["F"] == pytest.approx(res["ideal"]["n2"], abs=1e-7)
    assert res["n2"]["F00"] == pytest.approx(1.0, abs=1e-7)
