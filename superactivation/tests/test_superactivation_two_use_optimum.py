"""The optimal two-use code on N~_p^{(x)2} (two_use_optimum.py): family A's closed form, its one-CNOT circuit run
gate by gate, the CNOT-count classifier, the decoder bracket, the certificates, and the search. Every check is
paired with a case that must fail."""
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SF = ROOT / "experiments" / "superactivation" / "shield_family"
sys.path.insert(0, str(SF))
import two_use_optimum as t  # noqa: E402

EV = ROOT / "evidence" / "superactivation" / "shield_family"


@pytest.mark.parametrize("d", [16, 20, 24, 32, 40, 64])
def test_family_a_closed_form_is_the_evaluators_value_and_its_maximum(d):
    p = t.p_of(d)
    FA, delta = t.family_a_optimum(p)
    assert t.fidelity(t.family_a_code(delta), p) == pytest.approx(FA, abs=1e-12)
    assert t.family_a_value(p, delta) == pytest.approx(FA, abs=1e-14)
    grid = np.linspace(0, math.pi, 2001)
    vals = [t.fidelity(t.family_a_code(x), p) for x in grid]
    assert max(vals) <= FA + 1e-12                             # no Delta beats the closed-form optimum
    assert max(vals) > FA - 1e-6                               # and the grid gets close to it


def test_family_a_thresholds():
    for x in (5.0, 6.0, 6.4, 6.47, 7.0, 9.0):                  # the factorization behind D_STAR
        assert (x * x - 6 * x - 3) * (x * x + 6 * x + 1) == pytest.approx(x ** 4 - 38 * x * x - 24 * x - 3, abs=1e-9)
    below, above = t.family_a_optimum(t.p_of(D := t.D_STAR - 0.05)), t.family_a_optimum(t.p_of(t.D_STAR + 0.05))
    assert below[1] < math.pi and below[0] > t.closed_form(t.p_of(D))
    assert above[1] == math.pi and above[0] == t.closed_form(t.p_of(t.D_STAR + 0.05))
    pc = (math.sqrt(97) - 1) / 48                              # F_A* = 3/4 exactly here
    assert t.family_a_optimum(pc)[0] == pytest.approx(0.75, abs=1e-14)
    assert t.family_a_optimum(t.p_of(t.D_CROSS - 0.01))[0] < 0.75 < t.family_a_optimum(t.p_of(t.D_CROSS + 0.01))[0]
    assert t.family_a_optimum(t.p_of(20))[0] > 0.75 > t.family_a_optimum(t.p_of(19))[0]


@pytest.mark.parametrize("d", [16, 32, 64])
def test_the_one_cnot_protocol_run_gate_by_gate_reaches_family_a(d):
    p = t.p_of(d)
    c = t.family_a_circuit(p)
    V = t.family_a_code(c["delta"])
    assert abs(np.linalg.det(V.conj().T @ c["encoder"])) == pytest.approx(1, abs=1e-12)   # same code space
    sim = t.simulate_protocol(c, p)
    br = t.branch_values(V, p)
    for z in ("00", "01", "10", "11"):
        assert sim[z] == pytest.approx(br[z], abs=1e-12)
    assert sim["F"] == pytest.approx(c["F"], abs=1e-12)
    assert t.simulate_protocol(t.generic_protocol(V, p), p)["F"] == pytest.approx(c["F"], abs=1e-12)


@pytest.mark.parametrize("break_it", ["no_compromise", "conjugate_prep", "no_z_frame"])
def test_a_wrong_decoder_angle_loses_fidelity(break_it):
    """negative controls at d = 32: each closed-form angle is load-bearing"""
    p = t.p_of(32)
    c = t.family_a_circuit(p)
    h = c["delta"] / 2
    if break_it == "no_compromise":
        c["dec01"] = [np.diag([np.exp(-1j * k * h), np.exp(1j * k * h)]) @ t.H for k in (0, 1)]
    elif break_it == "conjugate_prep":
        c["dec11"] = {k: v.conj() for k, v in c["dec11"].items()}
    else:
        c["dec10"] = [c["dec10"][0], c["dec10"][0]]
    assert t.simulate_protocol(c, p)["F"] < c["F"] - 1e-5


def test_cnot_counts():
    p = t.p_of(32)
    assert t.min_cnots(t.family_a_code(t.family_a_optimum(p)[1])) == 1
    assert t.min_cnots(t.family_a_code(math.pi)) == 1                       # the closed form {|++>, |-->}
    prod = np.stack([np.kron([1, 0], [0.6, 0.8]), np.kron([0, 1], [0.6, 0.8])], axis=1).astype(complex)
    assert t.min_cnots(prod) == 0
    B = np.load(EV / "two_use_code_opt_d16.npy")
    assert t.min_cnots(B) == 2
    assert t.fit_cnot_circuit(B, 1, restarts=8)[0] > 1e-2                    # no 1-CNOT circuit makes this code
    assert t.fit_cnot_circuit(B, 2, restarts=8)[0] < 1e-9                    # two do
    assert t.fit_cnot_circuit(t.family_a_code(2.0), 1, restarts=8)[0] < 1e-9


@pytest.mark.parametrize("name, d", [("two_use_code_famA_d32.npy", 32), ("two_use_code_opt_d16.npy", 16)])
def test_the_decoder_bracket_closes_at_the_optima(name, d):
    """for these encoders the unitary corrections ARE the CPTP-optimal decoders (primal = dual)"""
    V = np.load(EV / name)
    p = t.p_of(d)
    br = t.decoder_bracket(V, p, iters=4000)
    assert br["F_dual"] - br["F_primal"] < 1e-12
    assert br["F_primal"] == pytest.approx(t.fidelity(V, p), abs=1e-12)


def test_the_dual_witness_is_an_upper_bound_not_an_echo():
    """negative control: an under-converged primal gives a STRICTLY larger dual, both bracketing the optimum"""
    p = t.p_of(16)
    V = np.load(EV / "two_use_code_opt_d16.npy")
    br = t.decoder_bracket(V, p, iters=3)
    assert br["F_primal"] < t.fidelity(V, p) - 1e-6 < br["F_dual"] - 1e-6


@pytest.mark.parametrize("cert, code, d, crosses", [
    ("certificate_two_use_famA_d32.json", "two_use_code_famA_d32.npy", 32, True),
    ("certificate_two_use_famA_d20.json", "two_use_code_famA_d20.npy", 20, True),
    ("certificate_two_use_famA_d16.json", "two_use_code_famA_d16.npy", 16, False),
    ("certificate_two_use_opt_d16.json", "two_use_code_opt_d16.npy", 16, False),
    ("certificate_two_use_d64_closed.json", None, 64, True),
])
def test_the_certificates_are_bound_to_their_codes(cert, code, d, crosses):
    c = json.loads((EV / cert).read_text(encoding="utf-8"))
    p = t.p_of(d)
    if code is None:
        V = t.family_a_code(math.pi)
        assert c["code_sha256"] is None
    else:
        raw = (EV / code).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c["code_sha256"]
        V = np.load(EV / code)
    F = t.fidelity(V, p)
    L = c["certified_lower_bound_float"]
    assert 0 <= F - L < 1e-12
    assert c["exceeds_three_quarters"] is crosses and (L > 0.75) is crosses
    if d == 32:
        assert L > t.closed_form(p) + 6.1e-4                  # the gain over {|++>, |-->} is certified
    if d == 64:
        assert 515 / 648 - L < 2e-15


def test_the_search_finds_family_a_at_d32_and_the_one_cnot_class_is_family_a():
    p = t.p_of(32)
    FA = t.family_a_optimum(p)[0]
    F, V, vals = t.search(p, 3, seed=1)
    assert F == pytest.approx(FA, abs=1e-9) and F > t.closed_form(p) + 6e-4
    assert t.best_one_cnot(p, starts=2)[0] == pytest.approx(FA, abs=1e-9)
