r"""The 2,002 held-out flat certificates are SHIPPED, not merely hashed.

Before 2026-09-12 a stranger holding the release could re-check the 17
branch-dual trees and none of the 2,002 flat receipts: each record carried
a 16-hex certificate hash and nothing else of the certificate, so the only
way to check one was to re-run the LP producer -- reproduction, not
checking. `experiments/m0_prior_art/qldpc_heldout_flat_certs.py` regenerates
every flat certificate from the shipped records, proves each IS the shipped
object by the runner's own hash formula, and stores the facet duals.

These tests hold that artifact to the campaign it claims to complete, and
re-check a sample of the stored certificates with BOTH checkers from the
stored payload alone -- no producer, which is the whole point.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from oneq.qldpc_check import QldpcCertificate, check_qldpc_exact   # noqa: E402

ART = ROOT / "evidence" / "qldpc_heldout_flat_certs.json"
HELDOUT = ROOT / "evidence" / "qldpc_heldout.json"
REF = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
FLAT_TIERS = ("tier1", "tier2")

needs_artifact = pytest.mark.skipif(not ART.exists(),
                                    reason="flat-certificate addendum not built")


@pytest.fixture(scope="module")
def d():
    return json.loads(ART.read_text(encoding="utf-8"))


@needs_artifact
def test_every_flat_receipt_of_the_campaign_is_stored_and_hash_bound(d):
    """Count must equal tier1 + tier2 over the held-out run, and every record
    that carries a hash must have been matched, none invented."""
    ho = json.loads(HELDOUT.read_text(encoding="utf-8"))
    expect = sum(r["tier1"] + r["tier2"] for r in ho["rungs"].values())
    t = d["totals"]
    assert t["attempted"] == expect == len(d["certificates"]), (t, expect)
    assert t["exact_accepted"] == expect
    assert t["ref_agreed"] == expect, "the pinned reference checker disagreed somewhere"
    # Every stored certificate is one of exactly three things: byte-identical
    # to the run's (the runner's own hash formula), a DIFFERENT optimal dual of
    # the same degenerate LP that both checkers still accept, or -- refsched
    # only -- unverifiable by hash because that rung's records carry none.
    # Nothing may fall outside those three.
    assert (t["hash_matched"] + t["hash_differs_but_valid"]
            + t["hash_unavailable"]) == expect, t
    # the ONLY rung allowed to lack shipped hashes is refsched (leaner schema)
    for tag, c in d["rungs"].items():
        if tag != "bb_refsched":
            assert c["hash_unavailable"] == 0, f"{tag} records carry hashes; none may be skipped"
            assert c["hash_matched"] + c["hash_differs_but_valid"] == c["attempted"], (tag, c)
    # and identity is the NORM, not the exception: a regeneration that mostly
    # disagreed with the run's own hashes would be a different producer
    assert t["hash_matched"] >= 0.9 * (expect - t["hash_unavailable"]), t
    assert all(("identical_to_run" in c) for c in d["certificates"])
    assert d["reference_checker_sha256"] == hashlib.sha256(REF.read_bytes()).hexdigest()
    assert d["freeze_commit"] == ho["freeze_commit"]


def _instance(tag, ho):
    """rebuild (checks, weights) for a rung -- the addendum's recipe"""
    import math
    import numpy as np
    from qldpc_bb_gate import bb_gross_code
    from qldpc_bb_phenom import spacetime_system
    from oneq.qldpc_cert import fixed_point_weights
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    cz = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    cx = [tuple(int(i) for i in np.flatnonzero(HX[j])) for j in range(m)]
    one = {i: 1 for i in range(n)}
    if tag in ("uniform_p01", "uniform_p02", "stress_p05"):
        return cz, one
    if tag == "xsector_p02":
        return cx, one
    if tag == "hetero_p02":
        rngw = np.random.default_rng(ho["rungs"]["hetero_p02"]["channel_seed"])
        p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
        llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
        return cz, fixed_point_weights(llr, scale=1000)[0]
    if tag.startswith("phenom"):
        p = 0.01 if tag.endswith("01") else 0.02
        cst, wst = spacetime_system(HZ, 4, p, p)
        return cst, fixed_point_weights(wst, scale=1000)[0]
    if tag == "bb_refsched":
        from qldpc_bb_refsched_gate import build_refsched_circuit
        from qldpc_circuit_gate import dem_to_matrices
        dem = build_refsched_circuit(3, 0.002).detector_error_model(decompose_errors=False)
        cc, wllr, _o, nm = dem_to_matrices(dem)
        return cc, {i: int(math.floor(1000 * float(wllr[i]) + 0.5)) for i in range(nm)}
    raise KeyError(tag)


@needs_artifact
def test_a_STORED_certificate_is_checkable_by_BOTH_checkers_with_no_producer(d):
    """Take stored certificates across rungs, rebuild only (checks, weights,
    syndrome) from the code and the record, and run the production exact
    checker AND the pinned reference checker on the STORED facet duals. No
    LP, no MIP, no BP-OSD is imported here."""
    ho = json.loads(HELDOUT.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("ref_checker", REF)
    ref = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = ref
    spec.loader.exec_module(ref)
    from qldpc_exact_all_gate import _cert_doc, _dense_H

    by_rung: dict[str, list] = {}
    for c in d["certificates"]:
        by_rung.setdefault(c["rung"], []).append(c)
    checked = 0
    for tag, certs in by_rung.items():
        checks, wint = _instance(tag, ho)
        n = len(wint)
        wx = {i: Fraction(int(wint[i])) for i in range(n)}
        dense = _dense_H(checks, n)
        for c in certs[:2]:                      # two per rung keeps this fast
            syn = [0] * len(checks)
            for j in c["syndrome_support"]:
                syn[j] = 1
            fd = [(tuple(r) if isinstance(r, list) else r, tuple(F), Fraction(num, den))
                  for (r, F, num, den) in c["facet_duals"]]
            ec = QldpcCertificate(error_support=tuple(c["candidate"]),
                                  facet_duals=fd, box_duals={})
            rx = check_qldpc_exact(checks=checks, weights=wx, syndrome=syn, cert=ec)
            assert rx.accepted, f"{tag} shot {c['shot_index']}: {rx.reason}"
            rep = ref.check_certificate(_cert_doc(dense, syn, wint, tuple(c["candidate"]), ec, n))
            assert rep["objective_optimal"], f"{tag} shot {c['shot_index']}: reference refused"
            checked += 1
    assert checked >= 16, checked


@needs_artifact
def test_a_HEAVIER_candidate_with_a_stored_dual_is_REFUSED(d):
    """The check can fail: the same stored dual must not certify a heavier
    feasible answer. Candidate XOR an H_X row stays in the syndrome coset."""
    import numpy as np
    from qldpc_bb_gate import bb_gross_code
    ho = json.loads(HELDOUT.read_text(encoding="utf-8"))
    HX, HZ = bb_gross_code()
    c = next(x for x in d["certificates"] if x["rung"] == "uniform_p02")
    checks, wint = _instance("uniform_p02", ho)
    wx = {i: Fraction(1) for i in range(len(wint))}
    syn = [0] * len(checks)
    for j in c["syndrome_support"]:
        syn[j] = 1
    heavy = tuple(sorted(set(c["candidate"]) ^ {int(i) for i in np.flatnonzero(HX[0])}))
    assert len(heavy) > len(c["candidate"])
    fd = [(tuple(r) if isinstance(r, list) else r, tuple(F), Fraction(num, den))
          for (r, F, num, den) in c["facet_duals"]]
    rx = check_qldpc_exact(checks=checks, weights=wx, syndrome=syn,
                           cert=QldpcCertificate(error_support=heavy, facet_duals=fd, box_duals={}))
    assert not rx.accepted, "a stored dual certified a HEAVIER feasible candidate"
