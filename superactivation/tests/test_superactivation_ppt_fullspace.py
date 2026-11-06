"""Proposition 11 of paper 4 on the full output space (ppt_relaxation_fullspace.py; adversary findings W1, W2).

Held here: the exact PSD tester is right over Q and Q(sqrt 2); the feasible point and the dual certificate give exactly
1 - 2^-(n+1) at n = 1, 2 for both channels and at every p tried, with no block structure assumed; the joint channel
P_2 (x) A_4 is exactly 3/4 at one use; and each check can fail -- a block scaled past 1/2, the dual with t - 1/64, and
the dual of a channel whose all-erased block is not PPT are refused, the last two by the condition that should refuse
them. The committed record is bound to the bytes of the code that wrote it.
"""
import hashlib
import json
import pathlib
import sys
from fractions import Fraction as Q

ROOT = pathlib.Path(__file__).resolve().parents[1]
SA = ROOT / "experiments" / "superactivation"
sys.path.insert(0, str(SA))
import ppt_relaxation_fullspace as fs  # noqa: E402


def test_psd_tester_is_calibrated_over_q_and_q_sqrt2():
    cal = fs.calibrate_psd()
    assert len(cal) >= 7 and all(cal.values()), cal
    s2 = fs.QS2(0, 1)
    assert (s2 * s2) == 2 and (1 / (s2 + 1)) == fs.QS2(-1, 1)
    assert fs.QS2(3, -2) > 0 and fs.QS2(-3, 2) < 0 and fs.QS2(-1, 1) > 0 and fs.QS2(1, -1) < 0


def test_feasible_point_and_dual_agree_exactly_on_the_full_space():
    for n in (1, 2):
        want = str(1 - Q(1, 2 ** (n + 1)))
        for m in (fs.model_flagged(), fs.model_erasure()):
            pr = fs.check_primal(m, n)
            assert pr["feasible"] and set(pr["value_at_p"].values()) == {want}, (m["name"], n, pr)
            for p in (Q(0), Q(1, 3), Q(1)):
                d = fs.check_dual(m, n, p=p)
                assert d["valid"] and d["t"] == want and d["Y + M P^TB == J"], (m["name"], n, p, d)


def test_joint_channel_is_exactly_three_quarters_at_one_use():
    jm = fs.model_joint(fs.QS2(-1, 1), "P_2 (x) A_4")
    chk = fs.p2_crosscheck(jm["JP"])
    assert chk["max_abs_difference_vs_ppt_family_choi_state"] < 1e-14
    assert all(v is True for k, v in chk.items() if k != "max_abs_difference_vs_ppt_family_choi_state"), chk
    pr = fs.check_primal(jm, 1, ps=(Q(0),))
    assert pr["full_dimension"] == 320 and pr["feasible"] and set(pr["value_at_p"].values()) == {"3/4"}
    d = fs.check_dual(jm, 1)
    assert d["valid"] and d["t"] == "3/4"


def test_controls_fail_and_for_the_right_reason():
    fl, er = fs.model_flagged(), fs.model_erasure()
    assert not fs.check_primal(fl, 2, kept_scale=Q(1))["feasible"]
    assert not fs.check_primal(er, 2, erased_scale=Q(1, 2) + Q(1, 64))["feasible"]
    t_low = fs.check_dual(fl, 2, t_shift=-Q(1, 64))
    assert not t_low["valid"] and t_low["P >= 0"] and not t_low["t 1 - Tr_B(Y + P) >= 0"]
    ident = fs.check_dual(fs.model_erased_branch_identity(), 2)
    assert not ident["valid"] and not ident["P >= 0"] and ident["Y >= 0"]
    non_ppt = fs.check_dual(fs.model_joint(fs.QS2(Q(1, 2)), "P_2 at p1 = 1/2"), 1)
    assert not non_ppt["valid"] and not non_ppt["P >= 0"] and non_ppt["Y >= 0"]


def test_committed_record_is_exact_and_bound_to_its_code():
    rec = json.loads((ROOT / "evidence" / "superactivation" / "ppt_relaxation_fullspace.json").read_text(
        encoding="utf-8"))
    assert rec["verdict"] == "EXACT"
    assert not any(c["passed_but_must_fail"] for c in rec["controls"]) and len(rec["controls"]) >= 13
    for f, blob in rec["provenance"]["code_git_blobs"].items():
        raw = (ROOT / f).read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob, f
