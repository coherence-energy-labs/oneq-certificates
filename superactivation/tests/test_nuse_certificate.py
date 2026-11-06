"""nuse_certificate.py (R186 part 4): the exact certifier of n-use superactivation codes, each check able to fail.
  - every committed certificate_nuse_*.json RE-DERIVES exactly from the code and decoder bytes it names (verify);
  - the DARK case is not certified: d = 8, n = 3's best code reads 0.7458 and its certificate says so (< 3/4);
  - at n = 2 it AGREES with R142's independent certifier (two_use_certificate.py) on the same code, to 1e-15;
  - tampered bytes, a record whose verdict was edited, a non-isometric code, a missing decoder and a decoder file for
    another (d, n) are all refused; the exact sign of a + b sqrt d is right on the knife edge.
"""
import json
import math
import pathlib
import shutil
import sys
from fractions import Fraction

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SF = ROOT / "experiments" / "superactivation" / "shield_family"
EV = ROOT / "evidence" / "superactivation" / "shield_family"
sys.path.insert(0, str(SF))
sys.path.insert(0, str(ROOT / "src"))
import nuse_certificate as NC  # noqa: E402

RECORDS = sorted(EV.glob("certificate_nuse_d*_n*.json"))


def test_there_are_certificates_to_check():
    assert len(RECORDS) >= 13                                   # d = 9..19 at n = 3, d = 20, 21 at n = 2, the dark d = 8


@pytest.mark.parametrize("rec", RECORDS, ids=[r.stem for r in RECORDS])
def test_every_committed_certificate_re_derives_exactly(rec):
    res = NC.verify(rec)
    want = json.loads(rec.read_text(encoding="utf-8"))
    assert res["exceeds_three_quarters"] == want["exceeds_three_quarters"]
    assert (Fraction(res["certified_lower_bound"]) > Fraction(3, 4)) == res["exceeds_three_quarters"]


def test_the_onset_table_claims_only_what_a_certificate_holds():
    """phase_diagram.json (R186 part 4): every onset labelled CERTIFIED names a certificate that exceeds 3/4 at that
    (d, n) with the bound the table quotes, and every certified crossing in the folder reaches the table"""
    table = json.loads((EV / "phase_diagram.json").read_text(encoding="utf-8"))
    certified = 0
    for d, row in table.items():
        n = row.get("onset_upper_bound")
        if row.get("onset_class") == "CERTIFIED":
            rec = json.loads((EV / f"certificate_nuse_d{d}_n{n}.json").read_text(encoding="utf-8"))
            assert rec["exceeds_three_quarters"] and row["onset_certified_lower_bound"] == rec["certified_lower_bound_float"]
            certified += 1
    assert certified >= 15                                       # d = 7..21
    for rec_path in RECORDS:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec["exceeds_three_quarters"]:
            row = table[str(rec["d"])]
            assert row["onset_upper_bound"] <= rec["n"], (rec_path.name, row["onset_upper_bound"])
    assert table["9"]["onset_upper_bound"] == 3 and table["9"]["onset_recorded_before_fix"] == 5


def test_the_dark_case_is_not_certified():
    rec = json.loads((EV / "certificate_nuse_d8_n3.json").read_text(encoding="utf-8"))
    assert rec["exceeds_three_quarters"] is False and rec["certified_lower_bound_float"] < 0.75


def test_it_agrees_with_the_independent_two_use_certifier():
    code = np.load(EV / "two_use_code_famA_d32.npy")
    decs, Ff = NC.power_decoders(code, 32, 2, iters=20000)
    mine = NC.certify(code, decs, 32, 2)
    theirs = json.loads((EV / "certificate_two_use_famA_d32.json").read_text(encoding="utf-8"))
    other = float(theirs["certified_lower_bound"].split(" ")[0].strip("["))
    assert mine["exceeds_three_quarters"] and abs(mine["certified_lower_bound_float"] - other) < 1e-15
    assert 0 <= Ff - mine["certified_lower_bound_float"] < 1e-9           # just below the pair's own float value


def _copy_record(tmp_path, stem):
    rec = json.loads((EV / f"{stem}.json").read_text(encoding="utf-8"))
    for name in (rec["code"], rec["decoders"]):
        shutil.copyfile(EV / name, tmp_path / name)
    out = tmp_path / f"{stem}.json"
    out.write_text(json.dumps(rec), encoding="utf-8")
    return out, rec


def test_tampered_bytes_and_an_edited_verdict_are_refused(tmp_path):
    out, rec = _copy_record(tmp_path, "certificate_nuse_d20_n2")
    assert NC.verify(out)["exceeds_three_quarters"]
    code = tmp_path / rec["code"]
    raw = bytearray(code.read_bytes())
    raw[-1] ^= 1
    code.write_bytes(bytes(raw))
    with pytest.raises(NC.CertificateRefusal, match="sha256 differs"):
        NC.verify(out)
    shutil.copyfile(EV / rec["code"], code)
    lie = dict(rec, certified_lower_bound="0.9")
    out.write_text(json.dumps(lie), encoding="utf-8")
    with pytest.raises(NC.CertificateRefusal, match="certified_lower_bound"):
        NC.verify(out)


def test_malformed_inputs_are_refused():
    V = np.load(EV / "general_code_d20_n2.npy")
    decs, _ = NC.power_decoders(V, 20, 2, iters=200)
    with pytest.raises(NC.CertificateRefusal, match="isometry"):
        NC.certify(2 * V, decs, 20, 2)
    with pytest.raises(NC.CertificateRefusal, match="one decoder per flag pattern"):
        NC.certify(V, {z: K for z, K in list(decs.items())[:-1]}, 20, 2)
    with pytest.raises(NC.CertificateRefusal, match="integers d >= 2"):
        NC.certify(V, decs, 1, 2)
    bad = dict(decs)
    bad[(0, 0)] = np.full((8, 2), np.nan)
    with pytest.raises(NC.CertificateRefusal, match="non-finite"):
        NC.certify(V, bad, 20, 2)


def test_a_decoder_file_for_another_d_or_n_is_refused(tmp_path):
    V = np.load(EV / "general_code_d20_n2.npy")
    decs, _ = NC.power_decoders(V, 20, 2, iters=200)
    code = tmp_path / "c.npy"
    np.save(code, V)
    dec = tmp_path / "dec.npz"
    NC.save_decoders(dec, 21, 2, decs)
    with pytest.raises(NC.CertificateRefusal, match=r"\(d, n\) = \(21, 2\)"):
        NC.certify_files(20, 2, code, dec, "test")


def test_the_exact_sign_of_a_plus_b_sqrt_d():
    F = Fraction
    assert NC.sign_of(F(3), F(-1), 9) == 0                      # 3 - sqrt 9 = 0 exactly
    assert NC.sign_of(F(3), F(-1), 8) == 1 and NC.sign_of(F(3), F(-1), 10) == -1
    assert NC.sign_of(F(-1), F(1), 2) == 1 and NC.sign_of(F(0), F(-1), 5) == -1 and NC.sign_of(F(2), F(0), 7) == 1
    lo = NC.lower_decimal(F(0), F(1), 2, digits=30)
    assert Fraction(lo) <= Fraction(math.isqrt(2 * 10 ** 60), 10 ** 30) + Fraction(1, 10 ** 30)
    assert lo.startswith("1.414213562373095048801688724")
