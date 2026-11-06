r"""Paper 3 after the two external audits of 2026-11-03 (docs/reviews/EXTERNAL{1,2}_2026-11-03.md): each repair,
held both ways.

* SERIALIZATION (audit 2, F04). Every hash-pinned file the lane writes has ONE byte format whatever the platform
  (serialization.py). Emulating the other platform's text-mode default in-process, the model writers reproduce the
  archived bytes, the JSON writers the released bytes; an AST scan refuses any text-mode write in the lane that does
  not name its newline; the committed files have the declared format.
* THEOREM 2'S HYPOTHESIS (audit 2, F08). Equal RAW noise on mirrored layers does not give the fold: a probe on the
  released morph72 gates has no solution at T = 1 and a weight-12 one at T = 2, while the hypothesis the proof uses
  -- equal mid-cycle IMAGES, checked by fold_uniform.check_PAL -- refuses that noise.
* SAME PARAMETERS, NOT THE SAME CODE (audit 2, F09). The morph144 end-cycle code, read off the emitted circuit, has
  weight-9 stabilizers; every element of the gross code's stabilizer group has even weight.
* PAN'S CIRCUITS (audit 1, P3-09). records/pan_crosscheck.jsonl -- our projected mechanism sets against Pan's own
  circuit files and builder -- is all-equal, every control fails, and it was produced from the current sources.
"""
from __future__ import annotations

import ast
import builtins
import collections
import hashlib
import io
import json
import pathlib
import sys

import numpy as np
import pytest
import stim

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "experiments" / "circuit_distance_exact"
EV = ROOT / "evidence" / "circuit_distance_exact"
RECORDS = EV / "records"
sys.path.insert(0, str(LANE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402
import morph2026 as m26  # noqa: E402

CR, LF = bytes([13]), bytes([10])
CRLF = CR + LF


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- serialization: the platform cannot choose the bytes

@pytest.fixture(params=["\n", "\r\n"], ids=["linux-default", "windows-default"])
def platform_default(request, monkeypatch):
    """Text-mode writes that leave newline unset get THIS platform's line end, as on that operating system; writes that
    name their newline are untouched. Patches every route a lane writer takes (open, io.open, hence Path.open)."""
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if any(c in mode for c in "wax") and "b" not in mode and newline is None:
            newline = request.param
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)
    monkeypatch.setattr(io, "open", fake)
    return request.param


def _recert_cell(family, basis, rounds):
    man = json.loads((EV / "recert_20261102" / "manifest.json").read_text(encoding="utf-8"))
    return next(c for c in man["cells"] if c["family"] == family and c["basis"] == basis and c["rounds"] == rounds)


def test_the_model_exporter_writes_the_archived_bytes_on_either_platform(platform_default, tmp_path):
    c = _recert_cell("morph72", "Z", 1)
    archived = (EV / "recert_20261102" / c["model"]).read_bytes()
    assert _sha(archived) == c["model_sha256"] and archived.count(CRLF) == archived.count(LF)
    out = tmp_path / "m.txt"
    em.export(c["circuit"], c["basis"], c["rounds"], out, time_order=c["maxround"] >= 0)
    assert out.read_bytes() == archived, "the exporter's bytes depend on the platform's newline"


def test_the_2026_exporter_writes_the_archived_bytes_on_either_platform(platform_default, tmp_path):
    man = json.loads((EV / "recert_20261102" / "manifest.json").read_text(encoding="utf-8"))
    c = next(c for c in man["cells"] if c["family"] == "morph2026" and c["rounds"] == 1)
    archived = (EV / "recert_20261102" / c["model"]).read_bytes()
    _, path, _ = m26.model(m26.spec(c["circuit"]), c["basis"], c["rounds"], tmp_path, c["maxround"] >= 0)
    assert path.read_bytes() == archived, "the 2026 exporter's bytes depend on the platform's newline"


def test_results_json_is_written_in_the_released_bytes_on_either_platform(platform_default, tmp_path, monkeypatch):
    import collect_results as cr
    monkeypatch.setattr(cr, "EVID", tmp_path)          # the output path only; the claims still read records/
    cr.main()
    fresh = (tmp_path / "results.json").read_bytes()
    assert CR not in fresh, "results.json picked up the platform's CRLF"
    assert fresh == (EV / "results.json").read_bytes()


def test_the_lean_receipt_is_written_in_the_released_bytes_on_either_platform(platform_default, tmp_path):
    import lean_receipt as lr
    committed = lr.RECEIPT.read_bytes()
    lr.write_receipt(json.loads(committed), tmp_path / "r.json")
    assert (tmp_path / "r.json").read_bytes() == committed and CR not in committed


def test_the_crlf_record_writer_reproduces_the_archived_record_on_either_platform(platform_default, tmp_path):
    import serialization as ser
    rec = (RECORDS / "3c_shift_certificates_w10.jsonl").read_bytes()
    ser.write_text_crlf(tmp_path / "r.jsonl", rec.decode("utf-8").replace("\r\n", "\n"))
    assert (tmp_path / "r.jsonl").read_bytes() == rec


def _text_writes_without_newline(path: pathlib.Path) -> list[str]:
    """Calls that write TEXT and leave the newline to the platform: open(..., "w"|"a"|"x"...) without "b" and without
    newline=, and Path.write_text(...) without newline=."""
    bad = []
    where = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    for node in ast.walk(ast.parse(path.read_bytes())):
        if not isinstance(node, ast.Call):
            continue
        kws = {k.arg for k in node.keywords}
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name == "open":
            mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
            if (isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax")
                    and "b" not in mode.value and "newline" not in kws):
                bad.append(f"{where}:{node.lineno} open(mode={mode.value!r})")
        elif name == "write_text" and "newline" not in kws:
            bad.append(f"{where}:{node.lineno} write_text")
    return bad


def test_no_lane_writer_leaves_its_newline_to_the_platform():
    bad = [b for p in sorted(LANE.rglob("*.py")) for b in _text_writes_without_newline(p)]
    assert not bad, "text-mode writes that inherit the platform's newline:\n  " + "\n  ".join(bad)


def test_the_scan_sees_the_defect_it_was_written_for(tmp_path):
    """The scan can fail: the two forms the audit found are flagged, the repaired forms are not."""
    probe = tmp_path / "probe.py"
    src = ('open(out, "w")\nopen(out, "w", encoding="ascii", newline="\\r\\n")\np.write_text(t)\n'
           'p.write_text(t, newline="\\n")\nopen(out, "rb")\nopen(out, "wb")\n')
    probe.write_bytes(src.encode("utf-8"))
    found = _text_writes_without_newline(probe)
    assert sorted(f.split(" ", 1)[1] for f in found) == ["open(mode='w')", "write_text"]


def test_the_committed_files_have_their_declared_format():
    for rel in ("results.json", "lean_r198_receipt.json", "records/recert_determinism.jsonl",
                "records/recert_old_prover.jsonl", "records/pan_crosscheck.jsonl"):
        assert CR not in (EV / rel).read_bytes(), f"{rel} should be LF"
    rec = (RECORDS / "3c_shift_certificates_w10.jsonl").read_bytes()
    assert rec.count(CRLF) == rec.count(LF) > 0, "the 3-colour certificate record should be CRLF"
    models = sorted((EV / "recert_20261102" / "models").rglob("*.txt")) + sorted(
        (EV / "ibm_recert_20261026" / "models").rglob("*.txt"))
    assert len(models) == 282 + 22
    for m in models:
        b = m.read_bytes()
        assert b.count(CRLF) == b.count(LF) > 0, f"{m.name} is not a CRLF model"


# ---------------------------------------------------------------- Theorem 2: images, not raw channels

def _raw_mirrored_x_noise(T: int, time_mirrored: bool = False):
    """The released morph72 memory-Z circuit with every noise channel removed except an independent X_ERROR on the 12
    qubits of one X logical of the mid-cycle code, after G_{0,0} in the contraction and in the expansion slot of the
    same layer (time_mirrored=False: the literal "noise of each expansion layer equals that of its mirrored
    contraction layer") or after G_{0,1} in the expansion (time_mirrored=True). Returns (circuit, tags)."""
    tags = []
    orig = bc.morphing_cycle("morph72", "Z", T, tags=tags)
    by = {t[0]: t for t in tags}
    hx, hz = bc.BB(12, 6).check_matrices()
    q = [int(x) for x in np.flatnonzero(bc.BB(12, 6).logicals(hx, hz)[0])]
    out, new_tags = stim.Circuit(), []
    exp_layer = 1 if time_mirrored else 2
    for idx, inst in enumerate(orig.flattened()):
        t = by.get(idx)
        if t is None:
            out.append(inst)
            continue
        _, _, phase, rnd, layer = t
        if (phase == "con" and rnd % 2 == 0 and layer == 0) or (phase == "exp" and (rnd - 1) % 2 == 0
                                                                 and layer == exp_layer):
            new_tags.append((len(out), "X_ERROR", phase, rnd, layer))
            out.append("X_ERROR", q, 1e-3)
    flat = list(out.flattened())
    assert all(flat[i].name == "X_ERROR" for i, *_ in new_tags), "stim fused instructions; the tags are off"
    return out, new_tags, q


def _min_solution(circuit):
    """Exact minimum weight of an undetectable logical set of the DEM's mechanisms (None if there is none)."""
    mech = []
    for inst in circuit.detector_error_model(decompose_errors=False).flattened():
        if inst.type == "error":
            d = sum(1 << t.val for t in inst.targets_copy() if t.is_relative_detector_id())
            l_ = sum(1 << t.val for t in inst.targets_copy() if t.is_logical_observable_id())
            mech.append((d, l_))
    piv, kernel = {}, []
    for i, (d, _) in enumerate(mech):
        comb, v = 1 << i, d
        while v:
            h = v.bit_length() - 1
            if h not in piv:
                piv[h] = (v, comb)
                break
            v, comb = v ^ piv[h][0], comb ^ piv[h][1]
        if not v:
            kernel.append(comb)
    assert len(kernel) <= 16
    best = None
    for bits in range(1, 1 << len(kernel)):
        comb = 0
        for j, k in enumerate(kernel):
            if bits >> j & 1:
                comb ^= k
        obs = 0
        for i in range(len(mech)):
            if comb >> i & 1:
                obs ^= mech[i][1]
        if obs and (best is None or bin(comb).count("1") < best):
            best = bin(comb).count("1")
    return len(mech), best


def test_raw_mirrored_noise_is_not_enough_and_the_image_condition_refuses_it():
    import fold_uniform as fu
    c1, _, q = _raw_mirrored_x_noise(1)
    c2, tags2, _ = _raw_mirrored_x_noise(2)
    n1, d1 = _min_solution(c1)
    n2, d2 = _min_solution(c2)
    assert (n1, d1) == (12, None), "the 1-round model should have 12 mechanisms and no solution"
    assert n2 == 24 and d2 == 12, "the 2-round model should have a weight-12 solution: d(T) != d(1)"
    D = fu.Data("morph72", "Z", "x")
    rows2, *_ = fu.components(D, c2, tags2)
    img = fu.Images(D)
    s = {t: fu.body_sets(D, rows2, img, t) for t in (0, 1)}
    raw_con, im_con = s[0][("con", 0)]
    raw_exp, im_exp = s[1][("exp", 2)]
    assert raw_con == raw_exp, "the raw channels of the mirrored layers should be equal (the literal wording holds)"
    assert im_con != im_exp, "their mid-cycle images should differ"
    pal = fu.check_PAL(D, rows2, img)
    assert not pal["holds"] and not pal["root_layer_invariance"], "PAL (images) must refuse this noise"
    # the lane's own noise passes the same check, so the refusal is about the noise, not the check
    c, tags = fu.build(D, 2)
    rows, *_ = fu.components(D, c, tags)
    assert fu.check_PAL(D, rows, fu.Images(D))["holds"]
    # the same raw channel at TIME-mirrored slots has equal images: no solution appears at T = 2
    ct, _, _ = _raw_mirrored_x_noise(2, time_mirrored=True)
    assert _min_solution(ct)[1] is None


# ---------------------------------------------------------------- the morph144 end-cycle code is not the gross code

def test_the_morph144_end_cycle_code_has_odd_weight_stabilizers_and_the_gross_code_has_none():
    gx, gz = bc.BB(12, 6).check_matrices()
    assert set(gx.sum(1).tolist()) == set(gz.sum(1).tolist()) == {6}
    for basis in "ZX":
        circ = bc.morphing_cycle("morph144", basis, 1)
        circ.detector_error_model(decompose_errors=False)  # refuses a non-deterministic detector
        first_data, k = circ.num_measurements - 144, 0
        weights = collections.Counter()
        for inst in circ.flattened():
            if inst.name in ("M", "MX"):
                k += len(inst.targets_copy())
            elif inst.name == "DETECTOR":
                n = sum(1 for t in inst.targets_copy() if k + t.value >= first_data)
                if n:
                    weights[n] += 1
        assert weights == {9: 72}, weights


# ---------------------------------------------------------------- our circuits against Pan's own

def test_the_pan_crosscheck_record_is_all_equal_with_live_controls_and_current_sources():
    rows = [json.loads(x) for x in (RECORDS / "pan_crosscheck.jsonl").read_text(encoding="utf-8").splitlines() if x]
    head, body = rows[0], rows[1:]
    assert head["kind"] == "header" and head["all_equal"] and head["controls_all_fail"]
    assert head["pan_release"]["tag_commit"] == "5ae848ab86d25e5235f7db4fe3c8df17f071fe60"
    for key, name in (("bb_circuits", "bb_circuits.py"), ("export_model", "export_model.py"),
                      ("this_script", "pan_crosscheck.py")):
        assert head["ours"][key] == _sha((LANE / name).read_bytes()), f"re-run pan_crosscheck.py: {name} changed"
    cmp_ = [r for r in body if r.get("kind") != "control"]
    ctl = [r for r in body if r.get("kind") == "control"]
    want = {(c, b, t) for c in ("ibm", "3c", "morph72", "morph144") for b in "ZX" for t in (1, 2, 3)}
    assert {(r["circuit"], r["basis"], r["T"]) for r in cmp_} == want
    assert all(r["equal"] and r["observable_map_rank"] == 12 and r["only_ours"] == r["only_pan"] == 0 for r in cmp_)
    assert len(ctl) == 8 and all(r["fails_as_required"] and not r["equal"] for r in ctl)


# ---------------------------------------------------------------- a record cited at a pinned commit is checked

def test_a_reference_pinned_to_a_commit_is_held_to_that_commit_and_its_readme():
    """Pan's E-037 and E-043 are in no release, so refs.bib cites them at their commits (final referee of
    2026-11-03, N1). verify_refs accepts the committed snapshot record and refuses a record for another commit, a
    missing file, another year, or a README that does not name the author."""
    sys.path.insert(0, str(ROOT / "paper" / "tools"))
    import verify_refs as vr
    entries = vr.parse_bib(vr.BIB.read_text(encoding="utf-8"))
    snap = json.loads(vr.SNAPSHOT.read_text(encoding="utf-8"))["entries"]
    _commit_pinned_entries_are_checked(vr, entries, snap)


def _commit_pinned_entries_are_checked(vr, entries, snap):
    for key in ("pan2026e037", "pan2026e043"):
        e, s = entries[key], snap[key]
        assert vr.GITHUB_COMMIT_FILE.fullmatch(e["url"]) and vr.compare(key, e, s) == []
        g = s["github_commit"]
        for bad in (dict(g, commit="0" * 40), dict(g, blob=None), dict(g, year=g["year"] - 1),
                    dict(g, readme_words="nobody here")):
            assert vr.compare(key, e, {"github_commit": bad}), bad
        # the ENTRY mutated, the stored record kept (final referee of 2026-11-03, residual 2): another file at the same
        # commit, another record's identifier in the title, a translated title that does not say so
        for bad_e in (dict(e, url=e["url"].replace(".md", "-NOPE.md")),
                      dict(e, title=e["title"].replace("E-0", "E-9")),
                      {k: v for k, v in e.items() if k != "note"}):
            assert vr.compare(key, bad_e, s), bad_e


# ---------------------------------------------------------------- Table II's CNOT-order pairs act on ONE code

def _end_cycle_span(name: str, basis: str, T: int) -> dict:
    """GF(2) span (pivot dict) of the data-readout parts of the emitted circuit's deterministic final detectors: the
    basis-type part of the end-cycle stabilizer group after round T."""
    sp = m26.spec(name)
    c = m26.build(sp, basis, T)
    c.detector_error_model(decompose_errors=False)  # refuses a non-deterministic detector
    first, k, piv = c.num_measurements - sp["code"].n2, 0, {}
    for inst in c.flattened():
        if inst.name in ("M", "MX"):
            k += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            v = 0
            for t in inst.targets_copy():
                if k + t.value >= first:
                    v ^= 1 << (k + t.value - first)
            while v:
                h = v.bit_length() - 1
                if h in piv:
                    v ^= piv[h]
                else:
                    piv[h] = v
                    break
    return piv


def _same(p1: dict, p2: dict) -> bool:
    joint = dict(p1)
    for v in p2.values():
        while v:
            h = v.bit_length() - 1
            if h in joint:
                v ^= joint[h]
            else:
                return False
    return len(p1) == len(p2)


def test_each_cnot_order_pair_of_table_ii_has_one_end_cycle_code():
    """main.tex Sec. VI C calls Table II's CNOT-order pairs the fixed-code comparisons: the two circuits of a pair
    share A and B as sets (one mid-cycle code) AND, read off their circuits, one end-cycle stabilizer group in both
    types after either round parity (final referee of 2026-11-03). The check can fail: two different codes differ."""
    for a, b in (("c42a", "c42a2"), ("c42b", "c42b2"), ("c48a", "c48a2"), ("c48b", "c48b2"), ("c48c", "c48c2")):
        ta, tb = m26.TARGETS[a], m26.TARGETS[b]
        assert sorted(map(str, ta["A"])) == sorted(map(str, tb["A"])) and sorted(map(str, ta["B"])) == sorted(map(str, tb["B"]))
        for basis in "ZX":
            for T in (1, 2):
                assert _same(_end_cycle_span(a, basis, T), _end_cycle_span(b, basis, T)), (a, b, basis, T)
    # controls: another code differs; and with A, B and the circuit FIXED, the end-cycle code after the other round
    # parity -- a different end-cycle code on the same qubits -- differs too, so the check discriminates on the axis
    # the claim is about (exchanging two contraction layers is no such control: measured, it keeps the end-cycle code)
    assert not _same(_end_cycle_span("c42a", "Z", 1), _end_cycle_span("c42b", "Z", 1))
    for name in ("c42a", "c48a"):
        for basis in "ZX":
            assert not _same(_end_cycle_span(name, basis, 1), _end_cycle_span(name, basis, 2)), (name, basis)
