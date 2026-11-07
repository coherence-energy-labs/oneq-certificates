"""The post hoc 2x2 of paper 1 Sec. VIII's two BP-OSD arms (experiments/m0_prior_art/gap_distribution_2x2_posthoc.py).

External audit 3 (2026-11-06): the preregistered arms differ in the input model AND the min-sum scaling, so their
difference is not the representation's. The 2x2 must (1) be a 2x2 -- four cells, each contrast changing one factor,
the preregistered cells configured exactly as preregistered; (2) score exactly as the preregistered run does, against
the RECORDED reference; (3) refuse to report anything unless it reproduces the two preregistered cells shot for shot
-- shown here on a small run of the preregistered runner, and by tampering with that run's record; (4) leave the
preregistered record untouched and say it is post hoc; and (5) the shipped artifact must re-summarise from its own
per-shot records joined with the preregistered ones.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
sys.path.insert(0, str(ROOT / "src"))

import gap_distribution as gd                                     # noqa: E402
import gap_distribution_2x2_posthoc as tx                         # noqa: E402

EV = ROOT / "evidence"
ART = EV / "gap_distribution_v4" / "posthoc_2x2_2026-11-06" / "summary.json"


def test_the_four_cells_are_a_2x2_and_each_contrast_changes_one_factor():
    assert set(tx.CELLS) == {(r, s) for r in ("merged", "component") for s in (0.625, 1.0)}
    assert set(tx.CELLS.values()) == set(tx.PREREGISTERED) | set(tx.NEW) and not set(tx.PREREGISTERED) & set(tx.NEW)
    cell_of = {v: k for k, v in tx.CELLS.items()}
    for a, b in tx.CONTRASTS.values():
        (ra, sa), (rb, sb) = cell_of[a], cell_of[b]
        assert (ra != rb) + (sa != sb) == 1, (a, b)                # exactly one factor differs
    # every cell, contrast by contrast, appears in one representation and one scaling contrast
    assert {k.split("_")[0] for k in tx.CONTRASTS} == {"representation", "scaling"}


def test_the_preregistered_cells_are_configured_as_preregistered():
    """bposd_merged is PRIMARY_BPOSD exactly; bposd_component is the released settings at the library default,
    which is min-sum scaling 1.0 -- read from the library, not assumed."""
    import numpy as np
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    assert tx.CELLS[("merged", 0.625)] == "bposd_merged" and tx.CELLS[("component", 1.0)] == "bposd_component"
    assert tx.settings_for(0.625) == gd.PRIMARY_BPOSD
    default = BpOsdDecoder(csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)), error_rate=0.1,
                           **gd.RELEASED_BPOSD).ms_scaling_factor
    assert default == 1.0
    assert tx.settings_for(1.0) == {**gd.RELEASED_BPOSD, "ms_scaling_factor": default}
    # the two settings differ in the scaling ONLY
    assert {k for k in tx.settings_for(0.625) if tx.settings_for(0.625)[k] != tx.settings_for(1.0)[k]} == \
        {"ms_scaling_factor"}


def test_a_candidate_is_scored_against_the_recorded_reference_exactly():
    W = {(0, 1): 1.5, (1, 2): 2.25, (-1, 0): 0.75}
    rec = {"pymatching": {"cost_exact": "9/4"}, "ref_eps": {"accepted": True, "epsilon": "1/1000000000"}}
    # the candidate selects edge (0,1) twice and (1,2) once: mod 2 it is {(1,2)}, odd at {1, 2}, cost 9/4 exactly
    cols = [(0, 1), (0, 1), (1, 2)]
    c = gd.project_selection(cols, [1, 1, 1], W)
    c["bp_converged"] = True
    e = tx.score(c, [1, 2], rec)
    assert e["edges"] == [[1, 2]] and e["consistent"] and e["edges_with_cancelled_copies"] == 1
    assert e["cost_exact"] == "9/4" and e["selected_columns"] == [0, 1, 2]
    assert (e["gap_lo"], e["gap_hi"]) == ("0", "1/1000000000") and tx.proven_within(e)
    # 3/2 heavier than the reference: a proven miss
    miss = tx.score(c, [1, 2], {**rec, "pymatching": {"cost_exact": "3/4"}})
    assert Fraction(miss["gap_lo"]) == Fraction(3, 2) and tx.proven_within(miss) is False
    # lighter than the proven lower bound w(J) - eps(J): the harness or the bound is broken, and it raises
    with pytest.raises(ValueError, match="below the proven lower bound"):
        tx.score(c, [1, 2], {**rec, "pymatching": {"cost_exact": "15/4"}})
    # off the reference (J not accepted) there is no interval and no verdict
    off = tx.score(c, [1, 2], {**rec, "ref_eps": {"accepted": False, "epsilon": None}})
    assert "gap_hi" not in off and tx.proven_within(off) is None
    # a syndrome-inconsistent candidate is never scored
    bad = tx.score(c, [0, 2], rec)
    assert not bad["consistent"] and "gap_hi" not in bad


def test_the_paired_contrast_counts_discordant_shots_and_their_exact_sign_test():
    from scipy.stats import binomtest

    def r(a_hi, b_hi):
        return {"A": {} if a_hi is None else {"gap_hi": a_hi}, "B": {"gap_hi": b_hi}}
    recs = [r("0", "0"), r("0", "1"), r("0", "1"), r("0", "1"), r("1", "0"), r("1", "1"), r(None, "0")]
    p = tx.paired(recs, "A", "B")
    assert (p["both_proven_within"], p["only_a"], p["only_b"], p["neither"]) == (1, 3, 1, 1)
    assert p["exact_mcnemar_p_two_sided"] == pytest.approx(binomtest(3, 4, 0.5).pvalue)
    assert tx.paired([r("0", "0")], "A", "B")["exact_mcnemar_p_two_sided"] == 1.0


def test_records_that_do_not_hash_to_the_summary_are_refused(tmp_path):
    body = b'{"shot": 0}\n'
    p = tmp_path / "per_shot_d3.jsonl.gz"
    p.write_bytes(gzip.compress(body))
    row = {"per_shot": str(p), "per_shot_sha256": hashlib.sha256(body).hexdigest()}
    assert tx.read_records(row) == [{"shot": 0}]
    with pytest.raises(SystemExit, match="does not hash"):
        tx.read_records({**row, "per_shot_sha256": "0" * 64})


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """A small run of the PREREGISTERED runner (seed 11, 12 shots, d = 3): the record the 2x2 must reproduce."""
    pytest.importorskip("beliefmatching")
    run = tmp_path_factory.mktemp("prereg")
    assert gd.main(["--seed", "11", "--shots", "12", "--distances", "3", "--tag", "t", "--out-dir", str(run)]) == 0
    return run


def _copy(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    summ = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    for row in summ["rows"]:
        row["per_shot"] = str(dst / pathlib.Path(row["per_shot"]).name)
    (dst / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
    return dst


def _rewrite(run: pathlib.Path, edit) -> None:
    """Apply `edit` to every per-shot record of the run and re-pin the file's digest in its summary (a consistent
    record of a DIFFERENT run, so only the reproduction gate can tell)."""
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    row = summ["rows"][0]
    p = pathlib.Path(row["per_shot"])
    recs = [json.loads(x) for x in gzip.decompress(p.read_bytes()).decode().splitlines()]
    body = "".join(json.dumps(edit(r), separators=(",", ":")) + "\n" for r in recs).encode()
    p.write_bytes(gzip.compress(body))
    row["per_shot_sha256"] = hashlib.sha256(body).hexdigest()
    (run / "summary.json").write_text(json.dumps(summ), encoding="utf-8")


def test_the_2x2_reproduces_a_preregistered_run_and_refuses_one_it_does_not(small_run, tmp_path, monkeypatch):
    run = _copy(small_run, tmp_path / "ok")
    monkeypatch.setattr(tx, "RUN", run)
    assert tx.main(["--out-dir", str(tmp_path / "out")]) == 0
    doc = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert doc["post_hoc"] is True and "POST HOC" in doc["label"]
    row = doc["rows"][0]
    stored = json.loads((run / "summary.json").read_text(encoding="utf-8"))["rows"][0]
    assert row["reproduced_preregistered_cells_on_every_shot"] and row["reproduced_preregistered_summaries"]
    for name in tx.PREREGISTERED:
        assert row["cells"][name] == stored["decoders"][name]
        assert row["settings_read_back"][name] == stored["bposd_settings"][name]
    assert row["settings_read_back"]["bposd_merged_unscaled"]["ms_scaling_factor"] == "1.0"
    assert row["settings_read_back"]["bposd_component_scaled"]["ms_scaling_factor"] == "0.625"
    # the per-shot file keeps the NEW cells only, and the preregistered record was not written to
    lines = gzip.decompress((tmp_path / "out" / "per_shot_d3.jsonl.gz").read_bytes()).decode().splitlines()
    assert len(lines) == row["nontrivial"] and set(json.loads(lines[0])) == {"shot", *tx.NEW}
    assert sorted(f.name for f in run.iterdir()) == sorted(f.name for f in small_run.iterdir())

    # a record whose merged arm selected different columns is a different run: refused, nothing written
    def drop_a_column(r):
        sel = r["bposd_merged"]["selected_columns"]
        r["bposd_merged"]["selected_columns"] = sel[1:] if sel else [0]
        return r
    bad = _copy(small_run, tmp_path / "bad")
    _rewrite(bad, drop_a_column)
    monkeypatch.setattr(tx, "RUN", bad)
    with pytest.raises(SystemExit, match="does not reproduce the record"):
        tx.main(["--out-dir", str(tmp_path / "out_bad")])
    assert not (tmp_path / "out_bad" / "summary.json").exists()

    # a recorded cost the regenerated graph does not give: the graph is not the record's
    def shift_cost(r):
        r["pymatching"]["cost_exact"] = str(Fraction(r["pymatching"]["cost_exact"]) + 1)
        return r
    off = _copy(small_run, tmp_path / "off")
    _rewrite(off, shift_cost)
    monkeypatch.setattr(tx, "RUN", off)
    with pytest.raises(SystemExit, match="costs differently on this graph"):
        tx.main(["--out-dir", str(tmp_path / "out_off")])


def test_a_cost_probe_never_writes_under_evidence():
    with pytest.raises(SystemExit, match="cost probe"):
        tx.main(["--limit", "3", "--out-dir", str(EV / "gap_distribution_v4" / "probe")])


def test_the_shipped_2x2_resummarises_from_its_own_records_and_the_preregistered_ones():
    """The artifact paper 1 Sec. VIII reads: post hoc, outside the preregistered record, both preregistered cells
    reproduced, and every new cell's summary recomputed here from its per-shot records joined, shot by shot, with the
    preregistered run's reference."""
    assert ART.is_file(), "missing: evidence/gap_distribution_v4/posthoc_2x2_2026-11-06/summary.json"
    doc = json.loads(ART.read_text(encoding="utf-8"))
    assert doc["post_hoc"] is True and doc["limit"] is None and "prereg_2026-11-03" not in ART.parent.name
    prereg = json.loads((tx.RUN / "summary.json").read_text(encoding="utf-8"))
    assert doc["preregistered_run"]["head"] == prereg["head"] and doc["preregistered_run"]["argv"] == prereg["argv"]
    by_d = {r["distance"]: r for r in prereg["rows"]}
    assert [r["distance"] for r in doc["rows"]] == [3, 5, 7]
    for row in doc["rows"]:
        pr = by_d[row["distance"]]
        assert row["nontrivial"] == pr["nontrivial"] and row["reproduced_preregistered_cells_on_every_shot"]
        assert row["reproduced_preregistered_summaries"]
        for name in tx.PREREGISTERED:
            assert row["cells"][name] == pr["decoders"][name], name
        body = gzip.decompress((ROOT / row["per_shot"]).read_bytes())
        assert hashlib.sha256(body).hexdigest() == row["per_shot_sha256"]
        mine = [json.loads(x) for x in body.decode().splitlines()]
        ref = tx.read_records(pr)
        assert [m["shot"] for m in mine] == [r["shot"] for r in ref]
        joined = [{**r, **{k: m[k] for k in tx.NEW}} for r, m in zip(ref, mine)]
        for name in tx.NEW:
            assert gd.summarise(joined, name) == row["cells"][name], (row["distance"], name)
        for k, (a, b) in tx.CONTRASTS.items():
            assert row["contrasts"][k] == tx.paired(joined, a, b), (row["distance"], k)
