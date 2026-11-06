"""The held-out d=7 partial-sample report (experiments/m0_prior_art/heldout_d7_partial.py).

It must (1) count only chunks whose receipts hash to their index, (2) use the preregistration's own statistics,
(3) give a worst-case view that counts every unfinished nontrivial shot as not proven, (4) refuse to read a run that
was not stopped, and (5) read nontrivial counts from the corpus correctly -- checked against the released development
receipts, which cover the same shots.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

import heldout_d7_partial as hp                                    # noqa: E402


def _chunk(root: pathlib.Path, arm: str, seed: int, status: dict, forg_acc: int = 0, corrupt: bool = False):
    body = (json.dumps({"seed": seed, "status": status}) + "\n").encode()
    rp = root / "data" / "epsilon_receipts" / arm / f"chunk_{seed:06d}.jsonl.gz"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_bytes(gzip.compress(body + (b"x" if corrupt else b""), 6, mtime=0))
    rec = {"seed": seed, "shots": 50, "status": status, "epsilon_max_accepted": "1/1000000000",
           "forgeries": {"D": 1}, "forgeries_accepted": {"D": forg_acc}, "battery_not_false": 0,
           "receipts_file": rp.relative_to(root).as_posix(), "receipts_sha256": hashlib.sha256(body).hexdigest(),
           "seconds": 1.0}
    idx = root / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    with open(idx, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _stop(arm, unfinished):
    return {"schema": "oneq-heldout-d7-stop/1", "current_arm": arm, "dispatched_unfinished_seeds": unfinished}


def test_the_amended_order_stop_record_names_unfinished_chunks_per_arm(tmp_path):
    """Deviation 1 (the arms interleaved): the watchdog's /2 record lists the chunks being computed per arm."""
    _chunk(tmp_path, "heldout_d7_X_r50", 1, {"EPSILON_OPTIMAL": 50})
    _chunk(tmp_path, "heldout_d7_Z_r10", 1, {"TRIVIAL": 10, "EPSILON_OPTIMAL": 40})
    stop = {"schema": "oneq-heldout-d7-stop/2",
            "dispatched_unfinished": {"heldout_d7_X_r50": [2], "samedev_d7_Z_r30": [1], "heldout_d7_Z_r10": [1]}}
    doc = hp.analyse(tmp_path, stop, nontrivial_of=lambda a, s: 50 * len(s))
    assert doc["arms"]["heldout_d7_X_r50"]["dispatched_worst_case"]["nontrivial"] == 100
    assert doc["arms"]["samedev_d7_Z_r30"]["dispatched_worst_case"] == {
        "chunks": 1, "nontrivial": 50, "not_proven_or_unfinished": 50, "upper_95_one_sided": 1.0}
    # a chunk that completed is never also counted as unfinished
    assert doc["arms"]["heldout_d7_Z_r10"]["dispatched_unfinished_seeds"] == []
    assert doc["pooled"]["r10"]["nontrivial"] == 40 and doc["pooled"]["r30"]["nontrivial"] == 0
    with pytest.raises(SystemExit, match="not a held-out d7 stop record"):
        hp.analyse(tmp_path, {"schema": "something/9"})


def test_partial_report_counts_verified_chunks_and_bounds_the_unfinished_ones(tmp_path):
    arm = "heldout_d7_X_r50"
    _chunk(tmp_path, arm, 1, {"TRIVIAL": 2, "EPSILON_OPTIMAL": 48})
    _chunk(tmp_path, arm, 2, {"TRIVIAL": 1, "EPSILON_OPTIMAL": 47, "NOT_PROVEN": 2})
    _chunk(tmp_path, arm, 4, {"EXACT_OPTIMAL": 1, "EPSILON_OPTIMAL": 49})
    _chunk(tmp_path, arm, 6, {"EPSILON_OPTIMAL": 50}, corrupt=True)        # receipts do not hash: not counted
    doc = hp.analyse(tmp_path, _stop(arm, [3, 5, 6]), nontrivial_of=lambda a, s: 49 * len(s))
    a = doc["arms"][arm]
    assert a["completed_seeds"] == [1, 2, 4]
    c = a["completed"]
    assert (c["nontrivial"], c["accepted"], c["not_proven"], c["exact_optimal"]) == (147, 145, 2, 1)
    assert a["prefix"]["chunks"] == 2 and a["prefix"]["nontrivial"] == 97
    assert a["dispatched_unfinished_seeds"] == [3, 5, 6] and a["dispatched_unfinished_nontrivial"] == 147
    w = a["dispatched_worst_case"]
    assert (w["nontrivial"], w["not_proven_or_unfinished"]) == (294, 149)
    assert w["upper_95_one_sided"] > c["not_proven_upper_95_one_sided"]
    # no development arm has data: the preregistered test is not computed, and says why
    assert not doc["tests"]["heldout_r50_vs_same_producer_dev_r30"]["computed"]
    assert doc["pooled"]["r50"]["arms_with_data"] == [arm]
    assert doc["forgeries_accepted_total"] == 0


def test_partial_report_uses_the_preregistered_statistics(tmp_path):
    import heldout_willow_d7 as hw                                  # the committed runner's own functions
    _chunk(tmp_path, "heldout_d7_X_r50", 1, {"EPSILON_OPTIMAL": 47, "NOT_PROVEN": 3})
    _chunk(tmp_path, "samedev_d7_X_r30", 1, {"TRIVIAL": 5, "EPSILON_OPTIMAL": 44, "NOT_PROVEN": 1})
    doc = hp.analyse(tmp_path, _stop("heldout_d7_X_r10", []), nontrivial_of=lambda a, s: 0)
    p = doc["tests"]["heldout_r50_vs_same_producer_dev_r30"]["p_one_sided"]
    assert p == hw.conditional_test(3, 50, 1, 45)
    assert doc["pooled"]["r50"]["not_proven_upper_95_one_sided"] == hw.cp_upper_one_sided(3, 50)
    assert doc["pooled"]["r30"]["not_proven_upper_95_one_sided"] == hw.cp_upper_one_sided(1, 45)


def _records(root, stop):
    """A run-records directory as the watchdog and the amendment stop leave it (the report refuses without one)."""
    rec = root / "records"
    (rec / "dispatch").mkdir(parents=True)
    (rec / "d7_stop_record.json").write_text(json.dumps(stop), encoding="utf-8")
    (rec / "d7_amendment_stop_record.json").write_text(json.dumps(
        {"stopped": "t", "core_hours_spent": 5.1, "completed_seeds": {}, "in_flight_seeds_lost": [2]}), encoding="utf-8")
    for nm in ("d7_watchdog.log", "d7_babysitter.log", "d7_run.log", "d7_run_interleaved.log"):
        (rec / nm).write_text("log\n", encoding="utf-8")
    # the producer and deviation files the head hashes, read from the run's tree
    for p in hp.PRODUCER_FILES + hp.DEVIATION_FILES:
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(f"# {p}\n", encoding="utf-8")
    return rec


def test_an_accepted_forgery_fails_the_report(tmp_path):
    _chunk(tmp_path, "heldout_d7_X_r50", 1, {"EPSILON_OPTIMAL": 50}, forg_acc=1)
    stop = _stop("heldout_d7_X_r50", [])
    rec = _records(tmp_path, stop)
    rc = hp.main(["--run-root", str(tmp_path), "--stop-record", str(rec / "d7_stop_record.json"), "--records",
                  str(rec), "--out", str(tmp_path / "o.json")])
    assert rc == 1
    doc = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert doc["head"]["records"]["amendment_stop"]["in_flight_seeds_lost"] == [2]
    assert doc["recorded_development_2026_09_17"]["not_proven"] == 2


def test_the_report_refuses_a_copy_that_differs_from_the_run(tmp_path):
    """When the sample is reported from a copy, every index line, receipt and instance must be the run's bytes."""
    run, copy = tmp_path / "run", tmp_path / "copy"
    for root in (run, copy):
        _chunk(root, "heldout_d7_X_r50", 1, {"EPSILON_OPTIMAL": 50})
        (root / "evidence" / "instances").mkdir(parents=True, exist_ok=True)
        for arm in hp.ARMS:
            (root / "evidence" / "instances" / f"matching_{arm}.json").write_text("{}", encoding="utf-8")
            (root / "evidence" / "epsilon_rerun" / arm).mkdir(parents=True, exist_ok=True)
            p = root / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
            if not p.exists():
                p.write_text("", encoding="utf-8")
    stop = _stop("heldout_d7_X_r50", [])
    rec = _records(run, stop)
    assert hp.head(copy, run, rec, rec / "d7_stop_record.json")["copied_from"]["index_receipts_and_instances_identical"]
    (copy / "evidence" / "instances" / "matching_heldout_d7_Z_r10.json").write_text("{ }", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the run"):
        hp.head(copy, run, rec, rec / "d7_stop_record.json")
    (rec / "d7_watchdog.log").unlink()
    with pytest.raises(SystemExit, match="missing"):
        hp.head(run, None, rec, rec / "d7_stop_record.json")


def test_the_real_report_pins_the_producer_and_its_records():
    """The record of the stopped run: its producer bytes are 2008c81f's, the deviation's are 4030f88d's, the run
    records it names are on disk unchanged, and its sample is the one replayed (the generator repeats these checks)."""
    import hashlib
    import subprocess
    p = ROOT / "evidence" / "epsilon_rerun" / "heldout_d7_partial.json"
    if not p.is_file():
        pytest.skip("the held-out d=7 report is not here")
    doc = json.loads(p.read_text(encoding="utf-8"))
    h = doc["head"]
    if (ROOT / ".git").exists():
        for commit, files in ((h["producer"]["commit"], h["producer"]["sha256"]),
                              (h["deviation"]["commit"], h["deviation"]["sha256"])):
            for path, sha in files.items():
                blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True).stdout
                assert hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == sha, (commit, path)
    else:                         # a reader's extracted bundle: the pins to 2008c81f / 4030f88d cannot be read
        import warnings
        warnings.warn("no repository here: the producer's and deviation's commit-pinned bytes were not compared")
    # the deviation's driver and watchdog ship as the bytes that ran: those are checked without git
    for path, sha in h["deviation"]["sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == sha, path
    rec = ROOT / h["records"]["dir"]
    for nm, sha in h["records"]["sha256"].items():
        assert hashlib.sha256((rec / nm).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == sha, nm
    assert json.loads((rec / "d7_stop_record.json").read_text(encoding="utf-8")) == doc["stop_record"]
    assert doc["status"] == "STOPPED_BY_RULE" and doc["stop_record"]["core_hours"] == 40.0
    for arm, v in doc["arms"].items():
        # the shipped index of each arm lists exactly the chunks the stopped run completed; the receipts themselves
        # are release assets a reader unpacks arm by arm, so they are hash-checked only where they are on disk
        # (2026-11-05: the first version needed every arm's receipts and failed the reader who had unpacked one)
        idx = ROOT / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
        rows = [json.loads(x) for x in idx.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert v["completed_seeds"] == sorted(r["seed"] for r in rows), arm
        on_disk = hp.completed_chunks(ROOT, arm)
        if on_disk:
            assert sorted(on_disk) == v["completed_seeds"], arm
    assert h["copied_from"]["index_receipts_and_instances_identical"] is True


def test_a_run_that_was_not_stopped_is_not_read(tmp_path):
    with pytest.raises(SystemExit, match="STOPPED run only"):
        hp.main(["--run-root", str(tmp_path), "--stop-record", str(tmp_path / "absent.json")])
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a held-out d7 stop record"):
        hp.main(["--run-root", str(tmp_path), "--stop-record", str(bad)])


def test_the_shipped_report_is_the_arithmetic_of_the_shipped_arm_records():
    """On bytes a reader has (the arm indices, the stop record, the report; no corpus): every completed view is the
    sum of its index lines, the worst case adds the recorded nontrivial count of each unfinished chunk as not proven,
    the prefix is the longest run from seed 1, and the stop record's seeds are the index's."""
    rep = ROOT / "evidence" / "epsilon_rerun" / "heldout_d7_partial.json"
    stop_p = ROOT / "evidence" / "epsilon_rerun" / "heldout_d7_stopped_2026-11-04" / "d7_stop_record.json"
    assert rep.is_file() and stop_p.is_file(), "missing: evidence/epsilon_rerun/heldout_d7_partial.json or the stop record"
    doc = json.loads(rep.read_text(encoding="utf-8"))
    stop = json.loads(stop_p.read_text(encoding="utf-8"))
    for arm, v in doc["arms"].items():
        idx = ROOT / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
        assert idx.is_file(), f"missing: {idx.relative_to(ROOT).as_posix()}"
        rows = {r["seed"]: r for r in (json.loads(x) for x in idx.read_text(encoding="utf-8").splitlines()
                                       if x.strip())}
        assert sorted(rows) == v["completed_seeds"] == stop["completed_seeds"].get(arm, [])
        c = v["completed"]
        tot = {}
        for r in rows.values():
            for k, n in r["status"].items():
                tot[k] = tot.get(k, 0) + n
        assert c["nontrivial"] == sum(n for k, n in tot.items() if k != "TRIVIAL")
        assert c["accepted"] == tot.get("EXACT_OPTIMAL", 0) + tot.get("EPSILON_OPTIMAL", 0)
        assert c["not_proven"] == tot.get("NOT_PROVEN", 0) and c["exact_optimal"] == tot.get("EXACT_OPTIMAL", 0)
        assert c["forgeries_accepted"] == sum(sum(r["forgeries_accepted"].values()) for r in rows.values()) == 0
        assert c["not_proven_upper_95_one_sided"] == hp.cp_upper_one_sided(c["not_proven"], c["nontrivial"])
        unf = stop["dispatched_unfinished"].get(arm, [])
        assert v["dispatched_unfinished_seeds"] == unf and not set(unf) & set(rows)
        assert 0 < v["dispatched_unfinished_nontrivial"] <= 50 * len(unf) if unf else \
            v["dispatched_unfinished_nontrivial"] == 0
        w = v["dispatched_worst_case"]
        assert w["nontrivial"] == c["nontrivial"] + v["dispatched_unfinished_nontrivial"]
        assert w["not_proven_or_unfinished"] == c["not_proven"] + v["dispatched_unfinished_nontrivial"]
        assert w["upper_95_one_sided"] == hp.cp_upper_one_sided(w["not_proven_or_unfinished"], w["nontrivial"])
        k = 0
        while (k + 1) in rows:
            k += 1
        assert v["prefix"]["chunks"] == k
    for rr_, row in doc["pooled"].items():
        sel = [v for v in doc["arms"].values() if v["rounds"] == rr_]
        assert row["nontrivial"] == sum(v["completed"]["nontrivial"] for v in sel)
        assert row["not_proven"] == sum(v["completed"]["not_proven"] for v in sel)
        assert row["not_proven_upper_95_one_sided"] == hp.cp_upper_one_sided(row["not_proven"], row["nontrivial"])
