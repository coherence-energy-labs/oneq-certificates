r"""Paper 3's first independent review (2026-10-29, docs/reviews/PAPER3_REFEREE_2026-10-29.md): the two pieces of NEW
evidence it led to, each held both ways.

* `dem_independent_morph2026.py` -- the fault set of every 2026 morphing model the campaign ran, derived without stim's
  error analysis: the committed record covers EXACTLY the campaign's models, is equal on every one, matches a fresh
  run on sampled cells, and the comparison disagrees with a mutated circuit (it can fail).
* `ibm_recert_models.py` -- every model shipped with the IBM re-certification is byte-identical to a fresh export of
  the cycle: the committed record says so for all 22, a fresh re-export of one cell agrees, and a tampered shipped
  model is caught.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import stim

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "experiments" / "circuit_distance_exact"
RECORDS = ROOT / "evidence" / "circuit_distance_exact" / "records"
sys.path.insert(0, str(LANE))
import dem_independent as di  # noqa: E402
import dem_independent_morph2026 as d26  # noqa: E402
import ibm_recert_models as irm  # noqa: E402
import morph2026 as m26  # noqa: E402


def _rows(name: str) -> list[dict]:
    return [json.loads(x) for x in (RECORDS / name).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- the 2026 circuits' independent DEMs

def test_the_2026_record_covers_exactly_the_campaigns_models_and_is_equal_on_every_one():
    rows = _rows("dem_independent_morph2026.jsonl")
    keys = [(r["target"], r["basis"], r["rounds"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(d26.campaign_models())
    assert set(keys) == set(d26.campaign_models())
    assert {k[0] for k in keys} == set(m26.TARGETS)
    assert all(r["equal"] and r["only_independent"] == 0 and r["only_stim"] == 0 for r in rows)


def test_a_window_rows_second_model_is_in_the_enumeration():
    """A window row compares T with T+1, so the model one round past the last search is a model the claims use."""
    camp = [json.loads(x) for x in (RECORDS / "morph2026_campaign.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    last_window = max(r["T"] for r in camp if r.get("kind") == "window" and r["code"] == "c42a" and r["basis"] == "Z")
    last_search = max(r["rounds"] for r in camp if r.get("kind") == "search" and r["code"] == "c42a"
                      and r["basis"] == "Z")
    assert last_window + 1 > last_search
    assert ("c42a", "Z", last_window + 1) in set(d26.campaign_models())


def test_the_2026_record_matches_a_fresh_run_on_sampled_cells():
    rec = {(r["target"], r["basis"], r["rounds"]): r for r in _rows("dem_independent_morph2026.jsonl")}
    for key in (("t30a", "X", 1), ("c42a", "Z", 2), ("c48c2", "X", 3)):
        name, basis, rounds = key
        fresh = di.compare(m26.build(m26.spec(name), basis, rounds))
        assert fresh["equal"]
        assert fresh["mechanisms"] == rec[key]["mechanisms"] == rec[key]["stim_mechanisms"]


def test_the_2026_comparison_disagrees_with_a_mutated_circuit():
    """CAN-FAIL: the independent pass on a copy with one two-qubit noise line dropped must disagree with stim on the
    original. The position is measured, not arbitrary: the fault set is redundant at the boundaries. For c42a,
    memory-Z, T = 2, dropping each of the 12 DEPOLARIZE2 lines removes 0, 0, 105, 168, 63, 0, 0, 63, 168, 105, 0, 0
    mechanisms, so the FIRST line would be a dark control; the third is used."""
    c = m26.build(m26.spec("c42a"), "Z", 2)
    lines = str(c).splitlines()
    k = [i for i, s in enumerate(lines) if s.startswith("DEPOLARIZE2")][2]
    mutated = stim.Circuit("\n".join(lines[:k] + lines[k + 1:]))
    assert di.pauli_dem(mutated)[0] != di.stim_dem(c)[0]
    assert di.pauli_dem(c)[0] == di.stim_dem(c)[0]


# ---------------------------------------------------------------- the IBM re-certification's models

def test_every_shipped_ibm_model_was_reexported_and_is_identical():
    manifest = json.loads((irm.PKG / "manifest.json").read_text(encoding="utf-8"))["observations"]
    rows = _rows("ibm_recert_reexport.jsonl")
    assert sorted((r["basis"], r["rounds"], r["shipped_sha256"]) for r in rows) == \
        sorted((o["basis"], o["rounds"], o["model_sha256"]) for o in manifest)
    assert len(rows) == 22 and all(r["equal"] and r["reexport_sha256"] == r["shipped_sha256"] for r in rows)


def test_a_fresh_reexport_agrees_and_a_tampered_shipped_model_is_caught(tmp_path, monkeypatch):
    manifest = json.loads((irm.PKG / "manifest.json").read_text(encoding="utf-8"))["observations"]
    o = next(x for x in manifest if x["basis"] == "Z" and x["rounds"] == 1)
    assert irm.reexport(o, tmp_path)["equal"]
    fake = tmp_path / "pkg"
    shutil.copytree(irm.PKG / "models", fake / "models")
    m = fake / o["model"]
    lines = m.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]          # two mechanisms swapped: the same set, different bytes
    m.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(irm, "PKG", fake)
    assert not irm.reexport(o, tmp_path)["equal"]
