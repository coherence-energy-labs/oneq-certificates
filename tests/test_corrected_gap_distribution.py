"""Regression, exact classification and adversarial replay of graph gaps."""
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap = load("corrected_gap_distribution", "experiments/m0_prior_art/gap_distribution.py")
replay = load("independent_corrected_gap_replay", "tools/replay_corrected_gaps.py")


def test_frozen_nine_copies_counterexample_is_exactly_optimal():
    from oneq.matching_cert import MatchingCertificate, check
    data = json.loads((ROOT / "tests/fixtures/gap_counterexample_20261103.json").read_text())
    mechs = [(tuple(row["edge"]), float.fromhex(row["graph_weight_hex"]),
              float.fromhex(row["probability_hex"])) for row in data["component_columns"]]
    selection = np.zeros(len(mechs), dtype=np.uint8)
    selection[data["selected_component_indices"]] = 1
    odd, cost = gap.odd_and_weight(mechs, selection)
    assert odd == {7, 9, 10}
    keys, _, _ = gap.component_projection(mechs, selection)
    assert keys == ((7, 10), (9, -1))
    assert float(cost) == 7.543721817379714
    duplicate_cost = sum((Fraction.from_float(mechs[j][1]) for j in data["selected_component_indices"]), Fraction())
    assert float(duplicate_cost - cost) == 36.81846169007079
    c = data["exact_certificate"]
    cert = MatchingCertificate(
        matched_edges=keys,
        node_potentials={r["vertex"]: float.fromhex(r["value_hex"]) for r in c["node_potentials"]},
        blossom_duals={frozenset(r["set"]): float.fromhex(r["value_hex"]) for r in c["blossom_duals"]})
    edges = [(r["edge"][0], r["edge"][1], float.fromhex(r["weight_hex"])) for r in data["graph_edges"]]
    assert check(edges=edges, syndrome=data["syndrome"], cert=cert).accepted
    assert cost == sum((Fraction.from_float(dict(((u, v), w) for u, v, w in edges)[edge])
                        for edge in keys), Fraction())


def test_every_small_component_selection_matches_direct_gf2_projection():
    mechs = [((0, 1), 0.1, 0.01), ((1, 0), 0.1, 0.02),
             ((0, 1), 0.1, 0.03), ((1, -1), 0.3, 0.01),
             ((-1, 1), 0.3, 0.02), ((0, -1), 0.0, 0.01)]
    graph_weights = {(0, 1): Fraction.from_float(0.1), (1, -1): Fraction.from_float(0.3),
                     (0, -1): Fraction()}
    incidence = np.asarray([[1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 0]], dtype=np.uint8)
    for bits in itertools.product((0, 1), repeat=6):
        keys, odd, cost = gap.component_projection(mechs, bits)
        expected = {(0, 1): (bits[0] + bits[1] + bits[2]) % 2,
                    (1, -1): (bits[3] + bits[4]) % 2, (0, -1): bits[5]}
        assert set(keys) == {edge for edge, selected in expected.items() if selected}
        assert cost == sum((graph_weights[edge] * selected for edge, selected in expected.items()), Fraction())
        assert odd == set(map(int, np.flatnonzero((incidence @ bits) % 2)))


@pytest.mark.parametrize("selection", [[0.1], [2], [float("nan")], [1, 0], [[1]]])
def test_lossy_or_wrong_shape_selections_are_refused(selection):
    with pytest.raises(ValueError):
        gap.odd_and_weight([((0, -1), 1.0, 0.1)], selection)


def test_unknown_edge_cannot_be_hidden_by_cancellation():
    with pytest.raises(ValueError, match="absent"):
        gap.graph_projection([(7, 8), (7, 8)], {(0, -1): Fraction(1)})


def test_duplicate_columns_must_agree_on_weight():
    with pytest.raises(ValueError, match="disagree"):
        gap.odd_and_weight([((0, 1), 1.0, 0.1), ((1, 0), 2.0, 0.1)], [1, 1])


def test_no_float_tolerance_in_zero_or_negative_gap_count():
    summary = gap.summarise([Fraction(), Fraction(1, 10**20), Fraction(-1, 10**20)])
    assert summary["zero_count_exact"] == 1
    assert summary["negative_count_exact"] == 1
    assert summary["frac_zero"] == 1 / 3


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    directory = tmp_path_factory.mktemp("corrected_gap")
    output = directory / "summary.json"
    result = subprocess.run([sys.executable, str(ROOT / "experiments/m0_prior_art/gap_distribution.py"),
                             "--shots", "8", "--train-shots", "8", "--distances", "3",
                             "--no-belief-matching", "--out", str(output)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert replay.verify(output)["accepted"]
    return output


def test_real_seeded_end_to_end_evidence_replays(measured):
    result = replay.verify(measured)
    assert result["families"] == 1
    assert result["reference_witnesses_rechecked"] > 0
    source = json.loads(measured.read_text())
    row = source["rows"][0]
    assert row["shots"] == row["trivial_excluded"] + row["nontrivial"]
    assert "bp_osd_train_selected" in row["decoders"]
    assert row["training"]["seed"] != source["protocol"]["heldout_seed"]


@pytest.mark.parametrize("attack", ["cost", "projection", "reference", "binding", "denominator", "zeros", "training"])
def test_resealed_semantic_tampering_is_refused(measured, tmp_path, attack):
    summary = json.loads(measured.read_text())
    compressed = (measured.parent / summary["trace"]["path"]).read_bytes()
    records = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
    shot = next(r for r in records if r["phase"] == "HELDOUT" and r["syndrome"])
    proposal = shot["candidates"]["bp_osd_train_selected"]
    if attack == "cost":
        proposal["weight_exact"]["num"] = str(int(proposal["weight_exact"]["num"]) + 1)
    elif attack == "projection":
        proposal["projected_edges"] = []
    elif attack == "reference":
        shot["reference"]["accepted_exact"] = not shot["reference"]["accepted_exact"]
    elif attack == "binding":
        shot["graph_sha256"] = "0" * 64
    elif attack == "denominator":
        summary["rows"][0]["nontrivial"] += 1
    elif attack == "zeros":
        summary["rows"][0]["decoders"]["bp_osd_train_selected"]["gap_vs_proven_optimum"]["zero_count_exact"] += 1
    else:
        summary["rows"][0]["training"]["selected_scale"] = 0.9
    modified = gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode(), mtime=0)
    summary["trace"]["sha256"] = hashlib.sha256(modified).hexdigest()
    (tmp_path / summary["trace"]["path"]).write_bytes(modified)
    output = tmp_path / "summary.json"
    output.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        replay.verify(output)


def test_historical_artifact_destination_is_refused_before_work():
    with pytest.raises(SystemExit):
        gap.main(["--out", str(ROOT / "evidence/gap_distribution.json")])
