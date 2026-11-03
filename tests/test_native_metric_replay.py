"""Complete-set and adversarial checks for the portable native benchmark."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.native_matching_duals.metric_dual_check import check
from tools.native_matching_duals.replay_native import replay

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools/native_matching_duals"


@pytest.fixture(scope="module")
def packet():
    return json.loads((HERE / "benchmark_final.json").read_text())


@pytest.fixture
def one_cell(packet):
    result = deepcopy(packet)
    result["models"] = result["models"][:1]
    result["cells"] = result["cells"][:1]
    result["records"] = [row for row in result["records"] if row["distance"] == 3]
    return result


def test_all_retained_integer_and_original_bounds_replay_without_dependencies():
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(HERE / "replay_native.py")],
        cwd=ROOT, stdin=subprocess.DEVNULL, text=True, capture_output=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["proofs_checked"] == 256
    assert result["complete_declared_shot_sets"]
    assert result["retained_aggregates_checked"]
    assert not result["timings_reproduced"]
    assert not result["physical_correctness_proven"]
    assert [cell["original_within_1e_9"] for cell in result["cells"]] == [60, 26, 4, 0]
    assert [cell["original_nontrivial_within_1e_9"] for cell in result["cells"]] == [19, 15, 3, 0]


@pytest.mark.parametrize("attack", [
    "omit_shot", "omit_all_shots", "duplicate_shot", "rename_shot", "omit_cell",
    "fake_denominator", "fake_acceptance", "fake_timing", "fake_graph_digest",
    "fake_original_cost", "fake_original_bound", "fake_integer_proof",
])
def test_tampered_or_incomplete_packet_is_refused(one_cell, attack):
    data = one_cell
    if attack == "omit_shot":
        data["records"].pop()
    elif attack == "omit_all_shots":
        data["records"] = []
    elif attack == "duplicate_shot":
        data["records"].append(deepcopy(data["records"][0]))
    elif attack == "rename_shot":
        data["records"][-1]["shot"] += 1
    elif attack == "omit_cell":
        data["cells"] = []
    elif attack == "fake_denominator":
        data["cells"][0]["shots"] -= 1
    elif attack == "fake_acceptance":
        data["cells"][0]["original_metric_within_1e_9"] += 1
    elif attack == "fake_timing":
        data["cells"][0]["same_shot_native_plus_check_ns"]["median"] += 1
    elif attack == "fake_graph_digest":
        data["cells"][0]["model_integer_sha256"] = "0" * 64
    elif attack == "fake_original_cost":
        data["records"][0]["original_exact_cost"][0] += 1
    elif attack == "fake_original_bound":
        data["records"][0]["original_epsilon_bound"][0] += 1
    elif attack == "fake_integer_proof":
        row = next(row for row in data["records"] if row["proof"]["dual"])
        row["proof"]["dual"][0]["value"] += 1
    with pytest.raises(ValueError):
        replay(data)


@pytest.mark.parametrize("bad", [None, [], True])
def test_top_level_nonobject_native_proofs_are_refused(packet, bad):
    model, row = packet["models"][0], packet["records"][0]
    result = check(model["nodes"], model["integer_edges"], row["syndrome"], bad)
    assert result["accepted"] is False
