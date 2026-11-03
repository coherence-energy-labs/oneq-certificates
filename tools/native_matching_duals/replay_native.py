"""Replay the retained native benchmark with the Python standard library only.

Usage: python -I -S tools/native_matching_duals/replay_native.py
No native extension, decoder, optimization package, or ONE_Q import is needed.
Every original metric bound is recomputed; retained verdicts are not trusted.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

# Isolated (-I -S) execution deliberately excludes the script directory.
# Add only this explicitly selected local checker package, never the project.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metric_dual_check import check
from reconcile_original import reconcile


def identity(row):
    values = row["distance"], row["rounds"]
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("invalid benchmark cell identity")
    return values


def indexed(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(name + " must be a nonempty list")
    result = {identity(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate " + name)
    return result


def stats(values):
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in values):
        raise ValueError("invalid observed timing")
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)]}


def check_cell(model, cell, rows, summary):
    """Check completeness and arithmetic, without attesting observed clocks."""
    shots = cell["shots"]
    if type(shots) is not int or shots <= 0:
        raise ValueError("invalid declared shot count")
    if {row["shot"] for row in rows} != set(range(shots)):
        raise ValueError("incomplete native benchmark shot set")
    nontrivial = [row for row in rows if row["syndrome"]]
    measured = [row for row in rows if "ladder_ns" in row]
    if any(not row["syndrome"] or type(row["ladder_accepted"]) is not bool
           for row in measured):
        raise ValueError("invalid ladder observation")
    counts = {
        "nodes": model["nodes"], "edges": len(model["integer_edges"]),
        "nontrivial": len(nontrivial), "native_verified": len(rows),
        "blossom_certificates": sum(bool(row["checked"]["blossom_terms"]) for row in rows),
        "ladder_accepted": sum(row["ladder_accepted"] for row in measured),
        "original_metric_exact": summary["original_exact"],
        "original_metric_within_1e_9": summary["original_within_1e_9"],
        "original_metric_nontrivial_within_1e_9": summary["original_nontrivial_within_1e_9"],
    }
    for key, expected in counts.items():
        if type(cell[key]) is not int or cell[key] != expected:
            raise ValueError("native benchmark count mismatch: " + key)
    graph_hash = hashlib.sha256(json.dumps(model["integer_edges"], separators=(",", ":")).encode()).hexdigest()
    if cell["model_integer_sha256"] != graph_hash:
        raise ValueError("native benchmark integer graph digest mismatch")
    float_costs = [weight for _, _, weight in model["original_edges"]]
    scale = 1.0 if all(float(weight).is_integer() for weight in float_costs) else ((1 << 24) - 1) / max(float_costs)
    expected_edges = [[u, v, 2 * math.floor(weight * scale + .5)]
                      for u, v, weight in model["original_edges"]]
    if cell["scale_float"] != scale or model["integer_edges"] != expected_edges:
        raise ValueError("declared input quantization mismatch")
    timings = {
        "native_solve_ns": [row["proof"]["native_solve_ns"] for row in nontrivial],
        "native_snapshot_ns": [row["proof"]["native_snapshot_ns"] for row in nontrivial],
        "native_export_wrapper_ns": [row["export_native_wrapper_ns"] for row in nontrivial],
        "independent_checker_ns": [row["independent_checker_ns"] for row in nontrivial],
        "same_shot_native_plus_check_ns": [row["export_native_wrapper_ns"] + row["independent_checker_ns"] for row in measured],
        "same_shot_ladder_ns": [row["ladder_ns"] for row in measured],
        "stock_pymatching_pairs_ns": [row["stock_pymatching_pairs_ns"] for row in nontrivial],
        "native_decode_noexport_wrapper_ns": [row["baseline_native_wrapper_ns"] for row in nontrivial],
        "original_metric_reconcile_ns": [row["original_reconcile_ns"] for row in nontrivial],
    }
    for key, values in timings.items():
        if cell[key] != stats(values):
            raise ValueError("native benchmark timing aggregation mismatch: " + key)
    epsilons = [float(Fraction(*row["original_epsilon_bound"])) for row in rows]
    values = {"max_original_epsilon_bound": max(epsilons),
              "mean_original_epsilon_bound": statistics.mean(epsilons),
              "original_metric_epsilon_max": float(summary["max_epsilon"])}
    for key, expected in values.items():
        if cell[key] != expected:
            raise ValueError("native benchmark bound aggregation mismatch: " + key)


def replay(packet):
    if not isinstance(packet, dict) or packet.get("schema") != "oneq.native_metric_dual_benchmark.v1":
        raise ValueError("unknown native benchmark schema")
    models = indexed(packet["models"], "graph model")
    declared = indexed(packet["cells"], "benchmark cell")
    if set(models) != set(declared):
        raise ValueError("benchmark cell and graph model sets differ")
    if not isinstance(packet["records"], list):
        raise ValueError("benchmark records must be a list")
    grouped = {key: [] for key in models}
    summary = {key: {"distance": key[0], "rounds": key[1], "checked": 0,
                     "nontrivial": 0, "original_exact": 0, "original_within_1e_9": 0,
                     "original_nontrivial_within_1e_9": 0, "max_epsilon": Fraction()}
               for key in models}
    seen = set()
    for row in packet["records"]:
        key = identity(row)
        if key not in models or type(row["shot"]) is not int or row["shot"] < 0:
            raise ValueError("invalid shot identity")
        shot_identity = (*key, row["shot"])
        if shot_identity in seen:
            raise ValueError("duplicate shot identity")
        seen.add(shot_identity)
        grouped[key].append(row)
        model, cell = models[key], summary[key]
        result = check(model["nodes"], model["integer_edges"], row["syndrome"], row["proof"])
        if not result["accepted"] or result != row["checked"]:
            raise ValueError("native integer proof refused or retained result differs")
        bounds = reconcile(model["nodes"], model["original_edges"], row["syndrome"], row["proof"], result)
        if bounds != row["original_metric_reconciliation"]:
            raise ValueError("original IEEE rational bound differs from retained result")
        if row["original_exact_cost"] != bounds["primal"]:
            raise ValueError("original correction cost differs from checked cost")
        original = {(min(u, v), max(u, v)): Fraction(weight) for u, v, weight in model["original_edges"]}
        alpha = min(original[(min(u, v), max(u, v))] / weight
                    for u, v, weight in model["integer_edges"] if weight > 0)
        edge_epsilon = Fraction(*bounds["primal"]) - alpha * result["dual_bound"]
        if row["original_epsilon_bound"] != [edge_epsilon.numerator, edge_epsilon.denominator]:
            raise ValueError("original edgewise bound differs from retained result")
        cell["checked"] += 1
        cell["nontrivial"] += bool(row["syndrome"])
        cell["original_exact"] += bounds["exact_optimal"]
        cell["original_within_1e_9"] += bounds["within_1e_9"]
        cell["original_nontrivial_within_1e_9"] += bool(row["syndrome"]) and bounds["within_1e_9"]
        cell["max_epsilon"] = max(cell["max_epsilon"], Fraction(*bounds["epsilon"]))
    cells = []
    for key, cell in summary.items():
        check_cell(models[key], declared[key], grouped[key], cell)
        epsilon = cell.pop("max_epsilon")
        cell["max_epsilon"] = [epsilon.numerator, epsilon.denominator]
        cells.append(cell)
    return {"status": "PASS", "checker_dependencies": "Python standard library only",
            "proofs_checked": len(seen), "cells": cells,
            "complete_declared_shot_sets": True,
            "retained_aggregates_checked": True,
            "timings_reproduced": False,
            "physical_correctness_proven": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", nargs="?", type=Path, default=HERE / "benchmark_final.json")
    args = parser.parse_args()
    raw = args.packet.read_bytes()
    result = replay(json.loads(raw))
    result["packet_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))
