#!/usr/bin/env python3
"""Replay corrected graph-gap evidence without Stim, BP-OSD, or the producer.

Independently derives XOR corrections, rational costs, training selection,
reference proof verdicts, exact classifications and aggregate observations.
Uses the existing mathematical checkers only for retained cut witnesses.
Timing values are observations checked for internal consistency, not attested
performance. Replay does not establish that a declared decoder produced an
answer or that the recorded syndromes came from physical hardware.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oneq.matching_cert import MatchingCertificate, check  # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon  # noqa: E402


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def rational(record):
    require(set(record) == {"num", "den"}, "invalid rational fields")
    numerator, denominator = int(record["num"]), int(record["den"])
    require(denominator > 0, "nonpositive rational denominator")
    return Fraction(numerator, denominator)


def key(edge):
    require(len(edge) == 2 and all(type(x) is int for x in edge), "noninteger edge")
    left, right = edge
    require(left != right, "self-loop")
    if left == -1:
        return right, left
    if right == -1:
        return left, right
    return min(left, right), max(left, right)


def xor_edges(edges, weights):
    parity = {}
    for edge in edges:
        normalized = key(edge)
        require(normalized in weights, "unknown edge")
        parity[normalized] = 1 - parity.get(normalized, 0)
    result = sorted(edge for edge, odd in parity.items() if odd)
    syndrome = set()
    for edge in result:
        for vertex in edge:
            if vertex != -1:
                if vertex in syndrome:
                    syndrome.remove(vertex)
                else:
                    syndrome.add(vertex)
    return result, syndrome, sum((weights[edge] for edge in result), Fraction())


def candidate(record, model, syndrome):
    status = record["status"]
    require(status in {"ERROR", "VALID", "SYNDROME_INCONSISTENT"}, "unknown candidate status")
    for field in ["decode_ns", "projection_ns"]:
        require(type(record[field]) is int and record[field] >= 0, "invalid timing")
    if status == "ERROR":
        require(bool(record.get("error_type")), "error omitted its type")
        return None
    weights, columns, _edges = model
    indices = record["selected_component_indices"]
    if indices is not None:
        require(len(set(indices)) == len(indices), "repeated component index")
        require(all(type(i) is int and 0 <= i < len(columns) for i in indices), "invalid component index")
        edges, odd, cost = xor_edges([columns[i] for i in indices], weights)
        require([list(edge) for edge in edges] == record["projected_edges"], "component projection mismatch")
    else:
        edges, odd, cost = xor_edges(record["projected_edges"], weights)
        require([list(edge) for edge in edges] == record["projected_edges"], "noncanonical edge set")
    require(cost == rational(record["weight_exact"]), "candidate weight mismatch")
    consistent = odd == set(syndrome)
    require((status == "VALID") == consistent, "syndrome status mismatch")
    return cost if consistent else None


def percentile(values, percentage):
    values = sorted(float(v) for v in values)
    position = (len(values) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def approximate_equal(got, expected, label):
    require(math.isfinite(got) and math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-15), label)


def verify_summary(summary, values):
    if not values:
        require(summary is None, "empty summary is not null")
        return
    n = len(values)
    zeros, negatives = sum(v == 0 for v in values), sum(v < 0 for v in values)
    require(summary["n"] == n, "summary count mismatch")
    require(summary["zero_count_exact"] == zeros, "exact zero count mismatch")
    require(summary["negative_count_exact"] == negatives == 0, "negative gap")
    require(rational(summary["sum_exact"]) == sum(values, Fraction()), "exact sum mismatch")
    require(rational(summary["max_exact"]) == max(values), "exact maximum mismatch")
    for field, expected in {"mean": float(sum(values, Fraction()) / n),
                            "median": percentile(values, 50), "p90": percentile(values, 90),
                            "p99": percentile(values, 99), "max": float(max(values)),
                            "frac_zero": zeros / n, "frac_below_zero": negatives / n}.items():
        approximate_equal(summary[field], expected, f"summary {field} mismatch")


def verify_timing(summary, values):
    if not values:
        require(summary is None, "empty timing summary is not null")
        return
    require(summary["count"] == len(values) and summary["total_ns"] == sum(values), "timing totals mismatch")
    for field, expected in {"mean_ms": sum(values) / len(values) / 1e6,
                            "p50_ms": percentile(values, 50) / 1e6,
                            "p95_ms": percentile(values, 95) / 1e6,
                            "max_ms": max(values) / 1e6}.items():
        approximate_equal(summary[field], expected, f"timing {field} mismatch")


def verify(path, *, verify_sources=True):
    path = Path(path)
    artifact = json.loads(path.read_text())
    require(artifact["schema"] == "oneq-gap-distribution/4", "unsupported schema")
    protocol = artifact["protocol"]
    require(protocol["heldout_seed"] != protocol["train_seed"], "training and heldout seeds coincide")
    trace_name = artifact["trace"]["path"]
    require(Path(trace_name).name == trace_name, "trace path must be a sibling filename")
    compressed = (path.parent / trace_name).read_bytes()
    require(hashlib.sha256(compressed).hexdigest() == artifact["trace"]["sha256"], "trace hash mismatch")
    if verify_sources:
        for name, expected in artifact["head"]["sha256"].items():
            require(hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected,
                    f"source hash mismatch: {name}")
    records = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
    require(len(records) == artifact["trace"]["records"], "trace record count mismatch")
    models, model_hashes = {}, {}
    for record in records:
        if record["phase"] != "MODEL":
            continue
        distance, graph = record["distance"], record["model"]
        require(distance not in models and graph["distance"] == distance, "duplicate or mismatched model")
        digest = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(digest == record["graph_sha256"], "model hash mismatch")
        weights, edges = {}, []
        for row in graph["edges"]:
            edge, weight = key(row["edge"]), float.fromhex(row["weight_hex"])
            require(edge not in weights and math.isfinite(weight) and weight >= 0, "invalid graph weight/duplicate")
            weights[edge] = Fraction.from_float(weight)
            edges.append((*edge, weight))
        columns = []
        for row in graph["component_columns"]:
            edge = key(row["edge"])
            require(edge in weights and 0 < float.fromhex(row["probability_hex"]) < 0.5, "invalid component")
            columns.append(edge)
        models[distance], model_hashes[distance] = (weights, columns, edges), digest
    require(set(models) == {row["distance"] for row in artifact["rows"]}, "model/summary families differ")
    checked = 0
    for row in artifact["rows"]:
        distance, model = row["distance"], models[row["distance"]]
        training = [r for r in records if r["phase"] == "TRAIN" and r["distance"] == distance]
        heldout = [r for r in records if r["phase"] == "HELDOUT" and r["distance"] == distance]
        require(len(heldout) == row["shots"] == protocol["heldout_shots"], "heldout denominator mismatch")
        require(sorted(r["index"] for r in heldout) == list(range(row["shots"])), "missing/duplicate heldout index")
        require(len({r["index"] for r in training}) == len(training), "duplicate training index")
        require(all(0 <= r["index"] < protocol["train_shots"] for r in training), "training index out of range")
        train_summary = row["training"]
        require(train_summary["shots"] == protocol["train_shots"] and train_summary["seed"] == protocol["train_seed"],
                "training protocol mismatch")
        require(train_summary["nontrivial"] == len(training), "training count mismatch")
        totals = {format(scale, ".17g"): [0, Fraction()] for scale in protocol["bp_scales"]}
        for record in training:
            require(set(record["candidates"]) == set(totals) and record["syndrome"], "training candidate set mismatch")
            for name, proposal in record["candidates"].items():
                cost = candidate(proposal, model, record["syndrome"])
                totals[name][0] += cost is None
                totals[name][1] += cost if cost is not None else 0
        selected = min(totals, key=lambda name: (*totals[name], float(name)))
        require(float(selected) == train_summary["selected_scale"], "training selection mismatch")
        require(set(train_summary["candidates"]) == set(totals), "training totals omitted candidate")
        for name, (failures, cost) in totals.items():
            require(train_summary["candidates"][name]["failures"] == failures, "training failure total mismatch")
            require(rational(train_summary["candidates"][name]["sum_weight_exact"]) == cost, "training cost mismatch")
        stats = {name: {"n": 0, "errors": 0, "inconsistent": 0, "gaps": [], "exact": [],
                        "decode": [], "projection": []} for name in row["decoders"]}
        exact = nontrivial = 0
        cert_times, replay_times = [], []
        for shot in heldout:
            require(shot["graph_sha256"] == model_hashes[distance], "shot/model binding mismatch")
            syndrome = shot["syndrome"]
            require(len(syndrome) == len(set(syndrome)) and all(type(v) is int and v >= 0 for v in syndrome),
                    "invalid syndrome")
            if not syndrome:
                require(shot["status"] == "TRIVIAL_EXCLUDED_FROM_GAP_COMPARISON", "trivial status mismatch")
                continue
            nontrivial += 1
            ref = shot["reference"]
            witness = ref["witness"]
            cert = MatchingCertificate(
                matched_edges=tuple(tuple(edge) for edge in witness["matched_edges"]),
                node_potentials={p["vertex"]: float.fromhex(p["value_hex"]) for p in witness["node_potentials"]},
                blossom_duals={frozenset(c["set"]): float.fromhex(c["value_hex"]) for c in witness["cut_duals"]})
            epsilon = check_epsilon(edges=model[2], syndrome=syndrome, cert=cert, eps_max=Fraction())
            require(epsilon.primal is not None and epsilon.lower_bound is not None, "invalid reference bound")
            require(epsilon.primal == rational(ref["weight_exact"]), "reference cost mismatch")
            require(epsilon.lower_bound == rational(ref["scaled_packing_bound_exact"]), "scaled bound mismatch")
            accepted = check(edges=model[2], syndrome=syndrome, cert=cert).accepted
            require(type(ref["accepted_exact"]) is bool and accepted == ref["accepted_exact"], "reference verdict mismatch")
            bound = epsilon.primal if accepted else epsilon.lower_bound
            require(bound == rational(ref["lower_bound_exact"]), "reference bound mismatch")
            exact += accepted
            checked += 1
            for field, target in [("certify_including_internal_checks_ns", cert_times),
                                  ("retained_witness_recheck_ns", replay_times)]:
                require(type(ref[field]) is int and ref[field] >= 0, "invalid reference timing")
                target.append(ref[field])
            require(set(shot["candidates"]) == set(stats), "heldout candidate set mismatch")
            for name, proposal in shot["candidates"].items():
                stat = stats[name]
                stat["n"] += 1
                cost = candidate(proposal, model, syndrome)
                stat["decode"].append(proposal["decode_ns"])
                stat["projection"].append(proposal["projection_ns"])
                if cost is None:
                    stat["errors" if proposal["status"] == "ERROR" else "inconsistent"] += 1
                    continue
                require(cost >= bound, "candidate undercuts checked bound")
                require(rational(proposal["gap_vs_certified_bound_exact"]) == cost - bound, "gap mismatch")
                stat["gaps"].append(cost - bound)
                if accepted:
                    require(rational(proposal["gap_vs_proven_optimum_exact"]) == cost - epsilon.primal,
                            "exact optimum gap mismatch")
                    stat["exact"].append(cost - epsilon.primal)
                else:
                    require("gap_vs_proven_optimum_exact" not in proposal, "unproved optimum claim")
        require(row["nontrivial"] == nontrivial and row["trivial_excluded"] == len(heldout) - nontrivial,
                "nontrivial denominator mismatch")
        require(row["mwpm_certified_exact"] == exact, "reference coverage count mismatch")
        require(row["exact_reference_coverage_nontrivial"] == (exact / nontrivial if nontrivial else None),
                "reference coverage ratio mismatch")
        verify_timing(row["certification_including_internal_checks"], cert_times)
        verify_timing(row["retained_witness_recheck"], replay_times)
        for name, stat in stats.items():
            result = row["decoders"][name]
            require(result["n"] == stat["n"] and result["errors"] == stat["errors"]
                    and result["syndrome_inconsistent"] == stat["inconsistent"], "candidate count mismatch")
            verify_summary(result["gap_vs_certified_bound"], stat["gaps"])
            verify_summary(result["gap_vs_proven_optimum"], stat["exact"])
            verify_timing(result["decode_latency"], stat["decode"])
            verify_timing(result["projection_latency"], stat["projection"])
    return {"accepted": True, "families": len(models), "reference_witnesses_rechecked": checked,
            "scope": "graph costs, proof verdicts, TRAIN selection, denominators and timing aggregation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.summary)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(json.dumps({"accepted": False, "reason": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
