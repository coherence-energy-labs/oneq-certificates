"""Exact graph-objective comparisons with XOR-projected decoder corrections.

A BP-OSD column in this experiment is a graphlike DEM component. Components
of a decomposed event are treated independently; this is NOT the original
correlated-event hypergraph model or an exact maximum-likelihood decoder.
Selected columns are projected onto canonical graph edges modulo two BEFORE
scoring. All costs and zero/negative-gap decisions use exact rational values
of the graph's IEEE-754 weights. Float quantiles are display summaries only.

The MWPM reference is an exact optimum only on shots accepted by the existing
exact checker. Other shots retain an independently checked rational lower
bound; failure to certify is not evidence that the correction is suboptimal.
BP scaling is chosen on a separate seeded TRAIN sample, frozen, and compared
with explicitly unscaled BP and MWPM on the same HELDOUT syndromes. This
compares graph cost, not logical error probability or logical-coset ML.

Historical evidence/gap_distribution*.json files are immutable historical
measurements. This corrected protocol writes to its own new evidence folder.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder  # noqa: E402
from oneq.matching_cert import BOUNDARY, MatchingCertificate, check  # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon  # noqa: E402


def fraction_json(value):
    value = Fraction(value)
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _exact_weight(value):
    value = value if isinstance(value, Fraction) else Fraction.from_float(float(value))
    if value < 0:
        raise ValueError("negative graph weight")
    return value


def graph_projection(keys, weights):
    """Return (canonical XOR edge set, odd detectors, exact graph cost).

    Unknown edges are refused even if they would cancel. Duplicate columns
    are not parallel selected edges: they represent the same binary variable
    in the comparison graph and must cancel modulo two.
    """
    selected = set()
    for edge in keys:
        key = CertifyingDecoder.key(edge)
        if key not in weights:
            raise ValueError(f"edge {key} is absent from the comparison graph")
        selected.symmetric_difference_update((key,))
    odd = set()
    cost = Fraction(0)
    for edge in sorted(selected):
        cost += _exact_weight(weights[edge])
        for vertex in edge:
            if vertex != BOUNDARY:
                odd.symmetric_difference_update((vertex,))
    return tuple(sorted(selected)), odd, cost


def component_projection(mechs, selection, weights=None):
    """Project a binary component vector without rounding malformed entries."""
    selected = np.asarray(selection)
    if selected.ndim != 1 or len(selected) != len(mechs):
        raise ValueError("component selection must have one entry per column")
    if not np.all((selected == 0) | (selected == 1)):
        raise ValueError("component selection must be binary")
    if weights is None:
        weights = {}
        for edge, weight, _probability in mechs:
            key = CertifyingDecoder.key(edge)
            exact = _exact_weight(weight)
            if key in weights and weights[key] != exact:
                raise ValueError("duplicate columns disagree on canonical graph weight")
            weights[key] = exact
    keys = [mechs[int(j)][0] for j in np.flatnonzero(selected)]
    return graph_projection(keys, weights)


def odd_and_weight(mechs, sel):
    """Compatibility helper: syndrome and EXACT cost of the XOR correction."""
    _keys, odd, weight = component_projection(mechs, sel)
    return odd, weight


def dem_mechanisms(dem, W, key):
    """One independent decoder variable per graphlike DEM component.

    The comparison graph's merged edge weights define the objective. The
    original event's probability is assigned separately to each component;
    original correlations and pure logical events are not modeled here.
    """
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        probability = float(inst.args_copy()[0])
        components, current = [], []
        for target in inst.targets_copy():
            if target.is_separator():
                components.append(current)
                current = []
            elif target.is_relative_detector_id():
                current.append(target.val)
        components.append(current)
        for detectors in components:
            if not detectors:
                continue
            if len(detectors) > 2:
                raise ValueError("DEM component is not graphlike")
            edge = key((int(detectors[0]), int(detectors[1])
                        if len(detectors) == 2 else BOUNDARY))
            if edge not in W:
                raise ValueError("DEM component is missing from the comparison graph")
            out.append((edge, W[edge], min(max(probability, 1e-15), 0.5 - 1e-15)))
    return out


def summarise(values):
    """Exact classifications; explicitly approximate display quantiles."""
    values = list(values)
    if not values:
        return None
    floats = np.asarray([float(v) for v in values])
    zeros = sum(v == 0 for v in values)
    negatives = sum(v < 0 for v in values)
    return {"n": len(values), "mean": float(sum(values, Fraction()) / len(values)),
            "median": float(np.median(floats)), "p90": float(np.percentile(floats, 90)),
            "p99": float(np.percentile(floats, 99)), "max": float(max(values)),
            "zero_count_exact": zeros, "negative_count_exact": negatives,
            "frac_zero": zeros / len(values), "frac_below_zero": negatives / len(values),
            "sum_exact": fraction_json(sum(values, Fraction())),
            "max_exact": fraction_json(max(values)),
            "quantile_arithmetic": "approximate float display; classifications are rational"}


def _certificate_json(cert):
    return {"matched_edges": [list(edge) for edge in cert.matched_edges],
            "node_potentials": [{"vertex": int(v), "value_hex": float(z).hex()}
                                for v, z in sorted(cert.node_potentials.items())],
            "cut_duals": [{"set": sorted(map(int, members)), "value_hex": float(z).hex()}
                          for members, z in sorted(cert.blossom_duals.items(),
                                                   key=lambda item: tuple(sorted(item[0])))]}


def _timing(values):
    if not values:
        return None
    return {"count": len(values), "total_ns": sum(values),
            "mean_ms": sum(values) / len(values) / 1e6,
            "p50_ms": float(np.percentile(values, 50)) / 1e6,
            "p95_ms": float(np.percentile(values, 95)) / 1e6,
            "max_ms": max(values) / 1e6}


def _bp_decoder(H, probabilities, scaling):
    from ldpc import BpOsdDecoder
    return BpOsdDecoder(H, error_channel=list(probabilities), max_iter=30,
                        bp_method="ms", ms_scaling_factor=scaling,
                        schedule="parallel", osd_method="osd_cs", osd_order=5)


def _decode_record(decoder, syndrome, mechs, weights):
    started = time.perf_counter_ns()
    selected_indices = None
    try:
        if getattr(decoder, "returns_keys", False):
            raw_keys = decoder.decode_keys(syndrome)
            decoded = time.perf_counter_ns()
            keys, odd, weight = graph_projection(raw_keys, weights)
        else:
            selection = np.asarray(decoder.decode(syndrome.astype(np.uint8)))
            decoded = time.perf_counter_ns()
            keys, odd, weight = component_projection(mechs, selection, weights)
            selected_indices = list(map(int, np.flatnonzero(selection)))
        projected = time.perf_counter_ns()
        consistent = odd == set(map(int, np.flatnonzero(syndrome)))
        record = {"status": "VALID" if consistent else "SYNDROME_INCONSISTENT",
                  "selected_component_indices": selected_indices,
                  "projected_edges": [list(edge) for edge in keys],
                  "weight_exact": fraction_json(weight),
                  "decode_ns": decoded - started, "projection_ns": projected - decoded}
        return record, weight if consistent else None
    except Exception as exc:  # decoder failures are reported, never silently dropped
        return {"status": "ERROR", "error_type": type(exc).__name__,
                "error": str(exc)[:200], "decode_ns": time.perf_counter_ns() - started,
                "projection_ns": 0}, None


def _train_scaling(circuit, H, mechs, weights, scales, shots, seed):
    syndromes = circuit.compile_detector_sampler(seed=seed).sample(shots)
    decoders = {format(scale, ".17g"): _bp_decoder(H, [m[2] for m in mechs], scale)
                for scale in scales}
    totals = {name: {"failures": 0, "cost": Fraction(0)} for name in decoders}
    records = []
    for index, syndrome in enumerate(syndromes):
        if not np.any(syndrome):
            continue
        candidates = {}
        for name, decoder in decoders.items():
            record, cost = _decode_record(decoder, syndrome, mechs, weights)
            candidates[name] = record
            if cost is None:
                totals[name]["failures"] += 1
            else:
                totals[name]["cost"] += cost
        records.append({"phase": "TRAIN", "index": index,
                        "syndrome": list(map(int, np.flatnonzero(syndrome))),
                        "candidates": candidates})
    # Fewer failures, then exact total graph cost, then numeric scale; no
    # HELDOUT observation or logical outcome can influence this decision.
    selected = min(totals, key=lambda name: (totals[name]["failures"],
                                           totals[name]["cost"], float(name)))
    return float(selected), records, {
        "seed": seed, "shots": shots, "nontrivial": len(records),
        "selection_objective": "lexicographic (failures, exact total graph cost, scale)",
        "selected_scale": float(selected),
        "candidates": {name: {"failures": value["failures"],
                               "sum_weight_exact": fraction_json(value["cost"])}
                       for name, value in totals.items()}}
class _BeliefMatchingCorrection:
    """Recovers the EDGE SET belief-matching actually applies.

    Reproduces `BeliefMatching.decode`'s own pipeline rather than calling it:
    BP first, and on non-convergence the reweighted matching it builds
    internally.

    RETURNS EDGE KEYS, NOT A MECHANISM SELECTION. beliefmatching enumerates its
    own graphlike edge columns, and those indices have nothing to do with our
    mechanism list -- reusing them as if they did would silently score the
    wrong edges and produce a plausible, meaningless gap. The detectors an edge
    touches are read out of its column in `edge_check_matrix`, which is the
    only thing that identifies it unambiguously.
    """

    returns_keys = True

    def __init__(self, bm, key):
        self.bm = bm
        self.mm = bm._matrices
        self.key = key
        H = self.mm.edge_check_matrix.tocsc()
        self.cols = []
        for j in range(H.shape[1]):
            dets = H.indices[H.indptr[j]:H.indptr[j + 1]]
            if len(dets) == 2:
                self.cols.append(key((int(dets[0]), int(dets[1]))))
            elif len(dets) == 1:
                self.cols.append(key((int(dets[0]), BOUNDARY)))
            else:
                self.cols.append(None)      # touches no detector: unweighable

    def decode_keys(self, syndrome):
        import numpy as np
        import pymatching
        s = np.asarray(syndrome).astype(np.uint8)
        corr = self.bm._bpd.decode(s)
        if self.bm._bpd.converge:
            sel = (self.mm.hyperedge_to_edge_matrix @ corr) % 2
            sel = np.asarray(sel).ravel().astype(bool)
        else:
            llrs = self.bm._bpd.log_prob_ratios
            ps_h = 1.0 / (1.0 + np.exp(llrs))
            ps_e = np.asarray(self.mm.hyperedge_to_edge_matrix @ ps_h).ravel()
            eps = 1e-14
            ps_e = np.clip(ps_e, eps, 1 - eps)
            M = pymatching.Matching.from_check_matrix(
                self.mm.edge_check_matrix, weights=-np.log(ps_e),
                faults_matrix=self.mm.edge_observables_matrix,
                use_virtual_boundary_node=True)
            # decode() RETURNS OBSERVABLE PREDICTIONS, NOT EDGES, because
            # faults_matrix is set -- reading it as an edge selection scored a
            # length-1 observable vector as if it indexed the 502 edge columns
            # and made every non-converged shot look syndrome-inconsistent.
            # 54 of 149 shots, which would have been written up as a property
            # of belief-matching rather than as my bug.
            return [self.key((int(u), int(v) if v >= 0 else BOUNDARY))
                    for u, v in M.decode_to_edges_array(s)]
        return [self.cols[j] for j in np.flatnonzero(sel) if self.cols[j] is not None]



def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None):
    import pymatching
    import stim
    from scipy.sparse import csc_matrix

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=400)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--p", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--train-seed", type=int, default=131337)
    parser.add_argument("--train-shots", type=int, default=128)
    parser.add_argument("--bp-scales", type=float, nargs="+", default=[0.5, 0.625, 0.75, 1.0])
    parser.add_argument("--no-belief-matching", action="store_true")
    parser.add_argument("--out", type=pathlib.Path,
                        default=ROOT / "evidence" / "corrected_gap_20261103" / "summary.json")
    args = parser.parse_args(argv)
    if args.shots <= 0 or args.train_shots <= 0 or args.seed == args.train_seed:
        parser.error("positive shot counts and distinct TRAIN/HELDOUT seeds are required")
    if not 0 < args.p < 0.5 or any(not 0 < x <= 1 for x in args.bp_scales):
        parser.error("p must be in (0, .5); BP scales must be in (0, 1]")
    if len(set(args.bp_scales)) != len(args.bp_scales) or len(set(args.distances)) != len(args.distances):
        parser.error("duplicate scales or distances are not allowed")
    if args.rounds is not None and args.rounds <= 0:
        parser.error("rounds must be positive")
    if any(d < 3 or d % 2 == 0 for d in args.distances):
        parser.error("rotated surface-code distances must be odd and at least three")
    protected = {ROOT / "evidence" / "gap_distribution.json",
                 ROOT / "evidence" / "gap_distribution_exact_rule.json"}
    if args.out.resolve() in {p.resolve() for p in protected}:
        parser.error("historical gap artifacts must not be overwritten")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error("output already exists; use a new path to retain earlier evidence")

    rows, trace = [], []
    for distance in args.distances:
        rounds = args.rounds or distance
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
            after_clifford_depolarization=args.p,
            before_measure_flip_probability=args.p,
            after_reset_flip_probability=args.p,
            before_round_data_depolarization=args.p)
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        edges = [(int(u), BOUNDARY if v is None else int(v), float(data["weight"]))
                 for u, v, data in matching.edges()]
        certifier = CertifyingDecoder(edges)
        weights = {key: _exact_weight(weight) for key, weight in certifier.W.items()}
        mechs = dem_mechanisms(dem, certifier.W, certifier.key)
        rr, cc = [], []
        for j, (edge, _weight, _probability) in enumerate(mechs):
            for detector in edge:
                if detector != BOUNDARY:
                    rr.append(detector)
                    cc.append(j)
        H = csc_matrix((np.ones(len(rr), dtype=np.uint8), (rr, cc)),
                       shape=(dem.num_detectors, len(mechs)))
        started = time.perf_counter_ns()
        selected_scale, training, training_summary = _train_scaling(
            circuit, H, mechs, weights, args.bp_scales, args.train_shots, args.train_seed)
        training_summary["elapsed_ns"] = time.perf_counter_ns() - started
        for record in training:
            record["distance"] = distance
        trace.extend(training)
        decoders = {"bp_osd_unscaled": _bp_decoder(H, [m[2] for m in mechs], 1.0),
                    "bp_osd_train_selected": _bp_decoder(H, [m[2] for m in mechs], selected_scale)}
        availability = {name: {"available": True} for name in decoders}
        if not args.no_belief_matching:
            try:
                from beliefmatching import BeliefMatching
                decoders["belief_matching"] = _BeliefMatchingCorrection(
                    BeliefMatching(dem, max_bp_iters=20), certifier.key)
                availability["belief_matching"] = {"available": True}
            except Exception as exc:
                availability["belief_matching"] = {"available": False,
                    "reason": f"{type(exc).__name__}: {exc}"[:240]}
        else:
            availability["belief_matching"] = {"available": False, "reason": "explicitly disabled"}

        graph = {"distance": distance, "rounds": rounds,
                 "detector_count": dem.num_detectors,
                 "edges": [{"edge": list(edge), "weight_hex": float(weight).hex()}
                           for edge, weight in sorted(certifier.W.items())],
                 "component_columns": [{"edge": list(edge), "probability_hex": float(p).hex()}
                                       for edge, _weight, p in mechs],
                 "stim_circuit": str(circuit), "dem": str(dem)}
        graph_bytes = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
        graph_hash = hashlib.sha256(graph_bytes).hexdigest()
        trace.append({"phase": "MODEL", "distance": distance, "graph_sha256": graph_hash,
                      "model": graph})
        detectors, observables = circuit.compile_detector_sampler(seed=args.seed).sample(
            args.shots, separate_observables=True)
        stats = {name: {"n": 0, "errors": 0, "inconsistent": 0, "gaps": [],
                        "exact_gaps": [], "decode_ns": [], "projection_ns": []}
                 for name in ["mwpm", *decoders]}
        certify_times, replay_times = [], []
        nontrivial = exact_count = 0
        for index, syndrome in enumerate(detectors):
            fired = list(map(int, np.flatnonzero(syndrome)))
            shot = {"phase": "HELDOUT", "distance": distance, "index": index,
                    "graph_sha256": graph_hash, "syndrome": fired,
                    "observables": list(map(int, observables[index]))}
            if not fired:
                shot["status"] = "TRIVIAL_EXCLUDED_FROM_GAP_COMPARISON"
                trace.append(shot)
                continue
            nontrivial += 1
            started = time.perf_counter_ns()
            raw = matching.decode_to_edges_array(syndrome)
            decoded = time.perf_counter_ns()
            mw_keys, mw_odd, mw_weight = graph_projection(raw, weights)
            projected = time.perf_counter_ns()
            if mw_odd != set(fired):
                raise RuntimeError("MWPM reference correction does not explain the syndrome")
            result = certifier.certify(fired, mw_keys)
            certified = time.perf_counter_ns()
            witness = result.cert if result.accepted else MatchingCertificate(
                matched_edges=mw_keys, node_potentials={}, blossom_duals=result.best_dual)
            if witness is None:
                raise RuntimeError("accepted reference omitted its witness")
            epsilon = check_epsilon(edges=edges, syndrome=fired, cert=witness, eps_max=Fraction(0))
            if epsilon.lower_bound is None or epsilon.primal != mw_weight:
                raise RuntimeError("reference packing failed exact lower-bound validation")
            if result.accepted and not check(edges=edges, syndrome=fired, cert=witness).accepted:
                raise RuntimeError("retained reference witness failed exact replay")
            replayed = time.perf_counter_ns()
            bound = mw_weight if result.accepted else epsilon.lower_bound
            exact_count += bool(result.accepted)
            certify_times.append(certified - projected)
            replay_times.append(replayed - certified)
            shot["reference"] = {"accepted_exact": bool(result.accepted), "reason": result.reason,
                "witness": _certificate_json(witness), "weight_exact": fraction_json(mw_weight),
                "lower_bound_exact": fraction_json(bound),
                "scaled_packing_bound_exact": fraction_json(epsilon.lower_bound),
                "certify_including_internal_checks_ns": certified - projected,
                "retained_witness_recheck_ns": replayed - certified}
            candidates = {"mwpm": ({"status": "VALID", "selected_component_indices": None,
                "projected_edges": [list(edge) for edge in mw_keys],
                "weight_exact": fraction_json(mw_weight), "decode_ns": decoded - started,
                "projection_ns": projected - decoded}, mw_weight)}
            for name, decoder in decoders.items():
                candidates[name] = _decode_record(decoder, syndrome, mechs, weights)
            shot["candidates"] = {}
            for name, (record, cost) in candidates.items():
                stat = stats[name]
                stat["n"] += 1
                stat["decode_ns"].append(record["decode_ns"])
                stat["projection_ns"].append(record["projection_ns"])
                if cost is None:
                    stat["errors" if record["status"] == "ERROR" else "inconsistent"] += 1
                else:
                    gap = cost - bound
                    if gap < 0:
                        raise RuntimeError(f"{name} correction undercuts a checked exact bound")
                    record["gap_vs_certified_bound_exact"] = fraction_json(gap)
                    stat["gaps"].append(gap)
                    if result.accepted:
                        record["gap_vs_proven_optimum_exact"] = fraction_json(cost - mw_weight)
                        stat["exact_gaps"].append(cost - mw_weight)
                shot["candidates"][name] = record
            trace.append(shot)
        rows.append({"distance": distance, "rounds": rounds, "p": args.p,
            "shots": args.shots, "nontrivial": nontrivial,
            "trivial_excluded": args.shots - nontrivial,
            "mwpm_certified_exact": exact_count,
            "exact_reference_coverage_nontrivial": exact_count / nontrivial if nontrivial else None,
            "training": training_summary, "decoder_availability": availability,
            "certification_including_internal_checks": _timing(certify_times),
            "retained_witness_recheck": _timing(replay_times),
            "decoders": {name: {"n": stat["n"], "errors": stat["errors"],
                "syndrome_inconsistent": stat["inconsistent"],
                "gap_vs_certified_bound": summarise(stat["gaps"]),
                "gap_vs_proven_optimum": summarise(stat["exact_gaps"]),
                "decode_latency": _timing(stat["decode_ns"]),
                "projection_latency": _timing(stat["projection_ns"])}
                         for name, stat in stats.items()}})
        print(f"d={distance}: {nontrivial} nontrivial, {exact_count} exact references; "
              f"TRAIN selected BP scale {selected_scale}", flush=True)
        for name, stat in rows[-1]["decoders"].items():
            gap = stat["gap_vs_proven_optimum"]
            print(f"  {name}: {gap}", flush=True)

    source_paths = ["experiments/m0_prior_art/gap_distribution.py", "src/oneq/matching_cert.py",
                    "src/oneq/matching_cert_epsilon.py", "src/oneq/matching_cert_strict.py",
                    "src/oneq/certifying_decoder.py", "tools/replay_corrected_gaps.py"]
    trace_path = args.out.with_name(args.out.stem + "_shots.jsonl.gz")
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                      for record in trace).encode()
    trace_path.write_bytes(gzip.compress(payload, mtime=0))
    artifact = {"schema": "oneq-gap-distribution/4", "rows": rows,
        "head": {"commit": _git("rev-parse", "HEAD"),
                 "uncommitted_source_changes": bool(_git("status", "--porcelain", "--", *source_paths)),
                 "sha256": {path: hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                            for path in source_paths}},
        "environment": {"python": sys.version.split()[0],
                        **{name: importlib.metadata.version(name)
                           for name in ["numpy", "scipy", "stim", "pymatching", "ldpc"]}},
        "trace": {"path": trace_path.name, "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  "records": len(trace)},
        "protocol": {"heldout_seed": args.seed, "train_seed": args.train_seed,
                     "train_shots": args.train_shots, "heldout_shots": args.shots,
                     "bp_scales": args.bp_scales, "bp_max_iter": 30, "bp_schedule": "parallel",
                     "osd_method": "osd_cs", "osd_order": 5,
                     "training_labels": "syndrome only; no logical labels or heldout observations",
                     "model": "independent graphlike DEM components; original event correlations discarded",
                     "projection": "XOR canonical graph edges before exact Fraction scoring",
                     "exact_zero_rule": "rational gap equals zero; no floating tolerance",
                     "coverage_denominator": "nontrivial HELDOUT syndromes; trivial shots retained separately"},
        "scope": {"objective": "minimum graph correction weight, not logical-coset ML or logical error rate",
                  "timing": "local wall-clock observations; certificate production includes internal checks",
                  "historical_comparison": "new sampled run, not corrected values for the original paper's samples",
                  "prior_art": "no novelty or universal decoder-superiority claim"}}
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"evidence -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
