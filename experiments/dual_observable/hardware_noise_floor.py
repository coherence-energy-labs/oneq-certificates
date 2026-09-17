r"""THE DUAL AS AN INSTRUMENT, ON A REAL DEVICE.

`certified_noise_floor.py` runs the same three instruments on stim, and
its ledger row `DO-1` is declared a PROXY for the sharpest possible
reason: on a simulator the shots are sampled FROM the model being tested
and decoded USING it, so `dem_calibration` cannot detect the thing it
exists to detect. The model IS the sampler.

This lane removes that circularity. The shots come from `ibm_marrakesh`;
the model comes from the device's published calibration. Neither
generated the other.

--------------------------------------------------------------------------
*** IT SPENDS NO CREDITS BY DEFAULT, AND THAT IS NOT A CONVENIENCE. ***

A completed IBM job is re-fetchable. `evidence/ibm_live_job.json` names
one -- 4,000 shots of a d=5 repetition-code memory already paid for --
so the first hardware reading of this instrument costs ZERO QPU
seconds.

That matters beyond thrift. `hardware_lambda.py` shipped with its
scorer inside the submission path, so the only way to execute it was to
spend the owner's budget; it reached production unrun and the bound
being quoted from it was hand arithmetic. Scoring is not a device act.
Here the device act is `--submit`, everything else reads stored or
re-fetched shots, and `--dry-run` exercises the entire path on
synthetic detectors so the code is proven before a single credit moves.

--------------------------------------------------------------------------
*** THE WEIGHTS ARE THE DEVICE'S, NOT ONES. ***

`evidence/ibm_live_certified.json` records "uniform, declared". Under
uniform weights the minimum-weight explanation counts MECHANISMS and
says nothing about probability, so `dem_calibration` would have no model
to test and `certified_weight` would report a count.

`oneq.device_dem.rep_code_edges` builds the weights from the device's
own numbers -- readout error, two-qubit gate error, T1 idle over a round
-- and `_weight(p) = log((1-p)/p)` is exactly the log-odds convention
the LOG_LIKELIHOOD reading requires. Checked here rather than assumed.

--------------------------------------------------------------------------
*** THE RAW DETECTORS ARE KEPT THIS TIME. ***

`evidence/ibm_live/*.job.json` stores `backend`, `config`, `job_id`,
`prereg_sha256`, `provider`, `submitted_utc` -- and no shots. Every
re-analysis therefore needs the network and IBM's retention policy. The
same shape as certified shots not keeping their weights, one level up.

This lane writes the detector array into its own artifact, so every
number here can be recomputed offline, forever, by someone who has only
the file.

    python experiments/dual_observable/hardware_noise_floor.py --dry-run
    python experiments/dual_observable/hardware_noise_floor.py --refetch
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ibm_live"))

from oneq.certified_weight import (LOG_LIKELIHOOD,              # noqa: E402
                                   device_floor,
                                   is_gameable_by_weakening_the_dual,
                                   survivorship_gap)
from oneq.certifying_decoder import BOUNDARY, CertifyingDecoder  # noqa: E402
from oneq.dem_families import (apply_family_scales, burst_edges,  # noqa: E402,F401
                               edge_family, parse_scales)
from oneq.dem_calibration import calibration_report              # noqa: E402
from oneq.exchangeability import (ExchangeabilityRefusal,       # noqa: E402
                                  block_scan, lag1_autocorrelation,
                                  permutation_test)
from oneq.soft_estimator import (EstimatorRefusal,               # noqa: E402
                                 measured_estimate,
                                 parity_calibration_split,
                                 weighted_estimate)


def lifted_matching(edges, logical_cut, n_nodes):
    r"""Two matching graphs: the plain one, and one with the observable
    promoted to a detector.

    *** FOR A REPETITION CODE THE LIFT IS EXACT AND CHEAP. *** In
    general, relabelling a logical observable as a detector turns every
    observable-flipping error into a three-target hyperedge, which a
    matching decoder cannot take. On a 1-D chain the logical cut IS the
    right-hand boundary, so those edges simply terminate on a real node
    instead of the virtual one -- an ordinary edge, no decomposition
    needed.

    Decoding `[syndrome, 0]` and `[syndrome, 1]` against the lifted
    graph then gives the minimum weight in each logical class, and the
    difference is the complementary gap: a log-likelihood ratio, which
    is what `soft_estimator` and `dem_calibration` both consume.
    """
    import pymatching

    plain = pymatching.Matching()
    lift = pymatching.Matching()
    obs_node = n_nodes                      # the promoted observable
    for (u, v, w) in edges:
        w = float(w)
        if v == BOUNDARY:
            plain.add_boundary_edge(int(u), weight=w)
            if (int(u), BOUNDARY) in logical_cut or (u, v) in logical_cut:
                lift.add_edge(int(u), obs_node, weight=w)
            else:
                lift.add_boundary_edge(int(u), weight=w)
        else:
            plain.add_edge(int(u), int(v), weight=w)
            lift.add_edge(int(u), int(v), weight=w)
    return plain, lift, obs_node


def score(edges, logical_cut, det, truth, n_nodes, *, budget_s=0.0):
    r"""Per-shot records: the numbers this repository kept discarding.

    Returns `(records, preds, gaps, correct)` -- the certified interval
    inputs and the soft-decision inputs, taken from the SAME shots so
    the correlation between confidence and error weight survives.
    """
    plain, lift, obs_node = lifted_matching(edges, logical_cut, n_nodes)
    cd = CertifyingDecoder(edges)

    # BATCHED, AND `return_weightS` IS decode_batch's SPELLING.
    # `Matching.decode` takes `return_weight` (singular); the plural was
    # accepted and IGNORED, so the first version indexed [1] into a bare
    # prediction array. Caught by --dry-run before a credit moved, which
    # is the entire reason that mode exists.
    dets = det.astype(np.uint8)
    zeros = np.zeros((dets.shape[0], 1), dtype=np.uint8)
    w0 = lift.decode_batch(np.concatenate([dets, zeros], axis=1),
                           return_weights=True)[1]
    w1 = lift.decode_batch(np.concatenate([dets, zeros ^ 1], axis=1),
                           return_weights=True)[1]
    preds = [1 if b < a_ else 0 for a_, b in zip(w0, w1)]
    gaps = [abs(float(b) - float(a_)) for a_, b in zip(w0, w1)]
    correct = ([bool(int(t)) == bool(p) for t, p in zip(truth, preds)]
               if truth is not None else [])

    records = []
    t0 = time.time()
    for s in range(det.shape[0]):
        row = dets[s]
        fired = [int(i) for i in np.flatnonzero(row)]
        if not fired:
            continue
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                     for a, b in plain.decode_to_edges_array(row))
        res = cd.certify_hard(fired, corr)
        records.append({"primal": float(res.primal),
                        "bound": float(res.bound),
                        "accepted": bool(res.accepted)})
        if budget_s and time.time() - t0 > budget_s:
            break
    return records, preds, gaps, correct


#: The published numbers a re-score must reproduce, by JSON path. Named
#: EXPLICITLY rather than diffing whole documents, because a whole-doc
#: diff would trip on `seconds` and on any prose reworded later, and a
#: check that fails for unrelated reasons gets switched off -- taking the
#: real one with it. Everything here is a number this repository asks a
#: third party to believe.
_CLAIMS = (
    ("certified_weight.proven_interval_on_mean_min_weight",),
    ("certified_weight.certified_fraction",),
    ("certified_weight.mean_width",),
    ("certified_weight.population_bound_reliable",),
    ("survivorship_gap",),
    ("anti_gaming.gameable",),
    ("anti_gaming.bound_fell",),
    ("soft_estimate.value",),
    ("soft_estimate.effective_shot_multiplier",),
    ("dem_calibration.calibration_error",),
    ("dem_calibration.chi2",),
    ("dem_calibration.p_value",),
    ("dem_calibration.powered_bins",),
    ("exchangeability.lag1.observed",),
    ("exchangeability.lag1.p_value",),
    ("exchangeability.lag1.sensitivity_3sigma",),
    ("circuit.nontrivial",),
    ("passed",),
)
_CLAIM_COUNT = len(_CLAIMS)


def _dig(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


_MISSING = object()


def _claims_differ(published, recomputed):
    r"""Which published claims a re-score failed to reproduce.

    *** A MISTYPED CLAIM PATH MUST NOT READ AS AGREEMENT. ***

    The first version of this skipped any claim missing from both
    documents, reasoning that an older artifact predates a field. Four
    of the fifteen paths were then found to be guesses at key names --
    `calibration.verdict` for `dem_calibration.p_value`, `iid_lag1` for
    `exchangeability.lag1` -- and all four had been quietly passing.
    A quarter of the check was dark on its first run.

    The RECOMPUTED document is always current, so it is the authority on
    whether a field exists at all: a path missing there is a defect in
    this list and raises. Missing only from the published side is a real
    difference and is reported as one.
    """
    bad, broken = [], []
    for (path,) in _CLAIMS:
        was, now = _dig(published, path), _dig(recomputed, path)
        if now is _MISSING:
            broken.append(path)
            continue
        if isinstance(was, float) and isinstance(now, float):
            if math.isclose(was, now, rel_tol=0.0, abs_tol=0.0):
                continue
        elif was == now:
            continue
        bad.append((path, "<absent>" if was is _MISSING else was, now))
    if broken:
        raise KeyError(
            "these claim paths do not exist in the document this run just "
            "built, so they could never have compared anything: "
            + ", ".join(broken))
    return bad


def _synthetic_edges(d, r, p_edge):
    r"""A rep-code graph at a REALISTIC per-edge error probability.

    *** THE FIRST DRY RUN REHEARSED AT 27% PER EDGE. *** It reused
    `decode_graph`, whose weights are a uniform 1.0, and the generator
    inverts weight to probability with `p = 1/(1+e^w)` -- so w = 1.0
    means p = 0.269. That is far past threshold, where min-weight
    matching is nowhere near maximum-likelihood, and the run duly
    reported MISCALIBRATED. Correct, and a rehearsal for nothing: no
    device runs there.

    Weights are `log((1-p)/p)`, the same log-odds convention
    `device_dem._weight` uses, so the dry run exercises the LOG_LIKELIHOOD
    path the hardware arm will take.
    """
    from rep_code_cert import decode_graph
    w = math.log((1.0 - p_edge) / p_edge)
    return [(u, v, w) for (u, v, _old) in decode_graph(d, r)]


def _synthetic(d, r, edges, seed, shots=400):
    r"""Detectors from the SAME edge model, for the dry run.

    This is a MIRROR and is labelled one: the model generates the shots,
    so nothing it produces is evidence about a device. It exists to
    prove the code path end to end before any credit is spent, which is
    the only claim made for it.
    """
    rng = np.random.default_rng(seed)
    n_nodes = (r + 1) * (d - 1)
    det = np.zeros((shots, n_nodes), dtype=np.uint8)
    truth = np.zeros(shots, dtype=np.uint8)
    cut = set()
    for (u, v, w) in edges:
        p = 1.0 / (1.0 + math.exp(w))       # invert log((1-p)/p)
        fire = rng.random(shots) < p
        det[fire, int(u)] ^= 1
        if v != BOUNDARY:
            det[fire, int(v)] ^= 1
        elif int(u) % (d - 1) == d - 2:     # right boundary flips logical
            truth[fire] ^= 1
            cut.add((int(u), BOUNDARY))
    return det, truth, cut


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="drive the whole path on synthetic detectors from "
                         "the same edge model; spends nothing and proves "
                         "nothing about a device")
    ap.add_argument("--refetch", action="store_true",
                    help="re-fetch the COMPLETED job named in "
                         "evidence/ibm_live_job.json; spends zero QPU "
                         "because the job is already paid for")
    ap.add_argument("--from-artifact", action="store_true",
                    dest="from_artifact",
                    help="re-score from the PINNED weights and stored "
                         "detectors in --json; no network, no calibration "
                         "snapshot, and therefore reproducible")
    ap.add_argument("--job", default=None,
                    help="re-fetch THIS completed job id instead of the "
                         "one in evidence/ibm_live_job.json; the record "
                         "must already exist under evidence/ibm_live/, "
                         "so this can only read work already paid for")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--dry-noise", type=float, default=0.01, dest="dry_noise",
                    help="per-edge error probability for --dry-run. The "
                         "lift can only be VERIFIED where logical errors "
                         "actually occur: at d=5, p=0.01 the code works "
                         "so well that no gap bin has power and the run "
                         "is honestly UNCHECKABLE")
    ap.add_argument("--dry-shots", type=int, default=400, dest="dry_shots")
    ap.add_argument("--dry-distance", type=int, default=5, dest="dry_d")
    ap.add_argument("--json", default="evidence/hardware_noise_floor.json")
    ap.add_argument("--scales", default=None,
                    help="a DECLARED per-family calibration layer. THE RULE (R19): "
                         "start at 'time-like=0.1,space-like=1,boundary=1' -- one "
                         "number, anywhere in [0.01, 0.2], which carries most of the "
                         "gain and transfers across chains and snapshots -- and fit "
                         "all three only when the calibration gate reads MISCALIBRATED. "
                         "e.g. 'time-like=0.0875,space-like=1.6459,boundary=0.9411': "
                         "the pinned weights become the SCALED ones (exact), the "
                         "decoder and every certificate use them, and weight_model "
                         "says so. Refused with --from-artifact, whose weights are "
                         "already what was decoded with (R14-R18)")
    ap.add_argument("--burst-cost", type=float, default=None, dest="burst_cost",
                    help="a DECLARED BURST model (R21): every time-like weight is "
                         "REPLACED by this cost and multi-round time-like edges are "
                         "added at it, because a burst -- a leaked or stuck ancilla -- "
                         "costs about one event whatever its span. MEASURED at 0.5 with "
                         "no fitting: 710 -> 147 logical errors over 120,000 marrakesh "
                         "shots, better than the fitted scales, and far WORSE on "
                         "synthetic shots where no burst exists. Composes with --scales, "
                         "which is applied first"
                    )
    ap.add_argument("--scales-provenance", default="", dest="scales_provenance",
                    help="where the scales come from (a preregistration or artifact "
                         "path), recorded beside them")
    a = ap.parse_args()
    t0 = time.time()

    if not (a.dry_run or a.refetch or a.from_artifact):
        print("Pick a mode. --dry-run proves the path on synthetic "
              "detectors; --refetch reads a completed job for zero QPU; "
              "--from-artifact re-scores stored shots against their "
              "PINNED weights and touches no network at all.",
              file=sys.stderr)
        return 2

    from oneq.device_dem import rep_code_edges, snapshot_calibration
    from rep_code_cert import logical_cut_edges

    if a.from_artifact:
        # *** THE SAME 4,000 SHOTS GAVE 4.9615, THEN 5.1264. ***
        #
        # `snapshot_calibration` reads the backend LIVE, so re-analysing
        # one identical completed job a day later builds DIFFERENT edge
        # weights and reports a different certified weight -- a 3.32%
        # shift on ibm_marrakesh over the same stored detectors. Nothing
        # was wrong with either number; the device's published
        # calibration simply moved between the two readings.
        #
        # But a number that changes when the data did not is not one
        # anyone can check, and this repository's whole claim is that a
        # third party can recompute what it publishes.
        #
        # `device_dem` already states the cure: the derivation is a
        # PRODUCER step and its output is DATA -- every weight is
        # emitted as an exact rational and the artifact carries it, so a
        # verifier reads the pinned value verbatim and never calls a
        # transcendental. It returns `prov` for exactly this purpose.
        # The first version of this lane threw `prov` away, which is how
        # a lane built on pinned weights ended up unreproducible.
        src = ROOT / a.json
        if not src.is_file():
            print(f"REFUSING: no artifact at {src}", file=sys.stderr)
            return 5
        old = json.loads(src.read_text(encoding="utf-8"))
        prov = old.get("edge_provenance")
        if not prov:
            print("REFUSING: that artifact carries no pinned weights, so "
                  "re-scoring it would have to take a FRESH calibration "
                  "snapshot -- which is the very thing that made the "
                  "number move. Re-run --refetch once to mint one.",
                  file=sys.stderr)
            return 5
        # `rep_code_edges` returns {"round_seconds": ..., "edges": {...}},
        # so the edge table is one level down. Read it STRICTLY: a
        # missing "edges" key means this artifact came from some other
        # producer and guessing at its layout is how the flat-shape bug
        # above got as far as hardware.
        if "edges" not in prov:
            print("REFUSING: edge_provenance has no 'edges' table; this "
                  "artifact was not written by this lane.", file=sys.stderr)
            return 5
        pinned = prov["edges"]
        # Keys are f"{u}-{v}" and v is -1 at the boundary, so "8--1"
        # partitions to ("8", "-", "-1"). The EXACT rational is what
        # gets read; a float is a binary rational, so this is lossless
        # and not merely close.
        edges = []
        for k, v in pinned.items():
            u, _, rest = k.partition("-")
            edges.append((int(u), int(rest),
                          float(Fraction(v["weight_exact"]))))
        d = int(old["circuit"]["distance"])
        r = int(old["circuit"]["rounds"])
        backend = old["backend"]
        mode = f"from-artifact({old.get('mode', '?')})"
        sh = old["detector_shape"]
        det = np.unpackbits(
            np.array(old["detectors_packed"], dtype=np.uint8)
        )[: sh[0] * sh[1]].reshape(sh).astype(np.uint8)
        truth = np.array(old["truth"], dtype=np.uint8)
        cut = logical_cut_edges(d, r)
        cal_note = (str(old["weight_model"]).split(" [PINNED")[0]
                    + " [PINNED: re-scored offline from the exact rationals "
                      "the artifact carries, with no calibration snapshot]")
        convention = LOG_LIKELIHOOD
    elif a.dry_run:
        d, r = a.dry_d, 5
        P_EDGE = a.dry_noise
        edges = _synthetic_edges(d, r, P_EDGE)
        # PINNED here too, so `--from-artifact` is a complete OFFLINE
        # round trip and the reproducibility property can be tested
        # without a network or a device.
        #
        # *** THE SHAPE IS `rep_code_edges`'s, EXACTLY. *** The first
        # version emitted a FLAT {edge: record} dict, so the offline
        # round trip passed on a shape the device path never produces --
        # and the first real `--from-artifact` on hardware died trying
        # to parse "round_seconds" as a node index. A fixture that does
        # not match production tests the fixture.
        prov = {"round_seconds": None,
                "edges": {f"{u}-{v}": {"p": P_EDGE, "weight": float(w),
                                       "weight_exact": str(Fraction(float(w))),
                                       "from": "synthetic dry-run"}
                          for (u, v, w) in edges}}
        backend, mode = "DRY-RUN (no device)", "dry-run"
        det, truth, _cut = _synthetic(d, r, edges, seed=20260831,
                                      shots=a.dry_shots)
        cut = logical_cut_edges(d, r)
        cal_note = (f"synthetic at p={P_EDGE} per edge, weights "
                    "log((1-p)/p) -- the same convention the device arm "
                    f"uses, so the dry run exercises the real path")
        # A MIRROR, and declared one: these shots were sampled from the
        # very model that decodes them, so a CONSISTENT calibration here
        # proves the LIFT is correct and nothing about any device.
        convention = LOG_LIKELIHOOD
    else:
        # *** THE DEFAULT JOB WAS THE ONLY JOB THIS LANE COULD SEE. ***
        # `evidence/ibm_live_job.json` names one d=3 r=13 run on
        # ibm_marrakesh, and every finding here was measured on it --
        # one device, one geometry. A SECOND completed job was sitting
        # in evidence/ibm_live/ the whole time (d=9 r=5 on ibm_fez,
        # 4,000 shots, already paid for) and nothing could point at it.
        # A lane that can only read one dataset cannot tell a property
        # of the solver from a property of that dataset.
        if a.job:
            cand = ROOT / "evidence" / "ibm_live" / f"{a.job}.job.json"
            if not cand.is_file():
                print(f"REFUSING: no job record at {cand}. This lane "
                      f"re-fetches jobs already paid for; it does not "
                      f"submit.", file=sys.stderr)
                return 4
            rec = json.loads(cand.read_text(encoding="utf-8"))
            rec.setdefault("job_id", a.job)
        else:
            rec = json.loads((ROOT / "evidence" / "ibm_live_job.json")
                             .read_text(encoding="utf-8"))
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService()
        job = svc.job(rec["job_id"])
        st = str(job.status())
        if st not in ("DONE", "JobStatus.DONE"):
            print(f"REFUSING: job {rec['job_id']} is {st}, not DONE. A "
                  f"re-fetch is free only because the work is finished.",
                  file=sys.stderr)
            return 3
        d = int(rec["config"]["distance"])
        r = int(rec["config"]["rounds"])
        backend = rec["backend"]
        mode = f"refetch({rec['job_id']})"
        circ = job.inputs["pubs"][0][0]
        idx = list(circ.layout.initial_index_layout())[: 2 * d - 1]
        data_q, anc_q = idx[:d], idx[d: 2 * d - 1]
        cal = snapshot_calibration(svc.backend(backend), idx)
        edges, prov = rep_code_edges(cal, d, r, data_q, anc_q)
        cut = logical_cut_edges(d, r)
        from rep_code_cert import detectors_from_records, truth_column
        res = job.result()[0]
        # ONE REGISTER PER ROUND, `m0`..`m{r-1}`, not a single `anc`
        # array. Read the way `recertify_calibrated` reads it, because
        # inventing a shape here is how a run gets decoded against the
        # wrong bits and still returns numbers.
        anc_regs = [np.array(getattr(res.data, f"m{t}").to_bool_array(),
                             dtype=np.uint8) for t in range(r)]
        fin = np.array(res.data.fin.to_bool_array(), dtype=np.uint8)
        det = np.stack([
            detectors_from_records(
                np.stack([anc_regs[t][sh] for t in range(r)]),
                fin[sh]).reshape(-1)
            for sh in range(fin.shape[0])])
        # A repetition-code memory prepared in |0..0> does NOT withhold
        # its ground truth: the destructive final readout carries it.
        truth = fin[:, truth_column(d)]
        cal_note = "device calibration snapshot; weights are log((1-p)/p)"
        convention = LOG_LIKELIHOOD

    if a.scales:
        # *** THE DECLARED CALIBRATION LAYER (R14-R18). *** One weight scale per
        # edge family, fitted once on half a run and fixed before the next,
        # cut the certifying decoder's logical errors 2-7x on runs it never
        # saw. Applied here EXACTLY to the pinned rationals; the float edges the
        # decoder uses are derived from those, so `--from-artifact` re-scores the
        # very numbers this run decoded with. It is a different model from the
        # calibration snapshot's, and weight_model says so: every certificate
        # below is a certificate about THIS model.
        if a.from_artifact:
            print("REFUSING: --scales with --from-artifact: the artifact's pinned "
                  "weights are already what was decoded with; re-scaling them "
                  "would score a model the artifact does not record",
                  file=sys.stderr)
            return 5
        scales = parse_scales(a.scales)
        positions = d - 1
        for key, rec in prov["edges"].items():
            head, _, tail = key.partition("-")
            u, v = int(head), int(tail)
            fam = edge_family(u, v, positions)
            factor = Fraction(str(scales.get(fam, 1)))
            rec["weight_exact"] = str(Fraction(rec["weight_exact"]) * factor)
            rec["weight"] = float(Fraction(rec["weight_exact"]))
            rec["family"] = fam
            rec["family_scale"] = float(factor)
        edges = [(int(k.partition("-")[0]), int(k.partition("-")[2]),
                  float(Fraction(rec["weight_exact"])))
                 for k, rec in prov["edges"].items()]
        prov["family_scales"] = dict(scales)
        prov["family_scales_provenance"] = a.scales_provenance
        cal_note += (f" x per-family scales {scales} -- a DECLARED calibration "
                     f"layer (provenance: {a.scales_provenance or 'unstated'}); "
                     f"the pinned weights ARE the scaled ones")

    if a.burst_cost is not None:
        # *** THE DECLARED BURST MODEL (R21). *** Not a scale: a PRICE. Under
        # a burst the per-round probability is not what sets the cost, so the
        # time-like weights are replaced rather than multiplied, and the
        # multi-round edges the DEM never had are added at the same price.
        # Pinned exactly, so `--from-artifact` re-scores the graph this run
        # decoded with -- including the edges that did not exist before.
        if a.from_artifact:
            print("REFUSING: --burst-cost with --from-artifact: the artifact's "
                  "pinned graph is already what was decoded with; adding edges "
                  "would score a model the artifact does not record",
                  file=sys.stderr)
            return 5
        cost = Fraction(str(a.burst_cost))
        positions, n_rounds = d - 1, r + 1
        for key, rec in prov["edges"].items():
            head, _, tail = key.partition("-")
            if edge_family(int(head), int(tail), positions) == "time-like":
                rec["weight_exact"] = str(cost)
                rec["weight"] = float(cost)
                rec["burst_priced"] = True
        for (u, v, _w) in burst_edges(positions, n_rounds, float(cost)):
            prov["edges"][f"{u}-{v}"] = {"weight_exact": str(cost), "weight": float(cost),
                                         "from": "burst model (R21)", "span_rounds":
                                         abs(u - v) // positions}
        edges = [(int(k.partition("-")[0]), int(k.partition("-")[2]),
                  float(Fraction(rec["weight_exact"])))
                 for k, rec in prov["edges"].items()]
        prov["burst_cost"] = float(cost)
        cal_note += (f" + a DECLARED BURST model at cost {float(cost)}: time-like "
                     f"weights REPLACED and multi-round time-like edges added, "
                     f"because a burst costs one event whatever its span (R21)")

    n_nodes = (r + 1) * (d - 1)
    records, preds, gaps, correct = score(
        edges, cut, det, truth, n_nodes, budget_s=a.budget)
    if not records:
        print("REFUSING: no nontrivial shots", file=sys.stderr)
        return 4

    floor = device_floor(records,
                         convention=convention or "undeclared")
    weak = device_floor([{"primal": x["primal"], "bound": x["bound"] * 0.6,
                          "accepted": False} for x in records],
                        convention=convention or "undeclared")
    gameable = is_gameable_by_weakening_the_dual(floor, weak)

    cal_rep = calibration_report([float(g) for g in gaps], correct)
    powered = [b["bin_low"] for b in cal_rep.bins if b.get("has_power")]
    max_gap = max(powered) if powered else 0.0
    # *** MEASURED CONFIDENCE, NOT THE MODEL'S. ***
    # The clamped model estimate is kept beside it because the two
    # disagreeing IS the finding: the model claims more confidence than
    # the shots support, and only the measured one stays in range.
    est, est_refusal = None, None
    model_est = None
    try:
        est = measured_estimate(preds, gaps, correct)
    except EstimatorRefusal as e:
        est_refusal = f"UNCHECKABLE: {e}"
    try:
        model_est = weighted_estimate(preds, gaps,
                                      max_calibrated_gap=max_gap)
    except EstimatorRefusal:
        model_est = None
    split = parity_calibration_split(preds, gaps, correct)

    # *** IS THE i.i.d. ASSUMPTION TRUE ON THIS DEVICE? ***
    #
    # Every DEM assumes it, dem_calibration assumes it, and the
    # aggregate in certified_weight assumes it -- a mean over correlated
    # shots does not have the variance reported for it. It is also the
    # rare assumption testable with NO MODEL: under i.i.d. shot order is
    # exchangeable, so permuting the observed values gives an EXACT
    # null.
    #
    # This needs shot ORDER, which is why the detector array is stored
    # in submission order rather than sorted or deduplicated.
    weights = det.astype(int).sum(axis=1).astype(float).tolist()
    drift, drift_refusal = None, None
    try:
        drift = permutation_test(weights, lag1_autocorrelation,
                                 name="lag1_autocorrelation").to_json()
    except ExchangeabilityRefusal as e:
        drift_refusal = str(e)
    try:
        blocks = block_scan(weights, permutations=4000)
    except ExchangeabilityRefusal as e:
        blocks = {"refused": str(e)}

    doc = {
        "gate": "hardware-noise-floor",
        "status": "MEASURED",
        "mode": mode, "backend": backend,
        "qpu_seconds": 0.0,
        "why_zero_qpu": (
            "a completed IBM job is re-fetchable, so the first hardware "
            "reading of this instrument costs nothing. Scoring is not a "
            "device act"),
        "circuit": {"code": "repetition-code memory", "distance": d,
                    "rounds": r, "shots": int(det.shape[0]),
                    "nontrivial": len(records)},
        "weight_model": cal_note,
        # PINNED, so `--from-artifact` reproduces this run exactly.
        "edge_provenance": prov,
        "certified_weight": floor.to_json(),
        "survivorship_gap": survivorship_gap(floor),
        "anti_gaming": {"weakened_dual_scale": 0.6,
                        "bound_fell": floor.mean_lower - weak.mean_lower,
                        "interval_widened": (weak.mean_width
                                             - floor.mean_width),
                        "gameable": gameable},
        "soft_estimate": (est.to_json() if est is not None
                          else {"status": est_refusal}),
        "calibrated_gap_ceiling": max_gap,
        "soft_estimate_from_MODEL_confidence": (
            model_est.to_json() if model_est is not None
            else {"status": "refused"}),
        "why_two_estimates": (
            "the measured estimate weights by confidence the shots "
            "actually demonstrated; the model estimate weights by "
            "1/(1+e^gap). Where they disagree, the model is claiming "
            "confidence the data does not carry -- and an |s| > 1 on "
            "either is proof its reliabilities are wrong, since "
            "(-1)^L cannot leave [-1, 1]"),
        "parity_calibration_split": split,
        "dem_calibration": cal_rep.to_json(),
        "exchangeability": {
            "lag1": drift or {"refused": drift_refusal},
            "block_scan": blocks,
            "why": ("an EXACT test of the assumption every DEM rests on. "
                    "Under i.i.d. shot order carries no information, so "
                    "permuting the observed syndrome weights gives the "
                    "null outright -- no distributional family to be "
                    "wrong about. A large p here means only that nothing "
                    "ABOVE the reported sensitivity was found"),
        },
        # *** THE SHOTS, KEPT. *** Every number above is recomputable
        # offline from this array by someone who has only this file.
        "detectors_packed": [int(x) for x in
                             np.packbits(det, axis=None).tolist()],
        "detector_shape": list(det.shape),
        "truth": ([int(x) for x in np.asarray(truth).tolist()]
                  if truth is not None else None),
        "does_not_claim": (
            "on --dry-run, ANYTHING about a device: the shots are "
            "sampled from the same edge model that decodes them, which "
            "is the circularity DO-1 is declared a proxy for. On "
            "--refetch the shots are the device's and the model is the "
            "device's published calibration, so neither generated the "
            "other -- but the calibration is a SNAPSHOT and the job is "
            "one session"),
        "passed": bool(not gameable),
        "seconds": round(time.time() - t0, 3),
    }
    if a.from_artifact:
        # *** A VERIFIER MUST NOT WRITE TO WHAT IT VERIFIES. ***
        #
        # The first version of this branch rewrote the artifact, and two
        # re-scores turned `mode` into
        # "from-artifact(from-artifact(dry-run))" -- the instrument
        # recording its own passes over the subject. It also meant a
        # failed re-score would have overwritten the evidence of the
        # disagreement with the disagreeing numbers.
        #
        # So this mode reads, recomputes, and reports. Nothing is
        # written.
        bad = _claims_differ(old, doc)
        print(f"  VERIFY {a.json}")
        print(f"    origin {old.get('mode')} on {old.get('backend')}, "
              f"re-scored offline from {len(pinned)} pinned weights")
        if bad:
            print(f"    *** {len(bad)} PUBLISHED CLAIM(S) DID NOT "
                  f"REPRODUCE ***")
            for name, was, now in bad:
                print(f"      {name}: published {was!r}, recomputed {now!r}")
            print("    The stored shots and the stored weights are the "
                  "only inputs, so a difference here means the CODE "
                  "changed its answer -- not the device.")
            return 6
        print(f"    {_CLAIM_COUNT} published claims reproduced EXACTLY "
              f"from the stored shots and pinned weights")
        print("    (`seconds` is wall clock and is not a claim; the "
              "numbers are)")
        return 0

    out = ROOT / a.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(doc, indent=2) + "\n").encode("utf-8"))

    cw = doc["certified_weight"]
    lo, hi = cw["proven_interval_on_mean_min_weight"]
    print(f"  mode {mode}  backend {backend}  d={d} r={r}")
    print(f"  shots {det.shape[0]}  nontrivial {len(records)}  "
          f"certified {cw['certified_fraction']:.4f}")
    print(f"  proven interval on mean min-weight : [{lo:.4f}, {hi:.4f}]"
          f"  width {cw['mean_width']:.4f}")
    print(f"  reading : {cw['physical_reading'][:88]}")
    print(f"  anti-gaming gameable={gameable}")
    if est is None:
        print(f"  soft estimate : {est_refusal[:110]}")
    else:
        flag = "  *** OUT OF RANGE ***" if est.out_of_range else ""
        print(f"  soft estimate {est.value:+.4f}  multiplier "
              f"{est.effective_shot_multiplier:.3f}  ceiling {max_gap}"
              f"{flag}")
        if est.out_of_range:
            print("      |s| > 1 is impossible for (-1)^L, so this is "
                  "NOT a measurement -- it proves the clamped "
                  "reliabilities understate the true confidence and the "
                  "debias over-corrects")
    if drift:
        print(f"  i.i.d. (lag-1)  r1={drift['observed']:+.5f}  "
              f"p={drift['p_value']:.4f}  "
              f"sensitivity {drift['sensitivity_3sigma']:.4f}")
        if drift["p_value"] >= 0.05:
            print(f"      no order effect ABOVE "
                  f"{drift['sensitivity_3sigma']:.3f} -- which is not the "
                  f"same as the device being stable")
    print(f"  DEM calibration: {cal_rep.verdict[:100]}")
    print(f"  err={cal_rep.calibration_error:.4f} "
          f"powered_bins={cal_rep.powered_bins}")
    print(f"hardware-noise-floor -> {out}")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
