r"""Certified decoding on LIVE IBM hardware -- the vendor duo's second half.

ONE_Q has certified 219,384 real hardware shots -- all of them Google's
published archive. This experiment runs a repetition-code memory on an IBM
Heron device, collects fresh syndromes from real hardware, and certifies
the decoding with the same instrument. Design choices, each with its
reason:

  REPETITION CODE, d=9, r=5 (17 qubits of 156): native to heavy-hex (any
  path through the coupling map), cheap in credits, and its decoding graph
  is 1-D matching -- the certifier's home turf. The claim is the same as
  every ONE_Q artifact: per-shot min-weight conformance against DECLARED
  weights (uniform, stated -- unlike the Google corpus, the error model
  here is ours and says so).

  THREE MODES, and only one talks to IBM:
    --validate   the ENTIRE pipeline against a stim mirror of the same
                 experiment -- sample, decode, certify, observable check --
                 no network, no credits, and the gate that must be green
                 before --submit is even legal.
    --submit     OWNER-GATED TWICE: requires both the flag
                 --i-understand-this-spends-credits AND the environment
                 variable ONEQ_IBM_SUBMIT=1. Pre-registers (hashes circuit
                 QASM + config + analysis plan, prints the hash) BEFORE
                 submission, then saves the job id.
    --fetch      retrieves a completed job, rebuilds detectors from the
                 per-round classical registers, decodes, certifies, and
                 writes the artifact.

  THE RECEIPT FIXES THE ESTATE'S RECORDED DEFECTS: the atlas documents the
  earlier IBM tournament receipts as (1) signed with the all-zeros Ed25519
  seed and (2) missing any device field. Here provider/backend/job_id are
  first-class fields of the signed core (passport's schema), the weak-key
  registry refuses the all-zeros seed at mint, and the artifact carries the
  pre-registration hash minted before the job existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import BOUNDARY, CertifyingDecoder  # noqa: E402
# The SAME canonical edge key and the SAME parity rule the rest of the
# estate uses. Rebuilding either here is how two copies of one predicate
# start disagreeing about which logical they are measuring.
from oneq.logical_cut import _key as _lc_key                    # noqa: E402
from oneq.logical_cut import logical_parity                     # noqa: E402

D = 9            # code distance (data qubits)
R = 5            # noisy rounds
SHOTS = 4000


def decode_graph(d: int, r: int):
    """The 1-D space-time matching graph: detectors (i, t), uniform weights.

    Node id = t * (d-1) + i for detector column i in round t (r+1 rounds of
    detectors: r ancilla rounds + the final data-derived round). Edges:
    time-like (same i, adjacent t), space-like (adjacent i, same t), and
    boundary edges at the chain ends.
    """
    def nid(i, t):
        return t * (d - 1) + i

    edges = []
    for t in range(r + 1):
        for i in range(d - 1):
            if t + 1 <= r:
                edges.append((nid(i, t), nid(i, t + 1), 1.0))
            if i + 1 < d - 1:
                edges.append((nid(i, t), nid(i + 1, t), 1.0))
        edges.append((nid(0, t), BOUNDARY, 1.0))
        edges.append((nid(d - 2, t), BOUNDARY, 1.0))
    return edges


def logical_cut_edges(d: int, r: int) -> frozenset:
    r"""The edges whose parity IS the logical class: the RIGHT boundary.

    *** THIS LANE HAD NO OBSERVABLE, AND ITS DOCSTRING SAID IT DID. ***
    The header above advertises `--validate` as "sample, decode, certify,
    observable check"; `certify_shots` only ever saw detectors, and
    `mode_fetch` read the final data register `fin` solely to build the
    last detector row and then threw it away. So the run reported a
    certified-conformance rate and never a LOGICAL ERROR RATE, and the
    estate's note that "a real device withholds ground truth" was read as
    a fact about hardware when it was a fact about this pipeline.

    A repetition-code MEMORY prepared in |0...0> does not withhold it.
    The circuit ends with `qc.measure(data, cfin)`, a destructive readout
    of every data qubit, so `fin` IS the accumulated X-error pattern.
    Z_L is a single-qubit Z, the coset is {e, e XOR 11...1}, and any one
    data qubit separates them -- so the truth is a column of `fin`,
    already fetched and thrown away.

    WHICH column, and which edges, is a CONVENTION, and a convention that
    disagrees with itself measures two different logicals. An X error on
    data qubit j fires detector columns j-1 and j, so only the two END
    qubits fire a single column: qubit 0 fires column 0 (the LEFT
    boundary) and qubit d-1 fires column d-2 (the RIGHT boundary).
    Either is a valid representative, provided the cut and the truth
    column name the SAME one.

    *** MEASURED, NOT ASSUMED, AND THE FIRST VERSION HAD IT BACKWARDS.
    *** This returned the LEFT boundary on the reasoning above. Against
    stim's own observable on the mirror the left cut disagreed on
    0.169 of shots and the right cut on 0.000 of 219 -- so stim's
    observable is the qubit-(d-1) representative, and the lane now uses
    that one end to end: this cut, and `fin[:, d-1]` on hardware. The
    check that caught it is in `mode_validate` and scores BOTH
    boundaries, because a convention that cannot be seen to be wrong has
    not been tested.
    """
    return frozenset(_lc_key(t * (d - 1) + (d - 2), BOUNDARY)
                     for t in range(r + 1))


#: The data-qubit column of `fin` that the cut above corresponds to. It
#: is derived from `d` rather than written as a literal so the two can
#: never be edited apart.
def truth_column(d: int) -> int:
    return d - 1


def opposite_cut_edges(d: int, r: int) -> frozenset:
    """The other boundary. Exists only to prove the cut discriminates."""
    return frozenset(_lc_key(t * (d - 1), BOUNDARY) for t in range(r + 1))


def detectors_from_records(anc_rounds: np.ndarray,
                           final_data: np.ndarray) -> np.ndarray:
    """anc_rounds: (r, d-1) 0/1; final_data: (d,) 0/1 -> (r+1, d-1)."""
    r, dm1 = anc_rounds.shape
    det = np.zeros((r + 1, dm1), dtype=np.uint8)
    det[0] = anc_rounds[0]
    for t in range(1, r):
        det[t] = anc_rounds[t] ^ anc_rounds[t - 1]
    final_par = np.array([final_data[i] ^ final_data[i + 1]
                          for i in range(dm1)], dtype=np.uint8)
    det[r] = final_par ^ anc_rounds[r - 1]
    return det


def certify_shots(det_all: np.ndarray, d: int, r: int, *, hard: bool,
                  edges=None, observed=None, ledger=None):
    """det_all: (shots, r+1, d-1). Returns summary dict.

    `edges` overrides the uniform graph. It exists because the frozen
    preregistration pins "Model N: the device's own DEM" while this lane
    decoded against weight 1.0 everywhere -- D1 in
    `docs/PHASE6_DEVIATIONS.md`. The model was never the hard part; the
    lane simply had no way to be handed one. Now it does, and the next
    run's preregistration can declare a model this code can actually
    honour.
    """
    edges = decode_graph(d, r) if edges is None else edges
    cd = CertifyingDecoder(edges)
    import pymatching
    m = pymatching.Matching()
    for (u, v, w) in edges:
        if v == BOUNDARY:
            m.add_boundary_edge(int(u), weight=w)
        else:
            m.add_edge(int(u), int(v), weight=w)
    n_nodes = (r + 1) * (d - 1)
    nontrivial = certified = 0
    t_cert = 0.0
    cut = logical_cut_edges(d, r)
    rcut = opposite_cut_edges(d, r)
    # A: the heuristic's logical class. Y: the truth, when supplied.
    # E is not a third computation -- `certify_hard` PROVES the submitted
    # correction is minimum weight, so on an accepted shot E == A by
    # theorem, and on a refused one E is simply unknown to this tier.
    obs_stats = {"observed_shots": 0, "logical_errors": 0,
                 "logical_errors_opposite_cut": 0,
                 # DISCORDANT PAIRS, because the two cuts are scored on
                 # the SAME shots. Comparing them with two independent
                 # variances inflates the error bar by ~2x and the first
                 # version of this check failed a 0.000-vs-0.169 result
                 # on exactly that mistake.
                 "declared_wrong_only": 0, "opposite_wrong_only": 0,
                 "delta_S_certified_zero": 0, "delta_S_unknown": 0}
    for s in range(det_all.shape[0]):
        flat = det_all[s].reshape(-1)
        fired = [int(i) for i in np.flatnonzero(flat)]
        if not fired:
            continue
        nontrivial += 1
        syn = np.zeros(n_nodes, dtype=np.uint8)
        syn[fired] = 1
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                     for a, b in m.decode_to_edges_array(syn))
        t0 = time.perf_counter()
        res = (cd.certify_hard(fired, corr) if hard
               else cd.certify(fired, corr))
        t_cert += time.perf_counter() - t0
        certified += res.accepted
        if observed is not None:
            y = int(observed[s]) & 1
            a_par = logical_parity(corr, cut)
            obs_stats["observed_shots"] += 1
            obs_stats["logical_errors"] += int(a_par != y)
            # THE SAME SHOT SCORED AGAINST THE OTHER BOUNDARY. A cut that
            # cannot be wrong in a visible way has not been tested, and
            # these two must NOT agree: if they do, the parity rule is
            # reading something that is not the logical class.
            opp_wrong = logical_parity(corr, rcut) != y
            dec_wrong = a_par != y
            obs_stats["logical_errors_opposite_cut"] += int(opp_wrong)
            obs_stats["declared_wrong_only"] += int(dec_wrong
                                                    and not opp_wrong)
            obs_stats["opposite_wrong_only"] += int(opp_wrong
                                                    and not dec_wrong)
            if res.accepted:
                obs_stats["delta_S_certified_zero"] += 1
            else:
                obs_stats["delta_S_unknown"] += 1
            if ledger is not None:
                ledger.append({"shot": s, "a": a_par, "y": y,
                               "e": a_par if res.accepted else None,
                               "e_is": ("A, PROVEN minimum weight"
                                        if res.accepted else
                                        "UNKNOWN: this shot was refused")})
    # WALL CLOCK IS SEGREGATED, because everything else here is a function
    # of the shots and the graph and therefore re-derivable byte-for-byte,
    # while this is not. Mixed in with the counts it silently made the whole
    # artifact non-reproducible: a re-fetch of the SAME job moved
    # cert_ms_per_shot 2.417 -> 4.388 with every scientific field
    # identical, so a stranger re-deriving it would see bytes that differ
    # and have no way to tell a timing jitter from a tampered record.
    # Named separately, a verifier knows which fields must match exactly and
    # which cannot.
    out = {"shots": int(det_all.shape[0]), "nontrivial": nontrivial,
           "certified": certified,
           "certified_rate": certified / max(nontrivial, 1)}
    if observed is not None:
        n_obs = max(obs_stats["observed_shots"], 1)
        out["observable"] = {
            **obs_stats,
            "logical_error_rate": obs_stats["logical_errors"] / n_obs,
            "logical_error_rate_opposite_cut": (
                obs_stats["logical_errors_opposite_cut"] / n_obs),
            "delta_S": 0.0 if obs_stats["delta_S_unknown"] == 0 else None,
            "delta_S_note": (
                "0 by THEOREM, not by estimate: every scored shot's "
                "correction was certified minimum weight, so E == A and "
                "L_A - L_E = 0 on each of them. A refused shot leaves E "
                "unknown to this tier and delta_S with it"
                if obs_stats["delta_S_unknown"] == 0 else
                f"{obs_stats['delta_S_unknown']} shot(s) were refused, so "
                f"E is unknown there and delta_S is not determined"),
            "does_not_claim": (
                "the other three budget terms. delta_O, R_I and delta_M "
                "all need the model's P(Y=1|s), which is a ratio of exact "
                "COSET MASSES -- and `bayes_coset` refuses above kernel "
                "dimension 22 while this graph's kernel is far larger. "
                "Search is the only term this lane can settle, and it "
                "settles it exactly")}
    out["nonreproducible"] = {
        "note": ("wall-clock, machine- and load-dependent; NOT part "
                 "of the re-derivable record and never compared"),
        "cert_ms_per_shot": 1e3 * t_cert / max(nontrivial, 1)}
    return out


def build_qiskit_circuit(d: int, r: int):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    data = QuantumRegister(d, "d")
    anc = QuantumRegister(d - 1, "a")
    cregs = [ClassicalRegister(d - 1, f"m{t}") for t in range(r)]
    cfin = ClassicalRegister(d, "fin")
    qc = QuantumCircuit(data, anc, *cregs, cfin)
    for t in range(r):
        for i in range(d - 1):
            qc.cx(data[i], anc[i])
            qc.cx(data[i + 1], anc[i])
        for i in range(d - 1):
            qc.measure(anc[i], cregs[t][i])
        if t < r - 1:
            for i in range(d - 1):
                qc.reset(anc[i])
        qc.barrier()
    qc.measure(data, cfin)
    return qc


def prereg_hash(qc, config: dict) -> str:
    from qiskit import qasm3
    blob = qasm3.dumps(qc) + json.dumps(config, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def mode_validate(a) -> int:
    """The stim mirror: same code, same rounds, measurement noise -- the
    ENTIRE pipeline proven offline before any credit is spent."""
    import stim
    circ = stim.Circuit.generated(
        "repetition_code:memory", distance=a.distance, rounds=a.rounds,
        before_measure_flip_probability=a.p,
        before_round_data_depolarization=a.p)
    dem = circ.detector_error_model()
    sampler = circ.compile_detector_sampler(seed=20260802)
    # SEPARATE_OBSERVABLES IS THE WHOLE FIX. Without it this returned
    # detectors only, and the "observable check" in this file's header
    # was a sentence describing work nothing did.
    det, obs = sampler.sample(a.shots, separate_observables=True)
    # stim's detector layout for repetition_code:memory is (round-major,
    # (d-1) per round, r+1 rounds) -- exactly our (r+1, d-1) reshape
    det3 = det.astype(np.uint8).reshape(det.shape[0], a.rounds + 1,
                                        a.distance - 1)
    ledger = []
    out = certify_shots(det3, a.distance, a.rounds, hard=True,
                        observed=np.asarray(obs, dtype=np.uint8)[:, 0],
                        ledger=ledger)
    out.update({"mode": "validate(stim mirror)", "p": a.p,
                "dem_mechanisms": dem.num_errors})
    print(json.dumps(out, indent=2))
    ok = out["certified_rate"] >= 0.999 if out["nontrivial"] else False

    # *** THE OBSERVABLE CHECK, AND IT CALIBRATES ITS OWN CONVENTION. ***
    # A cut naming the wrong boundary would measure a different logical
    # and report a logical error rate near 1 instead of near p -- so the
    # check is not "is there a number" but "is the number on the right
    # side, and does the OTHER boundary disagree". If both cuts agreed,
    # the parity rule would be reading something that is not the logical
    # class at all, and neither number would mean anything.
    ob = out.get("observable", {})
    n_obs = max(ob.get("observed_shots", 0), 1)
    lr = ob.get("logical_error_rate")
    lr_opp = ob.get("logical_error_rate_opposite_cut")
    # THE BAND IS THE PAIRED DIFFERENCE'S OWN ERROR BAR, NOT A CHOSEN
    # NUMBER -- AND NOT AN UNPAIRED ONE. Both cuts are scored on the SAME
    # shots, so only the DISCORDANT pairs carry information: with b
    # shots where only the declared cut is wrong and c where only the
    # opposite is, the difference is (b - c)/n with standard error
    # sqrt(b + c)/n. Treating them as two independent rates inflated the
    # bar to 0.191 and failed a 0.000-vs-0.169 result, which is the
    # cleanest separation this check can ever see.
    b = ob.get("declared_wrong_only", 0)
    c = ob.get("opposite_wrong_only", 0)
    se = (b + c) ** 0.5 / n_obs
    gap_needed = max(4.0 * se, 1.0 / n_obs)
    cuts_differ = (lr is not None and lr_opp is not None
                   and (lr_opp - lr) > gap_needed)
    # A d=9 memory at this p must essentially never fail logically. If
    # the DECLARED cut is the one tracking the observable, its rate is
    # tiny; a rate that is merely SMALLER than the other one is not
    # enough, because both could be reading something unrelated.
    declared_tracks = lr is not None and lr < 0.02
    obs_ok = cuts_differ and declared_tracks and ob.get("delta_S") == 0.0
    print(f"OBSERVABLE: logical error rate {lr} on the declared cut vs "
          f"{lr_opp} on the opposite boundary over {n_obs} shots "
          f"(gap must exceed {gap_needed:.4f}); delta_S {ob.get('delta_S')}")
    if not obs_ok:
        print("OBSERVABLE: RED -- either the declared cut does not track "
              "the observable, the two boundaries do not measurably "
              "differ (so the parity rule is not reading a logical class "
              "at all), or a refused shot left delta_S undetermined.")
    ok = ok and obs_ok
    out["observable_check"] = {
        "passed": obs_ok,
        "cuts_differ": cuts_differ, "declared_tracks": declared_tracks,
        "gap_required": gap_needed,
        "rule": ("the declared cut must disagree with stim's observable "
                 "on under 2% of shots AND the opposite boundary must "
                 "differ from it by more than 4 standard errors of the "
                 "difference -- a cut nobody can see to be wrong has not "
                 "been tested, and the first version of this file named "
                 "the WRONG boundary")}
    print("VALIDATE:", "GREEN -- pipeline proven offline"
          if ok else "RED -- do not submit until this is green")
    # *** A TEST OVERWROTE PUBLISHED EVIDENCE. *** This wrote the
    # committed slot unconditionally, so ANY caller replaced it as a side
    # effect of running -- and `test_the_stim_mirror_validates_the_
    # convention_end_to_end` is a caller. It ran 300 shots over the
    # committed run's 400 and silently swapped the smaller numbers into
    # `evidence/ibm_live_validate.json` mid-suite. Caught by
    # `test_published_bytes`, which exists for exactly this and worked.
    #
    # So the destination is now an ARGUMENT with the published slot as
    # its default: the CLI lane still writes the evidence it is supposed
    # to write, and anything exploratory says where its output goes.
    dest = (pathlib.Path(getattr(a, "out", None))
            if getattr(a, "out", None)
            else ROOT / "evidence" / "ibm_live_validate.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    return 0 if ok else 1


# *** TEXT MODE TURNS EVERY LF INTO CRLF ON WINDOWS. ***
#
# MEASURED 2026-08-26: a `--mode fetch` re-run rewrote
# `d9su5b9dsedc73ai79jg.certified.json` with 32 CRLF and 0 bare LF,
# against the 14 bare LF git had stored. The CONTENT was identical and
# correct -- the same 4,000 shots, the same 576 certified -- so the only
# symptom was `test_stored_bytes_equal_worktree_bytes` going red on a
# file nobody meant to touch, and a whole-file diff that looks like a
# rewrite and is a line-ending flip.
#
# Four of this module's five writers were text-mode. The fifth -- the
# per-shot ledger -- already passed `newline=""` and was the only one
# producing correct bytes, which is why the ledger was clean and its
# neighbours were not. `write_bytes` cannot be got wrong by a platform
# default, so all of them use it now.


def _archive(rec: dict, kind: str) -> None:
    """Write an IMMUTABLE per-run copy alongside the mutable latest-pointer.

    `ibm_live_job.json` and `ibm_live_certified.json` are single slots, and
    every run overwrote them. That is fine for "what ran last" and fatal for
    anything that CITED them: the 2026-08-10 marrakesh run silently destroyed
    the artifact backing `docs/PAPER_DRAFT.md`'s published claim of 2,296 of
    2,296 certified shots on ibm_fez -- caught only because the claims
    provenance gate refused to let the push through with an orphan number.

    A run's own record must not be erasable by the next run, so each one also
    lands at evidence/ibm_live/<job_id>.<kind>.json, which is keyed by
    something no later run can collide with. Same class as the recovery that
    reused a log path and truncated the black box it was meant to preserve.
    """
    jid = rec.get("job_id")
    if not jid:
        return
    # The MODEL is part of the archive key. Without it, re-fetching the same
    # job under `--model calibrated` would overwrite the uniform artifact at
    # the same path -- reintroducing, one commit later, exactly the defect
    # that destroyed the ibm_fez 2,296/2,296 evidence. One decode, one file.
    model = rec.get("model")
    suffix = f"{kind}.{model}" if model and model != "uniform" else kind
    p = ROOT / "evidence" / "ibm_live" / f"{jid}.{suffix}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))


def mode_submit(a) -> int:
    import os
    if not (a.i_understand_this_spends_credits
            and os.environ.get("ONEQ_IBM_SUBMIT") == "1"):
        print("REFUSED: submission spends the owner's IBM credits.\n"
              "Required: --i-understand-this-spends-credits AND "
              "ONEQ_IBM_SUBMIT=1 in the environment.\n"
              "Run --validate first; it must be GREEN.")
        return 3
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager)
    qc = build_qiskit_circuit(a.distance, a.rounds)

    # RESOLVE THE BACKEND AND LAYOUT *BEFORE* THE CONFIG LITERAL.
    # config named `layout` nineteen lines before it was assigned, so
    # mode_submit raised NameError unconditionally -- after the credit
    # guard had been satisfied and before prereg_hash could be minted.
    # The whole executable path was unreachable, and there was no test
    # for it.
    #
    # The ordering constraint the preregistration actually imposes is
    # that the hash precede SUBMISSION, not that it precede layout
    # selection. Nothing below touches the QPU until SamplerV2.run.
    svc = QiskitRuntimeService()
    # A PINNED LAYOUT IS A COMMITMENT THE CODE MUST BE ABLE TO KEEP.
    # Phase 7 pins specific physical qubits so the calibrated model differs
    # from uniform enough to be detectable, and so the choice cannot be made
    # after the fact. Declaring that while submit() only knew least_busy
    # would have repeated D1 exactly: a preregistration naming something the
    # executable path cannot do.
    if a.backend:
        backend = svc.backend(a.backend)
    else:
        backend = svc.least_busy(operational=True, simulator=False,
                                 min_num_qubits=2 * a.distance)
    layout = None
    if a.layout:
        layout = [int(q) for q in a.layout.split(",")]
        need = 2 * a.distance - 1
        if len(layout) != need:
            print(f"REFUSED: --layout needs {need} qubits "
                  f"(d data then d-1 ancilla), got {len(layout)}")
            return 3
        # Refuse rather than silently re-site. Phase 7 says an unavailable
        # pinned layout POSTPONES the run, because re-siting restores the
        # freedom the pin exists to remove.
        props = backend.target
        for q in layout:
            if q >= backend.num_qubits:
                print(f"REFUSED: qubit {q} not on {backend.name}")
                return 3
            e = props["measure"].get((q,)) if "measure" in props else None
            if e is not None and e.error is not None and e.error > 0.05:
                print(f"REFUSED: qubit {q} readout error {e.error:.4f} > 0.05 "
                      f"-- the pinned layout is unhealthy; POSTPONE, do not "
                      f"re-site")
                return 3
        print(f"layout PINNED: data={layout[:a.distance]} "
              f"ancilla={layout[a.distance:]}")

    # NOW the config is complete, so the preregistration can bind what
    # will actually run -- including the pinned layout, which is the
    # commitment Phase 7 exists to make. Minted before submission.
    config = {"distance": a.distance, "rounds": a.rounds, "shots": a.shots,
              "layout": layout,
              "code": "repetition(bit-flip)", "weights": "uniform, declared",
              "analysis": "detectors from round diffs; pymatching decode; "
                          "certify_hard per shot; artifact schema "
                          "oneq-ibm-live/1"}
    h = prereg_hash(qc, config)
    print(f"PRE-REGISTRATION sha256 = {h}  (minted BEFORE submission)")

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                      initial_layout=layout)
    isa = pm.run(qc)
    job = SamplerV2(mode=backend).run([isa], shots=a.shots)
    rec = {"prereg_sha256": h, "provider": "ibm_quantum",
           "backend": backend.name, "job_id": job.job_id(),
           "config": config,
           "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())}
    dest = ROOT / "evidence" / "ibm_live_job.json"
    dest.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    _archive(rec, "job")
    print(f"submitted to {backend.name}: job {job.job_id()}\n-> {dest}")
    return 0


def mode_fetch(a) -> int:
    from qiskit_ibm_runtime import QiskitRuntimeService
    rec = json.loads((ROOT / "evidence" / "ibm_live_job.json")
                     .read_text(encoding="utf-8"))
    svc = QiskitRuntimeService()
    job = svc.job(rec["job_id"])
    if job.status() not in ("DONE", "JobStatus.DONE"):
        print(f"job {rec['job_id']} status: {job.status()} -- not done")
        return 4
    res = job.result()[0]
    d, r = rec["config"]["distance"], rec["config"]["rounds"]
    anc_regs = [np.array(getattr(res.data, f"m{t}").to_bool_array(),
                         dtype=np.uint8) for t in range(r)]
    fin = np.array(res.data.fin.to_bool_array(), dtype=np.uint8)
    shots = fin.shape[0]
    det3 = np.zeros((shots, r + 1, d - 1), dtype=np.uint8)
    for s in range(shots):
        anc_rounds = np.stack([anc_regs[t][s] for t in range(r)])
        det3[s] = detectors_from_records(anc_rounds, fin[s])
    edges = None
    weights_note = ("uniform, declared (the error model is OURS and "
                    "says so -- unlike the Google corpus)")
    cal_snapshot = None
    if getattr(a, "model", "uniform") == "calibrated":
        # The layout comes from the SUBMITTED circuit, never a fresh
        # transpile: re-transpiling could pick different physical qubits and
        # would silently weight a different experiment than the one that ran.
        from qiskit_ibm_runtime import QiskitRuntimeService

        from oneq.device_dem import rep_code_edges, snapshot_calibration
        circ = job.inputs["pubs"][0][0]
        idx = list(circ.layout.initial_index_layout())[: 2 * d - 1]
        data_q, anc_q = idx[:d], idx[d: 2 * d - 1]
        backend = QiskitRuntimeService().backend(rec["backend"])
        cal_snapshot = snapshot_calibration(backend, data_q + anc_q)
        edges, _prov = rep_code_edges(cal_snapshot, d, r, data_q, anc_q)
        weights_note = (f"calibrated from {rec['backend']} published "
                        f"calibration for physical qubits data={data_q} "
                        f"ancilla={anc_q}; snapshot pinned in this artifact")
    # THE TRUTH WAS ALWAYS IN THE RECORD. `fin` is the destructive
    # readout of every data qubit; the lane read it only to build the
    # last detector row and discarded the rest. Prepared in |0...0>, its
    # column d-1 IS the logical class of the actual error -- the same
    # representative `logical_cut_edges` names, validated against stim's
    # own observable offline before any of this touches a device.
    ledger = []
    out = certify_shots(det3, d, r, hard=True, edges=edges,
                        observed=fin[:, truth_column(d)], ledger=ledger)
    out.update({"schema": "oneq-ibm-live/1", "mode": "fetch(live hardware)",
                "provider": rec["provider"], "backend": rec["backend"],
                "job_id": rec["job_id"],
                "prereg_sha256": rec["prereg_sha256"],
                "model": getattr(a, "model", "uniform"),
                "weights": weights_note})
    if cal_snapshot is not None:
        out["calibration"] = cal_snapshot
    dest = ROOT / "evidence" / "ibm_live_certified.json"
    dest.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    _archive(out, "certified")
    # The per-shot ledger, keyed by job id so a later fetch cannot
    # overwrite an earlier run's record -- the same rule `_archive`
    # follows, for the same reason.
    lp = (ROOT / "evidence" / "ibm_live"
          / f"{rec['job_id']}.ledger.jsonl")
    with lp.open("w", encoding="utf-8", newline="") as fh:
        for row in ledger:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"ledger -> {lp}  ({len(ledger)} scored shots)")
    print(json.dumps(out, indent=2))
    print(f"evidence -> {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "submit", "fetch"),
                    required=True)
    ap.add_argument("--distance", type=int, default=D)
    ap.add_argument("--rounds", type=int, default=R)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--p", type=float, default=0.01,
                    help="validate-mode noise rate for the stim mirror")
    ap.add_argument("--out", default=None,
                    help="where validate-mode writes its record. Defaults to "
                         "the published slot evidence/ibm_live_validate.json; "
                         "give a path for an exploratory run so it cannot "
                         "replace committed evidence")
    ap.add_argument("--i-understand-this-spends-credits",
                    action="store_true")
    ap.add_argument("--layout",
                    help="PIN the physical qubits: d data then d-1 ancilla, "
                         "comma-separated (e.g. 17,26,24,27,25). Without it "
                         "the transpiler chooses, which is a post-hoc choice "
                         "a preregistration should not leave open.")
    ap.add_argument("--backend",
                    help="pin the device instead of least_busy")
    ap.add_argument("--model", choices=("uniform", "calibrated"),
                    default="uniform",
                    help="decoding model. 'uniform' is what the 2026-08-10 "
                         "run used and what its artifacts say; 'calibrated' "
                         "derives weights from the device's published "
                         "calibration for the physical qubits the job ran "
                         "on. The default stays 'uniform' so a re-fetch of "
                         "an EXISTING job reproduces the bytes it already "
                         "published -- changing what an old artifact says "
                         "is not an upgrade, it is a forgery.")
    a = ap.parse_args()
    return {"validate": mode_validate, "submit": mode_submit,
            "fetch": mode_fetch}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())
