r"""Paper 2's circuit model against paper 3's IBM model at T = 3, and the weight-10 logical in paper 2's own instance.

    python experiments/m0_prior_art/qldpc_refsched_p3_equality.py

WHY (external audit 1 of release v2026.11.03, "should fix"; adjudicated 2026-11-03, finding P2-04). Paper 2's circuit
dataset does not fault idle locations; paper 3's IBM placement does, and proves d_circ = 10 under it. Paper 2 cites the
value 10 for "this cycle". Fewer fault locations can only keep a distance or raise it, so containment alone gives
paper 2's family d >= 10. This measures whether there is any gap.

WHAT IS COMPARED. Paper 2's model is build_refsched_circuit(3, 0.002) (memory-Z, one declared logical Z), exactly the
persisted instance evidence/instances/bb144_refsched_r3_p002.json. Paper 3's is bb_circuits.ibm_cycle("Z", 3),
projected to its Z-type detectors by export_model.project_same_type, and the stored certified model
evidence/circuit_distance_exact/ibm_recert_20261026/models/ibm_l12_Z3.txt (hash-checked against its manifest). A
mechanism is a signature (detector set, observable set); stim merges equal-signature errors into one line, and the
comparison is of SETS of signatures. Detectors are aligned by the measurement events they compare -- not by index --
and the alignment must be a bijection onto paper 3's Z-type detectors. Observables: paper 2's one logical, and paper
3's twelve with paper 2's added to both circuits as a thirteenth.

THE WITNESS. Paper 3's memory-Z IBM witness found at T = 2 (evidence/circuit_distance_exact/records/
ibm_witnesses.jsonl, first line), shifted one round later and written in paper 2's column order: a 10-column set
whose syndrome on the persisted instance is zero and which flips paper 2's observable, while no 9-column subset does
both. That bounds paper 2's d_DEM for its observable by 10, where the paper reports 12 (Stim's search).
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
P3 = ROOT / "experiments" / "circuit_distance_exact"
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
sys.path.insert(0, str(P3))
OUT = ROOT / "evidence" / "qldpc_refsched_p3_equality.json"
INSTANCE = ROOT / "evidence" / "instances" / "bb144_refsched_r3_p002.json"
MANIFEST = ROOT / "evidence" / "circuit_distance_exact" / "ibm_recert_20261026" / "manifest.json"
STORED = ROOT / "evidence" / "circuit_distance_exact" / "ibm_recert_20261026" / "models" / "ibm_l12_Z3.txt"
# paper 3's T=2 memory-Z witness (projected detector ids of its T=2 model), shifted one round, in paper 2's columns
WITNESS_P2_COLUMNS = [2088, 2089, 2101, 2146, 2159, 2212, 2236, 2238, 2242, 2287]
MEAS = {"M", "MX", "MZ", "MY", "MR", "MRX"}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def events(c):
    """Detectors and observables as sets of (qubit, k-th measurement of that qubit) events."""
    meas, cnt = [], collections.Counter()
    dets, obs = [], collections.defaultdict(set)
    for inst in c.flattened():
        if inst.name in MEAS:
            for t in inst.targets_copy():
                meas.append((t.value, cnt[t.value]))
                cnt[t.value] += 1
        elif inst.name == "DETECTOR":
            s = set()
            for t in inst.targets_copy():
                s ^= {meas[len(meas) + t.value]}
            dets.append(frozenset(s))
        elif inst.name == "OBSERVABLE_INCLUDE":
            k = int(inst.gate_args_copy()[0])
            for t in inst.targets_copy():
                obs[k] ^= {meas[len(meas) + t.value]}
    return dets, dict(obs)


def signatures(c, det_map=None, obs_map=None, keep_obs=None, adj=False):
    dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=adj)
    lines, sigs = 0, set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        lines += 1
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        os_ = [t.val for t in inst.targets_copy() if t.is_logical_observable_id()]
        if det_map is not None:
            ds = [det_map[d] for d in ds if d in det_map]
        if keep_obs is not None:                  # keep_obs names RAW observable ids, before any relabelling
            os_ = [o for o in os_ if o in keep_obs]
        if obs_map is not None:
            os_ = [obs_map[o] for o in os_]
        ds, os_ = frozenset(ds), frozenset(os_)
        if ds or os_:
            sigs.add((ds, os_))
    return lines, sigs


def relation(a: set, b: set) -> dict:
    rel = "equal" if a == b else "P2 subset of P3" if a < b else "P3 subset of P2" if b < a else "neither"
    return {"P2": len(a), "P3": len(b), "both": len(a & b), "P2_only": len(a - b), "P3_only": len(b - a),
            "relation": rel}


def main() -> int:
    import numpy as np
    import stim
    import bb_circuits as bc
    import export_model as em
    from qldpc_bb_circuit_gate import logical_z
    from qldpc_bb_refsched_gate import build_refsched_circuit
    from qldpc_circuit_gate import dem_to_matrices

    out = {"schema": "oneq-qldpc-refsched-p3-equality/1", "stim": stim.__version__, "rounds": 3, "basis": "Z"}
    c2 = build_refsched_circuit(3, 0.002)
    c3 = bc.ibm_cycle("Z", 3)
    lz = sorted(logical_z())

    # paper 2's model IS its persisted instance
    dem2 = c2.detector_error_model(decompose_errors=False)
    checks, _w, obs_rows, n_mech = dem_to_matrices(dem2)
    inst = json.loads(INSTANCE.read_text(encoding="utf-8"))
    out["p2_model_is_the_persisted_instance"] = ([list(c) for c in checks] == inst["checks"]
                                                 and n_mech == inst["n_variables"] == 2304)

    # detector alignment by measurement events
    D2, O2 = events(c2)
    D3, O3 = events(c3)
    idx2 = {d: i for i, d in enumerate(D2)}
    align = {j: idx2[d] for j, d in enumerate(D3) if d in idx2}
    coords3 = c3.get_detector_coordinates()
    ztype = sorted(j for j, v in coords3.items() if int(v[1]) == 1)
    out["detectors"] = {"p2": len(D2), "p3": len(D3), "p3_z_type": len(ztype),
                        "aligned_by_measurement_events": len(align),
                        "alignment_is_a_bijection_onto_p3_z_type": (sorted(align) == ztype
                                                                    and len(set(align.values())) == len(D2))}
    # observables: paper 2's logical is one of paper 3's twelve
    L3 = [sorted(q for (q, k) in O3[i]) for i in range(len(O3))]
    p2_obs = sorted(q for (q, k) in O2[0])
    out["observables"] = {"p2": len(O2), "p3": len(O3), "p2_logical_is_p3_observable": [i for i, s in enumerate(L3)
                                                                                           if s == p2_obs],
                          "p2_logical_matches_logical_z": p2_obs == lz}
    if not (out["p2_model_is_the_persisted_instance"] and out["detectors"]["alignment_is_a_bijection_onto_p3_z_type"]
            and out["observables"]["p2_logical_is_p3_observable"]):
        raise SystemExit(f"alignment controls failed: {out}")
    k0 = out["observables"]["p2_logical_is_p3_observable"][0]

    # (a) paper 2's declared observable on both sides
    _, S2 = signatures(c2)
    _, S3 = signatures(c3, det_map=align, obs_map={k0: 0}, keep_obs={k0}, adj=True)
    out["mechanisms_p2_observable"] = relation(S2, S3)

    # (b) all twelve of paper 3's observables, added to paper 2's circuit (P2's own logical becomes label 12)
    def with_obs(c, sups_ids):
        c = c.copy()
        for sup, k in sups_ids:
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(144 - q)) for q in sup], k)
        return c
    c2aug = with_obs(c2, [(L3[i], i + 1) for i in range(len(L3))])
    c3aug = with_obs(c3, [(lz, len(L3))])
    l2, S2_13 = signatures(c2aug, obs_map={0: len(L3), **{i + 1: i for i in range(len(L3))}})
    _, S3_13 = signatures(c3aug, det_map=align, adj=True)
    out["mechanisms_13_observables"] = relation(S2_13, S3_13)
    out["p2_dem_lines_equal_distinct_signatures"] = l2 == len(S2_13)

    # (c) paper 3's own export path and its stored, certified model
    _dem3, mech3, crd3 = em.mechanisms(c3)
    pm3, pc3 = em.project_same_type(mech3, crd3, "Z")
    S3_export = {(frozenset(72 * int(pc3[d][2]) + int(pc3[d][0]) for d in dets), frozenset(obs)) for dets, obs in pm3}
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pinned = [o for o in man["observations"] if o.get("model") == "models/ibm_l12_Z3.txt"]
    out["stored_model"] = {"file": STORED.relative_to(ROOT).as_posix(), "sha256": sha(STORED),
                           "manifest_sha256": pinned[0].get("model_sha256") if pinned else None}
    with open(STORED, encoding="utf-8") as f:
        hdr = list(map(int, f.readline().split()))
        nd, nm, no = hdr[0], hdr[1], hdr[2]
        rows = [list(map(int, f.readline().split())) for _ in range(nm)]
        f.readline()
        dco = [tuple(map(int, f.readline().split())) for _ in range(nd)]
    S3_stored = set()
    for row in rows:
        k = row[0]
        S3_stored.add((frozenset(72 * dco[d][2] + dco[d][0] for d in row[1:1 + k]),
                       frozenset(i for i in range(no) if row[1 + k] >> i & 1)))
    S2_12 = {(d, frozenset(o - {len(L3)})) for d, o in S2_13}
    S2_12 = {s for s in S2_12 if s[0] or s[1]}
    out["stored_model"].update(header=hdr, export_equals_stored=S3_export == S3_stored,
                               mechanisms_vs_p2_with_p3_observables=relation(S2_12, S3_stored))

    # the witness, on the persisted instance
    H = np.zeros((288, 2304), dtype=np.uint8)
    for r_, row in enumerate(inst["checks"]):
        H[r_, row] = 1
    oc = set(int(i) for i in obs_rows.get(0, ()))

    def undetectable_logical(cols) -> bool:
        x = np.zeros(2304, dtype=np.uint8)
        x[list(cols)] = 1
        return not ((H.astype(np.int64) @ x) % 2).any() and len(oc & set(cols)) % 2 == 1
    W = WITNESS_P2_COLUMNS
    out["witness"] = {"columns": W, "weight": len(W), "provenance": (
        "paper 3's memory-Z IBM witness at T = 2 (evidence/circuit_distance_exact/records/ibm_witnesses.jsonl, line 1), "
        "shifted one round later and written in paper 2's DEM column order"),
        "zero_syndrome_and_flips_p2_observable": undetectable_logical(W),
        "no_9_column_subset_does_both": not any(undetectable_logical(s) for s in itertools.combinations(W, 9))}
    ok = (out["mechanisms_p2_observable"]["relation"] == "equal" and out["mechanisms_13_observables"]["relation"]
          == "equal" and out["witness"]["zero_syndrome_and_flips_p2_observable"]
          and out["witness"]["no_9_column_subset_does_both"])
    out["conclusion"] = (
        "At T = 3, memory-Z, paper 2's 2,304-mechanism model and paper 3's IBM model projected to its Z-type detectors "
        "have the SAME mechanism set (paper 3's idle faults add no signature), so paper 3's certified lower bound on "
        "that model applies to paper 2's unchanged, and the 10-column witness bounds it from above: d_DEM = 10 for "
        "paper 2's declared observable at T = 3 (DERIVED: this measurement plus paper 3's certificate and time-shift "
        "theorem). Only T = 3 is compared." if ok else "NOT ESTABLISHED: see the fields above")
    me = pathlib.Path(__file__).resolve()
    out["code_sha256"] = {me.relative_to(ROOT).as_posix(): sha(me),
                          "experiments/m0_prior_art/qldpc_bb_refsched_gate.py":
                              sha(ROOT / "experiments" / "m0_prior_art" / "qldpc_bb_refsched_gate.py"),
                          "experiments/circuit_distance_exact/bb_circuits.py": sha(P3 / "bb_circuits.py"),
                          "experiments/circuit_distance_exact/export_model.py": sha(P3 / "export_model.py")}
    out["inputs_sha256"] = {INSTANCE.relative_to(ROOT).as_posix(): sha(INSTANCE),
                            MANIFEST.relative_to(ROOT).as_posix(): sha(MANIFEST)}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("mechanisms_p2_observable", "mechanisms_13_observables", "witness")},
                     indent=1))
    print(out["conclusion"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
