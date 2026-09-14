r"""THE DEM LEG: is stim's mechanism LIST the enumerated single faults?

Trust = min(decoder, DEM) is the build program's standing finding, and
after this campaign the decoder leg is closed while the DEM leg is not.
The fault cross-validation closed one half of the DEM leg: propagation.
It proved that a fault inserted at a location produces the detector
signature our independent symplectic engine predicts.

It did NOT prove the other half: that the DEM's mechanism LIST is
exactly the set of physical single-fault events, with the right
probabilities. A DEM could propagate every fault correctly and still
enumerate the wrong set -- omit a location, double-count a channel, or
merge two mechanisms that should stay distinct -- and every downstream
receipt would certify decoding of a model that is not the circuit.

This gate closes it, three ways:

  1. COVERAGE. Enumerate every single-fault event the noise channels
     can produce (X/Y/Z after each depolarizing location, flips at each
     reset/measurement), compute each signature symplectically, and
     require every signature with a nonzero probability to appear in
     the DEM's mechanism list.
  2. NO INVENTION. Require every DEM mechanism's signature to appear
     among the enumerated physical single-fault signatures -- a
     mechanism the circuit cannot produce is an invented one.
  3. PROBABILITY ACCOUNTING. For each signature, compare the DEM's
     probability against the sum of physical channel probabilities
     producing it, to first order (the DEM merges same-signature events
     by odd-parity combination; at p=0.002 the higher-order correction
     is ~p^2 and is reported, not hidden).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_refsched_gate import build_refsched_circuit
from qldpc_refsched_fault_xval import manifest_ops, propagate, stim_ops

P = 0.002
ROUNDS = 3


def main() -> int:
    import stim
    circ = build_refsched_circuit(ROUNDS, P)
    sops, srec = stim_ops(circ)
    mops, mrec = manifest_ops(ROUNDS)
    assert len(mops) == len(sops) and mrec == srec

    # ---- enumerate physical single-fault signatures ----
    # walk the noisy circuit; for each noise instruction, each qubit it
    # touches contributes single-qubit Pauli events with first-order
    # probabilities. Signatures come from the symplectic engine, which
    # the fault cross-validation already validated against stim.
    t0 = time.perf_counter()
    phys = defaultdict(float)          # signature -> total probability
    op_index = {}                      # map noisy op -> preceding op idx
    # INDEX ALIGNMENT. The op list that propagate() walks includes
    # DETECTOR and OBSERVABLE entries, so counting only gates drifts the
    # injection point by one per detector -- 1,440 phantom "invented"
    # mechanisms on the first run. Count exactly what stim_ops() counts.
    k = -1
    for inst in circ.flattened():
        name = inst.name
        if name in ("R", "RX", "CX", "M", "MX", "DETECTOR",
                    "OBSERVABLE_INCLUDE"):
            k += 1
            continue
        if name not in ("X_ERROR", "Z_ERROR", "DEPOLARIZE1",
                        "DEPOLARIZE2"):
            continue
        targs = [t.value for t in inst.targets_copy()]
        pr = inst.gate_args_copy()[0]
        if pr <= 0:
            continue
        if name == "X_ERROR":
            events = [(q, "X", pr) for q in targs]
        elif name == "Z_ERROR":
            events = [(q, "Z", pr) for q in targs]
        elif name == "DEPOLARIZE1":
            events = [(q, pa, pr / 3.0) for q in targs
                      for pa in ("X", "Y", "Z")]
        else:                           # DEPOLARIZE2: 15 two-qubit Paulis
            events = []
            for a in range(0, len(targs), 2):
                q1, q2 = targs[a], targs[a + 1]
                for pa in ("I", "X", "Y", "Z"):
                    for pb in ("I", "X", "Y", "Z"):
                        if pa == "I" and pb == "I":
                            continue
                        # first-order: treat each two-qubit Pauli as a
                        # pair of single-qubit events on the same tick
                        events.append(((q1, q2), (pa, pb), pr / 15.0))
        for ev in events:
            if isinstance(ev[0], tuple):
                (q1, q2), (pa, pb), w = ev
                s1 = (propagate(mops, mrec, k, q1, pa)
                      if pa != "I" else (frozenset(), 0))
                s2 = (propagate(mops, mrec, k, q2, pb)
                      if pb != "I" else (frozenset(), 0))
                sig = (frozenset(s1[0] ^ s2[0]), s1[1] ^ s2[1])
            else:
                q, pa, w = ev
                sig = propagate(mops, mrec, k, q, pa)
            if sig[0] or sig[1]:
                phys[sig] += w
    t_enum = time.perf_counter() - t0
    print(f"enumerated {len(phys)} distinct physical single-fault "
          f"signatures in {t_enum:.0f}s")

    # ---- the DEM's mechanism list ----
    dem = circ.detector_error_model(decompose_errors=False)
    demsig = defaultdict(float)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        pr = inst.args_copy()[0]
        dets = frozenset(int(t.val) for t in inst.targets_copy()
                         if t.is_relative_detector_id())
        obs = 0
        for t in inst.targets_copy():
            if t.is_logical_observable_id():
                obs ^= 1
        demsig[(dets, obs)] += pr
    print(f"DEM carries {len(demsig)} distinct mechanism signatures")

    missing = [s for s in phys if s not in demsig]
    invented = [s for s in demsig if s not in phys]
    common = [s for s in demsig if s in phys]
    dev = []
    for s in common:
        a, b = demsig[s], phys[s]
        if b > 0:
            dev.append(abs(a - b) / b)
    out = {"schema": "oneq-dem-mechanism-audit/1",
           "circuit": f"reference schedule r={ROUNDS}, p={P}",
           "physical_signatures": len(phys),
           "dem_signatures": len(demsig),
           "coverage_missing_from_dem": len(missing),
           "invented_by_dem": len(invented),
           "common": len(common),
           "probability_max_rel_dev": (round(max(dev), 4) if dev else None),
           "probability_median_rel_dev": (round(float(np.median(dev)), 6)
                                          if dev else None),
           "enumeration_seconds": round(t_enum, 1),
           "note": ("first-order accounting: the DEM merges identical "
                    "signatures by odd-parity combination, so deviations "
                    "of order p are expected and reported rather than "
                    "hidden; what must be ZERO is missing coverage and "
                    "invented mechanisms")}
    print(f"  coverage: {len(missing)} physical signatures missing from "
          f"the DEM")
    print(f"  invention: {len(invented)} DEM mechanisms the circuit "
          f"cannot produce")
    if dev:
        print(f"  probability: median rel dev "
              f"{out['probability_median_rel_dev']}, max "
              f"{out['probability_max_rel_dev']}")
    dest = ROOT / "evidence" / "qldpc_dem_mechanism_audit.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    assert not missing, f"{len(missing)} physical faults absent from DEM"
    assert not invented, f"{len(invented)} invented DEM mechanisms"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
