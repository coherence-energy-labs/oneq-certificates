r"""Single-fault symplectic cross-validation of the reference schedule.

Audit round 6, the last technical gate: noiseless syndrome validation
proves only the ideal Clifford action -- the sequential ablation is the
in-paper counterexample that fault structure can differ under identical
syndrome maps. This gate validates the TRANSCRIPTION AT THE FAULT
LEVEL, with two independent engines:

  PART 1 -- transcription. An op-sequence generator written HERE, from
  the published layer manifest (Table 5 structure: register layout,
  seven CNOT rounds with their monomial assignments and directions,
  init/measure placement, detector definitions), is compared op-by-op
  against the shipped stim circuit's text. Any difference in a control,
  target, order, or detector definition fails the gate.

  PART 2 -- fault propagation. Every single X and Z fault at every
  fault location (after each op, on that op's qubits) is propagated by
  a hand-rolled symplectic engine (CX: x_t ^= x_c, z_c ^= z_t; M flips
  its record iff x=1; MX iff z=1; R/RX clear the frame) and its
  detector+observable signature compared against stim's deterministic
  detector sample with that fault injected. Zero mismatches required.

Together: the stim circuit provably enacts the manifest, and the
manifest's fault structure -- hooks included -- is what stim samples.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code
from qldpc_bb_circuit_gate import logical_z, monomials
from qldpc_bb_refsched_gate import build_refsched_circuit

NHALF = 72
L = list(range(0, 72))
R = list(range(72, 144))
XA = list(range(144, 216))
ZA = list(range(216, 288))
DATA = L + R


def manifest_ops(r: int):
    """Op sequence generated from the Table-5 manifest, independently.

    Emits (name, targets) tuples in execution order, plus DETECTOR
    definitions as ('DETECTOR', [record indices]) with ABSOLUTE record
    indices (0-based, in measurement order), and the observable as
    ('OBSERVABLE', [record indices])."""
    A_perms, B_perms = monomials()

    def fwd(P, i):
        return int(np.flatnonzero(P[i])[0])

    def bwd(P, i):
        return int(np.flatnonzero(P[:, i])[0])

    ops = []
    ops.append(("R", list(DATA)))
    ops.append(("R", list(ZA)))
    nrec = 0
    za_rec = {}                          # cycle -> first record index
    for rd in range(r):
        ops.append(("RX", list(XA)))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (R[bwd(A_perms[0], i)], ZA[i])]))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (XA[i], L[fwd(A_perms[1], i)])]
                    + [q for i in range(NHALF)
                       for q in (R[bwd(A_perms[2], i)], ZA[i])]))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (XA[i], R[fwd(B_perms[1], i)])]
                    + [q for i in range(NHALF)
                       for q in (L[bwd(B_perms[0], i)], ZA[i])]))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (XA[i], R[fwd(B_perms[0], i)])]
                    + [q for i in range(NHALF)
                       for q in (L[bwd(B_perms[1], i)], ZA[i])]))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (XA[i], R[fwd(B_perms[2], i)])]
                    + [q for i in range(NHALF)
                       for q in (L[bwd(B_perms[2], i)], ZA[i])]))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (XA[i], L[fwd(A_perms[0], i)])]
                    + [q for i in range(NHALF)
                       for q in (R[bwd(A_perms[1], i)], ZA[i])]))
        ops.append(("CX", [q for i in range(NHALF)
                           for q in (XA[i], L[fwd(A_perms[2], i)])]))
        ops.append(("M", list(ZA)))
        za_rec[rd] = nrec
        nrec += NHALF
        for i in range(NHALF):
            if rd == 0:
                ops.append(("DETECTOR", [za_rec[rd] + i]))
            else:
                ops.append(("DETECTOR", [za_rec[rd] + i,
                                         za_rec[rd - 1] + i]))
        ops.append(("MX", list(XA)))
        nrec += NHALF
        ops.append(("R", list(ZA)))
    ops.append(("M", list(DATA)))
    data_rec = nrec
    nrec += len(DATA)
    _HX, HZ = bb_gross_code()
    for i in range(NHALF):
        sup = [int(j) for j in np.flatnonzero(HZ[i])]
        ops.append(("DETECTOR", [data_rec + j for j in sup]
                    + [za_rec[r - 1] + i]))
    lz = logical_z()
    ops.append(("OBSERVABLE", [data_rec + j for j in lz]))
    return ops, nrec


def stim_ops(circ):
    """Parse the shipped circuit's text into the same normal form."""
    import stim
    ops = []
    nrec = 0
    for inst in circ.flattened():
        name = inst.name
        if name in ("X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2",
                    "TICK", "QUBIT_COORDS"):
            continue
        if name == "DETECTOR":
            recs = [nrec + t.value for t in inst.targets_copy()]
            ops.append(("DETECTOR", recs))
            continue
        if name == "OBSERVABLE_INCLUDE":
            recs = [nrec + t.value for t in inst.targets_copy()]
            ops.append(("OBSERVABLE", recs))
            continue
        targs = [t.value for t in inst.targets_copy()]
        if name in ("M", "MX"):
            ops.append((name, targs))
            nrec += len(targs)
        elif name in ("R", "RX", "CX"):
            ops.append((name, targs))
        else:
            raise AssertionError(f"unexpected op {name}")
    return ops, nrec


def propagate(ops, nrec, fault_at, fault_q, fault_p):
    """Hand-rolled symplectic propagation of one fault.

    fault_at: index into ops AFTER which the fault occurs. Returns
    (frozenset flipped detector ordinals, observable flip bit)."""
    x = [0] * 288
    z = [0] * 288
    rec = [0] * nrec
    nrec_seen = 0
    det_flips = []
    obs = 0
    det_ord = -1
    active = False
    for k, (name, targs) in enumerate(ops):
        if active:
            if name == "CX":
                for a in range(0, len(targs), 2):
                    c, t = targs[a], targs[a + 1]
                    x[t] ^= x[c]
                    z[c] ^= z[t]
            elif name == "M":
                for q in targs:
                    rec[nrec_seen] ^= x[q]
                    nrec_seen += 1
            elif name == "MX":
                for q in targs:
                    rec[nrec_seen] ^= z[q]
                    nrec_seen += 1
            elif name in ("R", "RX"):
                for q in targs:
                    x[q] = 0
                    z[q] = 0
        else:
            if name in ("M", "MX"):
                nrec_seen += len(targs)
        if name == "DETECTOR":
            det_ord += 1
            if active and sum(rec[i] for i in targs) % 2:
                det_flips.append(det_ord)
        if name == "OBSERVABLE" and active:
            obs = sum(rec[i] for i in targs) % 2
        if k == fault_at:
            active = True
            if fault_p in ("X", "Y"):
                x[fault_q] ^= 1
            if fault_p in ("Z", "Y"):
                z[fault_q] ^= 1
    return frozenset(det_flips), obs


def stim_signature(circ_text_lines, insert_after_line, q, pauli):
    """One fault's detector signature, injected AS NOISE.

    A deterministic Pauli gate is INVISIBLE to stim's detector sampler:
    the sampler computes its reference including deterministic gates,
    so `X 4` shifts the reference too and reports zero detections
    (verified on a 5-line circuit). Faults must be injected as
    probability-1 noise channels, which is what the sampler measures
    against the noiseless reference. This was the harness bug behind a
    5,328-mismatch first run -- the instrument, not the circuit."""
    import stim
    lines = list(circ_text_lines)
    lines.insert(insert_after_line + 1, f"{pauli}_ERROR(1) {q}")
    c2 = stim.Circuit("\n".join(lines))
    det, obs = c2.compile_detector_sampler().sample(
        shots=1, separate_observables=True)
    return frozenset(int(i) for i in np.flatnonzero(det[0])), \
        int(obs[0][0])


def main() -> int:
    r = 3
    circ = build_refsched_circuit(r, 0.0)
    mops, mrec = manifest_ops(r)
    sops, srec = stim_ops(circ)

    # ---- PART 1: transcription equality, op by op ----
    assert mrec == srec, f"record counts differ: {mrec} vs {srec}"
    assert len(mops) == len(sops), \
        f"op counts differ: {len(mops)} vs {len(sops)}"
    for k, (a, b) in enumerate(zip(mops, sops)):
        assert a[0] == b[0] and list(a[1]) == list(b[1]), \
            f"op {k} differs: manifest {a[0]}{a[1][:6]}... vs " \
            f"stim {b[0]}{b[1][:6]}..."
    print(f"PART 1: transcription EXACT -- {len(mops)} ops, "
          f"{mrec} records, every control/target/detector identical "
          f"between the manifest generator and the shipped circuit")

    # ---- PART 2: single-fault signatures, both engines ----
    # op index -> line number in stim text (skip annotations)
    lines = str(circ).split("\n")
    op_line = []
    li = -1
    import re
    for k, (name, _t) in enumerate(sops):
        pat = {"DETECTOR": "DETECTOR", "OBSERVABLE": "OBSERVABLE_INCLUDE",
               }.get(name, name)
        while True:
            li += 1
            stripped = lines[li].strip()
            if stripped.startswith(pat + " ") or stripped == pat or \
                    stripped.startswith(pat + "("):
                op_line.append(li)
                break

    t0 = time.perf_counter()
    checked = mismatches = 0
    for k, (name, targs) in enumerate(sops):
        if name not in ("CX", "R", "RX", "M", "MX"):
            continue
        qs = sorted(set(targs))
        for q in qs:
            for pauli in ("X", "Z"):
                mine = propagate(mops, mrec, k, q, pauli)
                theirs = stim_signature(lines, op_line[k], q, pauli)
                checked += 1
                if mine != theirs:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"MISMATCH op{k} {name} q{q} {pauli}: "
                              f"mine {sorted(mine[0])[:8]}/{mine[1]} "
                              f"stim {sorted(theirs[0])[:8]}/{theirs[1]}")
    dt = time.perf_counter() - t0
    print(f"PART 2: {checked} single-fault signatures compared in "
          f"{dt:.0f}s -- {mismatches} mismatches")
    assert mismatches == 0

    out = {"schema": "oneq-refsched-fault-xval/1",
           "rounds": r,
           "part1_transcription": {"ops": len(mops), "records": mrec,
                                   "identical": True},
           "part2_fault_signatures": {"checked": checked,
                                      "mismatches": 0,
                                      "seconds": round(dt, 1)},
           "engines": ("manifest-derived op generator + hand-rolled "
                       "symplectic propagator vs stim deterministic "
                       "sampling; shared inputs: the published layer "
                       "manifest and the code matrices only")}
    dest = ROOT / "evidence" / "qldpc_refsched_fault_xval.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
