r"""R198: the detector error model derived WITHOUT stim's error analysis, and compared to stim's.

    python dem_independent.py --circuit ibm|3c [--l 12] --basis Z --tmin 1 --tmax 11     # JSON lines on stdout

WHY. Every model the engine searches is a projection of stim's DEM, so stim's error analysis sat in the trusted
base. This derives the same set of fault mechanisms by a different method and requires EXACT equality (and equal
detector coordinates and observable count) with the list the exporter projects from (`export_model.mechanisms`),
which removes stim's error analysis from the trusted base. stim still parses the instruction list both read.

HOW (Heisenberg picture, one backward pass). Number the measurements forward; give every measurement record the
bitmask of the detectors and observables that contain it. Then walk the circuit backwards keeping, per qubit, the
masks fx[q] / fz[q] of outputs an X / Z error at the current point would flip:
  M  (Z basis): an X before it flips its record          fx[q] ^= sens(record)
  MX (X basis): a Z before it flips its record           fz[q] ^= sens(record)
  R, RX: nothing before a reset reaches any output       fx[q] = fz[q] = 0
  H: X <-> Z                                             swap fx[q], fz[q]
  CX a->b: X_a -> X_a X_b and Z_b -> Z_a Z_b             fx[a] ^= fx[b];  fz[b] ^= fz[a]
A noise component then flips exactly the outputs of its Pauli at its own position: X_ERROR -> fx, Z_ERROR -> fz,
DEPOLARIZE1 -> {fx, fx^fz, fz}, DEPOLARIZE2 -> each of the 15 two-qubit Paulis. The DEM is the set of distinct
nonzero masks -- the same object stim's approximate_disjoint_errors DEM lists, one line per distinct symptom.
Instructions outside this list are refused by name (never silently skipped).

The idea was first run by the adversarial pass of 2026-10-20 in its own scratch code; this is an independent
implementation of it for the lane.
"""
from __future__ import annotations

import argparse
import json
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402

TWO_QUBIT_PAULIS = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]


def pauli_dem(circuit) -> tuple[set, dict, int]:
    """(set of (detectors tuple, observables tuple), {detector: coords}, number of observables)"""
    comps, coords, nd, nobs = pauli_components(circuit)
    out = set()
    for _, _, mk in comps:
        if mk:
            out.add((tuple(d for d in range(nd) if mk >> d & 1), tuple(k for k in range(nobs) if mk >> (nd + k) & 1)))
    return out, coords, nobs


def pauli_components(circuit) -> tuple[list, dict, int, int]:
    """Every noise component with what it flips: ([(instruction index, {qubit: "X" | "Y" | "Z"}, mask)], coords,
    number of detectors, number of observables); mask bit d < nd is detector d, bit nd + k is observable k."""
    insts = list(circuit.flattened())
    nmeas, rec_of, det_recs, det_coords, obs_recs = 0, [], [], [], {}
    for inst in insts:
        name = inst.name
        if name in ("M", "MX"):
            if inst.gate_args_copy():
                raise ValueError("a measurement with a built-in flip probability is not handled")
            rec_of.append(list(range(nmeas, nmeas + len(inst.targets_copy()))))
            nmeas += len(inst.targets_copy())
            continue
        rec_of.append(None)
        if name == "DETECTOR":
            det_recs.append([nmeas + t.value for t in inst.targets_copy()])
            det_coords.append(tuple(int(round(v)) for v in inst.gate_args_copy()))
        elif name == "OBSERVABLE_INCLUDE":
            obs_recs.setdefault(int(inst.gate_args_copy()[0]), []).extend(nmeas + t.value for t in inst.targets_copy())
    nd = len(det_recs)
    nobs = max(obs_recs) + 1 if obs_recs else 0
    sens = [0] * nmeas
    for d, recs in enumerate(det_recs):
        for r in recs:
            sens[r] ^= 1 << d
    for k, recs in obs_recs.items():
        for r in recs:
            sens[r] ^= 1 << (nd + k)
    nq = circuit.num_qubits
    fx, fz = [0] * nq, [0] * nq
    comps: list = []

    def flips(q, p):
        return {"I": 0, "X": fx[q], "Z": fz[q], "Y": fx[q] ^ fz[q]}[p]

    for i in range(len(insts) - 1, -1, -1):
        inst = insts[i]
        name, tg = inst.name, [t.value for t in inst.targets_copy()]
        if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS"):
            continue
        if name == "M":
            for q, r in zip(tg, rec_of[i]):
                fx[q] ^= sens[r]
        elif name == "MX":
            for q, r in zip(tg, rec_of[i]):
                fz[q] ^= sens[r]
        elif name in ("R", "RX"):
            for q in tg:
                fx[q] = fz[q] = 0
        elif name == "H":
            for q in tg:
                fx[q], fz[q] = fz[q], fx[q]
        elif name == "CX":
            for a, b in zip(tg[0::2], tg[1::2]):
                fx[a] ^= fx[b]
                fz[b] ^= fz[a]
        elif name == "X_ERROR":
            comps += [(i, {q: "X"}, fx[q]) for q in tg]
        elif name == "Z_ERROR":
            comps += [(i, {q: "Z"}, fz[q]) for q in tg]
        elif name == "DEPOLARIZE1":
            for q in tg:
                comps += [(i, {q: p}, flips(q, p)) for p in "XYZ"]
        elif name == "DEPOLARIZE2":
            for a, b in zip(tg[0::2], tg[1::2]):
                comps += [(i, {q: p for q, p in ((a, pa), (b, pb)) if p != "I"}, flips(a, pa) ^ flips(b, pb))
                          for pa, pb in TWO_QUBIT_PAULIS]
        else:
            raise ValueError(f"instruction {name} is not handled by the independent derivation")
    return comps, dict(enumerate(det_coords)), nd, nobs


def stim_dem(circuit) -> tuple[set, dict, int]:
    """The mechanisms exactly as the exporter takes them from stim (export_model.mechanisms -- the list every
    searched and certified model is projected from), as a set."""
    dem, mech, coords = em.mechanisms(circuit)
    return set(mech), {d: tuple(int(round(v)) for v in xs) for d, xs in coords.items()}, dem.num_observables


def compare(circuit) -> dict:
    mine, mc, mo = pauli_dem(circuit)
    theirs, tc_, to = stim_dem(circuit)
    return dict(mechanisms=len(mine), stim_mechanisms=len(theirs), equal=mine == theirs and mc == tc_ and mo == to,
                only_independent=len(mine - theirs), only_stim=len(theirs - mine))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=bc.CIRCUITS, required=True)
    ap.add_argument("--l", type=int, default=12)
    ap.add_argument("--basis", choices=("Z", "X"), required=True)
    ap.add_argument("--tmin", type=int, default=1)
    ap.add_argument("--tmax", type=int, default=2)
    a = ap.parse_args()
    for T in range(a.tmin, a.tmax + 1):
        print(json.dumps(dict(circuit=a.circuit, l=a.l, basis=a.basis, rounds=T,
                              **compare(bc.build(a.circuit, a.basis, T, a.l)))), flush=True)


if __name__ == "__main__":
    main()
