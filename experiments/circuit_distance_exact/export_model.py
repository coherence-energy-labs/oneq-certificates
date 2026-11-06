r"""R198: a circuit's detector error model as the text model the exact engine reads.

    python export_model.py --circuit ibm|3c --l 12 --basis Z --rounds 2 --out model.txt [--project]

THE MODEL. The DEM of the circuit (decompose_errors=False, approximate_disjoint_errors=True) as a list of
mechanisms (detector set, observable set). Every syndrome must be distinct -- the engine's leaf lookups rely
on it and check it at load.

*** THE SAME-TYPE PROJECTION (`--project`) IS A LOWER BOUND, NOT AN EQUALITY. *** In memory-Z keep only
Z-type detectors (memory-X: X-type). Any full solution X (no detector fires, some observable flips) maps to a
projected solution of weight <= |X|: project every member, drop the ones that become (no detector, no
observable), and cancel coincident projections in pairs; the observable flips are untouched. So
d(projected) <= d_circ. A projected witness becomes a full one only by lifting (`lift_witness.py`). On the
gross code it shrinks 8,568 mechanisms to 1,584 -- the representation change that made weight 10 reachable.
For this lane's CSS circuits under depolarizing noise it is EXACT: every projected mechanism has a pure-X
(memory-Z) / pure-Z (memory-X) fault component, which flips no detector of the other type, so any projected
solution lifts at its own weight and d(projected) = d_circ (adversarial pass 2026-10-21; held by a test).

*** ORBITS. *** The translations of `bb_circuits.translations` must map the mechanism set to itself (asserted
here, re-verified by the checker); orbits are numbered by first occurrence, the anchor of an orbit is its first
member, and the action must be free.

Format: "ND NM NO NORB L M" / per mechanism "k d1..dk obsmask orbit" / the NORB anchors / per detector
"site type round".
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import serialization as ser  # noqa: E402


def mechanisms(circuit):
    """(dem, [(detectors tuple, observables tuple)], {detector: (site, type, round)})"""
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    mech = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        mech.append((dets, obs))
    coords = dem.get_detector_coordinates()
    return dem, mech, coords


def project_same_type(mech, coords, basis: str):
    keep_type = 1 if basis == "Z" else 0
    keep = sorted(d for d, v in coords.items() if int(v[1]) == keep_type)
    new_id = {d: i for i, d in enumerate(keep)}
    seen = {}
    for dets, obs in mech:
        pd = tuple(sorted(new_id[d] for d in dets if d in new_id))
        if not pd and not obs:
            continue
        assert pd, "a single mechanism flips a logical with no same-type detector: d = 1"
        seen[(pd, obs)] = True
    return list(seen.keys()), {new_id[d]: coords[d] for d in keep}


def orbits(mech, coords, shifts, l: int, m: int):
    key_of = {d: (int(v[0]), int(v[1]), int(v[2])) for d, v in coords.items()}
    back = {v: d for d, v in key_of.items()}
    index = {mm[0]: i for i, mm in enumerate(mech)}

    def shift(k, a, b):
        i, j = divmod(k, m)
        return ((i + a) % l) * m + (j + b) % m

    orb = -np.ones(len(mech), dtype=int)
    reps = []
    for i, (dets, _) in enumerate(mech):
        if orb[i] >= 0:
            continue
        o = len(reps)
        reps.append(i)
        for a, b in shifts:
            img = tuple(sorted(back[(shift(key_of[d][0], a, b), key_of[d][1], key_of[d][2])] for d in dets))
            j = index.get(img)
            assert j is not None, "the model is not invariant under a declared translation"
            assert orb[j] in (-1, o), "orbit collision"
            orb[j] = o
    sizes = np.bincount(orb)
    assert (orb >= 0).all() and (sizes == len(shifts)).all(), "the translation action is not free"
    return orb, reps


def observables_affine(mech, coords, shifts, l: int, m: int) -> bool:
    """Anchoring needs "flips some logical" to be invariant on UNDETECTABLE sets: for every translation g there
    must be GF(2) maps with obs(g x) = M obs(x) + K syn(x) for every mechanism x (translated logicals pick up
    stabilizer terms, so plain linearity is false). Checked by Gaussian elimination on [obs | syn | obs(g x)];
    a row whose input part vanishes must have zero output. The checker (ccheck.c) re-verifies this; doing it here
    too makes an uncertified search self-contained (adversarial pass, 2026-10-20)."""
    key_of = {d: (int(v[0]), int(v[1]), int(v[2])) for d, v in coords.items()}
    back = {v: d for d, v in key_of.items()}
    index = {mm[0]: i for i, mm in enumerate(mech)}
    nd, no = len(coords), 1 + max((o for _, obs in mech for o in obs), default=0)
    for a, b in shifts:
        rows = np.zeros((len(mech), no + nd + no), dtype=np.uint8)
        for i, (dets, obs) in enumerate(mech):
            img = tuple(sorted(back[((((key_of[d][0] // m) + a) % l) * m + (key_of[d][0] % m + b) % m,
                                     key_of[d][1], key_of[d][2])] for d in dets))
            j = index[img]
            rows[i, list(obs)] = 1
            rows[i, [no + d for d in dets]] = 1
            rows[i, [no + nd + o for o in mech[j][1]]] = 1
        r = 0
        for col in range(no + nd):
            piv = next((i for i in range(r, len(rows)) if rows[i, col]), None)
            if piv is None:
                continue
            rows[[r, piv]] = rows[[piv, r]]
            hit = rows[:, col].astype(bool)
            hit[r] = False
            rows[hit] ^= rows[r]
            r += 1
        if rows[r:, no + nd:].any():
            return False
    return True


def export(circuit_kind: str, basis: str, rounds: int, out: pathlib.Path, *, l: int = 12, project: bool = True,
           time_order: bool = False) -> dict:
    """`time_order` re-lists the mechanisms by (first detector round, DEM index) so orbit ids are non-decreasing
    in round -- the precondition of TIME-SHIFT anchoring (cdist's MAXROUND argument; the checker re-verifies it).
    The DEM is a set; re-listing changes no answer, only which orbit anchors first."""
    circuit = bc.build(circuit_kind, basis, rounds, l)
    dem, mech, coords = mechanisms(circuit)
    n_full = len(mech)
    if project:
        mech, coords = project_same_type(mech, coords, basis)
    if time_order:
        mech = [m for _, m in sorted(enumerate(mech), key=lambda t: (min(int(coords[d][2]) for d in t[1][0]), t[0]))]
    l, m = bc.dims(circuit_kind, l)
    shifts = bc.translations(bc.group_kind(circuit_kind), l, m)
    orb, reps = orbits(mech, coords, shifts, l, m)
    assert observables_affine(mech, coords, shifts, l, m), "observables are not affine under a declared translation"
    # ASCII, CRLF on every platform: the archived models' bytes, which run records bind by sha256 (serialization.py)
    with ser.open_model(out) as f:
        f.write(f"{len(coords)} {len(mech)} {dem.num_observables} {len(reps)} {l} {m}\n")
        for (dets, obs), o in zip(mech, orb):
            mask = sum(1 << x for x in obs)
            f.write(f"{len(dets)} {' '.join(map(str, dets))} {mask} {o}\n")
        f.write(" ".join(map(str, reps)) + "\n")
        for d in range(len(coords)):
            k, typ, r = coords[d]
            f.write(f"{int(k)} {int(typ)} {int(r)}\n")
    return dict(circuit=circuit_kind, basis=basis, rounds=rounds, l=l, projected=project, full_mechanisms=n_full,
                mechanisms=len(mech), detectors=len(coords), orbits=len(reps), group=len(shifts), time_order=time_order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=bc.CIRCUITS, required=True)
    ap.add_argument("--l", type=int, default=12)
    ap.add_argument("--basis", choices=("Z", "X"), required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--time-order", action="store_true")
    a = ap.parse_args()
    print(export(a.circuit, a.basis, a.rounds, pathlib.Path(a.out), l=a.l, project=a.project, time_order=a.time_order))


if __name__ == "__main__":
    main()
