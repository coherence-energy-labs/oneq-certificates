r"""R198: the FOLD lemma for two-round morphing circuits, checked mechanically for each T: d_proj(T) >= d_proj(1).

    python fold_lemma.py --circuit morph144 --basis Z --tmin 2 --tmax 13      # JSON lines on stdout

WHY. Shaw & Terhal (arXiv 2604.09797, Prop. 1) prove d_circ(T) = d_circ(1) for two-round morphing circuits by
time-reversal folding -- in their convention, with the per-round meta-checks among the detectors. This file
re-derives the folding step for THIS lane's memory experiments (noisy preparation and readout, the derived
detectors, the same-type projection) and checks it on the computed models, so a single 1-round certificate
covers every checked T without the space-time sweep.

THE ARGUMENT. Around each measurement a morphing circuit is a palindrome: ... F_r, M_r | R_r, F_r^dagger ...
(contract, measure, reset the same qubits in the same basis, expand by the same layers reversed). Nothing is
measured between a mid-cycle point mu_t (after round t's expansion) and round t's measurements, nor between the
resets and the next mid-cycle point. So every fault is EQUIVALENT to a Pauli at a mid-cycle point: a fault in a
round's reset step or expansion, carried forward through the rest of the expansion, is a Pauli at that round's
mu_t; a fault in its contraction or measurement step (or the final readout), carried backward through the
contraction, is a Pauli at mu_t too. For each noise component of the T-round circuit this file computes that
mid-cycle Pauli (`fault_table`, from the builder's phase tags), keeps the part the projected model sees (X in
memory-Z), and looks it up among the single faults of the 1-ROUND circuit -- the map phi. If, over every
component, the projected effect in the 1-round model is a GF(2)-linear function of the projected syndrome in the
T-round model, with the observable change linear too (`tools_cde._affine_consistent` on rows
[syn_T | syn_1, obs_1 + obs_T]), then every undetectable logical set X of the T-round projected model maps to
phi(X): undetectable, the same logical flip, and no heavier (one image per mechanism; images can only cancel).
Hence d_proj(1) <= d_proj(T) <= d_circ(T). A component with no image (`missing`) or an inconsistency fails the
check for that T -- it does not mean the lemma is false, only that this route does not certify it.

WHAT IS TRUSTED. The mid-cycle-Pauli reasoning only PROPOSES phi; the conclusion rests on the linear check alone
(any map phi that passes it is sound, a wrong key can only make it fail), on the component masks (the
backward-sensitivity pass, equal to stim's DEM on every model -- `dem_independent.py`), and on the projection.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import dem_independent as di  # noqa: E402
import tools_cde as tc  # noqa: E402


def _cx_layer(layer, x: int, z: int) -> tuple[int, int]:
    """Conjugate a Pauli (X bits, Z bits) by a layer of CNOTs (its own inverse): X_c -> X_c X_t, Z_t -> Z_c Z_t."""
    for c, t in layer:
        if x >> c & 1:
            x ^= 1 << t
        if z >> t & 1:
            z ^= 1 << c
    return x, z


def fault_table(kind: str, basis: str, T: int, hom: str = "x") -> tuple[list, int]:
    """For every noise component of the T-round circuit: (projected detector bits, observable bits, key, round,
    phase), key = the memory-basis-visible part of its equivalent mid-cycle Pauli; and the projected detector count."""
    l, m, a_terms, b_terms = bc.MORPH_CODES[kind]
    gates, _ = bc.morphing_schedule(l, m, a_terms, b_terms, hom)
    tags: list = []
    circuit = bc.morphing_cycle(kind, basis, T, hom=hom, tags=tags)
    comps, coords, nd, _ = di.pauli_components(circuit)
    keep_type = 1 if basis == "Z" else 0
    pos = {d: i for i, d in enumerate(d for d in range(nd) if int(coords[d][1]) == keep_type)}
    tag_of = {t[0]: t for t in tags}
    rows = []
    for idx, pauli, mask in comps:
        _, _, phase, rnd, layer = tag_of[idx]
        prev, cur = (rnd - 1) % 2, rnd % 2
        expansion = list(reversed(gates[prev]))
        if phase in ("prep", "reset"):
            seq = expansion                                   # forward through the whole expansion
        elif phase == "exp":
            seq = expansion[layer + 1:]                       # noise after expansion layer j: forward through the rest
        elif phase == "con":
            seq = list(reversed(gates[cur][:layer + 1]))      # noise after contraction layer j: back through j..0
        else:                                                 # "meas", "readout": back through the whole contraction
            seq = list(reversed(gates[cur]))
        x = sum(1 << q for q, p in pauli.items() if p in "XY")
        z = sum(1 << q for q, p in pauli.items() if p in "ZY")
        for layer_pairs in seq:
            x, z = _cx_layer(layer_pairs, x, z)
        det = sum(1 << pos[d] for d in pos if mask >> d & 1)
        rows.append((det, mask >> nd, x if basis == "Z" else z, rnd, phase))
    return rows, len(pos)


def fold_check(kind: str, basis: str, T: int, hom: str = "x") -> dict:
    one, nd1 = fault_table(kind, basis, 1, hom)
    table: dict[int, tuple[int, int]] = {}
    conflicts = 0
    for det, obs, key, _, _ in one:
        if key in table and table[key] != (det, obs):
            conflicts += 1        # two 1-round faults with one mid-cycle Pauli must act alike; if not, the premise fails
        table.setdefault(key, (det, obs))
    rows, _ = fault_table(kind, basis, T, hom)
    pairs, missing = [], 0
    for det, obs, key, _, _ in rows:
        if not det and not obs:
            continue              # not a DEM mechanism: no solution contains it
        hit = table.get(key)
        if hit is None:
            missing += 1
            continue
        pairs.append((det, hit[0] | (hit[1] ^ obs) << nd1))
    consistent = tc._affine_consistent(pairs)
    return dict(circuit=kind, basis=basis, T=T, components=len(rows), mapped=len(pairs), missing=missing,
                conflicts=conflicts, consistent=consistent, holds=missing == 0 and conflicts == 0 and consistent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=sorted(bc.MORPH_CODES), required=True)
    ap.add_argument("--basis", choices=("Z", "X"), required=True)
    ap.add_argument("--tmin", type=int, default=2)
    ap.add_argument("--tmax", type=int, default=13)
    a = ap.parse_args()
    for T in range(a.tmin, a.tmax + 1):
        print(json.dumps(fold_check(a.circuit, a.basis, T)), flush=True)


if __name__ == "__main__":
    main()
