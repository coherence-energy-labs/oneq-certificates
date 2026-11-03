r"""fold_twin.py (R198) -- FOLD_PROOF.md Theorem (B) + (C) run on the lane's own weight-w 1-round witnesses at a T far
outside the checked range: the twin map iota_0 sends each witness mechanism of the 1-round projected model to a
mechanism of the T-round projected model; the images must be undetectable and logical there, and choosing for each
a pure-part fault component (Lemma 5) must give a FULL-DEM undetectable logical set of the same weight.

    python fold_twin.py --T 32

The witness ids index the exporter's projected 1-round model (export_model order).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

LANE = pathlib.Path(__file__).resolve().parent
REC = LANE.parents[1] / "evidence" / "circuit_distance_exact" / "records" / "morph144_witnesses.jsonl"
sys.path.insert(0, str(LANE))
import bb_circuits as bc  # noqa: E402
import dem_independent as di  # noqa: E402
import export_model as em  # noqa: E402


def projected_model(circuit, basis):
    _, mech, coords = em.mechanisms(circuit)
    pm, pc = em.project_same_type(mech, coords, basis)
    return pm, {d: tuple(int(round(v)) for v in c) for d, c in pc.items()}


def xor_check(mechs, ids):
    det, obs = set(), set()
    for j in ids:
        det ^= set(mechs[j][0])
        obs ^= set(mechs[j][1])
    return not det and bool(obs)


def run(basis: str, ids: list, T: int) -> dict:
    kind = "morph144"
    m1, c1 = projected_model(bc.morphing_cycle(kind, basis, 1), basis)
    ok1 = xor_check(m1, ids)
    cT = bc.morphing_cycle(kind, basis, T)
    mT, cTc = projected_model(cT, basis)
    keep = 1 if basis == "Z" else 0
    by_coord = {c: d for d, c in cTc.items()}
    index_T = {m: j for j, m in enumerate(mT)}
    # iota_0: the 1-round detector of g -> the T-round detector of g whose window holds round 0. Every generator
    # contracts in exactly one parity here (INFO in fold_uniform.py): parity 0 -> coordinate 0 in both models;
    # parity 1 -> the 1-round FINAL detector (coordinate 1) and the T-round checkpoint at round 1 (coordinate 1).
    images = []
    for j in ids:
        dets, obs = m1[j]
        new = []
        for d in dets:
            g, typ, r = c1[d]
            assert typ == keep and r in (0, 1)
            new.append(by_coord[(g, keep, r)])
        images.append((tuple(sorted(new)), obs))
    found = [index_T.get(m) for m in images]
    okT = None not in found and xor_check(mT, found) and len(set(found)) == len(ids)
    # full lift by pure components: a component with this projected symptom and no other-type detector
    comps, coords, nd, nobs = di.pauli_components(cT)
    pmask = sum(1 << d for d, c in coords.items() if c[1] == keep)
    obits = ((1 << nobs) - 1) << nd
    proj_index = {d: i for i, d in enumerate(sorted(d for d, c in coords.items() if c[1] == keep))}
    want = {}
    for i, (dets, obs) in enumerate(images):
        m = sum(1 << d for d, c in coords.items() if c[1] == keep and proj_index[d] in set(dets))
        m |= sum(1 << o for o in obs) << nd
        want[m] = i
    chosen = {}
    for idx, pauli, mask in comps:
        if (mask & (pmask | obits)) in want and not (mask & ((1 << nd) - 1) & ~pmask):
            chosen.setdefault(want[mask & (pmask | obits)], mask)
    full = 0
    for mask in chosen.values():
        full ^= mask
    full_ok = len(chosen) == len(ids) and not (full & ((1 << nd) - 1)) and bool(full >> nd)
    return dict(circuit=kind, basis=basis, T=T, weight=len(ids), one_round_witness_verified=ok1,
                twin_images_in_T_model=None not in found, projected_undetectable_logical_at_T=okT,
                full_dem_pure_lift_undetectable_logical_at_T=full_ok, projected_mechanisms_T=len(mT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=32)
    a = ap.parse_args()
    for line in REC.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            print(json.dumps(run(r["basis"], r["t1_projected"], a.T)), flush=True)


if __name__ == "__main__":
    main()
