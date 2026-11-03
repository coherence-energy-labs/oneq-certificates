r"""wu_stim.py -- the model of C_T from stim's error analysis ALONE (nothing from wu_core / wu_model).

stim builds the circuit from the lane's builder and analyses it (decompose_errors=False,
approximate_disjoint_errors=True: exactly how export_model.mechanisms calls it); detectors are identified by their
coordinates (k, type, round) and indexed s * 2 n2 + type * n2 + k, observables by their index.
"""
from __future__ import annotations

import pathlib
import sys

LANE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(LANE))

import bb_circuits as bc  # noqa: E402


def stim_mech_list(kind: str, basis: str, T: int, n2: int = 72) -> list:
    """The FULL DEM's mechanisms in stim's order (the order export_model.mechanisms lists them, which is what the
    lane's witness ids index), as (detector mask, observable mask)."""
    c = bc.build(kind, basis, T)
    dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    coords = dem.get_detector_coordinates()
    W = 2 * n2
    idx = {d: int(round(v[2])) * W + int(round(v[1])) * n2 + int(round(v[0])) for d, v in coords.items()}
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dm = om = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dm ^= 1 << idx[t.val]
            elif t.is_logical_observable_id():
                om ^= 1 << t.val
        out.append((dm, om))
    return out


def stim_model(kind: str, basis: str, T: int, projected: bool, circuit=None, n2: int = 72) -> set:
    c = circuit if circuit is not None else bc.build(kind, basis, T)
    dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    coords = dem.get_detector_coordinates()
    W = 2 * n2
    idx = {}
    for d, v in coords.items():
        k, ti, s = (int(round(x)) for x in v)
        idx[d] = s * W + ti * n2 + k
    keep = 1 if basis == "Z" else 0
    out = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dm = om = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                v = coords[t.val]
                if projected and int(round(v[1])) != keep:
                    continue
                dm ^= 1 << idx[t.val]
            elif t.is_logical_observable_id():
                om ^= 1 << t.val
        if dm or om:
            out.add((dm, om))
    return out
