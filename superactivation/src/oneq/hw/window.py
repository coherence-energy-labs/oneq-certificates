"""Opportunistic timing: is NOW a good calibration window for this design? (the owner's 25-item list, item 25)

The HEALTHY guard (oneq.hw.layout) refuses a bad window; this asks the opposite question for a frozen design: how
does the live calibration of the design's OWN qubits and edges rank against every calibration this repository has
recorded? A design is cheapest to run when its qubits are at their best, and the vendor calibration of a fixed
chip moves by tens of percent per qubit day to day (q149's offset 58.6 -> 50 -> 49.7 kHz across three jobs).

cost = sum over the design's qubits of (readout error + 1.5 x sx error) + sum over its edges of 1.25 x CZ error:
the first-order loss of one use of each (the engine's own Pauli conversions, noise.compile_ops). A missing or
non-finite number makes the cost infinite: a design cannot run on a qubit the vendor has not calibrated.

Every reader is pure; tools/calibration_window.py does the (read-only) polling."""
from __future__ import annotations

import math

from .snapshot import DeviceSnapshot


class WindowRefused(ValueError):
    """the window cannot be judged faithfully"""


def design_cost(snap: DeviceSnapshot, qubits, edges) -> float:
    """first-order loss of one use of each of the design's qubits and edges on this calibration (lower is better)"""
    qubits, edges = sorted(set(int(q) for q in qubits)), sorted({tuple(sorted(map(int, e))) for e in edges})
    if not qubits and not edges:
        raise WindowRefused("a design with no qubits and no edges has no cost")
    total = 0.0
    for q in qubits:
        if not (0 <= q < snap.num_qubits):
            raise WindowRefused(f"qubit {q} is not on {snap.name}")
        ro, sx = snap.readout_error(q), snap.sx_error[q]
        if not (math.isfinite(ro) and math.isfinite(sx)):
            return math.inf
        total += ro + 1.5 * sx
    for e in edges:
        if e not in snap.cz:
            return math.inf
        err = snap.cz[e][0]
        if not math.isfinite(err):
            return math.inf
        total += 1.25 * err
    return total


def rank(cost: float, history: list) -> float:
    """the fraction of recorded windows that were STRICTLY better than this one (0 = the best seen)"""
    finite = [h for h in history if math.isfinite(h)]
    if not finite:
        raise WindowRefused("no finite historical cost to rank against")
    return sum(1 for h in finite if h < cost) / len(finite)


def verdict(cost: float, history: list, best_fraction: float = 0.2, min_history: int = 5) -> dict:
    """submit now only if this window ranks in the best `best_fraction` of at least `min_history` recorded ones"""
    if not (0.0 < best_fraction < 1.0):
        raise WindowRefused(f"best_fraction must be in (0, 1): {best_fraction}")
    finite = [h for h in history if math.isfinite(h)]
    if len(finite) < min_history:
        return dict(submit_now=False, reason=f"only {len(finite)} recorded windows (need {min_history})",
                    cost=cost, n_history=len(finite))
    if not math.isfinite(cost):
        return dict(submit_now=False, reason="the design is not fully calibrated now", cost=cost,
                    n_history=len(finite))
    r = rank(cost, finite)
    return dict(submit_now=bool(r < best_fraction), rank=r, cost=cost, n_history=len(finite),
                best_recorded=min(finite), median_recorded=sorted(finite)[len(finite) // 2],
                reason=f"better than {100 * (1 - r):.0f}% of {len(finite)} recorded windows")
