"""Decision value per QPU-second: how sensitive a decision functional is to every coordinate of the device posterior,
and how to spend shots so the functional is known best for the quota.

For a functional F of the device (a protocol's predicted F_e, a code's logical error rate, a claim's margin), the
first-order variance from the posterior is  Var F = sum_k (dF/dtheta_k)^2 sigma_k^2  over the posterior's coordinates.
Each coordinate is measured by one calibration FAMILY, and its variance falls as 1/N in that family's shots, so

    Var F (N) = sum_j A_j / N_j,   A_j = N_j0 sum_{k in j} (dF/dtheta_k)^2 sigma_k0^2   (sigma_k0 at shots N_j0)

and the allocation minimizing it for a budget sum_j c_j N_j = B (c_j = QPU seconds per shot of the family, i.e. its
circuit count x the per-shot time) is  N_j  proportional to  sqrt(A_j / c_j)  -- the c-optimal design (a Neyman
allocation over families). A comparison that also carries MEASURED shot noise (V1: measured - predicted) adds that
family's A / N to the same sum.

dF/dtheta_k comes from the engine's linear response (sensitivity.py), chained through the gauge-fixing map for the
CZ coordinates: the posterior holds 9 learnable quantities per edge, the engine consumes 15 Pauli probabilities, and
the symmetric gauge connects them, in closed form: p = WHT f / 16 with f_P = sqrt(g) for a fixed P and for both members
of a pair, so dp_Q/dg_k = sum over k's members P of WHT[Q, P] / (32 sqrt g_k)."""
from __future__ import annotations

import math

from oneq.hw import cz_channel as CC

PER_SHOT_S = 320e-6


def channel_jacobian(g: dict) -> dict:
    """d p_Q / d g_k of the symmetric-gauge channel at g, as {k: {Q: dp_Q/dg_k}}, exactly"""
    out = {}
    idx = {p: i for i, p in enumerate(CC.LABELS)}
    for k, gk in g.items():
        if gk <= 0:
            raise ValueError(f"learnable quantity {k} = {gk}: the symmetric gauge's sqrt has no derivative there")
        members = (k,) if isinstance(k, str) else tuple(k)
        out[k] = {Q: sum(CC.WHT[idx[Q], idx[P]] for P in members) / (32 * math.sqrt(gk)) for Q in CC.NONI}
    return out


def coordinate_sensitivity(components: dict, posterior) -> dict:
    """dF/dtheta_k for every posterior coordinate from aggregated engine components (sensitivity.Sensitivity.components,
    possibly summed over many circuits)."""
    from oneq.hw.device import _KEYS     # the posterior's string keys for the learnable quantities
    out = {}
    by_edge: dict = {}
    for key, c in posterior.coords.items():
        if key[0] == "cz":
            by_edge.setdefault((key[1], key[2]), {})[key[3]] = c.mean
    for (a, b), gs in by_edge.items():
        if len(gs) != 9:
            # skipping it would report dF = 0 for the edge's coordinates, and an allocation would starve the family
            # that measures them; the engine refuses such an edge too (DevicePosterior.snapshot)
            raise ValueError(f"edge {(a, b)} has {len(gs)} of its 9 learnable quantities: no gauge map, no sensitivity")
        g = {_KEYS[k]: v for k, v in gs.items()}
        jac = channel_jacobian(g)
        for k, col in jac.items():
            out[("cz", a, b, str(k))] = sum(components.get(("cz", a, b, Q), 0.0) * dpq for Q, dpq in col.items())
    for key, c in posterior.coords.items():
        if key[0] in ("p01", "p10"):
            out[key] = components.get(key, 0.0)
        elif key[0] == "sx":
            out[key] = components.get(("sx", key[1]), 0.0)
        elif key[0] == "t2":
            out[key] = -components.get(("gamma2", key[1]), 0.0) / c.mean ** 2
        elif key[0] == "t1":
            out[key] = -components.get(("gamma1", key[1]), 0.0) / c.mean ** 2
    return out


def family_loads(sens: dict, posterior, family_of, shots_now: dict) -> dict:
    """A_j for each family: shots_now[j] x sum over its coordinates of (dF/dtheta)^2 sigma^2"""
    A: dict = {}
    for key, s in sens.items():
        c = posterior.coords.get(key)
        if c is None or c.sd <= 0:
            continue
        j = family_of(key)
        A[j] = A.get(j, 0.0) + shots_now[j] * (s * c.sd) ** 2
    return A


def allocate(A: dict, circuits: dict, budget_s: float) -> dict:
    """the c-optimal shots per circuit of each family for a QPU budget (seconds of per-shot time)"""
    cost = {j: circuits[j] * PER_SHOT_S for j in A}
    lam = sum(math.sqrt(A[j] * cost[j]) for j in A) / budget_s
    return {j: math.sqrt(A[j] / cost[j]) / lam for j in A}


def allocate_with_floors(A: dict, circuits: dict, budget_s: float, floors: dict) -> dict:
    """the optimal shots per circuit when some families may not go below a floor (a fit whose variance model needs
    enough shots per circuit to hold: a ladder's deep correlators at a few hundred shots are not Gaussian).

    KKT for  min sum A_j / N_j  s.t.  sum c_j N_j = B,  N_j >= f_j:  N_j = max(f_j, sqrt(A_j / c_j) / lambda), with
    lambda set so the budget binds -- water-filling over the families the floors do not pin. Refuses a budget the
    floors alone exceed, and floors that spend all of it while a family is left with none: that family would get zero
    shots, so its term A_j / N_j -- and the variance -- would be infinite (found 2026-10-16: floors costing exactly the
    budget divided by zero, and floors within the 1e-12 tolerance above it handed a family 0 shots)."""
    cost = {j: circuits[j] * PER_SHOT_S for j in A}
    fl = {j: float(floors.get(j, 0.0)) for j in A}
    if sum(cost[j] * fl[j] for j in A) > budget_s * (1 + 1e-12):
        raise ValueError("the floors alone cost more than the budget")
    pinned: set = set()
    while True:
        free = [j for j in A if j not in pinned]
        left = budget_s - sum(cost[j] * fl[j] for j in pinned)
        if not free:
            return {j: fl[j] for j in A}
        if left <= 0:
            raise ValueError(f"the floors spend the whole budget and leave no shots for {sorted(free)}")
        lam = sum(math.sqrt(A[j] * cost[j]) for j in free) / left
        n = {j: math.sqrt(A[j] / cost[j]) / lam for j in free}
        newly = {j for j in free if n[j] < fl[j]}
        if not newly:
            out = {j: fl[j] for j in pinned}
            out.update(n)
            return out
        pinned |= newly


def variance(A: dict, N: dict) -> float:
    return sum(A[j] / N[j] for j in A)


def budget_for(A: dict, circuits: dict, var_target: float) -> float:
    """the least QPU budget (per-shot seconds) whose optimal allocation reaches var_target"""
    s = sum(math.sqrt(A[j] * circuits[j] * PER_SHOT_S) for j in A)
    return s * s / var_target


__all__ = ["channel_jacobian", "coordinate_sensitivity", "family_loads", "allocate", "allocate_with_floors", "variance",
           "budget_for"]
