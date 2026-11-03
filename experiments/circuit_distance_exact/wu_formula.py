r"""wu_formula.py -- WINDOW_PROOF.md Theorem 1 (the window formula) and its window-structure corollary, mechanical.

For T >= T_0 = j_s + U the model of C_T is

    M_T = S  u  U_{i = j_s}^{T-1-U} tau_i(I)  u  tau_T(End)

with S (components at positions < j_s, absolute coordinates), I (a body component's profile relative to its
position) and End (body components at positions T-U..T-1 and the tail / readout components, relative to T), all
finite and T-independent. U = max(h, e) + 1 from FIR(h) / END(e). Masks: interior profiles are stored relative to
round i - 1, end profiles relative to round T - D (D = U + 1), so every shift is a left shift.
"""
from __future__ import annotations

import sys

import wu_model as wm  # noqa: E402


class Formula:
    def __init__(self, M: wm.Machine, projected: bool, h: int, e: int, T_ref: int | None = None,
                 i0: int | None = None, comps: str = "comps"):
        self.M, self.projected = M, projected
        self.W = M.W
        self.U = max(h, e) + 1
        other_delta_minus1 = any(M.delta[q] == -1 and M.code.check_of(q)[0] != M.code.bi for q in M.delta)
        self.j_s = 1 if projected or not other_delta_minus1 else 2
        self.T0 = self.j_s + self.U
        self.D = self.U + 1         # a body block at position T-U reads records of round >= T-U-1
        self.T_ref = T_ref if T_ref is not None else self.T0 + 3
        self.i0 = i0 if i0 is not None else self.j_s + 1
        assert self.j_s <= self.i0 <= self.T_ref - 1 - self.U, "i0 must be an interior position of C_T_ref"
        T, W = self.T_ref, self.W
        S, I, End = set(), set(), set()
        lost = 0
        for p, nm, idx, d, o in M.all_symptoms(T, comps):
            if projected:
                d = M.proj(d, T)
            if not (d or o):
                continue
            if p < self.j_s:
                S.add((d, o))
            elif nm == "B" and p == self.i0:
                base = (self.i0 - 1) * W
                lost += d & ((1 << base) - 1) != 0
                I.add((d >> base, o))
            elif (nm == "B" and p >= T - self.U) or nm in ("E", "R"):
                base = (T - self.D) * W
                lost += d & ((1 << base) - 1) != 0
                End.add((d >> base, o))
        assert lost == 0, "a profile reaches below its reference round"
        self.S, self.I, self.End = S, I, End

    def model(self, T: int) -> set:
        assert T >= self.T0, f"the window formula is stated for T >= T_0 = {self.T0}"
        W = self.W
        out = set(self.S)
        for i in range(self.j_s, T - self.U):
            sh = (i - 1) * W
            out.update((d << sh, o) for d, o in self.I)
        sh = (T - self.D) * W
        assert sh >= 0
        out.update((d << sh, o) for d, o in self.End)
        return out


# ------------------------------------------------------------------------------------------- windows
def start_round(d: int, W: int) -> int:
    low = d & -d
    return (low.bit_length() - 1) // W


def last_round(d: int, W: int) -> int:
    return (d.bit_length() - 1) // W


def windows(model: set, W: int):
    """by_start[r] = set of shapes (detector mask relative to r, observable mask); P1 = span <= 1 round. A
    mechanism with no detector cannot be windowed: it is reported (it would be a weight-1 logical)."""
    by, p1, nodet = {}, True, 0
    for d, o in model:
        if not d:
            nodet += 1
            continue
        r0, r1 = start_round(d, W), last_round(d, W)
        p1 &= r1 - r0 <= 1
        by.setdefault(r0, set()).add((d >> (r0 * W), o))
    return by, p1, nodet


def pair_check(wa, wb, Ta: int) -> dict:
    """tools_cde.window_preconditions (drift 0) on two window tables: models with T = Ta and Ta + 1."""
    sa, oka, na = wa
    sb, okb, nb = wb
    Tb = Ta + 1
    get = lambda s, r: s.get(r, set())  # noqa: E731
    interior = list(range(1, Tb - 1))
    ref = get(sb, interior[0]) if interior else set()
    p2 = all(get(sb, j) == ref for j in interior) and all(get(sa, j) == ref for j in range(1, Ta - 1))
    p3s, p3f, p3l = get(sa, 0) == get(sb, 0), get(sa, Ta) == get(sb, Tb), get(sa, Ta - 1) == get(sb, Tb - 1)
    p1 = bool(oka and okb and na == 0 and nb == 0)
    return dict(T=Ta, P1=p1, P2=bool(p2), P3=bool(p3s and p3f and p3l), P3_start=bool(p3s), P3_final=bool(p3f),
                P3_last=bool(p3l), interior_shapes=len(ref),
                holds=bool(p1 and p2 and p3s and p3f and p3l),
                time_shift_ok=bool(p1 and p2 and p3f and (Ta == 1 or p3l)))


def radii(F: Formula) -> dict:
    """R1, R2 and T_1 of Corollary 1: for T >= T_1 the window at start round r is
    Start_r (r < R1, T-independent), the interior shape set (R1 <= r <= T - R2), End_{T-r} (r > T - R2)."""
    W = F.W
    rho_S = max((last_round(d, W) for d, _ in F.S if d), default=-1)   # last round any S mechanism touches
    offs = [start_round(d, W) - 1 for d, _ in F.I if d]                  # start offset relative to i
    o_min, o_max = min(offs), max(offs)
    v_end = [F.D - start_round(d, W) for d, _ in F.End if d]           # T - start round
    rho_E = max(v_end)
    R1 = max(rho_S + 1, F.j_s + o_max)
    R2 = max(rho_E + 1, F.U + 1 - o_min)
    T1 = max(F.T0, R1 + R2, R1 - o_min + F.U, F.j_s + o_max + R2 - 1) + 1
    return dict(rho_S=rho_S, o_min=o_min, o_max=o_max, rho_E=rho_E, R1=R1, R2=R2, T1=T1)


def structure(F: Formula, rad: dict) -> dict:
    """At T_1: the near-start windows r = 1..R1-1 and the near-end windows r = T-v, v = 2..R2-1 equal the
    interior shape set (so only r = 0 and r = T-1, T differ, for EVERY T >= T_1); also every window r in
    [R1, T_1 - R2] equals it, and P1 holds."""
    T1 = rad["T1"]
    by, p1, nodet = windows(F.model(T1), F.W)
    get = lambda r: by.get(r, set())  # noqa: E731
    I_shapes = get(rad["R1"])
    mids = all(get(r) == I_shapes for r in range(rad["R1"], T1 - rad["R2"] + 1))
    near_start = {r: get(r) == I_shapes for r in range(1, rad["R1"])}
    near_end = {T1 - v: get(T1 - v) == I_shapes for v in range(2, rad["R2"])}
    return dict(T1=T1, P1=bool(p1 and nodet == 0), interior_shapes=len(I_shapes), middle_uniform=bool(mids),
                near_start_equal_interior=near_start, near_end_equal_interior=near_end,
                start_window=len(get(0)), last_window=len(get(T1 - 1)), final_window=len(get(T1)),
                holds=bool(p1 and nodet == 0 and mids and all(near_start.values()) and all(near_end.values())))
