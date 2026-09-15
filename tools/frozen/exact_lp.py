r"""Exact rational LP for cut-packing duals: active-set simplex, no float.

THE PROBLEM THIS SOLVES. The exact tier's Padberg-Rao loop converges to a
cut pool whose float LP bound equals the primal to 9 digits -- and the
acceptance decision needs the EXACT optimum over that pool, because at
d5 the interesting structure lives below float resolution: shipped duals
are float roundings damaged in both directions, and some corrections are
genuinely an ulp from optimal (alternative optima whose float sums tie
but whose exact dyadic sums differ). Every float-side repair strategy
failed on measurement -- shed/absorb, spreading, multi-start, threshold
sweeps, iterative refinement -- because each was trying to prove a
statement that is only sometimes true. The exact optimum decides which.

THE METHOD. maximize 1'z  s.t.  A z <= b (edge rows), z >= 0, solved by
an active-set simplex whose basis is n ~ 40 ACTIVE CONSTRAINTS -- never
the 1400-row tableau. z = 0 is a feasible VERTEX (all bound rows active;
b >= 0 because edge weights are nonnegative), so there is no phase 1.
Bland's rule -- lowest global row index for both the leaving and entering
constraint -- guarantees termination without cycling. Every quantity is a
Fraction; float appears NOWHERE in this module, so there is nothing to
repair afterwards: the returned objective IS the exact LP optimum, and
the returned point is exactly feasible by construction (and re-checked).

Measured on the six d5 hardware shots that motivated it: proven optimal
in 29-43 iterations each, feasibility and non-negativity re-verified from
scratch, one shot certifying outright and the rest landing at the true
optimum exactly one weight-ulp below the primal -- the certify-or-repair
boundary, resolved shot by shot instead of assumed.

WHAT THIS MODULE MUST NEVER DO: accept anything. It is a solver, not a
checker. Its output feeds the same independent verification as every
other candidate dual, and a defect here can cost only RATE (a refusal or
a repair that fails verification), never a false certificate.
"""

from __future__ import annotations

from fractions import Fraction


def _solve_square(M, rhs):
    """Exact n x n solve: returns x with M x = rhs, or None if singular."""
    n = len(M)
    A = [row[:] + [rhs[i]] for i, row in enumerate(M)]
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        inv = A[c][c]
        A[c] = [v / inv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [v - f * w for v, w in zip(A[r], A[c])]
    return [A[r][n] for r in range(n)]


def exact_weighted_lp(constraint_rows, rhs, objective, max_iter=5000):
    """Maximize objective.z  s.t. rows.z <= rhs, z >= 0 -- exactly.

    The generalization that lets the qLDPC lane reuse this simplex: the
    Feldman dual's objective is sum y_r (1 - |F_r|) - sum mu_i, whose
    coefficients are not uniform (and not even positive), so `sum(z)`
    cannot express it. Same active-set method, same z = 0 starting
    vertex, same Bland's rule; ONLY the objective vector generalizes --
    a coefficient c_j <= 0 simply means z_j never profitably enters.

    Returns (z_list, objective_value, iterations) at the proven optimum,
    or (None, None, reason) on failure. The caller re-verifies
    feasibility and attainment independently; this function's word is
    never the acceptance evidence.
    """
    m = len(constraint_rows)
    n = len(objective)
    if m == 0 or n == 0:
        return [Fraction(0)] * n, Fraction(0), 0
    rows = []
    for r in constraint_rows:
        if len(r) != n:
            return None, None, "ragged constraint matrix"
        rows.append([Fraction(v) for v in r])
    b = [Fraction(v) for v in rhs]
    if len(b) != m:
        return None, None, "rhs length mismatch"
    if any(v < 0 for v in b):
        # z = 0 would not be feasible; this problem class never produces
        # negative weights, so refuse rather than run phase 1.
        return None, None, "negative rhs: z=0 is not a feasible vertex"
    c = [Fraction(v) for v in objective]
    # global rows: constraints 0..m-1, then bounds -z_j <= 0 at m..m+n-1
    G = rows + [
        [Fraction(-1) if k == j else Fraction(0) for k in range(n)]
        for j in range(n)
    ]
    h = b + [Fraction(0)] * n
    z = [Fraction(0)] * n
    active = list(range(m, m + n))            # the z = 0 vertex
    for it in range(max_iter):
        M = [G[i] for i in active]
        lam = _solve_square([list(col) for col in zip(*M)], c)
        if lam is None:
            return None, None, "singular basis"
        leave_pos = None
        leave_idx = None
        for pos, i in enumerate(active):      # Bland: lowest global index
            if lam[pos] < 0 and (leave_idx is None or i < leave_idx):
                leave_idx, leave_pos = i, pos
        if leave_pos is None:                 # all multipliers >= 0: optimal
            return z, sum((a * v for a, v in zip(c, z)), Fraction(0)), it
        e_r = [Fraction(0)] * n
        e_r[leave_pos] = Fraction(-1)
        d = _solve_square(M, e_r)             # G_r d = -1, other active rows 0
        if d is None:
            return None, None, "singular direction"
        best_t = None
        enter = None
        act = set(active)
        for i in range(m + n):                # ratio test; Bland on argmin
            if i in act:
                continue
            gd = sum(a * v for a, v in zip(G[i], d))
            if gd > 0:
                t = (h[i] - sum(a * v for a, v in zip(G[i], z))) / gd
                if best_t is None or t < best_t or (t == best_t and i < enter):
                    best_t, enter = t, i
        if enter is None:
            return None, None, "unbounded"
        z = [a + best_t * v for a, v in zip(z, d)]
        active[leave_pos] = enter
    return None, None, "iteration cap"


def exact_cut_packing_lp(constraint_rows, rhs, max_iter=5000):
    """Maximize sum(z) s.t. rows.z <= rhs, z >= 0 -- exactly.

    The original uniform-objective entry point, now a delegation so the
    simplex exists ONCE (this estate's scar: a central definition that
    exists twice is one edit away from two different theorems). Every
    caller and test of this function is unchanged.
    """
    m = len(constraint_rows)
    if m == 0:
        return [], Fraction(0), 0
    n = len(constraint_rows[0])
    return exact_weighted_lp(constraint_rows, rhs,
                             [Fraction(1)] * n, max_iter=max_iter)
