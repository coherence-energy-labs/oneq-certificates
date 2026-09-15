r"""A PROOF THAT REPAIR CANNOT SUCCEED over the family it may use.

*** THE SHORTCUT THAT WAS CORRECTLY KILLED, AND THE ONE THAT REPLACES
IT. *** `matching_cert` runs FOUR exact greedy repair orders because the
order decides whether an attaining dual is found -- shedding can take
from a variable that absorbing then needs back. Dropping the later starts
was measured to lose real certifications, so that shortcut is dead and
stays dead. A heuristic "probably futile" pretest would trade
certifications for speed all over again.

This is not a pretest. It is a CERTIFICATE.

*** WHAT IT ACTUALLY BUYS, MEASURED -- AND IT IS NOT A SPEEDUP. ***
On surface-code shots at d=3 and d=5 (rotated memory Z, p=0.02),
certifying over the cut family each refused certificate actually carries:

    d=3   3 of 5 refusals proven futile      0 false skips
    d=5  17 of 25 refusals proven futile     0 false skips

So roughly two thirds of refusals ARE provably futile, exactly as the
proposal expected. But the exact LP that produces the proof costs 9.3 s
against the checker's 0.34 s for the same 150 shots -- twenty-seven times
more than the work it would skip. As a pre-test on this workload it is a
net loss, and saying otherwise would be reporting a mechanism at a value
its own measurement does not support.

WHERE IT IS FREE, AND THAT IS WHERE IT BELONGS. `certifying_decoder`'s
exact-rational tier ALREADY solves this LP, discovers that the optimum
falls short of the correction, and acts on it -- on the solver's word.
The dual multipliers that prove it were computed by the same simplex
iteration and thrown away. Returned instead (`exact_lp.exact_weighted_lp(
..., return_dual=True)`), they turn "our exact solver says the pool falls
short" into an object a stranger re-checks in one pass, at zero marginal
cost. That is an upgrade to the TRUST story, not to the clock, and it is
the honest reason to keep this module.

*** THE FAMILY IS THE ONE THE REPAIR MAY USE, AND GETTING THAT WRONG
PRODUCES EXACTLY A FALSE SKIP. *** Measured, on the way here: certifying
over the T-join separator's `pool` instead of over the certificate's own
`z` fired IMPOSSIBLE on 1 accepted shot at d=3 and 3 at d=5.
`MoatEngine.dual` ADDS its nested balls to the sets it is handed, so `z`
supports a strictly larger family than `pool`, and a sound certificate
about a subfamily says nothing about the superfamily. The certificate was
right; it was pointed at the wrong question.

THE LP. Let the repair's variables be the potentials it may raise -- y_n
for a node, z_S for a set -- each loading the edges of its cut, and let
`w_e` be the edge weights. The best objective any feasible dual can
reach is the value of the cut-packing LP

    PRIMAL      max  c^T x        s.t.  A x <= w,  x >= 0
    DUAL        min  w^T u        s.t.  A^T u >= c,  u >= 0

where `A[e][v] = 1` when edge e lies in variable v's cut, and `c_v = 1`
for the variables that PAY into the objective (every z_S, and y_n only
for a FIRED n -- a potential on an unfired node costs load and buys
nothing, which is why shedding spends those first).

Weak duality is the whole argument, and it is one line:

    for any feasible x and any feasible u,   c^T x  <=  w^T u.

So a single feasible `u` with `w^T u < W` proves that NO feasible dual
over this cut family attains the correction's primal weight W. All four
repair orders are then mathematically incapable of succeeding, and are
skipped -- not because they look unpromising, but because a checkable
object says they cannot work.

WHAT MAKES THIS SAFE TO ACT ON, and it is not the LP solver.

  THE SOLVER IS A PRODUCER, NEVER AN ACCEPT PATH. It proposes `u` in
  floats. Nothing downstream trusts that: `verify` re-checks every dual
  constraint and the objective in EXACT rational arithmetic, and a `u`
  that fails any of them yields INCONCLUSIVE. The same rule the rest of
  this repository runs on -- a producer may be wrong, slow or malicious
  and can only cost rate.

  INCONCLUSIVE NEVER MEANS POSSIBLE. Three verdicts, and only one of them
  licenses skipping work:

      IMPOSSIBLE     a checked feasible u with w^T u < W. Skip the repair
      INCONCLUSIVE   no such u was found, or the one found did not check.
                     Run the repair exactly as before
      (never)        "repair will succeed" -- this module cannot say that
                     and does not try

  THE FAILURE DIRECTION IS RATE, NOT SOUNDNESS -- but only just. A wrong
  IMPOSSIBLE skips a repair that would have certified, which costs a
  certification and publishes no false claim. That is still a false
  theorem, so the exact check is not optional and the sensitivity mode
  exists to prove it can refuse.

  IT IS ABOUT THIS CUT FAMILY, NOT ABOUT THE INSTANCE. `A` is built from
  the cuts the repair may actually use. A different family can have a
  larger packing value, so IMPOSSIBLE means "not from here", and the
  certificate says so in as many words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F

SCHEMA = "oneq-repair-impossible/1"

#: A verdict this module may return. `POSSIBLE` is deliberately absent.
IMPOSSIBLE = "IMPOSSIBLE"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Certificate:
    """A feasible dual point and the bound it proves. Self-contained.

    Everything a checker needs travels in the object: the edge weights it
    was built against, the cuts, and `u`. A certificate that needs the
    producer's memory to be re-checked is not a certificate.
    """
    schema: str
    verdict: str
    #: {edge key: u_e}, exact rationals as (numerator, denominator).
    u: dict
    #: {variable key: [edge keys in its cut]} -- only the PAYING ones.
    paying_cuts: dict
    #: {edge key: w_e}, as (numerator, denominator).
    weights: dict
    #: The correction weight the repair would have to attain.
    target: tuple
    #: w^T u, the certified upper bound on any dual from this family.
    bound: tuple
    why: str
    does_not_claim: str = (
        "that the correction is suboptimal, that a repair is impossible "
        "in general, or anything at all when the verdict is "
        "INCONCLUSIVE. It bounds what a dual built from THIS cut family "
        "can reach; a larger family can reach further")
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"schema": self.schema, "verdict": self.verdict,
                "u": {k: list(v) for k, v in self.u.items()},
                "paying_cuts": {k: list(v)
                                for k, v in self.paying_cuts.items()},
                "weights": {k: list(v) for k, v in self.weights.items()},
                "target": list(self.target), "bound": list(self.bound),
                "why": self.why, "does_not_claim": self.does_not_claim,
                "notes": dict(self.notes)}


def _frac(pair) -> F:
    return F(int(pair[0]), int(pair[1]))


def build_program(weights: dict, paying_cuts: dict):
    """(edge list, {paying variable: set of edges}) in a fixed order.

    The order is fixed because two implementations that enumerate the
    same LP in different orders can hand a solver different problems and
    disagree about a bound -- and disagreeing about a bound is how a
    conformance suite reports a defect in the wrong place.
    """
    edges = sorted(weights)
    cuts = {v: {e for e in cut if e in weights}
            for v, cut in sorted(paying_cuts.items())}
    return edges, cuts


def verify(cert: Certificate | dict) -> tuple[bool, str]:
    r"""Re-check an impossibility certificate in EXACT arithmetic.

    Independent of how `u` was produced -- that is the point. Four
    conditions, and all four are necessary:

        u >= 0                       a negative multiplier is not a dual
        A^T u >= c                   feasibility, one row per PAYING
                                     variable. An infeasible u bounds
                                     nothing, and this is the check an
                                     LP solver's floating-point answer
                                     most often fails by a whisker
        w^T u == the stated bound    the certificate must not misreport
                                     its own arithmetic
        bound < target               the bound has to actually bite.
                                     `w^T u == W` proves nothing: weak
                                     duality permits the packing to
                                     reach exactly W
    """
    d = cert.as_dict() if isinstance(cert, Certificate) else dict(cert)
    if d.get("schema") != SCHEMA:
        return False, f"not a {SCHEMA} document: {d.get('schema')}"
    if d.get("verdict") != IMPOSSIBLE:
        return False, (f"verdict is {d.get('verdict')!r}; only "
                       f"{IMPOSSIBLE} makes a claim to check")
    u = {k: _frac(v) for k, v in d["u"].items()}
    w = {k: _frac(v) for k, v in d["weights"].items()}
    target, bound = _frac(d["target"]), _frac(d["bound"])

    stray = sorted(set(u) - set(w))
    if stray:
        return False, (f"u puts weight on {stray[:3]}, which the "
                       f"certificate does not price -- an unpriced edge "
                       f"is a free multiplier and would prove anything")
    neg = sorted(k for k, v in u.items() if v < 0)
    if neg:
        return False, f"u is negative on {neg[:3]}; not a dual point"

    # *** A CERTIFICATE WITH NO CONSTRAINTS PROVES EVERYTHING. *** With no
    # paying variables the feasibility loop below runs zero times and any
    # non-negative u -- including the empty one -- "verifies" against any
    # positive target. That is genuinely true (an empty packing has value
    # 0) and it is exactly the shape a proof hides a bug in: hand this
    # function an empty family by mistake and it licenses skipping the
    # repair on every shot.
    #
    # So vacuity must be DECLARED. `notes.vacuous` is set by the producer
    # that meant it; an empty certificate that does not say so is
    # refused, which turns an accidental empty family from a silent
    # blanket skip into a visible refusal.
    if not d["paying_cuts"] and not d.get("notes", {}).get("vacuous"):
        return False, ("no paying variables and no declared vacuity: this "
                       "certificate constrains nothing, so it would "
                       "'prove' any positive target unreachable")

    for var, cut in d["paying_cuts"].items():
        total = sum((u.get(e, F(0)) for e in cut), F(0))
        if total < 1:
            return False, (f"dual constraint for {var} is violated: its "
                           f"cut carries {total} < 1, so u is infeasible "
                           f"and bounds nothing")
    got = sum((w[e] * u[e] for e in sorted(u)), F(0))
    if got != bound:
        return False, (f"the certificate states a bound of {bound} and "
                       f"its own u gives {got}")
    if not got < target:
        return False, (f"bound {got} does not beat the target {target}; "
                       f"weak duality permits the packing to reach it")
    return True, (f"feasible dual with w^T u = {got} < {target}: no dual "
                  f"over this cut family can attain the correction")


def _solve_dual(edges, cuts, weights):
    """Propose `u` with an LP solver. A PRODUCER, and may be wrong.

    Returns None when no solver is available or the LP is not solved --
    both of which mean INCONCLUSIVE, never IMPOSSIBLE. A missing
    dependency must not be able to license skipping work.
    """
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError:                                  # pragma: no cover
        return None
    if not edges or not cuts:
        return None
    idx = {e: i for i, e in enumerate(edges)}
    c = np.array([float(weights[e]) for e in edges], dtype=float)
    # A^T u >= 1  becomes  -A^T u <= -1
    rows = []
    for _v, cut in cuts.items():
        r = np.zeros(len(edges))
        for e in cut:
            r[idx[e]] = -1.0
        rows.append(r)
    res = linprog(c, A_ub=np.array(rows), b_ub=-np.ones(len(rows)),
                  bounds=[(0, None)] * len(edges), method="highs")
    if not res.success or res.x is None:
        return None
    return {e: float(res.x[i]) for e, i in idx.items()}


def _exactify(u_float: dict, cuts: dict, denom: int = 1 << 30) -> dict:
    r"""Float `u` -> exact rationals that are FEASIBLE.

    *** ROUNDING A FEASIBLE FLOAT SOLUTION USUALLY MAKES IT INFEASIBLE,
    BY A WHISKER. *** An LP solver returns a point on the constraint
    boundary; round it and half the tight rows fall a few units of the
    last place short. Verified exactly, that `u` bounds nothing, and the
    verdict would be INCONCLUSIVE on instances where the proof exists.

    So the point is rounded UP to a dyadic grid and then SCALED UP by the
    worst violated row's shortfall. Scaling a non-negative u by s >= 1
    multiplies every row by s and the objective by s: feasibility is
    restored and the bound gets weaker, never stronger. A weaker bound
    that verifies is worth more than a tight one that does not.
    """
    u = {e: F(int(v * denom) + (1 if v > 0 else 0), denom)
         for e, v in u_float.items()}
    worst = None
    for cut in cuts.values():
        total = sum((u.get(e, F(0)) for e in cut), F(0))
        if total <= 0:
            # A paying variable whose whole cut carries zero cannot be
            # rescued by scaling: 0 * s is 0. No certificate here.
            return {}
        if total < 1 and (worst is None or total < worst):
            worst = total
    if worst is not None:
        s = 1 / worst
        u = {e: v * s for e, v in u.items()}
    return u


def prove_impossible(weights: dict, paying_cuts: dict, target) -> Certificate:
    r"""Try to prove no dual from this cut family can attain `target`.

    Returns a Certificate whose verdict is IMPOSSIBLE (checked) or
    INCONCLUSIVE. Never raises for an ordinary failure to prove: the
    caller's correct response to INCONCLUSIVE is to run the repair, and a
    tool that threw on "I could not prove it" would make the common case
    an error path.
    """
    w = {e: F(v) for e, v in weights.items()}
    tgt = F(target)
    edges, cuts = build_program(w, paying_cuts)

    def inconclusive(why: str) -> Certificate:
        return Certificate(schema=SCHEMA, verdict=INCONCLUSIVE, u={},
                           paying_cuts={}, weights={},
                           target=(tgt.numerator, tgt.denominator),
                           bound=(0, 1), why=why)

    if not cuts:
        # NO PAYING VARIABLE MEANS THE PACKING IS EMPTY, so its value is
        # 0 -- which really does prove nothing can reach a positive
        # target. Stated explicitly rather than falling out of an empty
        # loop, because "vacuously true" is where a proof usually hides
        # a bug.
        if tgt > 0:
            return Certificate(
                schema=SCHEMA, verdict=IMPOSSIBLE, u={}, paying_cuts={},
                weights={}, target=(tgt.numerator, tgt.denominator),
                bound=(0, 1),
                why=("no variable pays into the objective, so the packing "
                     "value is exactly 0 and cannot reach a positive "
                     "target"),
                notes={"vacuous": True})
        return inconclusive("no paying variables and a non-positive target")

    proposed = _solve_dual(edges, cuts, w)
    if proposed is None:
        return inconclusive(
            "no LP solution was obtained (solver absent or the program "
            "was not solved). This is NOT evidence that repair can "
            "succeed -- run it")
    u = _exactify(proposed, cuts)
    if not u:
        return inconclusive("the proposed dual could not be made feasible "
                            "in exact arithmetic")
    bound = sum((w[e] * u[e] for e in sorted(u)), F(0))
    cert = Certificate(
        schema=SCHEMA, verdict=IMPOSSIBLE,
        u={str(e): (v.numerator, v.denominator) for e, v in u.items()},
        paying_cuts={str(v): [str(e) for e in sorted(cut)]
                     for v, cut in cuts.items()},
        weights={str(e): (v.numerator, v.denominator)
                 for e, v in w.items()},
        target=(tgt.numerator, tgt.denominator),
        bound=(bound.numerator, bound.denominator),
        why=f"w^T u = {bound} < {tgt}",
        notes={"edges": len(edges), "paying_variables": len(cuts)})
    ok, why = verify(cert)
    if not ok:
        # THE PRODUCER'S ANSWER DID NOT CHECK. That is an ordinary
        # outcome, not an error: the LP is solved in floats and the
        # verdict is decided in exact arithmetic, so they are allowed to
        # disagree. What is NOT allowed is acting on the float.
        return inconclusive(f"the proposed dual did not verify: {why}")
    return cert


def prove_impossible_exact(constraint_rows, weights, target, *,
                           objective=None, edge_keys=None,
                           variable_keys=None, max_iter=5000) -> Certificate:
    r"""The same proof, from the EXACT simplex. No float anywhere.

    *** THE FLOAT ROUTE COULD NOT RESOLVE THE GAP IT WAS BUILT TO
    CERTIFY. *** Measured on 356 surface-code shots at d=3 and d=5:

        the pool's LP optimum EXACTLY equals the correction   251
        the pool's LP optimum is strictly SHORT                19
        shortfall, when it happens        4.44e-16 to 8.88e-16

    One ulp of a double near 1. `prove_impossible` solves the dual in
    floats and rounds it onto a 2^30 grid, inflating its bound by about
    1e-9 -- seven orders of magnitude larger than the thing it is trying
    to detect. It fired on 0 of 356. An instrument that cannot resolve
    the gap it ranks does not report a small effect, it reports none.

    `exact_weighted_lp` already computes the exact dual multipliers at
    its optimum and was throwing them away. Taken instead of a float
    solve, the bound is EXACT, the certificate needs no rounding, and it
    fires on exactly the shots where a shortfall exists.

    `constraint_rows` is the same (edges x variables) matrix
    `certifying_decoder` already builds for `exact_cut_packing_lp`, and
    `weights` the same edge weights, so this is a second reading of a
    computation the decoder performs anyway.
    """
    from .exact_lp import exact_weighted_lp

    n_vars = len(constraint_rows[0]) if constraint_rows else 0
    c = ([F(1)] * n_vars if objective is None
         else [F(v) for v in objective])
    w = [F(v) for v in weights]
    tgt = F(target)
    ek = (list(edge_keys) if edge_keys is not None
          else list(range(len(constraint_rows))))
    vk = (list(variable_keys) if variable_keys is not None
          else list(range(n_vars)))

    def inconclusive(why: str) -> Certificate:
        return Certificate(schema=SCHEMA, verdict=INCONCLUSIVE, u={},
                           paying_cuts={}, weights={},
                           target=(tgt.numerator, tgt.denominator),
                           bound=(0, 1), why=why)

    z, obj, _it, u = exact_weighted_lp(constraint_rows, w, c,
                                       max_iter=max_iter, return_dual=True)
    if z is None or u is None:
        return inconclusive(f"the exact LP did not reach an optimum: {_it}. "
                            f"NOT evidence that repair can succeed -- run it")
    bound = sum((w[i] * v for i, v in sorted(u.items())), F(0))
    if not bound < tgt:
        return inconclusive(
            f"the exact optimum over this family is {bound}, which reaches "
            f"the target {tgt}. An attaining dual EXISTS here, so the "
            f"repair is not futile -- and the LP has already found one")

    cert = Certificate(
        schema=SCHEMA, verdict=IMPOSSIBLE,
        u={str(ek[i]): (v.numerator, v.denominator)
           for i, v in sorted(u.items())},
        paying_cuts={
            str(vk[j]): [str(ek[i]) for i in range(len(constraint_rows))
                         if constraint_rows[i][j]]
            for j in range(n_vars) if c[j] > 0},
        weights={str(ek[i]): (w[i].numerator, w[i].denominator)
                 for i in range(len(w))},
        target=(tgt.numerator, tgt.denominator),
        bound=(bound.numerator, bound.denominator),
        why=f"exact LP optimum {bound} < {tgt}",
        notes={"exact": True, "primal_objective": str(obj),
               "variables": n_vars, "edges": len(constraint_rows),
               # An empty program really does have packing value 0, so a
               # positive target really is unreachable -- but a
               # certificate that constrains nothing must SAY it is
               # vacuous or `verify` refuses it. See the guard there.
               **({"vacuous": True} if not any(c[j] > 0
                                               for j in range(n_vars))
                  else {})})
    ok, why = verify(cert)
    if not ok:                                       # pragma: no cover
        return inconclusive(f"the exact dual did not verify: {why}")
    return cert


def cut_family_from(fired, y_keys, z_keys, edges) -> dict:
    r"""The paying variables of `matching_cert`'s repair, as cuts.

    Mirrors `_repair_and_verify`: `y_n` loads every edge incident to n
    and pays only when n is FIRED; `z_S` loads every edge crossing S and
    always pays. A non-paying variable imposes the dual constraint
    `A^T u >= 0`, which any u >= 0 satisfies, so it is omitted rather
    than carried as a row that can never bind.
    """
    fired = {int(n) for n in fired}
    out: dict = {}
    for n in y_keys:
        if int(n) in fired:
            out[("y", int(n))] = {e for e in edges if int(n) in e}
    for S in z_keys:
        s = {int(x) for x in S}
        out[("z", tuple(sorted(s)))] = {
            e for e in edges if (e[0] in s) != (e[1] in s)}
    return out


__all__ = ["Certificate", "IMPOSSIBLE", "INCONCLUSIVE", "SCHEMA",
           "prove_impossible_exact",
           "build_program", "cut_family_from", "prove_impossible", "verify"]
