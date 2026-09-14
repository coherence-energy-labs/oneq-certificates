r"""Certificates for qLDPC decoding: Feldman-LP duals, checked independently.

THE SAME SEPARATION AS THE MATCHING CERTIFICATE, on the problem where it does
not come for free. Minimum-weight qLDPC decoding (He = s over GF(2)) is
NP-hard, so unlike matching there is no guarantee every shot certifies. What
survives is the two-tier receipt: when a feasible dual of Feldman's LP
relaxation meets the decoder's weight, the answer is PROVEN minimum-weight by
weak duality -- LP-dual value <= LP optimum <= integer optimum -- and when it
does not, the shot carries a measured gap instead of silence.

THE RELAXATION. Each check j with support N(j) and syndrome bit s_j relaxes to
its local codeword polytope: the convex hull of assignments on N(j) with
parity s_j. Its facets are, for every F subset of N(j) with |F| != s_j mod 2,

    sum_{i in F} (1 - x_i) + sum_{i in N(j) minus F} x_i  >=  1

(every wrong-parity vertex is Hamming distance >= 1 away). The LP minimizes
w.x over the intersection and [0,1]^n.

THE DUAL A STRANGER CAN CHECK. Assign y_{j,F} >= 0 to each facet and
mu_i >= 0 to each x_i <= 1 box. Rearranged per variable, the dual constraint is

    sum_{(j,F): i in N(j)} sign(i, F) * y_{j,F} - mu_i  <=  w_i
        where sign(i, F) = -1 if i in F else +1

and the dual objective is

    sum_{j,F} (1 - |F|) * y_{j,F} ... rearranged: each facet contributes
    y * (1 - |F|) from moving constants across; worked into canonical form
    below so the checker is one pass over the certificate plus one pass over
    the variables. Every quantity is re-derived from (H, w, s) and the listed
    support -- the producer is never consulted.

Facet counts are 2^(deg-1) per check, enumerable at LDPC degrees; the
certificate carries only its SUPPORT.

HONEST SCOPE, stated here because this module will be quoted: the LP bound is
not always tight (pseudocodewords exist), so unlike matching a 100% rate is
NOT promised by theory. The rate is an empirical property per code family, to
be measured. Min-weight is conformance to a specification, not
maximum-likelihood over degenerate cosets.
"""

from __future__ import annotations

# THE CHECKER LIVES IN ITS OWN MODULE (qldpc_check) with no producer
# imports -- independence by module graph, enforced by the standards gate.
# Re-exported here so every existing import path keeps working.
from fractions import Fraction

from .qldpc_check import (QldpcCertificate, QldpcCheckResult,  # noqa: F401
                          _exact_safe_dual_bound, check_qldpc,
                          check_qldpc_exact)


def _polish_if_short(checks, weights, syn, cert, primal):
    """Return `cert` unless its dual provably falls short of an integral
    primal in EXACT arithmetic -- then return the exact re-solve instead.

    The producer half of the exact acceptance contract: the checker's
    gate (2026-08-14) refuses ulp-short float duals, so a producer that
    ships them emits receipts its own checker refuses. This tests
    attainment with the checker's own safe-bound scan (one exact pass,
    cheap) and runs the LP re-solve ONLY on the shortfall cases (~6% on
    random ensembles, fewer on real corpora). Every failure path keeps
    the original certificate: this function can change RATE, never
    soundness, because its output faces the same checker as any other
    candidate.
    """
    if cert is None or primal is None or not cert.facet_duals:
        return cert
    try:
        wq = {int(i): Fraction(v) for i, v in weights.items()}
        sups = [frozenset(c) for c in checks]
        synq = [int(b) & 1 for b in syn]
        fd = [(j, tuple(F), Fraction(y)) for (j, F, y) in cert.facet_duals]
        pw = sum((wq[int(i)] for i in primal), Fraction(0))
        L, err = _exact_safe_dual_bound(sups, synq, wq, fd)
        if err is None and L is not None and L == pw:
            return cert                      # already exactly tight
        polished = exact_dual_polish(checks, weights, syn, cert)
        if polished is None:
            return cert
        fd2 = [(j, tuple(F), y) for (j, F, y) in polished.facet_duals]
        L2, err2 = _exact_safe_dual_bound(sups, synq, wq, fd2)
        if err2 is None and L2 is not None and L2 == pw:
            return polished
        return cert
    except (TypeError, ValueError, KeyError, OverflowError, IndexError):
        return cert


def exact_dual_polish(checks, weights, syndrome, cert: QldpcCertificate):
    """Re-solve the certificate's OWN facet pool exactly; return a dual
    that attains the primal exactly, or None.

    WHY. The exact acceptance gate (2026-08-14) refuses a certificate
    whose dual does not attain the correction's weight in exact
    arithmetic. feldman_dual's duals are scipy floats: on random
    ensembles ~6% of integral-primal certificates land ulp-scale short
    and are refused -- correctly, but the refusal is pure dual ROUNDING,
    not pool insufficiency. If the float LP's optimum really was the
    integral primal, then over the SAME facet pool the exact dual
    optimum attains it exactly, so an exact re-solve closes every such
    case. Measured before wiring: 23/23 refused certificates closed,
    0 errors (400-instance ensemble, rpc_rounds=4).

    NOT the dead approach: the atlas falsification library records
    float-side repair (limit_denominator, shed/absorb, rescale) as
    measured dead for exact closure, because the true dual vertex is a
    dyadic rational no bounded denominator recovers. This is the
    sanctioned operator instead -- float LP for STRUCTURE (which facets),
    exact_lp for the VALUE -- ported from coset_lower, where it took the
    parity-cut tier from 8/106 to 196/204 exact.

    mu (box) variables are included because they are genuinely needed:
    a variable at its upper bound with negative reduced cost prices its
    bound constraint, and a polish without mu is short on exactly those
    vertices. Soundness never rests here: the output goes through the
    same checker as every other candidate, so a defect costs RATE only.
    """
    supports = [frozenset(c) for c in checks]
    fd = list(cert.facet_duals)
    if not fd:
        return None
    resolved = []
    for (j, Fs, _y) in fd:
        try:
            if isinstance(j, (tuple, list, frozenset)):
                sup: frozenset = frozenset()
                for r in sorted({int(x) for x in j}):
                    sup ^= supports[r]
            else:
                sup = supports[int(j)]
            resolved.append((frozenset(int(i) for i in Fs), sup))
        except (TypeError, ValueError, IndexError):
            return None                       # malformed facet: not ours to fix
    var_set: set = set()
    for (_Fs, sup) in resolved:
        var_set |= sup
    vars_used = sorted(var_set)
    vidx = {v: k for k, v in enumerate(vars_used)}
    nR, nv = len(resolved), len(vars_used)
    rows, rhs = [], []
    for v in vars_used:
        row = [Fraction(0)] * (nR + nv)
        # fresh names: `Fs`/`sup` bind tuples in the resolution loop above
        # and mypy pins a name to its first binding
        for k, (fset, fsup) in enumerate(resolved):
            if v in fsup:
                row[k] = Fraction(-1) if v in fset else Fraction(1)
        row[nR + vidx[v]] = Fraction(-1)      # ... - mu_v <= w_v
        rows.append(row)
        try:
            w = Fraction(weights[v])
        except (KeyError, TypeError, ValueError):
            return None
        if w < 0:
            return None
        rhs.append(w)
    obj = ([Fraction(1 - len(Fs)) for (Fs, _s) in resolved]
           + [Fraction(-1)] * nv)
    from .exact_lp import exact_weighted_lp
    z, _val, _it = exact_weighted_lp(rows, rhs, obj)
    if z is None:
        return None
    new_fd = tuple((fd[k][0], tuple(fd[k][1]), z[k]) for k in range(nR))
    new_bd = tuple((vars_used[i], z[nR + i]) for i in range(nv)
                   if z[nR + i] != 0)
    return QldpcCertificate(error_support=cert.error_support,
                            facet_duals=new_fd, box_duals=new_bd)


def exactify_certificate(cert: QldpcCertificate, *,
                         max_denominator: int = 10**9) -> QldpcCertificate:
    """Rationalize a float dual into an exact-arithmetic certificate.

    Sound by the safe-dual construction: check_qldpc_exact derives the
    box slacks itself, so ANY exact nonnegative y is a fresh valid dual
    candidate whose bound is recomputed exactly from scratch. NOTE
    (external audit round 2): rationalization may move the bound in
    EITHER direction -- it perturbs both b.y and the derived slacks --
    and soundness does not depend on any monotonicity claim; it holds
    because the checker treats the rationalized multipliers as a new
    candidate and re-derives everything. The only risk of aggressive
    rationalization is a refused (not a false) certificate. Box duals
    are dropped: the exact checker's derived z supersedes them.
    """
    fd = tuple(
        (j, tuple(F),
         Fraction(max(0.0, float(y))).limit_denominator(max_denominator))
        for (j, F, y) in cert.facet_duals)
    return QldpcCertificate(error_support=cert.error_support,
                            facet_duals=fd, box_duals=())


def fixed_point_weights(float_weights, *, scale: int = 1000):
    """The published rational objective for heterogeneous channels.

    c_i = floor(scale * w_i + 1/2) as a plain integer, where w_i is the
    float64 log-likelihood ratio log((1-p_i)/p_i), nonnegative for
    p_i < 1/2. The rule is stated portably (round-half-up via floor;
    "round()" alone is not a specification -- ties and precision differ
    across languages, the external audit's point) and the certified
    specification is THIS integer objective: exact log-odds weights
    select the most likely independent mechanism pattern; the
    fixed-point weights select the optimum of the declared quantized
    surrogate, and the certificate proves the latter. Returns
    (weights, spec): the spec publishes the scale, the rounding rule,
    and a sha256 of the complete integer weight vector in index order,
    so a stranger can rebuild and compare the objective bit-for-bit.
    Integer weights put every feasible objective on the unit lattice,
    so the exact checker's acceptance rule is L > U - 1.
    """
    import hashlib
    import json as _json
    import math as _math
    c = {int(i): int(_math.floor(scale * float(w) + 0.5))
         for i, w in float_weights.items()}
    if any(v < 0 for v in c.values()):
        raise ValueError("negative fixed-point weight: check the channel")
    blob = _json.dumps([c[k] for k in sorted(c)]).encode()
    spec = {"scale": scale,
            "rounding": "floor(scale*llr + 1/2), float64 llr, half-up",
            "definition": "c_i = floor(scale * log((1-p_i)/p_i) + 1/2)",
            "weights_sha256": hashlib.sha256(blob).hexdigest()}
    return c, spec

# per-(code, parity) facet blocks; see the cache note inside feldman_dual
_FACET_CACHE: dict = {}


def most_violated_facet(xvals, parity):
    """The O(deg) separation rule, as a pure function of local values.

    xvals are the fractional coordinates on one check's support (local
    positions 0..deg-1); parity is that check's syndrome bit. Returns the
    LOCAL index tuple of the most violated facet: F0 = {k: x_k > 1/2}
    minimizes LHS unconstrained, and if its parity is wrong (a facet needs
    |F| != parity mod 2) the repair is the single flip nearest 1/2 --
    exact, not heuristic, since LHS(F) = LHS(F0) + sum_{F^F0} 2|x-1/2|
    (Taghavi & Siegel 2008). Lives as its own function so the exactness
    artifact (qldpc_claims_backing_gate.py) brute-forces THE SHIPPED RULE,
    not a transcription of it.
    """
    F = [k for k in range(len(xvals)) if xvals[k] > 0.5]
    if len(F) % 2 == (parity & 1):
        flip = min(range(len(xvals)), key=lambda k: abs(xvals[k] - 0.5))
        F = sorted(set(F) ^ {flip})
    return tuple(F)


def _lazy_separate(supports, syn, vidx, wvec, n, max_rounds):
    """Grow the LP by ADDING only the facets the solution path violates.

    Returns (res, x, rhs, meta), where res is None if nothing was ever
    violated (the zero-syndrome edge), or (None, None, None, None) if a
    solve failed.

    Round 0 seeds against x = 0: the most violated facet there is F = {}
    for parity-1 checks and F = {cheapest single} for parity-0, which is
    exactly what the greedy separator returns, so no special case is
    needed -- just run separation against the zero point.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    rows_r: list = []
    rows_c: list = []
    rows_v: list = []
    rhs: list = []
    meta: list = []
    seen: set = set()

    def add_facet(j, F):
        key = (j, F)
        if key in seen:
            return False
        seen.add(key)
        r = len(rhs)
        Fs = set(F)
        for i in supports[j]:
            rows_r.append(r)
            rows_c.append(vidx[i])
            rows_v.append(1.0 if i in Fs else -1.0)
        rhs.append(float(len(F) - 1))
        meta.append((j, F))
        return True

    x = np.zeros(n)
    res = None
    for _round in range(max_rounds):
        added = 0
        for j, sup in enumerate(supports):
            if not sup:
                continue
            xv = [float(x[vidx[i]]) for i in sup]
            F = tuple(sup[k] for k in most_violated_facet(xv, syn[j]))
            viol = sum(1.0 - x[vidx[i]] for i in F) \
                + sum(x[vidx[i]] for i in sup if i not in set(F))
            if viol < 1.0 - 1e-7 and add_facet(j, F):
                added += 1
        if added == 0:
            break                       # no facet in the FULL family violated
        A = csr_matrix((rows_v, (rows_r, rows_c)), shape=(len(rhs), n))
        res = linprog(wvec, A_ub=A, b_ub=np.array(rhs), bounds=(0.0, 1.0),
                      method="highs")
        if not res.success:
            return None, None, None, None
        x = np.asarray(res.x)
    return res, x, rhs, meta


def feldman_dual_lazy(checks, weights, syndrome, *, max_rounds: int = 60,
                      return_primal: bool = False, tol: float = 1e-9,
                      return_x: bool = False):
    r"""The Feldman LP by PURE LAZY FACET SEPARATION -- no enumeration, ever.

    The insight that retires both the 2^(deg-1) explosion and the capped-
    facet tightness question: for a check with support N(j) and a fractional
    point x*, the MOST VIOLATED facet is found greedily in O(deg) --
    include i in F exactly when x*_i > 1/2 (that choice minimizes
    sum_{F}(1-x_i) + sum_{N\F} x_i), then repair the parity by flipping the
    coordinate nearest 1/2. So the LP starts with NO facet rows at all and
    grows only the ones the solution path actually violates: rows stay
    support-sized, any detector degree is welcome, and at convergence no
    facet of the FULL family is violated -- the bound equals full
    enumeration's, not an approximation of it.

    Same certificate type, same checker, same duals-off-the-marginals --
    literally the same extraction function as the enumerating path, which
    is the point: two producers reading scipy's marginals two different
    ways is how the two paths would drift into disagreeing about the same
    LP.
    """
    import numpy as np

    # Annotated because the arity genuinely varies with the flags: mypy
    # otherwise pins the type to whichever branch it sees first and then
    # calls the `return_x` extension an incompatible assignment.
    _nores: tuple = (None, 0.0, None) if return_primal else (None, 0.0)
    if return_x:
        _nores = _nores + (None,)
    supports = [tuple(sorted(int(i) for i in c)) for c in checks]
    syn = [int(b) & 1 for b in syndrome]
    var_ids = sorted(weights)
    vidx = {v: k for k, v in enumerate(var_ids)}
    n = len(var_ids)
    wvec = np.array([float(weights[v]) for v in var_ids])

    res, x, rhs, meta = _lazy_separate(supports, syn, vidx, wvec, n,
                                       max_rounds)
    if res is not None and x is None:
        return _nores
    if res is None and rhs is None:     # a solve failed
        return _nores
    if res is None:                     # zero-syndrome edge: nothing violated
        empty = QldpcCertificate(error_support=(), facet_duals=())
        base = (empty, 0.0, ()) if return_primal else (empty, 0.0)
        return base + ({},) if return_x else base

    facets, boxes = _extract_dual(res, meta, rhs, var_ids, n,
                                  kind="lazy dual")
    cert = QldpcCertificate(error_support=(), facet_duals=facets,
                            box_duals=boxes)
    xdict = ({int(var_ids[k]): float(x[k]) for k in range(n)}
             if return_x else None)
    if not return_primal:
        return (cert, float(res.fun), xdict) if return_x \
            else (cert, float(res.fun))
    if np.max(np.abs(x - np.round(x))) < 1e-7:
        primal = tuple(int(var_ids[k]) for k in np.flatnonzero(
            np.round(x) > 0.5))
    else:
        primal = None
    cert = _polish_if_short(checks, weights, syndrome, cert, primal)
    if return_x:
        return cert, float(res.fun), primal, xdict
    return cert, float(res.fun), primal


def exact_milp(checks, weights, syndrome, *, time_limit: float = 30.0,
               return_status: bool = False):
    """Exact minimum-weight decoding as a MILP -- the DEGRADED receipt tier.

    He = s over GF(2) becomes sum_{i in N(j)} x_i - 2 k_j = s_j with integer
    slack k_j >= 0. HiGHS solves it exactly at [[144,12,12]] scale. The
    honesty line, stated where it will be quoted: this tier's optimality rests
    on TRUSTING THE MIP SOLVER -- there is no stranger-checkable witness, that
    is precisely what the certified tier above it has and this one lacks, and
    a receipt from this tier must say so. Its value is (1) exact decoding
    where the LP is loose, and (2) turning the unattributed residue into a
    measured attribution: LP-loose vs decoder-suboptimal.

    Returns (support tuple, weight) or (None, None) on failure/timeout.
    With return_status=True, returns (support, weight, status) where
    status carries the solver's own account -- status code, message, MIP
    gap, dual bound -- so a receipt can split SOLVER-REPORTED OPTIMUM
    (tier 3A: status optimal, zero gap) from FEASIBLE FALLBACK,
    OPTIMALITY UNKNOWN (tier 3B: time limit, nonzero gap, anything
    else). The external audit's point: feasibility plus a trusted label
    is not the same claim as a solver-asserted optimum, and the two must
    never share a name.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    supports = [tuple(sorted(int(i) for i in c)) for c in checks]
    syn = [int(b) & 1 for b in syndrome]
    var_ids = sorted(weights)
    vidx = {v: k for k, v in enumerate(var_ids)}
    n, m = len(var_ids), len(supports)
    maxdeg = max((len(s) for s in supports), default=0)
    A = lil_matrix((m, n + m))
    for j, sup in enumerate(supports):
        for i in sup:
            A[j, vidx[i]] = 1.0
        A[j, n + j] = -2.0
    c = np.concatenate([np.array([float(weights[v]) for v in var_ids]),
                        np.zeros(m)])
    lb = np.zeros(n + m)
    ub = np.concatenate([np.ones(n), np.full(m, maxdeg / 2.0 + 1.0)])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), syn, syn),
               integrality=np.ones(n + m), bounds=Bounds(lb, ub),
               options={"time_limit": float(time_limit)})
    gap = float(getattr(res, "mip_gap", float("nan")))
    status = {"status_code": int(getattr(res, "status", -1)),
              "message": str(getattr(res, "message", "")),
              "mip_gap": gap,
              "mip_dual_bound": float(getattr(res, "mip_dual_bound",
                                              float("nan"))),
              # tier 3A only on a clean optimal status with zero gap;
              # everything else -- time limit, nonzero gap, interruption
              # -- is tier 3B and may never be called optimal
              "tier": ("3A_solver_reported_optimum"
                       if res.success and getattr(res, "status", -1) == 0
                       and gap == 0.0
                       else "3B_feasible_optimality_unknown")}
    if not res.success or res.x is None:
        return (None, None, status) if return_status else (None, None)
    x = np.round(res.x[:n]).astype(int)
    support = tuple(var_ids[k] for k in range(n) if x[k] == 1)
    for j, sup in enumerate(supports):     # exactness re-checked, not assumed
        if sum(1 for i in sup if i in set(support)) % 2 != syn[j]:
            return (None, None, status) if return_status else (None, None)
    wt = float(sum(weights[i] for i in support))
    return (support, wt, status) if return_status else (support, wt)


def _facet_blocks(supports, syn, var_ids, vidx, n, max_check_degree,
                  max_facet_size):
    """Build (or reuse) the per-check facet row blocks for this syndrome.

    Returns (blocks, rhs, meta), or None if a check exceeds the degree cap.

    FACET BLOCKS ARE CACHED PER (CHECK, PARITY). Profiling on the
    [[144,12,12]] code put the LP at ~5 ms/shot and the pure-Python facet
    row construction at ~15 ms/shot (~50k list appends) -- rebuilt
    identically for every shot, even though each check has exactly TWO
    possible facet sets (syndrome bit 0 or 1). The blocks are built once
    per (code, parity) and per shot the right ones are stacked. The cache
    key carries the full supports and variable ordering, so a different
    code or weight-key set can never collide; values do not enter A.
    """
    from itertools import combinations

    from scipy.sparse import csr_matrix

    key = (tuple(supports), tuple(var_ids))
    cache = _FACET_CACHE.get(key)
    if cache is None:
        if len(_FACET_CACHE) >= 8:      # a handful of codes, never unbounded
            _FACET_CACHE.pop(next(iter(_FACET_CACHE)))
        cache = {}
        _FACET_CACHE[key] = cache
    blocks = []
    rhs: list = []
    meta: list = []
    for j, sup in enumerate(supports):
        d = len(sup)
        # DEGREE-CAPPED FACETS, the unlock for circuit-level detectors. A
        # circuit-level DEM detector can touch dozens of mechanisms, and full
        # facet families scale 2^(deg-1) -- unbuildable. Enumerating only the
        # subsets F with |F| <= max_facet_size keeps a SUBSET of the true
        # facets, which only ENLARGES the LP's feasible region: the minimum
        # can only drop, so the dual bound stays a sound lower bound on the
        # true optimum. What the cap costs is TIGHTNESS, and tightness is
        # measured per ensemble, never assumed. The checker needs no change:
        # every emitted facet is still a real facet.
        cap_sz = d if max_facet_size is None else min(d, max_facet_size)
        if d > max_check_degree and max_facet_size is None:
            return None                 # refuse loudly upstream, not silently
        ent = cache.get((j, syn[j], cap_sz))
        if ent is None:
            br, bc, bv, brhs, bmeta = [], [], [], [], []
            for size in range(cap_sz + 1):
                if size % 2 == syn[j]:
                    continue            # right parity: not a facet
                for F in combinations(sup, size):
                    r = len(brhs)
                    Fs = set(F)
                    for i in sup:
                        br.append(r)
                        bc.append(vidx[i])
                        bv.append(1.0 if i in Fs else -1.0)
                    brhs.append(float(len(F) - 1))
                    bmeta.append((j, F))
            ent = (csr_matrix((bv, (br, bc)), shape=(len(brhs), n))
                   if brhs else None, brhs, bmeta)
            cache[(j, syn[j], cap_sz)] = ent
        if ent[0] is not None:
            blocks.append(ent[0])
            rhs.extend(ent[1])
            meta.extend(ent[2])
    return blocks, rhs, meta


def _elimination_cuts(x, supports, syn, vidx, var_ids, ncols, rhs, meta,
                      ex, base_nrows, rpc_cap, seen):
    """Generate redundant-parity-check cuts that separate a fractional x.

    Zhang-Siegel-style adaptive cut generation. Redundant parity checks
    (GF(2) sums of rows of H) are implied constraints whose facets the
    base LP does not know. Naive PAIR sums were measured useless here.
    The generator that works ELIMINATES: row-reduce H over GF(2) pivoting
    on the fractional coordinates (most ambivalent first), so each
    surviving row touches at most one fractional position.

    Mutates rhs, meta and ex in place; returns the number of cuts added.
    """
    ex_r, ex_c, ex_v = ex
    fcols = sorted((k for k in range(ncols)
                    if abs(x[k] - round(x[k])) > 1e-7),
                   key=lambda k: abs(x[k] - 0.5))
    rowbits = []
    tags = []
    for jj, supt in enumerate(supports):
        b = 0
        for i in supt:
            b |= 1 << vidx[i]
        rowbits.append(b)
        tags.append(1 << jj)
    for c in fcols:
        mask = 1 << c
        piv = next((r for r in range(len(rowbits))
                    if rowbits[r] & mask), None)
        if piv is None:
            continue
        for r in range(len(rowbits)):
            if r != piv and (rowbits[r] & mask):
                rowbits[r] ^= rowbits[piv]
                tags[r] ^= tags[piv]
    added = 0
    for b, tg in zip(rowbits, tags):
        if b == 0 or tg == 0 or tg & (tg - 1) == 0:
            continue                # empty, or a single original row
        S = [k for k in range(ncols) if (b >> k) & 1]
        rows_used = tuple(r for r in range(len(supports))
                          if (tg >> r) & 1)
        par = 0
        for r in rows_used:
            par ^= syn[r]
        F = {k for k in S if x[k] > 0.5}
        if len(F) % 2 == par:
            flip = min(S, key=lambda k: abs(x[k] - 0.5))
            F ^= {flip}
        viol = sum(1.0 - x[k] for k in F) + sum(x[k] for k in S
                                                if k not in F)
        if viol > 1.0 - 1e-9:
            continue                # this point does not violate it
        key = (rows_used, frozenset(F))
        if key in seen:
            continue
        seen.add(key)
        r_ = len(rhs) - base_nrows
        for k in S:
            ex_r.append(r_)
            ex_c.append(k)
            ex_v.append(1.0 if k in F else -1.0)
        rhs.append(float(len(F) - 1))
        meta.append((rows_used,
                     tuple(sorted(var_ids[k] for k in F))))
        added += 1
        if added >= rpc_cap:
            break
    return added


def _rpc_cutting_loop(res, solve, rpc_rounds, supports, syn, vidx, var_ids,
                      ncols, rhs, meta, ex, base_nrows, rpc_cap):
    """Cut, re-solve, repeat -- keeping the last GOOD solve.

    Returns the final LP result.
    """
    import numpy as np

    ex_r, ex_c, ex_v = ex
    seen: set = set()
    for _round in range(int(rpc_rounds)):
        snap_rhs, snap_ex = len(rhs), len(ex_r)
        x = np.asarray(res.x)
        frac = np.abs(x - np.round(x))
        if frac.max() < 1e-7:
            break                       # integral: nothing left to cut
        added = _elimination_cuts(x, supports, syn, vidx, var_ids, ncols,
                                  rhs, meta, ex, base_nrows, rpc_cap, seen)
        if added == 0:
            break                   # no elimination cut separates this point
        res2 = solve()
        if not res2.success:
            # keep the last good solve AND roll back this round's rows,
            # or the dual extraction below indexes marginals the kept
            # solve never produced (latent until the certified-B&B
            # producer hit it on a reduced instance; the audit flagged
            # the first fix attempt as half-finished -- this is the
            # snapshot-based version)
            del rhs[snap_rhs:]
            del meta[snap_rhs:]
            del ex_r[snap_ex:]
            del ex_c[snap_ex:]
            del ex_v[snap_ex:]
            break
        res = res2
    return res


def _extract_dual(res, meta, rhs, var_ids, n, *, kind="dual"):
    """Read the facet and box duals off scipy's marginals, and CHECK them.

    PRODUCER SELF-CHECK: the extracted dual's objective, computed the way
    the CHECKER computes it, must equal the LP optimum. Strong duality
    guarantees the true dual does; if the reconstruction disagrees, the
    extraction (a sign, a marginal convention, a dropped term) is wrong and
    saying so here costs one comparison -- finding it downstream cost a
    refused-but-tight certificate and a debugging round.
    """
    import numpy as np

    y = np.abs(res.ineqlin.marginals)
    facets = tuple((meta[r][0], tuple(meta[r][1]), float(y[r]))
                   for r in range(len(rhs)) if y[r] > 1e-12)
    ub = np.abs(res.upper.marginals)
    boxes = tuple((var_ids[k], float(ub[k]))
                  for k in range(n) if ub[k] > 1e-12)
    recon = sum(v * (1.0 - len(F)) for (_j, F, v) in facets) \
        - sum(v for (_i, v) in boxes)
    if abs(recon - float(res.fun)) > 1e-6:
        raise AssertionError(
            f"{kind} extraction broken: reconstructed objective {recon} "
            f"!= LP optimum {float(res.fun)}")
    return facets, boxes


def feldman_dual(checks, weights, syndrome, *, max_check_degree: int = 12,
                 return_primal: bool = False, rpc_rounds: int = 0,
                 rpc_cap: int = 400, max_facet_size: int | None = None,
                 return_x: bool = False):
    """Produce a certificate by solving the Feldman LP -- scipy, sparse.

    The producer half: consults whatever it likes, is trusted for nothing.
    Enumerates each check's wrong-parity subsets (2^(deg-1) of them; refuses
    degrees beyond `max_check_degree` rather than exploding), solves the
    primal LP, and reads the facet/box duals off scipy's marginals.
    Returns (certificate, bound) with only the support retained.

    With return_primal=True, returns (certificate, bound, primal_support) where
    primal_support is the 0/1 support of the LP primal IF it is integral, else
    None. THE REPAIR PATH: an integral primal is itself a minimum-weight
    correction, certified by the very dual returned beside it -- found live
    when the certificate flagged BP-OSD returning weight 22 against a tight
    bound of 6 on the [[144,12,12]] code, the difference a logical operator.
    A refused decoder answer plus an integral LP is not just a detected fault;
    it is a repaired one.

    The three stages -- facet construction, RPC cutting, dual extraction --
    are separate functions. Nothing here is trusted by the checker, but a
    producer nobody can read is a producer whose YIELD regressions nobody
    can diagnose, and yield is what a producer is for.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    from scipy.sparse import vstack as spvstack

    # Annotated because the arity genuinely varies with the flags: mypy
    # otherwise pins the type to whichever branch it sees first and then
    # calls the `return_x` extension an incompatible assignment.
    _nores: tuple = (None, 0.0, None) if return_primal else (None, 0.0)
    if return_x:
        _nores = _nores + (None,)
    supports = [tuple(sorted(int(i) for i in c)) for c in checks]
    syn = [int(b) & 1 for b in syndrome]
    var_ids = sorted(weights)
    vidx = {v: k for k, v in enumerate(var_ids)}
    n = len(var_ids)
    wvec = np.array([float(weights[v]) for v in var_ids])

    built = _facet_blocks(supports, syn, var_ids, vidx, n, max_check_degree,
                          max_facet_size)
    if built is None:
        return _nores
    blocks, rhs, meta = built
    if not rhs:
        empty = QldpcCertificate(error_support=(), facet_duals=())
        base = (empty, 0.0, ()) if return_primal else (empty, 0.0)
        return base + ({},) if return_x else base
    A_base = spvstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    base_nrows = len(rhs)
    ex_r: list = []                     # RPC extras, row-indexed from 0
    ex_c: list = []
    ex_v: list = []

    def _solve():
        if ex_r:
            E = csr_matrix((ex_v, (ex_r, ex_c)),
                           shape=(len(rhs) - base_nrows, n))
            A = spvstack([A_base, E], format="csr")
        else:
            A = A_base
        return linprog(wvec, A_ub=A, b_ub=np.array(rhs), bounds=(0.0, 1.0),
                       method="highs")

    res = _solve()
    if not res.success:
        return _nores

    res = _rpc_cutting_loop(res, _solve, rpc_rounds, supports, syn, vidx,
                            var_ids, n, rhs, meta, (ex_r, ex_c, ex_v),
                            base_nrows, rpc_cap)

    facets, boxes = _extract_dual(res, meta, rhs, var_ids, n)
    cert = QldpcCertificate(error_support=(), facet_duals=facets,
                            box_duals=boxes)
    if not return_primal:
        base2 = (cert, float(res.fun))
        if return_x:
            xv = np.asarray(res.x)
            return base2 + ({int(var_ids[k]): float(xv[k])
                             for k in range(n)},)
        return base2
    x = np.asarray(res.x)
    if np.max(np.abs(x - np.round(x))) < 1e-7:
        # var_ids[k], NOT k: the internal soundness audit found the repair
        # support built from LP column indices -- harmless while every
        # caller uses contiguous 0..n-1 ids, wrong the day one does not
        # (the lazy path always did this correctly)
        primal = tuple(int(var_ids[k])
                       for k in np.flatnonzero(np.round(x) > 0.5))
    else:
        primal = None                   # fractional: a pseudocodeword, no repair
    cert = _polish_if_short(checks, weights, syndrome, cert, primal)
    if return_x:
        return cert, float(res.fun), primal, {int(var_ids[k]): float(x[k])
                                              for k in range(n)}
    return cert, float(res.fun), primal


def _name_infeasible_rows(red_checks, red_syn):
    """Name a GF(2) row combination that XORs to empty support, odd parity.

    Called when the LP says infeasible: the reduced system has no 0/1
    completion, and the certificate must say WHICH combination of rows
    proves it. Elimination with tags finds one; the checker re-derives the
    combination from (H, s) and trusts the naming for nothing.

    A PIVOT ROW SERVES ONCE. Reusing a pivot was a latent producer bug --
    it can eliminate a row back into a combination already consumed, and
    the tags then name a set whose XOR is not what the tags claim.

    Returns an ("infeasible", rows) node, or None if the LP failed for
    some other reason.
    """
    rowsets = [set(sup) for sup in red_checks]
    tags = [1 << j for j in range(len(rowsets))]
    pars = list(red_syn)
    used: set = set()           # a pivot row serves ONCE -- reusing
    for v in sorted(set(i for rs in rowsets for i in rs)):
        piv = next((k for k in range(len(rowsets))
                    if k not in used and v in rowsets[k]), None)
        if piv is None:
            continue
        used.add(piv)
        for k in range(len(rowsets)):
            if k != piv and v in rowsets[k]:
                rowsets[k] ^= rowsets[piv]
                pars[k] ^= pars[piv]
                tags[k] ^= tags[piv]
    for rs, pr, tg in zip(rowsets, pars, tags):
        if not rs and pr:
            rows = tuple(r for r in range(len(red_checks))
                         if (tg >> r) & 1)
            return ("infeasible", rows if len(rows) > 1 else rows[0])
    return None                  # LP failed for another reason


def _remap_facets_to_full(facet_duals, kept_idx):
    """Rewrite facet check-indices from the kept sublist to the FULL list.

    The sub-LP is built only over checks with a non-empty reduced support,
    so its row indices are not the instance's. The checker re-reduces from
    the FULL instance, so the names it is handed must be full-instance
    names or it validates a different constraint than the one proved.
    """
    fd = []
    for (j, F, y) in facet_duals:
        if isinstance(j, (tuple, list, frozenset)):
            fd.append((tuple(kept_idx[r] for r in sorted(j)), tuple(F), y))
        else:
            fd.append((kept_idx[int(j)], tuple(F), y))
    return fd


def certified_bnb(checks, weights, syndrome, U, *, rpc_rounds: int = 6,
                  max_nodes: int = 4000, max_depth: int = 40,
                  use_lazy: bool = False):
    r"""Produce a BRANCH-DUAL tree proving OPT >= the lattice level of U.

    The residue closer: where the flat LP (even with cuts) leaves a
    fractional pseudocodeword, branching on a near-half coordinate
    destroys the fractional face, and each child is again a syndrome-LP
    whose exact safe-dual either crosses U - step or branches further.
    The returned tree is checked by check_qldpc_bnb in exact arithmetic
    with NO solver -- the producer may consult anything (it uses the LP
    fractional point to pick branch variables); nothing it says is
    trusted. Returns (tree, nodes_used) or (None, nodes_used) if the
    node/depth budget ran out (an honest refusal, never a wrong tree).

    U is the candidate's weight in the same units as `weights` (use
    integer weights so the lattice step is 1).
    """
    from fractions import Fraction as _F

    from .qldpc_check import _exact_safe_dual_bound, _reduce_instance

    wx = {int(i): _F(int(v)) if float(v).is_integer() else _F(str(v))
          for i, v in weights.items()}
    wf = {i: float(v) for i, v in wx.items()}
    D = 1
    for v in wx.values():
        D = D * v.denominator // __import__("math").gcd(D, v.denominator)
    step = _F(1, D)
    Ux = _F(int(U)) if float(U).is_integer() else _F(str(U))
    base_checks = [tuple(sorted(int(i) for i in c)) for c in checks]
    base_syn = [int(b) & 1 for b in syndrome]
    counter = [0]

    def node(S0: frozenset, S1: frozenset, depth: int):
        counter[0] += 1
        if counter[0] > max_nodes or depth > max_depth:
            return None
        red_checks, red_syn = _reduce_instance(base_checks, base_syn,
                                               S0, S1)
        for j, (sup, sj) in enumerate(zip(red_checks, red_syn)):
            if not sup and sj:
                return ("infeasible", j)
        live = sorted(set(i for sup in red_checks for i in sup))
        W1 = sum((wx[i] for i in S1), _F(0))
        if not live:
            # no free variables and every check satisfied: the unique
            # completion has weight exactly W1
            return ("bound", ()) if W1 > Ux - step else None
        keep = [(sup, sj) for sup, sj in zip(red_checks, red_syn) if sup]
        kc = [c for c, _ in keep]
        ks = [sj for _, sj in keep]
        kept_idx = [j for j, sup in enumerate(red_checks) if sup]
        wsub = {i: wf[i] for i in live}
        if use_lazy:
            out = feldman_dual_lazy(kc, wsub, ks, return_primal=True,
                                    return_x=True)
        else:
            out = feldman_dual(kc, wsub, ks, rpc_rounds=rpc_rounds,
                               return_primal=True, return_x=True)
        cert, _bound, _primal, xd = out
        if cert is None:
            return _name_infeasible_rows(red_checks, red_syn)
        fd = _remap_facets_to_full(exactify_certificate(cert).facet_duals,
                                   kept_idx)
        red_sup_f = [frozenset(c) for c in red_checks]
        L, err = _exact_safe_dual_bound(red_sup_f, red_syn, wx, tuple(fd))
        if err is None and L is not None and L + W1 > Ux - step:
            return ("bound", tuple(fd))
        # branch on the most ambivalent coordinate of the LP point --
        # the fractional face dies on both children. Producer heuristic
        # only; the checker re-derives everything.
        if xd:
            pick = min(live, key=lambda i: abs(xd.get(i, 0.0) - 0.5))
        else:
            pick = min(live, key=lambda i: wf[i])
        c0 = node(S0 | {pick}, S1, depth + 1)
        if c0 is None:
            return None
        c1 = node(S0, S1 | {pick}, depth + 1)
        if c1 is None:
            return None
        return ("branch", pick, c0, c1)

    tree = node(frozenset(), frozenset(), 0)
    return tree, counter[0]


def frame_determinacy_trees(checks, weights, syndrome, U, logical_rows,
                            frame, *, rpc_rounds: int = 6,
                            max_nodes: int = 20000, use_lazy: bool = False):
    r"""FRAME-CERTIFIED: prove every weight-<=U solution has frame `frame`.

    A logical operator's support is one more parity row, so "no solution
    of weight <= U flips logical bit ell" is the statement "the system
    {He = s, (Ge)_ell = frame_ell XOR 1} has minimum weight > U" -- and
    that is exactly a bound-only branch-dual tree on the augmented
    instance (checker: check_qldpc_bnb(bound_only_threshold=U)). One
    tree per logical bit covers every differing frame, since a differing
    frame differs somewhere.

    Returns (trees, all_proven): trees is a list of
    (ell, tree_or_None); a None tree means the producer could not close
    that bit within budget -- the receipt for it is FRAME-UNCHECKED,
    never a claim. Soundness lives entirely in the checker.
    """
    trees = []
    all_ok = True
    for ell, row in enumerate(logical_rows):
        aug_checks = list(checks) + [tuple(sorted(int(i) for i in row))]
        aug_syn = list(syndrome) + [int(frame[ell]) ^ 1]
        # a logical row's support routinely exceeds the enumerating
        # path's degree cap (the demo's first run had EXACTLY the
        # over-cap bits come back unchecked -- the gate, not the
        # physics); the lazy separator handles any degree
        lazy_here = use_lazy or any(len(c) > 12 for c in aug_checks)
        # certified_bnb proves min >= lattice level of its U argument;
        # with integer weights, U_arg = U + 1 makes leaves cross U
        tree, _n = certified_bnb(aug_checks, weights, aug_syn,
                                 int(U) + 1, rpc_rounds=rpc_rounds,
                                 max_nodes=max_nodes, use_lazy=lazy_here)
        trees.append((ell, tree))
        if tree is None:
            all_ok = False
    return trees, all_ok
