r"""The forgery battery: certificates whose CLAIM is false.

Every scale run attacks the checker with these on every shot. They live here,
in ONE place, because the alternative is a copy per experiment -- and a battery
weakened in one copy still reports "0 forgeries accepted" from the other, which
reads exactly like a pass.

WHAT A CERTIFICATE CLAIMS, precisely: "matched_edges is a MINIMUM-WEIGHT edge
set whose odd-degree vertices are exactly the syndrome." A forgery is a
certificate for which that sentence is FALSE. Anything else is not an attack,
and counting it as one is how you get a fake alarm.

THIS MODULE LEARNED THAT THE HARD WAY, THREE TIMES.

  First, an attack yielded "send every detector to the boundary" without
  comparing weights. Sixteen were accepted and looked like program-ending
  false positives. They were equal-weight ALTERNATIVE OPTIMA -- minimum-weight
  corrections are not unique, and accepting a different optimum is correct.

  Then a "hidden negative dual" attack was accepted on 4 of 147 shots. An
  independent audit -- recomputing every packing constraint outside the
  checker's own code path -- found zero negative duals, zero violated
  constraints, the right odd-degree set, and an objective equal to the primal
  to within 9e-16. The construction had DEGENERATED into a valid certificate
  for a correction that really was optimal, so the checker was right again.

  Third (2026-11-03, adjudicating external review 1): D, G and H built their
  "strictly heavier" correction as `correction + (extra, extra)` -- a list
  naming one edge TWICE. As an edge set (mod 2) that is the genuine correction,
  so the claim "strictly heavier" held only in the multiset reading, and the
  falsity audit below confirmed it by summing the weight twice. The
  epsilon-checker refuses a repeated edge as malformed input, so all three were
  refused before the bound was ever computed: on the 10^6-shot simulation,
  8,000,000 of 8,000,001 forgeries were refused by input validation and one (A)
  by weak duality. They now add an even CYCLE edge-disjoint from the
  correction, a genuinely heavier edge set with the same odd-degree set; the
  doubled-edge constructions are kept, under the M_ names and the
  MALFORMED_INPUT tag, as what they always were.

The first two mistakes have one shape (an attack whose claim happened to be TRUE),
the third another (an attack whose object was not the object the claim is about).
So every attack carries `why_false`, naming the property that makes its claim
false, and `assert_all_false` re-establishes that property independently, in exact
arithmetic, before a "0 accepted" result may be trusted. A battery nobody audits
is a gate that cannot fail.

THE ATTACKS

    A  cheapest-boundary correction, kept only when STRICTLY heavier (by > eps_max)
    D  spurious even cycle added: same odd set, strictly more weight     (stealth)
    E  a dropped edge                                                    (odd set breaks)
    F  a duplicated edge                                                 (odd set breaks)
    G  D's correction + the genuine dual inflated to meet ITS weight     (stealth; reaches theta)
    H  the same, with a negative dual paying for the inflation           (stealth; the z >= 0 guard)
    I  a genuine dual paired with a different correction
    M_ the pre-2026-11-03 D, G, H: a doubled edge (MALFORMED_INPUT)

D and G are refused only by weak duality: their correction is a valid,
syndrome-consistent edge set and their packing is over genuine T'-odd sets, so the
bound -- after theta rescales G's overloaded packing -- falls short of the heavier
weight. H is refused by the non-negativity guard. `parity_forgery` is the
separate linear-programming adversary over sets that are NOT T'-odd, the class of
the only real checker defect found so far (R177).
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction

from .matching_cert import BOUNDARY, MatchingCertificate

# The claim is false because the correction does not explain the syndrome.
NOT_A_CORRECTION = "odd-degree set differs from the syndrome"
# The claim is false because a strictly cheaper correction is known to exist.
STRICTLY_SUBOPTIMAL = "a strictly cheaper syndrome-consistent correction is known"
# Not a claim about an edge set at all: the correction lists an edge more than once.
MALFORMED_INPUT = "the correction lists an edge twice, so it is not an edge set"

#: An attack is "strictly heavier" only by MORE than this, so it is false as an epsilon-claim too.
EXCESS_MARGIN = Fraction(1, 10**6)


def _key(e) -> tuple[int, int]:
    u, v = int(e[0]), int(e[1])
    if u == BOUNDARY:
        return (v, u)
    return (u, v) if v == BOUNDARY or u <= v else (v, u)


def _odd(correction) -> set[int]:
    deg: dict[int, int] = {}
    for u, v in correction:
        for x in (int(u), int(v)):
            if x != BOUNDARY:
                deg[x] = deg.get(x, 0) + 1
    return {k for k, c in deg.items() if c % 2 == 1}


def _exact_weight(W, corr) -> Fraction:
    return sum((Fraction(W[_key(e)]) for e in corr), Fraction(0))


def _adjacency(W):
    adj: dict[int, list] = {}
    for k in W:
        u, v = k
        adj.setdefault(u, []).append((v, k))
        adj.setdefault(v, []).append((u, k))
    return adj


def spurious_cycle(W, correction, *, max_hops: int = 8):
    """An even cycle edge-disjoint from `correction`, as a sorted edge list, or None.

    Adding it changes no vertex's parity, so the union is a syndrome-consistent edge set heavier by w(C). Edges
    are tried lightest first (ties by key) and the first that closes a cycle of at most `max_hops` edges in the
    graph minus the correction wins: deterministic, and usually the first edge tried.
    """
    used = {_key(e) for e in correction}
    adj = _adjacency(W)
    for (u, v) in sorted((k for k in W if k not in used), key=lambda k: (W[k], k)):
        prev = {u: None}
        q = deque([(u, 0)])
        found = False
        while q and not found:
            x, h = q.popleft()
            if h >= max_hops - 1:
                continue
            for y, k in adj.get(x, ()):
                if k in used or k == (u, v) or y in prev:
                    continue
                prev[y] = (x, k)
                if y == v:
                    found = True
                    break
                q.append((y, h + 1))
        if found:
            path, y = [], v
            while prev[y] is not None:
                x, k = prev[y]
                path.append(k)
                y = x
            return sorted(path + [(u, v)])
    return None


def forgeries(W, syndrome, correction, z, primal):
    """[(name, certificate, why_false)] -- every claim here is FALSE.

    `correction` is the decoder's own answer, already canonicalised, and
    `primal` is its weight: that is the known-feasible reference every
    "strictly suboptimal" attack is measured against. `z` is the shot's
    genuine dual, so the attacks cost checker calls and nothing else.
    """
    T = set(int(s) for s in syndrome)
    out: list[tuple[str, MatchingCertificate, str]] = []
    mset = set(correction)
    primal_x = Fraction(primal)

    def add(name, corr, dual, why):
        out.append((name, MatchingCertificate(matched_edges=tuple(corr),
                                              node_potentials={},
                                              blossom_duals=dual), why))

    # ---- corrections that do not explain the syndrome --------------------
    if len(correction) > 1:
        c = correction[:-1]
        if _odd(c) != T:
            add("E_dropped_edge", c, z, NOT_A_CORRECTION)
    if correction:
        c = tuple(correction) + (correction[0],)
        if _odd(c) != T:
            add("F_duplicated_edge", c, z, NOT_A_CORRECTION)
    other = tuple(k for k in W if k not in mset)[:max(len(correction), 1)]
    if other and _odd(other) != T:
        add("I_mismatched_pair", other, z, NOT_A_CORRECTION)

    # ---- corrections that explain it but are strictly worse --------------
    boundary_only = tuple((f, BOUNDARY) for f in syndrome if (f, BOUNDARY) in W)
    if (boundary_only and len(boundary_only) == len(syndrome)
            and _exact_weight(W, boundary_only) > primal_x + EXCESS_MARGIN):
        add("A_suboptimal", boundary_only, z, STRICTLY_SUBOPTIMAL)

    # D: the correction plus an even cycle disjoint from it. Same odd-degree set (a cycle changes no parity), a
    # valid edge set, and heavier by w(C) -- so only weak duality can refuse it.
    heavy = None
    cyc = spurious_cycle(W, correction)
    if cyc is not None:
        cand = tuple(sorted(set(correction) | set(cyc)))
        if _odd(cand) == T and _exact_weight(W, cand) > primal_x + EXCESS_MARGIN:
            heavy = cand
            add("D_spurious_cycle", cand, z, STRICTLY_SUBOPTIMAL)

    # G and H certify that same strictly-heavier correction, but repair the
    # arithmetic so the objective MEETS its weight -- which is what a
    # certificate is supposed to demonstrate.
    if heavy is not None and z:
        total = sum(z.values())
        heavy_w = sum(W[k] for k in heavy)
        if total > 1e-9:
            add("G_dual_inflated_to_meet_a_worse_primal", heavy,
                {S: v * (heavy_w / total) for S, v in z.items()},
                STRICTLY_SUBOPTIMAL)
            zz = _negative_hides_inflation(z, heavy_w)
            if zz is not None:
                add("H_negative_dual_hides_inflation", heavy, zz, STRICTLY_SUBOPTIMAL)

    # M: the pre-2026-11-03 D, G and H, kept as what they are -- documents that are not about an edge set.
    extra = next((k for k in W if k not in mset and BOUNDARY not in k), None)
    if extra is not None:
        doubled = tuple(correction) + (extra, extra)
        add("M_doubled_edge", doubled, z, MALFORMED_INPUT)
        if z:
            total = sum(z.values())
            dw = sum(W[k] for k in doubled)
            if total > 1e-9:
                add("M_doubled_edge_dual_inflated", doubled,
                    {S: v * (dw / total) for S, v in z.items()}, MALFORMED_INPUT)
                zz = _negative_hides_inflation(z, dw)
                if zz is not None:
                    add("M_doubled_edge_negative_dual", doubled, zz, MALFORMED_INPUT)
    return out


def _negative_hides_inflation(z, target):
    """z scaled past `target` with the excess paid by a negative value on the union set, or None."""
    total = sum(z.values())
    slack = 0.5 * total
    scale = (target + slack) / total
    zz = {S: v * scale for S, v in z.items()}
    wide = frozenset().union(*z.keys())
    zz[wide] = zz.get(wide, 0.0) - slack
    # ONLY AN ATTACK IF A NEGATIVE ACTUALLY SURVIVES. When z has a single set, `wide` IS that set and the
    # subtraction can land on a still-positive value -- which is how an earlier version degenerated into a valid
    # certificate and produced a fake alarm.
    return zz if min(zz.values()) < -1e-12 else None


def assert_all_false(W, syndrome, primal, battery):
    """Independently confirm every attack's claim really is false.

    Returns the list of attacks that do NOT falsify -- which must be empty. A
    non-empty result means the battery is scoring valid certificates as
    forgeries, and a "0 accepted" headline built on it is measuring nothing.
    Weights are compared in exact rationals; "strictly heavier" means by more
    than EXCESS_MARGIN, so the claim is false for an epsilon-checker too.
    """
    T = set(int(s) for s in syndrome)
    bogus = []
    for name, cert, why in battery:
        corr = cert.matched_edges
        keys = [_key(e) for e in corr]
        if why == NOT_A_CORRECTION:
            if _odd(corr) == T:
                bogus.append((name, "claims a bad odd set but the odd set matches"))
        elif why == STRICTLY_SUBOPTIMAL:
            if len(set(keys)) != len(keys):
                bogus.append((name, "lists an edge twice, so it is not an edge set and is not a heavier "
                                    "correction (mod 2 it may be the genuine one)"))
            elif any(k not in W for k in keys):
                bogus.append((name, "uses an edge the graph does not have"))
            elif _odd(corr) != T:
                bogus.append((name, "not syndrome-consistent, so it tests the wrong rule"))
            elif _exact_weight(W, corr) <= Fraction(primal) + EXCESS_MARGIN:
                bogus.append((name, "not actually heavier than a known correction"))
        elif why == MALFORMED_INPUT:
            if len(set(keys)) == len(keys):
                bogus.append((name, "tagged malformed but lists every edge once"))
        else:
            bogus.append((name, f"unknown falsity tag {why!r}"))
    return bogus


# ---------------------------------------------------------------- the T'-oddness adversary

def _blind_bound(W, duals) -> Fraction:
    """theta * sum(z) computed EXACTLY as the Lemma says, but WITHOUT the T'-oddness test: what a checker that forgot
    the parity rule would accept as a lower bound."""
    adj = _adjacency(W)
    load: dict = {}
    total = Fraction(0)
    for S, val in duals.items():
        zf = Fraction(val)
        if zf <= 0:
            continue
        total += zf
        members = set(S)
        for x in members:
            for _y, k in adj.get(x, ()):
                if (k[0] in members) != (k[1] in members):
                    load[k] = load.get(k, Fraction(0)) + zf
    theta = Fraction(1)
    for k, L in load.items():
        if L > Fraction(W[k]):
            theta = min(theta, Fraction(W[k]) / L)
    return theta * total


def parity_forgery(W, syndrome, correction, duals):
    """(certificate, w(heavier), parity-blind bound) for a heavier correction "proven" by sets that are NOT T'-odd.

    The linear-programming search that found the R177 defect, made to run on a real shot: take the heavier valid
    correction J' = J + C (C an even cycle disjoint from J), and maximise sum(z) over the genuine sets PLUS sets that
    a checker must refuse -- singletons of vertices outside T' near J' (|S n T'| = 0) and, when |T| is odd, the
    boundary-side traps {B, t} (odd against T, even against T' = T + {B}: the rule the strict checker got wrong) --
    subject to every edge load <= w_e. A checker that skipped the parity test would compute the third return value
    as its bound; when it reaches w(J') - eps the forgery is real, and only the T'-oddness test can refuse it.
    Returns None when no disjoint cycle exists.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    T = {int(s) for s in syndrome}
    Tp = T | ({BOUNDARY} if len(T) % 2 else set())
    cyc = spurious_cycle(W, correction)
    if cyc is None:
        return None
    heavy = tuple(sorted(set(_key(e) for e in correction) | set(cyc)))
    heavy_w = _exact_weight(W, heavy)
    adj = _adjacency(W)
    near = set()
    for (u, v) in heavy:
        for x in (u, v):
            near.add(x)
            near.update(y for y, _k in adj.get(x, ()))
    family = [frozenset(S) for S in duals]
    family += [frozenset({x}) for x in sorted(near) if x not in Tp]
    if len(T) % 2:
        family += [frozenset({BOUNDARY, t}) for t in sorted(T)]
    family = list(dict.fromkeys(family))
    rows, cols, edge_index = [], [], {}
    for j, S in enumerate(family):
        for x in S:
            for _y, k in adj.get(x, ()):
                if (k[0] in S) != (k[1] in S):
                    i = edge_index.setdefault(k, len(edge_index))
                    rows.append(i)
                    cols.append(j)
    keys = list(edge_index)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(family)))
    res = linprog(-np.ones(len(family)), A_ub=A, b_ub=np.array([W[k] for k in keys]),
                  bounds=[(0, None)] * len(family), method="highs")
    if res.status != 0:
        return None
    z = {S: float(x) for S, x in zip(family, res.x) if x > 1e-12}
    cert = MatchingCertificate(matched_edges=heavy, node_potentials={}, blossom_duals=z)
    return cert, heavy_w, _blind_bound(W, z)
