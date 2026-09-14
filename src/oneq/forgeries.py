r"""The forgery battery: certificates whose CLAIM is false.

Every scale run attacks the checker with these on every shot. They live here,
in ONE place, because the alternative is a copy per experiment -- and a battery
weakened in one copy still reports "0 forgeries accepted" from the other, which
reads exactly like a pass.

WHAT A CERTIFICATE CLAIMS, precisely: "matched_edges is a MINIMUM-WEIGHT edge
set whose odd-degree vertices are exactly the syndrome." A forgery is a
certificate for which that sentence is FALSE. Anything else is not an attack,
and counting it as one is how you get a fake alarm.

THIS MODULE LEARNED THAT THE HARD WAY, TWICE.

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

Both mistakes have the same shape: an attack whose claim happened to be TRUE.
So every attack here now carries `why_false`, naming the property that makes
its claim false, and the certified correction is compared against a
known-feasible one rather than assumed to be worse. `assert_all_false` lets a
caller verify the battery independently before trusting a "0 accepted" result.
A battery nobody audits is a gate that cannot fail.

THE ATTACKS

    A  cheapest-boundary correction, kept only when STRICTLY heavier
    D  spurious closed cycle: same odd set, strictly more weight   (stealth)
    E  a dropped edge                                              (odd set breaks)
    F  a duplicated edge                                           (odd set breaks)
    G  suboptimal correction + a dual inflated to meet ITS weight  (stealth)
    H  the same, with a negative dual paying for the inflation     (stealth)
    I  a genuine dual paired with a different correction

D, G and H are the ones worth staring at. All three leave the odd-degree set
EXACTLY as the syndrome demands, so syndrome-consistency -- the only check any
production decoder performs today -- passes them without complaint. G and H go
further and repair the arithmetic the naive way: G scales the real duals until
the objective meets the inflated primal, H uses a negative variable to pay for
constraints the scaling overran. Between them and an accepted forgery stand
exactly two rules: z_S >= 0 enforced as a membership condition rather than
inferred from sums, and every packing constraint checked rather than sampled.
"""

from __future__ import annotations

from .matching_cert import BOUNDARY, MatchingCertificate

# The claim is false because the correction does not explain the syndrome.
NOT_A_CORRECTION = "odd-degree set differs from the syndrome"
# The claim is false because a strictly cheaper correction is known to exist.
STRICTLY_SUBOPTIMAL = "a strictly cheaper syndrome-consistent correction is known"


def _odd(correction) -> set[int]:
    deg: dict[int, int] = {}
    for u, v in correction:
        for x in (int(u), int(v)):
            if x != BOUNDARY:
                deg[x] = deg.get(x, 0) + 1
    return {k for k, c in deg.items() if c % 2 == 1}


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
            and sum(W[k] for k in boundary_only) > primal + 1e-9):
        add("A_suboptimal", boundary_only, z, STRICTLY_SUBOPTIMAL)

    # A doubled edge raises both endpoints' degree by 2, so the odd set is
    # untouched and the weight strictly rises: syndrome-consistent and wrong.
    heavy = None
    extra = next((k for k in W if k not in mset and BOUNDARY not in k), None)
    if extra is not None:
        cand = tuple(correction) + (extra, extra)
        if _odd(cand) == T and sum(W[k] for k in cand) > primal + 1e-9:
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
            slack = 0.5 * total
            scale = (heavy_w + slack) / total
            zz = {S: v * scale for S, v in z.items()}
            wide = frozenset().union(*z.keys())
            zz[wide] = zz.get(wide, 0.0) - slack
            # ONLY AN ATTACK IF A NEGATIVE ACTUALLY SURVIVES. When z has a
            # single set, `wide` IS that set and the subtraction can land on a
            # still-positive value -- which is how the earlier version
            # degenerated into a valid certificate and produced a fake alarm.
            if min(zz.values()) < -1e-12:
                add("H_negative_dual_hides_inflation", heavy, zz,
                    STRICTLY_SUBOPTIMAL)
    return out


def assert_all_false(W, syndrome, primal, battery):
    """Independently confirm every attack's claim really is false.

    Returns the list of attacks that do NOT falsify -- which must be empty. A
    non-empty result means the battery is scoring valid certificates as
    forgeries, and a "0 accepted" headline built on it is measuring nothing.
    """
    T = set(int(s) for s in syndrome)
    bogus = []
    for name, cert, why in battery:
        corr = cert.matched_edges
        if why == NOT_A_CORRECTION:
            if _odd(corr) == T:
                bogus.append((name, "claims a bad odd set but the odd set matches"))
        elif why == STRICTLY_SUBOPTIMAL:
            if _odd(corr) != T:
                bogus.append((name, "not syndrome-consistent, so it tests the "
                                    "wrong rule"))
            elif sum(W[k] for k in corr) <= primal + 1e-9:
                bogus.append((name, "not actually heavier than a known correction"))
        else:
            bogus.append((name, f"unknown falsity tag {why!r}"))
    return bogus
