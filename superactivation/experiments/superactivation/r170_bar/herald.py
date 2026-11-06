"""R170 item 4: may heralded device failures be counted as extra erasures? The two checkable facts the theorem rests on.

The theorem (DERIVED; stated in full in the R170 ledger entry):
  Counting a heralded failure as an erasure of A's output is a DECODER action, so the channels are unchanged, PROVIDED
  (H1) the herald is read in Bob's laboratory from channel OUTPUTS (P's or A's) or Bob's decoder, or it is a classical
  record sent from Alice's laboratory (then it is a classical channel C, and A (x) C is still 2-extendible, so Q = 0 and
  every number of uses stays <= 3/4); and (H2) the herald carries no information about P's internal randomness (its
  sector bit or its twist/shield choice). Discarding heralded shots (post-selection) is never allowed.
  SCOPE (the R170 adversary pass): this holds against the 3/4 bar only. At an intermediate rung a classical side
  channel C changes the comparison -- two uses of the erasure channel plus ONE classical bit reach (7+sqrt2)/12 =
  0.7012 > 11/16 -- so "the certificate is unchanged" must not be read off at 11/16.
Two facts make (H2) and the post-selection ban NECESSARY, and they are what this module computes:
  * P_d is PPT only on the line p0/p1 = sqrt d (R142, PROVEN: the partial transpose has the closed-form eigenvalues
    p0/2d^2 +- p1/2d^{3/2} and p1/2d +- p0/2d^{3/2}). A herald correlated with the sector bit splits P into branches
    whose sector ratios straddle sqrt d, so one branch is NPT and the PPT certificate is gone -- for ANY correlation.
  * Post-selecting on a herald correlated with a channel's own randomness can beat the component bar outright: drop the
    shots in which A's (input-independent!) erasure flag fired and one use of the 50% erasure channel gives F = 1.
"""
from __future__ import annotations

import math


def pt_eigs(d: int, p1: float):
    """the partial-transpose spectrum of P_d's (normalised) Choi state at sector-1 probability p1 (R142's closed form):
    sector 0 gives p0/2d^2 +- p1/2d^{3/2}, sector 1 gives p1/2d +- p0/2d^{3/2} (each with its multiplicity)"""
    p0 = 1 - p1
    return (p0 / (2 * d * d) + p1 / (2 * d ** 1.5), p0 / (2 * d * d) - p1 / (2 * d ** 1.5),
            p1 / (2 * d) + p0 / (2 * d ** 1.5), p1 / (2 * d) - p0 / (2 * d ** 1.5))


def p1_ppt(d: int) -> float:
    return 1 / (math.sqrt(d) + 1)


def min_pt_eig(d: int, p1: float) -> float:
    return min(pt_eigs(d, p1))


def heralded_branches(d: int, delta: float):
    """a herald h in {0,1}, each with probability 1/2, that shifts the sector-1 probability to p1 (1 +- delta) in its two
    branches (the average stays at the PPT point). Returns the minimum partial-transpose eigenvalue of each branch."""
    p1 = p1_ppt(d)
    return min_pt_eig(d, p1 * (1 + delta)), min_pt_eig(d, p1 * (1 - delta))


def postselected_erasure_fidelity(m: int = 1) -> float:
    """one use of the 50% erasure channel, keeping only the shots whose erasure flag did NOT fire: the identity channel"""
    return 1.0 if m >= 1 else 0.25


def erasure_unassisted_one_use() -> float:
    """the same channel with every shot kept: kept branch 1, erased branch 1/4 (5/8 = F_c(A, 2), see erasure_codes)"""
    return 0.5 * 1.0 + 0.5 * 0.25
