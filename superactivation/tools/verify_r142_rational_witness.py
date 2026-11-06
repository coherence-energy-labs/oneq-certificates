#!/usr/bin/env python3
"""Exact rational, independently recheckable lower bounds for R142 two-use codes.

The witness consists ONLY of integer SU(2) quaternions and integer reference
state vectors. The encoder is the exact dyadic-rational matrix represented by
the printed binary64 code_columns. Its first Kraus operator is V/sqrt(lambda),
where a rational Gershgorin bound certifies V^dag V <= lambda I. Completing
the encoding CP map to trace preservation can only add nonnegative fidelity.

On the no-erasure branch Bob uses V^dag/sqrt(lambda), also completed to CPTP.
On one-erasure branches Bob applies the witness's normalized SU(2) rotations.
On both-erased branches Bob prepares normalized conjugate-reference states.
Integer sqrt bounds p=1/(1+sqrt(d)) by rationals. All fidelity lower-bound
arithmetic uses fractions.Fraction, NOT floating point, numpy or producer code.

What is proved: an explicit physically possible effective flagged channel has
fidelity strictly greater than 3/4 for the checked d=20,21,22 witnesses.
What is NOT proved here: the PPT family, its physical reduction to this channel,
its stand-alone upper bounds, hardware realization, or prior-art novelty.
"""
from __future__ import annotations

from fractions import Fraction as F
from math import isqrt
import hashlib
import json
import pathlib
import sys


def C(x, y=0):
    return (F(x), F(y))


def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def sub(x, y):
    return (x[0] - y[0], x[1] - y[1])


def times(x, y):
    return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def conj(x):
    return (x[0], -x[1])


def abs2(x):
    return x[0] * x[0] + x[1] * x[1]


def scale(a, x):
    return (a * x[0], a * x[1])


def complex_sum(items):
    total = C(0)
    for item in items:
        total = add(total, item)
    return total


class ProofRefused(ValueError):
    pass


def prove(cert: dict, witness: dict) -> dict:
    d = cert.get("d")
    if type(d) is not int or d < 2:
        raise ProofRefused("invalid shield dimension")
    if witness.get("schema") != "oneq.r142.rational-recovery/1" or witness.get("d") != d:
        raise ProofRefused("wrong witness schema or shield dimension")
    if type(witness.get("grid_scale")) is not int or witness["grid_scale"] < 1:
        raise ProofRefused("missing rational witness grid")
    digest = hashlib.sha256(json.dumps(
        cert.get("code_columns"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")).hexdigest()
    if witness.get("encoder_sha256") != digest:
        raise ProofRefused("witness encoder SHA256 does not match printed encoder")

    try:
        V = [[(F.from_float(float(pair[0])), F.from_float(float(pair[1])))
              for pair in row] for row in cert["code_columns"]]
        first, second, states = (witness["first_flagged"],
                                 witness["second_flagged"], witness["both_flagged"])
    except (TypeError, ValueError, IndexError, KeyError, OverflowError) as exc:
        raise ProofRefused("malformed encoder or rational witnesses") from exc
    if len(V) != 4 or any(len(row) != 2 for row in V):
        raise ProofRefused("encoder must be 4 x 2 complex matrix")
    for choices in (first, second):
        if (len(choices) != 2 or
                any(len(q) != 4 or any(type(x) is not int for x in q)
                    or not any(q) for q in choices)):
            raise ProofRefused("two nonzero integer SU(2) quaternion decoders required")
    if (len(states) != 4 or
            any(len(w) != 2 or any(len(z) != 2 or
                                   any(type(x) is not int for x in z) for z in w)
                or not any(x for z in w for x in z) for w in states)):
        raise ProofRefused("four nonzero rational reference-state decoders required")

    # lambda >= largest eigenvalue(V^dag V), by Gershgorin plus
    # |complex z| <= |Re z| + |Im z|. Exact dyadic-rational arithmetic.
    t00 = sum(abs2(V[i][0]) for i in range(4))
    t11 = sum(abs2(V[i][1]) for i in range(4))
    cross = complex_sum(times(conj(V[i][0]), V[i][1]) for i in range(4))
    off = abs(cross[0]) + abs(cross[1])
    lam = max(F(1), t00 + off, t11 + off)
    if lam <= 0:
        raise ProofRefused("invalid positivity normalization")

    # Let s_floor=floor(2^128 sqrt(d)), obtained using INTEGER sqrt.
    # sqrt(d) is between s_floor/N and (s_floor+1)/N.
    # All actual p or (1-p) branch probabilities are >= these lower bounds.
    N = 1 << 128
    s_floor = isqrt(d * N * N)
    p_lower = F(N, N + s_floor + 1)
    p_upper = F(N, N + s_floor)
    q_lower = F(1) - p_upper
    if not (0 < p_lower <= p_upper < 1 and 0 < q_lower < 1):
        raise ProofRefused("bad irrational flip-noise enclosure")

    # Both kept: primary encoder V/sqrt(lam), decoder V^dag/sqrt(lam).
    # Both CP subchannels have CPTP completions with positive Choi operators.
    f_kept = ((t00 + t11) / lam) ** 2 / 4

    def one_flagged(flagged: int, choices: list) -> F:
        subtotal = F(0)
        for observed in (0, 1):
            quaternion = choices[observed]
            norm = sum(q * q for q in quaternion)
            for actual in (0, 1):
                if flagged == 0:
                    M = [[V[2*actual+j][r] for r in range(2)] for j in range(2)]
                else:
                    M = [[V[2*j+actual][r] for r in range(2)] for j in range(2)]
                # Tr(U M) where U = (q0 I + i(q1 X+q2 Y+q3 Z))/sqrt(norm).
                tI = add(M[0][0], M[1][1])
                tX = times(C(0, 1), add(M[0][1], M[1][0]))
                tY = sub(M[1][0], M[0][1])
                tZ = times(C(0, 1), sub(M[0][0], M[1][1]))
                amplitude = complex_sum(scale(F(q), t)
                                        for q, t in zip(quaternion, (tI, tX, tY, tZ)))
                probability_lower = q_lower if observed == actual else p_lower
                subtotal += probability_lower * abs2(amplitude) / (4 * norm * lam)
        return subtotal

    f_first = one_flagged(0, first)
    f_second = one_flagged(1, second)

    # Both flagged: each observed (c0,c1) leads to an explicitly prepared
    # normalized pure state. Its reference conjugate is an integer complex
    # vector w; <w|S_c|w> is a nonnegative sum of rational |<w|v_m>|^2.
    f_both = F(0)
    for c0 in (0, 1):
        for c1 in (0, 1):
            w = [C(*z) for z in states[2*c0+c1]]
            norm = sum(abs2(z) for z in w)
            for m0 in (0, 1):
                for m1 in (0, 1):
                    v = V[2*m0+m1]
                    inner = add(times(conj(w[0]), v[0]), times(conj(w[1]), v[1]))
                    prob0 = q_lower if c0 == m0 else p_lower
                    prob1 = q_lower if c1 == m1 else p_lower
                    f_both += prob0*prob1*abs2(inner)/(4*norm*lam)

    lower = (f_kept + f_first + f_second + f_both) / 4
    # This is a strict exact-rational integer comparison, NOT double precision.
    positive = lower > F(3, 4)
    printed = lower.numerator * 10**15 // lower.denominator
    return {
        "dimension": d,
        "lower_bound_floor_15_decimals": f"{printed // 10**15}.{printed % 10**15:015d}",
        "crosses_three_quarters_PROVEN_on_effective_channel": positive,
        "strict_margin_sign": 1 if positive else (-1 if lower < F(3, 4) else 0),
        "p_interval_width_exact": str(p_upper - p_lower),
        "encoder_normalizer_upper_for_display_only": float(lam),
        "physical_decoder": "integer normalized SU2 and state-preparation CPTP completions",
        "proof_method": "exact fractions + integer sqrt enclosure + Gershgorin + positive CP completions",
        "scope": "conditional effective flagged channel; PPT and physical reduction not re-proved here",
    }


def main(args):
    if len(args) != 2:
        raise SystemExit("usage: python tools/verify_r142_rational_witness.py CERT.json WITNESS.json")
    cert = json.loads(pathlib.Path(args[0]).read_text(encoding="utf-8"))
    witness = json.loads(pathlib.Path(args[1]).read_text(encoding="utf-8"))
    print(json.dumps(prove(cert, witness), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
