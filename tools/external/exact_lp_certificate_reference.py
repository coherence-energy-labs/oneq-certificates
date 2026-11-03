#!/usr/bin/env python3
"""Exact reference checker for syndrome-LP lower-bound certificates.

This module is intentionally small and producer-independent. It uses Python's
arbitrary-precision integers and fractions.Fraction. It does not import an LP,
MIP, BP, OSD, NumPy, or SciPy implementation.

Certificate schema (JSON):
{
  "H": [[0,1,...], ...],
  "s": [0,1,...],
  "weights": [{"num": 1, "den": 1}, ...],
  "candidate": [0,1,...],
  "facets": [
    {
      "rows": [0, 3],
      "F": [2, 7],
      "y": {"num": 5, "den": 9}
    }
  ],
  "G": [[0,1,...], ...]  # optional logical-frame map
}

Each facet's parity row is the GF(2) XOR of H[rows], and its syndrome is the
XOR of s[rows]. F must be a subset of that row's support with parity different
from the reconstructed syndrome.

For listed facet inequalities a_r^T x >= b_r and multipliers y_r >= 0, the
checker derives z_i=max(0,(A^T y)_i-w_i) and the lower bound
L=b^T y-sum_i z_i. It then verifies the candidate and decides whether L reaches
the last objective lattice level below the candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence


class CertificateError(ValueError):
    """Raised when a certificate or instance is malformed or false."""


@dataclass(frozen=True)
class Limits:
    max_rows: int = 100_000
    max_cols: int = 1_000_000
    max_facets: int = 5_000_000
    max_row_tuple: int = 100_000
    max_fraction_bits: int = 1_000_000


def _plain_int(v: Any, name: str) -> int:
    if type(v) is not int:  # reject bool
        raise CertificateError(f"{name} must be an integer")
    return v


def _bit(v: Any, name: str) -> int:
    x = _plain_int(v, name)
    if x not in (0, 1):
        raise CertificateError(f"{name} must be 0 or 1")
    return x


def parse_fraction(obj: Any, name: str, limits: Limits) -> Fraction:
    if not isinstance(obj, dict) or set(obj) != {"num", "den"}:
        raise CertificateError(f"{name} must be exactly {{'num','den'}}")
    num = _plain_int(obj["num"], f"{name}.num")
    den = _plain_int(obj["den"], f"{name}.den")
    if den <= 0:
        raise CertificateError(f"{name}.den must be positive")
    if math.gcd(abs(num), den) != 1:
        raise CertificateError(f"{name} must be in canonical lowest terms")
    if num.bit_length() > limits.max_fraction_bits or den.bit_length() > limits.max_fraction_bits:
        raise CertificateError(f"{name} exceeds rational size limit")
    return Fraction(num, den)


def fraction_json(x: Fraction) -> dict[str, int]:
    return {"num": x.numerator, "den": x.denominator}


def _validate_matrix(H_obj: Any, limits: Limits, name: str = "H") -> list[list[int]]:
    if not isinstance(H_obj, list):
        raise CertificateError(f"{name} must be a list of rows")
    if len(H_obj) > limits.max_rows:
        raise CertificateError(f"{name} exceeds row limit")
    if not H_obj:
        return []
    if not all(isinstance(r, list) for r in H_obj):
        raise CertificateError(f"{name} rows must be lists")
    n = len(H_obj[0])
    if n > limits.max_cols:
        raise CertificateError(f"{name} exceeds column limit")
    out: list[list[int]] = []
    for j, row in enumerate(H_obj):
        if len(row) != n:
            raise CertificateError(f"{name}[{j}] has inconsistent width")
        out.append([_bit(v, f"{name}[{j}][{i}]") for i, v in enumerate(row)])
    return out


def _validate_vector(obj: Any, length: int, name: str) -> list[int]:
    if not isinstance(obj, list) or len(obj) != length:
        raise CertificateError(f"{name} must be a list of length {length}")
    return [_bit(v, f"{name}[{i}]") for i, v in enumerate(obj)]


def syndrome(H: Sequence[Sequence[int]], e: Sequence[int]) -> list[int]:
    return [sum(h * x for h, x in zip(row, e)) & 1 for row in H]


def logical_frame(G: Sequence[Sequence[int]], e: Sequence[int]) -> list[int]:
    return [sum(g * x for g, x in zip(row, e)) & 1 for row in G]


def reconstruct_parity_row(
    H: Sequence[Sequence[int]],
    s: Sequence[int],
    rows_obj: Any,
    limits: Limits,
) -> tuple[tuple[int, ...], list[int], int]:
    if not isinstance(rows_obj, list) or not rows_obj:
        raise CertificateError("facet.rows must be a nonempty list")
    if len(rows_obj) > limits.max_row_tuple:
        raise CertificateError("facet.rows exceeds tuple limit")
    rows = tuple(_plain_int(v, "facet.rows[]") for v in rows_obj)
    if tuple(sorted(rows)) != rows or len(set(rows)) != len(rows):
        raise CertificateError("facet.rows must be strictly increasing and duplicate-free")
    m = len(H)
    if any(r < 0 or r >= m for r in rows):
        raise CertificateError("facet.rows contains an out-of-range row")
    n = len(H[0]) if H else 0
    h = [0] * n
    sigma = 0
    for r in rows:
        sigma ^= s[r]
        for i, bit in enumerate(H[r]):
            h[i] ^= bit
    return rows, h, sigma


def validate_facet(
    H: Sequence[Sequence[int]],
    s: Sequence[int],
    facet_obj: Any,
    limits: Limits,
) -> tuple[tuple[int, ...], tuple[int, ...], Fraction, list[int], int]:
    if not isinstance(facet_obj, dict) or set(facet_obj) != {"rows", "F", "y"}:
        raise CertificateError("each facet must contain exactly rows, F, and y")
    rows, h, sigma = reconstruct_parity_row(H, s, facet_obj["rows"], limits)
    support = {i for i, bit in enumerate(h) if bit}
    F_obj = facet_obj["F"]
    if not isinstance(F_obj, list):
        raise CertificateError("facet.F must be a list")
    F = tuple(_plain_int(v, "facet.F[]") for v in F_obj)
    if tuple(sorted(F)) != F or len(set(F)) != len(F):
        raise CertificateError("facet.F must be strictly increasing and duplicate-free")
    if any(i < 0 or i >= len(h) for i in F):
        raise CertificateError("facet.F contains an out-of-range variable")
    if not set(F).issubset(support):
        raise CertificateError("facet.F is not a subset of the reconstructed support")
    if (len(F) & 1) == sigma:
        raise CertificateError("facet.F has correct parity; it is not a forbidden-set facet")
    y = parse_fraction(facet_obj["y"], "facet.y", limits)
    if y < 0:
        raise CertificateError("facet.y must be nonnegative")
    return rows, F, y, h, sigma


def objective(weights: Sequence[Fraction], e: Sequence[int]) -> Fraction:
    return sum((w * bit for w, bit in zip(weights, e)), Fraction(0))


def objective_lattice_denominator(weights: Sequence[Fraction]) -> int:
    d = 1
    for w in weights:
        d = math.lcm(d, w.denominator)
    return d


def check_certificate(data: dict[str, Any], limits: Limits = Limits()) -> dict[str, Any]:
    allowed = {"H", "s", "weights", "candidate", "facets", "G"}
    unknown = set(data) - allowed
    required = {"H", "s", "weights", "candidate", "facets"}
    if unknown or not required.issubset(data):
        raise CertificateError(f"top-level fields mismatch; unknown={sorted(unknown)}, missing={sorted(required-set(data))}")

    H = _validate_matrix(data["H"], limits, "H")
    m = len(H)
    n = len(H[0]) if H else len(data.get("candidate", []))
    if H and any(len(r) != n for r in H):
        raise CertificateError("H is ragged")
    s = _validate_vector(data["s"], m, "s")
    e = _validate_vector(data["candidate"], n, "candidate")

    w_obj = data["weights"]
    if not isinstance(w_obj, list) or len(w_obj) != n:
        raise CertificateError(f"weights must be a list of length {n}")
    weights = [parse_fraction(v, f"weights[{i}]", limits) for i, v in enumerate(w_obj)]
    if any(v < 0 for v in weights):
        raise CertificateError("weights must be nonnegative")

    actual_s = syndrome(H, e)
    if actual_s != s:
        raise CertificateError("candidate does not satisfy H e = s mod 2")

    facets_obj = data["facets"]
    if not isinstance(facets_obj, list):
        raise CertificateError("facets must be a list")
    if len(facets_obj) > limits.max_facets:
        raise CertificateError("facets exceeds limit")

    t = [Fraction(0) for _ in range(n)]
    b_dot_y = Fraction(0)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for k, facet_obj in enumerate(facets_obj):
        try:
            rows, F, y, h, _sigma = validate_facet(H, s, facet_obj, limits)
        except CertificateError as exc:
            raise CertificateError(f"facet[{k}]: {exc}") from exc
        key = (rows, F)
        if key in seen:
            raise CertificateError(f"facet[{k}] duplicates a prior facet")
        seen.add(key)
        F_set = set(F)
        for i, bit in enumerate(h):
            if bit:
                t[i] += y * (-1 if i in F_set else 1)
        b_dot_y += y * (1 - len(F))

    z = [max(Fraction(0), t_i - w_i) for t_i, w_i in zip(t, weights)]
    lower = b_dot_y - sum(z, Fraction(0))
    upper = objective(weights, e)
    D = objective_lattice_denominator(weights)
    lattice_step = Fraction(1, D)
    if lower > upper:
        # A valid dual lower bound cannot exceed a verified feasible upper bound.
        # Treat this as an instrument failure, not as an especially strong proof.
        raise CertificateError(
            "dual lower bound exceeds the feasible candidate objective"
        )
    exact_equal = lower == upper
    crosses_last_lower_level = lower > upper - lattice_step
    certified_optimal = exact_equal or crosses_last_lower_level

    report: dict[str, Any] = {
        "candidate_feasible": True,
        "candidate_objective": fraction_json(upper),
        "dual_lower_bound": fraction_json(lower),
        "objective_lattice_denominator": D,
        "objective_lattice_step": fraction_json(lattice_step),
        "exact_equality": exact_equal,
        "crosses_last_lower_objective_level": crosses_last_lower_level,
        "objective_optimal": certified_optimal,
        "facet_count": len(facets_obj),
        "derived_upper_bound_duals": [fraction_json(v) for v in z],
    }

    if "G" in data:
        G = _validate_matrix(data["G"], limits, "G")
        if G and len(G[0]) != n:
            raise CertificateError("G width must equal candidate length")
        report["predicted_logical_frame"] = logical_frame(G, e)
        report["logical_frame_determinate"] = "unknown"

    return report


def most_violated_forbidden_set(
    x: Sequence[Fraction], support: Sequence[int], syndrome_bit: int
) -> tuple[tuple[int, ...], Fraction]:
    sigma = _bit(syndrome_bit, "syndrome_bit")
    support_tuple = tuple(support)
    if tuple(sorted(support_tuple)) != support_tuple or len(set(support_tuple)) != len(support_tuple):
        raise CertificateError("support must be strictly increasing and duplicate-free")
    if any(i < 0 or i >= len(x) for i in support_tuple):
        raise CertificateError("support index out of range")

    F0 = {i for i in support_tuple if x[i] > Fraction(1, 2)}
    forbidden_parity = 1 - sigma
    if (len(F0) & 1) != forbidden_parity:
        if not support_tuple:
            # No subset of the opposite parity exists for an empty support.
            raise CertificateError("empty support has no forbidden subset of requested parity")
        i_star = min(support_tuple, key=lambda i: (abs(x[i] - Fraction(1, 2)), i))
        if i_star in F0:
            F0.remove(i_star)
        else:
            F0.add(i_star)
    F = tuple(sorted(F0))
    lhs = sum((1 - x[i] if i in F0 else x[i] for i in support_tuple), Fraction(0))
    return F, lhs


def brute_force_optimum(H: Sequence[Sequence[int]], s: Sequence[int], w: Sequence[Fraction]) -> Fraction | None:
    n = len(w)
    best: Fraction | None = None
    for e in product((0, 1), repeat=n):
        if syndrome(H, e) == list(s):
            val = objective(w, e)
            if best is None or val < best:
                best = val
    return best


def _self_test() -> None:
    # Known exact optimum: x0 xor x1 = 1, unit weights.
    instance = {
        "H": [[1, 1]],
        "s": [1],
        "weights": [{"num": 1, "den": 1}, {"num": 1, "den": 1}],
        "candidate": [1, 0],
        "facets": [
            {"rows": [0], "F": [], "y": {"num": 1, "den": 1}}
        ],
        "G": [[1, 0]],
    }
    r = check_certificate(instance)
    assert r["objective_optimal"] is True
    assert r["dual_lower_bound"] == {"num": 1, "den": 1}

    # A heavier candidate must not be certified by a lower bound of one.
    heavy = dict(instance)
    heavy["H"] = [[1, 1, 0]]
    heavy["weights"] = [{"num": 1, "den": 1}] * 3
    heavy["candidate"] = [1, 0, 1]
    heavy["G"] = [[1, 0, 0]]
    rh = check_certificate(heavy)
    assert rh["candidate_objective"] == {"num": 2, "den": 1}
    assert rh["dual_lower_bound"] == {"num": 1, "den": 1}
    assert rh["objective_optimal"] is False

    # Corrected separator against exhaustive enumeration.
    rng = random.Random(20260802)
    for degree in range(1, 11):
        support = tuple(range(degree))
        for sigma in (0, 1):
            for _ in range(500):
                x = [Fraction(rng.randrange(0, 101), 100) for _ in range(degree)]
                F, lhs = most_violated_forbidden_set(x, support, sigma)
                assert (len(F) & 1) != sigma
                exhaustive: list[Fraction] = []
                for bits in product((0, 1), repeat=degree):
                    if (sum(bits) & 1) != sigma:
                        exhaustive.append(sum((1 - x[i] if bits[i] else x[i] for i in support), Fraction(0)))
                assert lhs == min(exhaustive)

    # Random lower-bound soundness tests against brute force.
    for _ in range(500):
        n = rng.randint(1, 8)
        m = rng.randint(1, 5)
        H = [[rng.randrange(2) for _ in range(n)] for _ in range(m)]
        truth = [rng.randrange(2) for _ in range(n)]
        s = syndrome(H, truth)
        w = [Fraction(rng.randint(0, 7), rng.randint(1, 5)) for _ in range(n)]
        opt = brute_force_optimum(H, s, w)
        assert opt is not None

        row = rng.randrange(m)
        support = [i for i, b in enumerate(H[row]) if b]
        if not support:
            continue
        # Pick any forbidden set by enumerating.
        choices = [tuple(i for i, bit in zip(support, bits) if bit)
                   for bits in product((0, 1), repeat=len(support))
                   if (sum(bits) & 1) != s[row]]
        F = rng.choice(choices)
        y = Fraction(rng.randint(0, 5), rng.randint(1, 5))
        data = {
            "H": H,
            "s": s,
            "weights": [fraction_json(v) for v in w],
            "candidate": truth,
            "facets": [{"rows": [row], "F": list(F), "y": fraction_json(y)}],
        }
        report = check_certificate(data)
        L = Fraction(report["dual_lower_bound"]["num"], report["dual_lower_bound"]["den"])
        assert L <= opt

    print("SELF-TEST PASS: exact checker, separator, and 500 random lower-bound tests")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", nargs="?", help="JSON certificate to verify")
    parser.add_argument("--self-test", action="store_true", help="run deterministic reference tests")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            _self_test()
        if args.certificate:
            data = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
            report = check_certificate(data)
            print(json.dumps(report, indent=2, sort_keys=True))
            if not report["objective_optimal"]:
                return 2
        if not args.self_test and not args.certificate:
            parser.error("provide --self-test and/or a certificate JSON file")
    except (CertificateError, json.JSONDecodeError, OSError) as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
