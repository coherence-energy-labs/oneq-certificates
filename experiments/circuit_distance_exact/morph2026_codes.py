r"""Re-find the end-cycle codes 2604.09797 reports for its Abelian-2BGA morphing circuits (Figs. 11, 25, 26; App. C).

    python morph2026_codes.py --N 42 --w 5 --k 6 --d 6          # [[42,6,6]], degree 4 (weight-5 mid-cycle code)
    python morph2026_codes.py --N 30 --w 6 --k 8 --d 5          # [[30,8,5]], degree 5 (weight-6 mid-cycle code)

The paper names neither group nor polynomials, so this enumerates them. Mid-cycle code: G = Z_N (every abelian group
of a squarefree order is cyclic -- 30 and 42 are squarefree), A and B with |A| + |B| = w, |A| >= |B| >= 2, both
containing 0 (translation symmetry), deduplicated under translations of A and of B, multiplication by units
(automorphisms) and, when |A| = |B|, the ZX duality (A, B) -> (-B, -A). Circuit: the standard (homomorphism-based)
morphing schedule of morph2026.morphing_schedule with f(g) = g mod 2 (the only surjection Z_N -> Z_2). For every
ORDER of the non-special elements (it changes the CNOT layers, not the code) the end-cycle code of both rounds is
computed and its distance found exactly (meet in the middle); the mid-cycle distance is checked >= the end-cycle
distance as the paper does. JSON lines: one per circuit whose end-cycle code is [[N, k, d]] exactly, then a summary.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import morph2026_gf2 as gf2  # noqa: E402
import morph2026 as M  # noqa: E402


def canonical(N: int, A, B):
    units = [u for u in range(1, N) if math.gcd(u, N) == 1]
    best = None
    for u in units:
        for a0 in A:
            for b0 in B:
                key = (tuple(sorted(u * (a - a0) % N for a in A)), tuple(sorted(u * (b - b0) % N for b in B)))
                if len(A) == len(B):
                    key = min(key, (key[1], key[0]))  # (A, B) ~ (-B, -A) ~ (B, A) via the unit -1
                best = key if best is None or key < best else best
    return best


def k_of(N: int, A, B) -> int:
    code = M.Code2BGA(N, 1, [(a, 0) for a in A], [(b, 0) for b in B])
    hx, hz = code.rows()
    return 2 * N - gf2.rank(hx) - gf2.rank(hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    a = ap.parse_args()
    N, w, kt, dt = a.N, a.w, a.k, a.d
    t0 = time.time()
    seen, n_raw, n_valid, n_k, hits = set(), 0, 0, 0, []
    for nb in range(2, w // 2 + 1):
        na = w - nb
        for Arest in itertools.combinations(range(1, N), na - 1):
            for Brest in itertools.combinations(range(1, N), nb - 1):
                n_raw += 1
                A, B = (0,) + Arest, (0,) + Brest
                pa, pb = [x % 2 for x in A], [x % 2 for x in B]
                if min(pa.count(0), pa.count(1)) != 1 or min(pb.count(0), pb.count(1)) != 1:
                    continue
                key = canonical(N, A, B)
                if key in seen:
                    continue
                seen.add(key)
                n_valid += 1
                if k_of(N, *key) != kt:
                    continue
                n_k += 1
                A0, B0 = key
                half = (dt + 1) // 2
                # every distinct ordering of the lists (the schedule swaps the special element to the front; the
                # order of the others sets the layers)
                orders = set()
                for Ap in itertools.permutations(A0):
                    for Bp in itertools.permutations(B0):
                        try:
                            code = M.Code2BGA(N, 1, [(x, 0) for x in Ap], [(x, 0) for x in Bp])
                            gates, meas, As, Bs = M.morphing_schedule(code, (1, 0))
                        except (ValueError, AssertionError):
                            continue
                        orders.add((tuple(x[0] for x in As), tuple(x[0] for x in Bs)))
                for As, Bs in sorted(orders):
                    code = M.Code2BGA(N, 1, [(x, 0) for x in As], [(x, 0) for x in Bs])
                    try:
                        gates, meas, _, _ = M.morphing_schedule(code, (1, 0))
                    except (ValueError, AssertionError) as e:
                        continue
                    ds = []
                    for r in (0, 1):
                        hx, hz = M.end_cycle_code(code, gates, meas, r)
                        ds.append((gf2.min_logical(hx, hz, N, half), gf2.min_logical(hz, hx, N, half)))
                    dend = [min((x for x in pair if x is not None), default=None) for pair in ds]
                    if any(x is not None and x < dt for pair in ds for x in pair):
                        continue
                    exact = all(x == dt for pair in ds for x in pair)
                    dmid = code.distance(half)
                    if any(x is not None and x < dt for x in dmid):
                        continue
                    row = dict(N=N, A=list(As), B=list(Bs), k=kt, d_end_rounds=ds, d_end_exactly_target=exact,
                               d_mid_xz=dmid, degree=M.connectivity_degree(gates, 2 * N), layers=len(gates[0]))
                    hits.append(row)
                    print(json.dumps(row), flush=True)
    print(json.dumps(dict(summary=True, N=N, w=w, k=kt, d=dt, raw=n_raw, classes_valid=n_valid,
                          classes_with_k=n_k, circuits_hit=len(hits),
                          circuits_exact=sum(h["d_end_exactly_target"] for h in hits),
                          seconds=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
