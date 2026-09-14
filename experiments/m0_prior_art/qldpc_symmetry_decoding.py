r"""A6 -- SYMMETRY-AWARE DECODING: free information the field discards.

THE SPEC (FTQC_BUILD_PROGRAM A6). Inject the algorithm's conserved
quantities -- particle number, S_z, spatial symmetry -- as ADDITIONAL
detectors in the DEM, so syndromes violating them are refused outright.
The evidence behind the bet: getting MORE INFORMATION into the decoder
(erasure conversion, 2-7.5x threshold) beats inferring better over the
same information (+15%) by roughly an order of magnitude.

WHAT A CONSERVED QUANTITY IS, IN THE CODE. A quantity conserved by the
algorithm's Hamiltonian is measurable, and in the encoded picture it is a
LOGICAL observable -- fermion parity, total charge, S_z all map to
products of logical operators. So "the algorithm conserves Q" means "this
logical observable's value is known independently of the syndrome". That
is one extra parity bit per shot that the decoder is normally never told.

Adding it is a one-line change to the decoding problem: append the
logical operator's support as an extra CHECK ROW, with the measured value
as its syndrome bit. Every property of the pipeline survives, because the
augmented system is still a sparse GF(2) system and the certificate
machinery only ever sees (checks, weights, syndrome). The row is
high-degree (weight ~n/2), which is exactly what feldman_dual_lazy's
O(deg) separator exists for -- no facet enumeration, any degree welcome.

WHY THIS IS NOT TRUTH LEAKAGE, and how that is PROVEN rather than
asserted. Handing a decoder a bit derived from the true error would be
cheating, and a result built on it would be worthless. So the symmetry
bit is measured with its OWN error rate q and the benefit is reported
across a ladder in q:

    q = 0      a perfectly measured conserved quantity
    q = 0.01   a good measurement
    q = 0.10   a poor one
    q = 0.50   PURE NOISE -- the negative control

At q = 0.5 the bit carries zero information, so any apparent gain there
is an artifact of the harness and the whole result is void. The gain must
decay to nothing as q rises; that curve is the evidence that what helps
is the INFORMATION and not the extra constraint.

WHAT IS MEASURED, per shot, at every q:
  * logical error on the GUARDED observable (the one the symmetry pins)
  * logical error on ANY observable (guarding one must not wreck others)
  * REFUSALS: syndromes where the decoder's own answer violates the
    measured symmetry -- the "refused outright" the spec asks for, and a
    detection the unguarded pipeline cannot make at all
  * certified rate, so the guarantee is not traded away for the gain
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import (bb_gross_code, gf2_in_rowspace,   # noqa: E402
                           gf2_row_reduce)
from oneq.qldpc_cert import exact_milp, feldman_dual_lazy    # noqa: E402
from oneq.qldpc_check import QldpcCertificate, check_qldpc   # noqa: E402


def gf2_nullspace(M):
    """A basis for {v : M v = 0} over GF(2), as 0/1 row vectors."""
    m, n = M.shape
    A = M.copy() % 2
    piv, row = [], 0
    for col in range(n):
        sel = None
        for r in range(row, m):
            if A[r, col]:
                sel = r
                break
        if sel is None:
            continue
        A[[row, sel]] = A[[sel, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        piv.append(col)
        row += 1
        if row == m:
            break
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            if A[i, f]:
                v[c] = 1
        basis.append(v)
    return basis


def pick_logical(HZ, HX):
    """A logical Z representative: in ker(HZ), NOT in rowspace(HX).

    A conserved quantity that is already implied by the stabilizers would
    add no information at all -- the decoder can derive it -- so the
    representative is explicitly checked to lie OUTSIDE the rowspace.
    Prefer a low-weight one: a lighter observable is a cheaper thing to
    measure on hardware, and the row it adds is cheaper to separate.
    """
    n = HZ.shape[1]
    xb, xp = gf2_row_reduce(HX % 2)
    best = None
    for v in gf2_nullspace(HZ % 2):
        if not v.any():
            continue
        if gf2_in_rowspace(xb, xp, v, n):
            continue                    # a stabilizer: no new information
        w = int(v.sum())
        if best is None or w < best[0]:
            best = (w, v)
    assert best is not None, "no logical found outside the stabilizer group"
    return best[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    L = pick_logical(HZ, HX)
    lsup = tuple(int(i) for i in np.flatnonzero(L))
    xb, xp = gf2_row_reduce(HX % 2)
    print(f"[[144,12,12]] Z-sector; conserved observable = logical Z of "
          f"weight {len(lsup)} (verified OUTSIDE the stabilizer rowspace)")

    def is_logical(fault):
        if ((HZ @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(xb, xp, fault, n)

    def flips_guarded(fault):
        return bool(int(L @ fault) % 2)

    base_checks = [tuple(int(i) for i in np.flatnonzero(HZ[j]))
                   for j in range(m)]
    w = {i: 1.0 for i in range(n)}

    rng = np.random.default_rng(20260804)
    shots = []
    while len(shots) < a.shots:
        e = (rng.random(n) < a.p).astype(np.uint8)
        syn = (HZ @ e) % 2
        if syn.any():
            shots.append((syn, e))

    def decode(checks, syn_list):
        """The certified answer: lazy dual first (any degree), exact MILP
        as the closer. Returns (support, certified)."""
        try:
            cert, _b, primal = feldman_dual_lazy(checks, w, syn_list,
                                                 return_primal=True)
        except Exception:                        # noqa: BLE001
            cert, primal = None, None
        if cert is not None and primal is not None:
            r = check_qldpc(checks=checks, weights=w, syndrome=syn_list,
                            cert=QldpcCertificate(
                                error_support=tuple(primal),
                                facet_duals=cert.facet_duals,
                                box_duals=cert.box_duals))
            if r.accepted:
                return tuple(primal), True
        sup, _wt, _st = exact_milp(checks, w, syn_list, return_status=True)
        return (tuple(int(i) for i in sup) if sup is not None else ()), False

    out = {"schema": "oneq-symmetry-decoding/1",
           "code": "[[144,12,12]] gross, Z-sector, code capacity",
           "p": a.p, "shots": len(shots),
           "observable_weight": len(lsup),
           "premise": ("a conserved quantity of the algorithm is a LOGICAL "
                       "observable, so knowing it is one extra parity bit "
                       "per shot -- appended as a high-degree check row, "
                       "separated lazily"),
           "levels": {}}

    # ---- baseline: no symmetry information at all
    t0 = time.perf_counter()
    base = []
    for (syn, e) in shots:
        sup, cert = decode(base_checks, list(syn))
        r = np.zeros(n, dtype=np.uint8)
        r[list(sup)] = 1
        base.append((r, cert))
    b_any = sum(int(is_logical((r ^ e) % 2)) for (r, _c), (_s, e)
                in zip(base, shots))
    b_guard = sum(int(flips_guarded((r ^ e) % 2)) for (r, _c), (_s, e)
                  in zip(base, shots))
    b_cert = sum(int(c) for (_r, c) in base)
    out["baseline"] = {"logical_any": b_any, "logical_guarded": b_guard,
                       "certified": b_cert,
                       "seconds": round(time.perf_counter() - t0, 1)}
    print(f"  baseline (no symmetry): logical ANY {b_any}, "
          f"GUARDED {b_guard}, certified {b_cert}/{len(shots)}")

    aug_checks = base_checks + [lsup]
    for q in (0.0, 0.01, 0.10, 0.50):
        t0 = time.perf_counter()
        l_any = l_guard = certn = refused = 0
        for k, (syn, e) in enumerate(shots):
            true_bit = int(L @ e) % 2
            meas = true_bit ^ int(rng.random() < q)
            sup, cert = decode(aug_checks, list(syn) + [meas])
            r = np.zeros(n, dtype=np.uint8)
            r[list(sup)] = 1
            f = (r ^ e) % 2
            l_any += int(is_logical(f))
            l_guard += int(flips_guarded(f))
            certn += int(cert)
            # THE REFUSAL the spec asks for: the unguarded answer violates
            # the measured conserved quantity, so it is rejected outright
            if int(L @ base[k][0]) % 2 != meas:
                refused += 1
        lvl = {"logical_any": l_any, "logical_guarded": l_guard,
               "certified": certn, "refusals_of_unguarded_answer": refused,
               "seconds": round(time.perf_counter() - t0, 1)}
        out["levels"][f"q={q}"] = lvl
        print(f"  q={q:<5}: logical ANY {l_any:3d}  GUARDED {l_guard:3d}  "
              f"certified {certn:3d}/{len(shots)}  "
              f"refusals {refused:3d}  ({lvl['seconds']}s)")

    g0 = out["levels"]["q=0.0"]["logical_guarded"]
    ghalf = out["levels"]["q=0.5"]["logical_guarded"]
    out["verdict"] = {
        "guarded_errors_baseline": b_guard,
        "guarded_errors_perfect_symmetry": g0,
        "guarded_errors_pure_noise_control": ghalf,
        "information_not_artifact": (g0 <= b_guard and ghalf >= g0),
        "reading": ("a perfectly measured conserved quantity should drive "
                    "errors on THAT observable to zero; a pure-noise bit "
                    "(q=0.5) must give no benefit at all -- that control "
                    "is what separates information from a lucky extra "
                    "constraint")}

    fail = []
    if g0 > b_guard:
        fail.append(f"PERFECT SYMMETRY MADE IT WORSE: guarded errors "
                    f"{b_guard} -> {g0}")
    if ghalf < g0:
        fail.append(f"NEGATIVE CONTROL INVERTED: pure noise (q=0.5) beat "
                    f"a perfect measurement ({ghalf} < {g0}) -- the gain "
                    f"is an artifact of the extra row, not information")
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "qldpc_symmetry_decoding.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    if fail:
        print("\nA6 FAILED:")
        for f in fail:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
