"""How far may the erasure channel's probability drift from 1/2 before each certified code fails?

Each committed witness (n = 17, 20, 24) is held FIXED; only the flag probability q moves
(oneq.superactivation.fidelity_at_erasure). Writes evidence/superactivation/erasure_sensitivity.json.
q < 1/2 rows are descriptive: there A is no longer two-extendible, so 3/4 is not its bound.

The scoring is `tolerance(Fk)`: pure arithmetic over the branch fidelities, testable without a witness.

  python erasure_sensitivity.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import brentq

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation as sa  # noqa: E402
from oneq import superactivation_certificate as sc  # noqa: E402

EV = ROOT / "evidence" / "superactivation"
GRID = (0.499, 0.5, 0.50001, 0.5001, 0.501, 0.505, 0.51)
Q_HI = 0.6
SCAN = 4001                      # bracket the FIRST crossing on this grid before refining it


def tolerance(Fk, q_hi: float = Q_HI) -> dict:
    """The fixed code's margin against a drifting erasure probability, from its branch fidelities F_k.

    Refuses a code that does not exceed 3/4 at q = 1/2: it has no tolerance to report. The crossing returned
    is the FIRST q > 1/2 where F reaches 3/4 (scanned on SCAN points, then refined), so "F > 3/4 for all
    1/2 <= q < q*" holds up to the scan resolution, not merely "F(q*) = 3/4 somewhere". F(q) is a degree-n
    polynomial; a dip narrower than (q_hi - 1/2)/(SCAN - 1) between scan points is the residual blind spot.
    q* is None when F stays above 3/4 on the whole interval [1/2, q_hi].
    """
    def F(q):
        return sa.fidelity_at_erasure(Fk, q)
    f_half = F(0.5)
    if not f_half > 0.75:
        raise sa.SuperactivationRefusal(f"refusing: F(1/2) = {f_half!r} does not exceed 3/4, so there is no tolerance")
    qs = np.linspace(0.5, q_hi, SCAN)
    first = next((i for i, q in enumerate(qs) if F(q) <= 0.75), None)
    q_star = None if first is None else brentq(lambda q: F(q) - 0.75, qs[first - 1], qs[first], xtol=1e-12)
    return {
        "F_at_half": f_half,
        "q_where_F_reaches_three_quarters": q_star,
        "tolerance_above_half": None if q_star is None else q_star - 0.5,
        "dF_dq_at_half": (F(0.5 + 1e-6) - F(0.5 - 1e-6)) / 2e-6,
        "F_at_q": {f"{q:.5f}": F(q) for q in GRID},
    }


def main() -> int:
    out = {"meaning": "fixed-code fidelity vs the erasure probability q of the 50% erasure channel; "
                      "q < 0.5 voids A's zero-capacity premise and is descriptive only", "codes": {}}
    for n in (17, 20, 24):
        n_, X, decs = sc.load_witness(EV / f"witness_n{n}.npz")
        Fk = sa.branch_fidelities(X, decs, n_)
        row = {"witness": f"witness_n{n}.npz", **tolerance(Fk), "branch_fidelities": [float(v) for v in Fk]}
        out["codes"][str(n)] = row
        print(f"n={n}: F(1/2)={row['F_at_half']:.10f}  stays > 3/4 while q < "
              f"{row['q_where_F_reaches_three_quarters']}  dF/dq={row['dF_dq_at_half']:.4f}")
    (EV / "erasure_sensitivity.json").write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
