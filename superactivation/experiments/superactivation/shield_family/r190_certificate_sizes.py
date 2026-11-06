"""R190: how many single-shot units the error-indicator certificate needs at each shield size m, per device and arm.

Each high group's error indicator is exactly Bernoulli at its mean rate (a mixture of Bernoullis is Bernoulli), so a
simulated run needs only the group means: r190_m_scan.py's high-group errors per (m, device, arm), the EXACT
design-law weights (ion_run_package.committed_weights) and the preregistered rule -- Clopper-Pearson per high group at
delta / 2, low groups at 0, the claim F_e_lower > 3/4 -- with N units split as ion_run_package.allocation splits them.

  python r190_certificate_sizes.py MSCAN_JSON [--reps R]   -> JSON on stdout
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

SIZES = (500, 1000, 1500, 2000, 4000)
DELTAS = (("3 sigma", 0.00135), ("5 sigma", 2.87e-7))


def sizes(scan: list, reps: int = 400, seed: int = 190) -> list:
    import hw_two_use as H
    import ion_run_package as P
    import r186_certification as C
    rng = np.random.default_rng(seed)
    out = []
    for r in scan:
        m = r["m"]
        W = P.committed_weights(m)
        f_ideal = (2 * H.f_e_closed(m) + 1) / 3
        for arm in ("faithful", "barriers"):
            ebar = r["arms"][arm]["high_group_error"]
            rows = []
            for label, delta in DELTAS:
                for n_units in SIZES:
                    plan = P.allocation(n_units, m, ebar)
                    a = delta / len(plan["alloc"])
                    ok, lows = 0, []
                    for _ in range(reps):
                        f_lo = f_ideal
                        for g, n in plan["alloc"].items():
                            k = int((rng.random(n) < ebar[g]).sum())
                            f_lo -= plan["groups"][g]["width"] * W[g] * C.cp_upper(k, n, a)
                        F = 1.5 * f_lo - 0.5
                        lows.append(F)
                        ok += F > 0.75
                    rows.append(dict(claim=label, delta=delta, units=n_units, clears=ok, reps=reps,
                                     certified_F_e_median=float(np.median(lows))))
            out.append(dict(m=m, device=r["device"], arm=arm, F_e_model=r["arms"][arm]["F_e"], plans=rows))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mscan")
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args(argv)
    print(json.dumps(sizes(json.loads(pathlib.Path(a.mscan).read_text(encoding="utf-8")), a.reps), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
