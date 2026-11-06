"""R189 part 2: what the certificate clears when the device model is evaluated on each arm's circuits -- the frozen
level-3 compile (R186/R188, merges across the channel boundary), the channel-faithful barriered compile, and the
executed one-program stream.

Each group's unit law is BOOTSTRAPPED from that arm's per-unit error rates (r189_executed_projection.py, all 2,000
frozen units: themselves fresh draws from the group's law), so the simulated experiment is the preregistered one --
N fresh single-shot units, allocated as `ion_run_package.allocation` allocates them, an exact Clopper-Pearson bound
per high group at delta / 2, low groups at 0, the claim F_e_lower > 3/4 -- run on that arm's errors.

  python r189_pass_rates.py PROJECTION_JSON [PROJECTION_JSON ...]   -> JSON on stdout
  python r189_pass_rates.py --package DESIGN_DIR PROJECTION_JSON ... THE package's dry run: its own units, each erring
                                                                     at its projected rate (executed and barriers)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

PLANS = ((2000, 0.00135), (4000, 0.00135), (4000, 2.87e-7), (6000, 2.87e-7), (8000, 2.87e-7))
REPS = 400


def pass_rates(proj: dict, design: dict, reps: int = REPS, seed: int = 189) -> dict:
    import ion_run_package as P
    import r186_certification as C
    rng = np.random.default_rng(seed)
    groups = design["groups"]
    pu = proj["per_unit"]
    out = {}
    for arm in ("frozen", "barriers", "exec"):
        e_by = {g: np.array([e for e, gg in zip(pu[f"e_{arm}"], pu["group"]) if gg == g]) for g in groups}
        rows = []
        for n_units, delta in PLANS:
            alloc = P.allocation(n_units)["alloc"]
            a = delta / len(groups)
            ok, lows = 0, []
            for _ in range(reps):
                f_lo = design["f_ideal"]
                for g, meta in groups.items():
                    n = alloc[g]
                    p = np.clip(rng.choice(e_by[g], size=n), 0.0, 1.0)
                    k = int((rng.random(n) < p).sum())
                    f_lo -= meta["width"] * meta["W"] * C.cp_upper(k, n, a)
                F = 1.5 * f_lo - 0.5
                lows.append(F)
                ok += F > design["bar_F_e"]
            rows.append(dict(units=n_units, delta=delta, clears=ok, reps=reps,
                             certified_F_e_median=float(np.median(lows))))
        out[arm] = dict(F_e_projected=proj["F_e_projected"][arm], plans=rows)
    return dict(device=proj["device"], arms=out)


def package_dryrun(proj: dict, design: dict, arm: str = "exec", delta: float | None = None, reps: int = 2000,
                   seed: int = 4000) -> dict:
    """THE frozen package's own run, simulated: every unit of `design` runs its one shot and errs with ITS projected
    rate on `arm` (no bootstrap -- the units are the package's), then the preregistered certificate. Refused unless
    the projection covers exactly this design's units, in order"""
    import r186_certification as C
    pu = proj["per_unit"]
    units = design["units"]
    if proj["units"] != len(units) or pu["group"] != [u["group"] for u in units]:
        raise ValueError("the projection is not of this design's units, in this order")
    delta = design["delta"] if delta is None else delta
    e = np.clip(np.array(pu[f"e_{arm}"]), 0.0, 1.0)
    groups = design["groups"]
    idx = {g: np.array([i for i, u in enumerate(units) if u["group"] == g]) for g in groups}
    a = delta / len(groups)
    rng = np.random.default_rng(seed)
    ok, lows = 0, []
    for _ in range(reps):
        f_lo = design["f_ideal"]
        for g, meta in groups.items():
            k = int((rng.random(len(idx[g])) < e[idx[g]]).sum())
            f_lo -= meta["width"] * meta["W"] * C.cp_upper(k, len(idx[g]), a)
        F = 1.5 * f_lo - 0.5
        lows.append(F)
        ok += F > design["bar_F_e"]
    return dict(device=proj["device"], arm=arm, units=len(units), delta=delta, reps=reps, clears=ok,
                certified_F_e_median=float(np.median(lows)), certified_F_e_p05=float(np.quantile(lows, 0.05)))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--package":
        # --package DESIGN_DIR PROJECTION... [--sigma S]: the package's own dry run at its preregistered delta, or at
        # the one-sided delta = Phi(-S) of an S-sigma claim (R195: m = 8 at 5 sigma)
        delta = None
        if "--sigma" in argv:
            i = argv.index("--sigma")
            delta = 0.5 * math.erfc(float(argv[i + 1]) / math.sqrt(2))
            argv = argv[:i] + argv[i + 2:]
        design = json.loads((pathlib.Path(argv[1]) / "design.json").read_text(encoding="utf-8"))
        out = []
        for p in argv[2:]:
            proj = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
            out += [package_dryrun(proj, design, arm, delta=delta) for arm in ("exec", "barriers")]
        print(json.dumps(out, indent=1))
        return 0
    design = json.loads((ROOT / "evidence" / "superactivation" / "ion_run_m6" / "design_n2000" / "design.json")
                        .read_text(encoding="utf-8"))
    res = [pass_rates(json.loads(pathlib.Path(p).read_text(encoding="utf-8")), design) for p in argv]
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
