"""R200 -- how much worse than its data sheet may the trapped-ion device be, and the m = 8 package still clear 3/4?

The m = 8 package (R195) was priced on data-sheet error rates in a Pauli model. Its biggest risk is that model being
optimistic: leakage, crosstalk and drift are not in it (R186 part 2, "how it could still be wrong"). This scales
every unit's projected error probability by lambda and reruns the package's OWN certificate (r189_pass_rates.
package_dryrun: each unit errs at its scaled rate, then the preregistered Clopper-Pearson certificate), so the answer
is about THIS package's units, groups and weights.

  lambda*  -- the largest error multiplier at which the package still clears in >= 95% of replicates, per claim
              (3 sigma: delta = 0.00135; 5 sigma: delta = Phi(-5)), device and arm.

A stride-2 half of the package (2,000 units) is reported too, as the cheaper 3-sigma request's robustness. It keeps
the design's per-unit groups and weights, but its group counts are those of the stride, not of a fresh allocation.

`size_surface` maps clear rate over (lambda, package size); sizes above 4,000 tile the package's own unit-error
distribution (a PROJECTION, never a package). `pilot_lambda` shows why a pilot cannot size the claim cheaply: at the
projected 2.4% binding-group error, 190 units see about 5 errors and lambda's 95% interval is [0.36, 2.55].
`sequential` is the design that replaces it. The GROUP-SEQUENTIAL certificate runs preregistered batches (cumulative
looks at 1,000 / 2,000 / 4,000 / 8,000 units), certifies after each at delta / looks / groups (a union bound, so
optional stopping keeps the error rate <= delta), and stops at the first look that clears.

  python r200_ion_robustness.py [OUT_JSON]
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import r189_pass_rates as RP  # noqa: E402

PKG = ROOT / "evidence" / "superactivation" / "ion_run_m6" / "design_m8_n4000"
OUT = ROOT / "evidence" / "superactivation" / "r200_ion_robustness" / "robustness.json"
LAMS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
CLAIMS = {"3 sigma": 0.00135, "5 sigma": 0.5 * math.erfc(5 / math.sqrt(2))}
DEVICES = ("helios_1", "h2_2_mem56")
ARMS = ("exec", "barriers")
REPS = 1000
PASS = 0.95


def scaled(proj: dict, lam: float) -> dict:
    """the projection with every unit's error probability on every arm multiplied by lam (clipped to 1)"""
    out = copy.deepcopy(proj)
    for arm in ARMS:
        key = f"e_{arm}"
        out["per_unit"][key] = [min(1.0, lam * float(e)) for e in proj["per_unit"][key]]
    return out


def stride(design: dict, proj: dict, k: int = 2) -> tuple:
    """every k-th unit of the design and its projection, in order (the groups and weights of the design kept)"""
    d = dict(design, units=design["units"][::k])
    p = copy.deepcopy(proj)
    p["units"] = len(d["units"])
    for key, vals in proj["per_unit"].items():
        if isinstance(vals, list) and len(vals) == len(design["units"]):
            p["per_unit"][key] = vals[::k]
    return d, p


def tile(design: dict, proj: dict, k: int) -> tuple:
    """the design and projection repeated k times: a PROJECTION of a k-times larger package drawn from the same law
    (each unit's error rate recurs, as a fresh draw of the same distribution would on average); never a real package"""
    d = dict(design, units=design["units"] * k)
    p = copy.deepcopy(proj)
    p["units"] = len(d["units"])
    for key, vals in proj["per_unit"].items():
        if isinstance(vals, list) and len(vals) == len(design["units"]):
            p["per_unit"][key] = vals * k
    return d, p


def hqc(units: int, per_shot: float) -> float:
    """one parameterized-shot program (R189): 5 HQC plus the per-shot op cost"""
    return 5.0 + units * per_shot


SIZES = {"2000_stride2": None, "4000": 1, "8000_tiled": 2, "12000_tiled": 3}


def size_surface(dev: str = "helios_1", arm: str = "exec", lams=(1.0, 1.5, 2.0, 2.5, 3.0)) -> dict:
    """the clear rate on a (lambda, package size) grid, per claim: read off the smallest package that clears in >= 95%
    at a pilot-measured lambda, and its HQC"""
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    proj = json.loads((PKG / f"executed_projection_{dev}.json").read_text(encoding="utf-8"))
    per_shot = json.loads((PKG / "param_shots_receipt.json").read_text(encoding="utf-8"))["hqc"]["per_shot"]
    out: dict = {}
    for size, k in SIZES.items():
        dz, pz = stride(design, proj) if k is None else (design, proj) if k == 1 else tile(design, proj, k)
        for claim, delta in CLAIMS.items():
            rows = {}
            for lam in lams:
                r = RP.package_dryrun(scaled(pz, lam), dz, arm, delta=delta, reps=REPS, seed=201)
                rows[str(lam)] = round(r["clears"] / r["reps"], 4)
            out[f"{size}|{claim}"] = dict(units=len(dz["units"]), hqc=round(hqc(len(dz["units"]), per_shot), 1),
                                          clear_rate_by_lambda=rows)
    return out


def projected_group_rates(dev: str = "helios_1", arm: str = "exec") -> dict:
    """the package projection's mean per-unit error probability per group, on this device and arm"""
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    proj = json.loads((PKG / f"executed_projection_{dev}.json").read_text(encoding="utf-8"))
    out: dict = {}
    for u, e in zip(design["units"], proj["per_unit"][f"e_{arm}"]):
        out.setdefault(u["group"], []).append(float(e))
    return {g: float(np.mean(v)) for g, v in out.items()}


BINDING = "kept/high"             # W x width 0.73 of the certificate's 0.81 (design.json groups)


def pilot_lambda(errors: int, shots: int, dev: str = "helios_1", arm: str = "exec", conf: float = 0.95) -> dict:
    """a PILOT's measured error multiplier on the binding group: lambda_hat = observed rate / projected rate, with the
    exact two-sided Clopper-Pearson interval of the rate carried through. The UPPER end sizes the claim (`recommend`)"""
    from scipy.stats import beta
    if shots <= 0 or not 0 <= errors <= shots:
        raise ValueError(f"pilot counts {errors}/{shots} are not a binomial outcome")
    proj = projected_group_rates(dev, arm)[BINDING]
    a = (1 - conf) / 2
    lo = 0.0 if errors == 0 else float(beta.ppf(a, errors, shots - errors + 1))
    hi = 1.0 if errors == shots else float(beta.ppf(1 - a, errors + 1, shots - errors))
    return dict(group=BINDING, errors=errors, shots=shots, rate=errors / shots, projected_rate=proj,
                lambda_hat=errors / shots / proj, lambda_ci=[lo / proj, hi / proj], conf=conf, device=dev, arm=arm)


def recommend(lambda_upper: float, surface: dict, target: float = PASS) -> dict:
    """per claim, the cheapest package in `surface` (size_surface's output) whose clear rate at the first grid lambda
    >= lambda_upper is >= target; None where no size on the surface reaches it (or lambda_upper is off the grid)"""
    out: dict = {}
    for claim in CLAIMS:
        best = None
        for key, row in surface.items():
            size, c = key.split("|")
            if c != claim:
                continue
            grid = sorted(float(x) for x in row["clear_rate_by_lambda"])
            at = next((g for g in grid if g >= lambda_upper - 1e-12), None)
            if at is None or row["clear_rate_by_lambda"][str(at)] < target:
                continue
            if best is None or row["hqc"] < best["hqc"]:
                best = dict(package=size, units=row["units"], hqc=row["hqc"], at_lambda=at,
                            clear_rate=row["clear_rate_by_lambda"][str(at)])
        out[claim] = best
    return out


LOOKS = (1000, 2000, 4000, 8000)


def sequential(dev: str = "helios_1", arm: str = "exec", looks=LOOKS, lams=(1.0, 1.5, 2.0, 2.5), reps: int = REPS,
               seed: int = 202) -> dict:
    """the GROUP-SEQUENTIAL certificate: run the package in batches, certify after each cumulative look with the
    design's exact Clopper-Pearson bound at delta / len(looks) (a union bound over the preregistered looks, so the
    claim's error rate stays <= delta under this optional stopping), and STOP at the first look that clears 3/4.
    Units beyond the committed 4,000 are the package's own error distribution tiled (a PROJECTION). Returns, per claim
    and lambda: the probability of clearing by each look, of never clearing, and the expected HQC spent."""
    import r186_certification as C
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    proj = json.loads((PKG / f"executed_projection_{dev}.json").read_text(encoding="utf-8"))
    per_shot = json.loads((PKG / "param_shots_receipt.json").read_text(encoding="utf-8"))["hqc"]["per_shot"]
    k = math.ceil(max(looks) / len(design["units"]))
    d, p = tile(design, proj, k)
    groups = [u["group"] for u in d["units"]][:max(looks)]
    e_all = np.array(p["per_unit"][f"e_{arm}"][:max(looks)], dtype=float)
    names = list(design["groups"])
    gi = np.array([names.index(g) for g in groups])
    rng = np.random.default_rng(seed)
    out: dict = {}
    for claim, delta in CLAIMS.items():
        a = delta / len(looks) / len(names)                      # per look, per group (union bound)
        for lam in lams:
            e = np.clip(lam * e_all, 0.0, 1.0)
            stop = np.full(reps, -1)
            for r in range(reps):
                err = rng.random(len(e)) < e
                for li, n in enumerate(looks):
                    f_lo = design["f_ideal"]
                    for j, g in enumerate(names):
                        sel = gi[:n] == j
                        f_lo -= design["groups"][g]["width"] * design["groups"][g]["W"] * C.cp_upper(
                            int(err[:n][sel].sum()), int(sel.sum()), a)
                    if 1.5 * f_lo - 0.5 > design["bar_F_e"]:
                        stop[r] = li
                        break
            by_look = {str(n): float(np.mean(stop == li)) for li, n in enumerate(looks)}
            spent = [hqc(looks[s], per_shot) + 5.0 * s if s >= 0 else hqc(max(looks), per_shot) + 5.0 * (len(looks) - 1)
                     for s in stop]
            out[f"{claim}|{lam}"] = dict(clears_by_look=by_look, never=float(np.mean(stop < 0)),
                                         expected_hqc=float(np.mean(spent)), max_hqc=hqc(max(looks), per_shot)
                                         + 5.0 * (len(looks) - 1))
    return dict(looks=list(looks), alpha_split="delta / looks / groups (union bound)", device=dev, arm=arm,
                results=out)


def lambda_star(rows: list) -> float | None:
    """the largest lambda whose clear rate is >= PASS, provided every smaller lambda passes too (monotone);
    None when even lambda = 1 fails"""
    best = None
    for r in sorted(rows, key=lambda r: r["lambda"]):
        if r["clears"] / r["reps"] >= PASS:
            best = r["lambda"]
        else:
            break
    return best


def run() -> dict:
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    out: dict = dict(schema="oneq.r200.ion_robustness/1", package=PKG.relative_to(ROOT).as_posix() if PKG.is_relative_to(ROOT) else str(PKG), m=design["m"],
                     reps=REPS, pass_rate=PASS, evidence_class="SIMULATED on the package's own projected unit errors",
                     results={})
    for dev in DEVICES:
        proj = json.loads((PKG / f"executed_projection_{dev}.json").read_text(encoding="utf-8"))
        for size, (dz, pz) in (("4000", (design, proj)), ("2000_stride2", stride(design, proj))):
            for claim, delta in CLAIMS.items():
                for arm in ARMS:
                    rows = []
                    for lam in LAMS:
                        r = RP.package_dryrun(scaled(pz, lam), dz, arm, delta=delta, reps=REPS, seed=200)
                        rows.append(dict(**{"lambda": lam}, clears=r["clears"], reps=r["reps"],
                                         certified_F_e_median=r["certified_F_e_median"],
                                         certified_F_e_p05=r["certified_F_e_p05"]))
                    key = f"{dev}|{size}|{claim}|{arm}"
                    out["results"][key] = dict(lambda_star=lambda_star(rows), rows=rows)
                    print(f"{key:44s} lambda* = {lambda_star(rows)}", flush=True)
    return out


def main() -> int:
    res = run()
    res["size_surface"] = {f"{dev}|{arm}": size_surface(dev, arm) for dev in DEVICES for arm in ARMS}
    res["sequential"] = {f"{dev}|{arm}": sequential(dev, arm) for dev in DEVICES for arm in ARMS}
    for k, v in res["sequential"].items():
        for c, row in v["results"].items():
            print(f"SEQ {k:22s} {c:14s} E[HQC] {row['expected_hqc']:7.1f}  never {row['never']:.3f}  "
                  f"by look {row['clears_by_look']}", flush=True)
    for k, v in res["size_surface"].items():
        for s, row in v.items():
            print(f"{k:22s} {s:22s} HQC {row['hqc']:7.1f}  clear rate by lambda {row['clear_rate_by_lambda']}",
                  flush=True)
    dest = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
