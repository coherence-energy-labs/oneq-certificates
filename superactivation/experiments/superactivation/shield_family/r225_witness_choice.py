"""R225 -- the Helios run's test, chosen on operating characteristics: F_e > 3/4 (R203), an antidegradability witness
frozen from the device model (R213), or their fixed 50/50 mixture -- on the frozen R224 package, whose labels and
circuits every candidate shares (only the score each error carries differs).

THE NULL (R213). Everything a P-free or A-free device can implement (A^m, P^m, mixtures, shared randomness, P used
classically, the flags) is an antidegradable (AD) qubit channel: PPT qubit channels are entanglement-breaking, A is 50%
erasure. AD is convex: a channel is AD iff its Choi state has a symmetric extension on the output.

THE WITNESS BOUND (DERIVED; each bound used is certified in exact arithmetic here). For w >= 0,
    w . diag(T) <= lambda_max(M_w)  for every AD qubit channel,   M_w = (W_BA (x) I_B' + W_B'A (x) I_B) / 2,
with W = sum_i w_i s_i (x) s_i^T and T the Pauli transfer matrix: J's symmetric extension rho on (B, B', A) gives
Tr(W J) = Tr(M_w rho) <= lambda_max(M_w), and Tr(W J) = w . diag(T). (R213: the bound is attained -- a twirled top
eigenvector -- and agrees with the Chen-Ji-Kribs-Lutkenhaus-Zeng criterion; only the inequality is needed for validity.)
M_w is a real 8 x 8 matrix (Y (x) Y^T is real), so hbar(w) = a rational with hbar I - M_w positive definite, shown by
an exact LDL^T (`certify_bound`), is a proven upper bound. w = (1, 1, 1) gives 2: the cloning bound F_e <= 3/4.

THE TEST. The design's per-basis logical diagonal is lam_b = lam_b(ideal) - 6 sum_g width_g W_{g,b} e_{g,b}, with
W_{g,b} the exact design weight of group g's units of input basis b (ion_run_package, `W_exact_by_basis`) and e_{g,b}
their error rate; errors of LOW units only raise lam_b, so with w >= 0 dropping them is conservative (R190). A shot
labelled g (P(g) = pi_g, iid), its unit a fresh draw of basis b (probability W_{g,b} / W_g), scores
    S = kappa_{g,b} X,   kappa_{g,b} = 6 w_b width_g W_g / pi_g,     E[S | past] = sum_{g,b} 6 w_b width_g W_{g,b} e_{g,b},
and the null is  mean E[S] >= m_w = w . lam(ideal) - hbar(w). That is R203's test with other scores: the same Ville
argument, the same constant bets, the same looks. w = (1, 1, 1) / 6 IS the F_e test (checked against ion_betting).
The fixed mixture K = (K_F + K_w) / 2 is valid too: under the null (the shot-averaged channel AD) both components sit
on their nulls, so K is below a mean of nonnegative martingales from 1.

THE DEVICE. Per (group, basis) error rates from R195's executed Helios projection -- the committed source of the F_e
operating characteristics' group means -- times lambda. kept's basis PROFILE is varied at its fixed group mean over
DIRECTIONS: 'model' (the projection's own), 'isotropic', and 'adverse' (only the X and Y bases err: the excess noise a
model-chosen w weighs most). F_e sees only group means, so its operating characteristics are the same in every
direction; only a witness's move.

THE OPERATING CHARACTERISTICS. The wealth at a look is a function of the error counts by (group, basis), so a run is
simulated at the level of its per-look counts (multinomial per batch), common random numbers across the tests
(SIMULATED; F_e cross-checked against r203_betting.oc_exact's exact DP). No futility stop for any test: the comparison
is on P(holds) and on spend without it (R209's futility rule is F_e's, fitted to F_e).

THE DECISION RULE (declared in this file before its first run, 2026-10-26):
  1. w* maximizes the Kelly growth rate at lambda = 2.5 in the 'model' direction over w >= 0, sum w = 1/2, and is
     rounded to a rational with denominator 200. It comes from the device model alone; no device shot exists.
  2. Bets: F_e keeps R203's (0.3, 0.5, 0.8). A witness's bets are its Kelly-optimal bets at the lambdas where those
     are F_e's Kelly-optimal bets ('model' direction): the same rule, transferred, never tuned on the outcome.
     Rounded to 3 decimals, as preregistered constants are (amended after the first run, before any choice rested on
     it: the 200,000-run decision was the same with the unrounded bets).
  3. A candidate (W* or MIX) QUALIFIES if in EVERY direction, at every lambda where F_e's P(holds) >= 0.5, its
     P(holds) >= F_e's - 0.01: a model that is wrong about the noise's direction must not cost a run F_e would win.
  4. Among qualifying candidates, the larger P(holds) at lambda = 2.5 ('model') wins if it beats F_e's by >= 0.05;
     otherwise F_e stays (the simpler test, already machine-checked in Lean).

  python r225_witness_choice.py [REPS]     -> evidence/superactivation/r225_witness/choice.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import ad_witness as A  # noqa: E402
import ion_betting as B  # noqa: E402
import ion_run_package as P  # noqa: E402
import ion_sequential as Q  # noqa: E402
import r203_betting as R3  # noqa: E402

EV = ROOT / "evidence" / "superactivation"
PKG = EV / "ion_run_m8_r224"
PROJ = EV / "ion_run_m6" / "design_m8_n4000"          # R195's executed projection (per-unit, with inputs)
OUT = EV / "r225_witness" / "choice.json"
BASES = "XYZ"
CATS = (("ee_x/high", "X"), ("kept/high", "X"), ("kept/high", "Y"), ("kept/high", "Z"))
LAMS = (1.0, 1.5, 2.0, 2.5, 3.0)
DIRECTIONS = ("model", "isotropic", "adverse")
DESIGN_LAMBDA = 2.5
W_DENOMINATOR = 200
QUALIFY_TOL, WIN_MARGIN, QUALIFY_FLOOR = 0.01, 0.05, 0.5


# the witness bound and the package constants live in ad_witness, which the run's analysis owns
WitnessRefused = A.WitnessRefused
TERMS, m_matrix, lambda_max = A.TERMS, A.m_matrix, A.lambda_max
positive_definite, certify_bound, lam_ideal = A.positive_definite, A.certify_bound, A.lam_ideal


def basis_weights(m: int = 8) -> dict:
    """{(group, basis): exact W_{g,b}} from the committed record (R224's law)"""
    rec = json.loads(P.WEIGHTS.read_text(encoding="utf-8"))[P.weights_key(m, "both")]
    if "W_exact_by_basis" not in rec:
        raise WitnessRefused("the committed weights have no per-basis split: run ion_run_package.py weights "
                             f"{m} --erasure-orders both")
    return {(g, b): Fraction(*x) for g, v in rec["W_exact_by_basis"].items() for b, x in v.items()}


def witness_test(w, pi_rare: Fraction, m: int = 8) -> dict:
    """the betting test of witness w on the EXACT design law's group weights: kappa per (group, basis) rounded UP,
    m_w rounded DOWN (both conservative; ad_witness.constants)"""
    w = [Fraction(x) for x in w]
    W = basis_weights(m)
    group = {g: sum(x for (gg, _), x in W.items() if gg == g) for g in ("ee_x/high", "kept/high")}
    hbar = certify_bound(w)
    c = A.constants(w, hbar, group, pi_rare, m)
    return dict(w=[str(x) for x in w], hbar=str(hbar), m_exact=str(c["m_exact"]), m=c["m"], kappa=c["kappa"])


def fe_test(design: dict, pi_rare: float) -> dict:
    """R203's F_e test in the same form (its own constants, from ion_betting)"""
    a = B.scores(design, pi_rare)
    return dict(m=B.threshold(design)["m"], kappa={"ee_x/high|X": a[B.RARE], **{f"kept/high|{b}": a[B.COMMON]
                                                                              for b in BASES}})


# ------------------------------------------------------------------ the device
def device_rates(dev: str = "helios_1", arm: str = "exec") -> dict:
    """R195's executed projection per (group, basis): {cat: mean error probability}, and the group means the F_e
    operating characteristics use (r200_ion_robustness.projected_group_rates: the same file, the same units)"""
    design = json.loads((PROJ / "design.json").read_text(encoding="utf-8"))
    proj = json.loads((PROJ / f"executed_projection_{dev}.json").read_text(encoding="utf-8"))
    acc: dict = {}
    for u, e in zip(design["units"], proj["per_unit"][f"e_{arm}"]):
        acc.setdefault(f"{u['group']}|{u['inp'][0]}", []).append(float(e))
        acc.setdefault(u["group"], []).append(float(e))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def shot_probs(rates: dict, direction: str, lam: float, pi_rare: float, m: int = 8) -> dict:
    """{cat: P(a shot is labelled g, draws basis b, and errs)}: kept's group mean fixed at the projection's, its basis
    profile by direction"""
    W = basis_weights(m)
    share = {b: float(W[("kept/high", b)] / sum(W[("kept/high", c)] for c in BASES)) for b in BASES}
    mean = rates["kept/high"]
    if direction == "model":
        prof = {b: rates[f"kept/high|{b}"] for b in BASES}
    elif direction == "isotropic":
        prof = {b: 1.0 for b in BASES}
    elif direction == "adverse":
        prof = {"X": 1.0, "Y": 1.0, "Z": 0.0}
    else:
        raise WitnessRefused(f"direction {direction!r}")
    norm = sum(share[b] * prof[b] for b in BASES)
    out = {"ee_x/high|X": pi_rare * min(1.0, lam * rates["ee_x/high"])}
    for b in BASES:
        out[f"kept/high|{b}"] = (1 - pi_rare) * share[b] * min(1.0, lam * mean * prof[b] / norm)
    return out


# ------------------------------------------------------------------ growth, bets, w*
def growth(test: dict, probs: dict, b: float) -> float:
    """E log(1 + b (m - S)) per shot"""
    m = test["m"]
    p0 = 1 - sum(probs.values())
    return p0 * math.log1p(b * m) + sum(p * math.log1p(b * (m - test["kappa"][c])) for c, p in probs.items())


def bet_cap(test: dict) -> float:
    return 1 / (max(test["kappa"].values()) - test["m"])


def kelly(test: dict, probs: dict) -> tuple:
    from scipy.optimize import minimize_scalar
    cap = bet_cap(test)
    r = minimize_scalar(lambda b: -growth(test, probs, b), bounds=(1e-9, cap * (1 - 1e-9)), method="bounded",
                        options=dict(xatol=1e-10))
    return float(r.x), float(-r.fun)


def best_witness(rates: dict, pi_rare: float, lam: float = DESIGN_LAMBDA, direction: str = "model") -> list:
    """w* (rule 1): the Kelly growth rate's maximizer over w >= 0, sum w = 1/2, as Fractions over W_DENOMINATOR"""
    from scipy.optimize import minimize
    probs = shot_probs(rates, direction, lam, pi_rare)

    def neg(x):
        w = np.abs(x) / np.abs(x).sum() / 2
        test = _float_test(w, pi_rare)
        return -kelly(test, probs)[1] if test["m"] > 0 else 1.0
    best = min((minimize(neg, np.array(s, float), method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-12,
                                                                                    maxiter=2000))
                for s in ([1, 1, 1], [1, .4, .6], [1, .2, .3], [.6, .3, 1])), key=lambda r: r.fun)
    x = np.abs(best.x) / np.abs(best.x).sum() / 2
    num = [int(round(v * W_DENOMINATOR)) for v in x]
    num[int(np.argmax(num))] += W_DENOMINATOR // 2 - sum(num)          # keep sum w = 1/2 exactly
    return [Fraction(n, W_DENOMINATOR) for n in num]


def _float_test(w, pi_rare: float, m: int = 8) -> dict:
    """the witness test in floats (the optimizer's inner loop; the frozen w* gets witness_test's exact constants)"""
    W = basis_weights(m)
    group = {g: float(sum(x for (gg, _), x in W.items() if gg == g)) for g in ("ee_x/high", "kept/high")}
    pi = {"ee_x/high": pi_rare, "kept/high": 1 - pi_rare}
    width = {"ee_x/high": 1.0, "kept/high": 2.0}
    lam = [float(x) for x in lam_ideal(m)]
    return dict(m=float(np.dot(w, lam)) - lambda_max(w),
                kappa={f"{g}|{b}": 6 * w[BASES.index(b)] * width[g] * group[g] / pi[g] for g, b in CATS})


def transferred_bets(fe: dict, wit: dict, rates: dict, pi_rare: float, bets=Q.BETS) -> dict:
    """rule 2: for each of F_e's bets, the lambda at which it is F_e's Kelly bet ('model'), and the witness's Kelly
    bet there"""
    from scipy.optimize import brentq
    out = []
    for b in bets:
        f = lambda lam: kelly(fe, shot_probs(rates, "model", lam, pi_rare))[0] - b  # noqa: E731
        lam = brentq(f, 0.05, 6.0, xtol=1e-10)
        out.append(dict(fe_bet=b, at_lambda=lam, witness_bet=kelly(wit, shot_probs(rates, "model", lam, pi_rare))[0]))
    return dict(rule="each F_e bet's Kelly lambda ('model'); the witness's Kelly bet there", rows=out)


# ------------------------------------------------------------------ operating characteristics
def oc_mc(tests: dict, probs: dict, looks: list, costs: list, delta: float, reps: int, seed: int,
          chunk: int = 50_000) -> dict:
    """P(clears at each look) and E[HQC] for each test, from the same simulated per-look counts (common random
    numbers). tests: {name: [(m, kappa, b), ...]} -- a test's wealth is the mean over its components"""
    cats = list(probs)
    pv = [probs[c] for c in cats] + [1 - sum(probs.values())]
    if min(pv) < 0:
        raise WitnessRefused(f"shot probabilities {pv} are not a distribution")
    bar = math.log(1 / delta)
    cum = np.cumsum(costs)
    stops = {t: np.zeros(len(looks)) for t in tests}
    rng = np.random.default_rng(seed)
    done = 0
    while done < reps:
        n = min(chunk, reps - done)
        k = np.zeros((n, len(cats)), np.int64)
        alive = {t: np.ones(n, bool) for t in tests}
        prev = 0
        for li, L in enumerate(looks):
            k += rng.multinomial(L - prev, pv, size=n)[:, :len(cats)]
            prev = L
            n0 = L - k.sum(axis=1)
            for t, comps in tests.items():
                lw = np.stack([n0 * math.log1p(b * m) + sum(k[:, j] * math.log1p(b * (m - kap[c]))
                                                             for j, c in enumerate(cats)) for m, kap, b in comps])
                top = lw.max(axis=0)
                hit = (top + np.log(np.mean(np.exp(lw - top), axis=0)) >= bar) & alive[t]
                stops[t][li] += hit.sum()
                alive[t] &= ~hit
        done += n
    out = {}
    for t, s in stops.items():
        p = s / reps
        never = 1 - p.sum()
        out[t] = dict(clears_by_look=[round(float(x), 6) for x in p], holds=round(float(p.sum()), 6),
                      se=round(float(math.sqrt(max(p.sum() * (1 - p.sum()), 1e-12) / reps)), 6),
                      expected_hqc=round(float((p * cum).sum() + never * cum[-1]), 1))
    return out


def components(test: dict, bets) -> list:
    return [(test["m"], test["kappa"], float(b)) for b in bets]


def decide(oc: dict) -> dict:
    """rules 3 and 4 on oc[direction][lambda][test]"""
    verdicts = {}
    for cand in ("witness", "mixture"):
        worst = []
        for d in DIRECTIONS:
            for lam in LAMS:
                row = oc[d][str(lam)]
                if row["F_e"]["holds"] >= QUALIFY_FLOOR:
                    worst.append((row[cand]["holds"] - row["F_e"]["holds"], d, lam))
        low = min(worst)
        gain = oc["model"][str(DESIGN_LAMBDA)][cand]["holds"] - oc["model"][str(DESIGN_LAMBDA)]["F_e"]["holds"]
        verdicts[cand] = dict(qualifies=bool(low[0] >= -QUALIFY_TOL), worst_shortfall=dict(
            delta_holds=round(low[0], 6), direction=low[1], lam=low[2]), gain_at_design_lambda=round(gain, 6))
    ok = [c for c in verdicts if verdicts[c]["qualifies"] and verdicts[c]["gain_at_design_lambda"] >= WIN_MARGIN]
    choice = max(ok, key=lambda c: verdicts[c]["gain_at_design_lambda"]) if ok else "F_e"
    return dict(choice=choice, candidates=verdicts, rule=dict(qualify_tol=QUALIFY_TOL, qualify_floor=QUALIFY_FLOOR,
                                                              win_margin=WIN_MARGIN, design_lambda=DESIGN_LAMBDA))


def run(reps: int = 200_000, seed: int = 225) -> dict:
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((PKG / "sequential.json").read_text(encoding="utf-8"))
    Q.check_betting_package(design, seq)
    pi = float(seq["pi_rare"])
    pi_exact = Fraction(seq["pi_rare"]).limit_denominator(10**6)
    looks = list(seq["looks"])
    costs, cost_src = R3.batch_costs(looks)
    rates = device_rates()
    fe = fe_test(design, pi)
    w_star = best_witness(rates, pi)
    wit = witness_test(w_star, pi_exact)
    bets_w = transferred_bets(fe, wit, rates, pi)
    wb = [round(r["witness_bet"], 3) for r in bets_w["rows"]]
    for b in wb:
        if not 0 < b < bet_cap(wit):
            raise WitnessRefused(f"witness bet {b} is outside (0, {bet_cap(wit)})")
    tests = {"F_e": components(fe, Q.BETS), "witness": components(wit, wb),
             "mixture": components(fe, Q.BETS) + components(wit, wb)}
    oc: dict = {}
    for d in DIRECTIONS:
        oc[d] = {}
        for lam in LAMS:
            probs = shot_probs(rates, d, lam, pi)
            row = oc_mc(tests, probs, looks, costs, Q.CLAIM_DELTA, reps, seed + int(lam * 100) + 7 * DIRECTIONS.index(d))
            for name, comps in (("F_e", tests["F_e"]), ("witness", tests["witness"])):
                row[name]["drift_per_shot"] = round(sum(p * kap for p, kap in (
                    (probs[c], comps[0][1][c]) for c in probs)) - comps[0][0], 8)       # mean S - m: < 0 can clear
            oc[d][str(lam)] = row
    cross = {}
    for lam in LAMS:
        ex = R3.oc_exact(design, pi, Q.BETS, looks, costs, lam, {B.RARE: rates["ee_x/high"],
                                                                  B.COMMON: rates["kept/high"]}, Q.CLAIM_DELTA)
        mc = oc["model"][str(lam)]["F_e"]
        cross[str(lam)] = dict(exact_holds=ex["holds"], mc_holds=mc["holds"], mc_se=mc["se"],
                               z=round((mc["holds"] - ex["holds"]) / max(mc["se"], 1e-6), 2),
                               exact_hqc=ex["expected_hqc"], mc_hqc=mc["expected_hqc"])
    sens = {str(lam): [str(x) for x in best_witness(rates, pi, lam=lam)] for lam in (1.0, 2.0, 3.0)}
    return dict(schema="oneq.r225_witness_choice/1", package=PKG.relative_to(ROOT).as_posix() if PKG.is_relative_to(ROOT) else str(PKG),
                pi_rare=pi, looks=looks, batch_hqc=costs, batch_hqc_source=cost_src, claim_delta=Q.CLAIM_DELTA,
                rates=rates, rates_source="R195 executed projection, Helios, exec arm, per (group, input basis)",
                lam_ideal=[str(x) for x in lam_ideal()], F_e=dict(fe, bets=list(Q.BETS)),
                witness=dict(wit, bets=wb, bet_transfer=bets_w, w_float=[float(x) for x in w_star]),
                w_star_sensitivity=sens, reps=reps, seed=seed, oc=oc, fe_mc_vs_exact=cross, decision=decide(oc),
                caveats=["SIMULATED device rates (R195's projection) scaled by lambda; directions are scenarios",
                         "no futility stop in any arm",
                         "Monte Carlo over per-look counts: holds carries its standard error se"])


def main(argv) -> int:
    reps = int(argv[0]) if argv else 200_000
    out = run(reps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"w* {out['witness']['w']} (sum 1/2)  hbar {out['witness']['hbar']}  m_w {out['witness']['m']:.6f}  "
          f"bets {[round(b, 4) for b in out['witness']['bets']]}")
    for d, rows in out["oc"].items():
        for lam, row in rows.items():
            print(f"{d:9s} x{lam:4s} " + "  ".join(f"{t} {v['holds']:.4f}/{v['expected_hqc']:7.1f}"
                                                    for t, v in row.items()))
    print("F_e MC vs exact:", {k: v["z"] for k, v in out["fe_mc_vs_exact"].items()})
    print("decision:", json.dumps(out["decision"]))
    print("->", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
