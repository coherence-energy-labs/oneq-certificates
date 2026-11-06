"""R186 part 3: the certification design for the channel-faithful ion run -- how many DISTINCT circuits, and shots
each, a one-sided claim F_e > 3/4 needs. Computed from R186's own unit matrices (r186_faithful_requirements.py `run`
writes them to evidence/superactivation/r186_faithful_requirements/runs/; they are not committed, ~390 MB).

Why it matters: R144 named the number of distinct circuits as the binding resource of a claim over a SAMPLED
mixed-unitary design (P_d's random gates), and R186 part 2's shot budgets assumed a fresh circuit per shot.

The estimator is R186's control variate. Stratum b has weight W_b; each unit's ideal score ideal_u is EXACT; a shot of
unit u scores S. Then f_avg = f_ideal - mean_b,u (ideal_u - S), and F_e = 1.5 f_avg - 0.5, so the bar on f_avg is 5/6.
Unit-to-unit spread enters only through the LOSS ideal_u - E[S|u].

  gauss      Var(F_hat) = 2.25 / N * sum_b W_b (Var_u(loss_u) + mean_u Var(S|u) / s), with N distinct units of s shots
             each, allocated in proportion to W_b; N for a 3-sigma ASYMPTOTIC claim. Not a certificate.
  betting    a RIGOROUS, distribution-free, anytime-valid one-sided bound: the hedged-capital betting confidence
             bound for a bounded mean (Waudby-Smith & Ramdas 2024), at delta = 0.00135, on iid unit draws (stratum by
             W_b, unit uniform within it, the CV estimate averaged over its s shots). Simulated on the model's own unit
             laws: the fraction of experiments whose bound clears 3/4.
  raw        the same bound without the control variate (Y = S), for comparison.
  binomial   THE RECOMMENDED DESIGN (2026-10-19): every unit's ideal outcome is deterministic, so a shot is its ideal
             outcome or an ERROR; per (kind, sign of the ideal) group a fresh-unit single-shot draw's error indicator is
             EXACTLY Bernoulli, and an exact Clopper-Pearson bound needs no range or unit-sampling term. Only the groups
             whose errors LOWER f are sampled (the others are bounded below by 0, valid on every device); the constant
             both-erased Z/Y units need no shots. About HALF the betting design's circuits for the same pass rate.

  python r186_certification.py gauss                     the asymptotic table (fast)
  python r186_certification.py betting [REPS]            the rigorous betting table (minutes)
  python r186_certification.py binomial [REPS]           the error-indicator table (minutes)
  python r186_certification.py all [REPS]                all -> evidence/superactivation/r186_faithful_requirements/certification.json
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

RUNS = ROOT / "evidence" / "superactivation" / "r186_faithful_requirements" / "runs"
OUT = ROOT / "evidence" / "superactivation" / "r186_faithful_requirements" / "certification.json"
DELTA = 0.00135
BAR_F = 5 / 6                                   # f_avg at F_e = 3/4
DEVICES = ("spec_h2_2_m6_mix_standard", "spec_helios_1_m6_mix_standard", "spec_h2_2_m5_mix_standard",
           "spec_helios_1_m5_mix_standard")


class CertRefused(ValueError):
    """an input the design cannot be computed from"""


def unit_law(name: str) -> dict:
    """the run's per-unit exact E[S] (full noise), exact ideal, kinds and strata"""
    import hw_two_use as H
    import r186_faithful_requirements as R
    if not (RUNS / f"{name}.npz").exists():
        raise CertRefused(f"{name}: no unit matrix in {RUNS} (r186_faithful_requirements.py run writes it)")
    run = R.load_run(RUNS / name)
    V, keys, W = R._matrix(run)
    ix = R.index_of(run["configs"])
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    strata = [(float(W[idx[0]]), np.array(idx)) for idx in groups.values()]
    return dict(es=V[:, ix["full"]], ideal=V[:, ix["ideal"]], kinds=[r["kind"] for r in run["rows"]], strata=strata,
                f_ideal=(2 * H.f_e_closed(run["m"]) + 1) / 3, m=run["m"], units=len(run["rows"]))


SUPPORT = {"kept": (0.0, 2.0), "ee_x": (0.0, 1.0)}         # every other kind: S = +-1/2
RANGE_PAD = 1e-9


def _support(kind: str) -> tuple:
    return SUPPORT.get(kind, (-0.5, 0.5))


def cv_range(law: dict) -> tuple:
    """the range of the CV draw y = f_ideal - (ideal_u - S), DERIVED unit by unit from each unit's score support and
    its exact ideal, so every draw lies in it by construction (an assumed [f_ideal - 2, f_ideal + 2] was violated by
    the real H2-2 law: found by betting_lcb's own range refusal, 2026-10-19)"""
    lo = min(law["f_ideal"] - law["ideal"][i] + _support(k)[0] for i, k in enumerate(law["kinds"]))
    hi = max(law["f_ideal"] - law["ideal"][i] + _support(k)[1] for i, k in enumerate(law["kinds"]))
    # widened by RANGE_PAD: a draw is formed as f - (ideal - S), the bound as f - ideal + S_max, and the two round
    # differently by ~1e-16 at the extreme unit; a wider declared range only makes the bound more conservative
    return float(lo) - RANGE_PAD, float(hi) + RANGE_PAD


def raw_range(law: dict) -> tuple:
    kinds = set(law["kinds"])
    return min(_support(k)[0] for k in kinds), max(_support(k)[1] for k in kinds)


def _second_moment(kind: str, e: float) -> float:
    """E[S^2] from the score's support (as r186_faithful_requirements.shot_sd): kept S in {0, 2}; a both-erased X input
    S in {0, 1}; a both-erased Z/Y input S = +-1/2"""
    return 2 * e if kind == "kept" else (e if kind == "ee_x" else 0.25)


def components(law: dict) -> dict:
    """the W-weighted per-unit variances: between units (loss, and raw for comparison) and within a unit per shot"""
    bl = br = wi = 0.0
    for w, idx in law["strata"]:
        es = law["es"][idx]
        loss = law["ideal"][idx] - es
        wi += w * float(np.mean([_second_moment(law["kinds"][i], e) for i, e in zip(idx, es)] - es ** 2))
        if len(idx) > 1:
            bl += w * float(loss.var(ddof=1))
            br += w * float(es.var(ddof=1))
    return dict(between_loss=bl, between_raw=br, within=wi)


def gauss_units(c: dict, margin: float, s: float) -> int:
    v = c["between_loss"] + c["within"] / s
    return math.ceil(9 * 2.25 * v / margin ** 2)


def sample_scores(law: dict, u: np.ndarray, s: int, rng) -> np.ndarray:
    """each unit's mean score over s shots"""
    out = np.empty(len(u))
    for i, j in enumerate(u):
        k, e = law["kinds"][j], law["es"][j]
        if k == "kept":
            out[i] = 2.0 * (rng.random(s) < e / 2).mean()
        elif k == "ee_x":
            out[i] = (rng.random(s) < e).mean()
        else:
            out[i] = np.where(rng.random(s) < e + 0.5, 0.5, -0.5).mean()
    return out


def draw_units(law: dict, n: int, rng) -> np.ndarray:
    w = np.array([st[0] for st in law["strata"]])
    b = rng.choice(len(w), size=n, p=w / w.sum())
    return np.array([law["strata"][k][1][rng.integers(len(law["strata"][k][1]))] for k in b])


def betting_lcb(x: np.ndarray, lo: float, hi: float, window: tuple, delta: float = DELTA, grid: int = 400,
                cap: float = 0.5) -> float:
    """the hedged-capital lower confidence bound for the mean of iid x in [lo, hi] (positive bets, predictable plug-in
    sizes, capped at cap / m so wealth stays positive), searched on a fine grid over `window` (x's units). Wealth is
    monotone in m for positive bets, so the bound is the first m in the window not rejected; window[0] when none is.
    `cap` < 1 keeps every factor 1 + lambda (z - m) >= 1 - cap > 0, so validity holds for any cap in (0, 1); a
    higher cap matters when the variance is tiny and every bet sits at the cap (the clustered design, 2026-10-19)"""
    if not 0 < cap < 1:
        raise CertRefused(f"cap {cap} outside (0, 1): wealth could reach zero")
    if not (lo <= x.min() and x.max() <= hi):
        raise CertRefused(f"draws outside the declared range [{lo}, {hi}]: the bound would be invalid")
    z = (x - lo) / (hi - lo)
    n = len(z)
    ms = np.linspace((window[0] - lo) / (hi - lo), (window[1] - lo) / (hi - lo), grid)
    t = 1 + np.arange(n)
    mu = (0.5 + np.concatenate([[0], np.cumsum(z)[:-1]])) / t
    s2 = (0.25 + np.concatenate([[0], np.cumsum((z - mu) ** 2)[:-1]])) / t
    lam = np.sqrt(2 * math.log(1 / delta) / (s2 * t * np.log(t + 1)))
    logK = np.zeros(len(ms))
    for i in range(n):
        logK += np.log1p(np.minimum(lam[i], cap / ms) * (z[i] - ms))
    ok = logK < math.log(1 / delta)
    if ok.all():
        return window[0]
    return lo + float(ms[ok][0] if ok.any() else ms[-1]) * (hi - lo)


# ------------------------------------------------------------------ the ERROR-INDICATOR certificate (2026-10-19)
# Every unit of the faithful design has a DETERMINISTIC ideal outcome: its ideal score is the top or the bottom of its
# support (r186 unit laws: 2,048 of 2,048 units), and a both-erased Z/Y unit scores exactly 1/2 whatever happens. So a
# shot is either its known ideal outcome or an ERROR, and with e_u the device's error probability for unit u:
#     f = f_ideal - sum_{high units} w_u width_u e_u + sum_{low units} w_u width_u e_u        (constants: exact)
# For one group g (one kind, one sign of the ideal), a draw = a FRESH unit u ~ w restricted to g plus one shot has an
# error indicator that is EXACTLY Bernoulli(ebar_g) (a mixture of Bernoullis is Bernoulli), whatever the device does:
# an exact one-sided Clopper-Pearson bound, no range term, no unit-sampling term. The low groups' errors only RAISE f
# (e >= 0 on every device), so bounding them below by 0 -- not sampling them -- is valid and spends every shot where
# the risk is. Union bound over the sampled groups.
CONST_KIND = "ee_const"


def error_groups(law: dict) -> tuple:
    """{(kind, 'high' | 'low'): dict(width, W, p (unit draw law), e (the model's error rates), ebar)}, the constant
    units' weight, and the model's f. Refused when a unit's ideal outcome is not deterministic"""
    w_unit = np.zeros(len(law["es"]))
    for w, idx in law["strata"]:
        w_unit[idx] = w / len(idx)
    out: dict = {}
    const_w = 0.0
    for i, k in enumerate(law["kinds"]):
        if k == CONST_KIND:
            const_w += w_unit[i]
            continue
        lo, hi = _support(k)
        width = hi - lo
        p_hi = (law["ideal"][i] - lo) / width
        if abs(p_hi - 1) < 1e-9:
            sign, e = "high", (law["ideal"][i] - law["es"][i]) / width
        elif abs(p_hi) < 1e-9:
            sign, e = "low", (law["es"][i] - law["ideal"][i]) / width
        else:
            raise CertRefused(f"unit {i} ({k}): its ideal outcome is not deterministic (P(top) = {p_hi:.6g}), so a "
                              "shot is not an error indicator")
        if e < -1e-12:
            raise CertRefused(f"unit {i} ({k}): a negative error rate {e:.3g}")
        g = out.setdefault((k, sign), dict(width=width, w=[], e=[]))
        g["w"].append(w_unit[i])
        g["e"].append(max(e, 0.0))
    for g in out.values():
        g["W"] = float(np.sum(g["w"]))
        g["p"] = np.array(g["w"]) / g["W"]
        g["e"] = np.array(g["e"])
        g["ebar"] = float(g["p"] @ g["e"])
    f = law["f_ideal"] - sum(g["width"] * g["W"] * g["ebar"] * (1 if s == "high" else -1) for (_, s), g in out.items())
    return out, const_w, float(f)


def cp_upper(k: int, n: int, a: float) -> float:
    """the exact one-sided (1 - a) Clopper-Pearson upper bound on a binomial rate"""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))


# ------------------------------------------------------------------ TWO UNITS PER CIRCUIT: the exact pair bound (R189)
# Where a device cannot run a different unit per shot (R189's parameterized shots), two fresh units of one group can
# share a circuit (two 28-qubit units fill H2-2's 56 qubits), halving the per-program charge. The two slots of a shot
# may then err TOGETHER (shared calibration, crosstalk), so the 2N indicators are not independent and Clopper-Pearson
# on 2N is not exact. Per circuit, Y = (e1 + e2) / 2 lies in {0, 1/2, 1} with some law q = (q0, qh, q1), and EVERY
# joint law of the two slots is one such q; mu = qh / 2 + q1 is the per-slot error rate. Buehler's construction on the
# statistic T = n_half + K n_one (K > 2 penalizes double errors) is exact for any q: the set {q : P_q(T <= t_obs) > a}
# covers the truth with probability >= 1 - a, and the bound is the largest mu in it. P_q(T <= t) falls in qh and in q1
# (both move mass up), so qh_max(q1) is non-increasing and, on a cell [qa, qb], mu <= qh_max(qa) / 2 + qb: the cell
# refinement below never returns less than the true supremum. With no double errors it is the pooled 2N bound to
# within ~0.5%; with no errors at all it is the perfectly correlated N bound (the data cannot tell them apart).
PAIR_K = 4


def _bpmf(n, N, p):
    from scipy.special import gammaln
    if p <= 0:
        return (n == 0).astype(float)
    return np.exp(gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1) + n * math.log(p) + (N - n) * math.log1p(-p))


def pair_cdf(t: int, N: int, qh: float, q1: float, K: int = PAIR_K) -> float:
    """P(n_half + K n_one <= t) under the trinomial (N; qh, q1)"""
    from scipy.special import bdtr
    if q1 <= 0:
        return float(bdtr(t, N, qh)) if qh < 1 else float(t >= N)
    n1 = np.arange(0, min(N, t // K) + 1)
    cond = min(max(qh / (1 - q1), 0.0), 1.0)
    kk, rest = t - K * n1, N - n1
    c = (kk >= rest).astype(float) if cond >= 1 else bdtr(kk, rest, cond)
    return float(np.sum(_bpmf(n1, N, q1) * c))


def _pair_cdf_grid(t: int, N: int, qhs: np.ndarray, q1: float, K: int) -> np.ndarray:
    """pair_cdf at every qh of a grid, in one vectorized evaluation"""
    from scipy.special import bdtr
    if q1 <= 0:
        return np.where(qhs < 1, bdtr(t, N, np.minimum(qhs, 1.0)), float(t >= N))
    n1 = np.arange(0, min(N, t // K) + 1)
    cond = np.clip(qhs / (1 - q1), 0.0, 1.0)
    kk, rest = (t - K * n1)[:, None], (N - n1)[:, None]
    c = np.where(cond[None, :] >= 1, (kk >= rest).astype(float), bdtr(kk, rest, np.minimum(cond, 1.0)[None, :]))
    return (_bpmf(n1, N, q1)[:, None] * c).sum(axis=0)


def _qh_max(t: int, N: int, q1: float, a: float, K: int, rounds: int = 5, grid: int = 65) -> float:
    """an upper estimate (the rejected end of the final bracket) of the largest qh with pair_cdf > a; -1 if none.
    pair_cdf falls in qh, so the accepted set is an interval [0, qh*]; five rounds of a 65-point grid bracket qh* to
    (1/64)^5 ~ 1e-9 of the range"""
    if pair_cdf(t, N, 0.0, q1, K) <= a:
        return -1.0
    lo, hi = 0.0, 1.0 - q1
    if pair_cdf(t, N, hi, q1, K) > a:
        return hi
    for _ in range(rounds):
        g = np.linspace(lo, hi, grid)
        ok = _pair_cdf_grid(t, N, g, q1, K) > a
        j = int(np.flatnonzero(ok)[-1]) if ok.any() else 0
        if j == grid - 1:
            return float(hi)
        lo, hi = float(g[j]), float(g[j + 1])
    return hi


def pair_ucb(n_half: int, n_one: int, N: int, a: float, K: int = PAIR_K, rel_tol: float = 2e-3,
             max_splits: int = 400) -> float:
    """the exact (1 - a) upper bound on the per-slot error rate from N two-slot circuits, n_half with one error and
    n_one with two, valid under ANY dependence between the slots. The returned value is always the largest PROVEN
    cell bound, so stopping early only loosens it: it stops once that is within rel_tol of a point in the confidence
    set (the gap closes ~1/splits, the objective being nearly flat along the set's boundary), or after max_splits"""
    if not (0 <= n_half and 0 <= n_one and n_half + n_one <= N and 0 < a < 1 and K > 2):
        raise CertRefused(f"pair counts ({n_half}, {n_one}) of {N} at a = {a}, K = {K}")
    t = n_half + K * n_one
    q1_hi = cp_upper(t // K, N, a) if t // K < N else 1.0       # pair_cdf <= P(n_one <= t / K)
    q1_hi = min(1.0, q1_hi * (1 + 1e-9) + 1e-12)
    cache: dict = {}

    def m(q):
        if q not in cache:
            cache[q] = _qh_max(t, N, q, a, K)
        return cache[q]

    # branch and bound: always split the cell with the largest PROVEN upper value; the answer is that value once it is
    # within tol of a point actually in the confidence set, so it is never below the true supremum
    import heapq
    edges = np.linspace(0.0, q1_hi, 17)
    heap, achieved = [], 0.0
    for qa, qb in zip(edges[:-1], edges[1:]):
        ma = m(qa)
        if ma >= 0:
            heapq.heappush(heap, (-(ma / 2 + qb), qa, qb))
            achieved = max(achieved, ma / 2 + qa)
    for _ in range(max_splits):
        top, qa, qb = heap[0]
        if -top - achieved <= rel_tol * achieved:
            break
        heapq.heappop(heap)
        mid = 0.5 * (qa + qb)
        for lo_, hi_ in ((qa, mid), (mid, qb)):
            ml = m(lo_)
            if ml >= 0:
                heapq.heappush(heap, (-(ml / 2 + hi_), lo_, hi_))
                achieved = max(achieved, ml / 2 + lo_)
    return min(-heap[0][0], 1.0)


def binomial_lower_f(law_groups: dict, f_ideal: float, counts: dict, delta: float = DELTA) -> float:
    """f's lower bound from {group: (errors, draws)} over the HIGH groups sampled (union bound); the low groups at 0"""
    a = delta / len(counts)
    return f_ideal - sum(law_groups[g]["width"] * law_groups[g]["W"] * cp_upper(k, n, a) for g, (k, n) in counts.items())


def binomial_allocation(law_groups: dict, N: int) -> dict:
    """N single-shot draws over the HIGH groups, in proportion to width_g W_g sqrt(ebar_g) (Neyman, from the model)"""
    high = {g: v for g, v in law_groups.items() if g[1] == "high"}
    score = {g: v["width"] * v["W"] * math.sqrt(max(v["ebar"], 1e-6)) for g, v in high.items()}
    tot = sum(score.values())
    return {g: max(1, int(round(N * s / tot))) for g, s in score.items()}


def binomial_table(law: dict, totals: tuple, reps: int, rng) -> list:
    G, const_w, f_true = error_groups(law)
    rows = []
    for N in totals:
        alloc = binomial_allocation(G, N)
        ok, lows = 0, []
        for _ in range(reps):
            counts = {}
            for g, n in alloc.items():
                u = rng.choice(len(G[g]["e"]), size=n, p=G[g]["p"])
                counts[g] = (int((rng.random(n) < G[g]["e"][u]).sum()), n)
            f_lo = binomial_lower_f(G, law["f_ideal"], counts)
            lows.append(1.5 * f_lo - 0.5)
            ok += f_lo > BAR_F
        rows.append(dict(single_shot_distinct_circuits=N, allocation={f"{k}/{s}": n for (k, s), n in alloc.items()},
                         pass_rate=ok / reps, certified_F_e_median=float(np.median(lows)), reps=reps))
        print(f"   [error-indicator] {N:5d} single-shot circuits {rows[-1]['allocation']}: certified F_e median "
              f"{np.median(lows):.4f}, clears 3/4 in {ok}/{reps}", flush=True)
    return rows


def betting_table(law: dict, plans: tuple, reps: int, rng) -> list:
    win = ((0.70 + 0.5) / 1.5, (0.80 + 0.5) / 1.5)
    clo, chi = cv_range(law)
    rlo, rhi = raw_range(law)
    rows = []
    for s, n in plans:
        cv_ok, raw_ok, cv_l, raw_l = 0, 0, [], []
        for _ in range(reps):
            u = draw_units(law, n, rng)
            sb = sample_scores(law, u, s, rng)
            y = law["f_ideal"] - (law["ideal"][u] - sb)
            lc = betting_lcb(y, clo, chi, win)
            lr = betting_lcb(sb, rlo, rhi, ((0.5 + 0.5) / 1.5, win[1]))
            cv_ok += lc > BAR_F
            raw_ok += lr > BAR_F
            cv_l.append(1.5 * lc - 0.5)
            raw_l.append(1.5 * lr - 0.5)
        rows.append(dict(shots_per_unit=s, units=n, shots=s * n, cv_pass=cv_ok / reps,
                         cv_lcb_median=float(np.median(cv_l)), raw_pass=raw_ok / reps,
                         raw_lcb_median=float(np.median(raw_l)), reps=reps))
        print(f"   s={s:4d} units={n:5d} (shots {s * n:6d}): CV bound median {np.median(cv_l):.4f} "
              f"(clears 3/4 in {cv_ok}/{reps}); raw {np.median(raw_l):.4f} ({raw_ok}/{reps})", flush=True)
    return rows


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gauss"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    if cmd not in ("gauss", "betting", "binomial", "all"):
        print(__doc__)
        return 2
    rng = np.random.default_rng(186)
    out: dict = dict(delta=DELTA, bar_F_e=0.75, devices={})
    for name in DEVICES:
        law = unit_law(name)
        c = components(law)
        import r186_faithful_requirements as R
        F = float(R.analyse(R.load_run(RUNS / name), boot=2)["F_e"])
        margin = F - 0.75
        rec: dict = dict(F_e=F, margin=margin, m=law["m"], units_in_model=law["units"], **c,
                         cv_variance_reduction=c["between_raw"] / c["between_loss"], cv_range=cv_range(law),
                         raw_range=raw_range(law))
        rec["gauss_units"] = {str(s): (gauss_units(c, margin, s) if margin > 0 else None)
                              for s in (1, 10, 100, 1000)}
        print(f"== {name}: F_e {F:.4f} (margin {margin:+.4f}); between-unit variance raw {c['between_raw']:.2e} -> "
              f"loss {c['between_loss']:.2e} ({rec['cv_variance_reduction']:.0f}x); within {c['within']:.2e}")
        print("   gauss 3-sigma units at s = 1/10/100/1000: " + ", ".join(f"{v}" for v in rec["gauss_units"].values()))
        if cmd in ("betting", "all") and law["m"] == 6:
            rec["betting"] = betting_table(law, ((1, 1000), (1, 2000), (1, 3000), (10, 600), (10, 1000),
                                                 (10, 1500), (100, 1000)), reps, rng)
        if cmd in ("binomial", "all") and law["m"] == 6:
            G, const_w, f_model = error_groups(law)
            rec["error_groups"] = {f"{k}/{s}": dict(W=g["W"], width=g["width"], ebar=g["ebar"], units=len(g["e"]))
                                   for (k, s), g in G.items()}
            rec["constant_weight_no_shots"] = const_w
            rec["binomial"] = binomial_table(law, (800, 1000, 1200, 1500, 2000), max(reps, 200), rng)
        out["devices"][name] = rec
    if cmd == "all":
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"-> {OUT.relative_to(ROOT).as_posix() if OUT.is_relative_to(ROOT) else str(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
