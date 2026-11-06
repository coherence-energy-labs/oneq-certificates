"""R203 -- the m = 8 ion run's certificate as a one-sided BETTING test (Ville's inequality) on iid-labelled shots.

WHY IT REPLACES THE STRATIFIED CLOPPER-PEARSON RULE (R201/R202). That rule certified sum_g c_g pbar_g, the error of
each group averaged over the shots ITS units ran in, and an adversary pass broke it twice at the level of the
schedule:
  - R201's order: the groups ran at different times, so a drifting device cleared it with every channel at F_e <= 3/4;
  - R202's block order: a device whose kept circuits never err right after an ee_x circuit (natural one-shot memory)
    puts 24% of kept shots in that state against 20% of all slots, and the false-claim probability reached 5.2 delta.
Same failure, same level, twice: the deterministic schedule cannot stand in for identically distributed shots.
The fix is one level down: draw every shot's LABEL iid, and certify with a test whose validity needs no assumption
about the device beyond non-anticipation.

THE TEST. A shot labelled g (P(g) = pi_g, iid; its unit a fresh draw from g's law) scores W = (c_g / pi_g) X, where
X = 1 iff its outcome differs from the unit's deterministic ideal and c_g = width_g W_g. Given the past,
E[W_i | past] = sum_g c_g e_g(i) = mu_i, the deficit of the DESIGN channel the device implements at shot i (its
low-group and both-erased terms only raise f). H0: mu_bar_t = (1/t) sum_{i<=t} mu_i >= m, where m is the deficit at
which F_e = 3/4 -- that is, the shot-averaged implemented design channel has F_e <= 3/4.
  K_t(b) = prod_{i<=t} (1 + b (m - W_i)) for a CONSTANT bet b, and K_t = mean_j K_t(b_j) over the fixed bets.
  The claim holds at the first look with K_t >= 1 / delta.

VALIDITY (DERIVED, checked by simulation in the tests).
  - For each b, M_t(b) = prod (1 + b (m - W_i)) / (1 + b (m - mu_i)) is a nonnegative martingale from 1. That needs
    E[W_i | past] = mu_i, which holds when the label and unit of shot i are independent of everything before it.
  - By AM-GM, prod_i (1 + b (m - mu_i)) <= (1 + b (m - mu_bar_t))^t <= 1 whenever mu_bar_t >= m. So K_t(b) <= M_t(b)
    on that event, for every b at once. (A bet that changes over time would lose this step; the bets are constant.)
  - mean_j M_t(b_j) is a nonnegative martingale from 1. Ville: P(exists t: K_t >= 1/delta and mu_bar_t >= m) <= delta.
  - So ANY stopping rule and any number of looks keep the claim's error <= delta: looks cost no statistics.
  - It holds for drift, memory of earlier circuits, and errors that depend on the device's own earlier outcomes.
  - Every bet must keep 1 + b (m - max_g c_g / pi_g) > 0 (refused otherwise).
WHAT IT CERTIFIES. mu_bar_t < m at the stopping look: the shot-averaged design channel the device implemented, each
shot's taken given the device's own history, has F_e > 3/4. F_e is linear in the channel, so some shot's conditional
design channel had F_e > 3/4.
THE PREMISE: BEHAVIOURAL NON-ANTICIPATION (corrected 2026-10-21 by the R203 adversary pass). The submission form puts
the whole frozen table in the program at compile time, so the stack always "knows" every row; what the proof needs is
behavioural:
  - shot i's outcome depends on rows after i only through shot i's own unit;
  - the shots execute in an order that does not depend on their content (table order).
The risk lives in the software stack -- the vendor's compiler and shot scheduler -- not in the qubits. A witness:
  - a scheduler that ran every kept row first under one channel and the ee_x rows after under another;
  - each channel has F_e <= 3/4, the shot-averaged one 0.7228;
  - the records are error-free, byte-identical to a perfect device's, and the test clears at look 500.
The results carry only shot, uid and bits, so execution order is a device-time obligation (ask the vendor, record it).
A commit-reveal of the labels does not help here: the table is submitted whole.
THE OTHER OBLIGATIONS, enforced by ion_sequential.verdict / errors_from_results:
  - the package's parameters are the preregistered ones, and its labels are the seeded stream;
  - exactly ONE submitted job per batch: re-running a batch until it clears is outside Ville's guarantee.
"""
from __future__ import annotations

import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RARE, COMMON = "ee_x/high", "kept/high"


class BettingRefused(ValueError):
    pass


def threshold(design: dict) -> dict:
    """the certificate's constants from the package: c_g = width_g W_g and m, the deficit at which F_e = bar_F_e"""
    g = design["groups"]
    if set(g) != {RARE, COMMON}:
        raise BettingRefused(f"the betting test is written for the groups {RARE}, {COMMON}; the design has {sorted(g)}")
    return dict(c={k: g[k]["width"] * g[k]["W"] for k in g}, m=design["f_ideal"] - (design["bar_F_e"] + 0.5) / 1.5)


def scores(design: dict, pi_rare: float) -> dict:
    """{group: the W of one error in that group} = c_g / pi_g"""
    if not 0 < pi_rare < 1:
        raise BettingRefused(f"pi_rare {pi_rare} must lie in (0, 1)")
    c = threshold(design)["c"]
    return {RARE: c[RARE] / pi_rare, COMMON: c[COMMON] / (1 - pi_rare)}


def check_bets(design: dict, pi_rare: float, bets) -> list:
    m, a = threshold(design)["m"], scores(design, pi_rare)
    bets = [float(b) for b in bets]
    bad = [b for b in bets if not (b > 0 and 1 + b * (m - max(a.values())) > 0)]
    if not bets or bad:
        raise BettingRefused(f"bets {bad or bets} break 0 < b < 1 / (max W - m) = {1 / (max(a.values()) - m):.4f}")
    return bets


def log_wealth(n: int, k_rare: int, k_common: int, design: dict, pi_rare: float, bets) -> list:
    """log K_n(b) for each bet, from the look's shot count and its error counts by group (all it depends on)"""
    if min(n, k_rare, k_common) < 0 or k_rare + k_common > n:
        raise BettingRefused(f"counts n={n}, k_rare={k_rare}, k_common={k_common} are not a run")
    m, a = threshold(design)["m"], scores(design, pi_rare)
    return [(n - k_rare - k_common) * math.log1p(b * m) + k_rare * math.log1p(b * (m - a[RARE]))
            + k_common * math.log1p(b * (m - a[COMMON])) for b in check_bets(design, pi_rare, bets)]


def log_mean_exp(xs: list) -> float:
    top = max(xs)
    return top + math.log(sum(math.exp(x - top) for x in xs) / len(xs))


SCOPE = ("the shot-averaged design channel the device implemented (each shot's given the device's own history) has "
         "F_e > 3/4, so some shot's did; valid for any behaviourally non-anticipating device -- drift, memory and "
         "outcome-dependent errors included -- under any stopping rule (Ville), provided the shots ran in an order "
         "independent of their content and each batch was submitted once")


# R209: the FUTILITY stop. A run on a device far worse than its data sheet used to spend the whole package (1,401.7
# HQC) before admitting failure. After a look that does not clear, from `min_look` shots on, the run stops when even
# an OPTIMISTIC projection gives it less than `tau` chance of ever clearing. A futility stop never claims, so Ville's
# bound -- and with it the claim's validity -- is untouched; it can only give up claims, which the exact operating
# characteristics price (r203_betting.py `oc_futility`). Chosen there on the exact DP: at 5 sigma it saves 1,110 HQC
# on a 3x device and 374 at 2.5x, and costs at most 0.0011 of P(holds) at or below 2x.
FUTILITY = {"alpha_opt": 0.1, "min_look": 1000, "tau": 0.05}


def futility_bound(n, k_rare, k_common, later_looks, design: dict, pi_rare: float, bets, delta: float,
                   alpha_opt: float = FUTILITY["alpha_opt"]):
    """an UPPER estimate of P(the mixture clears at some later look), from a look of n shots with these error
    counts (scalars, or numpy arrays of counts for the operating-characteristics DP -- the same rule either way).

    Each group's per-shot error rate is projected at its OPTIMISTIC alpha_opt-quantile (Jeffreys, Beta(k + 1/2,
    n - k + 1/2)), each bet's log-wealth as a Gaussian random walk with that mean and variance per shot, and the
    chance it reaches log(1/delta) by each later look is Phi. The mixture can clear only if SOME bet's wealth does
    (log mean exp <= max), so the sum over bets of the best look is an upper estimate. Optimism and the sum both
    push the estimate UP, so the rule stops only when the run has plainly no chance."""
    import numpy as np
    from scipy.stats import beta as Beta
    from scipy.stats import norm
    m, a = threshold(design)["m"], scores(design, pi_rare)
    k1, k2 = np.asarray(k_rare, dtype=float), np.asarray(k_common, dtype=float)
    q1 = Beta.ppf(alpha_opt, k1 + 0.5, n - k1 + 0.5)
    q2 = Beta.ppf(alpha_opt, k2 + 0.5, n - k2 + 0.5)
    q0 = np.clip(1 - q1 - q2, 0.0, 1.0)
    bar = math.log(1 / delta)
    total = 0.0
    for b in check_bets(design, pi_rare, bets):
        x0, x1, x2 = math.log1p(b * m), math.log1p(b * (m - a[RARE])), math.log1p(b * (m - a[COMMON]))
        lw = (n - k1 - k2) * x0 + k1 * x1 + k2 * x2
        g = q0 * x0 + q1 * x1 + q2 * x2
        v = np.maximum(q0 * x0 ** 2 + q1 * x1 ** 2 + q2 * x2 ** 2 - g ** 2, 1e-12)
        best = 0.0
        for later in later_looks:
            r = later - n
            if r <= 0:
                raise BettingRefused(f"later look {later} is not after {n}")
            best = np.maximum(best, norm.cdf((lw + r * g - bar) / np.sqrt(r * v)))
        total = total + best
    return total


def verdict(design: dict, seq: dict, errors: dict) -> dict:
    """the preregistered rule: at each look in order, if every unit up to it has a result, compute the mixture
    wealth from the prefix's error counts; the claim holds at the FIRST look with K >= 1/delta. A look whose units
    have not all run is NOT REACHED (and so is every later look). A look that does not clear, from FUTILITY's
    min_look on and before the last look, STOPS the run when futility_bound < tau: no later batch is submitted, and
    the claim fails. Every row says what the operator does next. Rule 'betting_ad' (R225) goes to verdict_ad."""
    if seq.get("rule") == AD_RULE:
        return verdict_ad(design, seq, errors)
    if seq.get("rule") != "betting":
        raise BettingRefused(f"the package's rule is {seq.get('rule')!r}, not 'betting'")
    pi, bets, delta = float(seq["pi_rare"]), seq["bets"], float(seq["claim_delta"])
    check_bets(design, pi, bets)
    names = [u["name"] for u in design["units"]]
    bar = math.log(1 / delta)
    rows = []
    for look in seq["looks"]:
        if look > len(names):
            raise BettingRefused(f"look {look} is past the package's {len(names)} units")
        if any(n not in errors for n in names[:look]):
            rows.append(dict(look=look, status="NOT REACHED"))
            break
        tally = {RARE: [0, 0], COMMON: [0, 0]}
        for u in design["units"][:look]:
            tally[u["group"]][0] += int(errors[u["name"]])
            tally[u["group"]][1] += 1
        lw = log_wealth(look, tally[RARE][0], tally[COMMON][0], design, pi, bets)
        lk = log_mean_exp(lw)
        clears = lk >= bar
        row = dict(look=look, status="CLEARS" if clears else "below", log_wealth=lk, log_bar=bar,
                   log_wealth_per_bet=dict(zip([str(b) for b in bets], lw)),
                   errors={g: dict(errors=e, shots=n) for g, (e, n) in tally.items()})
        later = [x for x in seq["looks"] if x > look]
        if not clears and later and look >= FUTILITY["min_look"]:
            u = float(futility_bound(look, tally[RARE][0], tally[COMMON][0], later, design, pi, bets, delta))
            row["futility_bound"] = u
            if u < FUTILITY["tau"]:
                row["status"] = "STOP (futility)"
        row["next"] = ("stop: the claim holds" if clears else "stop: futility, submit nothing more"
                       if row["status"] == "STOP (futility)" else "submit the next batch" if later else
                       "stop: the last look")
        rows.append(row)
        if row["status"] != "below":
            break
    last = rows[-1]
    out = ("CLAIM HOLDS" if last["status"] == "CLEARS"
           else "NOT DECIDED (stopped before the last look)" if last["status"] == "NOT REACHED"
           else f"CLAIM FAILS (stopped for futility at look {last['look']})" if last["status"] == "STOP (futility)"
           else "CLAIM FAILS (every look below the bar)")
    return dict(verdict=out, stopped_at=last["look"], claim_delta=delta, rule="betting (R203)", pi_rare=pi,
                bets=list(bets), m=threshold(design)["m"], futility=dict(FUTILITY), looks=rows, claim_scope=SCOPE)


# ---------------------------------------------------------------------------------------------------------------------
# R225: the run's test is the fixed 50/50 MIXTURE of R203's F_e wealth and the antidegradability witness w*'s
# (ad_witness), chosen by r225_witness_choice.py's rule, declared before its first run, on simulated operating
# characteristics: at 2.5x the data sheet it holds 0.998 against F_e's 0.133, and in no noise direction tried does it
# cost a run F_e would win. Six components, three bets each test, equal weights. Each is a constant-bet wealth on its
# OWN score process against its OWN threshold:
#   - F_e: R203's scores by group (threshold, scores) against m_F;
#   - the witness: kappa by (group, input basis) against m_w = w . lam(ideal) - hbar (ad_witness.constants).
# THE NULL: the shot-averaged design channel the device implemented is antidegradable. That puts EVERY component on
# its null at once (F_e <= 3/4 and w . diag(T) <= hbar both hold on AD), so the mean of the six wealths is below a
# mean of nonnegative martingales from 1, and Ville bounds a false claim by delta (lean: ville_family).
# WHAT IT CERTIFIES: the shot-averaged design channel is NOT antidegradable -- its F_e exceeds 3/4 or its witness
# exceeds the AD bound -- which no device using P alone or A alone (mixtures, memory, shared randomness included)
# implements. The F_e number itself is not certified unless the F_e components carry the claim.
# THE FLOAT DECISION: the claim needs log K >= log(1/delta) + FLOAT_GUARD. FLOAT_GUARD is far above the derived bound
# on the double-precision error of log K (r225_float_guard.py), so a float claim implies the exact-real claim
# the Lean theorem is about. It can only withhold a claim within FLOAT_GUARD of the bar, never make one.
AD_RULE = "betting_ad"
FLOAT_GUARD = 1e-9

SCOPE_AD = ("the shot-averaged design channel the device implemented (each shot's given the device's own history) is "
            "not antidegradable -- its F_e exceeds 3/4 or its w* witness exceeds the antidegradable bound -- so no "
            "device using P alone or A alone, with any mixture, memory or shared randomness, implements it; valid for "
            "any behaviourally non-anticipating device -- drift, memory and outcome-dependent errors included -- under "
            "any stopping rule (Ville), provided the shots ran in an order independent of their content, each batch "
            "was submitted once, Bob's basis choice beta does not signal into the copy readout it is matched against, "
            "and no device moves an erased register's content before its reset; delta is over the seeded draw of the "
            "units (the design's randomization)")


def category(unit: dict) -> str:
    """a unit's (group, input basis) category: the witness scores an error by it"""
    import ad_witness as A
    cat = f"{unit['group']}|{unit['inp'][0]}"
    if cat not in A.CATS:
        raise BettingRefused(f"unit {unit.get('name')} is in category {cat}, not one of {A.CATS}")
    return cat


def ad_components(design: dict, seq: dict) -> list:
    """the six components of the R225 mixture: [{test, m, kappa (by category), bet}]. The witness's constants come
    from the package's own W_g and the preregistered pi (ad_witness.constants refuses an hbar that is not a
    certified bound); every bet must be safe against its test's largest score"""
    from fractions import Fraction

    import ad_witness as A
    pi = float(seq["pi_rare"])
    a = scores(design, pi)
    comps = [dict(test="F_e", m=threshold(design)["m"], kappa={c: a[c.split("|")[0]] for c in A.CATS}, bet=b)
             for b in check_bets(design, pi, seq["bets"])]
    wit = seq.get("witness")
    if not isinstance(wit, dict) or set(wit) != {"w", "hbar", "bets"}:
        raise BettingRefused(f"rule {AD_RULE} needs the witness's w, hbar and bets in sequential.json, not {wit!r}")
    c = A.constants([Fraction(x) for x in wit["w"]], Fraction(wit["hbar"]),
                    {g: design["groups"][g]["W"] for g in (RARE, COMMON)}, Fraction(str(pi)), m=int(design["m"]))
    top = max(c["kappa"].values())
    for b in wit["bets"]:
        if not (float(b) > 0 and 1 + float(b) * (c["m"] - top) > 0):
            raise BettingRefused(f"witness bet {b} breaks 0 < b < 1 / (max kappa - m_w) = {1 / (top - c['m']):.4f}")
        comps.append(dict(test="witness", m=c["m"], kappa=dict(c["kappa"]), bet=float(b)))
    return comps


def _component_name(comp: dict) -> str:
    return f"{comp['test']}@{comp['bet']}"


def ad_log_wealth(n: int, counts: dict, comps: list) -> list:
    """log K_n of each component from the look's shot count and its error counts by category (all it depends on)"""
    k = sum(counts.values())
    if n < 0 or min(counts.values(), default=0) < 0 or k > n or set(counts) - set(comps[0]["kappa"]):
        raise BettingRefused(f"counts n={n}, {counts} are not a run")
    return [(n - k) * math.log1p(c["bet"] * c["m"])
            + sum(kc * math.log1p(c["bet"] * (c["m"] - c["kappa"][cat])) for cat, kc in counts.items())
            for c in comps]


def futility_bound_ad(n, counts: dict, later_looks, comps: list, delta: float,
                      alpha_opt: float = FUTILITY["alpha_opt"]):
    """futility_bound for the R225 mixture: an UPPER estimate of P(the mixture clears at some later look). Each
    category's per-shot error rate at its OPTIMISTIC alpha_opt-quantile (Jeffreys); each component's log-wealth a
    Gaussian random walk with that mean and variance; the mixture clears only if SOME component does (mean <= max),
    so the sum over components of each one's best later look is an upper estimate. counts: {category: scalar or
    numpy array} -- the run's rule and the operating characteristics' DP apply the same function"""
    import numpy as np
    from scipy.stats import beta as Beta
    from scipy.stats import norm
    cats = list(comps[0]["kappa"])
    k = {c: np.asarray(counts.get(c, 0), dtype=float) for c in cats}
    q = {c: Beta.ppf(alpha_opt, k[c] + 0.5, n - k[c] + 0.5) for c in cats}
    q0 = np.clip(1 - sum(q.values()), 0.0, 1.0)
    n0 = n - sum(k.values())
    bar = math.log(1 / delta)
    total = 0.0
    for comp in comps:
        x0 = math.log1p(comp["bet"] * comp["m"])
        x = {c: math.log1p(comp["bet"] * (comp["m"] - comp["kappa"][c])) for c in cats}
        lw = n0 * x0 + sum(k[c] * x[c] for c in cats)
        g = q0 * x0 + sum(q[c] * x[c] for c in cats)
        v = np.maximum(q0 * x0 ** 2 + sum(q[c] * x[c] ** 2 for c in cats) - g ** 2, 1e-12)
        best = 0.0
        for later in later_looks:
            r = later - n
            if r <= 0:
                raise BettingRefused(f"later look {later} is not after {n}")
            best = np.maximum(best, norm.cdf((lw + r * g - bar) / np.sqrt(r * v)))
        total = total + best
    return total


def verdict_ad(design: dict, seq: dict, errors: dict) -> dict:
    """R225's preregistered rule, R203's verdict with the mixture's six components: at each look in order, if every
    unit up to it has a result, the equal-weight mixture's log wealth from the prefix's error counts by category; the
    claim holds at the FIRST look with log K >= log(1/delta) + FLOAT_GUARD. NOT REACHED and the futility stop as in
    R203 (futility_bound_ad, FUTILITY's parameters)"""
    if seq.get("rule") != AD_RULE:
        raise BettingRefused(f"the package's rule is {seq.get('rule')!r}, not {AD_RULE!r}")
    import ad_witness as A
    delta = float(seq["claim_delta"])
    comps = ad_components(design, seq)
    names = [u["name"] for u in design["units"]]
    cats = [category(u) for u in design["units"]]
    bar = math.log(1 / delta)
    rows = []
    for look in seq["looks"]:
        if look > len(names):
            raise BettingRefused(f"look {look} is past the package's {len(names)} units")
        if any(n not in errors for n in names[:look]):
            rows.append(dict(look=look, status="NOT REACHED"))
            break
        tally = {c: [0, 0] for c in A.CATS}
        for name, cat in zip(names[:look], cats[:look]):
            tally[cat][0] += int(errors[name])
            tally[cat][1] += 1
        counts = {c: e for c, (e, _) in tally.items()}
        lw = ad_log_wealth(look, counts, comps)
        lk = log_mean_exp(lw)
        clears = lk >= bar + FLOAT_GUARD
        row = dict(look=look, status="CLEARS" if clears else "below", log_wealth=lk, log_bar=bar,
                   float_guard=FLOAT_GUARD, log_wealth_per_component=dict(zip(map(_component_name, comps), lw)),
                   errors={c: dict(errors=e, shots=n) for c, (e, n) in tally.items()})
        later = [x for x in seq["looks"] if x > look]
        if not clears and later and look >= FUTILITY["min_look"]:
            u = float(futility_bound_ad(look, counts, later, comps, delta))
            row["futility_bound"] = u
            if u < FUTILITY["tau"]:
                row["status"] = "STOP (futility)"
        row["next"] = ("stop: the claim holds" if clears else "stop: futility, submit nothing more"
                       if row["status"] == "STOP (futility)" else "submit the next batch" if later else
                       "stop: the last look")
        rows.append(row)
        if row["status"] != "below":
            break
    last = rows[-1]
    out = ("CLAIM HOLDS" if last["status"] == "CLEARS"
           else "NOT DECIDED (stopped before the last look)" if last["status"] == "NOT REACHED"
           else f"CLAIM FAILS (stopped for futility at look {last['look']})" if last["status"] == "STOP (futility)"
           else "CLAIM FAILS (every look below the bar)")
    return dict(verdict=out, stopped_at=last["look"], claim_delta=delta, rule="betting_ad (R225)",
                pi_rare=float(seq["pi_rare"]), witness=dict(seq["witness"]),
                components=[dict(test=c["test"], m=c["m"], bet=c["bet"]) for c in comps], futility=dict(FUTILITY),
                looks=rows, claim_scope=SCOPE_AD)
