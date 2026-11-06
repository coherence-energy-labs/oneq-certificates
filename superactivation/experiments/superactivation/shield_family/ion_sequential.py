"""R201 -- the m = 8 GROUP-SEQUENTIAL trapped-ion run as a submittable package (R200 designed it; this builds it).

R200 showed the fixed 4,000-unit package survives only 1.5x its data sheet at 5 sigma, and that a group-sequential
certificate is cheaper at spec and survives 2x. This module makes that design real, reusing ion_run_package's freeze,
allocation and certificate unchanged:

  freeze   8,000 FRESH units (seed 201), allocated by the 4,000-unit package's own rule (Neyman over the exact
           design-law weights with H2-2's faithful-arm error rates, R190), written as ONE package;
  split    the package into the four preregistered batches -- units [0, 1000), [1000, 2000), [2000, 4000),
           [4000, 8000) -- each its own package directory, so ion_param_shots emits ONE parameterized-shot program
           per batch and tools/verify_ion_param_shots.py verifies every shot of it;
  analyse  after each batch, certify the cumulative prefix (looks at 1,000 / 2,000 / 4,000 / 8,000 units) with the
           package's exact Clopper-Pearson certificate at delta / looks per look (split again over the groups
           inside), and STOP at the first look that clears. The union bound over looks keeps the claim's error
           rate <= delta under this stopping rule.

Why prefixes are valid samples: freeze accepts units by rejection sampling, so each group's accepted units are iid
draws from that group's law, and every look's per-group counts are fixed before any data.

R202 (2026-10-21) -- the COMPOSITION and the ORDER. R201 froze the Neyman split (5.55% ee_x) in acceptance order,
where the ee_x quota fills by unit 2,599, so batch 4 holds no ee_x unit.
  - Under drift that order certifies a quantity that is not the F_e of any channel the device ran (`block_order`'s
    docstring has the witness).
  - Stratifying it to the Neyman split at every look would cost +27% expected HQC at spec: the exact 5 sigma bound
    pays a floor term ln(1/a)/n_g, so its optimal split is 15-25% ee_x, not 5.55%.
  - The default rule 'block' freezes a 1/5 share, one ee_x unit per block of 5 consecutive units. Both groups then
    share one time profile, and in the R202 dry run it beats R201's order at every lambda.

R203 replaced these rules with the BETTING certificate on iid-labelled shots (rule 'betting'), and R225 with its
F_e + antidegradability-witness mixture (rule 'betting_ad', the default; ion_betting.verdict_ad).

  python ion_sequential.py freeze OUT_DIR [--rule betting_ad|betting|block|neyman] [--seed N] [--n 8000]
  python ion_sequential.py adopt PKG_DIR [--rule betting_ad]   -> a frozen betting package read by another betting
                                                                  rule (before any shot; units and labels stay)
  python ion_sequential.py split PKG_DIR                  -> PKG_DIR/batch_1 .. batch_k (one per look)
  python ion_sequential.py analyse PKG_DIR RESULTS_DIR    -> the preregistered verdict, look by look, from the
                                                             provider's raw results (RESULTS_DIR/batch_<k>.json)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

LOOKS = (1000, 2000, 4000, 8000)
CLAIM_DELTA = 0.5 * math.erfc(5 / math.sqrt(2))        # one-sided 5 sigma
SEED, SEED_R202 = 201, 202                              # R201's committed package; R202's block-ordered one
M8_SCAN = ROOT / "evidence" / "superactivation" / "r190_m_scan" / "m_scan.json"
# R202: ONE ee_x/high unit in every block of BLOCK consecutive units (a 1/BLOCK share), at a seeded position
BLOCK, RARE, COMMON = 5, "ee_x/high", "kept/high"
# R203: the BETTING certificate on iid-labelled shots. Each slot's label is ee_x/high with probability PI_RARE
# (seeded, iid), its unit a fresh draw from that group's law; the test is a mixture of constant bets (Ville).
# Chosen on the exact operating characteristics (r203_betting.py, R195's Helios projection): looks are free under
# an anytime-valid test, so the early looks are fine
PI_RARE, BETS, SEED_R203 = 0.08, (0.3, 0.5, 0.8), 203
LOOKS_R203 = (500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000)
# R225: rule 'betting_ad' -- the same package, labels, looks, pi and F_e bets, read by the fixed 50/50 mixture of the
# F_e wealth and the antidegradability witness W_STAR (ion_betting.verdict_ad). Chosen by r225_witness_choice.py's
# pre-declared rule; W_STAR from the device model alone, HBAR its exact certified bound (ad_witness.certify_bound),
# WITNESS_BETS its Kelly bets at the lambdas where BETS are F_e's, rounded to 3 decimals
AD_RULE = "betting_ad"
W_STAR, HBAR, WITNESS_BETS = ("1/4", "19/200", "31/200"), "171076149241/500000000000", (0.467, 0.579, 0.699)
# R225 adversary pass (2026-10-26): the analysis read design.json's constants -- each high group's W and width,
# f_ideal, bar_F_e, the shield size -- UNBOUND. A design.json doctored consistently with its batches passed every check
# (the emitter's programs and tables do not carry them), and a null device then claimed 57 times in 400 with both group
# weights scaled by 0.8. The same gap fed R203's F_e scores. They are now the preregistered ones
# (check_design_constants): the design law's exact high-group weights (R224, rounded UP as the freeze rounds them),
# the score widths, bar 3/4, m = 8 and f_ideal at the closed form.
DESIGN_M, BAR_F_E = 8, 0.75
W_EXACT = {"ee_x/high": (257, 3468), "kept/high": (149, 408)}
WIDTH = {"ee_x/high": 1.0, "kept/high": 2.0}


class SequentialRefused(ValueError):
    pass


def block_order(groups: list, block: int = BLOCK, seed: int = SEED_R202) -> list:
    """R202: the permutation that puts exactly one RARE unit in every block of `block` consecutive units, at a
    position drawn from a seeded generator, keeping each group's acceptance order (each group's units stay an iid
    sequence from its law).

    WHY. The certificate bounds sum_g width_g W_g pbar_g, where pbar_g averages group g's error over the shots its
    units ran in. If the groups run at different times (R201's sampler put every ee_x unit in the first 2,600 shots),
    a drifting device can make that sum clear while every channel it ran has F_e <= 3/4 (a late channel that fails
    only ee_x units is never measured on them). When every group has the same time profile, the sum is the error of
    the TIME-AVERAGED channel. F_e is linear in the channel, so clearing it means some channel the device ran had
    F_e > 3/4 -- no stationarity premise, only that the channel is constant within a block of `block` shots."""
    rare = [i for i, g in enumerate(groups) if g == RARE]
    common = [i for i, g in enumerate(groups) if g == COMMON]
    if len(rare) + len(common) != len(groups) or len(groups) % block or len(rare) * block != len(groups):
        raise SequentialRefused(f"{len(groups)} units with {len(rare)} {RARE}: block order needs one {RARE} unit per "
                                f"block of {block} and every other unit {COMMON}")
    import numpy as np
    rng = np.random.default_rng([seed, block])
    perm: list = []
    for b in range(len(rare)):
        blk = common[b * (block - 1):(b + 1) * (block - 1)]
        blk.insert(int(rng.integers(block)), rare[b])
        perm += blk
    return perm


def iid_labels(n: int, pi: float = PI_RARE, seed: int = SEED_R203) -> list:
    """R203: each slot's group label, iid: RARE with probability pi. Drawn BEFORE any unit, from its own stream"""
    import numpy as np
    rng = np.random.default_rng([seed, 203])
    return [RARE if x < pi else COMMON for x in rng.random(n)]


def labels_sha(labels: list) -> str:
    import hashlib
    return hashlib.sha256("".join("1" if g == RARE else "0" for g in labels).encode("ascii")).hexdigest()


def label_order(labels: list):
    """the freeze's order hook for R203: RARE units go to the RARE label positions, COMMON units to the rest, each
    group in acceptance order (so each stays an iid sequence from its law, and the label sequence stays iid)"""
    def order(groups):
        if sorted(groups) != sorted(labels):
            raise SequentialRefused(f"the frozen groups ({groups.count(RARE)} {RARE}) do not match the labels "
                                    f"({labels.count(RARE)})")
        pools = {g: [i for i, x in enumerate(groups) if x == g] for g in (RARE, COMMON)}
        nxt = {RARE: 0, COMMON: 0}
        perm = []
        for g in labels:
            perm.append(pools[g][nxt[g]])
            nxt[g] += 1
        return perm
    return order


def time_profile(units: list) -> dict:
    """whether every group has the same time profile, at block resolution: exactly one RARE unit in every block of
    len(units) / n_RARE consecutive units (the claim's drift-free reading, R202). Anything else is reported, with
    the reason, so the verdict can scope its claim"""
    groups = [u["group"] for u in units]
    n_rare = groups.count(RARE)
    if set(groups) - {RARE, COMMON} or not n_rare or len(groups) % n_rare:
        return dict(uniform=False, reason=f"{n_rare} {RARE} units do not tile {len(groups)} units into equal blocks")
    block = len(groups) // n_rare
    bad = [b for b in range(n_rare) if groups[b * block:(b + 1) * block].count(RARE) != 1]
    if bad:
        return dict(uniform=False, reason=f"{len(bad)} of {n_rare} blocks of {block} do not hold exactly one {RARE} "
                                          f"unit (first: block {bad[0]})")
    return dict(uniform=True, block=block, rare=RARE)


def ebar_rule() -> dict:
    """the 4,000-unit package's allocation rule: H2-2 (N = 56 memory) faithful-arm high-group error rates at m = 8
    (R190). They reproduce its 222 / 3,778 split exactly; they decide the split only, never the bound's validity"""
    rows = json.loads(M8_SCAN.read_text(encoding="utf-8"))
    (row,) = [r for r in rows if r["m"] == 8 and r["device"] == "spec:h2_2_mem56"]
    return dict(row["arms"]["faithful"]["high_group_error"])


def plan(looks=LOOKS, claim_delta: float = CLAIM_DELTA) -> dict:
    looks = tuple(int(n) for n in looks)
    if not looks or any(b <= a for a, b in zip(looks, looks[1:])) or looks[0] <= 0:
        raise SequentialRefused(f"looks {looks} must be strictly increasing and positive")
    edges = (0,) + looks
    return dict(schema="oneq.ion_sequential/1", looks=list(looks), claim_delta=claim_delta,
                delta_per_look=claim_delta / len(looks),
                batches=[[edges[i], edges[i + 1]] for i in range(len(looks))],
                alpha_split="claim delta / looks per look; the package certificate splits it again over its groups")


RULE_SEED = {AD_RULE: SEED_R203, "betting": SEED_R203, "block": SEED_R202, "neyman": SEED}
RULE_LOOKS = {AD_RULE: LOOKS_R203, "betting": LOOKS_R203, "block": LOOKS, "neyman": LOOKS}


def witness_fields() -> dict:
    """the R225 rule's preregistered witness block for sequential.json"""
    return dict(w=list(W_STAR), hbar=HBAR, bets=list(WITNESS_BETS))


def freeze_sequential(outdir: pathlib.Path, seed: int | None = None, n: int = LOOKS[-1], looks=None,
                      rule: str = AD_RULE) -> dict:
    """rule 'betting_ad' (R225, the default) and 'betting' (R203): iid labels (ee_x with probability PI_RARE per
    slot), each unit a fresh draw from its label's group; the same package, read by the F_e + witness mixture
    (ion_betting.verdict_ad) or by F_e alone. Rule 'block' (R202): a 1/BLOCK ee_x share, one ee_x unit per block of
    BLOCK consecutive units. Rule 'neyman' (R201, seed 201): the 4,000-unit package's Neyman split in acceptance order
    -- the committed R201 package, kept so it re-freezes byte for byte. Seed and looks default to the rule's own"""
    import ion_run_package as P
    if rule not in RULE_SEED:
        raise SequentialRefused(f"unknown rule {rule!r}: '{AD_RULE}' (R225), 'betting' (R203), 'block' (R202) or "
                                "'neyman' (R201)")
    seed = RULE_SEED[rule] if seed is None else seed
    looks = RULE_LOOKS[rule] if looks is None else looks
    extra: dict = {}
    seq = plan(looks)
    if n != seq["looks"][-1]:
        raise SequentialRefused(f"a package of {n} units cannot serve looks ending at {seq['looks'][-1]}")
    if rule == "block":
        if any(x % BLOCK for x in seq["looks"]):
            raise SequentialRefused(f"every look {seq['looks']} must be a whole number of blocks of {BLOCK}")

        def order(groups):
            return block_order(groups, BLOCK, seed)
        order.rule = (f"block (R202): one {RARE} unit in every block of {BLOCK} consecutive units, at a position "
                      f"drawn from default_rng([{seed}, {BLOCK}]); each group keeps its acceptance order")
        design = P.freeze(n, outdir, seed=seed, m=8, ebar=ebar_rule(),
                          alloc={RARE: n // BLOCK, COMMON: n - n // BLOCK}, order=order)
    elif rule == "neyman":
        design = P.freeze(n, outdir, seed=seed, m=8, ebar=ebar_rule())
    elif rule in ("betting", AD_RULE):
        labels = iid_labels(n, PI_RARE, seed)
        order = label_order(labels)
        order.rule = (f"betting (R203): iid labels, {RARE} with probability {PI_RARE} per slot from "
                      f"default_rng([{seed}, 203]); each group keeps its acceptance order")
        # R224: BOTH one-erased orders, so the hardware's erasure resource is A (x) A and F_e > 3/4 binds (R213)
        design = P.freeze(n, outdir, seed=seed, m=8, ebar=ebar_rule(),
                          alloc={RARE: labels.count(RARE), COMMON: labels.count(COMMON)}, order=order,
                          erasure_orders="both")
        if [u["group"] for u in design["units"]] != labels:
            raise SequentialRefused("the frozen units are not in label order")
        extra = dict(delta_per_look=seq["claim_delta"], pi_rare=PI_RARE, bets=list(BETS),
                     alpha_split="none: the betting test is anytime-valid (Ville), so every look uses the whole delta",
                     labels_sha256=labels_sha(labels))
        if rule == AD_RULE:
            extra["witness"] = witness_fields()
    # the allocation rounds each group (at least one unit each), so the package can hold a unit more or less than n:
    # the LAST look is the whole package, never a prefix that leaves frozen units out of every batch (found by the
    # first smoke freeze, 2026-10-21: n = 8 froze 9 units and batch [4, 8) dropped the ninth)
    if len(design["units"]) != seq["looks"][-1]:
        seq = plan(tuple(seq["looks"][:-1]) + (len(design["units"]),), seq["claim_delta"])
    seq.update(seed=seed, rule=rule, units=len(design["units"]), requested_units=n,
               design_qpy_sha256=design["qpy_sha256"], time_profile=time_profile(design["units"]), **extra)
    (outdir / "sequential.json").write_text(json.dumps(seq, indent=1) + "\n", encoding="utf-8", newline="\n")
    return seq


def adopt_rule(pkg: pathlib.Path, rule: str = AD_RULE) -> dict:
    """move a frozen betting package to another betting rule BEFORE any shot. The units, labels, circuits, looks, pi
    and F_e bets are the package's own and stay; only the analysis rule and its preregistered fields change (R225:
    R224's package, frozen under 'betting', is read by 'betting_ad'). Refused unless the package binds under its
    current rule and under the new one"""
    if rule not in ("betting", AD_RULE):
        raise SequentialRefused(f"a betting package can adopt 'betting' or '{AD_RULE}', not {rule!r}")
    path = pkg / "sequential.json"
    seq = json.loads(path.read_text(encoding="utf-8"))
    design = json.loads((pkg / "design.json").read_text(encoding="utf-8"))
    check_betting_package(design, seq)
    new = {k: v for k, v in seq.items() if k != "witness"}
    new["rule"] = rule
    if rule == AD_RULE:
        new["witness"] = witness_fields()
    check_betting_package(design, new)
    path.write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8", newline="\n")
    return new


def split(pkg: pathlib.Path) -> list:
    """the package's preregistered batches, each a package directory (design.json with that batch's units,
    circuits.qasm.jsonl with their lines) for ion_param_shots / verify_ion_param_shots"""
    design = json.loads((pkg / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((pkg / "sequential.json").read_text(encoding="utf-8"))
    if seq["design_qpy_sha256"] != design["qpy_sha256"] or seq["units"] != len(design["units"]):
        raise SequentialRefused("sequential.json does not describe this package")
    spans = seq["batches"]
    if spans[0][0] != 0 or spans[-1][1] != len(design["units"]) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise SequentialRefused(f"the batches {spans} do not tile the package's {len(design['units'])} units")
    lines = [ln for ln in (pkg / "circuits.qasm.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if [json.loads(ln)["name"] for ln in lines] != [u["name"] for u in design["units"]]:
        raise SequentialRefused("circuits.qasm.jsonl is not the design's units, in order")
    out = []
    for k, (a, b) in enumerate(seq["batches"], 1):
        d = pkg / f"batch_{k}"
        d.mkdir(exist_ok=True)
        sub = dict(design, units=design["units"][a:b], sequential_batch=dict(index=k, units=[a, b],
                                                                                parent_qpy_sha256=design["qpy_sha256"]))
        (d / "design.json").write_text(json.dumps(sub, indent=1) + "\n", encoding="utf-8", newline="\n")
        (d / "circuits.qasm.jsonl").write_text("\n".join(lines[a:b]) + "\n", encoding="utf-8", newline="\n")
        out.append(d)
    return out


def errors_from_counts(design: dict, counts: dict) -> dict:
    """{unit name: bool error} from one-shot counts, by the package's own score (ion_run_package.analyse's rule):
    a shot errs when its score differs from the unit's deterministic ideal"""
    import hw_faithful as HF
    import ion_run_package as P
    out = {}
    for u in design["units"]:
        if u["name"] not in counts:
            continue
        c = counts[u["name"]]
        if sum(c.values()) != design["shots_per_circuit"] or design["shots_per_circuit"] != 1:
            raise SequentialRefused(f"{u['name']}: the design runs exactly one shot per circuit")
        sc = HF.score(design["m"], P._unit(u), design["decoder"])
        (bits,) = [b for b, k in c.items() if k]
        out[u["name"]] = abs(sc(P.bits_of(bits, u["n_clbits"], u["name"])) - u["ideal_score"]) > 1e-9
    return out


def errors_from_results(pkg: pathlib.Path, results_dir: pathlib.Path) -> tuple:
    """THE FIXED ANALYSIS INPUT (R202): the provider's raw per-shot results, one file per batch program
    (RESULTS_DIR/batch_<k>.json), each through ion_param_shots.counts_from_shots, which refuses a short, duplicated,
    re-ordered or mis-sized job. A batch with no file has not run, so its look and every later one is NOT REACHED.

    Refused before any scoring: a batch directory that is not the parent's slice by content (units and design fields),
    or whose receipt does not PASS for the program in it. Returns (design, seq, errors)."""
    import hashlib

    import ion_param_shots as PS
    design = json.loads((pkg / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((pkg / "sequential.json").read_text(encoding="utf-8"))
    if seq["design_qpy_sha256"] != design["qpy_sha256"] or seq["units"] != len(design["units"]):
        raise SequentialRefused("sequential.json does not describe this package")
    known = {f"batch_{k}{ext}" for k in range(1, len(seq["batches"]) + 1) for ext in (".json", ".job.json")}
    stray = sorted(p.name for p in results_dir.iterdir() if p.is_file() and p.name not in known | {"jobs.json"})
    if stray:
        raise SequentialRefused(f"results files that name no batch: {stray}")
    # the provider's list of EVERY job the account submitted in the run window (R203 adversary pass): a batch
    # re-run until it clears, or run with its results withheld, is outside the certificate's guarantee
    if not (results_dir / "jobs.json").is_file():
        raise SequentialRefused("no provider job list (jobs.json): nothing shows each batch ran exactly once")
    jobs = json.loads((results_dir / "jobs.json").read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or any(not isinstance(j, dict) or "hugr_sha256" not in j
                                         or "provider_job_id" not in j for j in jobs):
        raise SequentialRefused("jobs.json is not a list of {provider_job_id, hugr_sha256}")
    parent = [ln for ln in (pkg / "circuits.qasm.jsonl").read_text(encoding="utf-8").splitlines() if ln]

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    errors: dict = {}
    for k, (a, b) in enumerate(seq["batches"], 1):
        f = results_dir / f"batch_{k}.json"
        rc_path = pkg / f"batch_{k}" / "param_shots_receipt.json"
        rc_k = json.loads(rc_path.read_text(encoding="utf-8")) if rc_path.is_file() else {}
        ran = [j for j in jobs if rc_k and j["hugr_sha256"] == rc_k.get("hugr_sha256")]
        if len(ran) > 1:
            raise SequentialRefused(f"batch {k}'s program was submitted {len(ran)} times: one job per batch")
        if ran and not f.is_file():
            raise SequentialRefused(f"batch {k} ran but its results are not here (job {ran[0]['provider_job_id']})")
        if not f.is_file():
            continue
        if not ran:
            raise SequentialRefused(f"batch {k} has results but no job in the provider's list")
        d = pkg / f"batch_{k}"
        sub = json.loads((d / "design.json").read_text(encoding="utf-8"))
        lines = [ln for ln in (d / "circuits.qasm.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        if sub["units"] != design["units"][a:b] or sub.get("sequential_batch") != dict(
                index=k, units=[a, b], parent_qpy_sha256=design["qpy_sha256"]) or \
                {x: v for x, v in sub.items() if x not in ("units", "sequential_batch")} != \
                {x: v for x, v in design.items() if x != "units"} or lines != parent[a:b]:
            raise SequentialRefused(f"batch_{k} is not the parent package's units [{a}, {b})")
        # the program and table must be the emitter's output for THIS slice (the programs of all batches are the
        # same bytes; the table is what differs), and the receipt must bind all three files (R203 adversary pass:
        # an edited table, or another batch's receipt, used to pass)
        if (d / "param_shots_program.py").read_text(encoding="utf-8") != PS.emit(sub) or \
                (d / "param_shots_table.json").read_text(encoding="utf-8") != PS.table_json(PS.table(sub)):
            raise SequentialRefused(f"batch_{k}'s program or table is not the emitter's output for its slice")
        rc = json.loads((d / "param_shots_receipt.json").read_text(encoding="utf-8"))
        if rc.get("verdict") != "PASS" or rc.get("n_failures") != 0 or rc.get("rows") != b - a \
                or rc.get("shots_checked") != b - a or rc.get("program_sha256") != sha(d / "param_shots_program.py") \
                or rc.get("table_sha256") != sha(d / "param_shots_table.json") \
                or rc.get("circuits_qasm_sha256") != sha(d / "circuits.qasm.jsonl"):
            raise SequentialRefused(f"batch_{k}'s receipt does not PASS for the files in its directory")
        # the results carry only shot, uid and bits, which do not say which batch ran: the provider's job record
        # does. It must name the compiled program (HUGR digest) this batch's receipt verified
        job_f = results_dir / f"batch_{k}.job.json"
        job = json.loads(job_f.read_text(encoding="utf-8")) if job_f.is_file() else {}
        if not str(job.get("provider_job_id", "")).strip() or job.get("hugr_sha256") != rc.get("hugr_sha256") \
                or job.get("provider_job_id") != ran[0]["provider_job_id"]:
            raise SequentialRefused(f"batch_{k}.json has no job record naming batch {k}'s compiled program "
                                    f"(batch_{k}.job.json: provider_job_id and hugr_sha256 {rc.get('hugr_sha256')})")
        counts = PS.counts_from_shots(sub, PS.normalise_records(json.loads(f.read_text(encoding="utf-8"))))
        errors.update(errors_from_counts(sub, counts))
    return design, seq, errors


def check_betting_package(design: dict, seq: dict) -> None:
    """R203 adversary pass: the analysis read pi, the bets and delta from sequential.json unbound (pi edited to 0.3
    made a device BELOW the bar claim 100 times in 100), and labels_sha256 was written but never checked. A betting
    package's parameters must be the preregistered ones, and its labels the seeded iid stream. Rule 'betting_ad'
    (R225) binds the witness block too -- w*, its certified bound and its bets -- and re-proves the bound"""
    want = dict(pi_rare=PI_RARE, bets=list(BETS), claim_delta=CLAIM_DELTA, looks=list(LOOKS_R203), seed=SEED_R203)
    if seq.get("rule") == AD_RULE:
        import ad_witness as A
        want["witness"] = witness_fields()
        if not A.is_bound([Fraction(x) for x in W_STAR], HBAR):
            raise SequentialRefused(f"HBAR {HBAR} is not a certified antidegradability bound for W_STAR {W_STAR}")
    elif seq.get("rule") != "betting":
        raise SequentialRefused(f"rule {seq.get('rule')!r} is not a betting rule ('betting' or '{AD_RULE}')")
    bad = {k: seq.get(k) for k, v in want.items() if seq.get(k) != v}
    if bad:
        raise SequentialRefused(f"the package's betting parameters are not the preregistered ones: {bad}")
    labels = iid_labels(len(design["units"]), PI_RARE, SEED_R203)
    if [u["group"] for u in design["units"]] != labels or seq.get("labels_sha256") != labels_sha(labels):
        raise SequentialRefused("the package's labels are not the seeded iid stream it declares")
    check_erasure_resource(design)
    check_design_constants(design)


def check_design_constants(design: dict) -> None:
    """R225 adversary pass: the constants the scores and thresholds are computed from must be the preregistered ones,
    not whatever design.json says -- each high group's W (the exact law rounded UP), its width, bar_F_e, the shield
    size m, and f_ideal within 4e-16 of the closed form with the F_e threshold at or below the exact one"""
    import ad_witness as A
    import ion_betting as B
    if design.get("m") != DESIGN_M or design.get("bar_F_e") != BAR_F_E:
        raise SequentialRefused(f"the design's shield size {design.get('m')!r} or bar {design.get('bar_F_e')!r} is not "
                                f"the preregistered m = {DESIGN_M}, bar {BAR_F_E}")
    groups = design.get("groups", {})
    if set(groups) != set(W_EXACT):
        raise SequentialRefused(f"the design's groups {sorted(groups)} are not {sorted(W_EXACT)}")
    for g, (n, d) in W_EXACT.items():
        want = A.round_toward(Fraction(n, d), up=True)
        if groups[g].get("W") != want or groups[g].get("width") != WIDTH[g]:
            raise SequentialRefused(f"group {g}'s W {groups[g].get('W')!r} / width {groups[g].get('width')!r} are not "
                                    f"the design law's {want!r} ({n}/{d} rounded up) / {WIDTH[g]}")
    f_exact = Fraction(1, 2) + sum(A.lam_ideal(DESIGN_M)) / 6
    if abs(Fraction(design.get("f_ideal", 0.0)) - f_exact) > Fraction(4, 10**16):
        raise SequentialRefused(f"the design's f_ideal {design.get('f_ideal')!r} is not the closed form {f_exact}")
    m_exact = f_exact - (Fraction(BAR_F_E) + Fraction(1, 2)) / Fraction(3, 2)
    if Fraction(B.threshold(design)["m"]) > m_exact:
        raise SequentialRefused(f"the F_e threshold {B.threshold(design)['m']!r} lies above the exact {m_exact}")


def check_erasure_resource(design: dict) -> None:
    """R213/R224: the 3/4 bar holds only against what A (x) A and P (x) P can do -- each use erased with probability 1/2,
    independently. A package that folds the one-erased orders (use 1 always the erased one) hands the device an
    erasure resource in which use 0 survives 3/4 of the time, and a P-free strategy then reaches F_e = 13/16. Refused
    unless the design law has both orders AND the frozen units show both (a law recorded but not sampled is caught)"""
    import hw_faithful as HF
    if design.get("erasure_orders") != "both":
        raise SequentialRefused("the package folds the one-erased orders (R213): its hardware erasure resource keeps "
                                "use 0 with weight 3/4, so a P-free strategy reaches F_e = 13/16 and F_e > 3/4 does "
                                "not certify superactivation -- re-freeze with erasure_orders 'both'")
    marg = HF.erasure_marginals("both")
    if design.get("erasure_marginals") != marg or (marg["use0"], marg["use1"], marg["both"]) != (0.5, 0.5, 0.25):
        raise SequentialRefused(f"the package's erasure marginals {design.get('erasure_marginals')} are not A (x) A's")
    one = [u["branch"] for u in design["units"] if u["branch"] in ("kept_erased", "erased_kept")]
    k = one.count("erased_kept")
    if not one:
        raise SequentialRefused("the package samples no one-erased unit: its law's orders are recorded, not sampled")
    if abs(k - len(one) / 2) > 5 * (len(one) / 4) ** 0.5:
        raise SequentialRefused(f"{k} of the package's {len(one)} one-erased units erase use 0: not the 1/2 its law "
                                "declares (more than 5 sigma)")


def verdict(design: dict, seq: dict, errors: dict) -> dict:
    """the package's own preregistered rule: the betting test for rule 'betting' (R203), its parameters and labels
    bound first; the stratified Clopper-Pearson looks for the superseded rules (R201 'neyman', R202 'block'). Every
    rule first checks the erasure resource (R213): a folded package certifies nothing, whichever test reads it.
    Rule 'betting_ad' (R225, the F_e + witness mixture) is bound like 'betting', its witness block included"""
    if seq.get("rule") in ("betting", AD_RULE):
        import ion_betting as B
        check_betting_package(design, seq)
        return B.verdict(design, seq, errors)
    check_erasure_resource(design)
    check_design_constants(design)
    return sequential_verdict(design, seq, errors)


def sequential_verdict(design: dict, seq: dict, errors: dict) -> dict:
    """the preregistered rule on error indicators: for each look in order, if every unit up to it has a result,
    certify that prefix at delta_per_look (union over groups inside); the claim holds at the FIRST clearing look.
    A look whose units have not all run is NOT REACHED (and so is every later look)."""
    import r186_certification as C
    names = [u["name"] for u in design["units"]]
    groups = list(design["groups"])
    rows = []
    for look in seq["looks"]:
        if look > len(names):
            raise SequentialRefused(f"look {look} is past the package's {len(names)} units")
        if any(n not in errors for n in names[:look]):
            rows.append(dict(look=look, status="NOT REACHED"))
            break
        tally = {g: [0, 0] for g in groups}
        for u in design["units"][:look]:
            tally[u["group"]][0] += int(errors[u["name"]])
            tally[u["group"]][1] += 1
        a = seq["delta_per_look"] / len(groups)
        ucb = {g: C.cp_upper(e, n, a) if n else 1.0 for g, (e, n) in tally.items()}
        f_lo = design["f_ideal"] - sum(design["groups"][g]["width"] * design["groups"][g]["W"] * ucb[g] for g in groups)
        F_lo = 1.5 * f_lo - 0.5
        clears = bool(F_lo > design["bar_F_e"])
        # each batch's own tallies: not part of the decision, the record a reader needs to see drift between batches
        per_batch = []
        for a_, b_ in seq["batches"]:
            if a_ >= look:
                break
            t = {g: dict(errors=0, shots=0) for g in groups}
            for u in design["units"][a_:min(b_, look)]:
                t[u["group"]]["errors"] += int(errors[u["name"]])
                t[u["group"]]["shots"] += 1
            per_batch.append(dict(units=[a_, min(b_, look)], errors=t))
        rows.append(dict(look=look, status="CLEARS" if clears else "below", certified_F_e_lower=F_lo,
                         errors={g: dict(errors=e, shots=n, ucb=ucb[g]) for g, (e, n) in tally.items()},
                         batches=per_batch))
        if clears:
            break
    last = rows[-1]
    verdict = ("CLAIM HOLDS" if last["status"] == "CLEARS"
               else "NOT DECIDED (stopped before the last look)" if last["status"] == "NOT REACHED"
               else "CLAIM FAILS (every look below the bar)")
    prof = time_profile(design["units"])
    scope = (f"the time-averaged channel over the shots run (both groups share one time profile, one {RARE} unit per "
             f"{prof['block']} shots): clearing means some channel the device ran had F_e > 3/4, provided the channel "
             f"is constant within a block of {prof['block']} shots" if prof["uniform"] else
             f"a stationary device only: the groups do not share one time profile ({prof['reason']}), so under drift "
             "the certified quantity is not the F_e of any channel the device ran (R202)")
    return dict(verdict=verdict, stopped_at=last["look"], claim_delta=seq["claim_delta"],
                delta_per_look=seq["delta_per_look"], looks=rows, time_profile=prof, claim_scope=scope)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["freeze", "split", "analyse", "adopt"])
    ap.add_argument("path")
    ap.add_argument("counts", nargs="?")
    ap.add_argument("--seed", type=int, default=None, help="default: the rule's own (203 / 202 / 201)")
    ap.add_argument("--n", type=int, default=LOOKS[-1])
    ap.add_argument("--rule", choices=[AD_RULE, "betting", "block", "neyman"], default=AD_RULE)
    a = ap.parse_args()
    p = pathlib.Path(a.path)
    if a.mode == "freeze":
        print(json.dumps(freeze_sequential(p, a.seed, a.n, rule=a.rule), indent=1))
    elif a.mode == "adopt":
        print(json.dumps(adopt_rule(p, a.rule), indent=1))
    elif a.mode == "split":
        print("\n".join(str(d) for d in split(p)))
    else:
        if not a.counts or not pathlib.Path(a.counts).is_dir():
            raise SequentialRefused("analyse PKG_DIR RESULTS_DIR: RESULTS_DIR holds the provider's raw results, one "
                                    "batch_<k>.json per batch program that ran")
        design, seq, errors = errors_from_results(p, pathlib.Path(a.counts))
        print(json.dumps(verdict(design, seq, errors), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
