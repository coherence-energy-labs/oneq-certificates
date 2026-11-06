"""R188: the SUBMISSION PACKAGE for the first device-feasible superactivation run -- the channel-faithful two-use
protocol at m = 6 (shield dimension d = 64) on a trapped-ion device (R186: Quantinuum H2-2 / Helios clear 3/4 by
+0.030 in the validated model), certified by the exact ERROR-INDICATOR design (R186 part 3). Built so that, when
device access exists, nothing remains to be written. Spends nothing; no network.

The design (r186_certification.py `binomial`, whose reasoning this follows):
  - every unit (P_d's random gates on both uses, the erasure branch, the input) has a DETERMINISTIC ideal outcome,
    so a shot is its ideal score or an error; units whose ideal is the TOP of their score's support are the "high"
    groups (one per kind), and their errors are the only ones that lower F_e;
  - a draw is a FRESH unit from the design law restricted to a high group, run for ONE shot, so its error indicator
    is exactly Bernoulli(ebar_g) on any device, and an exact one-sided Clopper-Pearson bound applies;
  - f_lower = f_ideal - sum_g width_g W_g UCB_g(delta / G); F_e_lower = 1.5 f_lower - 0.5; the claim is F_e_lower > 3/4.
    The low groups' errors only RAISE f and are bounded by 0; the both-erased Z/Y units score exactly 1/2 (no shots).
  - The circuits are hw_faithful's with erasure by OMISSION (erase='none': an erased register is never touched again,
    the same outcome law as a reset), so there is no mid-circuit read, no reset and no feed-forward.

  python ion_run_package.py freeze N OUTDIR [--seed S]   draw N high-group units (allocated by width W sqrt(ebar)
                                                          from certification.json), build, classify by the EXACT
                                                          ideal, export design.json + circuits.qasm.jsonl + .qpy
  python ion_run_package.py analyse DESIGN COUNTS        COUNTS: {circuit name: {bitstring: count}} -> the verdict
  python ion_run_package.py dryrun DESIGN [--reps R]     every circuit sampled from the device MODEL's exact error
                                                          rate (R186's evaluator), analysed R times: the pass rate
  python ion_run_package.py recost DESIGN_DIR            re-run the cost producer on the frozen QASM (hash-checked)

The SUBMISSION FORM is R189 (ion_param_shots.py): the same frozen units as ONE parameterized-shot Helios program, one
unit per shot, which pays the fixed 5 HQC once instead of once per circuit.
"""
from __future__ import annotations

import argparse
import hashlib
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

M, DECODER, DEVICE = 6, "mix", "spec:h2_2"
CERT = ROOT / "evidence" / "superactivation" / "r186_faithful_requirements" / "certification.json"
MODEL_RUN = "spec_h2_2_m6_mix_standard"
DELTA = 0.00135
BAR_F = 5 / 6
PREMISE = {"": {"barriers": [0, 0], "if_else": [0, 0], "mid_circuit_measures": [0, 0]}}


class PackageRefused(ValueError):
    """an input the package cannot freeze or analyse faithfully"""


# the OpenQASM 2 standard header's own gates. qasm2.dumps writes qiskit's phase gate as `p`, which qelib1.inc does not
# define, so a strict OpenQASM 2 reader REFUSES the file (found 2026-10-19 by this package's own round-trip test,
# before any vendor saw one): circuits are lowered to these gates first, at optimization level 0 (nothing reordered)
QELIB1 = ["u3", "u2", "u1", "cx", "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cz", "measure"]


def export_qasm(qc) -> str:
    """the circuit as strict OpenQASM 2 (qelib1.inc gates only)"""
    from qiskit import qasm2, transpile
    return qasm2.dumps(transpile(qc, basis_gates=QELIB1, optimization_level=0))


WEIGHTS = ROOT / "evidence" / "superactivation" / "ion_run_m6" / "exact_group_weights.json"


def exact_group_weights(m: int, draws: int = 8, seed: int = 0, erasure_orders: str = "folded") -> dict:
    """{group: W_g} EXACT from the design law, for the certificate's f_lower = f_ideal - sum width W UCB.

    WHY (2026-10-19, R190). The packages carried W_g SAMPLED from R186's 2,048-unit law: kept/high 0.35441 against
    the exact 77/216 = 0.356481. Understating a high group's weight makes f_lower too HIGH, so the "exact"
    certificate was anti-conservative by ~1.5e-4 in F_e. A unit's group is a function of its stratum key and its
    input alone -- DERIVED from the design (P_d's SECTOR, not its sampled gate, fixes the logical effect; the label
    within a sector is the randomization), and MEASURED here: every (stratum, input) cell is evaluated on `draws`
    fresh label draws and REFUSED if they disagree. Then W_g = sum over cells of weight(stratum) / 6."""
    import hw_faithful as HF
    import hw_two_use as H
    import r186_faithful_requirements as R
    dev = R.make_device(DEVICE, m, DECODER)
    rng = np.random.default_rng(seed)
    W: dict = {}
    exact: dict = {}
    by_basis: dict = {}
    for key, w in R.faithful_strata(m, erasure_orders):
        for inp in H.INPUTS:
            seen = set()
            for _ in range(draws):
                u0 = R.sample_faithful(m, key, rng)
                u = HF.FUnit(u0.branch, u0.choices, u0.beta, inp)
                kind = _kind(u)
                if kind == "ee_const":
                    seen.add("ee_const")
                    continue
                qc = HF.circuit(m, u, DECODER, erase="none")
                tq = dev.transpile(qc, seed=11)
                ideal = float(R.evaluate(R.compile_circuit(tq, dev.snap, dev.params, [R.IDEAL]),
                                         HF.score(m, u, DECODER), tq.num_clbits)[0])
                lo, hi = SUPPORT[kind]
                if abs(ideal - hi) < 1e-9:
                    seen.add(f"{kind}/high")
                elif abs(ideal - lo) < 1e-9:
                    seen.add(f"{kind}/low")
                else:
                    raise PackageRefused(f"m={m} {key} {inp}: a non-deterministic ideal {ideal}")
            if len(seen) != 1:
                raise PackageRefused(f"m={m} {key} {inp}: its label draws fall in {sorted(seen)} -- the group is not "
                                     "a function of (stratum, input), so W_g cannot be read off the design law")
            g = seen.pop()
            W[g] = W.get(g, 0.0) + w / len(H.INPUTS)
            if erasure_orders != "folded":
                x = exact_stratum_weight(m, key, erasure_orders) / len(H.INPUTS)
                exact[g] = exact.get(g, Fraction(0)) + x
                cell = by_basis.setdefault(g, {})
                cell[inp[0]] = cell.get(inp[0], Fraction(0)) + x
    if abs(sum(W.values()) - 1) > 1e-12:
        raise PackageRefused(f"group weights sum to {sum(W.values())}, not 1")
    if erasure_orders == "folded":
        return W                     # the R190 record, as committed (its float sums sit a few ulps below the exact law)
    if sum(exact.values()) != 1:
        raise PackageRefused(f"exact group weights sum to {sum(exact.values())}, not 1")
    # R224: a HIGH group's weight is rounded UP to a double (understating one is anti-conservative, R190), a low one
    # down; the exact fractions travel with the record
    out = {g: _round_toward(x, up=g.endswith("/high")) for g, x in exact.items()}
    if max(abs(out[g] - W[g]) for g in W) > 1e-12:
        raise PackageRefused("the float and the exact group weights disagree beyond rounding")
    exact_group_weights.last_exact = {g: [x.numerator, x.denominator] for g, x in sorted(exact.items())}
    # R225: each group's weight split by its units' input basis -- a function of (stratum, input) like the group, so
    # exact too. The antidegradability witness weighs the bases differently and needs it.
    if any(sum(v.values()) != exact[g] for g, v in by_basis.items()):
        raise PackageRefused("a group's per-basis weights do not sum to its weight")
    exact_group_weights.last_exact_basis = {g: {b: [x.numerator, x.denominator] for b, x in sorted(v.items())}
                                            for g, v in sorted(by_basis.items())}
    return out


def _round_toward(x: Fraction, up: bool) -> float:
    f = float(x)
    if up and Fraction(f) < x:
        return math.nextafter(f, math.inf)
    if not up and Fraction(f) > x:
        return math.nextafter(f, -math.inf)
    return f


def exact_stratum_weight(m: int, key: tuple, erasure_orders: str) -> Fraction:
    """a faithful stratum's weight in F_avg as an exact rational (p_flip = 1/(2^(m/2) + 1) is rational for even m):
    branch weight x beta's 1/2 (kept branches) x the sector weights it names (1 - p for s0, p for s1)"""
    root = math.isqrt(2 ** m)
    if root * root != 2 ** m:
        raise PackageRefused(f"m = {m} is odd: p_flip is irrational, so the stratum weights are not rational")
    p = Fraction(1, root + 1)
    branch, s0, beta, s1 = key
    sector = {"s0": 1 - p, "s1": p}
    wb = {"kept_kept": Fraction(1, 4), "erased_erased": Fraction(1, 4),
          "kept_erased": Fraction(1, 2) if erasure_orders == "folded" else Fraction(1, 4),
          "erased_kept": Fraction(1, 4)}[branch]
    w = wb * sector[s0] * (sector[s1] if s1 is not None else 1)
    return w if branch == "erased_erased" else w / 2


def weights_key(m: int, erasure_orders: str = "folded") -> str:
    """the committed record's key: 'm' for the folded law (R190), 'm:both' for both erasure orders (R224)"""
    return str(m) if erasure_orders == "folded" else f"{m}:{erasure_orders}"


def committed_weights(m: int, erasure_orders: str = "folded") -> dict:
    rec = json.loads(WEIGHTS.read_text(encoding="utf-8")) if WEIGHTS.is_file() else {}
    key = weights_key(m, erasure_orders)
    if key not in rec:
        raise PackageRefused(f"no committed exact group weights for {key}: run `ion_run_package.py weights {m}` "
                             f"--erasure-orders {erasure_orders}")
    return rec[key]["W"]


def allocation(n: int, m: int = M, ebar: dict | None = None, erasure_orders: str = "folded") -> dict:
    """N single-shot draws over the HIGH groups, in proportion to width W sqrt(ebar) (Neyman), with W the EXACT
    design-law weights (`committed_weights`). ebar is the device model's high-group error rate: the committed
    certification.json for m = 6 (R186 part 3), or passed in (e.g. from r190_m_scan's faithful arm) for other m --
    it decides only the split, never the bound's validity"""
    W = committed_weights(m, erasure_orders)
    if ebar is None:
        if m != 6:
            raise PackageRefused(f"no model error rates for the allocation at m = {m}: pass ebar")
        rec = json.loads(CERT.read_text(encoding="utf-8"))["devices"][MODEL_RUN]["error_groups"]
        ebar = {g: v["ebar"] for g, v in rec.items() if g.endswith("/high")}
    high = {g: w for g, w in W.items() if g.endswith("/high")}
    if not high or set(high) != set(ebar):
        raise PackageRefused(f"high groups {sorted(high)} and their error rates {sorted(ebar)} do not match")
    width = {g: SUPPORT[g.split("/")[0]][1] - SUPPORT[g.split("/")[0]][0] for g in high}
    score = {g: width[g] * high[g] * math.sqrt(max(ebar[g], 1e-6)) for g in high}
    tot = sum(score.values())
    alloc = {g: max(1, int(round(n * s / tot))) for g, s in score.items()}
    return dict(alloc=alloc, groups={g: dict(W=high[g], width=width[g], W_source="exact design law (R190)")
                                     for g in high})


def _kind(u) -> str:
    if u.branch != "erased_erased":
        return "kept"
    return "ee_x" if u.inp[0] == "X" else "ee_const"


SUPPORT = {"kept": (0.0, 2.0), "ee_x": (0.0, 1.0)}


def _f_ideal(m: int = M) -> float:
    import hw_two_use as H
    return (2 * H.f_e_closed(m) + 1) / 3


def freeze(n: int, outdir: pathlib.Path, seed: int = 188, progress: bool = True, m: int = M,
           ebar: dict | None = None, alloc: dict | None = None, order=None, erasure_orders: str = "folded") -> dict:
    """draw, build, classify by the exact ideal outcome, export. Rejection sampling over the design law keeps every
    accepted unit distributed as w restricted to its group.

    alloc (R202): an explicit {group: units} in place of the Neyman split; it must name exactly the design's high
    groups and sum to n. order (R202): a function of the accepted units' groups (acceptance order) returning the
    permutation they are exported in; units are named after it, so names follow the order the device runs"""
    import hw_faithful as HF
    import r186_faithful_requirements as R

    from oneq.hw import freeze as F
    from oneq.hw import premise as PR
    plan = allocation(n, m, ebar, erasure_orders)
    if alloc is not None:
        if set(alloc) != set(plan["alloc"]):
            raise PackageRefused(f"the allocation names groups {sorted(alloc)}, the design's high groups are "
                                 f"{sorted(plan['alloc'])}")
        if sum(int(v) for v in alloc.values()) != n or any(int(v) < 1 for v in alloc.values()):
            raise PackageRefused(f"the allocation {alloc} must give every group a unit and sum to {n}")
        plan = dict(plan, alloc={g: int(v) for g, v in alloc.items()})
    need = dict(plan["alloc"])
    dev = R.make_device(DEVICE, m, DECODER)
    strata = R.faithful_strata(m, erasure_orders)
    w = np.array([s[1] for s in strata])
    rng = np.random.default_rng(seed)
    units, named, drawn = [], [], 0
    while any(v > 0 for v in need.values()):
        key = strata[int(rng.choice(len(strata), p=w / w.sum()))][0]
        u = R.sample_faithful(m, key, rng)
        drawn += 1
        kind = _kind(u)
        if kind == "ee_const":
            continue
        lo, hi = SUPPORT[kind]
        if need.get(f"{kind}/high", 0) <= 0:
            continue                                   # this group is full: the draw is rejected before evaluating
        qc = HF.circuit(m, u, DECODER, erase="none")
        tqc = dev.transpile(qc, seed=1000 + drawn)
        comp = R.compile_circuit(tqc, dev.snap, dev.params, [R.IDEAL])
        ideal = float(R.evaluate(comp, HF.score(m, u, DECODER), tqc.num_clbits)[0])
        if abs(ideal - hi) > 1e-9:
            if abs(ideal - lo) > 1e-9:
                raise PackageRefused(f"a {kind} unit with a non-deterministic ideal score {ideal}: the error-indicator "
                                     "design does not apply")
            continue                                   # a LOW unit: its errors only raise f; not sampled
        g = f"{kind}/high"
        need[g] -= 1
        name = f"u{len(units):05d}_{kind}"
        units.append(dict(name=name, group=g, key=[str(x) for x in key], branch=u.branch,
                          choices=[[c[0], list(c[1])] for c in u.choices], beta=int(u.beta),
                          inp=[u.inp[0], int(u.inp[1])], ideal_score=hi, n_clbits=qc.num_clbits))
        named.append((name, qc))
        if progress and len(units) % 100 == 0:
            print(f"   {len(units)} frozen ({drawn} drawn)", flush=True)
    bad = PR.check(dict(named), PREMISE)
    if bad:
        raise PackageRefused(f"the frozen circuits violate the structural premise: {bad}")
    if order is not None:
        perm = [int(i) for i in order([u["group"] for u in units])]
        if sorted(perm) != list(range(len(units))):
            raise PackageRefused("the order is not a permutation of the frozen units")
        units, named = [units[i] for i in perm], [named[i][1] for i in perm]
        for i, rec in enumerate(units):
            rec["name"] = f"u{i:05d}_{rec['name'].split('_', 1)[1]}"
        named = list(zip([u["name"] for u in units], named))
    outdir.mkdir(parents=True, exist_ok=True)
    qasm_lines = []
    for (name, qc), rec in zip(named, units):
        text = export_qasm(qc)
        rec["qasm_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        qasm_lines.append(json.dumps({"name": name, "qasm": text}))
    (outdir / "circuits.qasm.jsonl").write_text("\n".join(qasm_lines) + "\n", encoding="utf-8", newline="\n")
    frec, blob = F.freeze(named, {"prereg_sha256": "", "not_for_submission": True,
                                  "note": "a design package: re-freeze against the device's own compiler at submission"})
    (outdir / "circuits.qpy").write_bytes(blob)
    design = dict(schema="oneq.ion_run_package/1", m=m, d=2 ** m, decoder=DECODER, model_device=DEVICE,
                  delta=DELTA, bar_F_e=0.75, f_ideal=_f_ideal(m), groups=plan["groups"], allocation=plan["alloc"],
                  draws_examined=drawn, seed=seed, shots_per_circuit=1, structural_premise=PR.declaration(PREMISE),
                  qpy_sha256=frec["qpy_sha256"], units=units,
                  cost_estimate=hqc_estimate(named))
    if order is not None:
        design["order"] = getattr(order, "rule", "an explicit permutation of the accepted units")
    if erasure_orders != "folded":       # (absent for the folded law, so the R186-R203 packages re-freeze byte for byte)
        design["erasure_orders"] = erasure_orders
        design["erasure_marginals"] = HF.erasure_marginals(erasure_orders)
    (outdir / "design.json").write_text(json.dumps(design, indent=1) + "\n", encoding="utf-8", newline="\n")
    return design


def hqc_estimate(named: list) -> dict:
    """Quantinuum's model, as its own Helios costing page states it (docs.quantinuum.com, read 2026-10-19; the same
    weights as the Azure Quantum provider doc): HQC = 5 + C (N1q + 10 N2q + 5 N_M) / 5000 per program, C shots, where
    N_M counts measurements AND qubit initialisations (ion allocations). Corrected 2026-10-19 (R189): the R188 freeze
    counted measurements only (mean 8.6 instead of 36.6 at m = 6). On the LOGICAL counts, one program per circuit;
    the one-program parameterized-shot form is ion_param_shots.py, priced on native counts by its verifier"""
    tot, n1, n2, nm = 0.0, [], [], []
    for _, qc in named:
        # by ARITY, not by name: a name list missed `ch` (the cz2 decoder's), so the R188 freeze priced every CH as a
        # one-qubit gate (mean N2q 28.1 against the true 33.1)
        one = two = meas = 0
        for inst in qc.data:
            op = inst.operation
            if op.name == "barrier":
                continue
            if op.name == "measure":
                meas += 1
            elif op.num_qubits == 2:
                two += 1
            elif op.num_qubits == 1:
                one += 1
            else:
                raise PackageRefused(f"a {op.num_qubits}-qubit {op.name} has no price in the HQC model")
        meas += qc.num_qubits
        n1.append(one)
        n2.append(two)
        nm.append(meas)
        tot += 5 + (one + 10 * two + 5 * meas) / 5000
    return dict(model="HQC = 5 + C (N1q + 10 N2q + 5 N_M) / 5000 per program, C = 1, one program per circuit; "
                      "N_M = measurements + qubit allocations (Helios costing page, read 2026-10-19)",
                total_HQC=round(tot, 1), mean_N1q=float(np.mean(n1)), mean_N2q=float(np.mean(n2)),
                mean_Nm=float(np.mean(nm)),
                caveat="logical gate counts; the vendor's compile decides the native ones. The same circuits as ONE "
                       "parameterized-shot program: param_shots_receipt.json")


def regroup(design_dir: pathlib.Path) -> dict:
    """rewrite an existing package's groups[*].W with the EXACT design-law weights (`committed_weights`), keeping
    its units, allocation and widths: W enters only the analysis bound, never which circuits were frozen. Returns
    {group: (old W, exact W)}"""
    path = design_dir / "design.json"
    design = load(path)
    W = committed_weights(design["m"])
    changed = {}
    for g, v in design["groups"].items():
        changed[g] = (v["W"], W[g])
        v["W"] = W[g]
        v["W_source"] = "exact design law (R190)"
    path.write_text(json.dumps(design, indent=1) + "\n", encoding="utf-8", newline="\n")
    return changed


def recost(design_dir: pathlib.Path) -> dict:
    """re-run the cost producer on the FROZEN circuits (circuits.qasm.jsonl, hash-checked against design.json) and
    rewrite design.json's cost_estimate only: an artifact is only as current as the last run of its producer"""
    from qiskit import qasm2
    path = design_dir / "design.json"
    design = load(path)
    by_name = {}
    for line in (design_dir / "circuits.qasm.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_name[rec["name"]] = rec["qasm"]
    named = []
    for u in design["units"]:
        text = by_name.get(u["name"])
        if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != u["qasm_sha256"]:
            raise PackageRefused(f"{u['name']}: the exported QASM is missing or is not the frozen one")
        named.append((u["name"], qasm2.loads(text)))
    design["cost_estimate"] = hqc_estimate(named)
    path.write_text(json.dumps(design, indent=1) + "\n", encoding="utf-8", newline="\n")
    return design["cost_estimate"]


def load(design_path: pathlib.Path) -> dict:
    d = json.loads(design_path.read_text(encoding="utf-8"))
    if d.get("schema") != "oneq.ion_run_package/1":
        raise PackageRefused(f"{design_path.name} is not an ion-run design package")
    return d


def _unit(rec):
    import hw_faithful as HF
    choices = tuple((c[0], tuple(c[1])) for c in rec["choices"])
    return HF.FUnit(rec["branch"], choices, rec["beta"], (rec["inp"][0], rec["inp"][1]))


def bits_of(key: str, n_clbits: int, name: str = "") -> list:
    """a counts key (clbit 0 rightmost; spaces between registers allowed) -> [clbit 0, clbit 1, ...]. REFUSED unless
    it holds exactly n_clbits binary digits. A short, long or non-binary record is a defect of whatever produced it,
    never a shot. Found 2026-10-21 by the R201 adversary pass: the reader used to pad with zeros and truncate, so the
    EMPTY string scored as an error-free shot of u00000_kept"""
    s = str(key).replace(" ", "")
    if len(s) != n_clbits or set(s) - {"0", "1"}:
        raise PackageRefused(f"{name}: the record {key!r} is not {n_clbits} binary digits")
    return [int(ch) for ch in s[::-1]]


def analyse(design: dict, counts: dict) -> dict:
    """counts {name: {bitstring (clbit 0 rightmost): count}} -> per group (errors, shots), the exact Clopper-Pearson
    upper bounds, and the certified F_e lower bound. Every frozen circuit must have run; shots beyond one per circuit
    are refused (the exact-Bernoulli argument is per FRESH unit)"""
    import hw_faithful as HF
    import r186_certification as C
    if "sequential_batch" in design:
        # a batch carries the fixed package's delta; certifying it alone skips the union bound over the looks (R202)
        raise PackageRefused("a sequential batch is never analysed alone: analyse its parent package with "
                             "`ion_sequential.py analyse`, which applies delta / looks at every look")
    missing = [u["name"] for u in design["units"] if u["name"] not in counts]
    if missing:
        raise PackageRefused(f"{len(missing)} frozen circuits have no counts, e.g. {missing[:3]}")
    tally: dict = {g: [0, 0] for g in design["groups"]}
    for u in design["units"]:
        c = counts[u["name"]]
        if sum(c.values()) != design["shots_per_circuit"]:
            raise PackageRefused(f"{u['name']}: {sum(c.values())} shots, the design runs exactly "
                                 f"{design['shots_per_circuit']} per circuit")
        sc = HF.score(design["m"], _unit(u), design["decoder"])
        for bits, k in c.items():
            b = bits_of(bits, u["n_clbits"], u["name"])
            tally[u["group"]][0] += int(k) * int(abs(sc(b) - u["ideal_score"]) > 1e-9)
            tally[u["group"]][1] += int(k)
    a = design["delta"] / len(design["groups"])
    ucb = {g: C.cp_upper(e, n, a) for g, (e, n) in tally.items()}
    f_lo = design["f_ideal"] - sum(v["width"] * v["W"] * ucb[g] for g, v in design["groups"].items())
    F_lo = 1.5 * f_lo - 0.5
    return dict(errors={g: dict(errors=e, shots=n, rate=e / n if n else None, ucb=ucb[g]) for g, (e, n) in tally.items()},
                f_lower=f_lo, certified_F_e_lower=F_lo, claim_holds=bool(F_lo > design["bar_F_e"]),
                delta=design["delta"], method="exact Clopper-Pearson per high group, union bound; low groups at 0")


def dryrun(design: dict, reps: int = 50, seed: int = 7) -> dict:
    """each frozen circuit's error probability under the device MODEL (R186's exact evaluator at FULL), then reps
    synthetic single-shot experiments through `analyse`: the package end to end before any spend"""
    import hw_faithful as HF
    import r186_faithful_requirements as R
    dev = R.make_device(design["model_device"], design["m"], design["decoder"])
    e = []
    for i, u in enumerate(design["units"]):
        qc = HF.circuit(design["m"], _unit(u), design["decoder"], erase="none")
        if hashlib.sha256(export_qasm(qc).encode("utf-8")).hexdigest() != u["qasm_sha256"]:
            raise PackageRefused(f"{u['name']}: the rebuilt circuit is not the frozen one")
        tqc = dev.transpile(qc, seed=5000 + i)
        comp = R.compile_circuit(tqc, dev.snap, dev.params, [R.FULL])
        es = float(R.evaluate(comp, HF.score(design["m"], _unit(u), design["decoder"]), tqc.num_clbits)[0])
        lo, hi = SUPPORT[u["group"].split("/")[0]]
        e.append((u["ideal_score"] - es) / (hi - lo))
    e = np.clip(np.array(e), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    passes, lows = 0, []
    for _ in range(reps):
        err = rng.random(len(e)) < e
        res = _analyse_indicators(design, {u["name"]: bool(b) for u, b in zip(design["units"], err)})
        lows.append(res["certified_F_e_lower"])
        passes += res["claim_holds"]
    return dict(reps=reps, pass_rate=passes / reps, certified_F_e_median=float(np.median(lows)),
                model_error_rates={g: float(np.mean([x for x, u in zip(e, design["units"]) if u["group"] == g]))
                                   for g in design["groups"]})


def _analyse_indicators(design: dict, errors: dict) -> dict:
    """analyse() on error indicators directly (the dry run's synthetic shots)"""
    import r186_certification as C
    tally = {g: [0, 0] for g in design["groups"]}
    for u in design["units"]:
        tally[u["group"]][0] += int(errors[u["name"]])
        tally[u["group"]][1] += 1
    a = design["delta"] / len(design["groups"])
    f_lo = design["f_ideal"] - sum(v["width"] * v["W"] * C.cp_upper(*tally[g], a) for g, v in design["groups"].items())
    return dict(certified_F_e_lower=1.5 * f_lo - 0.5, claim_holds=bool(1.5 * f_lo - 0.5 > design["bar_F_e"]))


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["freeze", "analyse", "dryrun", "recost", "weights", "regroup"])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--seed", type=int, default=188)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--ebar", default="", help="freeze at m != 6: JSON {group: model error rate} for the split")
    ap.add_argument("--draws", type=int, default=8, help="weights: label draws checked per (stratum, input) cell")
    ap.add_argument("--erasure-orders", choices=["folded", "both"], default="folded",
                    help="weights / freeze: the one-erased law; 'both' is the one a hardware certificate needs (R224)")
    a = ap.parse_args(argv)
    if a.mode == "weights":
        m = int(a.a)
        W = exact_group_weights(m, draws=a.draws, erasure_orders=a.erasure_orders)
        rec = json.loads(WEIGHTS.read_text(encoding="utf-8")) if WEIGHTS.is_file() else {}
        rec[weights_key(m, a.erasure_orders)] = dict(W=W, draws_per_cell=a.draws, method="exact_group_weights: a unit's group is a function "
                           "of (stratum, input); every cell checked on fresh label draws")
        if a.erasure_orders != "folded":
            rec[weights_key(m, a.erasure_orders)].update(
                W_exact=exact_group_weights.last_exact,
                W_exact_by_basis=exact_group_weights.last_exact_basis,
                rounding="W is the exact rational weight rounded UP to a double for high groups, down for low ones")
        WEIGHTS.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(rec[weights_key(m, a.erasure_orders)], indent=1))
        return 0
    if a.mode == "regroup":
        print(json.dumps(regroup(pathlib.Path(a.a)), indent=1))
        return 0
    if a.mode == "freeze":
        ebar = json.loads(a.ebar) if a.ebar else None
        d = freeze(int(a.a), pathlib.Path(a.b), seed=a.seed, m=a.m, ebar=ebar, erasure_orders=a.erasure_orders)
        print(f"frozen {len(d['units'])} circuits {d['allocation']} from {d['draws_examined']} draws; cost "
              f"{d['cost_estimate']['total_HQC']} HQC (estimate) -> {a.b}")
        return 0
    if a.mode == "recost":
        print(json.dumps(recost(pathlib.Path(a.a)), indent=1))
        return 0
    design = load(pathlib.Path(a.a))
    if a.mode == "analyse":
        res = analyse(design, json.loads(pathlib.Path(a.b).read_text(encoding="utf-8")))
    else:
        res = dryrun(design, reps=a.reps)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
