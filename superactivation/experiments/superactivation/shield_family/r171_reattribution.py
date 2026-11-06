"""R171 (item 17): R152's per-unit attribution RE-RUN on the residuals of the CURRENT best models, with the server's
single-qubit template as a location class of its own (EXPLORATORY: outside every preregistered verdict).

R152's attribution (r152_attribution.py) regressed every unit's residual on its first-order loss per error source
(oneq.hw.sensitivity) on the V0 base and found the excess at single-qubit-gate locations, ~4x the vendor sx error,
collinear with CZ (0.99) on R146. Since then the base moved: V3c (vendor sx, R155's white T2 + sigma, coherent
windows) and the twirl DRESSING at R165's 1.54 leave R146 at -0.023 held out. This module asks where that remainder
sits, on the same operator, with two additions the base now needs:

1. EVERY loss class the budget carries. r152_attribution.CLASSES = (cz, sx, idle, readout) predates the dressing
   and the quasi-static windows: its `losses` adds a budget entry to a class only `if key[0] in out`, so on today's
   model the dressing's loss would be dropped from every class fit, silently. class_losses here takes every class
   and refuses a budget key it does not know.
2. THE 1q SLOT (R166, the server's executed schedule, jobs dba0ds8c77vs73824c3g / dba0dufi0auc7391ap5g): with
   twirling on, the server plays every single-qubit run as a fixed two-sx template, whatever was compiled -- an H
   compiled with one sx plays two, an rz-only slot plays two, and every twirled CZ layer's qubits get a slot before
   each layer and after the last. So a qubit with n CZs owns n + 1 slots whether or not the compiled circuit put a
   gate in them; a qubit with no CZ owns one slot per run. The engine charges only the COMPILED pulses. A slot's
   error is a depolarizing channel of 2 pulses (2 x 1.5 x the qubit's sx error): it commutes with every 1q unitary,
   so placing it anywhere inside the slot is the same channel. Its first-order loss needs the linear response at a
   location where the engine has NO error row (an empty or rz-only slot): slot_derivatives propagates each slot's
   X, Y and Z through the engine's own plan (its conjugation tables; a blocked one by one statevector run), exactly
   as oneq.hw.sensitivity does for the rows it has. Nothing is added to the engine: the lead is putting the
   re-synthesis into oneq.hw.noise; this is the measurement of where it should matter.

Models (nothing fitted to R146 or R147), each an ARM of main:
  r146                    V3c + dressing x1.54 (R165), the fe146 rule of r152_posthoc.predictors (its per-unit
                          random streams), with the slot class
  r146_template           V3c + the template IN the engine (NoiseParams.twirl_template, the lead's
                          noise.template_resynthesis, after R166) + dressing x2 (two sx per idle layer, as R166
                          scheduled): where the remainder sits once the template is modelled, not regressed
  r147_physical           V3c + dressing x1.54 on R147's stand-ins with the hold at R166's measured 2.7 us (read
                          start to the feed-forward), no global stall -- R166 measured that a qubit independent of
                          the conditional is NOT stalled; with the slot class
  r147_template           the same stand-ins on the r146_template model
  r147_stall_sensitivity  R167 b's global sync + stall at L = 0.85 us (r167_global_stall_heldout.stalled_stand_ins
                          with r155_heldout_r152.with_hold) on V3c + dressing x1.54 -- what R166 falsified as the
                          server's behaviour, kept as a what-if
Every arm re-derives a committed held-out number on the same streams (reproduces): R165's x1.54, R166's template
x2 (the lead's r166_template_heldout.py), R167 b's L = 0.85.

  python r171_reattribution.py REPO_ROOT [--traj N] [--limit N]
      -> evidence/superactivation/r171_residual/reattribution_r152.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import r152_attribution as A  # noqa: E402  -- fit, fe_shift, _mix: the R152 operator, reused

from oneq.hw import engine as E  # noqa: E402
from oneq.hw import sensitivity as SE  # noqa: E402

CLASSES = ("cz", "sx", "idle", "readout", "dressing", "quasistatic")
KNOWN_BUDGET = set(CLASSES) | {"spectator", "crosstalk"}
GATE_1Q = ("sx", "x", "rz")
PULSES_PER_SLOT = 2                      # R166: every single-qubit run plays as exactly two sx
MODEL_KW = dict(quasistatic="coherent", twirl="active-accum", dressing_factor=1.54)
# after R166: the server's template in the engine (noise.template_resynthesis, the lead's) + two sx per idle layer
TEMPLATE_KW = dict(quasistatic="coherent", twirl="active-accum", dressing_factor=2.0, twirl_template=True)
HOLD_MEASURED_S = 2.7e-6                 # R166: the feed-forward lands 2.7 us after its read starts
STALL_SENSITIVITY_US = 0.85              # R167 b's middle R155 stall (a what-if since R166)
OUT = pathlib.Path("evidence/superactivation/r171_residual")


class SlotError(ValueError):
    """a slot cannot be given a faithful linear response"""


# ------------------------------------------------------------------ the server's 1q template (pure)
def slots_of(ops, virtual=frozenset()) -> list:
    """per active qubit, its 1q SLOTS as the server's twirl template plays them (module docstring, R166).

    ops: the engine's compiled op list (noise.Op; only cz / sx / x / rz are GATES here -- idles and inserted errors
    are not). virtual: the CZs (frozensets of two active qubits) that are not twirled layers -- a stand-in's
    zero-duration virtual edge (noise.template_resynthesis draws the same line) -- skipped as if absent. A qubit with
    CZs owns n_cz + 1 slots: 'pre' (before its first CZ), 'between' (one per consecutive pair of its CZs), 'post'
    (after its last); a qubit with no CZ owns one 'free' slot holding all its gates. Each slot:
      a        the active qubit
      kind     pre | between | post | free
      gates    compiled 1q gates inside it (rz counted)
      pulses   compiled sx / x inside it (what the engine charges today)
      after    the op index the slot's error is placed right after: the slot's last gate, else the CZ that opens it,
               else -1 (the slot precedes every op: an empty 'pre' slot of a qubit whose first CZ is op 0)
    """
    seq: dict = {}
    for i, op in enumerate(ops):
        if op.name == "cz" and frozenset(op.qubits) in virtual:
            continue
        if op.name == "cz" or op.name in GATE_1Q:
            for a in op.qubits:
                seq.setdefault(a, []).append((i, op.name))
    out = []
    for a in sorted(seq):
        s = seq[a]
        czs = [k for k, (_, nm) in enumerate(s) if nm == "cz"]
        if not czs:
            out.append(dict(a=a, kind="free", gates=len(s), pulses=sum(nm in ("sx", "x") for _, nm in s),
                            after=s[-1][0]))
            continue
        bounds = [-1] + czs + [len(s)]
        for j in range(len(bounds) - 1):
            inside = s[bounds[j] + 1:bounds[j + 1]]
            kind = "pre" if j == 0 else ("post" if j == len(bounds) - 2 else "between")
            if inside:
                after = inside[-1][0]
            elif j > 0:
                after = s[bounds[j]][0]                    # right after the CZ that opens the slot
            else:
                after = s[czs[0]][0] - 1                   # right before the first CZ (-1: before every op)
            out.append(dict(a=a, kind=kind, gates=len(inside), pulses=sum(nm in ("sx", "x") for _, nm in inside),
                            after=after))
    return out


def _propagate(P, start: int, x: np.ndarray, z: np.ndarray) -> bool:
    """push the Pauli (x, z bit vectors over the active qubits, IN PLACE) through ops[start:] with the plan's own
    conjugation tables -- the loop of engine.plan for one row. Returns True when a gate blocks it (not a Pauli)."""
    return any(_propagate_one(P, j, x, z) for j in range(start, len(P.ops)))


def _flip(P, x: np.ndarray) -> int:
    m = len(P.meas)
    f = 0
    for j, (a, _) in enumerate(P.meas):
        f |= int(x[a]) << (m - 1 - j)
    return f


def slot_derivatives(P, g_hat: np.ndarray, sv, slots: list) -> list:
    """dS/dp of a DEPOLARIZING error (X, Y, Z at p/3) at each slot, from the engine's own plan P: the same linear
    response oneq.hw.sensitivity gives for an existing error row, at a location that has none. Reachable: EXACT,
    2^-m sum_s chi(s) q^(s) g^(s) (sigma(s) - 1); blocked by a later non-Clifford gate: one statevector run, value
    with the Pauli applied minus value without (first order in the blocked mass, as sensitivity). chi is the product
    of EVERY op's reachable factor: the slot is a new location, not part of any op's own channel."""
    k, m = len(P.active), len(P.meas)
    factors, _ = E._op_factors(P, len(P.ops), m)
    chi = np.ones(1 << m)
    for f in factors.values():
        chi = chi * f
    q_hat = E.wht(E._true_probs(sv.run(P.ops, {}), P.meas))
    A_ = q_hat * g_hat / (1 << m)
    base = float(chi @ A_)
    out = []
    for sl in slots:
        a = sl["a"]
        d = []
        for g in ("X", "Y", "Z"):
            x = np.zeros(k, dtype=np.int64)
            z = np.zeros(k, dtype=np.int64)
            x[a], z[a] = int(g in "XY"), int(g in "YZ")
            at = sl["after"]
            paulis = [(a, g)]
            if at < 0:                                      # before op 0: conjugate through op 0, then after it
                if _propagate_one(P, 0, x, z):
                    raise SlotError("a slot before op 0 whose first op does not keep it a Pauli")
                at = 0
                paulis = [(q, "IZXY"[2 * int(x[q]) + int(z[q])]) for q in range(k) if x[q] or z[q]]
            blocked = _propagate(P, at + 1, x, z)
            if not blocked:
                fl = _flip(P, x)
                if fl == 0:
                    d.append(0.0)
                    continue
                signs = E._parity_signs(np.array([fl], dtype=np.int64), m)[0]
                d.append(float((chi * A_) @ (signs - 1.0)))
            else:
                qh = E.wht(E._true_probs(sv.run(P.ops, {at: paulis}), P.meas))
                d.append(float(chi @ (qh * g_hat)) / (1 << m) - base)
        out.append(dict(sl, dS_dp=sum(d) / 3.0, dS=dict(zip("XYZ", d))))
    return out


def _propagate_one(P, j: int, x: np.ndarray, z: np.ndarray) -> bool:
    """through op j alone (the conjugation of a Pauli placed BEFORE op j into one placed after it)"""
    tab = P.tables[j]
    if tab is None:
        return False
    qs = P.ops[j].qubits
    if len(qs) == 1:
        q = qs[0]
        new = int(tab[2 * x[q] + z[q]])
        if new < 0:
            return True
        x[q], z[q] = (new >> 1) & 1, new & 1
        return False
    q0, q1 = qs
    new = int(tab[4 * (2 * x[q0] + z[q0]) + (2 * x[q1] + z[q1])])
    if new < 0:
        return True
    c0, c1 = new >> 2, new & 3
    x[q0], z[q0], x[q1], z[q1] = (c0 >> 1) & 1, c0 & 1, (c1 >> 1) & 1, c1 & 1
    return False


def slot_sensitivity(qc, score, snap, params, prefer: str = "numpy") -> tuple:
    """(the plan's active qubits, every slot of the circuit with dS/dp) on the model (snap, params)"""
    P = E.plan(qc, snap, params)
    k, m = len(P.active), len(P.meas)
    if k > E.MAX_ACTIVE or m > E.MAX_MEASURED:
        raise E.EngineError("circuit beyond the engine's limits")
    sv = E.simulator(k, prefer)
    g_hat = E.wht(E._smoothed_score(E.score_vector(score, m), E._confusions(P.meas, snap, P.active, params)))
    virtual = frozenset(frozenset(op.qubits) for op in P.ops if op.name == "cz"
                        and snap.edge(*(P.active[a] for a in op.qubits))[1] <= 0)
    return P.active, slot_derivatives(P, g_hat, sv, slots_of(P.ops, virtual))


# ------------------------------------------------------------------ losses (pure given the responses)
def class_losses(sens, snap, params, active=None, slots=None, edges: bool = True, qubits: bool = True) -> dict:
    """a unit's first-order loss per source CLASS (every class the budget carries: CLASSES), per CZ edge, per qubit and
    per (class, qubit) -- r152_attribution.losses' operator with nothing dropped -- plus, when slots are given, the 1q
    TEMPLATE class: each slot's p_slot dS/dp with p_slot = PULSES_PER_SLOT x 1.5 x sx_factor x the qubit's sx error.
    '_value' is the linear response's own expectation. A budget key outside KNOWN_BUDGET is refused, never dropped."""
    out = dict.fromkeys(CLASSES, 0.0)
    out["_value"] = sens.value

    def add(key, v):
        out[key] = out.get(key, 0.0) + v
    for key, v in sens.loss_budget.items():
        if key[0] not in KNOWN_BUDGET:
            raise ValueError(f"a loss-budget class this attribution does not know: {key!r}")
        add(key[0], v)
        if edges and key[0] == "cz":
            add(f"cz_{key[1]}_{key[2]}", v)
        if qubits and key[0] != "cz" and len(key) == 2:
            add(f"{key[0]}@q{key[1]}", v)
    if qubits:
        for _, _, _, paulis, w, ds in sens.per_error:
            hit = [q for q, g in paulis if g != "I"]
            for q in hit:
                add(f"q{q}", w * ds / len(hit))
    for key, v in sens.components.items():                  # readout: p01 dS/dp01 + p10 dS/dp10, each linear
        if key[0] in ("p01", "p10"):
            q = key[1]
            p = (snap.p01 if key[0] == "p01" else snap.p10)[q] * params.readout_factor
            add("readout", p * v)
            if qubits:
                add(f"q{q}", p * v)
                add(f"readout@q{q}", p * v)
    if slots is not None:
        out["slot"] = 0.0
        for sl in slots:
            q = active[sl["a"]]
            e = PULSES_PER_SLOT * 1.5 * params.sx_factor * snap.sx_error[q]
            e = 0.0 if not math.isfinite(e) else min(e, 0.75)
            add("slot", e * sl["dS_dp"])
            if qubits:
                add(f"slot@q{q}", e * sl["dS_dp"])
    return out


def played(qc, snap, params) -> tuple:
    """(circuit, params) with the server's template applied ONCE (noise.template_resynthesis) and the params asking
    for it no more: exactly what compile_ops does first under twirl_template, so every later engine call on the pair
    gives the same numbers without re-synthesizing. A model without the template is returned as it is."""
    from oneq.hw import noise as N
    if not getattr(params, "twirl_template", False):
        return qc, params
    dress = getattr(params, "twirl_template_dress", False)
    out = N.template_resynthesis(qc, snap, dress_idle=dress) if dress else N.template_resynthesis(qc, snap)
    d = {**params.describe(), "twirl_template": False}
    d.pop("twirl_template_dress", None)
    return out, N.NoiseParams(**d)


def evidence_text(out: dict) -> str:
    """the record as JSON: every summary indented for reading, every row of out['rows'] (a dict of lists) on ONE
    compact line -- the rows are most of the bytes. Round-trips exactly (checked before it is returned)."""
    head = {k: v for k, v in out.items() if k != "rows"}
    text = json.dumps(head, indent=1, default=float)
    if "rows" in out:
        arms = []
        for arm, rr in out["rows"].items():
            body = ",\n".join(json.dumps(r, separators=(",", ":"), default=float) for r in rr)
            arms.append(f"  {json.dumps(arm)}: [\n{body}\n  ]")
        text = (text[:-2] + ",\n" if head else "{\n") + ' "rows": {\n' + ",\n".join(arms) + "\n }\n}"
    back = json.loads(text)
    if json.dumps(back, sort_keys=True, default=float) != json.dumps(json.loads(json.dumps(out, default=float)),
                                                                      sort_keys=True):
        raise ValueError("the evidence text does not round-trip")
    return text + "\n"


def provenance(root: pathlib.Path) -> dict:
    """the sources that made the numbers: sha256 of the engine and these modules as RUN, and the commit they sit on"""
    import hashlib
    import subprocess
    files = ["src/oneq/hw/noise.py", "src/oneq/hw/engine.py", "src/oneq/hw/sensitivity.py",
             "experiments/superactivation/shield_family/r171_reattribution.py",
             "experiments/superactivation/shield_family/r171_syndromes.py",
             "experiments/superactivation/shield_family/r152_attribution.py"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL).stdout.strip()
    out = dict(head=head, sha256={})
    for f in files:
        p = root / f
        if p.exists():
            here = hashlib.sha256(p.read_bytes()).hexdigest()
            at_head = subprocess.run(["git", "show", f"HEAD:{f}"], cwd=root, capture_output=True,
                                     stdin=subprocess.DEVNULL).stdout
            out["sha256"][f] = dict(run=here, equals_head=bool(at_head) and hashlib.sha256(at_head).hexdigest() == here)
    return out


def slot_counts(slots: list) -> dict:
    """per circuit: template slots, those the compiled circuit left EMPTY or rz-only (the engine charges nothing
    there), compiled pulses inside slots"""
    return dict(slots=len(slots), empty=sum(s["gates"] == 0 for s in slots),
                rz_only=sum(s["gates"] > 0 and s["pulses"] == 0 for s in slots),
                no_pulse=sum(s["pulses"] == 0 for s in slots), compiled_pulses=sum(s["pulses"] for s in slots),
                template_pulses=PULSES_PER_SLOT * len(slots))


def mix_counts(a: dict, b: dict, r: float) -> dict:
    return {k: (1 - r) * a[k] + r * b[k] for k in a}


def residual_vs(rows: list, key: str) -> dict:
    """WLS slope of the per-unit residual (measured - predicted) on one per-unit regressor `key` (in rows[i]['x']),
    with an intercept: does the residual TRACK it? slope, sd, z, and the weighted correlation"""
    x = np.array([r["x"][key] for r in rows], dtype=float)
    y = np.array([r["measured"] - r["predicted"] for r in rows])
    w = np.array([1 / r["var"] for r in rows])
    X = np.column_stack([np.ones_like(x), x])
    Xw, yw = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.pinv(Xw.T @ Xw)
    chi0 = float(((yw - np.sqrt(w) * np.average(y, weights=w)) ** 2).sum())
    chi1 = float(((yw - Xw @ beta) ** 2).sum())
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    sxy = float(np.sum(w * (x - xm) * (y - ym)))
    sxx, syy = float(np.sum(w * (x - xm) ** 2)), float(np.sum(w * (y - ym) ** 2))
    sd = float(math.sqrt(max(cov[1, 1], 0.0)))
    return dict(key=key, n=len(rows), slope=float(beta[1]), sd=sd, z=float(beta[1] / sd) if sd > 0 else 0.0,
                intercept=float(beta[0]), chi2_mean_only=chi0, chi2_with_slope=chi1,
                weighted_corr=sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0,
                x_range=[float(x.min()), float(x.max())])


def compact(d: dict, digits: int = 6) -> dict:
    return {k: (float(f"{v:.{digits}g}") if isinstance(v, float) else v) for k, v in d.items()}


# ------------------------------------------------------------------ rows (engine runs)
def unit_losses(qc, score, snap, params, with_slots: bool = True) -> tuple:
    """(losses, slot counts). with_slots False: the model already plays the template (NoiseParams.twirl_template),
    so the slot class would count it twice -- only the counts of the circuit AS COMPILED are kept"""
    s = SE.sensitivity(qc, score, snap, params)
    if with_slots:
        active, slots = slot_sensitivity(qc, score, snap, params)
        return class_losses(s, snap, params, active, slots), slot_counts(slots)
    from oneq.hw import noise as N
    plain = N.NoiseParams(**{**params.describe(), "twirl_template": False})
    P = E.plan(qc, snap, plain)
    virtual = frozenset(frozenset(op.qubits) for op in P.ops if op.name == "cz"
                        and snap.edge(*(P.active[a] for a in op.qubits))[1] <= 0)
    return class_losses(s, snap, params), slot_counts(slots_of(P.ops, virtual))


def r146_rows(art, P, model, params, traj, limit=None, with_slots=True) -> tuple:
    import hw_superact_run as HS
    import hw_two_use as H
    from oneq.hw import freeze as F
    from oneq.hw import stats as T
    d, r146 = art["dir"], art["parts"]["r146"]
    c146 = F.load(r146, (d / "r146.qpy").read_bytes(), check_sources=False)
    stat = [T.Unit(u["name"], u["weight"], u["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in r146["units"]]
    scores = {u["name"]: H.score(r146["m"], HS.unit_from(u["unit"])) for u in r146["units"]}
    tallies = T.tally_all(stat, {u["name"]: art["counts"]["r146__" + u["name"]] for u in r146["units"]}, scores)
    rows = []
    for k, u in enumerate(r146["units"]):
        if limit is not None and k >= limit:
            break
        n = u["name"]
        pred = E.expectation(c146[n], scores[n], model, params, traj, np.random.default_rng([11, k]))
        t = tallies[n]
        L, cnt = unit_losses(c146[n], scores[n], model, params, with_slots)
        L.pop("_value")
        cz = sum(1 for i in c146[n].data if i.operation.name == "cz")
        rows.append(dict(name=n, kind="p0", stratum=u["stratum"], weight=u["weight"], measured=t.mean,
                         predicted=pred.mean, var=max(t.s2 / t.n - t.mean ** 2, 1e-6) / t.n + pred.se ** 2,
                         x=dict(cnt, cz=cz), L=compact(L)))
    units = [u for u in r146["units"] if u["name"] in {r["name"] for r in rows}]
    est = HS.estimate({r["name"]: r["predicted"] for r in rows}, units)
    return rows, dict(F_e_predicted=est["F_e"])


def r147_rows(art, P, model, params, stand_ins, traj, limit=None, with_slots=True) -> tuple:
    import hw_dynamic as D
    import hw_dynamic_run as R147
    import hw_two_use as H
    from oneq.hw import freeze as F
    from oneq.hw import stats as T
    d, r147 = art["dir"], P["record147"]
    c147 = F.load(r147, (d / "r147.qpy").read_bytes(), check_sources=False)
    names = None if limit is None else {u["name"] for u in r147["units"][:limit]}
    preds = R147.predict_units(r147, c147, stand_ins, model, params, traj=traj, names=names)
    stat = [T.Unit(u["name"], u["weight"], r147["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in r147["units"]]
    tallies = T.tally_all(stat, {u["name"]: art["counts"]["r147__" + u["name"]] for u in r147["units"]},
                          {u["name"]: R147._score(u) for u in r147["units"]})
    rows = []
    for u in r147["units"]:
        n, inp = u["name"], u["prediction_inputs"]
        if names is not None and n not in names:
            continue
        uu = R147.unit_from(u["unit"])
        score = R147.FA.score(R147.M, uu, R147.DECODER) if inp["score"] == "FA" else H.score(R147.M, uu)
        if inp["kind"] == "dynamic":
            sn = D.augmented(model, [tuple(e) for e in inp["virt"]])
            r = 0.5 * (1 - math.prod(1 - 2 * model.readout_error(q) for q in inp["mix_qubits"]))
            L0, n0 = unit_losses(stand_ins[inp["c0"]], score, sn, params, with_slots)
            L1, n1 = unit_losses(stand_ins[inp["c1"]], score, sn, params, with_slots) if inp["c1"] else (L0, n0)
            L = A._mix(L0, L1, r)
            L["mcm_readout"] = r * (L1["_value"] - L0["_value"])
            cnt = mix_counts(n0, n1, r)
            cz = sum(1 for i in stand_ins[inp["c0"]].data if i.operation.name == "cz"
                     and tuple(sorted(stand_ins[inp["c0"]].find_bit(x).index for x in i.qubits))
                     not in {tuple(sorted(e)) for e in inp["virt"]})
        else:
            L, cnt = unit_losses(c147[n], score, model, params, with_slots)
            L["mcm_readout"] = 0.0
            cz = sum(1 for i in c147[n].data if i.operation.name == "cz")
        L.pop("_value")
        t = tallies[n]
        rows.append(dict(name=n, kind=u["kind"], stratum=u["stratum"], weight=u["weight"], measured=t.mean,
                         predicted=preds[n][0], var=max(t.s2 / t.n - t.mean ** 2, 1e-6) / t.n + preds[n][1] ** 2,
                         x=dict(cnt, cz=cz), L=compact(L)))
    summary: dict = {}
    if names is None:                                    # the record's own estimator needs every unit
        an = R147.analyse(R147.with_predictions(r147, preds, model),
                          {nm: art["counts"]["r147__" + nm] for nm in r147["circuit_order"]})
        kinds: dict = {}
        for v in an["per_unit"].values():
            kinds.setdefault(v["kind"], []).append(v["measured"] - v["predicted"])
        summary = dict(F_e_predicted=an["predicted"]["F_e"], residual={k: float(np.mean(v)) for k, v in kinds.items()})
    return rows, summary


# ------------------------------------------------------------------ fits (pure over rows)
def fits_for(rows: list, extra: tuple = ()) -> dict:
    """the class fits a protocol's rows are read by: every class jointly; + the template slot; the slot REPLACING
    the compiled sx (the server's synthesis); each class alone; per edge; per qubit; per qubit slot"""
    base = list(CLASSES) + list(extra)
    present = [k for k in base if any(abs(r["L"].get(k, 0.0)) > 0 for r in rows)]
    slot = any(abs(r["L"].get("slot", 0.0)) > 0 for r in rows)
    out = dict(classes=A.fit(rows, present))
    if slot:                                               # a model that plays the template has no slot class
        out.update(classes_slot=A.fit(rows, present + ["slot"]),
                   slot_for_sx=A.fit(rows, [k for k in present if k != "sx"] + ["slot"]))
    for one in present + (["slot"] if slot else []):
        out[f"only_{one}"] = A.fit(rows, [one])
    e = sorted({k for r in rows for k in r["L"] if k.startswith("cz_")})
    q = sorted({k for r in rows for k in r["L"] if k[0] == "q" and k[1:].isdigit()}, key=lambda s: int(s[1:]))
    sq = sorted({k for r in rows for k in r["L"] if k.startswith("slot@q")}, key=lambda s: int(s[6:]))
    for name, keys, rest in (("edges", e, [k for k in present if k != "cz"]),
                             ("qubits", q, [k for k in extra if k in present]),
                             ("slot_qubits", sq, [k for k in present if k != "sx"])):
        if keys:                                           # a per-location fit needs locations
            out[name] = A.fit(rows, keys + rest)
    return out


def template_shift(rows: list) -> dict:
    """the server's synthesis to first order, NOTHING fitted: every compiled pulse removed (beta_sx = -1) and every
    template slot charged (beta_slot = +1). Per protocol F_e (main / p0 units) and per kind, against the measured
    residual"""
    beta = {"sx": -1.0, "slot": 1.0}
    has = any("slot" in r["L"] for r in rows)
    out = dict(F_e_shift=A.fe_shift(rows, beta) if has else None,
               F_e_slot_only=A.fe_shift(rows, {"slot": 1.0}) if has else None,
               F_e_sx_removed=A.fe_shift(rows, {"sx": -1.0}) if has else None,
               F_e_residual=1.5 * sum(r["weight"] * (r["measured"] - r["predicted"]) for r in rows
                                      if r["kind"] in ("p0", "main")))
    kinds: dict = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    out["per_kind"] = {k: dict(n=len(v), mean_residual=float(np.mean([r["measured"] - r["predicted"] for r in v])),
                               mean_template_shift=float(np.mean([sum(b * r["L"].get(c, 0.0) for c, b in beta.items())
                                                                  for r in v])),
                               mean_slot_loss=float(np.mean([r["L"].get("slot", 0.0) for r in v])),
                               mean_sx_loss=float(np.mean([r["L"].get("sx", 0.0) for r in v])))
                       for k, v in kinds.items()}
    return out


def tracking(rows: list) -> dict:
    """does the residual track the slot count -- and does it track the slot count BETTER than the CZ count or the
    compiled pulses (the regressors are collinear; each is reported alone and their correlations)"""
    keys = ("slots", "empty", "no_pulse", "compiled_pulses", "cz")
    out = {k: residual_vs(rows, k) for k in keys}
    X = np.array([[r["x"][k] for k in keys] for r in rows], dtype=float)
    out["correlation"] = dict(keys=list(keys), matrix=np.corrcoef(X.T).round(3).tolist())
    return out


def design_split(rows: list, design_of: dict) -> dict:
    """the residual by (kind, design, sector): R147's dynamic units sit on layout 0, its static ones on layout 1, so
    'dynamic vs static' is also 'layout 0 vs layout 1'. design_of: unit name -> 'dynamic' | 'isa'"""
    groups: dict = {}
    for r in rows:
        sector = r["stratum"].split("|")[1] if "|" in r["stratum"] else r["stratum"]
        groups.setdefault(f"{r['kind']}|{design_of.get(r['name'], 'isa')}|{sector}", []).append(r)
    out = {}
    for key, v in sorted(groups.items()):
        res = np.array([x["measured"] - x["predicted"] for x in v])
        var = np.array([x["var"] for x in v])
        out[key] = dict(n=len(v), mean_residual=float(res.mean()), sd=float(math.sqrt(var.sum()) / len(v)),
                        chi2_per_unit=float((res ** 2 / var).sum() / len(v)))
    return out


def twin_gaps(rows: list) -> dict:
    """R147's matched twins (the same unit as the early main circuit, the inline circuit, the static circuit): per
    pair the main - twin difference, measured (sd from the pairs' spread) against predicted"""
    by: dict = {}
    for r in rows:
        if r["kind"] in ("twin_inline", "twin_static"):
            by.setdefault(r["name"].split("__")[1], {})[r["kind"]] = r
    for r in rows:
        if r["kind"] == "main" and r["name"].split("__")[0] in by:
            by[r["name"].split("__")[0]]["main"] = r
    out = {}
    for other in ("twin_inline", "twin_static"):
        g = [p for p in by.values() if "main" in p and other in p]
        if len(g) < 2:
            continue
        dm = np.array([p["main"]["measured"] - p[other]["measured"] for p in g])
        dp = np.array([p["main"]["predicted"] - p[other]["predicted"] for p in g])
        sd = float(dm.std(ddof=1) / math.sqrt(len(g)))
        out[f"main_minus_{other}"] = dict(n=len(g), measured=float(dm.mean()), sd=sd, predicted=float(dp.mean()),
                                          z=float((dm.mean() - dp.mean()) / sd) if sd > 0 else None)
    return out


def class_qubit_fits(rows: list) -> dict:
    """per class, one factor per QUBIT for that class (key 'class@qN') with every other class as one factor: where a
    class's excess or over-charge sits (read only the strongest; these designs are ill-conditioned)"""
    out = {}
    for cls in ("idle", "readout", "sx", "dressing"):
        keys = sorted({k for r in rows for k in r["L"] if k.startswith(f"{cls}@q")}, key=lambda s: int(s.split("@q")[1]))
        others = [k for k in CLASSES if k != cls and any(abs(r["L"].get(k, 0.0)) > 0 for r in rows)]
        if keys:
            out[cls] = A.fit(rows, keys + others)
    return out


def summarize(rows: list, extra=(), design_of: dict | None = None) -> dict:
    out = dict(n=len(rows), fits=fits_for(rows, extra), template=template_shift(rows), tracking=tracking(rows),
               class_qubits=class_qubit_fits(rows))
    if design_of is not None:
        out.update(design_split=design_split(rows, design_of), twins=twin_gaps(rows))
    return out


# ------------------------------------------------------------------ error bars that count the twirl's frames
def frame_bootstrap(shots_by_pub: dict, estimator, n_boot: int = 300, seed: int = 171) -> dict:
    """per statistic: the estimate, its sd over resampled RANDOMIZATIONS (each circuit's 8 frames with replacement)
    and over resampled SHOTS by the same route, and their ratio (oneq.hw.randomization, canonical: R176). The
    predictions and the fit weights stay fixed; only the measured side is resampled."""
    from oneq.hw import randomization as RZ
    fr = RZ.bootstrap(estimator, shots_by_pub, n_boot=n_boot, seed=seed, mode="randomization")
    sh = RZ.bootstrap(estimator, shots_by_pub, n_boot=n_boot, seed=seed + 1, mode="shot")
    infl = RZ.inflation(fr, sh)
    return {k: dict(estimate=fr.estimate[k], sd_frames=fr.sd[k], sd_shots=sh.sd[k], inflation=infl[k])
            for k in fr.estimate}


def unit_specs(art) -> dict:
    """unit name -> (PUB name, score, n_bits) for R146 p0 and R147 units: how a unit's measured mean is tallied"""
    import hw_dynamic_run as R147
    import hw_superact_run as HS
    import hw_two_use as H
    out = {}
    r146 = art["parts"]["r146"]
    for u in r146["units"]:
        out[u["name"]] = ("r146__" + u["name"], H.score(r146["m"], HS.unit_from(u["unit"])), u["n_bits"])
    for u in art["parts"]["r147"]["units"]:
        out[u["name"]] = ("r147__" + u["name"], R147._score(u), u["n_bits"])
    return out


def means_from(counts_by_pub: dict, specs: dict, names, cache: dict | None = None) -> dict:
    """each unit's mean score from (resampled) counts; `cache` keeps each (unit, outcome)'s score across resamples"""
    from oneq.hw import stats as T
    cache = {} if cache is None else cache
    out = {}
    for n in names:
        pub, score, nb = specs[n]
        c = counts_by_pub[pub]
        tot = sum(c.values())
        acc = 0.0
        for k, v in c.items():
            s = cache.get((n, k))
            if s is None:
                s = cache[(n, k)] = float(score(T.bits_of(k, nb)))
            acc += v * s
        out[n] = acc / tot
    return out


def arm_statistics(rows: list, means: dict, design_of: dict | None) -> dict:
    """the headline numbers of one arm with every unit's MEASURED mean replaced by `means` (weights fixed): the F_e
    residual, the single- and joint-class factors, and for R147 the design split and the early - inline twin gap"""
    rr = [dict(r, measured=means[r["name"]]) for r in rows]
    out = {"F_e_residual": 1.5 * sum(r["weight"] * (r["measured"] - r["predicted"]) for r in rr
                                     if r["kind"] in ("p0", "main"))}
    base = CLASSES + (("mcm_readout",) if design_of is not None else ())     # as fits_for: R147's read mix too
    present = [k for k in base if any(abs(r["L"].get(k, 0.0)) > 0 for r in rr)]
    for one in ("sx", "idle", "slot", "cz", "dressing"):
        if any(abs(r["L"].get(one, 0.0)) > 0 for r in rr):
            out[f"only_{one}"] = A.fit(rr, [one])["beta"][one]
    f = A.fit(rr, present)
    out.update({f"classes.{k}": v for k, v in f["beta"].items()})
    if any(abs(r["L"].get("slot", 0.0)) > 0 for r in rr):
        f = A.fit(rr, [k for k in present if k != "sx"] + ["slot"])
        out.update({f"slot_for_sx.{k}": v for k, v in f["beta"].items()})
    if design_of is not None:
        for k, v in design_split(rr, design_of).items():
            if k in ("main|dynamic|s0", "main|isa|s1", "twin_inline|dynamic|s0", "twin_static|isa|s0"):
                out[f"split.{k}"] = v["mean_residual"]
        tw = twin_gaps(rr)
        if "main_minus_twin_inline" in tw:
            g = tw["main_minus_twin_inline"]
            out["twins.early_minus_inline.measured_minus_predicted"] = g["measured"] - g["predicted"]
    return out


ARM_PROTOCOL = {"r146": "r146", "r146_template": "r146", "r147_physical": "r147", "r147_template": "r147",
                "r147_stall_sensitivity": "r147"}


def assemble(out: dict, rows: dict, design_of: dict) -> dict:
    """every arm's summary, the joint fit and the cross-predictions, from the rows alone (PURE: --resummarize rebuilds
    them from a stored run without the engine); each arm keeps what its row builder measured (F_e_predicted, the
    per-kind residual)"""
    for name, rr in rows.items():
        keep = {k: v for k, v in out.get(name, {}).items() if k in ("F_e_predicted", "residual")}
        r147 = ARM_PROTOCOL[name] == "r147"
        out[name] = dict(summarize(rr, ("mcm_readout",) if r147 else (), design_of if r147 else None), **keep)
    if "r146" in rows and "r147_physical" in rows:
        out["joint_physical"] = dict(classes_slot=A.fit(rows["r146"] + rows["r147_physical"],
                                                        list(CLASSES) + ["mcm_readout", "slot"]))
    out["cross"] = {}
    for a6, a7 in (("r146", "r147_physical"), ("r146_template", "r147_template")):
        if a6 not in rows or a7 not in rows:
            continue
        f6, f7 = out[a6]["fits"], out[a7]["fits"]
        out["cross"][f"{a6}<->{a7}"] = {
            fn: dict(r146_fit_predicts_r147=A.fe_shift(rows[a7], f6[fn]["beta"]),
                     r147_fit_predicts_r146=A.fe_shift(rows[a6], {k: v for k, v in f7[fn]["beta"].items()
                                                                 if k != "mcm_readout"}),
                     r146_residual=out[a6]["template"]["F_e_residual"],
                     r147_residual=out[a7]["template"]["F_e_residual"])
            for fn in ("classes", "classes_slot", "slot_for_sx", "only_slot", "only_sx") if fn in f6 and fn in f7}
    return out


def resummarize(root: pathlib.Path, name: str = "reattribution_r152.json") -> dict:
    """rebuild a stored run's summaries from its rows (no engine), with R147's design labels from its record"""
    import r155_heldout_r152 as HO
    p = root / OUT / name
    out = json.loads(p.read_text(encoding="utf-8"))
    rec147 = json.loads((root / HO.R152_DIR / "r147.json").read_text(encoding="utf-8"))
    assemble(out, out["rows"], {u["name"]: u["prediction_inputs"]["kind"] for u in rec147["units"]})
    out["resummarized_by"] = provenance(root)
    p.write_text(evidence_text(out), encoding="utf-8", newline="\n")
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="only the first N units of each protocol (a smoke run)")
    ap.add_argument("--resummarize", action="store_true",
                    help="rebuild the stored run's summaries from its rows (pure; no engine) and rewrite it")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="error bars counting the twirl's frames: N resamples of the stored run's measured side, by "
                         "randomization and by shot (pure; no engine) -> bootstrap_reattribution_r152.json")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    if a.bootstrap:
        import r152_posthoc as P152
        import r155_heldout_r152 as HO
        from oneq.hw import randomization as RZ
        t0 = time.time()
        stored = json.loads((root / OUT / "reattribution_r152.json").read_text(encoding="utf-8"))
        art = P152.load(root / HO.R152_DIR)
        specs = unit_specs(art)
        design_of = {u["name"]: u["prediction_inputs"]["kind"] for u in art["parts"]["r147"]["units"]}
        names = sorted({r["name"] for rr in stored["rows"].values() for r in rr})
        # R152's span SLICES live in its spans file, not the flight index (r176_frame_bars.run_r152 does the same)
        index = dict(art["index"], execution_spans=art["spans"])
        shots = RZ.from_flight(index, art["arrays"], names=[specs[n][0] for n in names])
        cache: dict = {}

        def estimator(counts):
            means = means_from(counts, specs, names, cache)
            out = {}
            for arm, rr in stored["rows"].items():
                st = arm_statistics(rr, means, design_of if ARM_PROTOCOL[arm] == "r147" else None)
                out.update({f"{arm}.{k}": v for k, v in st.items()})
            return out
        res = frame_bootstrap(shots, estimator, n_boot=a.bootstrap)
        rec = dict(schema="oneq.r171.bootstrap/1", label="EXPLORATORY -- outside every preregistered verdict",
                   source=f"{OUT.as_posix()}/reattribution_r152.json", n_boot=a.bootstrap,
                   frames_per_circuit=sorted({s.R for s in shots.values()}), provenance=provenance(root),
                   statistics=res, seconds=time.time() - t0)
        (root / OUT / "bootstrap_reattribution_r152.json").write_text(evidence_text(rec), encoding="utf-8",
                                                                       newline="\n")
        for k, v in res.items():
            print(f"{k:70s} {v['estimate']:+.4f}  sd frames {v['sd_frames']:.4f}  shots {v['sd_shots']:.4f}  "
                  f"x{v['inflation']:.2f}")
        print(f"wrote bootstrap_reattribution_r152.json ({rec['seconds']:.0f} s)")
        return 0
    if a.resummarize:
        out = resummarize(root)
        for arm in ARM_PROTOCOL:
            if arm in out:
                print(arm, "residual F_e", round(out[arm]["template"]["F_e_residual"], 4),
                      json.dumps(out[arm].get("design_split", {}))[:600], json.dumps(out[arm].get("twins", {})))
        return 0
    import psutil
    if psutil.virtual_memory().available < 4e9:
        raise SystemExit("less than 4 GB available: not starting")
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    import r167_global_stall_heldout as G
    from oneq.hw import noise as N
    t0 = time.time()
    prov = provenance(root)
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    m6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    rec147 = P["record147"]
    dt = rec147["snapshot"]["dt"]
    held, _ = HO.with_hold(P["stand_ins"], rec147, HOLD_MEASURED_S)
    stall_dt = int(round(STALL_SENSITIVITY_US * 1e-6 / dt / 16)) * 16
    stalled, _ = G.stalled_stand_ins(P["stand_ins"], rec147, stall_dt, HO.with_hold)
    arms = (("r146", "r146", MODEL_KW, None, True), ("r146_template", "r146", TEMPLATE_KW, None, False),
            ("r147_physical", "r147", MODEL_KW, held, True),
            ("r147_template", "r147", TEMPLATE_KW, held, False),
            ("r147_stall_sensitivity", "r147", MODEL_KW, stalled, True))
    out: dict = dict(schema="oneq.r171.reattribution/2", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, limit=a.limit, provenance=prov,
                     measured=dict(r146=P["meas146"], r147=P["meas147"]), models={},
                     stand_ins=dict(r147_physical=f"hold {HOLD_MEASURED_S} s (R166), no global stall",
                                    r147_template=f"hold {HOLD_MEASURED_S} s (R166), no global stall",
                                    r147_stall_sensitivity=f"global sync + {STALL_SENSITIVITY_US} us stall (R167 b; "
                                                           "R166 falsified it as the server's behaviour)"),
                     slot_rule=dict(pulses_per_slot=PULSES_PER_SLOT, source="R166 executed schedule",
                                    rule="n_cz + 1 slots per qubit with real CZs (virtual zero-duration edges "
                                         "skipped); one per qubit without"))
    rows: dict = {}
    for name, proto, kw, stand, slots in arms:
        if proto == "r146":
            params = N.NoiseParams(**{**P["params146"].describe(), **kw})
            rows[name], out[name] = r146_rows(art, P, m6, params, a.traj, a.limit, slots)
        else:
            params = N.NoiseParams(cz_factor=1.0, schedule=P["params146"].schedule, **kw)
            rows[name], out[name] = r147_rows(art, P, m7, params, stand, a.traj, a.limit, slots)
        out["models"][name] = dict(base="V3c", slot_class=slots, **params.describe())
        print(f"{name}: {len(rows[name])} units ({time.time() - t0:.0f} s)", flush=True)
    assemble(out, rows, {u["name"]: u["prediction_inputs"]["kind"] for u in rec147["units"]})
    # the same rule on the same streams must reproduce the committed held-out predictions
    ev = root / "evidence" / "superactivation"
    r165 = json.loads((ev / "r165_dressing_bench" / "heldout_r152.json").read_text("utf-8"))
    r167 = json.loads((ev / "r167_standin" / "global_stall_heldout_r152.json").read_text("utf-8"))
    r166 = json.loads((ev / "r166_executed_schedule" / "template_heldout_r152.json").read_text("utf-8"))
    out["reproduces"] = dict(
        r146_committed_r165=r165["variants"]["+ dressing x1.54 (R165, 132 ns inverted)"]["r146"],
        r146_here=out["r146"].get("F_e_predicted"),
        r146_template_committed_r166=r166["variants"]["V3c + template + dressing x2 (two sx per layer, as scheduled)"]["r146"],
        r146_template_here=out["r146_template"].get("F_e_predicted"),
        r147_stall_committed_r167=r167["variants"][f"global stall L = {STALL_SENSITIVITY_US} us"]["r147"],
        r147_stall_here=out["r147_stall_sensitivity"].get("F_e_predicted"))
    out["rows"] = rows
    out["seconds"] = time.time() - t0
    d = root / OUT
    d.mkdir(parents=True, exist_ok=True)
    name = "reattribution_r152.json" if a.limit is None else f"reattribution_r152_smoke{a.limit}.json"
    (d / name).write_text(evidence_text(out), encoding="utf-8", newline="\n")
    for key, *_ in arms:
        s = out[key]
        for fname in ("classes", "classes_slot", "slot_for_sx", "only_slot"):
            if fname not in s["fits"]:
                continue
            f = s["fits"][fname]
            print(key, fname, {k: f"{f['beta'][k]:+.2f}+-{f['sd'][k]:.2f}" for k in f["keys"]},
                  f"chi2 {f['chi2_no_excess']:.0f} -> {f['chi2_fit']:.0f} (n {f['n']}, cond {f['condition']:.0f})")
        print(key, "template", json.dumps({k: v for k, v in s["template"].items() if k != "per_kind"}))
        print(key, "tracks slots", json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                               for k, v in s["tracking"]["slots"].items()}))
    print("reproduces", json.dumps(out["reproduces"]))
    print("wrote", name, f"({out['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
