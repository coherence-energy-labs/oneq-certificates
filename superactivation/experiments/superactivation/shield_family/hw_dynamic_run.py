"""A preregistered MODEL TEST of measure-and-correct P on IBM hardware (R148 candidate; design: R147).

What it tests, in one job, at m = 5 so every number sits next to R145's measured 0.615:
  MAIN     the two-use protocol with P_d's sector 0 by mid-circuit read + feed-forward on the EARLY schedule
           (hw_dynamic.lean_dynamic), sector 1 by its routed gates; Neyman-allocated kept strata, the 24 both-erased
           units enumerated. Every circuit predicted at freeze (sector 0 through hw_dynamic.engine_circuit's exact
           engine stand-in, mid-circuit read-out error mixed in exactly).
  TWINS    for TWINS sector-0 matched units of MAIN, the SAME unit as (a) the INLINE schedule (feed-forward after the
           key is prepared) and (b) the R145 static circuit (routed CZ star). Paired differences test the model's idle
           accounting (early - inline) and the design-vs-design gain (dynamic - static, each on its own layout), with
           day-to-day drift cancelled.
  MCMCAL   the shield qubits read twice in a row from |0...0> and |1...1>: mid-circuit assignment error and
           measurement back-action (a flip between the two reads), directly.
It cannot license a superactivation claim (V1 is reported only): the circuits are the LEAN branch compilation.

Modes (as hw_superact_run.py):
  freeze  --live B | --fake      read-only: snapshot, layouts, design, QPY + record with every prediction inside
  submit  RECORD                 OWNER ACT, behind every oneq.hw.submit guard
  fetch   RECORD                 read-only: counts -> analyse() -> result JSON
  dryrun  RECORD                 the frozen circuits through SamplerV2 LOCAL on FakeKingston (Aer) -> analyse()
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_dynamic as D  # noqa: E402
import hw_faithful as FA  # noqa: E402
import hw_levers as V  # noqa: E402
import hw_superact_run as R145  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

M = 5
DECODER = "cz2"
N_KEPT = 160
MIN_PER_STRATUM = 4
TWINS = 16
SHOTS = 400
CAL_SHOTS = 2000
RANDOMIZATIONS = 8
TRAJ = 60
ROUTE_ATTEMPTS = 20
MODEL_TOLERANCE_FE = 0.03
SEED = 20261016
PREREG = ROOT / "docs" / "PREREG_superactivation_dynamic_validation_m5.md"
PRIOR = ROOT / "evidence" / "superactivation" / "shield_family" / "dynamic" / "dynamic_m5_cz2_early_idle1.json"
OUTDIR = ROOT / "evidence" / "superactivation" / "hw_dynamic_m5"
TAG = "superact_dyn_m5"


# ------------------------------------------------------------------ design
def stratum_name(key) -> str:
    return "|".join(map(str, key))


def allocation(prior: dict) -> dict:
    """Neyman over the 12 kept strata: C_b ~ W_b x (predicted spread of per-unit LOSS), floor MIN_PER_STRATUM"""
    rows = {stratum_name(k): w for k, w in H.strata(M)}
    spread = {n: prior["strata"][n]["loss_se"] * math.sqrt(prior["units_per_stratum"]) for n in rows}
    score = {n: rows[n] * max(spread[n], 1e-3) for n in rows}
    tot = sum(score.values())
    return {n: max(MIN_PER_STRATUM, round(N_KEPT * s / tot)) for n, s in score.items()}


def design(prior: dict) -> dict:
    rng = np.random.default_rng(SEED)
    alloc = allocation(prior)
    main = []
    for key, W in H.strata(M):
        n = stratum_name(key)
        for _ in range(alloc[n]):
            main.append((n, W / alloc[n], H.sample_in_stratum(M, key, rng)))
    ee = [("erased_erased", w, u) for w, u in H.erased_erased_units(M)]
    pool = [i for i, (n, _, u) in enumerate(main) if u.choice[0] == "s0" and u.beta == 0
            and D.measure_type(u.choice)]
    twins = sorted(rng.choice(pool, size=min(TWINS, len(pool)), replace=False).tolist())
    return dict(main=main, ee=ee, twins=twins, allocation=alloc)


def unit_record(u):
    return R145.unit_record(u)


def unit_from(rec):
    return R145.unit_from(rec)


def mcm_cal_circuit(m, pattern: int):
    """the m shield qubits: prepare |pattern>^m, read mid-circuit, read again. Bits 0..m-1 first read, m..2m-1 second"""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    q = QuantumRegister(m, "q")
    c = ClassicalRegister(2 * m, "c")
    qc = QuantumCircuit(q, c, name=f"mcmcal_{pattern}")
    if pattern:
        qc.x(range(m))
    qc.measure(range(m), range(m))
    qc.measure(range(m), range(m, 2 * m))
    return qc


# ------------------------------------------------------------------ freeze
def _policy_ok(snap, tqc):
    return not L.refusals(snap, tqc, L.QubitPolicy())


def _route(be, snap, qc, layout, seed0):
    from qiskit import transpile
    for attempt in range(ROUTE_ATTEMPTS):
        tqc = transpile(qc, be, initial_layout=layout, optimization_level=3, seed_transpiler=seed0 + 1000 * attempt)
        if _policy_ok(snap, tqc):
            return tqc
    raise SystemExit(f"{qc.name}: cannot be routed inside the policy: {L.refusals(snap, tqc, L.QubitPolicy())}")


_UNIT_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def _delays_in_dt(qc, dt: float):
    """the circuit with every delay in whole dt (rounded). qiskit 2.5's QPY cannot load a delay stored in seconds: it
    rebuilds it in 'dt' and refuses the float (found 2026-10-16, R152; 2.5.0 and 2.5.2 both), so a frozen stand-in
    with one could never be re-predicted. The engine schedules in dt anyway."""
    from qiskit.circuit import Delay
    out = qc.copy_empty_like()
    for inst in qc.data:
        op = inst.operation
        if op.name == "delay" and op.unit != "dt":
            op = Delay(int(round(float(op.duration) * _UNIT_S[op.unit] / dt)), unit="dt")
        out.append(op, inst.qubits, inst.clbits)
    return out


def _dynamic_engine_inputs(u, layout0, be, tau, early):
    """the static engine stand-ins of a dynamic unit (c0: every mid-circuit read correct; c1: their parity flipped),
    the virtual edges they need, and the shields whose read-out error mixes the two -- everything its prediction reads
    except the device model, so it can be FROZEN and the prediction recomputed on any model (R152). Delays in dt, so
    the frozen QPY loads (_delays_in_dt)."""
    (c0, virt), (c1, _) = D.engine_circuit(M, u, DECODER, layout0, be, tau, early=early)
    mt = D.measure_type(u.choice)
    return (_delays_in_dt(c0, be.dt), _delays_in_dt(c1, be.dt), [list(map(int, e)) for e in virt],
            [layout0[4 + t] for t in mt])


def _predict_from(c0, c1, virt, mix_qubits, score, model, rng, params, traj=TRAJ):
    sn = D.augmented(model, [tuple(e) for e in virt])
    r = 0.5 * (1 - math.prod(1 - 2 * model.readout_error(q) for q in mix_qubits))
    e0 = E.expectation(c0, score, sn, params, traj, rng)
    e1 = E.expectation(c1, score, sn, params, traj, rng) if mix_qubits else e0
    return (1 - r) * e0.mean + r * e1.mean, math.hypot((1 - r) * e0.se, r * e1.se)


def unit_rng(k: int):
    """the k-th unit's own random stream: a prediction never depends on which units were predicted before it"""
    return np.random.default_rng([7, k])


def _predict_dynamic(u, layout0, be, model, tau, early, rng, params):
    c0, c1, virt, mix = _dynamic_engine_inputs(u, layout0, be, tau, early)
    return _predict_from(c0, c1, virt, mix, FA.score(M, u, DECODER), model, rng, params)


def model_coverage(model, named) -> list:
    """every CZ edge a frozen circuit uses that the prediction model has NO measured channel for (cz_structure)"""
    missing = set()
    for _, tqc in named:
        for inst in tqc.data:
            if inst.operation.name == "cz":
                a, b = sorted(tqc.find_bit(x).index for x in inst.qubits)
                if model.edge_structure(a, b) is None:
                    missing.add((a, b))
    return sorted(missing)


def freeze_run(be, snap, prereg_sha: str, model=None, cz_factor: float = V.CZ_FACTOR_V4, require_repaired=False,
               schedule: str = "alap", model_path: str | None = None, idle: bool = True,
               return_prediction_blob: bool = False):
    """snap: the live calibration (layout, routing, the HEALTHY policy). model: the snapshot the PREDICTIONS use --
    for a real freeze, a repaired in-circuit model (R146's) whose cz_structure covers every edge the circuits use;
    require_repaired refuses to freeze otherwise. Default (rehearsal): the vendor snapshot at R145's CZ factor, the
    parameters R146 showed reproduce R145's miss -- a rehearsal of the machinery, not a prediction to test.

    Every unit also records its PREDICTION INPUTS (which circuit the engine reads, the virtual edges, the read-out mix,
    the score function); the dynamic units' engine stand-ins are frozen in a second QPY, never submitted, whose hash
    the record pins. predict_units() then recomputes every prediction on ANY model, exactly as this freeze computed
    it -- the same-job model of R152, fitted from a calibration block in the same job. return_prediction_blob=True
    returns that QPY as a third element."""
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    model = model or snap
    params = N.NoiseParams(cz_factor=cz_factor, schedule=schedule, idle=idle)
    layout0, _ = D.hand_layout(snap, be, M)
    layout1 = L.choose_layout(V.representatives(M, DECODER), be, snap, params).initial_layout
    tau = float(np.median([x for x in snap.meas_duration if math.isfinite(x)])) + D.TAU_FF
    des = design(prior)
    named, units, pred_named = [], [], []

    def add(name, stratum, weight, u, tqc, kind, pred, pred_se, hi, inputs):
        named.append((name, tqc))
        units.append(dict(name=name, stratum=stratum, weight=weight, unit=unit_record(u), kind=kind,
                          ideal=FA.lean_ideal(M, u, DECODER) if u.branch != "erased_erased" else H.ideal_expectation(M, u),
                          predicted=pred, predicted_se=pred_se, cz=tqc.count_ops().get("cz", 0),
                          n_bits=tqc.num_clbits, hi=hi, prediction_inputs=inputs))

    def dynamic(name, u, early):
        c0, c1, virt, mix = _dynamic_engine_inputs(u, layout0, be, tau, early)
        pred_named.append((f"pred0__{name}", c0))
        if mix:
            pred_named.append((f"pred1__{name}", c1))
        p = _predict_from(c0, c1, virt, mix, FA.score(M, u, DECODER), model, unit_rng(len(units)), params)
        return p, dict(kind="dynamic", c0=f"pred0__{name}", c1=f"pred1__{name}" if mix else None, virt=virt,
                       mix_qubits=mix, score="FA")
    for i, (st, w, u) in enumerate(des["main"]):
        name = f"m{i:03d}__{st}"
        if u.choice[0] == "s0":
            tqc = _route(be, snap, D.lean_dynamic(M, u, DECODER, early=True, one_register=True), layout0, 9000 + i)
            (pred, se), inputs = dynamic(name, u, True)
        else:
            tqc = _route(be, snap, FA.lean_circuit(M, u, DECODER), layout1, 9000 + i)
            r = E.expectation(tqc, FA.score(M, u, DECODER), model, params, TRAJ, unit_rng(len(units)))
            (pred, se), inputs = (r.mean, r.se), dict(kind="isa", score="FA")
        add(name, st, w, u, tqc, "main", pred, se, 2.0, inputs)
    for j, (st, w, u) in enumerate(des["ee"]):
        tqc = _route(be, snap, H.circuit(M, u), layout0, 7000 + j)
        r = E.expectation(tqc, H.score(M, u), model, params, TRAJ, unit_rng(len(units)))
        add(f"e{j:03d}__erased_erased", st, w, u, tqc, "main", r.mean, r.se, 1.0, dict(kind="isa", score="H"))
    for k, i in enumerate(des["twins"]):
        st, _, u = des["main"][i]
        name = f"t{k:02d}_inline__m{i:03d}"
        tqc = _route(be, snap, D.lean_dynamic(M, u, DECODER, early=False, one_register=True), layout0, 5000 + k)
        (pred, se), inputs = dynamic(name, u, False)
        add(name, st, 0.0, u, tqc, "twin_inline", pred, se, 2.0, inputs)
        # each design on its OWN best layout: V4 compares the designs as they would be run, not routing alone (the
        # dynamic layout scatters its pairs across the chip; the static star routed from there needs 170-270 CZs)
        tqc = _route(be, snap, H.circuit(M, u), layout1, 6000 + k)
        r = E.expectation(tqc, H.score(M, u), model, params, TRAJ, unit_rng(len(units)))
        add(f"t{k:02d}_static__m{i:03d}", st, 0.0, u, tqc, "twin_static", r.mean, r.se, 2.0,
            dict(kind="isa", score="H"))
    cal = []
    for pat in (0, 1):
        tqc = _route(be, snap, mcm_cal_circuit(M, pat), [layout0[4 + t] for t in range(M)], 3000 + pat)
        named.append((f"mcmcal_{pat}", tqc))
        cal.append(dict(name=f"mcmcal_{pat}", pattern=pat, qubits=[layout0[4 + t] for t in range(M)],
                        readout_calibrated=[model.readout_error(layout0[4 + t]) for t in range(M)]))
    missing = model_coverage(model, named)
    if require_repaired and missing:
        raise SystemExit(f"REFUSED: the prediction model has no measured CZ channel for {len(missing)} edge(s) the "
                         f"frozen circuits use: {missing[:8]} -- calibrate them in-circuit first (R146's method)")
    main_units = [u for u in units if u["kind"] == "main"]
    pred = R145.estimate({u["name"]: u["predicted"] for u in main_units}, main_units)
    meta = dict(protocol="two-use superactivation, measure-and-correct P (hw_dynamic.py), lean branch compilation",
                m=M, d=2 ** M, decoder=DECODER, schedule="early", backend=snap.name, snapshot=snap.to_json(),
                snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc, layout_sector0=layout0,
                layout_sector1=layout1, tau_s=tau, tau_ff_assumed_s=D.TAU_FF, cz_factor=cz_factor, schedule_model=schedule,
                model_snapshot=None if model is snap else model.to_json(), model_snapshot_sha256=model.sha256(),
                model_snapshot_path=model_path, idle_modelled=idle,
                model_unmeasured_edges=[list(e) for e in missing], repaired_model=bool(model is not snap),
                units=units, allocation=des["allocation"], shots=SHOTS, cal_shots=CAL_SHOTS, mcm_cal=cal,
                randomizations=RANDOMIZATIONS, prediction=pred, model_tolerance_F_e=MODEL_TOLERANCE_FE,
                prereg=PREREG.name, prereg_sha256=prereg_sha,
                sources=[str(p) for p in [HERE / "hw_dynamic_run.py", HERE / "hw_dynamic.py", HERE / "hw_faithful.py",
                                          HERE / "hw_two_use.py", HERE / "hw_levers.py", HERE / "hw_superact_run.py"]
                         + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))])
    pred_rec, pred_blob = F.freeze(pred_named, dict(kind="R147 engine stand-ins of the dynamic units: read by "
                                                         "predict_units, never submitted"))
    meta["prediction_circuits"] = dict(qpy_sha256=pred_rec["qpy_sha256"], circuit_order=pred_rec["circuit_order"],
                                       circuit_sha256=pred_rec["circuit_sha256"])
    rec, blob = F.freeze(named, meta)
    return (rec, blob, pred_blob) if return_prediction_blob else (rec, blob)


def load_prediction_circuits(rec: dict, pred_blob: bytes) -> dict:
    """the frozen engine stand-ins, refused unless they are byte-for-byte the ones this record pins"""
    pc = rec["prediction_circuits"]
    return F.load(dict(pc, sources=[]), pred_blob, check_sources=False)


def predict_units(rec: dict, circuits: dict, pred_circuits: dict, model, params, traj: int = TRAJ,
                  names: set | None = None) -> dict:
    """every unit's prediction on `model` (a DeviceSnapshot) under `params`, recomputed from the frozen inputs in the
    freeze's own per-unit random streams -- so on the freeze's model it reproduces the frozen predictions exactly,
    and on a same-job model (R152) it is the preregistered prediction rule applied to that model. names: only these
    units (each unit's stream is its own, so a subset reproduces exactly too)"""
    out = {}
    for k, u in enumerate(rec["units"]):
        if names is not None and u["name"] not in names:
            continue
        rng = unit_rng(k)
        uu = unit_from(u["unit"])
        inp = u["prediction_inputs"]
        score = FA.score(M, uu, DECODER) if inp["score"] == "FA" else H.score(M, uu)
        if inp["kind"] == "dynamic":
            c1 = pred_circuits[inp["c1"]] if inp["c1"] else None
            out[u["name"]] = _predict_from(pred_circuits[inp["c0"]], c1, inp["virt"], inp["mix_qubits"], score,
                                           model, rng, params, traj)
        else:
            r = E.expectation(circuits[u["name"]], score, model, params, traj, rng)
            out[u["name"]] = (r.mean, r.se)
    return out


def with_predictions(rec: dict, preds: dict, model) -> dict:
    """the record with every unit's prediction (and the headline prediction, and the mid-circuit calibration's
    reference read-out) taken from `preds` and `model` -- the input analyse() reads, nothing else changed"""
    units = [dict(u, predicted=preds[u["name"]][0], predicted_se=preds[u["name"]][1]) for u in rec["units"]]
    main = [u for u in units if u["kind"] == "main"]
    cal = [dict(c, readout_calibrated=[model.readout_error(q) for q in c["qubits"]]) for c in rec["mcm_cal"]]
    return dict(rec, units=units, mcm_cal=cal,
                prediction=R145.estimate({u["name"]: u["predicted"] for u in main}, main))


# ------------------------------------------------------------------ analysis (pure)
def _score(u):
    uu = unit_from(u["unit"])
    return H.score(M, uu) if uu.branch == "erased_erased" else FA.score(M, uu, DECODER)


def control_variate(means: dict, units: list, var: dict | None = None, shots: dict | None = None) -> dict:
    """F_avg = sum_b W_b (exact_b - mean loss_b) + both-erased; the R147 estimator (exact stratum ideals)"""
    by: dict = {}
    for u in units:
        if u["kind"] == "main" and u["stratum"] != "erased_erased":
            by.setdefault(u["stratum"], []).append(u)
    f_avg = 0.0
    var_ = 0.0
    for key, W in H.strata(M):
        us = by[stratum_name(key)]
        losses = [u["ideal"] - means[u["name"]] for u in us]
        f_avg += W * (V.stratum_exact(M, list(key)) - float(np.mean(losses)))
        var_ += W ** 2 * (float(np.var(losses, ddof=1)) / len(us) if len(us) > 1 else 0.0)
    ee = [u for u in units if u["stratum"] == "erased_erased"]
    f_avg += sum(u["weight"] * means[u["name"]] for u in ee)
    if var is not None:                                  # the enumerated branch carries shot noise only
        var_total = var_ + sum(u["weight"] ** 2 * var[u["name"]] / shots[u["name"]] for u in ee)
    else:
        var_total = var_
    return dict(F_e=(3 * f_avg - 1) / 2, sd_F_e=1.5 * math.sqrt(var_total))


def paired(means: dict, units: list, a: str, b: str) -> dict:
    """mean over twin units of (kind a - kind b), measured and predicted, with the measured sd"""
    by = {}
    for u in units:
        if u["kind"] in ("twin_inline", "twin_static"):
            by.setdefault(u["name"].split("__")[1], {})[u["kind"]] = u
    for u in units:
        if u["kind"] == "main":
            key = u["name"].split("__")[0]
            if key in by:
                by[key]["main"] = u
    rows = [g for g in by.values() if a in g and b in g]
    dm = [means[g[a]["name"]] - means[g[b]["name"]] for g in rows]
    dp = [g[a]["predicted"] - g[b]["predicted"] for g in rows]
    return dict(n=len(rows), measured=float(np.mean(dm)), measured_sd=float(np.std(dm, ddof=1) / math.sqrt(len(dm))),
                predicted=float(np.mean(dp)))


def mcm_cal(record: dict, counts: dict) -> dict:
    out = []
    for cal in record["mcm_cal"]:
        cnt = counts[cal["name"]]
        tot = sum(int(c) for c in cnt.values())
        for t, q in enumerate(cal["qubits"]):
            wrong1 = sum(int(c) for k, c in cnt.items() if T.bits_of(k, 2 * M)[t] != cal["pattern"])
            flip = sum(int(c) for k, c in cnt.items() if T.bits_of(k, 2 * M)[t] != T.bits_of(k, 2 * M)[M + t])
            a_ = wrong1 / tot
            # two reads disagree by assignment noise alone with probability ~2a(1-a): back-action is the EXCESS
            out.append(dict(qubit=q, pattern=cal["pattern"], assignment_error=a_, disagreement=flip / tot,
                            backaction_flip=max(0.0, flip / tot - 2 * a_ * (1 - a_)),
                            calibrated_readout=cal["readout_calibrated"][t]))
    per_q: dict = {}
    for r in out:
        per_q.setdefault(r["qubit"], []).append(r)
    summary = {q: dict(assignment=float(np.mean([r["assignment_error"] for r in rs])),
                       backaction=float(np.mean([r["backaction_flip"] for r in rs])),
                       calibrated=rs[0]["calibrated_readout"]) for q, rs in per_q.items()}
    ok = all(s["assignment"] <= 2 * s["calibrated"] + 1e-3 and s["backaction"] <= 0.01 for s in summary.values())
    return dict(per_qubit=summary, rows=out, V5_met=bool(ok))


def analyse(record: dict, counts: dict) -> dict:
    units = record["units"]
    stat = [T.Unit(u["name"], u["weight"], record["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    tallies = T.tally_all(stat, {u["name"]: counts[u["name"]] for u in units}, {u["name"]: _score(u) for u in units})
    means = {n: t.mean for n, t in tallies.items()}
    var = {n: max(t.s2 / t.n - t.mean ** 2, 0.0) for n, t in tallies.items()}
    shots = {n: t.n for n, t in tallies.items()}
    main = [u for u in units if u["kind"] == "main"]
    strat = R145.estimate(means, main, var, shots)
    cv = control_variate(means, main, var, shots)
    pred = R145.estimate({u["name"]: u["predicted"] for u in main}, main)
    pred_cv = control_variate({u["name"]: u["predicted"] for u in main}, main)
    early_inline = paired(means, units, "main", "twin_inline")
    dyn_static = paired(means, units, "main", "twin_static")
    cal = mcm_cal(record, counts)
    delta = cv["F_e"] - pred_cv["F_e"]
    return dict(
        measured=dict(F_e=cv["F_e"], sd_F_e=cv["sd_F_e"], F_e_stratified=strat["F_e"], sd_stratified=strat["sd_F_e"]),
        predicted=dict(F_e=pred_cv["F_e"], F_e_stratified=pred["F_e"]),
        verdicts=dict(
            V1_superactivation_reported_only=bool(cv["F_e"] - 2.326 * cv["sd_F_e"] > 0.75),
            V2_model_within_tolerance=bool(abs(delta) <= record["model_tolerance_F_e"]), delta_F_e=delta,
            V3_idle_accounting=bool(abs(early_inline["measured"] - early_inline["predicted"])
                                    <= 2 * early_inline["measured_sd"]),
            V4_dynamic_beats_static=bool(dyn_static["measured"] - 2 * dyn_static["measured_sd"] > 0),
            V5_mcm_readout=cal["V5_met"]),
        early_minus_inline=early_inline, dynamic_minus_static=dyn_static, mcm_calibration=cal,
        R145_measured_F_e=0.6151,
        per_unit={u["name"]: dict(measured=means[u["name"]], predicted=u["predicted"], ideal=u["ideal"], cz=u["cz"],
                                  kind=u["kind"]) for u in units})


# ------------------------------------------------------------------ modes
def job_for(rec: dict, outdir: pathlib.Path) -> dict:
    """the ONE job record submitted from THIS frozen record (matched by the QPY hash, never by recency): circuit names
    repeat across freezes, so the newest job file may belong to another record"""
    jobs = [j for j in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(outdir.glob(f"{TAG}_*.job.json")))
            if j.get("qpy_sha256") == rec["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records were submitted from THIS frozen record (qpy "
                         f"{rec['qpy_sha256'][:12]}); expected exactly one")
    return jobs[0]


def tracked_in_repo(path: pathlib.Path) -> bool:
    """the file is committed in this repository (git ls-files knows it, and the working copy matches HEAD)"""
    import subprocess
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    ls = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(rel)], capture_output=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", str(rel)], capture_output=True)
    return ls.returncode == 0 and dirty.returncode == 0


REPAIRED_CZ_FACTOR = 1.0        # R146's repaired model carries its CZ rates in the snapshot and runs at cz_factor 1


def submit_refusals(rec: dict, blob: bytes) -> list:
    """this run's own preconditions, RECOMPUTED from the record (no field is trusted): the predictions' model is a
    stored snapshot that hashes to the recorded value and carries a measured CZ channel for every edge the frozen
    circuits use, and the schedule the model assumed was stated. Pure: no network."""
    out = []
    ms = rec.get("model_snapshot")
    if not ms:
        return ["the record stores no repaired model snapshot (a rehearsal, or frozen before that rule): re-freeze "
                "with --model-snapshot"]
    model = S.DeviceSnapshot.from_json(ms)
    if model.sha256() != rec.get("model_snapshot_sha256"):
        out.append("the stored model snapshot does not hash to model_snapshot_sha256")
    # PROVENANCE: the model must be a COMMITTED file (R146's repaired snapshot), byte-identical in content to what the
    # record stores -- a record cannot carry a model made up for it
    src = rec.get("model_snapshot_path")
    if not src:
        out.append("the record does not name the committed file its model came from")
    else:
        fp = ROOT / src
        if not fp.is_file() or not tracked_in_repo(fp):
            out.append(f"the model file {src} is not a committed, unmodified file of this repository")
        elif S.DeviceSnapshot.from_json(json.loads(fp.read_text(encoding="utf-8"))).sha256() != model.sha256():
            out.append(f"the committed model file {src} is not the model this record stores")
    if rec.get("cz_factor") != REPAIRED_CZ_FACTOR:
        out.append(f"cz_factor {rec.get('cz_factor')} is not the repaired model's {REPAIRED_CZ_FACTOR}")
    try:
        circuits = F.load(rec, blob, check_sources=False)   # source drift is oneq.hw.submit's FROZEN guard
    except F.FrozenMismatch as e:
        return out + [f"frozen circuits: {e}"]
    missing = model_coverage(model, list(circuits.items()))
    if missing:
        out.append(f"the model has no measured CZ channel for {len(missing)} edge(s) used: {missing[:6]}")
    if rec.get("schedule_model") not in N.SCHEDULES:
        out.append("the record does not state the schedule its predictions assumed")
    if rec.get("idle_modelled") is not True:
        out.append("the record's predictions did not model idle errors (a rehearsal setting)")
    return out



def _plan(rec):
    shots = {u["name"]: rec["shots"] for u in rec["units"]} | {c["name"]: rec["cal_shots"] for c in rec["mcm_cal"]}
    return U.Plan(tag=TAG, backend=rec["backend"], shots=shots, prereg_path=str(PREREG),
                  prereg_sha256=rec["prereg_sha256"], twirl_randomizations=rec["randomizations"],
                  records_dir=str(OUTDIR), predicted_pass_probability=None,
                  measurement_reason=("a preregistered MODEL TEST of measure-and-correct P (mid-circuit read-out, "
                                      "feed-forward, design-vs-design gain); no superactivation claim is possible from it"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["freeze", "submit", "fetch", "dryrun"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true", help="REHEARSAL: freeze on FakeKingston's OWN calibration at CZ "
                    "factor 1 with idle errors OFF -- the device Aer simulates in `dryrun` (qiskit cannot pad idles "
                    "around if_else, so Aer has none) -- making the dry run an end-to-end test of every prediction "
                    "except the idle accounting, which the device run's early/inline twins test")
    ap.add_argument("--model-snapshot", help="REQUIRED for a live freeze: the repaired in-circuit DeviceSnapshot JSON "
                    "(R146's repaired_snapshot) the predictions use; it must carry a CZ channel for every edge used")
    ap.add_argument("--model-cz-factor", type=float, default=1.0, help="cz_factor for the repaired model (R146: 1.0)")
    ap.add_argument("--schedule", choices=list(N.SCHEDULES), help="REQUIRED for a live freeze: the schedule R146's "
                    "timing circuits measured on the device (the early schedule's benefit assumes ALAP)")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.mode == "freeze":
        if not a.fake and (not a.model_snapshot or not a.schedule):
            raise SystemExit("REFUSED: a live freeze needs --model-snapshot (a repaired in-circuit model; the vendor "
                             "calibration at R145's CZ factor reproduces R145's miss, R146) and --schedule (measured).")
        if a.fake:
            be = V.backend()
            snap = S.from_backend(be)
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            be = QiskitRuntimeService().backend(a.live)
            snap = S.from_backend(be)
        if not PREREG.is_file():
            raise SystemExit(f"write {PREREG.name} first: the frozen record pins its hash")
        if a.fake:
            rec, blob = freeze_run(be, snap, U.file_sha256(PREREG), cz_factor=1.0, idle=False)
        else:
            if not a.model_snapshot:
                raise SystemExit("REFUSED: a live freeze needs --model-snapshot (a repaired in-circuit model). The "
                                 "vendor calibration at R145's CZ factor reproduces R145's miss (R146).")
            mp = pathlib.Path(a.model_snapshot).resolve()
            if not tracked_in_repo(mp):
                raise SystemExit(f"REFUSED: {a.model_snapshot} must be a committed, unmodified file of this repository "
                                 "(the repaired model's provenance)")
            model = S.DeviceSnapshot.from_json(json.loads(mp.read_text(encoding="utf-8")))
            rec, blob = freeze_run(be, snap, U.file_sha256(PREREG), model=model, cz_factor=a.model_cz_factor,
                                   require_repaired=True, schedule=a.schedule,
                                   model_path=str(mp.relative_to(ROOT.resolve())))
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"{'rehearsal_' if a.fake else ''}frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        plan = _plan(rec)
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; predicted F_e {rec['prediction']['F_e']:.4f}; "
              f"{plan.total_shots} shots, cost estimate {plan.cost_s():.1f} s")
        return 0
    rp = pathlib.Path(a.record)
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    plan = _plan(rec)
    if a.mode == "dryrun":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        # oneq.hw.submit.local_run's default: the target noise model, method pinned to statevector (R146: Aer's
        # automatic method mis-simulated mixed-size noisy batches)
        counts = U.local_run(rec, blob, FakeKingston(), plan.shots)     # no idle on either side (see --fake)
        out = analyse(rec, counts)
        (OUTDIR / f"dryrun_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n",
                                                     encoding="utf-8", newline="\n")
        print(json.dumps(out["verdicts"], indent=1))
        print("measured", out["measured"], "\npredicted", out["predicted"])
        return 0
    if a.mode == "submit":
        why = submit_refusals(rec, blob)
        if why:
            raise SystemExit("REFUSED: " + "; ".join(why))
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    if a.mode == "submit":
        be = svc.backend(rec["backend"])
        live = S.from_backend(be)
        circuits = F.load(rec, blob)
        health = sorted({r for qc in circuits.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(plan, rec, blob, be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        return 0
    job = job_for(rec, OUTDIR)
    got = U.fetch(job, svc)
    if "counts" not in got:
        print(f"job {job['job_id']}: {got['status']}")
        return 4
    out = dict(job=job, execution_seconds=got["execution_seconds"], analysis=analyse(rec, got["counts"]),
               counts=got["counts"])
    (OUTDIR / f"result_{job['job_id']}.json").write_text(json.dumps(out, indent=1, default=str) + "\n",
                                                         encoding="utf-8", newline="\n")
    print(json.dumps(out["analysis"]["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
