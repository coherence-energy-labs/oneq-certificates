"""The first execution of a Smith-Yard-type superactivation protocol on a quantum computer -- preregistered, predicted
to the circuit before submission, and analysed to the frame bit. A MEASUREMENT: oneq.hw predicts it falls short of
3/4 on today's IBM; the run measures by how much, tests that prediction at 14 qubits and ~40 CZ, and re-measures the
CZ factor on the layout's own most-used edge in the same job.

Design (docs/PREREG_superactivation_hw_run_m5_2026-10-15.md):
  - the two-use protocol of hw_two_use.py at m = 5 (d = 32), the 12 kept strata sampled with NEYMAN allocation
    (circuits proportional to weight x the predicted spread of unit means, at least MIN_PER_STRATUM each), the 24
    both-erased units ENUMERATED with exact weights;
  - a twirled CZ depth ladder (0, 2, 4, 8 CZ x XX/YY/ZZ, hw_flagged_channel's circuits) on the most-used edge;
  - every protocol circuit's expected score predicted by oneq.hw.engine at freeze, on the frozen ISA circuits.
Modes:
  freeze  --live B        read-only: snapshot, layout, design, freeze QPY + record (predictions inside)
  submit  RECORD          OWNER ACT: oneq.hw.submit behind every guard (flag + ONEQ_IBM_SUBMIT=1 + prereg hash + ...)
  fetch   RECORD          read-only: counts -> analyse() -> result JSON
  dryrun  RECORD          the frozen circuits through SamplerV2 LOCAL on the fake twin -> analyse() (tw-venv)
analyse() is pure arithmetic over the record and the counts, tested on synthetic counts before any submission.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_predict as HP  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

M = 5
N_KEPT = 176
MIN_PER_STRATUM = 4
SHOTS = 400
LADDER_SHOTS = 1000
RANDOMIZATIONS = 8
TRAJ = 100
ROUTE_ATTEMPTS = 20
MODEL_TOLERANCE_FE = 0.03          # preregistered: |measured - predicted| F_e for "the model holds at scale"
PREREG = ROOT / "docs" / "PREREG_superactivation_hw_run_m5_2026-10-15.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "hw_run_m5"
PRIOR = ROOT / "evidence" / "superactivation" / "shield_family" / "hw_predict_m5_ibm_kingston_cz2.29.json"
LADDER_DEPTHS = (0, 2, 4, 8)
BASES = ("XX", "YY", "ZZ")


# ------------------------------------------------------------------ design
def stratum_name(key: tuple) -> str:
    return "|".join(map(str, key))


def allocation(prior: dict) -> dict:
    """Neyman: C_b proportional to W_b x spread_b (the predicted std of unit losses in stratum b), floor
    MIN_PER_STRATUM, N_KEPT circuits over the 12 kept strata"""
    rows = {stratum_name(k): w for k, w in H.strata(M)}
    spread = {n: prior["strata"][n]["se_loss"] * math.sqrt(prior["strata"][n]["units"]) for n in rows}
    score = {n: rows[n] * max(spread[n], 1e-3) for n in rows}
    tot = sum(score.values())
    return {n: max(MIN_PER_STRATUM, round(N_KEPT * s / tot)) for n, s in score.items()}


def design_units(prior: dict, seed: int = 20261015) -> list:
    """[(stratum, weight_within_estimand, Unit)]: kept units sampled per the allocation (each weight W_b / C_b),
    both-erased units enumerated with their exact weights"""
    rng = np.random.default_rng(seed)
    alloc = allocation(prior)
    out = []
    for key, W in H.strata(M):
        n = stratum_name(key)
        for _ in range(alloc[n]):
            out.append((n, W / alloc[n], H.sample_in_stratum(M, key, rng)))
    for w, u in H.erased_erased_units(M):
        out.append(("erased_erased", w, u))
    return out


def ladder_circuits():
    spec = importlib.util.spec_from_file_location("hw_flagged_channel", HERE.parent / "hw_flagged_channel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    named = dict(mod.circuits())
    return [(f"ladder_d{d}_{b}", named[f"pauli_I_{b}" if d == 0 else f"depth_{d}_{b}"]) for d in LADDER_DEPTHS
            for b in BASES]


def unit_record(u: H.Unit) -> dict:
    return dict(branch=u.branch, choice=None if u.choice is None else [u.choice[0], list(u.choice[1])],
                beta=u.beta, inp=list(u.inp), flips=list(u.flips))


def unit_from(rec: dict) -> H.Unit:
    ch = None if rec["choice"] is None else (rec["choice"][0], tuple(rec["choice"][1]))
    return H.Unit(rec["branch"], ch, rec["beta"], tuple(rec["inp"]), tuple(rec["flips"]))


def freeze_run(be, snap: S.DeviceSnapshot, prereg_sha: str) -> tuple:
    from qiskit import transpile
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    choice = HP.choose_layout(be, M, snap, N.NoiseParams())
    units = design_units(prior)
    named, meta_units = [], []
    rng = np.random.default_rng(7)
    for i, (stratum, w, u) in enumerate(units):
        # EVERY frozen circuit must pass the policy the HEALTHY guard applies at submission, not just the layout's
        # representatives: routing can carry a measured logical onto a bad qubit. Re-route with a new seed, a
        # bounded number of times; refuse to freeze rather than ship a circuit the guard will refuse.
        for attempt in range(ROUTE_ATTEMPTS):
            tqc = HP.isa(be, H.circuit(M, u), choice.initial_layout, seed=9000 + i + 1000 * attempt,
                         coupling_map=choice.coupling_map)
            if not L.refusals(snap, tqc, L.QubitPolicy()):
                break
        else:
            raise SystemExit(f"unit {i} ({stratum}) cannot be routed inside the policy in {ROUTE_ATTEMPTS} tries: "
                             f"{L.refusals(snap, tqc, L.QubitPolicy())}")
        r = E.expectation(tqc, H.score(M, u), snap, N.NoiseParams(), TRAJ, rng)
        name = f"u{i:03d}__{stratum}"
        named.append((name, tqc))
        meta_units.append(dict(name=name, stratum=stratum, weight=w, unit=unit_record(u),
                               ideal=H.ideal_expectation(M, u), predicted=r.mean, predicted_se=r.se,
                               exact=r.exact, cz=tqc.count_ops().get("cz", 0), n_bits=tqc.num_clbits,
                               hi=1.0 if u.branch == "erased_erased" else 2.0))
    edge_use: dict = {}
    for _, tqc in named:
        for inst in tqc.data:
            if inst.operation.name == "cz":
                a, b = sorted(tqc.find_bit(x).index for x in inst.qubits)
                edge_use[(a, b)] = edge_use.get((a, b), 0) + 1
    edge = max(edge_use, key=lambda e: (edge_use[e], -e[0]))
    ladder = []
    for name, qc in ladder_circuits():
        tqc = transpile(qc, be, initial_layout=list(edge), optimization_level=1, seed_transpiler=1)
        if L.refusals(snap, tqc, L.QubitPolicy()):
            raise SystemExit(f"ladder edge {edge} is outside the policy: {L.refusals(snap, tqc, L.QubitPolicy())}")
        par = E.expectation(tqc, lambda bits: 1.0 if bits[0] == bits[1] else -1.0, snap, N.NoiseParams(), 0,
                            np.random.default_rng(0))
        ladder.append(dict(name=name, depth=int(name.split("_")[1][1:]), basis=name.split("_")[2],
                           predicted_correlator=par.mean))
        named.append((name, tqc))
    pred = estimate({u["name"]: u["predicted"] for u in meta_units}, meta_units)
    meta = dict(protocol="two-use superactivation, hw_two_use.py", m=M, d=2 ** M, backend=snap.name,
                snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
                layout=choice.initial_layout, layout_proxy=choice.best.proxy, noise=N.NoiseParams().describe(),
                units=meta_units, allocation=allocation(prior), shots=SHOTS, ladder_shots=LADDER_SHOTS,
                ladder=ladder, ladder_edge=list(edge), ladder_edge_vendor_cz=snap.edge(*edge)[0],
                edge_use=sorted([[a, b, n] for (a, b), n in edge_use.items()], key=lambda t: -t[2])[:10],
                randomizations=RANDOMIZATIONS, prediction=pred, model_tolerance_F_e=MODEL_TOLERANCE_FE,
                prereg=PREREG.name, prereg_sha256=prereg_sha,
                sources=[str(p) for p in [HERE / "hw_superact_run.py", HERE / "hw_two_use.py", HERE / "hw_predict.py"]
                         + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))])
    return F.freeze(named, meta)


# ------------------------------------------------------------------ analysis (pure)
def estimate(means: dict, units: list, variances: dict | None = None, shots: dict | None = None) -> dict:
    """the design's stratified estimate from per-circuit means (predicted or measured):
        F_avg = sum over units of weight x mean    (kept: W_b / C_b each; both-erased: exact weights)
    with, when variances are given, the shot sd and the design sd (spread of unit means within each sampled
    stratum; the enumerated stratum has none)"""
    f_avg = sum(u["weight"] * means[u["name"]] for u in units)
    out = dict(F_avg=f_avg, F_e=(3 * f_avg - 1) / 2)
    by: dict = {}
    for u in units:
        by.setdefault(u["stratum"], []).append(u)
    per = {}
    for s, us in by.items():
        W = sum(u["weight"] for u in us)
        per[s] = dict(weight=W, n=len(us), mean=sum(u["weight"] * means[u["name"]] for u in us) / W)
    out["strata"] = per
    if variances is not None:
        shot_var = sum((u["weight"] ** 2) * variances[u["name"]] / shots[u["name"]] for u in units)
        design_var = 0.0
        for s, us in by.items():
            if s == "erased_erased" or len(us) < 2:
                continue
            vals = [means[u["name"]] for u in us]
            W = sum(u["weight"] for u in us)
            design_var += W ** 2 * float(np.var(vals, ddof=1)) / len(us)
        out.update(shot_sd_F_avg=math.sqrt(shot_var), design_sd_F_avg=math.sqrt(design_var),
                   sd_F_e=1.5 * math.sqrt(shot_var + design_var), shot_sd_F_e=1.5 * math.sqrt(shot_var))
    return out


def branch_f_e(means: dict, units: list) -> dict:
    out = {}
    for b, wb in H.BRANCH_WEIGHT.items():
        us = [u for u in units if u["unit"]["branch"] == b]
        f_avg_b = sum(u["weight"] * means[u["name"]] for u in us) / wb
        out[b] = (3 * f_avg_b - 1) / 2
    return out


def ladder_fit(correlators: dict, vendor_r: float) -> dict:
    """Bell fidelity per depth F = (1 + <XX> - <YY> + <ZZ>)/4, lambda = (4F - 1)/3 fitted as lam0 (1 - eps)^depth
    (the R133 estimator: fit lambda, not F), then the CZ factor in process-infidelity units: (15 eps/16) / (5 r/4)"""
    F = {d: (1 + correlators[(d, "XX")] - correlators[(d, "YY")] + correlators[(d, "ZZ")]) / 4 for d in LADDER_DEPTHS}
    lam = {d: (4 * F[d] - 1) / 3 for d in LADDER_DEPTHS}
    ds = np.array(LADDER_DEPTHS, dtype=float)
    slope = float(np.polyfit(ds, np.log(np.clip([lam[d] for d in LADDER_DEPTHS], 1e-12, None)), 1)[0])
    eps = 1 - math.exp(slope)
    return dict(F=F, eps_per_cz=eps, cz_factor=(15 / 16 * eps) / (5 / 4 * vendor_r))


def analyse(record: dict, counts: dict) -> dict:
    units = record["units"]
    stat_units = [T.Unit(u["name"], u["weight"], u.get("shots", record["shots"]), u["n_bits"], 0.0, u["hi"],
                         u["stratum"]) for u in units]
    scores = {u["name"]: H.score(record["m"], unit_from(u["unit"])) for u in units}
    tallies = T.tally_all(stat_units, {u["name"]: counts[u["name"]] for u in units}, scores)
    means = {n: t.mean for n, t in tallies.items()}
    var = {n: max(t.s2 / t.n - t.mean ** 2, 0.0) for n, t in tallies.items()}
    shots = {n: t.n for n, t in tallies.items()}
    meas = estimate(means, units, var, shots)
    pred_means = {u["name"]: u["predicted"] for u in units}
    pred = estimate(pred_means, units)
    eng_var = sum((u["weight"] ** 2) * (u["predicted_se"] or 0.0) ** 2 for u in units)
    delta_fe = meas["F_e"] - pred["F_e"]
    delta_sd = 1.5 * math.sqrt(meas["shot_sd_F_avg"] ** 2 + eng_var)
    # the copy bit l equals the sector in the ideal circuit: its agreement is Bob's first frame bit, measured directly
    agree, total = 0, 0
    for u in units:
        if u["unit"]["branch"] == "erased_erased":
            continue
        want = 0 if u["unit"]["choice"][0] == "s0" else 1
        for key, c in counts[u["name"]].items():
            total += int(c)
            agree += int(c) * (T.bits_of(key, u["n_bits"])[0] == want)
    corr = {}
    for lad in record["ladder"]:
        n = lad["name"]
        tot = sum(int(c) for c in counts[n].values())
        s = sum(int(c) * (1 if T.bits_of(k, 2)[0] == T.bits_of(k, 2)[1] else -1) for k, c in counts[n].items())
        corr[(lad["depth"], lad["basis"])] = s / tot
    ladder = ladder_fit(corr, record["ladder_edge_vendor_cz"])
    ladder_pred = ladder_fit({(lad["depth"], lad["basis"]): lad["predicted_correlator"] for lad in record["ladder"]},
                             record["ladder_edge_vendor_cz"])
    upper = meas["F_e"] + 2.326 * meas["sd_F_e"]
    return dict(
        measured=dict(F_e=meas["F_e"], sd_F_e=meas["sd_F_e"], shot_sd_F_e=meas["shot_sd_F_e"], F_avg=meas["F_avg"],
                      upper99_F_e=upper, branches=branch_f_e(means, units)),
        predicted=dict(F_e=pred["F_e"], branches=branch_f_e(pred_means, units)),
        verdicts=dict(
            superactivation_on_this_device=bool(meas["F_e"] - 2.326 * meas["sd_F_e"] > 0.75),
            excluded_at_99=bool(upper < 0.75),
            model_holds_at_scale=bool(abs(delta_fe) <= record["model_tolerance_F_e"]),
            delta_F_e=delta_fe, delta_sd=delta_sd),
        strata={s: dict(measured=meas["strata"][s]["mean"], predicted=pred["strata"][s]["mean"],
                        weight=meas["strata"][s]["weight"], n=meas["strata"][s]["n"]) for s in meas["strata"]},
        copy_bit_agreement=agree / total if total else None,
        ladder=dict(measured=ladder, predicted=ladder_pred, edge=record["ladder_edge"]),
        per_unit={u["name"]: dict(measured=means[u["name"]], predicted=u["predicted"], ideal=u["ideal"], cz=u["cz"])
                  for u in units})


# ------------------------------------------------------------------ modes
def _paths(record_path: pathlib.Path):
    return record_path, record_path.with_suffix(".qpy")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["freeze", "submit", "fetch", "dryrun"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.mode == "freeze":
        from qiskit_ibm_runtime import QiskitRuntimeService
        be = QiskitRuntimeService().backend(a.live)
        snap = S.from_backend(be)
        if not PREREG.is_file():
            raise SystemExit(f"write {PREREG.name} first: the frozen record pins its hash")
        rec, blob = freeze_run(be, snap, U.file_sha256(PREREG))
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        p = rec["prediction"]
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; predicted F_e {p['F_e']:.4f}; layout "
              f"{rec['layout']}; ladder edge {rec['ladder_edge']} (vendor CZ {rec['ladder_edge_vendor_cz']:.2e})")
        return 0
    rp, qp = _paths(pathlib.Path(a.record))
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), qp.read_bytes()
    shots = {u["name"]: rec["shots"] for u in rec["units"]} | {lad["name"]: rec["ladder_shots"] for lad in rec["ladder"]}
    plan = U.Plan(tag="superact_m5", backend=rec["backend"], shots=shots, prereg_path=str(PREREG),
                  prereg_sha256=rec["prereg_sha256"], twirl_randomizations=rec["randomizations"],
                  records_dir=str(OUTDIR), predicted_pass_probability=None,
                  measurement_reason=("a preregistered MEASUREMENT of how far the device falls short of 3/4 and a test "
                                      "of the model at 14 qubits; the superactivation verdict is predicted to fail"))
    if a.mode == "dryrun":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        counts = U.local_run(rec, blob, FakeKingston(), shots)
        out = analyse(rec, counts)
        (OUTDIR / f"dryrun_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                                     newline="\n")
        print(json.dumps(out["verdicts"], indent=1), "\nmeasured", out["measured"]["F_e"], "predicted",
              out["predicted"]["F_e"])
        return 0
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
    jobs = sorted(OUTDIR.glob("superact_m5_*.job.json"))
    job = json.loads(jobs[-1].read_text(encoding="utf-8"))
    got = U.fetch(job, svc)
    if "counts" not in got:
        print(f"job {job['job_id']}: {got['status']}")
        return 4
    out = dict(job=job, execution_seconds=got["execution_seconds"], analysis=analyse(rec, got["counts"]),
               counts=got["counts"])
    dest = OUTDIR / f"result_{job['job_id']}.json"
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    an = out["analysis"]
    print(json.dumps(an["verdicts"], indent=1))
    print(f"measured F_e {an['measured']['F_e']:.4f} +- {an['measured']['sd_F_e']:.4f}; predicted "
          f"{an['predicted']['F_e']:.4f}; ladder CZ factor {an['ladder']['measured']['cz_factor']:.2f} -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
