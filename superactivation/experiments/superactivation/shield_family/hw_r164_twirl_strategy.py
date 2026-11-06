"""R164: is the server's twirl dressing real? R146's p0 circuits under twirling strategy "active" against "active-accum".

R163 (offline, held out on R152): under qiskit-ibm-runtime's default strategy, active-accum, every qubit touched so far
is twirled at every CZ layer, so a qubit IDLE in a layer gets a random Pauli before and after it -- two physical
pulses whenever it is X or Y. No ladder sees it. Charged at one pulse's error (1.5 x the qubit's vendor sx error), it
closes 40% of R146's held-out miss (-0.051 -> -0.031) with nothing fitted. Strategy "active" twirls only a layer's own
qubits, so it dresses no idle qubit: same circuits, same CZ twirling, no idle pulses.

THE TEST. R146's 208 p0 circuits -- R145's static superactivation units, byte-identical in operation to R152's frozen
ones (semantic fingerprint checked) -- run twice, once per strategy, 264 shots each, 8 randomizations, measurement
twirling and DD off. Every unit is measured under both, so the analysis is PAIRED:
  Delta = F_e(active) - F_e(active-accum), and, per unit, the measured difference against the engine's predicted
  dressing loss L_u (oneq.hw.sensitivity, deterministic): the weighted slope kappa = sum w L D / sum w L^2 IS the
  dressing factor -- 1 if R163's model is right, 0 if there is no dressing cost.

  python hw_r164_twirl_strategy.py freeze --fake | --live B      read-only
  python hw_r164_twirl_strategy.py dryrun RECORD                  Aer on FakeKingston for both (Aer ignores the
                                                                  strategy: a NULL rehearsal, Delta must be ~0)
  python hw_r164_twirl_strategy.py submit RECORD --strategy S     owner act, every oneq.hw.submit guard
  python hw_r164_twirl_strategy.py fetch RECORD --strategy S      flight record, counts
  python hw_r164_twirl_strategy.py analyse RECORD                 both strategies' counts -> the verdicts
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_diag_run as D146  # noqa: E402
import hw_superact_run as HS  # noqa: E402
import hw_two_use as H  # noqa: E402
import r155_heldout_r152 as HO  # noqa: E402

from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import sensitivity as SE  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

STRATEGIES = ("active", "active-accum")          # the preregistered submission order
TAGS = {"active": "r164_active", "active-accum": "r164_accum"}
SHOTS = 264
SHOTS_RO = 2000                                  # the readout block at each job's start and end (the drift check)
RO_START, RO_END = ("ro_all0", "ro_all1"), ("ro_end0", "ro_end1")
MEAS_TYPE = "kerneled"                           # raw IQ on every shot (the owner's maximum-information rule)
RANDOMIZATIONS = 8
TRAJ = 24
Z_SIGNAL, Z_NULL = 3.0, 2.0
R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
PREREG = ROOT / "docs" / "PREREG_r164_twirl_strategy.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "r164_twirl_strategy"


# ------------------------------------------------------------------ the circuits and the model
def r146_part():
    rec = json.loads((R152_DIR / "r146.json").read_text(encoding="utf-8"))
    blob = (R152_DIR / "r146.qpy").read_bytes()
    return rec, F.load(rec, blob, check_sources=False)


def fingerprint(qc) -> str:
    """the circuit's operations, qubits, clbits and parameters (rounded to 1e-12): what the device executes, blind to
    QPY's serialization details"""
    rows = []
    for inst in qc.data:
        params = [round(float(p), 12) for p in inst.operation.params]
        rows.append([inst.operation.name, [qc.find_bit(x).index for x in inst.qubits],
                     [qc.find_bit(x).index for x in inst.clbits], params])
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def v3c_model():
    """R163's base: R152's in-circuit repair of R146, R155's white T2 + sigma, the vendor sx error; and its params"""
    rec, _ = r146_part()
    res = json.loads((R152_DIR / "result_db9c50oc77vs73817rog.json").read_text(encoding="utf-8"))
    idle = json.loads((ROOT / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    vendor = S.DeviceSnapshot.from_json(rec["snapshot"])
    m = HO.with_vendor_sx(HO.with_idle_map(D146.repaired_snapshot(rec, res["r146"]["fit"]), idle, True), vendor)
    return m, D146.repaired_params(res["r146"]["fit"])


def predictions(rec146: dict, circuits: dict) -> dict:
    """per unit: the predicted mean under each strategy and the dressing loss L_u (the linear response's dressing
    budget, deterministic); and each strategy's predicted F_e"""
    m, q = v3c_model()
    base = q.describe()
    out: dict = {"units": {}, "F_e": {}}
    means: dict = {s: {} for s in STRATEGIES}
    for k, u in enumerate(rec146["units"]):
        qc, score = circuits[u["name"]], H.score(rec146["m"], HS.unit_from(u["unit"]))
        row = {}
        for s, twirl in (("active", "off"), ("active-accum", "active-accum")):
            p = N.NoiseParams(**{**base, "quasistatic": "coherent", "twirl": twirl})
            means[s][u["name"]] = E.expectation(qc, score, m, p, TRAJ, np.random.default_rng([164, k])).mean
            row[s] = means[s][u["name"]]
        sens = SE.sensitivity(qc, score, m, N.NoiseParams(**{**base, "quasistatic": "coherent", "twirl": "active-accum"}))
        row["dressing_loss"] = float(sum(v for key, v in sens.loss_budget.items() if key[0] == "dressing"))
        out["units"][u["name"]] = row
    for s in STRATEGIES:
        out["F_e"][s] = HS.estimate(means[s], rec146["units"])["F_e"]
    out["model"] = "V3c (R152's repair + R155 white T2 + sigma + vendor sx), quasistatic coherent; 'active' = no " \
                   "idle dressing, 'active-accum' = R163's first-order dressing at one pulse"
    return out


def readout_circuits(be, qubits: list) -> list:
    """every R146 qubit prepared in |0> or |1> and read at once: the IQ discriminators, and the drift check"""
    import hw_r155_diag as R155
    from qiskit import ClassicalRegister, transpile
    out = []
    for name, prep in ((RO_START[0], 0), (RO_START[1], 1), (RO_END[0], 0), (RO_END[1], 1)):
        qc = R155._wide(be.num_qubits)
        qc.add_register(ClassicalRegister(len(qubits), "c"))
        qc.name = name
        if prep:
            qc.x(qubits)
        qc.measure(qubits, range(len(qubits)))
        out.append((name, transpile(qc, be, initial_layout=list(range(be.num_qubits)), optimization_level=0,
                                    seed_transpiler=1)))
    return out


def freeze_r164(be, snap, prereg_sha):
    rec146, circ = r146_part()
    names = [u["name"] for u in rec146["units"]]
    ro = readout_circuits(be, list(rec146["layout"]))
    named = ro[:2] + [(n, circ[n]) for n in names] + ro[2:]
    fp = {n: fingerprint(circ[n]) for n in names}
    bad = sorted({r for _, qc in named for r in L.refusals(snap, qc, L.QubitPolicy())})
    rec, blob = F.freeze(named, dict(
        protocol="R164: R146's p0 circuits under twirling strategy active vs active-accum (the twirl dressing test)",
        readout_qubits=list(rec146["layout"]), meas_type=MEAS_TYPE, iq_calibration=list(RO_START),
        backend=snap.name, snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
        origin=dict(job="db9c50oc77vs73817rog", part="r146", qpy_sha256=rec146["qpy_sha256"], semantic_sha256=fp),
        units=rec146["units"], m=rec146["m"], strategies=list(STRATEGIES), tags=TAGS, shots_per_unit=SHOTS,
        randomizations=RANDOMIZATIONS, predictions=predictions(rec146, circ), policy_refusals=bad,
        prereg=PREREG.name, prereg_sha256=prereg_sha, z_signal=Z_SIGNAL, z_null=Z_NULL,
        sources=[str(HERE / "hw_r164_twirl_strategy.py"), str(HERE / "r155_heldout_r152.py")]
        + [str(p) for p in sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))
    loaded = F.load(rec, blob, check_sources=False)
    drift = [n for n in names if fingerprint(loaded[n]) != fp[n]]
    if drift:
        raise SystemExit(f"REFUSED: {len(drift)} re-frozen circuits do not execute what R152's did: {drift[:3]}")
    return rec, blob


def plan(rec: dict, strategy: str) -> U.Plan:
    ro = set(RO_START) | set(RO_END)
    return U.Plan(tag=TAGS[strategy], backend=rec["backend"],
                  shots={n: SHOTS_RO if n in ro else SHOTS for n in rec["circuit_order"]},
                  prereg_path=str(PREREG), prereg_sha256=rec["prereg_sha256"], twirl_randomizations=RANDOMIZATIONS,
                  records_dir=str(OUTDIR), twirl_strategy=strategy, meas_type=rec.get("meas_type", "classified"),
                  iq_calibration=tuple(rec["iq_calibration"]) if rec.get("iq_calibration") else None,
                  measurement_reason="a preregistered DIAGNOSTIC: the same circuits under two twirling strategies")


def counts_from_iq(rec: dict, blob: bytes, index: dict, arrays: dict) -> tuple:
    """(counts, IQ report) for one job: discriminators from its own start-of-job readout circuits, every circuit
    classified, each unit's outlier (leakage-like) fraction, and the readout at the job's END against its start"""
    from oneq.hw import iq as IQ
    circs = F.load(rec, blob, check_sources=False)
    qbn = {n: IQ.measured_qubits(qc) for n, qc in circs.items()}
    iq = {n: IQ.iq_columns(index, arrays, n) for n in rec["circuit_order"]}
    discs = IQ.calibrate(iq, qbn, *rec["iq_calibration"])
    counts = IQ.classify_all(iq, qbn, discs)
    ends = {}
    for name, prep in ((RO_END[0], 0), (RO_END[1], 1)):
        bits = np.stack([IQ.classify(discs[q], iq[name][:, j]) for j, q in enumerate(qbn[name])], axis=1)
        ends[name] = {str(q): float(np.mean(bits[:, j] != bool(prep))) for j, q in enumerate(qbn[name])}
    return counts, dict(discriminators={str(q): d.to_json() for q, d in discs.items()},
                        outlier_fractions=IQ.outlier_fractions(iq, qbn, discs), end_assignment_error=ends)


def readout_drift(rec: dict, counts: dict) -> dict:
    """per strategy's job and qubit, the readout assignment error at the job's start and at its end (from bits): two
    jobs compared must not differ by their readout"""
    qubits = rec["readout_qubits"]
    n = len(qubits)
    out: dict = {}
    for s, c in counts.items():
        row = {}
        for i, q in enumerate(qubits):
            def err(name, prep):
                cnt = c[name]
                tot = sum(cnt.values())
                ones = sum(v for k, v in cnt.items() if k.zfill(n)[n - 1 - i] == "1")
                return (ones if prep == 0 else tot - ones) / tot
            row[str(q)] = dict(start=(err(RO_START[0], 0), err(RO_START[1], 1)), end=(err(RO_END[0], 0), err(RO_END[1], 1)))
        out[s] = row
    return out


# ------------------------------------------------------------------ analysis (pure)
def unit_means(rec: dict, counts: dict) -> tuple:
    """(means, variances of the mean) per unit from one strategy's counts"""
    units = rec["units"]
    stat = [T.Unit(u["name"], u["weight"], SHOTS, u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    scores = {u["name"]: H.score(rec["m"], HS.unit_from(u["unit"])) for u in units}
    tallies = T.tally_all(stat, {u["name"]: counts[u["name"]] for u in units}, scores)
    hi = {u["name"]: u["hi"] for u in units}
    mv = {n: mean_var(t, hi[n]) for n, t in tallies.items()}
    return {n: m for n, (m, _) in mv.items()}, {n: v for n, (_, v) in mv.items()}


def mean_var(t, hi: float) -> tuple:
    """(mean, variance of the mean) of one unit's tally. A unit whose every shot scored the same has a SAMPLE variance
    of 0, but its mean is still uncertain by up to ~range / n: without a floor such units took near-infinite weight
    and the dressing slope's sd read 7e-4 on a null (rehearsal, 2026-10-17). The floor is (hi / n)^2 -- one shot in
    n different."""
    return t.mean, max(max(t.s2 / t.n - t.mean ** 2, 0.0) / t.n, (hi / t.n) ** 2)


def paired(units: list, means: dict, var: dict, loss: dict) -> dict:
    """Delta F_e = F_e(active) - F_e(active-accum) with its shot sd, and the dressing factor kappa: the weighted slope
    of each unit's measured difference against its predicted dressing loss (loss < 0 is the accum arm's predicted
    deficit, so a REAL dressing makes active - accum = -kappa x loss)"""
    a, b = "active", "active-accum"
    w = {u["name"]: u["weight"] for u in units}
    delta = 1.5 * sum(w[n] * (means[a][n] - means[b][n]) for n in w)
    sd = 1.5 * math.sqrt(sum(w[n] ** 2 * (var[a][n] + var[b][n]) for n in w))
    x = np.array([-loss[n] for n in w])
    y = np.array([means[a][n] - means[b][n] for n in w])
    iv = np.array([1 / (var[a][n] + var[b][n]) for n in w])
    sxx = float((iv * x * x).sum())
    kappa = float((iv * x * y).sum() / sxx) if sxx > 0 else float("nan")
    kappa_sd = float(1 / math.sqrt(sxx)) if sxx > 0 else float("nan")
    return dict(delta_F_e=delta, sd=sd, z=delta / sd if sd > 0 else float("nan"), kappa=kappa, kappa_sd=kappa_sd)


def verdicts(p: dict, predicted_delta: float, zs: float = Z_SIGNAL, zn: float = Z_NULL) -> dict:
    return dict(D1_dressing_is_real=bool(p["z"] > zs), D2_no_material_dressing=bool(abs(p["z"]) < zn),
                kappa_consistent_with_model=bool(abs(p["kappa"] - 1) < zn * p["kappa_sd"]),
                predicted_delta_F_e=predicted_delta)


def analyse(rec: dict, counts: dict) -> dict:
    """counts: {strategy: {circuit: histogram}}"""
    mv = {s: unit_means(rec, counts[s]) for s in STRATEGIES}
    means = {s: mv[s][0] for s in STRATEGIES}
    var = {s: mv[s][1] for s in STRATEGIES}
    loss = {n: r["dressing_loss"] for n, r in rec["predictions"]["units"].items()}
    est = {s: HS.estimate(means[s], rec["units"], {n: v * SHOTS for n, v in var[s].items()},
                          {n: SHOTS for n in var[s]}) for s in STRATEGIES}
    p = paired(rec["units"], means, var, loss)
    pred = rec["predictions"]["F_e"]["active"] - rec["predictions"]["F_e"]["active-accum"]
    return dict(schema="oneq.r164.analysis/1", F_e={s: est[s]["F_e"] for s in STRATEGIES},
                shot_sd_F_e={s: est[s]["shot_sd_F_e"] for s in STRATEGIES}, paired=p, verdicts=verdicts(p, pred),
                predicted_F_e=rec["predictions"]["F_e"],
                readout_drift=readout_drift(rec, counts) if "readout_qubits" in rec else None)


# ------------------------------------------------------------------ modes
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["freeze", "dryrun", "submit", "fetch", "analyse"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--strategy", choices=STRATEGIES)
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.mode == "freeze":
        if not PREREG.is_file():
            raise SystemExit(f"write {PREREG.name} first: the frozen record pins its hash")
        if a.fake:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
            be = FakeKingston()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            be = QiskitRuntimeService().backend(a.live)
        rec, blob = freeze_r164(be, S.from_backend(be), U.file_sha256(PREREG))
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        costs = {s: plan(rec, s).cost_s() for s in STRATEGIES}
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; per job {plan(rec, 'active').total_shots} shots, "
              f"cost estimates {costs} (x{U.QUOTA_MARGIN} each); predicted F_e {rec['predictions']['F_e']}; "
              f"policy refusals {rec['policy_refusals'][:4]}")
        return 0
    rp = pathlib.Path(a.record).resolve()
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    if a.mode == "dryrun":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        counts = {s: U.local_run(rec, blob, FakeKingston(), plan(rec, s).shots) for s in STRATEGIES}
        out = analyse(rec, counts)
        out["dryrun"] = "Aer on FakeKingston ignores the twirling strategy: a NULL rehearsal (Delta must be ~0)"
        (rp.parent / f"dryrun_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=float) + "\n",
                                                        encoding="utf-8", newline="\n")
        print(json.dumps({k: out[k] for k in ("F_e", "paired", "verdicts")}, indent=1, default=float))
        return 0
    if a.mode == "analyse":
        counts = {}
        for s in STRATEGIES:
            jobs = [p for p in sorted(OUTDIR.glob(f"{TAGS[s]}_*.job.json"))
                    if json.loads(p.read_text(encoding="utf-8")).get("qpy_sha256") == rec["qpy_sha256"]]
            if len(jobs) != 1:
                raise SystemExit(f"REFUSED: {len(jobs)} {s} job records for this frozen record; expected one")
            jid = json.loads(jobs[0].read_text(encoding="utf-8"))["job_id"]
            counts[s] = json.loads((rp.parent / f"counts_{jid}.json").read_text(encoding="utf-8"))
        out = analyse(rec, counts)
        (rp.parent / f"result_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=float) + "\n",
                                                        encoding="utf-8", newline="\n")
        print(json.dumps({k: out[k] for k in ("F_e", "paired", "verdicts")}, indent=1, default=float))
        return 0
    if not a.strategy:
        raise SystemExit("--strategy is required for submit and fetch")
    from qiskit_ibm_runtime import QiskitRuntimeService

    from oneq.hw import campaign as C
    svc = QiskitRuntimeService()
    be = svc.backend(rec["backend"])
    pl = plan(rec, a.strategy)
    if a.mode == "submit":
        if "fake" in rec["backend"].lower() or rp.name.startswith("frozen_fake"):
            raise SystemExit("REFUSED: a --fake freeze is a rehearsal; freeze --live first")
        live = S.from_backend(be)
        circs = F.load(rec, blob)
        health = sorted({r for qc in circs.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(pl, rec, blob, be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted {a.strategy} job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        return 0
    jobs = [p for p in sorted(OUTDIR.glob(f"{TAGS[a.strategy]}_*.job.json"))
            if json.loads(p.read_text(encoding="utf-8")).get("qpy_sha256") == rec["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} {a.strategy} job records for this frozen record; expected one")
    jid = json.loads(jobs[0].read_text(encoding="utf-8"))["job_id"]
    job = svc.job(jid)
    if "DONE" not in str(job.status()):
        print(f"job {jid}: {job.status()}")
        return 4
    result = job.result()
    index, arrays = C.flight_record(result, rec)
    index["job_metrics"] = C._jsonable(job.metrics()) if hasattr(job, "metrics") else None
    np.savez_compressed(rp.parent / f"flight_{jid}.npz", **arrays)
    (rp.parent / f"flight_{jid}.json").write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8",
                                                 newline="\n")
    (rp.parent / f"after_{jid}.snapshot.json").write_text(json.dumps(S.from_backend(be).to_json(), indent=1) + "\n",
                                                         encoding="utf-8", newline="\n")
    if rec.get("meas_type") == "kerneled":
        counts, report = counts_from_iq(rec, blob, index, arrays)
        (rp.parent / f"iq_{jid}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    else:
        counts = U.counts_from_result(result, rec["circuit_order"])
    (rp.parent / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    print(f"fetched {a.strategy} job {jid}: {len(counts)} circuits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
