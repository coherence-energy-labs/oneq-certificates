"""R152: R146 and R147 in ONE job -- calibrate, model and test on one device in one calibration window.

Three parts, one SamplerV2 job (oneq.hw.campaign), every part's own frozen record and analysis unchanged:
  r146    R146 as preregistered and amended (hw_diag_run.freeze_diag): in-circuit calibration of R145's 14 qubits,
          the timing family that decides the device's schedule, and R145's circuits held out.
  blk147  R146's device families on R147's OWN qubits and on EVERY CZ edge R147's circuits use
          (hw_diag_run.block_circuits): the model R147's predictions are made on, measured in the same job.
  r147    R147's model test of measure-and-correct P (hw_dynamic_run.freeze_run), its predictions FROZEN AS A RULE:
          every unit's engine inputs are frozen (the dynamic units' static stand-ins in a second QPY), and
          predict_units() applies them to the block's fitted model under the schedule R146's timing family measures.
          The vendor-model prediction made at freeze is reported beside it.
Order in the job: r146, blk147, r147. All three are designed and predicted for 8 gate-twirl randomizations with
measurement twirling and dynamical decoupling off: one options set, so one job (compose refuses otherwise).

Modes:
  freeze  --live B | --fake    read-only: snapshot, the three freezes, the composite (records + QPYs + plan)
  dryrun  RECORD               each part through SamplerV2 LOCAL on FakeKingston under its own simulation rules
                               (R146 and the block padded ALAP; R147 not: qiskit cannot pad around if_else), then the
                               full analysis -- needs qiskit-aer (the tw-venv)
  submit  RECORD               OWNER ACT: every oneq.hw.submit guard on the composite, the block's coverage of R147,
                               the live calibration's health; one job; pointers in R146's and R147's own record dirs
                               so their ONE RUN guards see it
  fetch   RECORD               read-only: the flight record (every shot, the job's clock, metadata), the
                               post-job calibration, counts -> analyse
  analyse RECORD COUNTS        the analysis on saved counts (a dry run's or a fetch's)
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_diag_run as D146  # noqa: E402
import hw_dynamic as HD  # noqa: E402
import hw_dynamic_run as R147  # noqa: E402
from oneq.hw import campaign as C  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

PREREG = ROOT / "docs" / "PREREG_campaign_r152_one_job.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "campaign_r152"
TAG = "campaign_r152"
RANDOMIZATIONS = 8
PART_TAGS = ("r146", "blk147", "r147", "info")
T1_WAIT_S = 50e-6                    # long enough for a T1 of ~100-300 us to show a 15-40% decay at 3,672 shots
INFO_SHOTS = 3672                    # the ro family's allocation (R146 Amendment 1)


# ------------------------------------------------------------------ the EXPLORATORY part: outside every verdict
def info_circuits(be, snap: S.DeviceSnapshot, p146: C.Part, pblk: C.Part) -> tuple[list, int]:
    """the job's last circuits, read by no verdict:
      end_r146_ro_*, end_blk147_ro_*   R146's and the block's readout circuits AGAIN, byte-identical, renamed -- the
                                       readout drift across the job (the models' largest input, R146 Amendment 1)
      t1_r146, t1_blk147               every qubit of each set in |1>, a 50 us wait, read: an in-circuit T1 beside the
                                       vendor's (the models take T1 from the vendor; only R146's q57 is measured)"""
    from qiskit import QuantumCircuit
    out = []
    for p in (p146, pblk):
        circ = F.load(p.record, p.blob, check_sources=False)
        for n in p.record["circuit_order"]:
            if n.startswith("ro_"):
                out.append((f"end_{p.tag}_{n}", circ[n]))
    g = D146.GRANULARITY
    wait = int(round(T1_WAIT_S / snap.dt / g)) * g
    for p in (p146, pblk):
        qubits = p.record["layout"]
        qc = QuantumCircuit(len(qubits), len(qubits), name=f"t1_{p.tag}")
        qc.x(range(len(qubits)))
        qc.barrier()
        qc.delay(wait, range(len(qubits)), unit="dt")
        qc.barrier()
        qc.measure(range(len(qubits)), range(len(qubits)))
        out.append((f"t1_{p.tag}", D146.place(qc, be, qubits)))
    return out, wait


def _readout_from(counts: dict, prepared: list, qubits: list) -> dict:
    n = len(qubits)
    tot = {q: [0, 0, 0, 0] for q in qubits}          # n0, k0 (read 1 when 0), n1, k1 (read 0 when 1)
    for name, bits in prepared:
        p1, shots = D146._p1(counts[name], n)
        for i, q in enumerate(qubits):
            if bits[i]:
                tot[q][2] += shots
                tot[q][3] += (1 - p1[i]) * shots
            else:
                tot[q][0] += shots
                tot[q][1] += p1[i] * shots
    return {q: dict(p01=t[1] / t[0], p10=t[3] / t[2], n0=t[0], n1=t[2]) for q, t in tot.items()}


def analyse_info(rec: dict, parts: dict, per: dict) -> dict:
    """EXPLORATORY: readout at the job's end against its start (binomial z per qubit), and the in-circuit T1"""
    out: dict = dict(label="EXPLORATORY -- outside every preregistered verdict", readout_drift={}, t1={})
    info = per["info"]
    snap = S.DeviceSnapshot.from_json(rec["snapshot"])
    for tag in ("r146", "blk147"):
        p = parts[tag]
        qubits = p.record["layout"]
        diag = {d["name"]: d for d in p.record["diag"]}
        pats = [(f"end_{tag}_{n}", diag[n]["meta"]["prepared"]) for n in diag if n.startswith("ro_")]
        start = _readout_from(per[tag], [(n.split(f"end_{tag}_", 1)[1], b) for n, b in pats], qubits)
        end = _readout_from(info, pats, qubits)
        rows = {}
        for q in qubits:
            a, b = start[q], end[q]
            z = {}
            for k, n in (("p01", "n0"), ("p10", "n1")):
                pool = (a[k] * a[n] + b[k] * b[n]) / (a[n] + b[n])
                sd = (pool * (1 - pool) * (1 / a[n] + 1 / b[n])) ** 0.5
                z[k] = (b[k] - a[k]) / sd if sd > 0 else 0.0
            rows[str(q)] = dict(start=dict(p01=a["p01"], p10=a["p10"]), end=dict(p01=b["p01"], p10=b["p10"]), z=z)
        out["readout_drift"][tag] = dict(per_qubit=rows, max_abs_z=max(max(abs(v) for v in r["z"].values())
                                                                         for r in rows.values()))
        # T1: the |1> population after the wait, readout-inverted with the job-START readout, over its t = 0 value
        p1w, _ = D146._p1(info[f"t1_{tag}"], len(qubits))
        p1_0, _ = D146._p1(per[tag]["ro_all1"], len(qubits))
        t = parts["info"].record["t1_wait_dt"] * snap.dt
        rows = {}
        for i, q in enumerate(qubits):
            ro = start[q]
            s = 1 - ro["p01"] - ro["p10"]
            c0 = (p1_0[i] - ro["p01"]) / s
            cw = (p1w[i] - ro["p01"]) / s
            ratio = cw / c0 if c0 > 0 else float("nan")
            t1 = -t / np.log(ratio) if 0 < ratio < 1 else float("inf")
            rows[str(q)] = dict(t1_in_circuit_s=t1, t1_vendor_s=snap.t1[q],
                                ratio_to_vendor=(t1 / snap.t1[q]) if np.isfinite(t1) else None)
        out["t1"][tag] = dict(wait_s=t, per_qubit=rows)
    return out


# ------------------------------------------------------------------ what R147 needs measured
def _cz_edges(circuits) -> set:
    out = set()
    for qc in circuits:
        for inst in qc.data:
            if inst.operation.name == "cz":
                out.add(tuple(sorted(qc.find_bit(x).index for x in inst.qubits)))
    return out


def r147_needs(rec147: dict, circ147: dict, pred147: dict) -> tuple[list, list]:
    """the qubits R147's predictions read (every qubit its frozen circuits touch) and the REAL CZ edges they use --
    the submitted circuits' and the stand-ins', minus the stand-ins' virtual feed-forward edges, which the engine
    augments at zero error and no device has"""
    virt = {tuple(sorted(e)) for u in rec147["units"] if u["prediction_inputs"]["kind"] == "dynamic"
            for e in u["prediction_inputs"]["virt"]}
    edges = (_cz_edges(circ147.values()) | _cz_edges(pred147.values())) - virt
    qubits: set = set()
    for qc in list(circ147.values()) + list(pred147.values()):
        for inst in qc.data:
            if inst.operation.name not in ("barrier", "delay"):
                qubits.update(qc.find_bit(x).index for x in inst.qubits)
    return sorted(qubits), sorted(edges)


def r147_window_dt(rec147: dict, pred147: dict, snap: S.DeviceSnapshot) -> int:
    """R147's own median gate window (the dynamic units' static stand-ins, final readout removed), rounded to the
    delay granularity: the wait the block's X-basis family measures dephasing over. Timed on the snapshot the engine
    times them on -- augmented with the stand-ins' virtual feed-forward edges, which no device calibrates"""
    durs = []
    for u in rec147["units"]:
        inp = u["prediction_inputs"]
        if inp["kind"] == "dynamic":
            sn = HD.augmented(snap, [tuple(e) for e in inp["virt"]])
            durs.append(N.duration_s(pred147[inp["c0"]].remove_final_measurements(inplace=False), sn))
    g = D146.GRANULARITY
    return int(round(statistics.median(durs) / snap.dt / g)) * g


# ------------------------------------------------------------------ freeze
def freeze_all(be, snap: S.DeviceSnapshot, fake: bool) -> dict:
    sha146 = U.file_sha256(D146.PREREG)
    sha147 = U.file_sha256(R147.PREREG)
    rec146, blob146 = D146.freeze_diag(be, snap, sha146, check_health=not fake)
    # the at-freeze prediction is the VENDOR model's (reported only); the preregistered one is the rule applied to the
    # block's fit. --fake: idle off (the device Aer simulates has no idle noise around if_else, R147's rehearsal rule)
    rec147, blob147, pblob147 = R147.freeze_run(be, snap, sha147, cz_factor=R147.REPAIRED_CZ_FACTOR, idle=not fake,
                                                schedule="alap", return_prediction_blob=True)
    circ147 = F.load(rec147, blob147, check_sources=False)
    pred147 = R147.load_prediction_circuits(rec147, pblob147)
    qubits, edges = r147_needs(rec147, circ147, pred147)
    named, meta = D146.block_circuits(be, snap, qubits, edges, r147_window_dt(rec147, pred147, snap),
                                      check_health=not fake)
    recblk, blobblk = F.freeze(named, dict(meta, prereg=PREREG.name, prereg_sha256=U.file_sha256(PREREG)))
    shots147 = R147._plan(rec147).shots
    parts = [
        C.Part("r146", rec146, blob146, D146.shots_of(rec146), rec146["randomizations"], str(D146.PREREG), sha146,
               str(D146.OUTDIR), "diag_r146"),
        C.Part("blk147", recblk, blobblk, D146.shots_of(recblk), recblk["randomizations"], str(PREREG),
               U.file_sha256(PREREG)),
        C.Part("r147", rec147, blob147, shots147, rec147["randomizations"], str(R147.PREREG), sha147,
               str(R147.OUTDIR), R147.TAG, extra=dict(prediction_qpy_sha256=rec147["prediction_circuits"]["qpy_sha256"])),
    ]
    inamed, iwait = info_circuits(be, snap, parts[0], parts[1])
    recinf, blobinf = F.freeze(inamed, dict(
        kind="R152 EXPLORATORY: end-of-job readout sentinel and in-circuit T1, outside every verdict",
        t1_wait_dt=iwait, t1_wait_s=iwait * snap.dt, prereg=PREREG.name, prereg_sha256=U.file_sha256(PREREG)))
    parts.append(C.Part("info", recinf, blobinf, {n: INFO_SHOTS for n, _ in inamed}, RANDOMIZATIONS, str(PREREG),
                        U.file_sha256(PREREG)))
    rec, blob = C.compose(parts, dict(
        protocol="R152: R146 + an in-circuit calibration block on R147's qubits and edges + R147, in one job",
        backend=snap.name, snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
        fake=fake, prereg=PREREG.name, prereg_sha256=U.file_sha256(PREREG),
        r147_block=dict(qubits=qubits, edges=[list(e) for e in edges]),
        sources=[str(p) for p in [HERE / "hw_campaign_run.py", HERE / "hw_diag_run.py", HERE / "hw_dynamic_run.py"]
                 + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))
    return dict(rec=rec, blob=blob, parts={p.tag: p for p in parts}, pred147=pblob147)


def save(out: dict, stamp: str) -> pathlib.Path:
    d = OUTDIR / stamp
    d.mkdir(parents=True, exist_ok=True)
    for tag, p in out["parts"].items():
        (d / f"{tag}.json").write_text(json.dumps(p.record, indent=1) + "\n", encoding="utf-8", newline="\n")
        (d / f"{tag}.qpy").write_bytes(p.blob)
    (d / "r147.pred.qpy").write_bytes(out["pred147"])
    rp = d / "composite.json"
    rp.write_text(json.dumps(out["rec"], indent=1) + "\n", encoding="utf-8", newline="\n")
    rp.with_suffix(".qpy").write_bytes(out["blob"])
    return rp


def load(rp: pathlib.Path) -> dict:
    """the composite and its parts from disk, each part re-verified against the composite's pins"""
    rec = json.loads(rp.read_text(encoding="utf-8"))
    blob = rp.with_suffix(".qpy").read_bytes()
    parts = {}
    for pin in rec["parts"]:
        prec = json.loads((rp.parent / f"{pin['tag']}.json").read_text(encoding="utf-8"))
        pblob = (rp.parent / f"{pin['tag']}.qpy").read_bytes()
        parts[pin["tag"]] = C.Part(pin["tag"], prec, pblob, pin["shots"], rec["randomizations"], pin["prereg"],
                                   pin["prereg_sha256"], pin["records_dir"], pin["run_tag"], pin["extra"])
    C.verify_parts(rec, list(parts.values()))
    pred = (rp.parent / "r147.pred.qpy").read_bytes()
    if F.sha256(pred) != parts["r147"].record["prediction_circuits"]["qpy_sha256"]:
        raise SystemExit("REFUSED: r147.pred.qpy is not the prediction QPY R147's record pins")
    return dict(rec=rec, blob=blob, parts=parts, pred147=pred)


def plan(rec: dict) -> U.Plan:
    return U.Plan(tag=TAG, backend=rec["backend"], shots=rec["shots"], prereg_path=str(PREREG),
                  prereg_sha256=rec["prereg_sha256"], twirl_randomizations=rec["randomizations"],
                  records_dir=str(OUTDIR), predicted_pass_probability=None,
                  measurement_reason=("a preregistered calibration (R146), a calibration block and a MODEL TEST (R147) "
                                      "in one job; no protocol claim is at stake"))


# ------------------------------------------------------------------ analysis
def coverage_refusals(r147_rec: dict, circ147: dict, pred147: dict, block_rec: dict) -> list:
    """R147's precondition, recomputed: the block ladders EVERY real CZ edge R147's circuits use and reads every
    qubit they touch (no edge's channel may be the repaired model's median fill-in)"""
    qubits, edges = r147_needs(r147_rec, circ147, pred147)
    laddered = {tuple(e) for r in block_rec["ladder_rounds"] for e in map(tuple, r)}
    out = []
    missing = sorted(set(edges) - laddered)
    if missing:
        out.append(f"the block ladders none of R147's edges {missing[:6]}")
    unread = sorted(set(qubits) - set(block_rec["layout"]))
    if unread:
        out.append(f"the block does not read R147's qubits {unread[:6]}")
    return out


def analyse(loaded: dict, counts: dict) -> dict:
    rec, parts = loaded["rec"], loaded["parts"]
    per = C.split(rec, counts)
    p146, pblk, p147 = parts["r146"], parts["blk147"], parts["r147"]
    out146 = D146.analyse(p146.record, per["r146"], p146.blob)
    schedule = out146["fit"]["timing"]["schedule"]
    fitb = D146.fit_device(pblk.record, per["blk147"])
    model = D146.repaired_snapshot(pblk.record, fitb)
    circ147 = F.load(p147.record, p147.blob, check_sources=False)
    pred147 = R147.load_prediction_circuits(p147.record, loaded["pred147"])
    refusals = coverage_refusals(p147.record, circ147, pred147, pblk.record)
    measured = D146.measured_edges(fitb)
    _, needed = r147_needs(p147.record, circ147, pred147)
    if not set(needed) <= measured:
        refusals.append(f"the block's fit returned no channel for {sorted(set(needed) - measured)[:6]}")
    params = N.NoiseParams(cz_factor=R147.REPAIRED_CZ_FACTOR, schedule="alap" if schedule == "undetermined" else schedule,
                           idle=bool(p147.record["idle_modelled"]))
    preds = R147.predict_units(p147.record, circ147, pred147, model, params)
    rec147p = R147.with_predictions(p147.record, preds, model)
    out147 = R147.analyse(rec147p, per["r147"])
    vendor = R147.analyse(p147.record, per["r147"])["predicted"]
    info = analyse_info(rec, parts, per)
    return dict(
        schema="oneq.campaign_r152.analysis/1", composite_qpy_sha256=rec["qpy_sha256"], backend=rec["backend"],
        r146=out146, block=dict(fit=fitb, measured_edges=sorted(map(list, measured)), coverage_refusals=refusals,
                                model_sha256=model.sha256()),
        r147=dict(out147, prediction_rule=dict(model="the block's repaired snapshot", noise=params.describe(),
                                               schedule_from_r146=schedule),
                  vendor_prediction=vendor, verdicts_valid=not refusals),
        exploratory=info,
        headline=dict(
            exploratory_readout_drift_max_z={t: v["max_abs_z"] for t, v in info["readout_drift"].items()},
            r146_V1_repaired_model_holds=out146["verdicts"]["V1_repaired_model_holds"],
            r146_delta_repaired=out146["held_out"]["p0"]["delta_repaired"],
            r146_schedule=schedule,
            r147_F_e_measured=out147["measured"]["F_e"], r147_F_e_predicted=out147["predicted"]["F_e"],
            r147_F_e_vendor=vendor["F_e"], r147_V2=out147["verdicts"]["V2_model_within_tolerance"],
            r147_V3=out147["verdicts"]["V3_idle_accounting"], r147_V4=out147["verdicts"]["V4_dynamic_beats_static"],
            r147_V5=out147["verdicts"]["V5_mcm_readout"], r147_verdicts_valid=not refusals))


# ------------------------------------------------------------------ modes
def dryrun_counts(loaded: dict) -> dict:
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    be = FakeKingston()
    counts = {}
    for tag in PART_TAGS:
        p = loaded["parts"][tag]
        # the block reads ~26 qubits at once: too wide for a per-shot statevector, and only product states and pairwise
        # CZ ladders, which a matrix-product state holds exactly (checked against statevector on R146's 14-qubit
        # families, worst z 2.4 over every marginal and ladder parity)
        c = U.local_run(p.record, p.blob, be, p.shots, pad_alap=(tag != "r147"),
                        method="matrix_product_state" if tag in ("blk147", "info") else "statevector")
        counts.update({f"{tag}{C.SEP}{n}": v for n, v in c.items()})
    return counts


def submit_refusals(loaded: dict, live: S.DeviceSnapshot) -> list:
    rec, parts = loaded["rec"], loaded["parts"]
    out = []
    if rec.get("fake"):
        out.append("a --fake freeze is a rehearsal: re-freeze --live")
    p147 = parts["r147"]
    circ147 = F.load(p147.record, p147.blob, check_sources=False)
    pred147 = R147.load_prediction_circuits(p147.record, loaded["pred147"])
    out += coverage_refusals(p147.record, circ147, pred147, parts["blk147"].record)
    if p147.record.get("idle_modelled") is not True:
        out.append("R147's predictions do not model idle errors (a rehearsal setting)")
    if p147.record.get("cz_factor") != R147.REPAIRED_CZ_FACTOR:
        out.append("R147 was not frozen at the repaired model's CZ factor")
    for tag, p in parts.items():
        if p.prereg_path and U.file_sha256(p.prereg_path) != p.prereg_sha256:
            out.append(f"{tag}: {p.prereg_path} changed since the freeze")
        if p.records_dir and p.run_tag:
            prior = sorted(pathlib.Path(p.records_dir).glob(f"{p.run_tag}_*.job.json"))
            if any(json.loads(x.read_text(encoding="utf-8")).get("prereg_sha256") == p.prereg_sha256 for x in prior):
                out.append(f"{tag}: ONE RUN -- a job for its preregistration already exists in {p.records_dir}")
    return out


def _rel(p: pathlib.Path) -> str:
    """repository-relative when inside it, else as given (an unguarded relative_to raised here AFTER a real job was
    created, on 2026-10-17: the record path was relative and ROOT absolute -- the pointers below were not written)"""
    p = p.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def write_pointers(loaded: dict, job: dict, rp: pathlib.Path) -> list:
    """the composite's job record, written into each part's OWN record directory under its own run tag and
    preregistration, so that part's runner (hw_diag_run.py / hw_dynamic_run.py submit) sees the job and refuses a
    second run. Idempotent: the same job gives the same files."""
    out = []
    for tag, p in loaded["parts"].items():
        if not (p.records_dir and p.run_tag):
            continue
        d = pathlib.Path(p.records_dir)
        d.mkdir(parents=True, exist_ok=True)
        ptr = dict(job, tag=p.run_tag, prereg=p.prereg_path, prereg_sha256=p.prereg_sha256,
                   qpy_sha256=p.record["qpy_sha256"], campaign=TAG, campaign_part=tag, campaign_record=_rel(rp),
                   campaign_qpy_sha256=job["qpy_sha256"])
        f = d / f"{p.run_tag}_{job['job_id']}.job.json"
        f.write_text(json.dumps(ptr, indent=1) + "\n", encoding="utf-8", newline="\n")
        out.append(_rel(f))
    return out


def the_job(loaded: dict) -> dict:
    """the ONE composite job record submitted from this frozen composite (matched by its QPY hash)"""
    jobs = [j for j in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json")))
            if j.get("qpy_sha256") == loaded["rec"]["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records for this composite; expected one")
    return jobs[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["freeze", "dryrun", "submit", "fetch", "analyse", "pointers"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("counts", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    if a.mode == "freeze":
        if not PREREG.is_file():
            raise SystemExit(f"write {PREREG.name} first: the frozen record pins its hash")
        if a.fake:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
            be = FakeKingston()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            be = QiskitRuntimeService().backend(a.live)
        out = freeze_all(be, S.from_backend(be), a.fake)
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = save(out, stamp)
        pl = plan(out["rec"])
        print(f"frozen {out['rec']['n_circuits']} circuits -> {_rel(rp)}; {pl.total_shots} "
              f"shots, cost estimate {pl.cost_s():.1f} s (x{U.QUOTA_MARGIN} margin: {pl.cost_s() * U.QUOTA_MARGIN:.0f} "
              f"s); R147 vendor-model F_e {out['parts']['r147'].record['prediction']['F_e']:.4f}")
        return 0
    rp = pathlib.Path(a.record).resolve()
    loaded = load(rp)
    if a.mode == "pointers":                     # network-free: (re)write the parts' pointers from the job record
        print("pointers:", write_pointers(loaded, the_job(loaded), rp))
        return 0
    if a.mode in ("dryrun", "analyse"):
        counts = (json.loads(pathlib.Path(a.counts).read_text(encoding="utf-8")) if a.mode == "analyse"
                  else dryrun_counts(loaded))
        if a.mode == "dryrun" and a.counts:
            pathlib.Path(a.counts).write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
        res = analyse(loaded, counts)
        dest = rp.parent / f"{'dryrun' if a.mode == 'dryrun' else 'analysis'}_{rp.stem}.json"
        dest.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(res["headline"], indent=1))
        return 0
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    be = svc.backend(loaded["rec"]["backend"])
    pl = plan(loaded["rec"])
    if a.mode == "submit":
        live = S.from_backend(be)
        bad = submit_refusals(loaded, live)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        circuits = F.load(loaded["rec"], loaded["blob"])
        health = sorted({r for qc in circuits.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(pl, loaded["rec"], loaded["blob"], be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        print("pointers:", write_pointers(loaded, job, rp))
        return 0
    jobs = [the_job(loaded)]
    job = svc.job(jobs[0]["job_id"])
    if "DONE" not in str(job.status()):
        print(f"job {jobs[0]['job_id']}: {job.status()}")
        return 4
    result = job.result()
    index, arrays = C.flight_record(result, loaded["rec"])
    index["job_metrics"] = C._jsonable(job.metrics()) if hasattr(job, "metrics") else None
    after = S.from_backend(be)
    snap = S.DeviceSnapshot.from_json(loaded["rec"]["snapshot"])
    blk = loaded["parts"]["blk147"].record
    p146 = loaded["parts"]["r146"].record
    edges = sorted({tuple(e) for e in blk["edges"]} | {tuple(e) for r in p146["ladder_rounds"] for e in r})
    index["calibration_drift"] = C.calibration_drift(snap, after, sorted(set(blk["layout"]) | set(p146["layout"])),
                                                     edges)
    stem = rp.parent / f"flight_{jobs[0]['job_id']}"
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8",
                                         newline="\n")
    (rp.parent / f"after_{jobs[0]['job_id']}.snapshot.json").write_text(json.dumps(after.to_json(), indent=1) + "\n",
                                                                       encoding="utf-8", newline="\n")
    counts = U.counts_from_result(result, loaded["rec"]["circuit_order"])
    (rp.parent / f"counts_{jobs[0]['job_id']}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8",
                                                               newline="\n")
    res = analyse(loaded, counts)
    (rp.parent / f"result_{jobs[0]['job_id']}.json").write_text(json.dumps(res, indent=1, default=str) + "\n",
                                                                encoding="utf-8", newline="\n")
    print(json.dumps(res["headline"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
