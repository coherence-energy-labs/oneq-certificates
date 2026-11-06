"""R173: R166's circuits with twirling OFF -- does the server then play the compiled gates? (PREREG_r173_twirl_off_schedule)

R166 MEASURED, with twirling on, that every single-qubit run plays as one two-sx template and every CZ layer takes
132 ns. This runs the same eight circuits (hw_r166_executed_schedule's pick and circuits, on today's healthiest
qubits) with gate twirling OFF and the executed schedule requested, and reads them with R166's own readers.

  python r173_twirl_off.py freeze --fake | --live B
  python r173_twirl_off.py submit RECORD       owner act (authorized: "do all 25", item 21), every oneq.hw.submit guard
  python r173_twirl_off.py fetch RECORD
  python r173_twirl_off.py analyse RECORD
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_r166_executed_schedule as R166  # noqa: E402

from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import submit as U  # noqa: E402
from oneq.hw import timing as T  # noqa: E402

TAG = "r173_off"
PREREG = ROOT / "docs" / "PREREG_r173_twirl_off_schedule.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "r173_twirl_off"


def freeze_r173(be, snap, prereg_sha):
    from qiskit import transpile
    roles = R166.pick(snap)
    named, meta = [], []
    for name, qc, m in R166.circuits(be, roles):
        named.append((name, transpile(qc, be, initial_layout=list(range(be.num_qubits)), optimization_level=0,
                                      seed_transpiler=1)))
        meta.append(dict(name=name, shots=R166.SHOTS, **m))
    bad = sorted({r for _, qc in named for r in L.refusals(snap, qc, L.QubitPolicy())})
    return F.freeze(named, dict(
        protocol="R173: R166's circuits with gate twirling OFF -- the executed schedule of the compiled gates",
        backend=snap.name, snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
        roles=roles, used=[roles[r] for r in R166.ROLES], circuits=meta, twirling="off", dt=snap.dt,
        policy_refusals=bad, prereg=PREREG.name, prereg_sha256=prereg_sha,
        sources=[str(HERE / "r173_twirl_off.py"), str(HERE / "hw_r166_executed_schedule.py")]
        + [str(p) for p in sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))


def plan(rec: dict) -> U.Plan:
    return U.Plan(tag=TAG, backend=rec["backend"], shots={c["name"]: c["shots"] for c in rec["circuits"]},
                  prereg_path=str(PREREG), prereg_sha256=rec["prereg_sha256"], twirl_randomizations=None,
                  records_dir=str(OUTDIR), scheduler_timing=True,
                  measurement_reason="a preregistered INSTRUMENT READ: the executed schedule with twirling off")


def readings(report: dict) -> dict:
    """the prereg's readings from one schedule_report (pure)"""
    got = {n: r is not None for n, r in report.items()}
    out = dict(timing_returned=got)
    lad = report.get("ladder_bar")
    if lad is None:
        return out
    extra = {n: {role: v["extra"] for role, v in r["roles"].items() if v["extra"]}
             for n, r in report.items() if r is not None}
    out.update(extra_pulses=extra, plays_as_compiled=all(not v for v in extra.values()),
               cz_period_ns=(sorted(set(round(x) for x in lad["cz_period_ns"])) if lad["cz_period_ns"] else None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["freeze", "submit", "fetch", "analyse"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.mode == "freeze":
        if a.fake:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
            be = FakeKingston()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            be = QiskitRuntimeService().backend(a.live)
        rec, blob = freeze_r173(be, S.from_backend(be), U.file_sha256(PREREG))
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; cost {plan(rec).cost_s():.1f} s; roles {rec['roles']}; "
              f"policy refusals {rec['policy_refusals'][:4]}")
        return 0
    rp = pathlib.Path(a.record).resolve()
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    if a.mode == "analyse":
        jobs = [p for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json"))
                if json.loads(p.read_text(encoding="utf-8")).get("qpy_sha256") == rec["qpy_sha256"]]
        if len(jobs) != 1:
            raise SystemExit(f"REFUSED: {len(jobs)} job records for this frozen record; expected one")
        jid = json.loads(jobs[0].read_text(encoding="utf-8"))["job_id"]
        index = json.loads((rp.parent / f"flight_{jid}.json").read_text(encoding="utf-8"))
        rec_r = dict(rec, roles=rec["roles"])
        report = R166.schedule_report(rec_r, blob, index)
        out = dict(report=report, readings=readings(report))
        (rp.parent / f"result_{rp.stem}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8",
                                                        newline="\n")
        print(json.dumps(out["readings"], indent=1))
        return 0
    from qiskit_ibm_runtime import QiskitRuntimeService

    from oneq.hw import campaign as C
    svc = QiskitRuntimeService()
    be = svc.backend(rec["backend"])
    pl = plan(rec)
    if a.mode == "submit":
        if "fake" in rec["backend"].lower() or rp.name.startswith("frozen_fake"):
            raise SystemExit("REFUSED: a --fake freeze is a rehearsal; freeze --live first")
        live = S.from_backend(be)
        circs = F.load(rec, blob)
        health = sorted({r for qc in circs.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(pl, rec, blob, be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        return 0
    jobs = [p for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json"))
            if json.loads(p.read_text(encoding="utf-8")).get("qpy_sha256") == rec["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records for this frozen record; expected one")
    jid = json.loads(jobs[0].read_text(encoding="utf-8"))["job_id"]
    job = svc.job(jid)
    if "DONE" not in str(job.status()):
        print(f"job {jid}: {job.status()}")
        return 4
    result = job.result()
    index, arrays = C.flight_record(result, rec)
    index["job_metrics"] = C._jsonable(job.metrics()) if hasattr(job, "metrics") else None
    import numpy as np
    np.savez_compressed(rp.parent / f"flight_{jid}.npz", **arrays)
    (rp.parent / f"flight_{jid}.json").write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8",
                                                 newline="\n")
    (rp.parent / f"after_{jid}.snapshot.json").write_text(json.dumps(S.from_backend(be).to_json(), indent=1) + "\n",
                                                         encoding="utf-8", newline="\n")
    counts = U.counts_from_result(result, rec["circuit_order"])
    (rp.parent / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    print(f"fetched job {jid}: {len(counts)} circuits; schedule returned for "
          f"{sum(1 for p in index['pubs'] if T.from_metadata(p.get('metadata')) is not None)} of {len(index['pubs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
