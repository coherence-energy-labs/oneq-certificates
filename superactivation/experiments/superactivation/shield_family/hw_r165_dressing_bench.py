"""R165: the twirl dressing measured DIRECTLY -- idle qubits beside a CZ ladder, under strategy "active" and "active-accum".

R163 (offline) charged the server's twirl dressing at one pulse per idle qubit per CZ layer and closed 40% of R146's
held-out gap; R164 tests it on R146's own circuits but is blocked today (four of R146's edges at 2-4% vendor CZ
error). This measures the mechanism itself, on healthy qubits of the live calibration, in two small jobs that differ
ONLY in twirling.strategy:

  ladder   one healthy edge, d CZs (d = 0, 16, 48): each CZ is one twirled layer
  Z group  idle qubits put through SX then SXdg (back in |0>, but TOUCHED, so active-accum twirls them): a |0> qubit
           is not dephased and barely relaxes, so bit flips that grow with d ARE the dressing pulses' errors -- 2/3 of
           a depolarizing pulse error flips it (X and Y do, Z does not)
  X free   idle qubits in |+>, read back in X: the NET of the pulses' errors and their random-echo credit (a random
           X or Y between halves of a wait partly refocuses a quasi-static detuning)
  X echo   idle qubits in |+> with an explicit X at mid-ladder: the quasi-static part already echoed, so what remains
           is closer to the pulse error alone
All idle qubits are policy-admissible, pairwise uncoupled and uncoupled to the ladder edge (no ZZ, no spectator CZ).
Raw IQ on every shot; a readout block at each job's start and end.

  python hw_r165_dressing_bench.py freeze --fake | --live B      read-only
  python hw_r165_dressing_bench.py dryrun RECORD                  Aer for both strategies (Aer ignores the strategy:
                                                                  a NULL rehearsal)
  python hw_r165_dressing_bench.py submit RECORD --strategy S     owner act, every oneq.hw.submit guard
  python hw_r165_dressing_bench.py fetch RECORD --strategy S
  python hw_r165_dressing_bench.py analyse RECORD
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
import hw_r155_diag as R155  # noqa: E402
import hw_r164_twirl_strategy as R164  # noqa: E402

from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

STRATEGIES = ("active", "active-accum")
TAGS = {"active": "r165_active", "active-accum": "r165_accum"}
DEPTHS = (0, 16, 48)
GROUPS = ("z", "xfree", "xecho")
PER_GROUP = 7
SHOTS, SHOTS_RO = 4000, 2000
RANDOMIZATIONS = 8
RO_START, RO_END = ("ro_all0", "ro_all1"), ("ro_end0", "ro_end1")
Z_SIGNAL, Z_NULL = 3.0, 2.0
PREREG = ROOT / "docs" / "PREREG_r165_dressing_bench.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "r165_dressing_bench"


# ------------------------------------------------------------------ design
def pick(snap: S.DeviceSnapshot, per_group: int = PER_GROUP, pol=L.QubitPolicy()) -> dict:
    """the healthiest admissible edge inside the CZ policy, and 3 x per_group idle qubits: admissible, pairwise
    uncoupled and uncoupled to the edge, by readout error (greedy), dealt round-robin into the three groups"""
    nb = R155.graph(snap)
    ok = [e for e, (err, _) in snap.cz.items()
          if err <= pol.max_cz and R155.admissible(snap, e[0], pol) and R155.admissible(snap, e[1], pol)]
    if not ok:
        raise ValueError("no admissible edge")
    edge = min(ok, key=lambda e: snap.cz[e][0] + snap.readout_error(e[0]) + snap.readout_error(e[1]))
    blocked = set(edge) | nb.get(edge[0], set()) | nb.get(edge[1], set())
    idle: list = []
    for q in sorted((q for q in range(snap.num_qubits) if R155.admissible(snap, q, pol)), key=snap.readout_error):
        if q in blocked:
            continue
        idle.append(q)
        blocked |= {q} | nb.get(q, set())
        if len(idle) == 3 * per_group:
            break
    if len(idle) < 3 * per_group:
        raise ValueError(f"only {len(idle)} uncoupled admissible idle qubits")
    groups = {g: idle[i::3] for i, g in enumerate(GROUPS)}
    return dict(edge=list(edge), groups=groups)


def circuits(be, design: dict) -> list:
    """(name, circuit, meta). Clbits: the edge's two qubits, then the groups in GROUPS order"""
    from qiskit import ClassicalRegister
    a, b = design["edge"]
    g = design["groups"]
    used = [a, b] + g["z"] + g["xfree"] + g["xecho"]
    n = len(used)
    xs = g["xfree"] + g["xecho"]
    out = []
    for name, prep in ((RO_START[0], 0), (RO_START[1], 1)):
        out.append((name, _ro(be, used, prep), dict(family="ro", prepared=prep, qubits=used, shots=SHOTS_RO)))
    for d in DEPTHS:
        qc = R155._wide(be.num_qubits)
        qc.add_register(ClassicalRegister(n, "c"))
        qc.name = f"bench_d{d}"
        qc.sx(g["z"])
        qc.sxdg(g["z"])                                   # back in |0>, TOUCHED (active-accum now twirls them)
        qc.h(xs)
        qc.barrier(used)
        for k in range(d):
            qc.cz(a, b)
            qc.barrier(used)
            if k == d // 2 - 1:
                qc.x(g["xecho"])                          # the echo at mid-ladder; no barrier after it (the edge is
                                                          # not held: the X runs during the next layer)
        qc.h(xs)
        qc.measure(used, range(n))
        out.append((qc.name, qc, dict(family="bench", depth=d, qubits=used, shots=SHOTS)))
    for name, prep in ((RO_END[0], 0), (RO_END[1], 1)):
        out.append((name, _ro(be, used, prep), dict(family="ro", prepared=prep, qubits=used, shots=SHOTS_RO,
                                                     sentinel=True)))
    return out


def _ro(be, qubits, prep):
    from qiskit import ClassicalRegister
    qc = R155._wide(be.num_qubits)
    qc.add_register(ClassicalRegister(len(qubits), "c"))
    if prep:
        qc.x(qubits)
    qc.measure(qubits, range(len(qubits)))
    return qc


def freeze_r165(be, snap, prereg_sha):
    from qiskit import transpile
    design = pick(snap)
    named, meta = [], []
    for name, qc, m in circuits(be, design):
        qc.name = name
        named.append((name, transpile(qc, be, initial_layout=list(range(be.num_qubits)), optimization_level=0,
                                      seed_transpiler=1)))
        meta.append(dict(name=name, **m))
    bad = sorted({r for _, qc in named for r in L.refusals(snap, qc, L.QubitPolicy())})
    pred = {q: 1.5 * snap.sx_error[q] for qs in design["groups"].values() for q in qs}
    rec, blob = F.freeze(named, dict(
        protocol="R165: the twirl dressing measured directly -- idle qubits beside a CZ ladder, active vs active-accum",
        backend=snap.name, snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
        design=design, depths=list(DEPTHS), circuits=meta, strategies=list(STRATEGIES), tags=TAGS,
        randomizations=RANDOMIZATIONS, meas_type=R164.MEAS_TYPE, iq_calibration=list(RO_START),
        readout_qubits=meta[0]["qubits"], predicted_pulse_error=pred,
        predicted_z_flip_per_layer={str(q): 2 / 3 * p for q, p in pred.items() if q in design["groups"]["z"]},
        policy_refusals=bad, prereg=PREREG.name, prereg_sha256=prereg_sha, z_signal=Z_SIGNAL, z_null=Z_NULL,
        sources=[str(HERE / "hw_r165_dressing_bench.py"), str(HERE / "hw_r164_twirl_strategy.py")]
        + [str(p) for p in sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))
    return rec, blob


def plan(rec: dict, strategy: str) -> U.Plan:
    return U.Plan(tag=TAGS[strategy], backend=rec["backend"], shots={c["name"]: c["shots"] for c in rec["circuits"]},
                  prereg_path=str(PREREG), prereg_sha256=rec["prereg_sha256"], twirl_randomizations=RANDOMIZATIONS,
                  records_dir=str(OUTDIR), twirl_strategy=strategy, meas_type=rec["meas_type"],
                  iq_calibration=tuple(rec["iq_calibration"]),
                  measurement_reason="a preregistered DIAGNOSTIC: idle qubits under two twirling strategies")


# ------------------------------------------------------------------ analysis (pure)
def per_qubit(rec: dict, counts: dict) -> dict:
    """per idle qubit and depth: P(1) for the Z group, readout-corrected <X> for the X groups; with shot sds"""
    used = rec["readout_qubits"]
    n = len(used)
    ro = {}
    for i, q in enumerate(used):
        p1 = {}
        for name in RO_START:
            cnt = counts[name]
            p1[name] = sum(v for k, v in cnt.items() if k.zfill(n)[n - 1 - i] == "1") / sum(cnt.values())
        ro[q] = (p1[RO_START[0]], 1 - p1[RO_START[1]])
    out: dict = {}
    for gname, qs in rec["design"]["groups"].items():
        for q in qs:
            i = used.index(q)
            p01, p10 = ro[q]
            sa = 1 - p01 - p10
            pts = []
            for d in rec["depths"]:
                cnt = counts[f"bench_d{d}"]
                tot = sum(cnt.values())
                p1 = sum(v for k, v in cnt.items() if k.zfill(n)[n - 1 - i] == "1") / tot
                if gname == "z":                          # the corrected flip probability
                    val, sd = (p1 - p01) / sa, math.sqrt(max(p1 * (1 - p1), 1 / tot) / tot) / sa
                else:                                     # the corrected X contrast
                    val, sd = (1 - 2 * p1) / sa, 2 * math.sqrt(max(p1 * (1 - p1), 1 / tot) / tot) / sa
                pts.append((d, val, sd))
            out[str(q)] = dict(group=gname, points=pts)
    return out


def slope(points: list) -> tuple:
    """weighted least-squares slope per layer and its sd"""
    d = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    w = 1 / np.array([max(p[2], 1e-9) for p in points]) ** 2
    dm = (w * d).sum() / w.sum()
    sxx = (w * (d - dm) ** 2).sum()
    b = float((w * (d - dm) * (y - (w * y).sum() / w.sum())).sum() / sxx)
    return b, float(1 / math.sqrt(sxx))


def compare(rec: dict, counts: dict) -> dict:
    """per group: the mean per-layer slope under each strategy and accum minus active (flips: positive is a cost;
    contrast: negative is a cost), with sds; the Z group's measured dressing flip per layer against its prediction"""
    pq = {s: per_qubit(rec, counts[s]) for s in STRATEGIES}
    out: dict = {}
    for gname, qs in rec["design"]["groups"].items():
        rows = {}
        for s in STRATEGIES:
            sl = [slope(pq[s][str(q)]["points"]) for q in qs]
            w = np.array([1 / v ** 2 for _, v in sl])
            rows[s] = (float((w * np.array([b for b, _ in sl])).sum() / w.sum()), float(1 / math.sqrt(w.sum())))
        diff = rows["active-accum"][0] - rows["active"][0]
        sd = math.hypot(rows["active-accum"][1], rows["active"][1])
        out[gname] = dict(slope_active=rows["active"], slope_accum=rows["active-accum"], accum_minus_active=diff,
                          sd=sd, z=diff / sd if sd > 0 else float("nan"))
    pred = float(np.mean(list(rec["predicted_z_flip_per_layer"].values())))
    out["z"]["predicted_flip_per_layer"] = pred
    out["z"]["measured_over_predicted"] = out["z"]["accum_minus_active"] / pred if pred > 0 else float("nan")
    return out


def verdicts(cmp: dict, zs: float = Z_SIGNAL, zn: float = Z_NULL) -> dict:
    z = cmp["z"]
    return dict(E1_dressing_pulses_flip_idle_qubits=bool(z["z"] > zs), E2_no_dressing_pulse_error=bool(abs(z["z"]) < zn),
                net_on_superposed_idle_free=cmp["xfree"]["accum_minus_active"],
                net_on_superposed_idle_echoed=cmp["xecho"]["accum_minus_active"])


def analyse(rec: dict, counts: dict) -> dict:
    cmp = compare(rec, counts)
    return dict(schema="oneq.r165.analysis/1", groups=cmp, verdicts=verdicts(cmp),
                per_qubit={s: per_qubit(rec, counts[s]) for s in STRATEGIES},
                readout_drift=R164.readout_drift(dict(readout_qubits=rec["readout_qubits"]), counts))


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
        rec, blob = freeze_r165(be, S.from_backend(be), U.file_sha256(PREREG))
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; per job {plan(rec, 'active').total_shots} shots, "
              f"cost {plan(rec, 'active').cost_s():.1f} s each (x{U.QUOTA_MARGIN}); edge {rec['design']['edge']}; "
              f"groups {rec['design']['groups']}; policy refusals {rec['policy_refusals'][:4]}")
        return 0
    rp = pathlib.Path(a.record).resolve()
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    if a.mode == "dryrun":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        # matrix product state: 23 qubits that barely entangle (the idle qubits never interact) -- a per-shot
        # statevector of 2^23 amplitudes took over 10 minutes and was stopped (2026-10-17)
        counts = {s: U.local_run(rec, blob, FakeKingston(), plan(rec, s).shots, method="matrix_product_state")
                  for s in STRATEGIES}
        out = analyse(rec, counts)
        out["dryrun"] = "Aer on FakeKingston ignores the twirling strategy: a NULL rehearsal"
        (rp.parent / f"dryrun_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=float) + "\n",
                                                        encoding="utf-8", newline="\n")
        print(json.dumps(dict(groups=out["groups"], verdicts=out["verdicts"]), indent=1, default=float))
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
        print(json.dumps(dict(groups=out["groups"], verdicts=out["verdicts"]), indent=1, default=float))
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
    counts, report = R164.counts_from_iq(rec, blob, index, arrays)
    (rp.parent / f"iq_{jid}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    (rp.parent / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    print(f"fetched {a.strategy} job {jid}: {len(counts)} circuits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
