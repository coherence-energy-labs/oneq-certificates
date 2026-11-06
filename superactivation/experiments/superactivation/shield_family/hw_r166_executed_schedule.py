"""R166: what the server ACTUALLY PLAYS under twirling -- its executed schedule, read under strategy "active" and
"active-accum".

R165 measured the twirl dressing's total effect on an idle qubit (+6.86e-4 flips per CZ layer under active-accum, none
under "active": 2.65x the one-pulse estimate). It could not tell what the server puts there. Two readings fit its
shots about equally: an x for every X or Y Pauli, or a fixed rz.sx.rz.sx.rz dressing in every layer. The same unknown
sits under R152's 1q excess, so the engine cannot carry the twirl faithfully until it is read. SamplerV2 returns the
compiled schedule of every PUB when asked (oneq.hw.timing); no job before 2026-10-17 asked for it.

Roles (all policy-admissible; the idle qubits pairwise uncoupled and uncoupled to both edges):
  a-b       a healthy edge: the CZ ladders
  c-d       a second healthy edge, uncoupled to a-b: a parallel layer, and the feed-forward pair
  it        idle, TOUCHED (sx then sxdg, back in |0>): active-accum twirls it
  ip        idle in |+>, read back in X
  iu        idle, never gated: no strategy here should twirl it
  il        idle, touched only AFTER the ladder: active-accum must leave it alone during it

Circuits (one classical register; clbit i is used[i]):
  ro_all0, ro_all1    readout of every used qubit, prepared 0 / 1 (bit order and readout)
  ladder_bar          8 CZs on a-b, a barrier over every used qubit after each (R165's structure)
  ladder_nobar        the same 8 CZs with no barriers between them
  slots               three parallel layers on a-b and c-d with different single-qubit slots between them: one sx,
                      a virtual rz only, an x, two sx, and nothing -- what the server merges and re-synthesizes
  dynamic             c measured mid-circuit; if 1, an x on d; two CZs on a-b before and after (the feed-forward
                      stall and how twirling treats a dynamic circuit)
  ro_end0, ro_end1    the readout repeated last

Bits, not raw IQ. Whether kerneled IQ works with a mid-circuit feed-forward is not documented, and a rejected job
would lose the schedule this probe exists for. The prereg records this as its stated exception to the
maximum-information rule.

  python hw_r166_executed_schedule.py freeze --fake | --live B
  python hw_r166_executed_schedule.py dryrun RECORD                 Aer: no schedule comes back (a NULL rehearsal)
  python hw_r166_executed_schedule.py submit RECORD --strategy S    owner act, every oneq.hw.submit guard
  python hw_r166_executed_schedule.py fetch RECORD --strategy S
  python hw_r166_executed_schedule.py analyse RECORD
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_r155_diag as R155  # noqa: E402

from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import submit as U  # noqa: E402
from oneq.hw import timing as T  # noqa: E402

STRATEGIES = ("active", "active-accum")
TAGS = {"active": "r166_active", "active-accum": "r166_accum"}
ROLES = ("a", "b", "c", "d", "it", "ip", "iu", "il")
IDLE = ("it", "ip", "iu", "il")
D_LADDER = 8
SHOTS = 1000
RANDOMIZATIONS = 4
ORDER = ("ro_all0", "ro_all1", "ladder_bar", "ladder_nobar", "slots", "dynamic", "ro_end0", "ro_end1")
LADDERS = ("ladder_bar", "ladder_nobar")
T_CZ_NS = 68.0
PREREG = ROOT / "docs" / "PREREG_r166_executed_schedule.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "r166_executed_schedule"


# ------------------------------------------------------------------ design
def pick(snap: S.DeviceSnapshot, pol=L.QubitPolicy()) -> dict:
    """two healthy admissible edges, uncoupled to each other, and four idle qubits uncoupled to both and to each
    other, by readout error: {role: physical qubit}"""
    nb = R155.graph(snap)
    ok = sorted((e for e, (err, _) in snap.cz.items()
                 if err <= pol.max_cz and R155.admissible(snap, e[0], pol) and R155.admissible(snap, e[1], pol)),
                key=lambda e: snap.cz[e][0] + snap.readout_error(e[0]) + snap.readout_error(e[1]))
    if not ok:
        raise ValueError("no admissible edge")
    e1 = ok[0]
    near1 = set(e1) | nb.get(e1[0], set()) | nb.get(e1[1], set())
    e2 = next((e for e in ok if not (set(e) & near1) and not ((nb.get(e[0], set()) | nb.get(e[1], set())) & set(e1))),
              None)
    if e2 is None:
        raise ValueError("no second admissible edge uncoupled to the first")
    blocked = near1 | set(e2) | nb.get(e2[0], set()) | nb.get(e2[1], set())
    idle: list = []
    for q in sorted((q for q in range(snap.num_qubits) if R155.admissible(snap, q, pol)), key=snap.readout_error):
        if q in blocked:
            continue
        idle.append(q)
        blocked |= {q} | nb.get(q, set())
        if len(idle) == len(IDLE):
            break
    if len(idle) < len(IDLE):
        raise ValueError(f"only {len(idle)} uncoupled admissible idle qubits")
    return dict(zip(ROLES, [e1[0], e1[1], e2[0], e2[1]] + idle))


def circuits(be, roles: dict) -> list:
    """(name, circuit, meta) in ORDER"""
    from qiskit import ClassicalRegister
    used = [roles[r] for r in ROLES]
    n = len(used)
    a, b, c, d = (roles[r] for r in ("a", "b", "c", "d"))
    it, ip, il = roles["it"], roles["ip"], roles["il"]

    def new(name):
        qc = R155._wide(be.num_qubits)
        qc.add_register(ClassicalRegister(n, "c"))
        qc.name = name
        return qc

    def prep(qc):
        qc.sx(it)
        qc.sxdg(it)                                      # back in |0>, touched
        qc.h(ip)

    def finish(qc, skip=()):
        qc.h(ip)
        qc.measure([q for q in used if q not in skip], [i for i, q in enumerate(used) if q not in skip])

    out = []
    for name, p in (("ro_all0", 0), ("ro_all1", 1)):
        out.append(_ro(be, used, p, name))
    for name, bar in (("ladder_bar", True), ("ladder_nobar", False)):
        qc = new(name)
        prep(qc)
        qc.barrier(used)
        for _ in range(D_LADDER):
            qc.cz(a, b)
            if bar:
                qc.barrier(used)
        if not bar:
            qc.barrier(used)
        qc.h(il)                                         # touched only after the ladder
        finish(qc)
        out.append((name, qc, dict(family="ladder", depth=D_LADDER, barriers=bar)))
    qc = new("slots")
    prep(qc)
    qc.barrier(used)
    qc.cz(a, b)
    qc.cz(c, d)                                          # layer 1
    qc.sx(a)                                             # one sx
    qc.rz(0.7, b)                                        # virtual only
    qc.x(c)                                              # an x; d: nothing
    qc.cz(a, b)
    qc.cz(c, d)                                          # layer 2
    qc.rz(0.3, a)
    qc.sx(a)
    qc.rz(1.1, a)
    qc.sx(a)                                             # two sx (a generic single-qubit gate)
    qc.x(b)
    qc.cz(a, b)
    qc.cz(c, d)                                          # layer 3
    qc.barrier(used)
    finish(qc)
    out.append(("slots", qc, dict(family="slots", depth=3)))
    qc = new("dynamic")
    prep(qc)
    qc.h(c)
    qc.barrier(used)
    qc.cz(a, b)
    qc.cz(a, b)
    ic = used.index(c)
    qc.measure(c, ic)                                    # mid-circuit; c is not read again
    with qc.if_test((qc.clbits[ic], 1)):
        qc.x(d)
    qc.cz(a, b)
    qc.cz(a, b)
    qc.barrier(used)
    finish(qc, skip=(c,))
    out.append(("dynamic", qc, dict(family="dynamic", depth=4)))
    for name, p in (("ro_end0", 0), ("ro_end1", 1)):
        out.append(_ro(be, used, p, name, sentinel=True))
    return out


def _ro(be, used, prep, name, sentinel=False):
    from qiskit import ClassicalRegister
    qc = R155._wide(be.num_qubits)
    qc.add_register(ClassicalRegister(len(used), "c"))
    qc.name = name
    if prep:
        qc.x(used)
    qc.measure(used, range(len(used)))
    return name, qc, dict(family="ro", prepared=prep, sentinel=sentinel)


def freeze_r166(be, snap, prereg_sha):
    from qiskit import transpile
    roles = pick(snap)
    named, meta = [], []
    for name, qc, m in circuits(be, roles):
        named.append((name, transpile(qc, be, initial_layout=list(range(be.num_qubits)), optimization_level=0,
                                      seed_transpiler=1)))
        meta.append(dict(name=name, shots=SHOTS, **m))
    bad = sorted({r for _, qc in named for r in L.refusals(snap, qc, L.QubitPolicy())})
    return F.freeze(named, dict(
        protocol="R166: the server's executed schedule under twirling -- active vs active-accum",
        backend=snap.name, snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
        roles=roles, used=[roles[r] for r in ROLES], circuits=meta, strategies=list(STRATEGIES), tags=TAGS,
        randomizations=RANDOMIZATIONS, dt=snap.dt, policy_refusals=bad, prereg=PREREG.name, prereg_sha256=prereg_sha,
        sources=[str(HERE / "hw_r166_executed_schedule.py")]
        + [str(p) for p in sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))


def plan(rec: dict, strategy: str) -> U.Plan:
    return U.Plan(tag=TAGS[strategy], backend=rec["backend"], shots={c["name"]: c["shots"] for c in rec["circuits"]},
                  prereg_path=str(PREREG), prereg_sha256=rec["prereg_sha256"], twirl_randomizations=RANDOMIZATIONS,
                  records_dir=str(OUTDIR), twirl_strategy=strategy, scheduler_timing=True,
                  measurement_reason="a preregistered INSTRUMENT READ: the server's executed schedule under twirling")


# ------------------------------------------------------------------ analysis (pure)
def _cz_starts(rows: list, q: int) -> list:
    return sorted({r.t0 for r in rows if r.qubit == q and r.gate == "cz" and r.pulse == T.PHYSICAL})


def circuit_report(rows: list, qc, rec: dict, name: str) -> dict:
    """one PUB's schedule read against its submitted circuit, by role"""
    roles = rec["roles"]
    dt = rec["dt"]
    cmp = T.compare(rows, qc)
    meta = next(c for c in rec["circuits"] if c["name"] == name)
    depth = meta.get("depth")
    by_role = {}
    for r in ROLES:
        q = roles[r]
        v = cmp.get(q, dict(compiled={}, executed={}, extra={}))
        extra = sum(v["extra"].values())
        by_role[r] = dict(qubit=q, compiled=v["compiled"], executed=v["executed"], extra=v["extra"],
                          extra_per_layer=(extra / depth) if depth else None)
    starts = _cz_starts(rows, roles["a"])
    period = np.diff(starts) if len(starts) > 1 else np.array([])
    gates = Counter(r.gate for r in rows)
    return dict(duration_ns=T.duration_dt(rows) * dt * 1e9, roles=by_role,
                cz_period_ns=[float(x * dt * 1e9) for x in period],
                barrier_rows=int(sum(1 for r in rows if r.gate == "barrier")),
                channels=sorted({r.channel.split(" ")[0].rstrip("0123456789_") for r in rows}),
                gate_names=dict(gates),
                control=[dict(gate=r.gate, channel=r.channel, t0_ns=r.t0 * dt * 1e9, dur_ns=r.duration * dt * 1e9)
                         for r in rows if r.gate in ("broadcast", "receive") or r.channel in ("Hub", "Receive")][:40])


def schedule_report(rec: dict, blob: bytes, index: dict) -> dict:
    """{circuit: report | None (no schedule returned)} for one job's flight record"""
    circs = F.load(rec, blob, check_sources=False)
    out = {}
    for pub in index["pubs"]:
        rows = T.from_metadata(pub.get("metadata"))
        out[pub["name"]] = None if rows is None else circuit_report(rows, circs[pub["name"]], rec, pub["name"])
    return out


def verdicts(reports: dict) -> dict:
    """the readings the prereg fixes, per strategy (reports: {strategy: schedule_report})"""
    out = {}
    for s, rep in reports.items():
        got = {n: r is not None for n, r in rep.items()}
        lad = rep.get("ladder_bar")
        v = dict(timing_returned=got)
        if lad is not None:
            idle = {r: lad["roles"][r] for r in IDLE}
            v.update(
                dressing_pulses_per_layer={r: idle[r]["extra_per_layer"] for r in IDLE},
                dressing_gates={r: sorted(idle[r]["extra"]) for r in IDLE},
                cz_period_ns=(float(np.mean(lad["cz_period_ns"])) if lad["cz_period_ns"] else None),
                padding_per_layer_ns=(float(np.mean(lad["cz_period_ns"])) - T_CZ_NS if lad["cz_period_ns"] else None))
        out[s] = v
    acc, act = out.get("active-accum", {}), out.get("active", {})
    pa, pc = acc.get("dressing_pulses_per_layer"), act.get("dressing_pulses_per_layer")
    if pa is not None and pc is not None:
        out["reading"] = dict(
            schedule_shows_the_dressing=bool(pa["it"] and pa["it"] > 0 and not pc["it"]),
            schedule_blind_to_twirling=bool(not pa["it"]),     # R165 proved the dressing exists under accum
            untouched_left_alone=bool(not pa["iu"]),
            late_left_alone_during_ladder=bool(not pa["il"]))
    return out


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
        rec, blob = freeze_r166(be, S.from_backend(be), U.file_sha256(PREREG))
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        pl = plan(rec, "active")
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; per job {pl.total_shots} shots, "
              f"cost {pl.cost_s():.1f} s each (x{U.QUOTA_MARGIN}); roles {rec['roles']}; "
              f"policy refusals {rec['policy_refusals'][:4]}")
        return 0
    rp = pathlib.Path(a.record).resolve()
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    if a.mode == "dryrun":
        from qiskit_ibm_runtime.fake_provider import FakeKingston

        from oneq.hw import campaign as C
        out = {}
        for s in STRATEGIES:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            from qiskit_ibm_runtime import SamplerV2
            be = FakeKingston()
            sim = AerSimulator(noise_model=NoiseModel.from_backend(be), method="matrix_product_state")
            circs = F.load(rec, blob)
            res = SamplerV2(mode=sim).run([(circs[n], None, SHOTS) for n in rec["circuit_order"]]).result()
            index, _arrays = C.flight_record(res, rec)
            out[s] = schedule_report(rec, blob, index)
        v = verdicts(out)
        (rp.parent / f"dryrun_{rp.stem}.json").write_text(json.dumps(dict(reports=out, verdicts=v), indent=1) + "\n",
                                                        encoding="utf-8", newline="\n")
        print(json.dumps(v, indent=1))
        return 0
    if a.mode == "analyse":
        reports = {}
        for s in STRATEGIES:
            jobs = [p for p in sorted(OUTDIR.glob(f"{TAGS[s]}_*.job.json"))
                    if json.loads(p.read_text(encoding="utf-8")).get("qpy_sha256") == rec["qpy_sha256"]]
            if len(jobs) != 1:
                raise SystemExit(f"REFUSED: {len(jobs)} {s} job records for this frozen record; expected one")
            jid = json.loads(jobs[0].read_text(encoding="utf-8"))["job_id"]
            index = json.loads((rp.parent / f"flight_{jid}.json").read_text(encoding="utf-8"))
            reports[s] = schedule_report(rec, blob, index)
        v = verdicts(reports)
        (rp.parent / f"result_{rp.stem}.json").write_text(json.dumps(dict(reports=reports, verdicts=v), indent=1)
                                                        + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(v, indent=1))
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
    counts = U.counts_from_result(result, rec["circuit_order"])
    (rp.parent / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    print(f"fetched {a.strategy} job {jid}: {len(counts)} circuits; schedule returned for "
          f"{sum(1 for p in index['pubs'] if T.from_metadata(p.get('metadata')) is not None)} of {len(index['pubs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
