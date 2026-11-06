"""R146: calibrate ibm_kingston IN-CIRCUIT on R145's own 14 qubits, then predict R145's own circuits HELD OUT -- one job.

R145's model missed its preregistered test by 0.042 of F_e. Post hoc (R146 ledger) two mechanisms account for ~72% of it
-- excess SPAM, lumped (the ladder's XX/YY sit low at EVERY depth, ZZ does not), and a symmetric flip on the q57 copy
path (+2.9%, 30 sd) -- and neither the R145 counts nor the vendor calibration can separate the rest. This job measures
the device where the protocol runs it, and then tests the repaired model on circuits the fit never saw.

FIT families (the repaired model is a function of these counts ONLY):
  ro_*   simultaneous readout calibration of the 14 qubits: all-0, all-1, and the two alternating patterns -> per-qubit
         p01 / p10 as the protocol reads them (all measured at once)
  xs_*   all 14 in |+>, rotated back at once (w0) and after the protocol's own ALAP duration (w) -> per-qubit X-basis
         SPAM and the in-circuit dephasing over that window, as an effective T2
  sk_*   all 14 in |+>, then 64 / 256 sx (identities in fours), rotated back -> per-qubit in-circuit sx error from the
         contrast's DECAY with the count (xs_w0 is the zero point), immune to the X-basis SPAM that sets its intercept
  pb*_*  product-basis ladders on the layout's 10 busiest edges, run in disjoint PARALLEL rounds (the protocol runs its
         CZs in parallel too): 7 bases (ZZ, XZ, ZX, YZ, ZY, XX, XY) x depth 0 / 16 / 48, twirled. Each basis reads three
         correlators, and each decays per CZ by sqrt(f_P f_G(P)) -- so together they measure all 9 learnable quantities
         of each edge's CZ channel (oneq.hw.cz_channel), and the FULL Pauli channel follows in the symmetric gauge.
         Rates only: every (observable, basis) series has its own free intercept, so SPAM cannot leak into a gate rate.
         (It replaces Bell + |++> ladders: the Bell lambda is blind to XX/YY/ZZ errors, and a 3-parameter family missed
         the total error by up to 19% on realistic channels.)
  tm_*   timing: q57 flipped to |1> then left alone while q47 runs 400 sx (14 us); measured with and without a final
         barrier. Aligned terminal readout (the model's ALAP) decays q57 by ~5% in BOTH; as-soon-as-possible readout
         only in the barrier one. Decides which schedule the model must use.
HELD OUT (never enter the fit; analyse() fits on a count dict with these names REMOVED, then predicts them):
  cp_*   the copy primitive alone: q47 in |0>/|1>, CNOT 47 -> 57, both read -> the copy flip the fit predicts
  p0_*   R145's 208 frozen protocol circuits, byte-identical -> F_e under the repaired model, preregistered to land
         within REPAIR_TOLERANCE_FE of the measured value; and the same circuits' day-over-day drift against R145
Modes: freeze --live B | --fake (read-only) . submit RECORD (OWNER ACT) . fetch RECORD . dryrun RECORD (fake twin, Aer)
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_superact_run as HS  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import cz_channel as CC  # noqa: E402
from oneq.hw import device as DV  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import sensitivity as SE  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402
from oneq.hw import submit as U  # noqa: E402
from oneq.hw import value as V  # noqa: E402

R145_DIR = ROOT / "evidence" / "superactivation" / "hw_run_m5"
R145_RECORD = R145_DIR / "frozen_20261015T195926Z.json"
R145_RESULT = R145_DIR / "result_db8jpre94fic73etigpg.json"
OUTDIR = ROOT / "evidence" / "superactivation" / "hw_diag_r146"
PREREG = ROOT / "docs" / "PREREG_hw_diag_r146.md"
SHOTS_DIAG = 1000                    # as first preregistered (every calibration circuit), the baseline amendment 1
SHOTS_P0 = 200                       # was priced against (tools/hw_diag_allocation.py); a record frozen before it
RANDOMIZATIONS = 8
# AMENDMENT 1 (2026-10-16, before any device data; docs/PREREG_hw_diag_r146.md): shots per circuit by DECISION VALUE
# (tools/hw_diag_allocation.py, oneq.hw.value). V1's uncertainty was 51% the held-out circuits' shot noise and 40%
# readout; the ladders carried < 6%. Option C of evidence/superactivation/hw_diag_r146/allocation.json: every ladder
# kept at 1,000 (the per-edge CZ channel is a product of this job, not only V1's input) and the rest bought by decision
# value: V1's sd 0.00397 -> 0.00303 for 37.6 -> 45.5 s of per-shot time. Multiples of RANDOMIZATIONS. tm and cp decide
# V5 and V2 and keep their shots.
SHOTS_BY_FAMILY = dict(ro=3672, xs=1120, sk=1112, pb=1000, tm=1000, cp=1000)
SHOTS_P0_AMENDED = 264
# DEEP ladders: a decay RATE is measured over a long ladder, and a deeper circuit costs almost no QPU time (a shot is
# mostly the repetition delay). Simulated at 1,000 shots: 0/4/8 left each learnable quantity 23-42% uncertain, 0/16/48
# leaves 7-8.5%. (pauli_ladder design, R146.)
LADDER_DEPTHS = (0, 16, 48)
SX_COUNTS = (64, 256)                # multiples of 4: sx^4 = I; xs_w0 is the zero point
PB_DRAWS = 400                       # draws of the slopes per edge, for the channel's total-rate uncertainty
N_EDGES = 10
TIMING_SX = 400                      # a multiple of 4: sx^400 = I, so q47 ends where it began
TIMING_PAIR = (57, 47)               # (left alone in |1>, busy): the copy qubit and its busiest neighbour
COPY_PAIR = (47, 57)                 # (control, target) of the copy path R145 localized the excess on
REPAIR_TOLERANCE_FE = 0.01           # preregistered: |measured - repaired prediction| F_e on the held-out p0 circuits
MODEL_TOLERANCE_VENDOR = 0.03        # R145's own tolerance: V3 asks whether the vendor model fails it again
BOOT = 8                             # parametric draws of the fit, for the repaired prediction's error bar
HELD_OUT = ("cp_", "p0_")
SCHEDULE_MARGIN = 9.0                # chi2 gap the winning schedule must clear (3 sd) to be chosen at all
GRANULARITY = 16                     # IBM Heron delays are multiples of 16 dt


# ------------------------------------------------------------------ circuits (logical; placed by place())
def ro_circuits(n: int) -> list:
    from qiskit import QuantumCircuit
    pats = {"all0": [0] * n, "all1": [1] * n, "alt01": [i % 2 for i in range(n)], "alt10": [1 - i % 2 for i in range(n)]}
    out = []
    for tag, bits in pats.items():
        qc = QuantumCircuit(n, n, name=f"ro_{tag}")
        for i, b in enumerate(bits):
            if b:
                qc.x(i)
        qc.barrier()
        qc.measure(range(n), range(n))
        out.append((qc, dict(family="ro", prepared=bits)))
    return out


def xs_circuits(n: int, wait_dt: int) -> list:
    from qiskit import QuantumCircuit
    out = []
    for tag, wait in (("w0", 0), ("w", wait_dt)):
        qc = QuantumCircuit(n, n, name=f"xs_{tag}")
        qc.h(range(n))
        qc.barrier()
        if wait:
            qc.delay(wait, range(n), unit="dt")
            qc.barrier()
        qc.h(range(n))
        qc.measure(range(n), range(n))
        out.append((qc, dict(family="xs", wait_dt=wait)))
    return out


def ladder_rounds(edge_use: list, k: int = N_EDGES) -> list:
    """the k busiest edges, greedily packed into rounds of vertex-disjoint edges (busiest first)"""
    top = [tuple(sorted(e[:2])) for e in edge_use[:k]]
    rounds: list = []
    for e in top:
        for r in rounds:
            if all(not set(e) & set(f) for f in r):
                r.append(e)
                break
        else:
            rounds.append([e])
    return rounds


def pb_circuits(r: int, edges: list) -> list:
    """the round's product-basis ladders: for each of the 7 bases (oneq.hw.cz_channel.BASES) and each depth, every
    edge's two qubits in the +1 eigenstate of their basis letters (the first letter on the edge's lower-numbered
    qubit), `d` CZs on every edge, read back in the same basis. One circuit gives a(x)I, I(x)b and a(x)b per edge."""
    from qiskit import QuantumCircuit
    out = []
    n = 2 * len(edges)
    for basis in CC.BASES:
        for d in LADDER_DEPTHS:
            qc = QuantumCircuit(n, n, name=f"pb{r}_{basis}_d{d}")
            for q in range(n):
                letter = basis[q % 2]
                if letter in "XY":
                    qc.h(q)
                if letter == "Y":
                    qc.s(q)
            qc.barrier()
            for k in range(d):
                for i in range(len(edges)):
                    qc.cz(2 * i, 2 * i + 1)
                if k < d - 1:
                    qc.barrier()
            qc.barrier()
            for q in range(n):
                letter = basis[q % 2]
                if letter == "Y":
                    qc.sdg(q)
                if letter in "XY":
                    qc.h(q)
            qc.measure(range(n), range(n))
            out.append((qc, dict(family="pb", round=r, edges=[list(e) for e in edges], depth=d, basis=basis)))
    return out


def sk_circuits(n: int) -> list:
    """all n in |+>, then k sx (k a multiple of 4: the identity), rotated back: an sx error of probability p flips the
    X contrast with probability 2p/3 (its Y and Z parts), so the contrast decays by 1 - 4p/3 per sx"""
    from qiskit import QuantumCircuit
    out = []
    for k in SX_COUNTS:
        qc = QuantumCircuit(n, n, name=f"sk_n{k}")
        qc.h(range(n))
        qc.barrier()
        for _ in range(k):
            qc.sx(range(n))
        qc.barrier()
        qc.h(range(n))
        qc.measure(range(n), range(n))
        out.append((qc, dict(family="sk", n_sx=k)))
    return out


def tm_circuits() -> list:
    """three ways q57 can wait in |1> while q47 runs 400 sx (14 us):
      free     x(57); barrier; ...; measure both           -- decays only if terminal READOUT is aligned at the end
      aligned  the same with a barrier before the readout  -- decays always (the control)
      gate     x(57) with NO barrier, then cz(57, 47) after q47's run, then readout -- decays only if GATES run as
               soon as possible (ALAP would play the x just before the cz)"""
    from qiskit import QuantumCircuit
    out = []
    for tag in ("free", "aligned", "gate"):
        qc = QuantumCircuit(2, 2, name=f"tm_{tag}")
        qc.x(0)
        if tag != "gate":
            qc.barrier()
        for _ in range(TIMING_SX):
            qc.sx(1)
            qc.barrier(1)
        if tag == "aligned":
            qc.barrier()
        if tag == "gate":
            qc.cz(0, 1)                      # q47 is back in |0>: the cz acts as the identity on the populations
        qc.measure([0, 1], [0, 1])
        out.append((qc, dict(family="tm", variant=tag)))
    return out


def cp_circuits() -> list:
    from qiskit import QuantumCircuit
    out = []
    for v in (0, 1):
        qc = QuantumCircuit(2, 2, name=f"cp_{v}")
        if v:
            qc.x(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        out.append((qc, dict(family="cp", control_value=v)))
    return out


def place(qc, be, phys: list):
    """level 0: the diagnostic's barriers, delays and back-to-back gates are the experiment, and must not be optimized
    away; every two-qubit gate is on a device edge, so nothing is routed"""
    from qiskit import transpile
    t = transpile(qc, be, initial_layout=list(phys), optimization_level=0)
    t.name = qc.name
    if t.count_ops().get("swap", 0):
        raise SystemExit(f"{qc.name}: placement {phys} needed routing")
    return t


# ------------------------------------------------------------------ the model's view of each observable
def marginal(i: int, v: int = 1):
    return lambda bits: 1.0 if bits[i] == v else 0.0


def parity(i: int, j: int):
    return lambda bits: 1.0 if bits[i] == bits[j] else -1.0


def predict_diag(tqc, meta: dict, snap: S.DeviceSnapshot, params: N.NoiseParams) -> dict:
    """the model's prediction for every observable the analysis reads off this circuit (exact: all are Clifford)"""
    rng = np.random.default_rng(0)
    n = tqc.num_clbits

    def ex(f):
        return E.expectation(tqc, f, snap, params, 64, rng).mean

    if meta["family"] in ("ro", "xs", "sk", "tm", "cp"):
        return dict(p1=[ex(marginal(i)) for i in range(n)])
    if meta["family"] == "pb":
        return dict(parity=[ex(parity(2 * i, 2 * i + 1)) for i in range(n // 2)],
                    p1=[ex(marginal(i)) for i in range(n)])
    return dict(parity=[ex(parity(2 * i, 2 * i + 1)) for i in range(n // 2)])


# ------------------------------------------------------------------ freeze
def wait_dt_from(r145: dict, circ: dict, snap: S.DeviceSnapshot) -> int:
    """the protocol's own median gate window -- its critical path WITHOUT the final readout, which duration_s counts --
    rounded to the delay granularity"""
    durs = [N.duration_s(circ[u["name"]].remove_final_measurements(inplace=False), snap)
            for u in r145["units"] if u["unit"]["branch"] != "erased_erased"]
    return int(round(statistics.median(durs) / snap.dt / GRANULARITY)) * GRANULARITY


def freeze_diag(be, snap: S.DeviceSnapshot, prereg_sha: str, check_health: bool = True, p0_limit: int | None = None,
                traj: int = HS.TRAJ, vendor_families: set | None = None) -> tuple:
    """p0_limit, traj and vendor_families exist for the tests (a spread subset of R145's circuits, fewer trajectories,
    vendor predictions only for the families the analysis reads); a real freeze
    takes every circuit at the R145 trajectory count"""
    r145 = json.loads(R145_RECORD.read_text(encoding="utf-8"))
    rcirc = F.load(r145, R145_RECORD.with_suffix(".qpy").read_bytes(), check_sources=False)
    qubits = list(r145["layout"])
    wait = wait_dt_from(r145, rcirc, snap)
    rounds = ladder_rounds(r145["edge_use"])
    built = []
    for qc, meta in ro_circuits(len(qubits)) + xs_circuits(len(qubits), wait) + sk_circuits(len(qubits)):
        built.append((place(qc, be, qubits), dict(meta, physical=qubits)))
    for r, edges in enumerate(rounds):
        phys = [q for e in edges for q in e]
        for qc, meta in pb_circuits(r, edges):
            built.append((place(qc, be, phys), dict(meta, physical=phys)))
    for qc, meta in tm_circuits():
        built.append((place(qc, be, list(TIMING_PAIR)), dict(meta, physical=list(TIMING_PAIR))))
    for qc, meta in cp_circuits():
        built.append((place(qc, be, list(COPY_PAIR)), dict(meta, physical=list(COPY_PAIR))))
    policy = L.QubitPolicy()
    named, diag = [], []
    params = N.NoiseParams()
    for tqc, meta in built:
        why = L.refusals(snap, tqc, policy) if check_health else ()
        if why:
            raise SystemExit(f"{tqc.name} is outside the policy on {snap.name} now: {why}")
        diag.append(dict(name=tqc.name, meta=meta, n_bits=tqc.num_clbits,
                         vendor_prediction=(predict_diag(tqc, meta, snap, params)
                                            if vendor_families is None or meta["family"] in vendor_families else None)))
        named.append((tqc.name, tqc))
    # the held-out protocol circuits: R145's ISA circuits byte for byte, renamed
    p0_units = []
    rng = np.random.default_rng(7)
    chosen = r145["units"] if p0_limit is None else r145["units"][:: max(1, len(r145["units"]) // p0_limit)][:p0_limit]
    for u in chosen:
        qc = rcirc[u["name"]].copy(name="p0_" + u["name"])
        why = L.refusals(snap, qc, policy) if check_health else ()
        if why:
            raise SystemExit(f"{qc.name} (R145's circuit) is outside the policy on {snap.name} now: {why}")
        r = E.expectation(qc, H.score(r145["m"], HS.unit_from(u["unit"])), snap, params, traj, rng)
        p0_units.append(dict(u, name=qc.name, r145_name=u["name"], predicted=r.mean, predicted_se=r.se,
                             shots=SHOTS_P0_AMENDED))
        named.append((qc.name, qc))
    meta = dict(protocol="R146 in-circuit calibration + held-out re-run of R145", backend=snap.name,
                snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
                layout=qubits, wait_dt=wait, ladder_rounds=[[list(e) for e in r] for r in rounds],
                diag=diag, units=p0_units, m=r145["m"], shots=SHOTS_P0_AMENDED,
                shots_diag=dict(SHOTS_BY_FAMILY),
                randomizations=RANDOMIZATIONS, noise=params.describe(),
                vendor_prediction=HS.estimate({u["name"]: u["predicted"] for u in p0_units}, p0_units),
                r145_record=R145_RECORD.name, r145_record_sha256=U.file_sha256(R145_RECORD),
                repair_tolerance_F_e=REPAIR_TOLERANCE_FE, prereg=PREREG.name, prereg_sha256=prereg_sha,
                sources=[str(p) for p in [HERE / "hw_diag_run.py", HERE / "hw_superact_run.py", HERE / "hw_two_use.py"]
                         + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))])
    return F.freeze(named, meta)


BLOCK_FAMILIES = ("ro", "xs", "sk", "pb")


def block_circuits(be, snap: S.DeviceSnapshot, qubits: list, edges: list, wait_dt: int, check_health: bool = True,
                   vendor_families: frozenset = frozenset()) -> tuple:
    """R146's device families pointed at ANOTHER experiment's qubits and edges (R152): ro, xs, sk on `qubits` at once,
    and the product-basis ladders on EVERY edge of `edges`, packed into vertex-disjoint rounds -- so the fitted model
    carries a MEASURED CZ channel for each one (repaired_snapshot gives an unladdered edge the median shape, which a
    coverage check would mistake for a measurement). No timing or copy family: those are R145-specific, and the
    schedule is the device's, read once off R146's tm family in the same job. Returns (named, meta) in R146's record
    layout, so fit_device and repaired_snapshot read the block's record unchanged."""
    qubits = [int(q) for q in qubits]
    if len(set(qubits)) != len(qubits):
        raise SystemExit("REFUSED: the block's qubit list repeats a qubit")
    es = sorted({tuple(sorted((int(a), int(b)))) for a, b in edges})
    outside = [e for e in es if not set(e) <= set(qubits)]
    if outside:
        raise SystemExit(f"REFUSED: edges {outside[:4]} touch qubits outside the block, whose readout the ladder fit "
                         f"needs")
    for a, b in es:
        snap.edge(a, b)                                  # SnapshotError names an edge the device does not have
    rounds = ladder_rounds([list(e) for e in es], k=len(es))
    built = []
    for qc, meta in ro_circuits(len(qubits)) + xs_circuits(len(qubits), wait_dt) + sk_circuits(len(qubits)):
        built.append((place(qc, be, qubits), dict(meta, physical=qubits)))
    for r, round_edges in enumerate(rounds):
        phys = [q for e in round_edges for q in e]
        for qc, meta in pb_circuits(r, round_edges):
            built.append((place(qc, be, phys), dict(meta, physical=phys)))
    policy = L.QubitPolicy()
    params = N.NoiseParams()
    named, diag = [], []
    for tqc, meta in built:
        why = L.refusals(snap, tqc, policy) if check_health else ()
        if why:
            raise SystemExit(f"{tqc.name} is outside the policy on {snap.name} now: {why}")
        diag.append(dict(name=tqc.name, meta=meta, n_bits=tqc.num_clbits,
                         vendor_prediction=(predict_diag(tqc, meta, snap, params)
                                            if meta["family"] in vendor_families else None)))
        named.append((tqc.name, tqc))
    meta = dict(protocol="R146 device families on another experiment's qubits and edges (calibration block)",
                backend=snap.name, snapshot=snap.to_json(), snapshot_sha256=snap.sha256(),
                calibrated_utc=snap.calibrated_utc, layout=qubits, edges=[list(e) for e in es], wait_dt=int(wait_dt),
                ladder_rounds=[[list(e) for e in r] for r in rounds], diag=diag, units=[],
                shots_diag={f: SHOTS_BY_FAMILY[f] for f in BLOCK_FAMILIES}, randomizations=RANDOMIZATIONS,
                noise=params.describe())
    return named, meta


def measured_edges(fit: dict) -> set:
    """the edges a fit carries a LADDERED (measured) CZ channel for, as sorted pairs"""
    return {tuple(sorted(map(int, k.split("_")))) for k in fit["ladders"]}


def shots_of(rec: dict) -> dict:
    """shots per circuit. A record's shots_diag is one number for every calibration circuit (frozen before amendment
    1) or {family: shots} (amendment 1); a family the dict does not name is refused, never defaulted."""
    sd = rec["shots_diag"]

    def diag_shots(d):
        if isinstance(sd, int):
            return sd
        fam = d["meta"]["family"]
        if fam not in sd:
            raise SystemExit(f"{d['name']}: the record allocates no shots to family {fam!r}")
        return sd[fam]
    return {d["name"]: diag_shots(d) for d in rec["diag"]} | {u["name"]: u["shots"] for u in rec["units"]}


# ------------------------------------------------------------------ the fit (pure; never sees a held-out count)
def _p1(counts: dict, n_bits: int) -> list:
    tot = sum(int(c) for c in counts.values())
    ones = [0] * n_bits
    for k, c in counts.items():
        b = T.bits_of(k, n_bits)
        for i in range(n_bits):
            ones[i] += int(c) * b[i]
    return [x / tot for x in ones], tot


def _parity(counts: dict, n_bits: int, i: int, j: int) -> float:
    tot = sum(int(c) for c in counts.values())
    s = 0
    for k, c in counts.items():
        b = T.bits_of(k, n_bits)
        s += int(c) * (1 if b[i] == b[j] else -1)
    return s / tot


def fit_in_circuit(rec: dict, counts: dict, circ: dict) -> dict:
    """the repaired model's parameters from the FIT families alone. Refuses a count dict that holds a held-out name:
    the separation is enforced here, not trusted to the caller."""
    leak = sorted(n for n in counts if n.startswith(HELD_OUT))
    if leak:
        raise ValueError(f"the fit was handed held-out counts ({leak[:3]}...): the test would not be held out")
    dev = fit_device(rec, counts)
    return dict(dev, timing=fit_timing(rec, counts, circ, dev))


def fit_device(rec: dict, counts: dict) -> dict:
    """readout, sx, the X-basis effective T2 and every laddered edge's CZ channel: the families every calibration
    block carries -- R146's own, and a block on another experiment's qubits (block_circuits, R152). The timing family
    is R145-specific and stays in fit_in_circuit."""
    diag = {d["name"]: d for d in rec["diag"]}
    qubits = rec["layout"]
    n = len(qubits)
    # readout: each qubit is prepared 0 in two patterns and 1 in two
    n0 = [0.0] * n
    k0 = [0.0] * n
    n1 = [0.0] * n
    k1 = [0.0] * n
    for name in ("ro_all0", "ro_all1", "ro_alt01", "ro_alt10"):
        p1, tot = _p1(counts[name], n)
        for i, v in enumerate(diag[name]["meta"]["prepared"]):
            if v:
                k1[i] += (1 - p1[i]) * tot
                n1[i] += tot
            else:
                k0[i] += p1[i] * tot
                n0[i] += tot
    raw = {q: dict(p01=k0[i] / n0[i], p10=k1[i] / n1[i], shots0=n0[i], shots1=n1[i]) for i, q in enumerate(qubits)}
    # the |1> is PREPARED by an x gate, whose error (its X and Y parts, 2/3 of p_x) flips it: raw p10 carries that
    # gate's error, which the engine charges again on every x. The sx rate comes from a decay slope, which readout
    # cannot bias at first order, so: sx on the raw readout, then the readout corrected for the preparation, then sx
    # once more on the corrected readout
    sx = fit_sx(rec, counts, raw)
    readout = {}
    for q, r in raw.items():
        e = 2 * sx[q]["p_sx"] / 3
        readout[q] = dict(r, p10=max(0.0, (r["p10"] - e * (1 - r["p01"])) / (1 - e)), p10_raw=r["p10"])
    sx = fit_sx(rec, counts, readout)
    # X-basis: |+> -> H -> |0>; the contrast is 1 - 2 P(1), corrected for the measured readout by the exact affine
    # inverse (a qubit reading p01 / p10 maps contrast c to (1 - p01 - p10) c + (p10 - p01))
    w0, n0 = _p1(counts["xs_w0"], n)
    ww, nw = _p1(counts["xs_w"], n)
    t_wait = rec["wait_dt"] * rec["snapshot"]["dt"]
    xs = {}
    for i, q in enumerate(qubits):
        c0, cw = _readout_inverse(1 - 2 * w0[i], readout[q]), _readout_inverse(1 - 2 * ww[i], readout[q])
        ratio = cw / c0 if c0 > 0 else float("nan")
        t2 = -t_wait / math.log(ratio) if 0 < ratio < 1 else float("inf")
        # its sd from the two contrasts' shot noise (raw values, through the readout inverse): var ln(ratio) =
        # var(cw)/cw^2 + var(c0)/c0^2, and t2 = -t / ln(ratio) moves by t / ln(ratio)^2 per unit of ln(ratio)
        scale = 1 - readout[q]["p01"] - readout[q]["p10"]
        t2_sd = float("inf")
        if math.isfinite(t2) and cw > 0:
            v_log = (_corr_var(1 - 2 * ww[i], nw, scale) / cw ** 2 + _corr_var(1 - 2 * w0[i], n0, scale) / c0 ** 2)
            t2_sd = t_wait / math.log(ratio) ** 2 * math.sqrt(v_log)
        xs[q] = dict(contrast_w0=c0, contrast_w=cw, ratio=ratio, t2_eff_s=t2, t2_sd=t2_sd)
    ladders = fit_cz_channels(rec, counts, readout)
    return dict(readout=readout, xs=xs, sx=sx, ladders=ladders)


def fit_sx(rec: dict, counts: dict, readout: dict) -> dict:
    """per qubit, the in-circuit sx error from the X contrast's DECAY with the number of sx (xs_w0 is n = 0; sk_n32,
    sk_n64), intercept free: each sx multiplies the contrast by 1 - 4 p_sx / 3 (its Y and Z parts flip it)"""
    qubits = rec["layout"]
    n = len(qubits)
    w0, n_w0 = _p1(counts["xs_w0"], n)
    sk = {k: _p1(counts[f"sk_n{k}"], n) for k in SX_COUNTS}
    out = {}
    for i, q in enumerate(qubits):
        ro = readout[q]
        s = 1 - ro["p01"] - ro["p10"]
        raw = [(0, 1 - 2 * w0[i], n_w0)] + [(k, 1 - 2 * sk[k][0][i], sk[k][1]) for k in SX_COUNTS]
        pts = [(k, _readout_inverse(m, ro), _corr_var(m, nn, s)) for k, m, nn in raw]
        slope, sd = _log_slope(pts)
        m = math.exp(slope)
        out[q] = dict(points=[[k, c] for k, c, _ in pts], slope=slope, slope_sd=sd, p_sx=max(0.0, 3 * (1 - m) / 4),
                      p_sx_sd=3 * m * sd / 4, vendor_p_sx=1.5 * rec["snapshot"]["sx_error"][q])
    return out


def _log_slope(points: list) -> tuple:
    """(slope, sd) of ln c against depth by weighted least squares with a FREE intercept, from (depth, c, var_c):
    var(ln c) = var_c / c^2. A free intercept is what makes a rate immune to SPAM.
    var_c must come from the RAW measured correlator, propagated through the readout inverse (_corr_var): a corrected
    c can exceed 1 by a readout estimate's own noise, and the dry run (R146) found the old (1 - c^2) weight, clipped at
    c = 1, gave such a point near-infinite weight and four edges a fitted rate of zero."""
    x = np.array([p[0] for p in points], dtype=float)
    c = np.clip(np.array([p[1] for p in points], dtype=float), 1e-6, None)
    w = c ** 2 / np.clip(np.array([p[2] for p in points], dtype=float), 1e-12, None)
    y = np.log(c)
    W = w.sum()
    xm, ym = (w * x).sum() / W, (w * y).sum() / W
    sxx = (w * (x - xm) ** 2).sum()
    return float((w * (x - xm) * (y - ym)).sum() / sxx), float(math.sqrt(1 / sxx))


def _corr_var(m: float, shots: float, scale: float) -> float:
    """variance of a readout-corrected correlator c = (m - offset) / scale from its RAW measured value m (|m| <= 1
    always): var(m) = (1 - m^2) / shots, floored at one shot's worth so a perfect count is not infinitely sure"""
    return max(1 - m * m, 1.0 / shots) / shots / (scale * scale)


def _readout_inverse(m1: float, ro: dict) -> float:
    """the true single-qubit correlator from a measured one: readout maps z to (1 - p01 - p10) z + (p10 - p01)"""
    return (m1 - (ro["p10"] - ro["p01"])) / (1 - ro["p01"] - ro["p10"])


def _pair_inverse(mij: float, ci: float, cj: float, ri: dict, rj: dict) -> float:
    """the true two-qubit correlator from a measured one, independent asymmetric readout on each qubit:
    m_ij = s_i s_j c_ij + s_i o_j c_i + o_i s_j c_j + o_i o_j"""
    si, oi = 1 - ri["p01"] - ri["p10"], ri["p10"] - ri["p01"]
    sj, oj = 1 - rj["p01"] - rj["p10"], rj["p10"] - rj["p01"]
    return (mij - si * oj * ci - oi * sj * cj - oi * oj) / (si * sj)


def fit_cz_channels(rec: dict, counts: dict, readout: dict) -> dict:
    """per laddered edge, the CZ's FULL learnable Pauli channel (oneq.hw.cz_channel) from the product-basis ladders.

    Each (observable, basis) series is readout-corrected and fitted for its own decay RATE with a FREE intercept: every
    basis has its own preparation and readout gates, so its SPAM is its own, and no SPAM can leak into a rate. A
    correlator P decays per layer by ln(f_P f_G(P)) / 2, so every series that reads the same learnable quantity (a
    CZ-fixed Pauli, or a CZ pair) is pooled by inverse variance. The 9 quantities give the channel in the symmetric
    gauge; its total rate p and its SHAPE (15 fractions) go to the snapshot. p's uncertainty comes from PB_DRAWS draws
    of the pooled slopes. A slope set inconsistent across bases shows in chi2 (dof = series - quantities)."""
    out = {}
    rng = np.random.default_rng(146)
    for r, edges in enumerate(rec["ladder_rounds"]):
        nb = 2 * len(edges)
        for i, (a, b) in enumerate(edges):
            ra, rb = readout[a], readout[b]
            sa_, sb_ = 1 - ra["p01"] - ra["p10"], 1 - rb["p01"] - rb["p10"]
            series: dict = {}
            for basis in CC.BASES:
                for d in LADDER_DEPTHS:
                    cnt = counts[f"pb{r}_{basis}_d{d}"]
                    p1, tot = _p1(cnt, nb)
                    mi, mj, mij = 1 - 2 * p1[2 * i], 1 - 2 * p1[2 * i + 1], _parity(cnt, nb, 2 * i, 2 * i + 1)
                    ci, cj = _readout_inverse(mi, ra), _readout_inverse(mj, rb)
                    cij = _pair_inverse(mij, ci, cj, ra, rb)
                    for P, c, v in zip(CC.observables(basis), (ci, cj, cij),
                                       (_corr_var(mi, tot, sa_), _corr_var(mj, tot, sb_), _corr_var(mij, tot, sa_ * sb_))):
                        series.setdefault((P, basis), []).append((d, c, v))
            per_key: dict = {}
            fitted = {}
            for (P, basis), pts in series.items():
                slope, sd = _log_slope(pts)
                fitted[f"{P}@{basis}"] = dict(slope=slope, sd=sd, points=[[d, c] for d, c, _ in pts])
                per_key.setdefault(CC.learnable_key(P), []).append((slope, sd))
            keys = list(per_key)
            pooled = {}
            chi2 = 0.0
            for k in keys:
                w = np.array([1 / max(sd, 1e-9) ** 2 for _, sd in per_key[k]])
                s = np.array([sl for sl, _ in per_key[k]])
                m = float((w * s).sum() / w.sum())
                pooled[k] = (m, float(1 / math.sqrt(w.sum())))
                chi2 += float((w * (s - m) ** 2).sum())

            def channel(slopes: dict) -> dict:
                # per layer a correlator falls by sqrt(f_P f_G(P)) (f_P for a CZ-fixed P), so the per-two-layer
                # quantity cz_channel takes is exp(2 x the per-layer log slope), for pairs and fixed Paulis alike
                g = {k: math.exp(2 * sl) for k, sl in slopes.items()}
                return CC.project_physical(CC.symmetric_channel(g))
            ch = channel({k: v[0] for k, v in pooled.items()})
            p = 1 - ch["II"]
            draws = []
            for _ in range(PB_DRAWS):
                chd = channel({k: v[0] + rng.normal(0, v[1]) for k, v in pooled.items()})
                draws.append(1 - chd["II"])
            raw = CC.symmetric_channel({k: math.exp(2 * v[0]) for k, v in pooled.items()})
            clipped = float(sum(-v for kk, v in raw.items() if kk != "II" and v < 0))
            vendor = rec["snapshot"]["cz"][f"{a}_{b}"][0]
            shape = tuple(ch[q] / p for q in CC.NONI) if p > 0 else tuple(1 / 15 for _ in CC.NONI)
            out[f"{a}_{b}"] = dict(p=p, p_sd=float(np.std(draws, ddof=1)), shape=shape,
                                   channel={q: ch[q] for q in CC.NONI}, clipped_mass=clipped,
                                   learnable={str(k): math.exp(2 * v[0]) for k, v in pooled.items()},
                                   # g = exp(2 slope): its sd from the pooled slope's, to first order (delta method)
                                   learnable_sd={str(k): 2 * math.exp(2 * v[0]) * v[1] for k, v in pooled.items()},
                                   chi2=chi2, dof=len(series) - len(keys), r_equiv=p / 1.25, vendor_r=vendor,
                                   factor=p / (1.25 * vendor), series=fitted)
    return out


def fit_timing(rec: dict, counts: dict, circ: dict, partial: dict) -> dict:
    """which schedule the device runs. q57 waits in |1> through q47's 400 sx in 'aligned' under EVERY schedule, so that
    circuit measures q57's in-circuit T1 over the window; each candidate schedule then predicts all three circuits on
    the snapshot with everything fitted before it (readout, sx, T2, the CZ channels -- `partial`, at cz_factor 1) and
    that T1, and the device's schedule is the one with the least chi2. Which circuits decay is the discriminant, so an
    error in the T1 scale moves every candidate together. (The gate-timed circuit ends in a CZ on 47-57: predicting it
    on the VENDOR channel let an X/Y-type CZ error on q57 read as decay and pick the wrong schedule -- the closed loop
    found it when the synthetic channels became full.)"""
    q57 = TIMING_PAIR[0]
    readout = partial["readout"]
    p = {}
    n = {}
    for v in ("free", "aligned", "gate"):
        p1, tot = _p1(counts[f"tm_{v}"], 2)
        p[v], n[v] = p1[0], tot
    ro = readout[q57]
    snap0 = S.DeviceSnapshot.from_json(rec["snapshot"])
    t_busy = TIMING_SX * snap0.sx_duration[TIMING_PAIR[1]]
    surv = (p["aligned"] - ro["p01"]) / (1 - ro["p01"] - ro["p10"])
    t1_eff = -t_busy / math.log(surv) if 0 < surv < 1 else float("inf")
    d = repaired_snapshot(rec, partial).to_json()
    # the barrier control IS q57's in-circuit T1 over the window. If it shows no decay, a wait costs nothing
    # measurable and no schedule can be told from another: every candidate must then predict no decay (T1 effectively
    # infinite), so they tie -- keeping the vendor T1 there would invent decays the counts never showed and let the
    # variance floor turn them into a confident choice
    d["t1"][q57] = t1_eff if math.isfinite(t1_eff) else 1.0
    d["t2"][q57] = min(d["t2"][q57], 2 * d["t1"][q57])
    snap = S.DeviceSnapshot.from_json(d)
    chi2 = {}
    preds = {}
    for s in N.SCHEDULES:
        pr = {v: E.expectation(circ[f"tm_{v}"], marginal(0), snap, N.NoiseParams(cz_factor=1.0, schedule=s), 64,
                               np.random.default_rng(0)).mean for v in p}
        preds[s] = pr
        chi2[s] = sum((p[v] - pr[v]) ** 2 / (max(p[v] * (1 - p[v]), 1e-4) / n[v]) for v in p)
    best = min(chi2, key=chi2.get)
    margin = min(c for s, c in chi2.items() if s != best) - chi2[best]
    # a choice the counts cannot make is not made: under SCHEDULE_MARGIN (3 sd in chi2) the schedule is UNDETERMINED
    # and the repaired model keeps the default (found by the dry run: Aer applies no idle noise without explicit
    # delays, nothing decays, every schedule ties -- and a bare argmin "chose" one)
    schedule = best if margin >= SCHEDULE_MARGIN else "undetermined"
    return dict(p1=p, t1_eff_s=t1_eff, predictions=preds, chi2=chi2, best=best, chi2_margin=margin, schedule=schedule)


def repaired_snapshot(rec: dict, fit: dict) -> S.DeviceSnapshot:
    """the vendor snapshot overlaid with what the fit measured:
      - per-qubit readout; per-qubit sx error (the engine's p = 1.5 x error); per-qubit effective T2 where resolved;
      - per laddered edge the CZ channel: r = p / 1.25 (so the repaired model runs at cz_factor = 1) and its full
        SHAPE, 15 fractions -- DeviceSnapshot.cz_structure;
      - edges the ladders did not cover: vendor r x the median measured factor, with the component-wise median shape
        (renormalized)."""
    d = json.loads(json.dumps(rec["snapshot"]))
    for q, ro in fit["readout"].items():
        d["p01"][int(q)] = ro["p01"]
        d["p10"][int(q)] = ro["p10"]
    for q, x in fit["xs"].items():
        t2 = x["t2_eff_s"]
        if math.isfinite(t2) and t2 > 0:
            d["t2"][int(q)] = min(t2, 2 * d["t1"][int(q)])
    for q, x in fit.get("sx", {}).items():
        d["sx_error"][int(q)] = x["p_sx"] / 1.5
    lads = fit["ladders"]
    if not lads:                                   # nothing laddered: the vendor CZ rates stand, uniform
        return S.DeviceSnapshot.from_json(d)
    med = statistics.median(lad["factor"] for lad in lads.values())
    med_shape = np.median(np.array([lad["shape"] for lad in lads.values()], dtype=float), axis=0)
    med_shape = (med_shape / med_shape.sum()).tolist()
    structure = {}
    for k, (r, dur) in d["cz"].items():
        if k in lads:
            d["cz"][k] = [lads[k]["r_equiv"], dur]
            structure[k] = [float(x) for x in lads[k]["shape"]]
        else:
            d["cz"][k] = [r * med, dur]
            structure[k] = med_shape
    d["cz_structure"] = structure
    return S.DeviceSnapshot.from_json(d)


def repaired_params(fit: dict) -> N.NoiseParams:
    s = fit["timing"]["schedule"]
    return N.NoiseParams(cz_factor=1.0, schedule="alap" if s == "undetermined" else s)


# ------------------------------------------------------------------ analysis
def analyse(rec: dict, counts: dict, blob: bytes, traj: int = HS.TRAJ, boot: int = BOOT) -> dict:
    fit_counts = {k: v for k, v in counts.items() if not k.startswith(HELD_OUT)}
    circ = F.load(rec, blob, check_sources=False)
    fit = fit_in_circuit(rec, fit_counts, circ)
    snap_r = repaired_snapshot(rec, fit)
    params_r = repaired_params(fit)
    draws = [perturbed_fit(fit, np.random.default_rng(1000 + b)) for b in range(boot)]
    snaps_b = [repaired_snapshot(rec, f) for f in draws]
    # held-out 1: the copy primitive. Its prediction rests on ONE fitted edge (47-57) and two qubits, so its error bar
    # carries their uncertainty (the parametric draws), not only the shot noise of the measurement
    cp = {}
    for dd in rec["diag"]:
        if dd["meta"]["family"] != "cp":
            continue
        v = dd["meta"]["control_value"]
        p1, tot = _p1(counts[dd["name"]], 2)
        flip_meas = p1[1] if v == 0 else 1 - p1[1]

        def flip(s, dd=dd, v=v):
            pr = predict_diag(circ[dd["name"]], dd["meta"], s, params_r)["p1"][1]
            return pr if v == 0 else 1 - pr
        flip_pred = flip(snap_r)
        flips_b = [flip(s) for s in snaps_b]
        flip_vendor = (dd["vendor_prediction"]["p1"][1] if v == 0 else 1 - dd["vendor_prediction"]["p1"][1])
        sd_meas = math.sqrt(max(flip_meas * (1 - flip_meas), 1e-4) / tot)
        sd_pred = float(np.std(flips_b, ddof=1)) if len(flips_b) > 1 else 0.0
        cp[str(v)] = dict(flip_measured=flip_meas, flip_repaired=flip_pred, flip_vendor=flip_vendor, sd=sd_meas,
                          sd_repaired=sd_pred, z=(flip_meas - flip_pred) / math.hypot(sd_meas, sd_pred))
    # held-out 2: R145's circuits under the repaired model, and the spread of that prediction over the parametric draws
    # (common random numbers: every draw reuses the same trajectory stream, so the spread is the parameters')
    units = rec["units"]

    def predict_p0(s, t):
        rng = np.random.default_rng(11)
        return {u["name"]: E.expectation(circ[u["name"]], H.score(rec["m"], HS.unit_from(u["unit"])), s, params_r,
                                         t, rng).mean for u in units}
    rep = predict_p0(snap_r, traj)
    traj_b = max(1, traj // 4)
    fe_b = [HS.estimate(predict_p0(s, traj_b), units)["F_e"] for s in snaps_b]
    param_sd = float(np.std(fe_b, ddof=1)) if len(fe_b) > 1 else 0.0
    # the same spread by LINEAR RESPONSE (R151): every p0 circuit's derivative in every error location (exact for the
    # reachable part, first order for the blocked), pushed onto the fit's posterior coordinates and their sds.
    # Deterministic -- no draws, no trajectories. Reported beside the draws, which stay as a nonlinearity alarm.
    post = DV.from_diag_fit(rec, fit, "hw_diag_run.fit_in_circuit", "analysis")
    comp: dict = {}
    for u in units:
        sens = SE.sensitivity(circ[u["name"]], H.score(rec["m"], HS.unit_from(u["unit"])), snap_r, params_r)
        for key, dv in sens.components.items():
            comp[key] = comp.get(key, 0.0) + 1.5 * u["weight"] * dv          # F_e = (3 F_avg - 1) / 2
    dfe = V.coordinate_sensitivity(comp, post)
    param_sd_linear = math.sqrt(sum((dfe.get(k, 0.0) * c.sd) ** 2 for k, c in post.coords.items()))
    stat_units = [T.Unit(u["name"], u["weight"], u["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    scores = {u["name"]: H.score(rec["m"], HS.unit_from(u["unit"])) for u in units}
    tallies = T.tally_all(stat_units, {u["name"]: counts[u["name"]] for u in units}, scores)
    means = {k: t.mean for k, t in tallies.items()}
    var = {k: max(t.s2 / t.n - t.mean ** 2, 0.0) for k, t in tallies.items()}
    meas = HS.estimate(means, units, var, {k: t.n for k, t in tallies.items()})
    est_r = HS.estimate(rep, units)
    est_v = HS.estimate({u["name"]: u["predicted"] for u in units}, units)
    delta_r = meas["F_e"] - est_r["F_e"]
    # day-over-day: the same circuits in R145 (paired by unit; only shot noise separates them)
    r145 = json.loads(R145_RESULT.read_text(encoding="utf-8"))
    old_units = [dict(u, name=u["r145_name"], shots=400) for u in units]
    old_stat = [T.Unit(u["name"], u["weight"], 400, u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in old_units]
    old_scores = {u["name"]: scores["p0_" + u["name"]] for u in old_units}
    old_t = T.tally_all(old_stat, {u["name"]: r145["counts"][u["name"]] for u in old_units}, old_scores)
    old = HS.estimate({k: t.mean for k, t in old_t.items()}, old_units,
                      {k: max(t.s2 / t.n - t.mean ** 2, 0.0) for k, t in old_t.items()}, {k: t.n for k, t in old_t.items()})
    drift_sd = math.hypot(meas["shot_sd_F_e"], old["shot_sd_F_e"])
    snap_v = S.DeviceSnapshot.from_json(rec["snapshot"])
    ro_ratio = float(np.mean([0.5 * (r["p01"] + r["p10"]) / snap_v.readout_error(int(q))
                              for q, r in fit["readout"].items()]))
    return dict(
        fit=fit, repaired_noise=params_r.describe(),
        held_out=dict(
            copy_primitive=cp,
            p0=dict(F_e_measured=meas["F_e"], shot_sd_F_e=meas["shot_sd_F_e"], F_e_repaired=est_r["F_e"],
                    F_e_repaired_param_sd=param_sd, param_draws=len(fe_b), param_draw_trajectories=traj_b,
                    F_e_repaired_param_sd_linear=param_sd_linear, posterior_sha256=post.sha256(),
                    F_e_vendor=est_v["F_e"], delta_repaired=delta_r, delta_vendor=meas["F_e"] - est_v["F_e"],
                    # amendment 1: V1's z with the deterministic linear-response sd; the draws' z stays as the alarm
                    z_repaired=delta_r / math.hypot(meas["shot_sd_F_e"], param_sd_linear),
                    z_repaired_draws=delta_r / math.hypot(meas["shot_sd_F_e"], param_sd),
                    branches_measured=HS.branch_f_e(means, units), branches_repaired=HS.branch_f_e(rep, units)),
            drift=dict(F_e_r145=old["F_e"], F_e_today=meas["F_e"], delta=meas["F_e"] - old["F_e"], sd=drift_sd)),
        readout_in_circuit_over_vendor=ro_ratio,
        verdicts=dict(
            V1_repaired_model_holds=bool(abs(delta_r) <= REPAIR_TOLERANCE_FE),
            V2_copy_primitive_holds=bool(all(abs(c["z"]) <= 3 for c in cp.values())),
            V3_vendor_model_fails=bool(abs(meas["F_e"] - est_v["F_e"]) > MODEL_TOLERANCE_VENDOR),
            V4_readout_worse_in_circuit=bool(ro_ratio > 1),
            V5_schedule=fit["timing"]["schedule"],
            device_drifted=bool(abs(meas["F_e"] - old["F_e"]) > 3 * drift_sd)))


def perturbed_fit(fit: dict, rng) -> dict:
    """one parametric draw of the fit: every measured rate moved within its own sd, kept physical (non-negative;
    readout within its binomial sd). Structure and T2 are held: they move the prediction less than the rates do."""
    f = json.loads(json.dumps(fit))
    for lad in f["ladders"].values():
        p = max(0.0, lad["p"] + rng.normal(0, lad["p_sd"]))
        lad.update(p=p, r_equiv=p / 1.25, factor=p / (1.25 * lad["vendor_r"]))
    for x in f["sx"].values():
        x["p_sx"] = max(0.0, x["p_sx"] + rng.normal(0, x["p_sx_sd"]))
    for ro in f["readout"].values():
        for k, n in (("p01", "shots0"), ("p10", "shots1")):
            ro[k] = float(np.clip(ro[k] + rng.normal(0, math.sqrt(max(ro[k] * (1 - ro[k]), 1e-6) / ro[n])), 0, 0.5))
    return f


# ------------------------------------------------------------------ modes
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["freeze", "submit", "fetch", "dryrun"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true", help="freeze against FakeKingston (for dryrun only)")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    ap.add_argument("--save-counts", help="dryrun: write the simulated counts here (outside the repository)")
    ap.add_argument("--load-counts", help="dryrun: analyse these saved counts instead of simulating")
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
        snap = S.from_backend(be)
        rec, blob = freeze_diag(be, snap, U.file_sha256(PREREG), check_health=not a.fake)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{'fake_' if a.fake else ''}{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        plan = make_plan(rec)
        print(f"frozen {rec['n_circuits']} circuits -> {rp.name}; {plan.total_shots} shots, ~{plan.cost_s():.1f} s; "
              f"vendor-model F_e for the held-out p0 {rec['vendor_prediction']['F_e']:.4f}")
        return 0
    rp = pathlib.Path(a.record)
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    plan = make_plan(rec)
    if a.mode == "dryrun":
        if a.load_counts:                    # re-analyse a rehearsal's own counts without re-simulating them
            counts = json.loads(pathlib.Path(a.load_counts).read_text(encoding="utf-8"))
        else:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
            # padded: the rehearsal must be a device whose idles are physical, or it cannot test whether the repair
            # closes
            counts = U.local_run(rec, blob, FakeKingston(), shots_of(rec), pad_alap=True)
        if a.save_counts:
            pathlib.Path(a.save_counts).write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
        out = analyse(rec, counts, blob)
        (OUTDIR / f"dryrun_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                                     newline="\n")
        print(json.dumps(out["verdicts"], indent=1), json.dumps(out["held_out"]["p0"], indent=1))
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
    jobs = sorted(OUTDIR.glob("diag_r146_*.job.json"))
    job = json.loads(jobs[-1].read_text(encoding="utf-8"))
    got = U.fetch(job, svc)
    if "counts" not in got:
        print(f"job {job['job_id']}: {got['status']}")
        return 4
    out = dict(job=job, execution_seconds=got["execution_seconds"], analysis=analyse(rec, got["counts"], blob),
               counts=got["counts"])
    dest = OUTDIR / f"result_{job['job_id']}.json"
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["analysis"]["verdicts"], indent=1), json.dumps(out["analysis"]["held_out"], indent=1))
    return 0


def make_plan(rec: dict) -> U.Plan:
    return U.Plan(tag="diag_r146", backend=rec["backend"], shots=shots_of(rec), prereg_path=str(PREREG),
                  prereg_sha256=rec["prereg_sha256"], twirl_randomizations=rec["randomizations"],
                  records_dir=str(OUTDIR), predicted_pass_probability=None,
                  measurement_reason=("a preregistered CALIBRATION of the device in-circuit and a held-out test of the "
                                      "repaired model on R145's own circuits; no protocol verdict is at stake"))


if __name__ == "__main__":
    raise SystemExit(main())
