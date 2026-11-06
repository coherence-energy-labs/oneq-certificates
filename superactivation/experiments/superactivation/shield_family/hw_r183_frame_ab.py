"""R183 -- the paired test of R182: R147 FRAME-TRACKED against R147 AS COMPILED, in one job (spends nothing until the
owner authorizes a run; this module freezes, rehearses and analyses).

R182 found that R147's key idled 2.572 us at every conditional, waiting for the feed-forward, and that the Pauli
correction it waits for can leave the device exactly (oneq.hw.pauli_frame). The device model projects +0.030 to
+0.047 in F_e. This is the experiment that decides it.

RUN 2026-10-19 (job dbap0i1gv2cc73e76mb0; docs/RESEARCH_LEDGER.md "R183 -- RUN"). The frame gained +0.0123 (paired sd
0.0042, z 2.91), and no preregistered reading fired; V3 and V4 failed. ROOT: the fresh freeze compiled TODAY's
builder, which carries R153's schedule barrier. The circuits R182's 2.57 us was derived from (R152's) have none. With
the barrier the server holds the key in |0> until the feed-forward is received: it meets its conditional 0.04-0.18 us
after its last play. R182's no-barrier hypothesis is therefore untested here.

Two parts of ONE composite job (oneq.hw.campaign), so they share the device, the calibration and the sweeps:
  asc  R147's main units as compiled (mid-circuit reads + if_test corrections on the key);
  frm  the same circuits with every conditional moved into the frame (pauli_frame.track; each checked EXACT on
       every input on the key wire before it is frozen), so the key never waits.
Only main units run: they are the units R147's F_e estimator (control_variate) uses, and the cost model prices the
pair at 22.8 s at 250 shots, against about 50 s of quota.

Analysis (analyse_ab): each arm's counts -- the frame arm's corrected by its frames first -- through R147's own unit
score (hw_dynamic_run._score) and estimator (control_variate). The verdict reads delta = F_e(frm) - F_e(asc) and its z,
with the arms' sds combined. Frame-counting bars come from the flight record at fetch (oneq.hw.randomization).

Rehearsal (rehearse): shots SAMPLED from each circuit's exact outcome law (oneq.hw.dynamic_exact). As a NULL both arms
must agree. As a POSITIVE control, the compiled arm's key takes a Z with probability p during its wait (the
mechanism R182 names), and the analysis must find delta > 0 at the size that p predicts.

  python hw_r183_frame_ab.py rehearse [--p 0.02] [--shots 250]   exact rehearsal on R152's frozen R147
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

TAG = "r183_frame_ab"
PART_ASC, PART_FRM = "asc", "frm"
R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
EXACT = 1e-12


class ABRefused(ValueError):
    """a unit or a record the A/B cannot carry faithfully"""


def main_units(rec147: dict) -> list:
    return [u for u in rec147["units"] if u["kind"] == "main"]


def per_stratum(rec147: dict, k: int) -> list:
    """the names of the first k main units of every SAMPLED stratum, and EVERY both-erased unit (the estimator
    enumerates those with exact weights, so a subset of them would bias F_e): all main units when k is 0. The
    estimator needs every stratum, so a smaller rehearsal or run is cut per stratum, never off the end of the list"""
    out, seen = [], {}
    for u in main_units(rec147):
        s = u["stratum"]
        if k and s != "erased_erased" and seen.get(s, 0) >= k:
            continue
        seen[s] = seen.get(s, 0) + 1
        out.append(u["name"])
    return out


def arms(rec147: dict, circs147: dict, names: list) -> tuple:
    """(compiled circuits: every named unit; frame-tracked circuits and their frames: ONLY the units the frame changes,
    each CHECKED exact on every input on its key wire). A unit with no conditional is the same circuit in both arms, so
    it runs once and both arms read its counts; a unit whose frame misses is refused, never run"""
    from oneq.hw import contract_synth as CS
    from oneq.hw import pauli_frame as PF
    asc, frm, frames = {}, {}, {}
    for n in names:
        qc = circs147[n]
        asc[n] = qc
        new, frame = PF.track(qc)
        if not frame.removed:
            continue
        (wire,) = CS.input_wires(qc)
        dp = PF.check(qc, new, frame, wire=wire)
        if dp > EXACT:
            raise ABRefused(f"{n}: the frame-corrected law differs by {dp:.3e}")
        frm[n], frames[n] = new, frame.to_json()
    return asc, frm, frames


def _part_record(rec147: dict, circuits: dict, extra: dict):
    from oneq.hw import freeze as F
    keep = ("backend", "snapshot", "snapshot_sha256", "calibrated_utc", "m", "d", "decoder", "randomizations",
            "layout_sector0", "layout_sector1", "schedule", "protocol")
    meta = {k: rec147[k] for k in keep if k in rec147}
    names = list(circuits)
    meta.update(units=[u for u in rec147["units"] if u["name"] in set(names)], shots_each=extra.pop("shots_each"),
                derived_from_qpy_sha256=rec147["qpy_sha256"], **extra)
    return F.freeze([(n, circuits[n]) for n in names], meta)


def compose_ab(rec147: dict, blob147: bytes, shots: int, prereg_path: str = "", prereg_sha: str = "",
               names: list | None = None, meta_extra: dict | None = None) -> tuple:
    """(composite record, QPY): the two arms of the named (default: main) units as ONE job. meta_extra joins the
    composite's metadata (a live freeze passes `sources`, which the FROZEN guard checks at submission)"""
    from oneq.hw import campaign as C
    from oneq.hw import freeze as F
    circs = F.load(rec147, blob147, check_sources=False)
    names = names or [u["name"] for u in main_units(rec147)]
    asc, frm, frames = arms(rec147, circs, names)
    if not frm:
        raise ABRefused("no unit carries a conditional: there is nothing for the frame arm to test")
    ra, ba = _part_record(rec147, asc, dict(shots_each=shots, arm="as compiled"))
    rf, bf = _part_record(rec147, frm, dict(shots_each=shots, arm="frame-tracked", frames=frames))
    parts = [C.Part(PART_ASC, ra, ba, {n: shots for n in asc}, rec147.get("randomizations"), prereg_path,
                    prereg_sha),
             C.Part(PART_FRM, rf, bf, {n: shots for n in frm}, rec147.get("randomizations"), prereg_path,
                    prereg_sha)]
    meta = dict(protocol="R183: R147 frame-tracked against as compiled, paired in one job",
                backend=rec147.get("backend"), snapshot=rec147.get("snapshot"),
                snapshot_sha256=rec147.get("snapshot_sha256"), prereg=prereg_path,
                prereg_sha256=prereg_sha, derived_from_qpy_sha256=rec147["qpy_sha256"])
    meta.update(meta_extra or {})
    # INTERLEAVED unit by unit, so both arms of a unit sit in the same execution span (R168): concatenated, every
    # compiled circuit ran before every frame-tracked one and a span drift would read as Delta (adversary, 10-18)
    return C.compose(parts, meta, interleave=True), dict(asc=(ra, ba), frm=(rf, bf))


def arm_estimate(units: list, counts: dict) -> dict:
    """R147's own estimate on one arm: hw_dynamic_run._score per unit, tallies, control_variate"""
    import hw_dynamic_run as R147

    from oneq.hw import stats as T
    stat = [T.Unit(u["name"], u["weight"], sum(counts[u["name"]].values()), u["n_bits"], 0.0, u["hi"],
                   u["stratum"]) for u in units]
    tallies = T.tally_all(stat, {u["name"]: counts[u["name"]] for u in units},
                          {u["name"]: R147._score(u) for u in units})
    means = {n: t.mean for n, t in tallies.items()}
    var = {n: max(t.s2 / t.n - t.mean ** 2, 0.0) for n, t in tallies.items()}
    cv = R147.control_variate(means, units, var, {n: t.n for n, t in tallies.items()})
    return dict(F_e=cv["F_e"], sd_F_e=cv["sd_F_e"], means=means)


def frames_of(record_frm: dict) -> dict:
    from oneq.hw import pauli_frame as PF
    return {n: PF.Frame(flips={int(k): frozenset(v) for k, v in f["flips"].items()}, removed=f["removed"])
            for n, f in record_frm["frames"].items()}


def analyse_ab(crec: dict, counts: dict, rec_asc: dict, rec_frm: dict) -> dict:
    """both arms by R147's estimator, the frame arm corrected first; delta = frm - asc with its z and the paired
    per-unit differences"""
    from oneq.hw import campaign as C
    from oneq.hw import pauli_frame as PF
    parts = C.split(crec, counts)
    fr = frames_of(rec_frm)
    nbits = {u["name"]: u["n_bits"] for u in rec_asc["units"]}
    width = {n: len(next(iter(c))) for n, c in parts[PART_FRM].items()}
    # the frame arm: its own (corrected) counts where the frame changed the circuit, the shared circuit's otherwise
    corrected = dict(parts[PART_ASC])
    corrected.update({n: PF.corrected_counts(c, fr[n], width[n]) for n, c in parts[PART_FRM].items()})
    a = arm_estimate(rec_asc["units"], parts[PART_ASC])
    f = arm_estimate(rec_asc["units"], corrected)
    delta = f["F_e"] - a["F_e"]                    # exact: the estimator is linear in the unit means
    per_unit = {n: f["means"][n] - a["means"][n] for n in a["means"]}
    sd = paired_sd(rec_asc["units"], per_unit)
    return dict(F_e_asc=a["F_e"], sd_asc=a["sd_F_e"], F_e_frm=f["F_e"], sd_frm=f["sd_F_e"], delta=delta,
                frame_units=len(parts[PART_FRM]),
                sd_paired=sd, z=delta / sd if sd > 0 else float("nan"), units=len(per_unit),
                sd_unpaired=math.hypot(a["sd_F_e"], f["sd_F_e"]),
                paired_mean_score_delta=float(np.mean(list(per_unit.values()))), n_bits=nbits)


def paired_sd(units: list, d: dict) -> float:
    """the sd of delta = F_e(frm) - F_e(asc) when both arms run the SAME units: control_variate's own weights applied
    to the per-unit differences, so the unit-to-unit spread that dominates each arm's design sd cancels. Per stratum
    b of n_b main units: var = (1.5 W_b)^2 var_b(d) / n_b (both-erased units enumerated, weight^2 var(d))"""
    import hw_dynamic_run as R147
    by: dict = {}
    for u in units:
        if u["kind"] == "main" and u["stratum"] != "erased_erased":
            by.setdefault(u["stratum"], []).append(d[u["name"]])
    var = 0.0
    for key, W in R147.H.strata(R147.M):
        xs = by[R147.stratum_name(key)]
        var += (1.5 * W) ** 2 * (float(np.var(xs, ddof=1)) / len(xs) if len(xs) > 1 else 0.0)
    return math.sqrt(var)


# ================================================================== the exact rehearsal
def _key(bits: tuple) -> str:
    return "".join(str(b) for b in reversed(bits))


def _law(qc) -> dict:
    from oneq.hw import contract_synth as CS
    from oneq.hw import dynamic_exact as DX
    small, _ = CS.compact(qc)
    return DX.distribution(small).probs


def with_wait_dephasing(qc, p: float):
    """the compiled circuit's law if its key takes a Z with probability p during its wait: a mixture of the circuit
    and the circuit with a Z inserted just before its first conditional on the key (the R182 mechanism)"""
    from oneq.hw import contract_synth as CS
    (key,) = CS.input_wires(qc)
    hit = qc.copy_empty_like()
    placed = False
    for inst in qc.data:
        if not placed and getattr(inst.operation, "blocks", None) and \
                key in [qc.find_bit(q).index for q in inst.qubits]:
            hit.z(key)
            placed = True
        hit.append(inst)
    a, b = _law(qc), _law(hit)
    return {k: (1 - p) * a.get(k, 0.0) + p * b.get(k, 0.0) for k in set(a) | set(b)}


def sample(law: dict, shots: int, rng) -> dict:
    keys = list(law)
    probs = np.clip(np.array([law[k] for k in keys]), 0, None)
    draws = rng.multinomial(shots, probs / probs.sum())
    return {_key(k): int(c) for k, c in zip(keys, draws) if c}


class Rehearsal:
    """a composed A/B and every circuit's exact laws, computed ONCE: the law of each circuit, and for each compiled
    circuit with a conditional the law with a Z on its key during the wait (the two mix linearly in p)"""

    def __init__(self, rec147: dict, blob147: bytes, shots: int, names: list | None = None):
        from oneq.hw import freeze as F
        (self.crec, self.cblob), parts = compose_ab(rec147, blob147, shots, names=names)
        self.ra, self.rf = parts["asc"][0], parts["frm"][0]
        self.shots = shots
        circs = F.load(self.crec, self.cblob, check_sources=False)
        frm = set(self.rf["circuit_order"])
        self.law, self.hit = {}, {}
        for n, qc in circs.items():
            self.law[n] = _law(qc)
            tag, inner = n.split("__", 1)
            if tag == PART_ASC and inner in frm:
                full = with_wait_dephasing(qc, 1.0)          # p = 1: the law with the Z always inserted
                self.hit[n] = full

    def run(self, p_wait: float, seed: int = 180) -> dict:
        rng = np.random.default_rng(seed)
        counts = {}
        for n, base in self.law.items():
            if n in self.hit and p_wait > 0:
                h = self.hit[n]
                law = {k: (1 - p_wait) * base.get(k, 0.0) + p_wait * h.get(k, 0.0) for k in set(base) | set(h)}
            else:
                law = base
            counts[n] = sample(law, self.shots, rng)
        return analyse_ab(self.crec, counts, self.ra, self.rf)


def rehearse(rec147: dict, blob147: bytes, shots: int, p_wait: float, seed: int = 180,
             names: list | None = None) -> dict:
    """compose the A/B from a frozen R147, sample every circuit's shots from its EXACT law (the compiled arm's key
    dephased with probability p_wait during its wait), and analyse"""
    return Rehearsal(rec147, blob147, shots, names).run(p_wait, seed)


# ================================================================== the live run (owner-authorized 2026-10-18)
OUTDIR = ROOT / "evidence" / "superactivation" / "r183_frame_ab"
PREREG = ROOT / "docs" / "PREREG_r183_frame_ab.md"
SHOTS = 250
R152_RESULT = R152_DIR / "result_db9c50oc77vs73817rog.json"


def _rel(p: pathlib.Path) -> str:
    p = pathlib.Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def freeze_live(be, fake: bool, shots: int = SHOTS) -> dict:
    """R147 frozen LIVE on the backend's current calibration -- the asc arm, frozen exactly as R152 froze it (the
    repaired CZ factor, idle modelled, alap) -- then both arms composed and every frame checked exact (compose_ab)"""
    import hw_dynamic_run as R147

    from oneq.hw import snapshot as S
    from oneq.hw import submit as U
    snap = S.from_backend(be)
    rec147, blob147, pblob = R147.freeze_run(be, snap, U.file_sha256(R147.PREREG), cz_factor=R147.REPAIRED_CZ_FACTOR,
                                             idle=not fake, schedule="alap", return_prediction_blob=True)
    sources = [HERE / "hw_r183_frame_ab.py", HERE / "hw_dynamic_run.py"] + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))
    (crec, cblob), parts = compose_ab(rec147, blob147, shots, str(PREREG), U.file_sha256(PREREG), meta_extra=dict(
        fake=fake, calibrated_utc=snap.calibrated_utc, sources=[str(p) for p in sources]))
    return dict(rec=crec, blob=cblob, r147=(rec147, blob147, pblob), parts=parts)


def save(out: dict, stamp: str) -> pathlib.Path:
    d = OUTDIR / stamp
    d.mkdir(parents=True, exist_ok=True)
    rec147, blob147, pblob = out["r147"]
    (d / "r147.json").write_text(json.dumps(rec147, indent=1) + "\n", encoding="utf-8", newline="\n")
    (d / "r147.qpy").write_bytes(blob147)
    (d / "r147.pred.qpy").write_bytes(pblob)
    for tag, (rec, blob) in out["parts"].items():
        (d / f"{tag}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        (d / f"{tag}.qpy").write_bytes(blob)
    rp = d / "composite.json"
    rp.write_text(json.dumps(out["rec"], indent=1) + "\n", encoding="utf-8", newline="\n")
    rp.with_suffix(".qpy").write_bytes(out["blob"])
    return rp


def load(rp: pathlib.Path) -> dict:
    """the composite and both arms from disk, each arm re-verified against the composite's pins"""
    from oneq.hw import campaign as C
    rec = json.loads(rp.read_text(encoding="utf-8"))
    blob = rp.with_suffix(".qpy").read_bytes()
    parts, recs = [], {}
    for pin in rec["parts"]:
        prec = json.loads((rp.parent / f"{pin['tag']}.json").read_text(encoding="utf-8"))
        pblob = (rp.parent / f"{pin['tag']}.qpy").read_bytes()
        parts.append(C.Part(pin["tag"], prec, pblob, pin["shots"], rec["randomizations"], pin["prereg"],
                            pin["prereg_sha256"]))
        recs[pin["tag"]] = prec
    C.verify_parts(rec, parts)
    return dict(rec=rec, blob=blob, asc=recs[PART_ASC], frm=recs[PART_FRM])


def plan(rec: dict):
    from oneq.hw import submit as U
    return U.Plan(tag=TAG, backend=rec["backend"], shots=rec["shots"], prereg_path=str(PREREG),
                  prereg_sha256=rec["prereg_sha256"], twirl_randomizations=rec["randomizations"],
                  records_dir=str(OUTDIR), predicted_pass_probability=None,
                  measurement_reason=("a preregistered paired A/B of ONE lever (R182's Pauli frame) on R147's units; "
                                      "no superactivation claim is possible from it"))


def submit_refusals(loaded: dict) -> list:
    """this run's own preconditions, recomputed (no network): a live freeze, the preregistration unchanged, ONE RUN,
    and a frame arm that is not empty"""
    from oneq.hw import submit as U
    rec, out = loaded["rec"], []
    if rec.get("fake"):
        out.append("a --fake freeze is a rehearsal: freeze live")
    if U.file_sha256(PREREG) != rec.get("prereg_sha256"):
        out.append(f"{PREREG.name} changed since the freeze")
    prior = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json"))]
    if any(j.get("prereg_sha256") == rec.get("prereg_sha256") for j in prior):
        out.append("ONE RUN: a job for this preregistration already exists")
    if not loaded["frm"].get("frames"):
        out.append("the frame arm holds no frame-tracked unit")
    return out


def the_job(loaded: dict) -> dict:
    jobs = [j for j in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json")))
            if j.get("qpy_sha256") == loaded["rec"]["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records for this composite; expected one")
    return jobs[0]


Z_THRESHOLDS = (3.0, 2.0, -2.0, -3.0)          # V1, the two edges of V1-null, V1-neg


def verdicts(res: dict, frame: dict | None, r152: dict | None) -> dict:
    """the preregistered decision rules V1 / V1-null / V1-neg and V2 (V3 reads the executed schedule: `v3_schedule`),
    on the LARGER of the paired sd and the frame-counting sd. FAIL-CLOSED: the preregistration makes the verdict use
    the larger, so without a frame-counting sd (`frame` None or refused) V1 / V1-null / V1-neg are NOT decided -- until
    2026-10-18 a refused bootstrap returned None and the verdict was issued on the paired sd alone, silently.
    AT THRESHOLD (prereg): a z within 3 Monte-Carlo sds of a threshold. Only a bootstrap sd carries Monte-Carlo
    error, about sd / sqrt(2 (n_boot - 1)); the paired sd is analytic."""
    frame_sd = None if not frame or frame.get("refused") else float(frame["sd"])
    sd = max(res["sd_paired"], frame_sd or 0.0)
    z = res["delta"] / sd if sd > 0 else float("nan")
    out: dict = dict(sd_used=sd, sd_rule="max(paired sd, frame-counting sd)", z=z, sd_paired=res["sd_paired"],
                     sd_frame_counting=frame_sd)
    if frame_sd is None:
        why = (frame or {}).get("refused", "not computed")
        out.update(decided=False, undecided_because=f"no frame-counting sd ({why})", V1=None, V1_null=None,
                   V1_neg=None, at_threshold=None)
    else:
        from_frames = frame_sd >= res["sd_paired"]
        z_mc = abs(z) / math.sqrt(2 * (int(frame["n_boot"]) - 1)) if from_frames else 0.0
        out.update(decided=True, sd_from="frame-counting" if from_frames else "paired", z_mc_sd=z_mc,
                   V1=bool(z >= 3 and res["delta"] >= 0.015), V1_null=bool(abs(z) < 2), V1_neg=bool(z <= -3),
                   at_threshold=[t for t in Z_THRESHOLDS if abs(z - t) <= 3 * z_mc])
        out["reading"] = ("V1" if out["V1"] else "V1-null" if out["V1_null"] else "V1-neg" if out["V1_neg"]
                          else "NONE of V1 / V1-null / V1-neg (the preregistration names no reading here)")
    out["V4_projection_band"] = dict(lo=0.030, hi=0.047, inside_2sd=bool(0.030 - 2 * sd <= res["delta"] <= 0.047 + 2 * sd))
    if r152:
        d2 = res["F_e_asc"] - r152["F_e"]
        out["V2_drift"] = dict(r152_F_e=r152["F_e"], asc_minus_r152=d2, within_2sd=bool(abs(d2) <= 2 * res["sd_asc"]))
    return out


V3_WAIT_S, V3_TOL_S = 2.57e-6, 0.15e-6          # prereg V3: the asc key waits 2.57 +- 0.15 us, the frm key not at all


def _dt(loaded: dict) -> float:
    """the schedule's cycle from the freeze's own snapshot -- refused, never defaulted, when it is not there"""
    dt = (loaded["asc"].get("snapshot") or {}).get("dt")
    if not dt:
        raise ValueError("the compiled arm's snapshot records no dt")
    return float(dt)


def v3_schedule(loaded: dict, index: dict) -> dict:
    """prereg V3 READ from the executed schedules (oneq.hw.dynamic_timing.executed_key_wait / quantum_idle): every
    frame-tracked unit's compiled key wait at its conditional, and its frame twin's key idle in a quantum state.
    PASS when the median compiled wait is inside 2.57 +- 0.15 us AND no frame key idles more than 0.15 us. Refused
    (`refused`) when a schedule is missing -- never read as a pass."""
    from oneq.hw import dynamic_timing as DT
    from oneq.hw import freeze as F
    from oneq.hw import timing as TM
    dt = _dt(loaded)
    circs = F.load(loaded["rec"], loaded["blob"], check_sources=False)
    pubs = {p["name"]: p for p in index["pubs"]}
    waits, ground, frm_idle, missing = [], [], [], []
    for n in loaded["frm"]["frames"]:
        a, f = f"{PART_ASC}__{n}", f"{PART_FRM}__{n}"
        ra, rf = TM.from_metadata(pubs[a]["metadata"]), TM.from_metadata(pubs[f]["metadata"])
        if ra is None or rf is None:
            missing.append(n)
            continue
        qc = circs[a]
        key = next(qc.find_bit(inst.qubits[0]).index for inst in qc.data if inst.operation.name == "if_else")
        w = DT.executed_key_wait(ra, key, dt)
        waits.append(w["wait_s"])
        ground.append(w["ground_s"] or 0.0)
        frm_idle.append(DT.quantum_idle(rf, dt)["per_qubit"].get(key, {}).get("quantum_s", 0.0))
    if missing:
        return dict(refused=f"{len(missing)} units have no executed schedule", missing=missing[:10])
    med = float(np.median(waits))
    ok_asc = abs(med - V3_WAIT_S) <= V3_TOL_S
    ok_frm = max(frm_idle) <= V3_TOL_S
    return dict(units=len(waits), dt=dt, asc_key_wait_us=dict(min=min(waits) * 1e6, median=med * 1e6,
                                                               max=max(waits) * 1e6),
                asc_key_ground_idle_us_median=float(np.median(ground)) * 1e6,
                frm_key_quantum_idle_us_max=max(frm_idle) * 1e6,
                asc_inside_band=sum(abs(w - V3_WAIT_S) <= V3_TOL_S for w in waits),
                asc_wait_as_preregistered=bool(ok_asc), frm_key_does_not_wait=bool(ok_frm), PASS=bool(ok_asc and ok_frm))


def stall_posthoc(loaded: dict, index: dict, counts: dict, n_perm: int = 4000) -> dict:
    """POST HOC (not preregistered): where the compiled arm's extra idle falls, from the executed schedules, and
    whether per-unit Delta follows it (a permutation test of the correlation)"""
    from oneq.hw import campaign as C
    from oneq.hw import dynamic_timing as DT
    from oneq.hw import pauli_frame as PF
    from oneq.hw import timing as TM
    dt = _dt(loaded)
    pubs = {p["name"]: p for p in index["pubs"]}
    parts = C.split(loaded["rec"], counts)
    fr = frames_of(loaded["frm"])
    width = {n: len(next(iter(c))) for n, c in parts[PART_FRM].items()}
    corrected = dict(parts[PART_ASC])
    corrected.update({n: PF.corrected_counts(c, fr[n], width[n]) for n, c in parts[PART_FRM].items()})
    a = arm_estimate(loaded["asc"]["units"], parts[PART_ASC])
    f = arm_estimate(loaded["asc"]["units"], corrected)
    dq, dp, de = [], [], []
    for n in loaded["frm"]["frames"]:
        qa = DT.quantum_idle(TM.from_metadata(pubs[f"{PART_ASC}__{n}"]["metadata"]), dt)
        qf = DT.quantum_idle(TM.from_metadata(pubs[f"{PART_FRM}__{n}"]["metadata"]), dt)
        dq.append((qa["quantum_s"] - qf["quantum_s"]) * 1e6)
        dp.append((qa["post_read_s"] - qf["post_read_s"]) * 1e6)
        de.append(f["means"][n] - a["means"][n])
    rng = np.random.default_rng(183)
    out: dict = dict(label="POST HOC", units=len(de))
    for name, x in (("quantum_idle_diff_us", np.array(dq)), ("post_read_idle_diff_us", np.array(dp))):
        y = np.array(de)
        r = float(np.corrcoef(x, y)[0, 1])
        perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
        out[name] = dict(min=float(x.min()), median=float(np.median(x)), max=float(x.max()), mean=float(x.mean()),
                         slope_delta_per_us=float(np.polyfit(x, y, 1)[0]), r=r,
                         perm_p_two_sided=float((np.abs(perm) >= abs(r)).mean()))
    return out


def r152_main_fe() -> dict | None:
    """R152's R147 main-unit F_e as analysed then (the V2 drift control)"""
    if not R152_RESULT.is_file():
        return None
    m = json.loads(R152_RESULT.read_text(encoding="utf-8"))["r147"]["measured"]
    return dict(F_e=float(m["F_e"]), sd=float(m["sd_F_e"]))


def frame_counting_sd(loaded: dict, index: dict, arrays: dict, n_boot: int = 300) -> dict:
    """Delta's sd over twirling FRAMES (oneq.hw.randomization): resample whole randomizations per circuit, recompute
    Delta through the same estimator; with the shot bootstrap of the same Delta and their ratio (the inflation), and
    the frame-homogeneity map. A refusal is RETURNED with its reason (`refused`), never swallowed: `verdicts` then
    declines to decide (R183's first fetch lost its frame bars to a truncated last frame this way, 2026-10-18)."""
    from oneq.hw import randomization as R
    try:
        shots = R.from_flight(index, arrays)

        def est(counts):
            return analyse_ab(loaded["rec"], counts, loaded["asc"], loaded["frm"])["delta"]
        fr = R.bootstrap(est, shots, n_boot=n_boot, seed=183, mode="randomization")
        sh = R.bootstrap(est, shots, n_boot=n_boot, seed=183, mode="shot")
        hm = R.homogeneity_map(shots)
    except R.RandomizationRefused as e:
        return dict(refused=f"RandomizationRefused: {e}")
    frames = sorted({(s.R, s.n, s.last) for s in shots.values()}, key=str)
    return dict(sd=float(fr.sd["value"]), n_boot=n_boot, blocks=fr.blocks, scale=fr.scale,
                shot_sd=float(sh.sd["value"]), inflation=R.inflation(fr, sh)["value"],
                frames=[dict(R=r, n=n, last=last) for r, n, last in frames],
                homogeneity=dict((k, hm[k]) for k in ("tests", "threshold", "flagged", "flagged_circuits"))
                | dict(strongest=hm["strongest"][:5]))


def dryrun_counts(loaded: dict, seed: int = 183) -> dict:
    """every circuit of the FROZEN composite sampled from its exact outcome law (no device noise): both arms must agree
    on the ideal F_e and Delta must be 0 within shot noise -- the analysis path end to end before any spend"""
    from oneq.hw import freeze as F
    rng = np.random.default_rng(seed)
    circs = F.load(loaded["rec"], loaded["blob"], check_sources=False)
    shots = loaded["rec"]["shots"]
    return {n: sample(_law(qc), shots[n], rng) for n, qc in circs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rehearse", "freeze", "recompose", "dryrun", "submit", "fetch", "analyse",
                                     "reanalyse"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("counts", nargs="?")
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--per-stratum", type=int, default=0, help="the first N main units of each stratum (0: all)")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    if a.mode == "rehearse":
        rec = json.loads((R152_DIR / "r147.json").read_text(encoding="utf-8"))
        blob = (R152_DIR / "r147.qpy").read_bytes()
        names = per_stratum(rec, a.per_stratum)
        out = rehearse(rec, blob, a.shots, a.p, names=names)
        out.pop("n_bits", None)
        print(json.dumps(out, indent=1))
        return 0
    import datetime

    from oneq.hw import campaign as C
    from oneq.hw import freeze as F
    from oneq.hw import layout as L
    from oneq.hw import snapshot as S
    from oneq.hw import submit as U
    if a.mode == "freeze":
        if a.fake:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
            be = FakeKingston()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            be = QiskitRuntimeService().backend(a.live)
        out = freeze_live(be, a.fake, a.shots)
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = save(out, stamp)
        pl = plan(out["rec"])
        print(f"frozen {out['rec']['n_circuits']} circuits ({len(out['parts']['frm'][0]['frames'])} frame-tracked) -> "
              f"{_rel(rp)}; {pl.total_shots} shots, cost estimate {pl.cost_s():.1f} s "
              f"(x{U.QUOTA_MARGIN} margin: {pl.cost_s() * U.QUOTA_MARGIN:.0f} s)")
        return 0
    rp = pathlib.Path(a.record).resolve()
    if a.mode == "recompose":
        # the same live R147 freeze (its snapshot, layout and predictions), recomposed under today's compose_ab --
        # e.g. interleaved -- into a NEW stamped folder; the old composite is left as it was
        d = rp.parent
        rec147 = json.loads((d / "r147.json").read_text(encoding="utf-8"))
        blob147, pblob = (d / "r147.qpy").read_bytes(), (d / "r147.pred.qpy").read_bytes()
        old = json.loads(rp.read_text(encoding="utf-8"))
        sources = ([HERE / "hw_r183_frame_ab.py", HERE / "hw_dynamic_run.py"]
                   + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py")))
        (crec, cblob), parts = compose_ab(rec147, blob147, old["shots"][next(iter(old["shots"]))], str(PREREG),
                                          U.file_sha256(PREREG), meta_extra=dict(
                                              fake=old.get("fake", False), calibrated_utc=old.get("calibrated_utc"),
                                              recomposed_from=_rel(rp), sources=[str(p) for p in sources]))
        stamp = d.name + "_recomposed_" + datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%SZ")
        out = save(dict(rec=crec, blob=cblob, r147=(rec147, blob147, pblob), parts=parts), stamp)
        print(f"recomposed {crec['n_circuits']} circuits (interleaved={crec.get('interleaved')}) -> {_rel(out)}")
        return 0
    loaded = load(rp)
    if a.mode == "reanalyse":
        dest = reanalyse(rp, loaded)
        res = json.loads(dest.read_text(encoding="utf-8"))
        print(json.dumps(dict((k, res[k]) for k in ("F_e_asc", "F_e_frm", "delta", "sd_paired", "z")), indent=1))
        print(json.dumps(res["verdicts"], indent=1, default=str))
        print(f"-> {_rel(dest)}")
        return 0
    if a.mode in ("dryrun", "analyse"):
        counts = (dryrun_counts(loaded) if a.mode == "dryrun"
                  else json.loads(pathlib.Path(a.counts).read_text(encoding="utf-8")))
        res = analyse_ab(loaded["rec"], counts, loaded["asc"], loaded["frm"])
        res.pop("n_bits", None)
        res["verdicts"] = verdicts(res, None, r152_main_fe())
        dest = rp.parent / f"{'dryrun' if a.mode == 'dryrun' else 'analysis'}_{rp.stem}.json"
        dest.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in res.items() if k != "verdicts"}, indent=1, default=str))
        print(json.dumps(res["verdicts"], indent=1, default=str))
        return 0
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    be = svc.backend(loaded["rec"]["backend"])
    pl = plan(loaded["rec"])
    if a.mode == "submit":
        bad = submit_refusals(loaded)
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        live = S.from_backend(be)
        circuits = F.load(loaded["rec"], loaded["blob"])
        health = sorted({r for qc in circuits.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(pl, loaded["rec"], loaded["blob"], be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        return 0
    jid = the_job(loaded)["job_id"]
    job = svc.job(jid)
    if "DONE" not in str(job.status()):
        print(f"job {jid}: {job.status()}")
        return 4
    result = job.result()
    index, arrays = C.flight_record(result, loaded["rec"])
    index["job_metrics"] = C._jsonable(job.metrics()) if hasattr(job, "metrics") else None
    after = S.from_backend(be)
    stem = rp.parent / f"flight_{jid}"
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8",
                                         newline="\n")
    (rp.parent / f"after_{jid}.snapshot.json").write_text(json.dumps(after.to_json(), indent=1) + "\n",
                                                         encoding="utf-8", newline="\n")
    counts = U.counts_from_result(result, loaded["rec"]["circuit_order"])
    (rp.parent / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    res = full_analysis(loaded, index, arrays, counts)
    (rp.parent / f"result_{jid}.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8",
                                                  newline="\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


def full_analysis(loaded: dict, index: dict, arrays: dict, counts: dict) -> dict:
    """everything the preregistration asks of a fetched job: both arms, Delta, the frame-counting bars, V1-V4, and
    the post-hoc stall reading -- from the flight record alone (fetch and reanalyse call the same function)"""
    res = analyse_ab(loaded["rec"], counts, loaded["asc"], loaded["frm"])
    res.pop("n_bits", None)
    res["frame_counting"] = frame_counting_sd(loaded, index, arrays)
    res["verdicts"] = verdicts(res, res["frame_counting"], r152_main_fe())
    try:
        res["verdicts"]["V3_schedule"] = v3_schedule(loaded, index)
    except ValueError as e:
        res["verdicts"]["V3_schedule"] = dict(refused=str(e))
    try:
        res["stall_posthoc"] = stall_posthoc(loaded, index, counts)
    except (ValueError, TypeError) as e:
        res["stall_posthoc"] = dict(refused=str(e))
    return res


def reanalyse(rp: pathlib.Path, loaded: dict) -> pathlib.Path:
    """the job's saved flight record and counts, through today's `full_analysis`, into reanalysis_<job>.json beside
    the result written at fetch (which stays as it was); the counts must be exactly the flight record's shots"""
    from oneq.hw import randomization as R
    jid = the_job(loaded)["job_id"]
    index = json.loads((rp.parent / f"flight_{jid}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(rp.parent / f"flight_{jid}.npz"))
    counts = json.loads((rp.parent / f"counts_{jid}.json").read_text(encoding="utf-8"))
    try:
        shots = R.from_flight(index, arrays)
    except R.RandomizationRefused:
        shots = None
    if shots is not None and any(shots[n].counts() != {k: v for k, v in counts[n].items() if v} for n in shots):
        raise SystemExit("REFUSED: the saved counts are not the flight record's shots")
    res = full_analysis(loaded, index, arrays, counts)
    res["reanalysed_from"] = dict(job=jid, flight=f"flight_{jid}.json", counts=f"counts_{jid}.json",
                                  fetch_result=f"result_{jid}.json")
    dest = rp.parent / f"reanalysis_{jid}.json"
    dest.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    return dest


if __name__ == "__main__":
    raise SystemExit(main())
