"""R194 -- the 2 x 2 that R183 could not be: R147's feed-forward key WITH and WITHOUT R153's schedule barrier, each as
compiled and frame-tracked, in ONE job (spends nothing until the owner authorizes a run).

Why (ledger R183). R182 projected +0.030 to +0.047 F_e from moving the key's Pauli correction into the frame. It derived
that from R152's circuits, whose key waits 2.572 us in a quantum state at its conditional (the ASAP model). R183 froze
fresh, and its compiled arm carried R153's barrier: the server held the key in |0> until the feed-forward arrived, the
key met its conditional 0.04-0.18 us after its last play, and R182's mechanism was absent. R183 measured the frame
against the BARRIER-EARLY design (+0.0123, z 2.9). R182's own hypothesis is untested. R183's module is left unchanged
(its record pins it); this one reuses its estimator, frames and paired sd.

Four arms on one calibration, so every comparison is paired:

  asc  R147's main units as compiled TODAY (R153's barrier on its dynamic units)       -- R183's compiled arm
  nob  the same compiled circuits with the barrier REMOVED, nothing else changed       -- R152's design
  frm  asc's conditional units frame-tracked (the barrier kept)                         -- R183's frame arm
  fnb  nob's conditional units frame-tracked (no barrier, no conditional: static)       -- R182's frame arm

A unit an arm does not change is its parent's circuit and runs once (PARENT: nob reads asc, frm reads asc, fnb reads
nob). On R183's freeze that is 195 + 131 + 130 + 130 = 586 circuits.

Contrasts, each linear in the arms' F_e, each with its PAIRED sd (R183's paired_sd on per-unit differences) and a
frame-counting sd from ONE randomization bootstrap of all of them (CONTRASTS):
  R182          fnb - nob    PRIMARY: R182's hypothesis on R182's own design
  R183          frm - asc    R183 on a new calibration (a replication)
  barrier_asc   asc - nob    does the barrier protect a compiled key, which then waits in |0>?
  barrier_frm   frm - fnb    what the barrier costs when nothing waits for the feed-forward
  interaction   (fnb - nob) - (frm - asc)

The STRUCTURAL PREMISE is checked on the frozen bytes before any spend: the oneq.hw.premise ranges (STRUCTURAL_PREMISE,
which the preregistration carries verbatim) AND `premise_refusals`. Every compiled conditional unit must carry exactly
one barrier and, with it removed, model a 2.42-2.72 us key wait (R182's 2.572 us). Every frame unit must be exact on
every key-wire input. After the run, V3 reads what the server actually played from the executed schedules.

  python hw_r194_barrier_frame_2x2.py rehearse [--per-stratum N] [--shots S] [--p P]   exact rehearsal (R183's freeze)
  python hw_r194_barrier_frame_2x2.py freeze [--live ibm_kingston]                      live freeze, no spend
  python hw_r194_barrier_frame_2x2.py dryrun RECORD | check RECORD                      exact-law dry run / preflight
  ONEQ_IBM_SUBMIT=1 python hw_r194_barrier_frame_2x2.py submit RECORD --i-understand-this-spends-credits
  python hw_r194_barrier_frame_2x2.py fetch RECORD | reanalyse RECORD
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
import hw_r183_frame_ab as AB  # noqa: E402

TAG = "r194_barrier_frame_2x2"
ARMS = ("asc", "nob", "frm", "fnb")
PARENT = dict(nob="asc", frm="asc", fnb="nob")
FRAMED = ("frm", "fnb")
LABELS = dict(asc="as compiled (R153's barrier)", nob="as compiled, barrier removed",
              frm="frame-tracked, barrier kept", fnb="frame-tracked, barrier removed")
CONTRASTS = {
    "R182": {"fnb": 1, "nob": -1},
    "R183": {"frm": 1, "asc": -1},
    "barrier_asc": {"asc": 1, "nob": -1},
    "barrier_frm": {"frm": 1, "fnb": -1},
    "interaction": {"fnb": 1, "nob": -1, "frm": -1, "asc": 1},
}
PRIMARY = "R182"
R182_WAIT_S = 2.572e-6
MODEL_BAND_S = (2.42e-6, 2.72e-6)              # the frozen no-barrier key's modelled wait: R182's 2.572 us +- 0.15
EXECUTED_BAND_S = (2.07e-6, 3.07e-6)           # V3: the median EXECUTED no-barrier key wait, R182's +- 0.5 us
FRAME_IDLE_MAX_S = 0.15e-6                     # V3: R183's bound on a frame key's quantum idle
STRUCTURAL_PREMISE = {
    "nob__": {"barriers": [0, 0], "modelled_conditional_wait_s": [0.0, 2.72e-6]},
    "frm__": {"barriers": [1, 1], "if_else": [0, 0]},
    "fnb__": {"barriers": [0, 0], "if_else": [0, 0]},
}
OUTDIR = ROOT / "evidence" / "superactivation" / TAG
PREREG = ROOT / "docs" / "PREREG_r194_barrier_frame_2x2.md"
SHOTS = 250
R183_DIR = ROOT / "evidence" / "superactivation" / "r183_frame_ab" / "20261019T032044Z_recomposed_033301Z"
R183_RESULT = R183_DIR / "reanalysis_dbap0i1gv2cc73e76mb0.json"
SOURCES = ("hw_r194_barrier_frame_2x2.py", "hw_r183_frame_ab.py", "hw_dynamic_run.py", "hw_dynamic.py",
           "hw_faithful.py", "hw_levers.py", "hw_superact_run.py", "hw_two_use.py", "hardware_design.py",
           "ppt_family.py")


def _names(qc) -> list:
    return [inst.operation.name for inst in qc.data]


def strip_barriers(qc):
    """the circuit with every barrier removed and nothing else changed: a barrier is a scheduling directive, so the
    gates, their order on every qubit and the ideal law are the same"""
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name != "barrier":
            out.append(inst)
    return out


def arms4(circs147: dict, names: list, check_frames: bool = True) -> tuple:
    """({arm: {unit: circuit}}, {framed arm: {unit: frame json}}): asc every named unit; nob ONLY the units that carry a
    barrier, stripped; frm / fnb ONLY the units whose asc / nob circuit carries a conditional, frame-tracked and each
    CHECKED exact on every input on its key wire (a frame that misses is refused, never run). The check runs on the
    barrier-free circuit, whose law is the same (a barrier is the identity on states) and faster to compute. It runs
    ONCE per unit when the two frame twins carry the same frame and the same barrier-free instructions (law_key), and
    separately for each twin otherwise. check_frames=False skips only that exact check (six preparations per frame)
    for a structural reading; compose_2x2, and so every freeze and rehearsal, always checks"""
    from oneq.hw import contract_synth as CS
    from oneq.hw import pauli_frame as PF

    def exact(arm, n, base, new, frame):
        (wire,) = CS.input_wires(base)
        dp = PF.check(strip_barriers(base), strip_barriers(new), frame, wire=wire)
        if dp > AB.EXACT:
            raise AB.ABRefused(f"{arm}/{n}: the frame-corrected law differs by {dp:.3e}")
    out: dict = {a: {} for a in ARMS}
    frames: dict = {a: {} for a in FRAMED}
    for n in names:
        qc = circs147[n]
        out["asc"][n] = qc
        bare = qc
        if "barrier" in _names(qc):
            bare = out["nob"][n] = strip_barriers(qc)
        got = {}
        for arm, base in (("frm", qc), ("fnb", bare)):
            new, frame = PF.track(base)
            if frame.removed:
                got[arm] = (base, new, frame)
        if check_frames and got:
            same = (set(got) == set(FRAMED) and got["frm"][2].to_json() == got["fnb"][2].to_json()
                    and law_key(got["frm"][1]) == law_key(got["fnb"][1]))
            for arm in (["fnb"] if same else list(got)):
                exact(arm, n, *got[arm])
        for arm, (_, new, frame) in got.items():
            out[arm][n], frames[arm][n] = new, frame.to_json()
    if not out["frm"]:
        raise AB.ABRefused("no unit carries a conditional: there is nothing for the frame arms to test")
    if set(out["frm"]) != set(out["fnb"]):
        raise AB.ABRefused(f"the frame arms track different units: {sorted(set(out['frm']) ^ set(out['fnb']))[:5]}")
    if not out["nob"]:
        raise AB.ABRefused("no unit carries a barrier: there is no barrier factor to test")
    return out, frames


def premise_refusals(arm_circs: dict) -> list:
    """the per-unit premise oneq.hw.premise's prefix ranges cannot state: every compiled unit with a conditional carries
    exactly ONE barrier (R153's), has a barrier-free twin that models R182's key wait (MODEL_BAND_S), and has both
    frame twins. Empty = the premise holds. Each check fails on a builder without the barrier (nob would be asc), on a
    timing model that moved, or on a missing twin"""
    from oneq.hw import premise as PR
    bad = []
    cond = sorted(n for n, qc in arm_circs["asc"].items() if "if_else" in _names(qc))
    if not cond:
        return ["no compiled unit carries a conditional"]
    for n in cond:
        if _names(arm_circs["asc"][n]).count("barrier") != 1:
            bad.append(f"asc/{n}: {_names(arm_circs['asc'][n]).count('barrier')} barriers, not R153's one")
        for arm in ("nob", "frm", "fnb"):
            if n not in arm_circs[arm]:
                bad.append(f"{arm}/{n}: missing")
        if n in arm_circs["nob"]:
            fp = PR.fingerprint(arm_circs["nob"][n])
            w = fp["modelled_conditional_wait_s"]
            if fp["barriers"] or w is None or not MODEL_BAND_S[0] <= w <= MODEL_BAND_S[1]:
                bad.append(f"nob/{n}: barriers {fp['barriers']}, modelled key wait {w!r} s outside {MODEL_BAND_S}")
    return bad


def by_arm(circuits: dict) -> dict:
    """a composite's circuits ('tag__unit') regrouped as {arm: {unit: circuit}}"""
    out: dict = {a: {} for a in ARMS}
    for name, qc in circuits.items():
        tag, unit = name.split("__", 1)
        out[tag][unit] = qc
    return out


def compose_2x2(rec147: dict, blob147: bytes, shots: int, prereg_path: str = "", prereg_sha: str = "",
                names: list | None = None, meta_extra: dict | None = None) -> tuple:
    """((composite record, QPY), {arm: (part record, part QPY)}, {arm: {unit: circuit}}): the four arms of the named
    (default: main) units as ONE job, INTERLEAVED unit by unit so every arm of a unit sits in the same execution span"""
    from oneq.hw import campaign as C
    from oneq.hw import freeze as F
    circs = F.load(rec147, blob147, check_sources=False)
    names = names or [u["name"] for u in AB.main_units(rec147)]
    arm_circs, frames = arms4(circs, names)
    recs, parts = {}, []
    for a in ARMS:
        extra = dict(shots_each=shots, arm=LABELS[a], reads_unchanged_units_from=PARENT.get(a))
        if a in frames:
            extra["frames"] = frames[a]
        r, b = AB._part_record(rec147, arm_circs[a], extra)
        recs[a] = (r, b)
        parts.append(C.Part(a, r, b, {n: shots for n in arm_circs[a]}, rec147.get("randomizations"), prereg_path,
                            prereg_sha))
    meta = dict(protocol="R194: R147 with and without R153's barrier, as compiled and frame-tracked, one 2 x 2 job",
                backend=rec147.get("backend"), snapshot=rec147.get("snapshot"),
                snapshot_sha256=rec147.get("snapshot_sha256"), prereg=prereg_path, prereg_sha256=prereg_sha,
                derived_from_qpy_sha256=rec147["qpy_sha256"], arms=list(ARMS), parent=PARENT,
                contrasts=CONTRASTS, primary=PRIMARY)
    meta.update(meta_extra or {})
    return C.compose(parts, meta, interleave=True), recs, arm_circs


def arm_counts(crec: dict, counts: dict, part_recs: dict) -> dict:
    """{arm: {unit: counts}}: each arm's own circuits (the framed arms' counts corrected by their frames), and its
    parent's for every unit it does not change"""
    from oneq.hw import campaign as C
    from oneq.hw import pauli_frame as PF
    parts = C.split(crec, counts)
    out = {"asc": dict(parts["asc"])}
    for a in ARMS[1:]:
        own = dict(parts.get(a, {}))
        if a in FRAMED:
            fr = AB.frames_of(part_recs[a])
            own = {n: PF.corrected_counts(c, fr[n], len(next(iter(c)))) for n, c in own.items()}
        out[a] = {**out[PARENT[a]], **own}
    return out


def analyse_2x2(crec: dict, counts: dict, part_recs: dict) -> dict:
    """every arm by R147's estimator and every contrast with its paired sd. A contrast is exact as the combination of
    the arms' F_e (the estimator is linear in the unit means); its paired sd applies control_variate's weights to the
    same combination of per-unit means, so the unit-to-unit spread every arm shares cancels"""
    units = part_recs["asc"]["units"]
    cnt = arm_counts(crec, counts, part_recs)
    est = {a: AB.arm_estimate(units, cnt[a]) for a in ARMS}
    contrasts = {}
    for k, coef in CONTRASTS.items():
        value = sum(c * est[a]["F_e"] for a, c in coef.items())
        per_unit = {n: sum(c * est[a]["means"][n] for a, c in coef.items()) for n in est["asc"]["means"]}
        sd = AB.paired_sd(units, per_unit)
        contrasts[k] = dict(value=value, sd_paired=sd, z=value / sd if sd > 0 else float("nan"))
    return dict(arms={a: dict(F_e=est[a]["F_e"], sd_F_e=est[a]["sd_F_e"]) for a in ARMS}, contrasts=contrasts,
                units=len(units), own_circuits={a: len(part_recs[a]["circuit_order"]) for a in ARMS},
                per_unit_means={a: est[a]["means"] for a in ARMS})


# ================================================================== the exact rehearsal
def with_key_z(qc):
    """the circuit with a Z on the key just before its first conditional (R182's mechanism, as R183's
    `with_wait_dephasing` places it); refused when the key has no conditional"""
    from oneq.hw import contract_synth as CS
    (key,) = CS.input_wires(qc)
    hit = qc.copy_empty_like()
    placed = False
    for inst in qc.data:
        if not placed and getattr(inst.operation, "blocks", None) and key in [qc.find_bit(q).index for q in inst.qubits]:
            hit.z(key)
            placed = True
        hit.append(inst)
    if not placed:
        raise AB.ABRefused("the key has no conditional to wait at")
    return hit


def _structure(qc) -> list:
    """the circuit's instructions as plain data, recursively: name, qubit and clbit indices, parameters, a control-flow
    op's condition and blocks. Deterministic across processes -- a QPY dump is NOT (two dumps of one circuit differed
    between runs of this lane's own test, 2026-10-20), so it cannot key a cache"""
    out = []
    for inst in qc.data:
        op = inst.operation
        row = [op.name, [qc.find_bit(b).index for b in inst.qubits], [qc.find_bit(b).index for b in inst.clbits],
               [repr(float(p)) if isinstance(p, (int, float)) else repr(p) for p in op.params
                if not hasattr(p, "data")]]
        blocks = getattr(op, "blocks", None)
        if blocks:
            from qiskit.circuit import Clbit
            cond = getattr(op, "condition", None)
            if isinstance(cond, tuple):                  # (Clbit, value) or (ClassicalRegister, value)
                bit, val = cond
                cond = [qc.find_bit(bit).index if isinstance(bit, Clbit) else repr(bit), int(val)]
            row += [repr(cond), [_structure(b) for b in blocks]]  # an expr's repr names bits by register and index
        out.append(row)
    return out


def law_key(qc) -> str:
    """a digest of the circuit's instructions with its barriers removed: a barrier is the identity on states, so two
    circuits with one key have one exact law (nob shares asc's, fnb shares frm's) and it is computed once"""
    import hashlib
    return hashlib.sha256(json.dumps(_structure(strip_barriers(qc))).encode()).hexdigest()


class Rehearsal:
    """a composed 2 x 2 and every circuit's exact law, computed ONCE per barrier-free circuit (law_key), and for each
    no-barrier conditional circuit its law with a Z on the key during the wait (R182's mechanism; the two mix
    linearly in p)"""

    def __init__(self, rec147: dict, blob147: bytes, shots: int, names: list | None = None):
        from oneq.hw import freeze as F
        (self.crec, self.cblob), recs, _ = compose_2x2(rec147, blob147, shots, names=names)
        self.part_recs = {a: r for a, (r, _) in recs.items()}
        self.shots = shots
        framed = set(self.part_recs["fnb"]["circuit_order"])
        self.law, self.hit, cache = {}, {}, {}
        for n, qc in F.load(self.crec, self.cblob, check_sources=False).items():
            k = law_key(qc)
            if k not in cache:
                cache[k] = AB._law(strip_barriers(qc))     # the same law, computed faster
            self.law[n] = cache[k]
            tag, inner = n.split("__", 1)
            if tag == "nob" and inner in framed:
                self.hit[n] = AB._law(with_key_z(qc))          # p = 1: the law with the Z always inserted
        self.distinct_laws = len(cache)

    def run(self, p_wait: float, seed: int = 194) -> dict:
        rng = np.random.default_rng(seed)
        counts = {}
        for n, base in self.law.items():
            law = base
            if n in self.hit and p_wait > 0:
                h = self.hit[n]
                law = {k: (1 - p_wait) * base.get(k, 0.0) + p_wait * h.get(k, 0.0) for k in set(base) | set(h)}
            counts[n] = AB.sample(law, self.shots, rng)
        return analyse_2x2(self.crec, counts, self.part_recs)


# ================================================================== the live run
def _rel(p) -> str:
    return AB._rel(p)


def source_paths() -> list:
    return [HERE / s for s in SOURCES] + sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))


def freeze_live(be, fake: bool, shots: int = SHOTS) -> dict:
    """R147 frozen LIVE on the backend's current calibration exactly as R183 froze it (R152's settings: the repaired CZ
    factor, idle modelled, alap), the four arms composed, every frame checked exact, and the premise checked -- a
    freeze whose premise fails is refused here, before anything is written"""
    import hw_dynamic_run as R147

    from oneq.hw import snapshot as S
    from oneq.hw import submit as U
    snap = S.from_backend(be)
    rec147, blob147, pblob = R147.freeze_run(be, snap, U.file_sha256(R147.PREREG), cz_factor=R147.REPAIRED_CZ_FACTOR,
                                             idle=not fake, schedule="alap", return_prediction_blob=True)
    (crec, cblob), recs, arm_circs = compose_2x2(rec147, blob147, shots, str(PREREG), U.file_sha256(PREREG),
                                                 meta_extra=dict(fake=fake, calibrated_utc=snap.calibrated_utc,
                                                                 sources=[str(p) for p in source_paths()]))
    bad = premise_refusals(arm_circs)
    if bad:
        raise AB.ABRefused("the structural premise fails on this freeze: " + "; ".join(bad[:5]))
    return dict(rec=crec, blob=cblob, r147=(rec147, blob147, pblob), parts=recs)


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
    """the composite and its four arms from disk, each arm re-verified against the composite's pins"""
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
    if set(recs) != set(ARMS):
        raise SystemExit(f"REFUSED: the composite holds arms {sorted(recs)}, not {list(ARMS)}")
    return dict(rec=rec, blob=blob, parts=recs)


def plan(rec: dict):
    from oneq.hw import submit as U
    return U.Plan(tag=TAG, backend=rec["backend"], shots=rec["shots"], prereg_path=str(PREREG),
                  prereg_sha256=rec["prereg_sha256"], twirl_randomizations=rec["randomizations"],
                  records_dir=str(OUTDIR), predicted_pass_probability=None, structural_premise=STRUCTURAL_PREMISE,
                  measurement_reason=("a preregistered paired 2 x 2 of two levers (R153's barrier, R182's Pauli "
                                      "frame) on R147's units; no superactivation claim is possible from it"))


def submit_refusals(loaded: dict, circuits: dict | None = None) -> list:
    """this run's own preconditions, recomputed (no network): a live freeze, the preregistration unchanged, ONE RUN,
    non-empty frame arms, and the per-unit premise on the frozen circuits"""
    from oneq.hw import submit as U
    rec, out = loaded["rec"], []
    if rec.get("fake"):
        out.append("a --fake freeze is a rehearsal: freeze live")
    if U.file_sha256(PREREG) != rec.get("prereg_sha256"):
        out.append(f"{PREREG.name} changed since the freeze")
    prior = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json"))]
    if any(j.get("prereg_sha256") == rec.get("prereg_sha256") for j in prior):
        out.append("ONE RUN: a job for this preregistration already exists")
    for a in FRAMED:
        if not loaded["parts"][a].get("frames"):
            out.append(f"the {a} arm holds no frame-tracked unit")
    if circuits is not None:
        out.extend(f"PREMISE: {r}" for r in premise_refusals(by_arm(circuits)))
    return out


def the_job(loaded: dict) -> dict:
    jobs = [j for j in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json")))
            if j.get("qpy_sha256") == loaded["rec"]["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records for this composite; expected one")
    return jobs[0]


def references() -> dict:
    """the measured comparators the preregistration names: R183's reanalysed arms and delta, R152's R147"""
    out: dict = {}
    if R183_RESULT.is_file():
        r = json.loads(R183_RESULT.read_text(encoding="utf-8"))
        sd = max(float(r["sd_paired"]), float(r["frame_counting"]["sd"]))
        out["R183"] = dict(F_e_asc=float(r["F_e_asc"]), sd_asc=float(r["sd_asc"]), F_e_frm=float(r["F_e_frm"]),
                           sd_frm=float(r["sd_frm"]), delta=float(r["delta"]), sd_delta=sd)
    r152 = AB.r152_main_fe()
    if r152:
        out["R152"] = r152
    return out


def _reading(z: float, value: float, floor: float = 0.0) -> str:
    if z >= 3 and value >= floor:
        return "positive"
    if z <= -3:
        return "negative"
    if abs(z) < 2:
        return "null"
    return "none (2 <= |z| < 3, or under the effect floor)"


def verdicts(res: dict, frame: dict | None, refs: dict, v3: dict | None = None) -> dict:
    """the preregistered rules. Every contrast's sd is the LARGER of its paired sd and its frame-counting sd; with no
    frame-counting sd nothing is decided (FAIL-CLOSED, R183's lesson). V1 (PRIMARY, R182): z >= 3 and >= 0.015; V1-null
    |z| < 2; V1-neg z <= -3. Secondary contrasts read positive / negative / null at the same z edges. AT THRESHOLD: a z
    within 3 Monte-Carlo sds of an edge, which only a bootstrap sd carries."""
    frame_sd = None if not frame or frame.get("refused") else frame["sd"]
    out: dict = dict(sd_rule="max(paired sd, frame-counting sd)", contrasts={})
    for k, c in res["contrasts"].items():
        f = None if frame_sd is None else float(frame_sd[k])
        sd = max(c["sd_paired"], f or 0.0)
        z = c["value"] / sd if sd > 0 else float("nan")
        row = dict(value=c["value"], sd_used=sd, z=z, sd_paired=c["sd_paired"], sd_frame_counting=f)
        if f is None:
            row.update(decided=False, reading=None)
        else:
            from_frames = f >= c["sd_paired"]
            z_mc = abs(z) / math.sqrt(2 * (int(frame["n_boot"]) - 1)) if from_frames else 0.0
            row.update(decided=True, sd_from="frame-counting" if from_frames else "paired", z_mc_sd=z_mc,
                       at_threshold=[t for t in AB.Z_THRESHOLDS if abs(z - t) <= 3 * z_mc],
                       reading=_reading(z, c["value"], 0.015 if k == PRIMARY else 0.0))
        out["contrasts"][k] = row
    p = out["contrasts"][PRIMARY]
    if not p["decided"]:
        out.update(decided=False, undecided_because=f"no frame-counting sd ({(frame or {}).get('refused', 'not computed')})",
                   V1=None, V1_null=None, V1_neg=None)
    else:
        out.update(decided=True, V1=p["reading"] == "positive", V1_null=p["reading"] == "null",
                   V1_neg=p["reading"] == "negative")
        out["reading"] = ("V1" if out["V1"] else "V1-null" if out["V1_null"] else "V1-neg" if out["V1_neg"]
                          else "NONE of V1 / V1-null / V1-neg (the preregistration names no reading here)")
    sd = p["sd_used"]
    band = dict(lo=0.030, hi=0.047, inside_2sd=bool(0.030 - 2 * sd <= p["value"] <= 0.047 + 2 * sd))
    if v3 and not v3.get("refused"):
        s = v3["nob_key_wait_us"]["median"] * 1e-6 / R182_WAIT_S
        band["scaled_by_executed_wait"] = dict(factor=s, lo=0.030 * s, hi=0.047 * s,
                                               inside_2sd=bool(0.030 * s - 2 * sd <= p["value"] <= 0.047 * s + 2 * sd))
    out["V4_projection_band"] = band
    arms = res["arms"]
    if "R152" in refs:
        r = refs["R152"]
        d = arms["nob"]["F_e"] - r["F_e"]
        out["V2_drift_nob_vs_R152"] = dict(ref=r["F_e"], diff=d, within_2sd=bool(abs(d) <= 2 * math.hypot(
            arms["nob"]["sd_F_e"], r["sd"])))
    if "R183" in refs:
        r = refs["R183"]
        d = arms["asc"]["F_e"] - r["F_e_asc"]
        out["V2_drift_asc_vs_R183"] = dict(ref=r["F_e_asc"], diff=d, within_2sd=bool(abs(d) <= 2 * math.hypot(
            arms["asc"]["sd_F_e"], r["sd_asc"])))
        c = out["contrasts"]["R183"]
        joint = math.hypot(c["sd_used"], r["sd_delta"])
        w1, w2 = 1 / c["sd_used"] ** 2, 1 / r["sd_delta"] ** 2
        pooled = (w1 * c["value"] + w2 * r["delta"]) / (w1 + w2)
        out["V5_R183_replication"] = dict(R183_delta=r["delta"], R183_sd=r["sd_delta"], this=c["value"],
                                          consistent_2sd=bool(abs(c["value"] - r["delta"]) <= 2 * joint),
                                          pooled=pooled, pooled_sd=(w1 + w2) ** -0.5,
                                          pooled_z=pooled * (w1 + w2) ** 0.5)
    best = max(ARMS, key=lambda a: arms[a]["F_e"])
    out["best_arm"] = dict(arm=best, label=LABELS[best], F_e=arms[best]["F_e"], sd=arms[best]["sd_F_e"])
    if v3 is not None:
        out["V3_schedule"] = v3
    return out


def v3_schedule(loaded: dict, index: dict) -> dict:
    """prereg V3, READ from the executed schedules: every conditional unit's no-barrier key wait at its conditional
    (dynamic_timing.executed_key_wait), the barrier key's wait and ground idle, and the two frame keys' quantum idle.
    PASS when the median no-barrier wait is inside EXECUTED_BAND_S (R182's mechanism is present as played) AND no
    barrier-kept frame key idles more than 0.15 us. Refused when a schedule is missing -- never read as a pass."""
    from oneq.hw import dynamic_timing as DT
    from oneq.hw import freeze as F
    from oneq.hw import timing as TM
    dt = AB._dt(dict(asc=loaded["parts"]["asc"]))
    circs = F.load(loaded["rec"], loaded["blob"], check_sources=False)
    pubs = {p["name"]: p for p in index["pubs"]}
    rows: dict = {a: [] for a in ("nob_wait", "asc_wait", "asc_ground", "frm_idle", "fnb_idle")}
    missing = []
    for n in loaded["parts"]["fnb"]["frames"]:
        got = {a: TM.from_metadata(pubs[f"{a}__{n}"]["metadata"]) for a in ARMS}
        if any(v is None for v in got.values()):
            missing.append(n)
            continue
        qc = circs[f"asc__{n}"]
        key = next(qc.find_bit(inst.qubits[0]).index for inst in qc.data if inst.operation.name == "if_else")
        wn = DT.executed_key_wait(got["nob"], key, dt)
        wa = DT.executed_key_wait(got["asc"], key, dt)
        rows["nob_wait"].append(wn["wait_s"])
        rows["asc_wait"].append(wa["wait_s"])
        rows["asc_ground"].append(wa["ground_s"] or 0.0)
        for a in FRAMED:
            rows[f"{a}_idle"].append(DT.quantum_idle(got[a], dt)["per_qubit"].get(key, {}).get("quantum_s", 0.0))
    if missing:
        return dict(refused=f"{len(missing)} units have no executed schedule", missing=missing[:10])

    def us(xs):
        return dict(min=min(xs) * 1e6, median=float(np.median(xs)) * 1e6, max=max(xs) * 1e6)
    med = float(np.median(rows["nob_wait"]))
    ok_nob = EXECUTED_BAND_S[0] <= med <= EXECUTED_BAND_S[1]
    ok_frm = max(rows["frm_idle"]) <= FRAME_IDLE_MAX_S
    return dict(units=len(rows["nob_wait"]), dt=dt, nob_key_wait_us=us(rows["nob_wait"]),
                asc_key_wait_us=us(rows["asc_wait"]), asc_key_ground_idle_us=us(rows["asc_ground"]),
                frm_key_quantum_idle_us=us(rows["frm_idle"]), fnb_key_quantum_idle_us=us(rows["fnb_idle"]),
                nob_inside_band=sum(EXECUTED_BAND_S[0] <= w <= EXECUTED_BAND_S[1] for w in rows["nob_wait"]),
                nob_wait_as_preregistered=bool(ok_nob), frm_key_does_not_wait=bool(ok_frm),
                PASS=bool(ok_nob and ok_frm), per_unit_nob_wait_s=dict(zip(loaded["parts"]["fnb"]["frames"],
                                                                         rows["nob_wait"])))


def dose_response(res: dict, v3: dict, n_perm: int = 4000) -> dict:
    """SECONDARY (preregistered as reported, not decided): does the per-unit R182 contrast follow the unit's own
    executed no-barrier key wait? Slope and a permutation test of the correlation"""
    if not v3 or v3.get("refused"):
        return dict(refused="no executed schedule")
    m = res["per_unit_means"]
    names = list(v3["per_unit_nob_wait_s"])
    x = np.array([v3["per_unit_nob_wait_s"][n] * 1e6 for n in names])
    y = np.array([m["fnb"][n] - m["nob"][n] for n in names])
    if float(np.std(x)) == 0.0:
        return dict(units=len(names), refused="every unit waited the same: no dose spread to regress on")
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(194)
    perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
    return dict(units=len(names), wait_us=dict(min=float(x.min()), median=float(np.median(x)), max=float(x.max())),
                slope_per_us=float(np.polyfit(x, y, 1)[0]), r=r, perm_p_two_sided=float((np.abs(perm) >= abs(r)).mean()))


def frame_counting(loaded: dict, index: dict, arrays: dict, n_boot: int = 300) -> dict:
    """every contrast's (and every arm's) sd over twirling FRAMES from ONE randomization bootstrap, the shot bootstrap
    of the same statistics, their ratio, and the frame-homogeneity map. A refusal is RETURNED, never swallowed"""
    from oneq.hw import randomization as R
    try:
        shots = R.from_flight(index, arrays)

        def est(counts):
            res = analyse_2x2(loaded["rec"], counts, loaded["parts"])
            return {**{k: c["value"] for k, c in res["contrasts"].items()},
                    **{f"F_e_{a}": v["F_e"] for a, v in res["arms"].items()}}
        fr = R.bootstrap(est, shots, n_boot=n_boot, seed=194, mode="randomization")
        sh = R.bootstrap(est, shots, n_boot=n_boot, seed=194, mode="shot")
        hm = R.homogeneity_map(shots)
    except R.RandomizationRefused as e:
        return dict(refused=f"RandomizationRefused: {e}")
    frames = sorted({(s.R, s.n, s.last) for s in shots.values()}, key=str)
    return dict(sd=fr.sd, n_boot=n_boot, blocks=fr.blocks, scale=fr.scale, shot_sd=sh.sd,
                inflation=R.inflation(fr, sh), frames=[dict(R=r, n=n, last=last) for r, n, last in frames],
                homogeneity=dict((k, hm[k]) for k in ("tests", "threshold", "flagged", "flagged_circuits"))
                | dict(strongest=hm["strongest"][:5]))


def dryrun_counts(loaded: dict, seed: int = 194) -> dict:
    """every circuit of the FROZEN composite sampled from its exact outcome law (no device noise): all four arms must
    give the ideal F_e and every contrast 0 -- the analysis path end to end before any spend"""
    from oneq.hw import freeze as F
    rng = np.random.default_rng(seed)
    circs = F.load(loaded["rec"], loaded["blob"], check_sources=False)
    cache: dict = {}
    out = {}
    for n, qc in circs.items():
        k = law_key(qc)
        if k not in cache:
            cache[k] = AB._law(strip_barriers(qc))     # the same law, computed faster
        out[n] = AB.sample(cache[k], loaded["rec"]["shots"][n], rng)
    return out


def full_analysis(loaded: dict, index: dict, arrays: dict, counts: dict) -> dict:
    """everything the preregistration asks of a fetched job, from the flight record alone"""
    res = analyse_2x2(loaded["rec"], counts, loaded["parts"])
    res["frame_counting"] = frame_counting(loaded, index, arrays)
    try:
        v3 = v3_schedule(loaded, index)
    except ValueError as e:
        v3 = dict(refused=str(e))
    res["verdicts"] = verdicts(res, res["frame_counting"], references(), v3)
    res["dose_response"] = dose_response(res, v3)
    return res


def reanalyse(rp: pathlib.Path, loaded: dict) -> pathlib.Path:
    """the saved flight record and counts through today's `full_analysis`, into reanalysis_<job>.json beside the result
    written at fetch; the counts must be exactly the flight record's shots"""
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
    res["reanalysed_from"] = dict(job=jid, flight=f"flight_{jid}.json", counts=f"counts_{jid}.json")
    dest = rp.parent / f"reanalysis_{jid}.json"
    dest.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    return dest


def _summary(res: dict) -> dict:
    return dict(arms=res["arms"], contrasts=res["contrasts"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rehearse", "freeze", "dryrun", "check", "submit", "fetch", "reanalyse"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--per-stratum", type=int, default=0, help="the first N main units of each stratum (0: all)")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    if a.mode == "rehearse":
        rec = json.loads((R183_DIR / "r147.json").read_text(encoding="utf-8"))
        blob = (R183_DIR / "r147.qpy").read_bytes()
        out = Rehearsal(rec, blob, a.shots, AB.per_stratum(rec, a.per_stratum)).run(a.p)
        print(json.dumps(_summary(out), indent=1))
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
        print(f"frozen {out['rec']['n_circuits']} circuits "
              f"({ {t: len(r['circuit_order']) for t, (r, _) in out['parts'].items()} }) -> {_rel(rp)}; "
              f"{pl.total_shots} shots, cost estimate {pl.cost_s():.1f} s (x{U.QUOTA_MARGIN} margin: "
              f"{pl.cost_s() * U.QUOTA_MARGIN:.1f} s)")
        return 0
    rp = pathlib.Path(a.record).resolve()
    loaded = load(rp)
    if a.mode == "reanalyse":
        dest = reanalyse(rp, loaded)
        res = json.loads(dest.read_text(encoding="utf-8"))
        print(json.dumps(_summary(res), indent=1))
        print(json.dumps(res["verdicts"], indent=1, default=str))
        print(f"-> {_rel(dest)}")
        return 0
    if a.mode == "dryrun":
        res = analyse_2x2(loaded["rec"], dryrun_counts(loaded), loaded["parts"])
        res.pop("per_unit_means")
        dest = rp.parent / f"dryrun_{rp.stem}.json"
        dest.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(res, indent=1, default=str))
        return 0
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    be = svc.backend(loaded["rec"]["backend"])
    pl = plan(loaded["rec"])
    if a.mode in ("check", "submit"):
        circuits = F.load(loaded["rec"], loaded["blob"])
        bad = submit_refusals(loaded, circuits)
        live = S.from_backend(be)
        health = sorted({r for qc in circuits.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        remaining = U.remaining_quota_s(svc)
        if a.mode == "check":
            pre = U.preflight(pl, loaded["rec"], loaded["blob"], authorized_flag=a.i_understand_this_spends_credits,
                              remaining_s=remaining, health_refusals=health)
            print(json.dumps(dict(stored_calibration=loaded["rec"].get("calibrated_utc"),
                                  current_calibration=live.calibrated_utc, remaining_quota_s=remaining,
                                  circuits=len(circuits), total_shots=pl.total_shots, cost_s=pl.cost_s(),
                                  cost_with_margin_s=pl.cost_s() * U.QUOTA_MARGIN, lane_refusals=bad,
                                  preflight_refusals=pre), indent=1, default=str))
            return 0
        if bad:
            raise SystemExit("REFUSED: " + "; ".join(bad))
        job = U.submit(pl, loaded["rec"], loaded["blob"], be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=remaining, health_refusals=health)
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
    print(json.dumps(_summary(res), indent=1, default=str))
    print(json.dumps(res["verdicts"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
