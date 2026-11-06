"""R176 -- every committed twirled verdict re-read with error bars that count the twirl's frames (OFFLINE, 2026-10-17).

R169 found that on 148-149 the shot-level error bars of R146's CZ-channel estimator were 1.4-2.7x too small: the R
randomizations of a twirled circuit are R frames, and a coherent error that the twirl turns into a Pauli channel only
on average spreads the frame means beyond what the shots inside them show. This lane asks the same question of EVERY
twirled verdict whose shots are committed, through ONE canonical instrument (`oneq.hw.randomization`) and each
protocol's OWN estimator, unchanged:

  R155  db9dsqpgv2cc73e5c8fg  B (a read and a feed-forward against distant and neighbouring qubits), C1 (spectator Z
                              per CZ), D1 (early - inline). Its ro / xs / B circuits carry NO two-qubit gate, so their
                              8 frames are the SAME circuit: a hardware NULL for the instrument (inflation ~1)
  R161  db9q3bs24mrc7384ecn0  S1-S3 on the three edges, the IQ re-classified by the job's own discriminators
  R152  db9c50oc77vs73817rog  R146's F_e, R147's F_e and per-kind means (the model regression's per-kind misses),
                              R147's V3 and V4

For each statistic: the estimate (equal to the recorded one: checked, not assumed), its sd by the SHOT bootstrap
(what counts alone can say) and by the RANDOMIZATION bootstrap (what the frames add), and every recorded verdict
re-evaluated with the frame sd by the verdict's own rule. Nothing is fitted and no device model changes.

Speed, with the arithmetic untouched:
  - fit_cz_channels draws PB_DRAWS slope sets per edge for p_sd only; p does not depend on them (checked on the full
    data: p with 2 draws == p with 400, exactly). The bootstrap replaces p_sd, so it runs with 2.
  - R155's analysis recomputes its chip-wide A family every call; A is not audited, its circuits are not resampled,
    and their bit matrices are memoized by object identity (the cache holds each dict, so no id can be reused).

  python r176_frame_bars.py [--boot N] [--only r155|r161|r152] -> evidence/superactivation/r176_frame_bars/<job>.json
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

from oneq.hw import randomization as RZ  # noqa: E402

EV = ROOT / "evidence" / "superactivation"
OUT = EV / "r176_frame_bars"
R155_DIR = EV / "r155_diag"
R155_JOB, R155_RECORD = "db9dsqpgv2cc73e5c8fg", "frozen_20261017T023024Z"
R161_DIR = EV / "r161_spectator_cz" / "job_20261017T124413Z"
R161_JOB = "db9q3bs24mrc7384ecn0"
R152_DIR = EV / "campaign_r152" / "20261017T003152Z"
R152_JOB = "db9c50oc77vs73817rog"
N_BOOT = 300


class AuditRefused(ValueError):
    """the audit cannot reproduce a recorded number, so it cannot re-read its error bar"""


def _json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ================================================================== the instrument
def audit(estimator, shots: dict, n_boot: int, seed: int) -> dict:
    """every statistic of `estimator` with its shot sd, frame sd and their ratio (oneq.hw.randomization)"""
    fr = RZ.bootstrap(estimator, shots, n_boot=n_boot, seed=seed, mode="randomization")
    sh = RZ.bootstrap(estimator, shots, n_boot=n_boot, seed=seed + 1, mode="shot")
    infl = RZ.inflation(fr, sh)
    return dict(n_boot=n_boot, blocks=fr.blocks, scale=fr.scale,
                stats={k: dict(estimate=fr.estimate[k], sd_shot=sh.sd[k], sd_frames=fr.sd[k], inflation=infl[k])
                       for k in fr.estimate})


def _key_bits(s: RZ.Shots) -> np.ndarray:
    """(keys, clbits) 0/1, clbit 0 first (a key's RIGHTMOST character)"""
    return np.array([[int(ch) for ch in k[::-1]] for k in s.keys], dtype=np.int8)


def hamming_inflation(shots: dict, n_boot: int, seed: int, drop: dict | None = None) -> dict:
    """per circuit, the frame/shot sd ratio of its mean Hamming fraction over its clbits less those in drop[name]
    (circuits whose shots never vary skipped)"""
    out = {}
    for j, (name, s) in enumerate(sorted(shots.items())):
        keep = np.ones(len(s.keys[0]), bool)
        for c in (drop or {}).get(name, ()):
            keep[c] = False
        if not keep.any():
            continue
        weight = dict(zip(s.keys, (_key_bits(s)[:, keep].sum(axis=1) / keep.sum()).tolist()))

        def est(c, name=name, weight=weight):
            cc = c[name]
            return sum(v * weight[k] for k, v in cc.items()) / sum(cc.values())
        fr = RZ.bootstrap(est, {name: s}, n_boot=n_boot, seed=seed + 2 * j, mode="randomization")
        sh = RZ.bootstrap(est, {name: s}, n_boot=n_boot, seed=seed + 2 * j + 1, mode="shot")
        if sh.sd["value"] > 0:
            out[name] = fr.sd["value"] / sh.sd["value"]
    return out


def group_inflation(per_circuit: dict, two_qubit_gates: dict) -> dict:
    """the per-circuit ratios pooled (rms) by whether the circuit has a two-qubit gate: a frame can differ from
    another only through a twirled two-qubit gate, so circuits with none are the null"""
    out = {}
    for label, keep in (("no_two_qubit_gate", lambda n: two_qubit_gates[n] == 0),
                        ("with_two_qubit_gates", lambda n: two_qubit_gates[n] > 0)):
        v = np.array([x for n, x in per_circuit.items() if keep(n)])
        if v.size:
            out[label] = dict(circuits=int(v.size), rms=float(np.sqrt(np.mean(v ** 2))), median=float(np.median(v)),
                              max=float(v.max()), above_1_5=int((v > 1.5).sum()))
    return out


def two_qubit_gates(circuits: dict) -> dict:
    return {n: sum(1 for i in qc.data if i.operation.num_qubits == 2) for n, qc in circuits.items()}


FLAG_Z = 4.0


def frame_position(shots: dict, qubits_by_name: dict, names) -> dict:
    """per frame INDEX, over the named circuits and every clbit: the z of that frame's P(1) against the circuit's
    other frames (binomial), as an rms, a mean and a count beyond FLAG_Z -- plus every flagged (circuit, qubit,
    frame). A frame index is a position in the sweep, not a Pauli frame, in a circuit with no two-qubit gate: every
    index runs the same circuit, so a position effect (what ran just before, a reset that carries it) is all that
    can make one index differ."""
    zs: dict = {}
    flagged = []
    for name in names:
        s = shots[name]
        if s.R < 2:
            continue
        K = _key_bits(s).astype(float)
        per = np.stack([np.bincount(s.code[r * s.n:(r + 1) * s.n], minlength=len(s.keys)) for r in range(s.R)])
        p = per @ K / s.n                                           # (frames, clbits) P(1)
        pool = p.mean(axis=0)
        sd = np.sqrt(np.clip(pool * (1 - pool), 1e-6, None) * (1 / s.n + 1 / (s.n * (s.R - 1))))
        for r in range(s.R):
            rest = np.delete(p, r, axis=0).mean(axis=0)
            z = (p[r] - rest) / sd
            zs.setdefault(r, []).extend(z.tolist())
            for j in np.flatnonzero(np.abs(z) > FLAG_Z):
                flagged.append(dict(circuit=name, qubit=int(qubits_by_name[name][j]), clbit=int(j), frame=r,
                                    z=float(z[j]), p1_frame=float(p[r, j]), p1_other_frames=float(rest[j])))
    per_index = {str(r): dict(n=len(v), rms_z=float(np.sqrt(np.mean(np.square(v)))), mean_z=float(np.mean(v)),
                              beyond=int((np.abs(np.array(v)) > FLAG_Z).sum())) for r, v in sorted(zs.items())}
    return dict(flag_z=FLAG_Z, per_frame_index=per_index, flagged=flagged,
                flagged_qubits=sorted({f["qubit"] for f in flagged}))


def null_control(shots: dict, qubits_by_name: dict, g2: dict, n_boot: int, seed: int) -> dict:
    """the hardware null: circuits with no two-qubit gate run ONE circuit in all their frames, in the same sweeps
    (R168), so their frame/shot ratio must read ~1. Read per circuit, pooled by two-qubit gates; the frame-position
    test on the null circuits; and the null again with every flagged qubit's clbit dropped"""
    null_names = [n for n in shots if g2[n] == 0]
    per = hamming_inflation(shots, n_boot, seed)
    pos = frame_position(shots, qubits_by_name, null_names)
    drop = {n: [j for j, q in enumerate(qubits_by_name[n]) if q in pos["flagged_qubits"]] for n in shots}
    out = dict(per_circuit=per, by_two_qubit_gates=group_inflation(per, g2), frame_position=pos)
    if pos["flagged_qubits"]:
        per_d = hamming_inflation(shots, n_boot, seed, drop=drop)
        out["without_flagged_qubits"] = dict(dropped=pos["flagged_qubits"], by_two_qubit_gates=group_inflation(per_d, g2))
    return out


def _z(est: float, sd: float, ref: float = 0.0) -> float:
    return (est - ref) / sd if sd > 0 else float("nan")


def z_mc_sd(z: float, n_boot: int) -> float:
    """the Monte-Carlo sd of a z whose sd came from n_boot resamples: an sd read off B draws carries a relative
    error of 1 / sqrt(2 (B - 1)), and z = estimate / sd inherits it"""
    return abs(z) / math.sqrt(2 * (n_boot - 1))


def threshold_call(z: float, threshold: float, n_boot: int) -> dict:
    """|z| against a verdict threshold: the margin, its Monte-Carlo sd, and whether the side is DECIDED (the margin
    exceeds three Monte-Carlo sds). A verdict whose governing z sits within that band is reported as AT THRESHOLD,
    never as met or failed -- the data cannot carry either."""
    mc = z_mc_sd(z, n_boot)
    margin = threshold - abs(z)
    return dict(z=z, threshold=threshold, margin=margin, z_mc_sd=mc, decided=bool(abs(margin) > 3 * mc),
                side="below" if margin > 0 else "at_or_above")


# ================================================================== R155
class _BitsMemo:
    """R155's _bits, memoized for the dicts registered as FIXED. The cache HOLDS each registered dict, so while it
    lives no other object can carry its id: a key hit is that very object (an equal dict is another object, and is
    recomputed)."""

    def __init__(self, inner):
        self.inner, self.cache = inner, {}

    def register(self, d: dict, n: int) -> None:
        self.cache[(id(d), n)] = (d, self.inner(d, n))

    def __call__(self, counts, n):
        hit = self.cache.get((id(counts), n))
        return hit[1] if hit is not None else self.inner(counts, n)


def r155_statistics(a: dict) -> dict:
    out = {}
    for s, b in a["B"].items():
        for k in ("far_readff_over_read", "far_read_over_noread", "neighbour_read_over_noread"):
            out[f"B{s}|{k}"] = b[k]
    for s, c in a["C"].items():
        out[f"C{s}|spectator_z_per_cz"] = c["spectator_z_per_cz"]
    out["D|early_minus_inline"] = a["D"]["early_minus_inline"]
    return out


def r155_verdicts(stats: dict, recorded: dict, n_boot: int) -> dict:
    """B2 (PREREG_r155: read_ff / read < 1 - 2 sd at a site) with the propagated shot sd the analysis implies and with
    the frame sd; D1 with its recorded across-twin sd (already empirical) beside the frame sd"""
    b2 = {}
    for k, v in stats.items():
        if k.endswith("far_readff_over_read"):
            b2[k.split("|")[0]] = dict(ratio=v["estimate"], z_shot=_z(v["estimate"], v["sd_shot"], 1.0),
                                       z_frames=_z(v["estimate"], v["sd_frames"], 1.0),
                                       met_shot=v["estimate"] < 1 - 2 * v["sd_shot"],
                                       met_frames=v["estimate"] < 1 - 2 * v["sd_frames"],
                                       call_frames=threshold_call(_z(v["estimate"], v["sd_frames"], 1.0), 2.0, n_boot))
    d = stats["D|early_minus_inline"]
    return dict(B2=b2, B2_sites_met_shot=sum(x["met_shot"] for x in b2.values()),
                B2_sites_met_frames=sum(x["met_frames"] for x in b2.values()),
                D1=dict(estimate=d["estimate"], recorded_sd=recorded["D"]["early_minus_inline_sd"],
                        sd_frames=d["sd_frames"], z_frames=_z(d["estimate"], d["sd_frames"])))


def run_r155(n_boot: int) -> dict:
    import hw_r155_diag as R155

    from oneq.hw import freeze as F
    rec = _json(R155_DIR / f"{R155_RECORD}.json")
    index = _json(R155_DIR / f"flight_{R155_JOB}.json")
    arrays = dict(np.load(R155_DIR / f"flight_{R155_JOB}.npz"))
    counts = _json(R155_DIR / f"counts_{R155_JOB}.json")
    recorded = _json(R155_DIR / f"result_{R155_JOB}.json")
    shots = RZ.from_flight(index, arrays)
    bad = [n for n, s in shots.items() if s.counts() != {k: int(v) for k, v in counts[n].items()}]
    if bad:
        raise AuditRefused(f"R155: the flight record's shots do not give the committed counts for {bad[:3]}")
    circs = F.load(rec, (R155_DIR / f"{R155_RECORD}.qpy").read_bytes(), check_sources=False)
    g2 = two_qubit_gates(circs)
    fixed = {n: counts[n] for n in counts if not (n.startswith("B") or n.startswith("C") or n.startswith("D"))}
    resampled = {n: s for n, s in shots.items() if n not in fixed}
    memo = _BitsMemo(R155._bits)
    n_chip = len(rec["chip_qubits"])
    for n, d in fixed.items():
        memo.register(d, n_chip)
    inner, R155._bits = R155._bits, memo
    try:
        def est(c):
            return r155_statistics(R155.analyse(rec, {**fixed, **c}))
        full = est({n: s.counts() for n, s in resampled.items()})
        want = r155_statistics(recorded)
        off = {k: (full[k], want[k]) for k in want if abs(full[k] - want[k]) > 1e-12}
        if off:
            raise AuditRefused(f"R155: the audit's estimator does not reproduce the record: {off}")
        table = audit(est, resampled, n_boot, seed=155)
    finally:
        R155._bits = inner
    from oneq.hw import iq as IQ
    null = null_control(shots, {n: IQ.measured_qubits(qc) for n, qc in circs.items()}, g2, max(60, n_boot // 3), 1550)
    return dict(job=R155_JOB, record=R155_RECORD, resampled=sorted(resampled), fixed_not_resampled=sorted(fixed),
                audit=table, verdicts=r155_verdicts(table["stats"], recorded, n_boot),
                null_control=null)


# ================================================================== R161
def r161_shots(index: dict, arrays: dict, crec: dict, circs: dict) -> dict:
    """composite name -> Shots: every shot's IQ classified by the job's OWN discriminators (R161's analyse_job)"""
    from oneq.hw import iq as IQ
    qbn = {n: IQ.measured_qubits(qc) for n, qc in circs.items()}
    iq = {n: IQ.iq_columns(index, arrays, n) for n in crec["circuit_order"]}
    discs = IQ.calibrate(iq, qbn, *crec["iq_calibration"])
    pub = {p["name"]: p["index"] for p in index["pubs"]}
    out = {}
    for n in crec["circuit_order"]:
        bits = np.stack([IQ.classify(discs[q], iq[n][:, j]) for j, q in enumerate(qbn[n])], axis=1)
        R, _ = RZ.shape(index["execution_spans"], pub[n], len(bits))
        out[n] = RZ.from_bits(bits, R)
    return out


def r161_statistics(rec161: dict, ch: dict) -> dict:
    out = {}
    for a, b in (tuple(e) for e in rec161["edges"]):
        key = f"{a}_{b}"
        for arm in ch:
            out[f"p|{arm}|{key}"] = ch[arm][key]["p"]
            if arm != "0":
                out[f"diff|{arm}|{key}"] = ch[arm][key]["p"] - ch["0"][key]["p"]
        if "+e" in ch and "+" in ch:
            out[f"diff|+e_vs_+|{key}"] = ch["+e"][key]["p"] - ch["+"][key]["p"]
    return out


def r161_verdicts(rec161: dict, stats: dict, recorded_contrasts: dict, n_boot: int) -> dict:
    """R161's own verdict rule (hw_r161_spectator_cz.verdicts) on contrasts whose z uses the SHOT and the FRAME sd.
    S3 (every target's arm-1 and arm-+ |z| below z_null) is governed by the largest of those |z|: its call says
    whether that z is decided against z_null at this many resamples."""
    import hw_r161_spectator_cz as R161
    out = {}
    targets = [f"{a}_{b}" for a, b in rec161["targets"]]
    for which in ("sd_shot", "sd_frames"):
        con = {}
        for k, v in stats.items():
            if not k.startswith("diff|"):
                continue
            _, arm, edge = k.split("|")
            con.setdefault(edge, {})[arm] = dict(z=_z(v["estimate"], v[which]))
        out[which] = R161.verdicts(rec161, con)
        out[which + "_z"] = {e: {arm: round(x["z"], 3) for arm, x in row.items()} for e, row in con.items()}
        gov = max(((e, arm) for e in targets for arm in ("1", "+")), key=lambda t: abs(con[t[0]][t[1]]["z"]))
        out[which + "_S3_call"] = dict(governed_by=f"{gov[1]}|{gov[0]}",
                                       **threshold_call(con[gov[0]][gov[1]]["z"], rec161["z_null"], n_boot))
    out["recorded_z"] = {e: {arm: round(x["z"], 3) for arm, x in row.items()} for e, row in recorded_contrasts.items()}
    return out


def run_r161(n_boot: int) -> dict:
    import hw_diag_run as D146
    import hw_r161_spectator_cz as R161

    from oneq.hw import cz_channel as CC
    from oneq.hw import freeze as F
    index = _json(R161_DIR / f"flight_{R161_JOB}.json")
    arrays = dict(np.load(R161_DIR / f"flight_{R161_JOB}.npz"))
    crec = _json(R161_DIR / "job.json")
    rec161 = _json(R161_DIR / "r161.json")
    counts = _json(R161_DIR / f"counts_{R161_JOB}.json")
    recorded = _json(R161_DIR / f"result_{R161_JOB}.json")["r161"]
    circs = F.load(crec, (R161_DIR / "job.qpy").read_bytes(), check_sources=False)
    shots = r161_shots(index, arrays, crec, circs)
    bad = [n for n, s in shots.items() if s.counts() != {k: int(v) for k, v in counts[n].items()}]
    if bad:
        raise AuditRefused(f"R161: re-classified shots do not give the committed counts for {bad[:3]}")
    pre = R161.PART + "__"
    part = {n[len(pre):]: c for n, c in counts.items() if n.startswith(pre)}
    ro = R161.readout(rec161, part)                     # held fixed, as R169 did: the bootstrap is of the ladders
    ladders = [pre + f"{R161.arm_tag(arm)}_{b}_d{d}" for arm in rec161["arms"] for b in CC.BASES
               for d in rec161["depths"]]
    resampled = {n: shots[n] for n in ladders}
    full_ch = R161.channels(rec161, part, ro)
    draws = D146.PB_DRAWS
    D146.PB_DRAWS = 2
    try:
        def est(c):
            return r161_statistics(rec161, R161.channels(rec161, {n[len(pre):]: v for n, v in c.items()}, ro))
        full = est({n: s.counts() for n, s in resampled.items()})
        want = r161_statistics(rec161, full_ch)
        off = {k: (full[k], want[k]) for k in want if full[k] != want[k]}
        if off:
            raise AuditRefused(f"R161: p with 2 slope draws differs from p with {draws}: {list(off)[:3]}")
        rec_p = {f"p|{arm}|{e}": v["p"] for arm, row in recorded["channels"].items() for e, v in row.items()}
        off = {k: (full[k], rec_p[k]) for k in rec_p if abs(full[k] - rec_p[k]) > 1e-12}
        if off:
            raise AuditRefused(f"R161: the audit's p does not reproduce the record: {list(off)[:3]}")
        table = audit(est, resampled, n_boot, seed=161)
    finally:
        D146.PB_DRAWS = draws
    from oneq.hw import iq as IQ
    null = null_control(shots, {n: IQ.measured_qubits(qc) for n, qc in circs.items()}, two_qubit_gates(circs),
                        max(60, n_boot // 3), 1610)
    return dict(job=R161_JOB, resampled=len(resampled), readout_fixed=True, audit=table,
                verdicts=r161_verdicts(rec161, table["stats"], recorded["contrasts"], n_boot),
                null_control=null)


# ================================================================== R152
def r152_estimator(art: dict):
    """counts (composite names) -> R146's measured F_e (hw_diag_run.analyse's own tallies and HS.estimate), R147's
    measured F_e, per-kind mean, V3 and V4 measured values (hw_dynamic_run.analyse, unchanged)"""
    import hw_diag_run as D146
    import hw_dynamic_run as R147
    r146, r147 = art["parts"]["r146"], art["parts"]["r147"]
    units = r146["units"]
    stat = [D146.T.Unit(u["name"], u["weight"], u["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    scores = {u["name"]: D146.H.score(r146["m"], D146.HS.unit_from(u["unit"])) for u in units}

    def est(c):
        t = D146.T.tally_all(stat, {u["name"]: c["r146__" + u["name"]] for u in units}, scores)
        means = {k: x.mean for k, x in t.items()}
        var = {k: max(x.s2 / x.n - x.mean ** 2, 0.0) for k, x in t.items()}
        f146 = D146.HS.estimate(means, units, var, {k: x.n for k, x in t.items()})["F_e"]
        a = R147.analyse(r147, {n: c["r147__" + n] for n in r147["circuit_order"]})
        out = dict(r146_F_e=f146, r147_F_e=a["measured"]["F_e"],
                   V3_early_minus_inline=a["early_minus_inline"]["measured"],
                   V4_dynamic_minus_static=a["dynamic_minus_static"]["measured"])
        kinds: dict = {}
        for v in a["per_unit"].values():
            kinds.setdefault(v["kind"], []).append(v["measured"])
        out.update({f"r147_mean|{k}": float(np.mean(v)) for k, v in kinds.items()})
        return out
    return est


def r152_verdicts(stats: dict, recorded: dict, regression: dict, n_boot: int) -> dict:
    """R147's V4 (dynamic - static - 2 sd > 0) and V3 (|measured - predicted| <= 2 sd) by their own rules, and the
    model regression's v2 per-kind misses in units of each kind's frame sd"""
    r147 = recorded["r147"]
    v4, v3 = stats["V4_dynamic_minus_static"], stats["V3_early_minus_inline"]
    pred3 = r147["early_minus_inline"]["predicted"]
    v2 = next(v for k, v in regression["variants"].items() if k.startswith("v2"))
    kinds = dict(main="main", inline="twin_inline", static="twin_static")
    return dict(
        V4=dict(measured=v4["estimate"], recorded_sd=r147["dynamic_minus_static"]["measured_sd"],
                sd_shot=v4["sd_shot"], sd_frames=v4["sd_frames"], recorded_met=r147["verdicts"]["V4_dynamic_beats_static"],
                met_frames=bool(v4["estimate"] - 2 * v4["sd_frames"] > 0),
                call_frames=threshold_call(_z(v4["estimate"], v4["sd_frames"]), 2.0, n_boot)),
        V3=dict(measured=v3["estimate"], predicted=pred3, recorded_sd=r147["early_minus_inline"]["measured_sd"],
                sd_frames=v3["sd_frames"], recorded_met=r147["verdicts"]["V3_idle_accounting"],
                met_frames=bool(abs(v3["estimate"] - pred3) <= 2 * v3["sd_frames"])),
        regression_v2=dict(
            r146=dict(miss=v2["r146_miss"], sd_regression=regression["sd"]["r146"], sd_frames=stats["r146_F_e"]["sd_frames"],
                      z_frames=_z(v2["r146_miss"], stats["r146_F_e"]["sd_frames"])),
            r147=dict(miss=v2["r147_miss"], sd_frames=stats["r147_F_e"]["sd_frames"],
                      z_frames=_z(v2["r147_miss"], stats["r147_F_e"]["sd_frames"])),
            **{name: dict(miss=v2["residual"][k], sd_regression=regression["sd"][f"r147_{name}"],
                          sd_frames=stats[f"r147_mean|{k}"]["sd_frames"],
                          z_regression=_z(v2["residual"][k], regression["sd"][f"r147_{name}"]),
                          z_frames=_z(v2["residual"][k], stats[f"r147_mean|{k}"]["sd_frames"]))
               for name, k in kinds.items()}))


def run_r152(n_boot: int) -> dict:
    import r152_posthoc as P152
    art = P152.load(R152_DIR)
    index = dict(art["index"], execution_spans=art["spans"])
    names = ["r146__" + u["name"] for u in art["parts"]["r146"]["units"]] + \
            ["r147__" + n for n in art["parts"]["r147"]["circuit_order"]]
    shots = RZ.from_flight(index, art["arrays"], names=names)
    bad = [n for n, s in shots.items() if s.counts() != {k: int(v) for k, v in art["counts"][n].items()}]
    if bad:
        raise AuditRefused(f"R152: the flight record's shots do not give the committed counts for {bad[:3]}")
    est = r152_estimator(art)
    full = est({n: s.counts() for n, s in shots.items()})
    res = art["result"]
    want = dict(r146_F_e=res["r146"]["held_out"]["p0"]["F_e_measured"], r147_F_e=res["r147"]["measured"]["F_e"],
                V3_early_minus_inline=res["r147"]["early_minus_inline"]["measured"],
                V4_dynamic_minus_static=res["r147"]["dynamic_minus_static"]["measured"])
    off = {k: (full[k], v) for k, v in want.items() if abs(full[k] - v) > 1e-12}
    if off:
        raise AuditRefused(f"R152: the audit's estimator does not reproduce the record: {off}")
    table = audit(est, shots, n_boot, seed=152)
    regression = _json(ROOT / "evidence" / "hw" / "model_regression.json")
    return dict(job=R152_JOB, resampled=len(shots), audit=table,
                recorded_shot_sd=dict(r146_F_e=res["r146"]["held_out"]["p0"]["shot_sd_F_e"],
                                      r147_F_e=res["r147"]["measured"]["sd_F_e"]),
                verdicts=r152_verdicts(table["stats"], res, regression, n_boot))


RUNS = dict(r155=run_r155, r161=run_r161, r152=run_r152)


def _default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--only", choices=sorted(RUNS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, run in RUNS.items():
        if a.only and name != a.only:
            continue
        rec = dict(schema="oneq.r176.frame_bars/1", label="EXPLORATORY -- re-reads recorded verdicts; fits nothing",
                   protocol=name, **run(a.boot))
        p = OUT / f"{name}_{rec['job']}.json"
        p.write_text(json.dumps(rec, indent=1, default=_default) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p}", flush=True)
        print(json.dumps(rec["verdicts"], indent=1, default=_default)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
