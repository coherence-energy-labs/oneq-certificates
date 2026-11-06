"""R161: is a CZ worse when its NEIGHBOURS are not idle? -- and everything else one job on these 14 qubits can tell.

R152's per-unit attribution (r152_attribution.py --edges) found two laddered edges costing ~10x their own same-job
ladder INSIDE the protocol: 41-42 (+11.4 +- 2.1) and 148-149 (+9.4 +- 1.6), while 22-23 is consistent with its ladder
(+1.5 +- 0.95). A ladder parks every neighbour in |0>; a protocol does not.

LADDERS (the question). Each edge's full learnable CZ channel (oneq.hw.cz_channel via hw_diag_run.fit_cz_channels,
R146's estimator unchanged) in four ARMS that differ ONLY in the edges' coupled neighbours (the spectators):
  0    spectators in |0>                        the ordinary ladder
  1    spectators in |1>                        a static shift (a spectator's ZZ moves the CZ's conditional phase)
  +    spectators in |+>                        superposition, the protocol's situation
  +e   spectators in |+>, an X at mid-ladder    the same, ECHOED: a coherent spectator phase cancels, a decoherence
                                                does not -- so if + is worse, +e says whether dynamical decoupling of
                                                idle neighbours would fix it
All three edges run in parallel in every circuit (far apart; spectator sets disjoint), 7 product bases x depths 0, 16,
48; twirled CZs, 8 randomizations, as R152 and R155 ran.

SAME-JOB CALIBRATION (everything else, on every used qubit, in the job's own calibration window):
  ro_all0 / ro_all1 / ro_alt01 / ro_alt10   readout, and its CROSSTALK (alternating patterns)
  xs_w0 / xs_w1 / xs_w2 / xs_echo           Ramsey at 0, W, 2W and the echo: white rate and quasi-static sigma
                                            (r155_idle_map's estimator) in this window
  xs_w1_y                                   the Y quadrature at W: with xs_w1's X, the SIGN of each qubit's frequency
                                            offset (a frame correction needs it; R155 could not give it)
  zz_{a|s}{0|1}_{x|y}                       static ZZ, sign-resolved: targets in |+>, their partners in |0> or |1>,
                                            the targets' X and Y after W. a: each edge's lower qubit against its
                                            partner; s: every spectator against its edge qubit
RAW IQ (meas_type kerneled) on every shot: bits come from this job's own readout circuits (oneq.hw.iq), and LEAKAGE
(shots far from both clusters, growing with ladder depth) is measured -- bits alone cannot see it.

ONE JOB WITH R158 (PR #45): R158's frozen native sign-Ramsey bundle (evidence/superactivation/r158/kingston_148_149)
is composed INTO this job (oneq.hw.campaign, parts r161 and r158): one calibration window, one options set, R161's IQ
discriminators read R158's q148/q149 too, and R158's own independent analysis runs on its split counts. R161's ZZ
family and ladder arms 0/1 prepare a neighbour's |0> and |1> with R158's MATCHED native pulses (SX.SXdg / SX.SX).

  python hw_r161_spectator_cz.py freeze --fake | --live B       read-only; writes a job folder job_<stamp>/
  python hw_r161_spectator_cz.py dryrun JOB [--inject X] [--leak L]
        Aer on FakeKingston, then SYNTHETIC IQ drawn from the simulated bits (the IQ path end to end). --inject X:
        the TARGET edges' CZ error x X in arms + only (a positive control for S1); --leak L: a fraction L per CZ of
        the target edge qubits' shots moved to a third IQ cluster in arm + (a positive control for leakage)
  python hw_r161_spectator_cz.py analyse JOB                    from the flight record in the job folder
  submit / fetch go through oneq.hw.submit behind every guard (an OWNER act).
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
import hw_diag_run as D146  # noqa: E402
import hw_r155_diag as R155  # noqa: E402
import r155_idle_map as IM  # noqa: E402
from oneq.hw import cz_channel as CC  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import iq as IQ  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

TARGETS = ((41, 42), (148, 149))
CONTROLS = ((22, 23),)
ARMS = ("0", "1", "+", "+e")
DEPTHS = D146.LADDER_DEPTHS
W_DT = R155.W_DT
ECHO_DT = R155.ECHO_DT
SHOTS = dict(pb=1000, ro=3672, xs=2000, zz=2000)
RANDOMIZATIONS = 8
MEAS_TYPE = "kerneled"
PREREG = ROOT / "docs" / "PREREG_r161_spectator_cz.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "r161_spectator_cz"
TAG = "r161_spectator_cz"
Z_SIGNAL, Z_NULL = 3.0, 2.0                    # the preregistered decision thresholds (docs/PREREG_r161...)


def arm_tag(arm: str) -> str:
    return {"0": "s0", "1": "s1", "+": "sp", "+e": "se"}[arm]


# ------------------------------------------------------------------ design
def spectators(nb: dict, edges) -> dict:
    """each edge's coupled neighbours (of either qubit, not the edge itself); refused when two edges' qubit and
    spectator sets touch -- the arms must differ ONLY in each edge's own neighbours"""
    out, used = {}, set()
    for a, b in edges:
        sp = sorted((nb.get(a, set()) | nb.get(b, set())) - {a, b})
        block = {a, b, *sp}
        if block & used:
            raise ValueError(f"edge {(a, b)}'s block {sorted(block)} touches another edge's")
        used |= block
        out[(a, b)] = sp
    return out


def spectator_partner(nb: dict, edges, spect: dict) -> dict:
    """each spectator's partner for the zz family: the edge qubit it is coupled to (the lower one if both)"""
    out = {}
    for a, b in edges:
        for s in spect[(a, b)]:
            out[s] = a if a in nb.get(s, set()) else b
    return out


def _z_prep(qc, qubits, z: int) -> None:
    """|0> as SX then SXdg, |1> as SX then SX: two physical pulses either way, the states differing only in a
    virtual-Z frame (R158's matched native preparation). An X against nothing would let the drive itself -- its
    crosstalk, its heating -- masquerade as the neighbour's state."""
    if not qubits:
        return
    qc.sx(qubits)
    if z:
        qc.sx(qubits)
    else:
        qc.sxdg(qubits)


def _new(be, n, name):
    from qiskit import ClassicalRegister
    qc = R155._wide(be.num_qubits)
    qc.add_register(ClassicalRegister(n, "c"))
    qc.name = name
    return qc


def circuits(be, edges, spect: dict, partner: dict | None = None) -> list:
    """(name, circuit, meta) for every family. Clbits: each edge's two qubits first (lower-numbered qubit first, edges
    in order), then every spectator -- the ladder fit reads the edge bits exactly as R146's pb circuits lay them out,
    and every circuit reads every used qubit (raw IQ for all of them, always)."""
    eq = [q for e in edges for q in e]
    sq = [q for e in edges for q in spect[e]]
    used = eq + sq
    n = len(used)
    out = []
    # readout and its crosstalk: all 0, all 1, and the two alternating patterns over the used qubits
    for name, pattern in (("ro_all0", [0] * n), ("ro_all1", [1] * n), ("ro_alt01", [i % 2 for i in range(n)]),
                          ("ro_alt10", [1 - i % 2 for i in range(n)])):
        qc = _new(be, n, name)
        ones = [q for q, v in zip(used, pattern) if v]
        if ones:
            qc.x(ones)
        qc.measure(used, range(n))
        out.append((name, qc, dict(family="ro", pattern=pattern, qubits=used, shots=SHOTS["ro"])))
    # Ramsey at 0, W, 2W, the echo, and the Y quadrature at W, every used qubit at once
    for name, wait, echo, yq in (("xs_w0", 0, False, False), ("xs_w1", W_DT, False, False),
                                 ("xs_w2", 2 * W_DT, False, False), ("xs_echo", ECHO_DT, True, False),
                                 ("xs_w1_y", W_DT, False, True)):
        qc = _new(be, n, name)
        qc.h(used)
        qc.barrier(used)
        if echo:
            qc.delay(wait // 2, used, unit="dt")
            qc.barrier(used)
            qc.x(used)
            qc.barrier(used)
            qc.delay(wait // 2, used, unit="dt")
        elif wait:
            qc.delay(wait, used, unit="dt")
        qc.barrier(used)
        if yq:
            qc.sdg(used)
        qc.h(used)
        qc.measure(used, range(n))
        out.append((name, qc, dict(family="xs", wait_dt=wait, echo=echo, quadrature="y" if yq else "x",
                                   shots=SHOTS["xs"])))
    # static ZZ, sign-resolved: targets in |+>, partners in |0> or |1>, targets' X or Y after W
    partner = partner or {}
    for side in ("a", "s"):
        targets = [e[0] for e in edges] if side == "a" else sq
        partners = [e[1] for e in edges] if side == "a" else sorted({partner[s] for s in sq})
        for z in (0, 1):
            for quad in ("x", "y"):
                qc = _new(be, n, f"zz_{side}{z}_{quad}")
                _z_prep(qc, partners, z)
                qc.h(targets)
                qc.barrier(used)
                qc.delay(W_DT, used, unit="dt")
                qc.barrier(used)
                if quad == "y":
                    qc.sdg(targets)
                qc.h(targets)
                qc.measure(used, range(n))
                out.append((qc.name, qc, dict(family="zz", side=side, partner_state=z, quadrature=quad,
                                              targets=targets, partners=partners, shots=SHOTS["zz"])))
    # the ladders, four arms
    for arm in ARMS:
        for basis in CC.BASES:
            for d in DEPTHS:
                qc = _new(be, n, f"{arm_tag(arm)}_{basis}_d{d}")
                for i, q in enumerate(eq):
                    letter = basis[i % 2]
                    if letter in "XY":
                        qc.h(q)
                    if letter == "Y":
                        qc.s(q)
                if arm in ("0", "1"):
                    _z_prep(qc, sq, int(arm))                # matched pulses: arms 0 and 1 differ in virtual Z only
                elif arm in ("+", "+e"):
                    qc.h(sq)
                qc.barrier(used)
                for k in range(d):
                    for a, b in edges:
                        qc.cz(a, b)
                    qc.barrier(used)
                    if arm == "+e" and k == d // 2 - 1:
                        # the echo, after the first half of the layers. NO barrier after it: one would hold the edge
                        # qubits ~36 ns that no other arm holds them, and the arms must differ only on the spectators
                        # -- so the X runs on the spectators during the next CZ layer (the echo point is mid-layer)
                        qc.x(sq)
                for i, q in enumerate(eq):
                    letter = basis[i % 2]
                    if letter == "Y":
                        qc.sdg(q)
                    if letter in "XY":
                        qc.h(q)
                if arm in ("+", "+e"):
                    qc.h(sq)
                qc.measure(used, range(n))
                out.append((qc.name, qc, dict(family="pb", arm=arm, basis=basis, depth=d, shots=SHOTS["pb"])))
    # END-OF-JOB sentinels (last in the circuit order): the readout clusters at the job's end against its start
    for name, prep in (("ro_end0", 0), ("ro_end1", 1)):
        qc = _new(be, n, name)
        if prep:
            qc.x(used)
        qc.measure(used, range(n))
        out.append((name, qc, dict(family="ro", pattern=[prep] * n, qubits=used, shots=SHOTS["ro"], sentinel=True)))
    return out


def freeze_r161(be, snap, prereg_sha):
    from qiskit import transpile
    edges = [tuple(sorted(e)) for e in TARGETS + CONTROLS]
    for e in edges:
        snap.edge(*e)                                    # SnapshotError names an uncalibrated edge
    nb = R155.graph(snap)
    spect = spectators(nb, edges)
    partner = spectator_partner(nb, edges, spect)
    named, meta = [], []
    for name, qc, m in circuits(be, edges, spect, partner):
        tqc = transpile(qc, be, initial_layout=list(range(be.num_qubits)), optimization_level=0, seed_transpiler=1)
        named.append((name, tqc))
        meta.append(dict(name=name, n_bits=tqc.num_clbits, **m))
    bad = sorted({r for _, tqc in named for r in L.refusals(snap, tqc, L.QubitPolicy())})
    rec, blob = F.freeze(named, dict(
        protocol="R161: CZ channel per edge against its spectators' state (|0>, |1>, |+>, |+> echoed), with a "
                 "same-job calibration block, sign-resolved ZZ and raw IQ", backend=snap.name,
        snapshot=snap.to_json(), snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
        targets=[list(e) for e in TARGETS], controls=[list(e) for e in CONTROLS], edges=[list(e) for e in edges],
        spectators={f"{a}_{b}": spect[(a, b)] for a, b in edges}, spectator_partner={str(k): v for k, v in
                                                                                      partner.items()},
        arms=list(ARMS), depths=list(DEPTHS), w_dt=W_DT, echo_dt=ECHO_DT, circuits=meta,
        randomizations=RANDOMIZATIONS, meas_type=MEAS_TYPE, iq_calibration=["ro_all0", "ro_all1"],
        policy_refusals=bad, prereg=PREREG.name, prereg_sha256=prereg_sha, z_signal=Z_SIGNAL, z_null=Z_NULL,
        sources=[str(HERE / "hw_r161_spectator_cz.py"), str(HERE / "hw_diag_run.py"), str(HERE / "hw_r155_diag.py"),
                 str(HERE / "r155_idle_map.py")]
        + [str(p) for p in sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))
    return rec, blob


# ------------------------------------------------------------------ analysis (pure)
def _used(rec):
    return next(c["qubits"] for c in rec["circuits"] if c["name"] == "ro_all0")


def _p1(cnt: dict, n: int, i: int) -> float:
    tot = sum(cnt.values())
    return sum(v for k, v in cnt.items() if k.zfill(n)[n - 1 - i] == "1") / tot


def readout(rec: dict, counts: dict) -> dict:
    """per used qubit, P(read 1 | prep 0) and P(read 0 | prep 1) from the all-0 and all-1 circuits"""
    qubits = _used(rec)
    n = len(qubits)
    return {q: dict(p01=_p1(counts["ro_all0"], n, i), p10=1 - _p1(counts["ro_all1"], n, i))
            for i, q in enumerate(qubits)}


def readout_crosstalk(rec: dict, counts: dict, ro: dict) -> dict:
    """per qubit, its assignment error when its neighbours alternate, minus the all-same error for the same state"""
    qubits = _used(rec)
    n = len(qubits)
    out = {}
    for i, q in enumerate(qubits):
        row = {}
        for name in ("ro_alt01", "ro_alt10"):
            pat = next(c["pattern"] for c in rec["circuits"] if c["name"] == name)
            p1 = _p1(counts[name], n, i)
            err = p1 if pat[i] == 0 else 1 - p1
            row[name] = err - (ro[q]["p01"] if pat[i] == 0 else ro[q]["p10"])
        out[str(q)] = row
    return out


def edge_counts(counts: dict, n_edge_bits: int) -> dict:
    """the histogram marginalized to the edge bits (clbits 0 .. n_edge_bits - 1: the rightmost characters)"""
    out: dict = {}
    for k, v in counts.items():
        kk = k[-n_edge_bits:] if len(k) >= n_edge_bits else k.zfill(n_edge_bits)
        out[kk] = out.get(kk, 0) + v
    return out


def channels(rec: dict, counts: dict, ro: dict) -> dict:
    """arm -> edge -> R146's fit (p, p_sd, learnable quantities and sds, chi2) on that arm's ladders"""
    edges = [tuple(e) for e in rec["edges"]]
    nbits = 2 * len(edges)
    out = {}
    for arm in rec["arms"]:
        sub = {f"pb0_{b}_d{d}": edge_counts(counts[f"{arm_tag(arm)}_{b}_d{d}"], nbits)
               for b in CC.BASES for d in rec["depths"]}
        fit = D146.fit_cz_channels(dict(ladder_rounds=[[list(e) for e in edges]], snapshot=rec["snapshot"]), sub, ro)
        out[arm] = {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in fit.items()}
    return out


def _contrast(x: dict, base: dict) -> dict:
    diff = x["p"] - base["p"]
    sd = math.hypot(x["p_sd"], base["p_sd"])
    return dict(p=x["p"], p0=base["p"], diff=diff, sd=sd, z=diff / sd if sd > 0 else float("nan"),
                ratio=x["p"] / base["p"] if base["p"] > 0 else float("nan"))


def contrasts(ch: dict, edges: list) -> dict:
    """per edge, each arm's total CZ error against arm 0, and the echoed arm against the unechoed one"""
    out = {}
    for a, b in edges:
        key = f"{a}_{b}"
        row = {arm: _contrast(ch[arm][key], ch["0"][key]) for arm in ch if arm != "0"}
        if "+e" in ch and "+" in ch:
            row["+e_vs_+"] = _contrast(ch["+e"][key], ch["+"][key])
        out[key] = row
    return out


def verdicts(rec: dict, con: dict) -> dict:
    zs, zn = rec["z_signal"], rec["z_null"]
    tk = [f"{a}_{b}" for a, b in rec["targets"]]
    ck = [f"{a}_{b}" for a, b in rec["controls"]]
    arms = [a for a in ("1", "+", "+e") if a in next(iter(con.values()))]
    ctrl_quiet = all(abs(con[k][arm]["z"]) < zn for k in ck for arm in arms)
    s1 = [k for k in tk if con[k]["+"]["z"] > zs]
    s2 = [k for k in tk if con[k]["1"]["z"] > zs]
    s3 = all(abs(con[k][arm]["z"]) < zn for k in tk for arm in ("1", "+"))
    out = dict(S1_superposed_spectators_raise_the_CZ_error=dict(edges=s1, met=bool(s1) and ctrl_quiet),
               S2_excited_spectators_raise_the_CZ_error=dict(edges=s2, met=bool(s2) and ctrl_quiet),
               S3_no_spectator_dependence=dict(met=s3), control_quiet=ctrl_quiet)
    if "+e" in arms:                                     # S4 is read only where S1 holds
        coherent = [k for k in s1 if abs(con[k]["+e"]["z"]) < zn and con[k]["+e_vs_+"]["z"] < -zs]
        incoherent = [k for k in s1 if abs(con[k]["+e_vs_+"]["z"]) < zn]
        out["S4_the_echo_removes_it_coherent"] = dict(edges=coherent, met=bool(coherent) and ctrl_quiet)
        out["S4_the_echo_leaves_it_incoherent"] = dict(edges=incoherent, met=bool(incoherent) and ctrl_quiet)
    return out


def _xy(counts: dict, n: int, i: int, sa: float) -> float:
    return (1 - 2 * _p1(counts, n, i)) / sa


def ramsey(rec: dict, counts: dict, ro: dict) -> dict:
    """per used qubit: contrast ratios at W, 2W and the echo, the idle map's (G, sigma) from them (r155_idle_map's
    estimator), and the frequency offset's SIGNED angle at W from the X and Y quadratures"""
    qubits = _used(rec)
    n = len(qubits)
    dt = rec["snapshot"]["dt"]
    W, E = rec["w_dt"] * dt, rec["echo_dt"] * dt
    out = {}
    for i, q in enumerate(qubits):
        sa = 1 - ro[q]["p01"] - ro[q]["p10"]
        c = {k: _xy(counts[k], n, i, sa) for k in ("xs_w0", "xs_w1", "xs_w2", "xs_echo", "xs_w1_y")}
        n_xs = sum(counts["xs_w1"].values())
        sd = math.sqrt(max(1 - c["xs_w1"] ** 2, 1e-4) / n_xs) / max(abs(c["xs_w0"]), 1e-3)
        row = dict(ratio_w1=c["xs_w1"] / c["xs_w0"], ratio_w2=c["xs_w2"] / c["xs_w0"],
                   ratio_echo=c["xs_echo"] / c["xs_w0"], sd_ratio_w1=sd, sd_ratio_echo=sd,
                   t2_vendor_s=rec["snapshot"]["t2"][q])
        # <X> = r cos(phi), <Y> = r sin(phi) with phi = Delta W: the SIGNED offset (aliased beyond |Delta W| < pi)
        phi = math.atan2(c["xs_w1_y"], c["xs_w1"])
        row.update(phase_w_rad=phi, detuning_hz=phi / W / (2 * math.pi),
                   contrast_w=math.hypot(c["xs_w1"], c["xs_w1_y"]) / max(abs(c["xs_w0"]), 1e-3),
                   idle_map=IM.fit_qubit(row, W, E))
        out[str(q)] = row
    return out


def zz(rec: dict, counts: dict, ro: dict) -> dict:
    """per (target, partner): the target's phase after W with the partner in |0> and in |1>; their difference over W
    is the static ZZ conditional-phase rate, SIGN included (the target's own offset cancels in the difference)"""
    qubits = _used(rec)
    n = len(qubits)
    pos = {q: i for i, q in enumerate(qubits)}
    dt = rec["snapshot"]["dt"]
    W = rec["w_dt"] * dt
    out = {}
    for side in ("a", "s"):
        meta = next(c for c in rec["circuits"] if c["name"] == f"zz_{side}0_x")
        pairs = list(zip(meta["targets"], meta["partners"])) if side == "a" else \
            [(int(s), rec["spectator_partner"][str(s)]) for s in meta["targets"]]
        for t, p in pairs:
            i = pos[t]
            sa = 1 - ro[t]["p01"] - ro[t]["p10"]
            ph = {}
            for z in (0, 1):
                x = _xy(counts[f"zz_{side}{z}_x"], n, i, sa)
                y = _xy(counts[f"zz_{side}{z}_y"], n, i, sa)
                ph[z] = (math.atan2(y, x), math.hypot(x, y))
            d = (ph[1][0] - ph[0][0] + math.pi) % (2 * math.pi) - math.pi
            out[f"{t}_{p}"] = dict(phase0=ph[0][0], phase1=ph[1][0], contrast0=ph[0][1], contrast1=ph[1][1],
                                   zeta_rad_s=d / W, zeta_over_2pi_khz=d / W / (2 * math.pi) / 1e3)
    return out


def analyse_counts(rec: dict, counts: dict) -> dict:
    ro = readout(rec, counts)
    ch = channels(rec, counts, ro)
    con = contrasts(ch, [tuple(e) for e in rec["edges"]])
    return dict(schema="oneq.r161.analysis/2", readout={str(q): v for q, v in ro.items()},
                readout_crosstalk=readout_crosstalk(rec, counts, ro), channels=ch, contrasts=con,
                ramsey=ramsey(rec, counts, ro), zz=zz(rec, counts, ro), verdicts=verdicts(rec, con))


def leakage(rec: dict, frac: dict) -> dict:
    """per edge qubit and arm, the outlier fraction against ladder depth (averaged over the 7 bases), its per-CZ slope,
    and the calibration circuits' own outlier floor"""
    edges = [tuple(e) for e in rec["edges"]]
    floor = {q: float(np.mean([frac[n][q] for n in ("ro_all0", "ro_all1")])) for q in frac["ro_all0"]}
    out: dict = {}
    for a, b in edges:
        for q in (str(a), str(b)):
            row = {}
            for arm in rec["arms"]:
                pts = [float(np.mean([frac[f"{arm_tag(arm)}_{bs}_d{d}"][q] for bs in CC.BASES]))
                       for d in rec["depths"]]
                ds = np.array(rec["depths"], float)
                slope = float(np.polyfit(ds, pts, 1)[0])
                row[arm] = dict(fraction_by_depth=pts, per_cz=slope)
            out[q] = dict(floor=floor[q], arms=row)
    return out


def iq_report(rec: dict, iq_by_name: dict, qubits_by_name: dict, discs: dict, prefix: str = "") -> dict:
    """R161's IQ report from a job's IQ (names carry `prefix` inside a composite): outlier fractions, leakage against
    depth, the readout drift from the job's start to its end, and the assignment error at the end"""
    mine = {n[len(prefix):]: v for n, v in iq_by_name.items() if n.startswith(prefix)}
    qbn = {n[len(prefix):]: v for n, v in qubits_by_name.items() if n.startswith(prefix)}
    frac = IQ.outlier_fractions(mine, qbn, discs)
    report = dict(discriminators={str(q): d.to_json() for q, d in discs.items()}, outlier_fractions=frac,
                  leakage=leakage(rec, frac))
    if "ro_end0" in mine:                                # start against end of the job, per prepared state
        for prep in (0, 1):
            pos = IQ.drift(mine, qbn, discs, [f"ro_all{prep}", f"ro_end{prep}"])
            report[f"drift_prep{prep}_sd"] = {q: v[1] - v[0] for q, v in pos.items()}
        ends = IQ.classify_all({k: mine[k] for k in ("ro_end0", "ro_end1")}, qbn, discs)
        n = len(_used(rec))
        report["end_assignment_error"] = {str(q): dict(p01=_p1(ends["ro_end0"], n, i),
                                                        p10=1 - _p1(ends["ro_end1"], n, i))
                                          for i, q in enumerate(_used(rec))}
    return report


def analyse(rec: dict, counts: dict, iq_rep: dict | None = None) -> dict:
    out = analyse_counts(rec, counts)
    if iq_rep is not None:
        out["iq"] = iq_rep
    return out


# ------------------------------------------------------------------ the JOB: R161 and R158 in one submission
R158_DIR = ROOT / "evidence" / "superactivation" / "r158" / "kingston_148_149"
PART = "r161"


def r158_part():
    """R158's frozen bundle as a part of R161's job. Loaded WITH its source check: the bundle must still name the
    code that built it. R158 has no two-qubit gate, so the job's gate twirling leaves its circuits unchanged (only
    its shots are grouped into randomizations)."""
    from oneq.hw import campaign as C
    rec = json.loads((R158_DIR / "frozen.json").read_text(encoding="utf-8"))
    blob = (R158_DIR / "circuits.qpy").read_bytes()
    F.load(rec, blob)
    return C.Part("r158", rec, blob, dict(rec["shots"]), RANDOMIZATIONS, rec["prereg_path"], rec["prereg_sha256"],
                  str(R158_DIR), "r158", dict(note="no two-qubit gate: gate twirling leaves its circuits unchanged"))


def compose_job(rec161: dict, blob161: bytes, prereg_sha: str) -> tuple:
    from oneq.hw import campaign as C
    p161 = C.Part(PART, rec161, blob161, {c["name"]: c["shots"] for c in rec161["circuits"]}, RANDOMIZATIONS,
                  str(PREREG), prereg_sha, str(OUTDIR), TAG)
    return C.compose([p161, r158_part()], dict(
        protocol="R161 (spectator-state CZ, maximum information) with R158 (native sign Ramsey) inside one job",
        backend=rec161["backend"], snapshot=rec161["snapshot"], snapshot_sha256=rec161["snapshot_sha256"],
        calibrated_utc=rec161["calibrated_utc"], prereg=PREREG.name, prereg_sha256=prereg_sha, meas_type=MEAS_TYPE,
        iq_calibration=[f"{PART}__ro_all0", f"{PART}__ro_all1"],
        sources=[str(HERE / "hw_r161_spectator_cz.py"), str(ROOT / "src" / "oneq" / "hw" / "campaign.py"),
                 str(ROOT / "src" / "oneq" / "hw" / "iq.py")]))


def plan_job(crec: dict) -> U.Plan:
    return U.Plan(tag=TAG, backend=crec["backend"], shots=dict(crec["shots"]), prereg_path=str(PREREG),
                  prereg_sha256=crec["prereg_sha256"], twirl_randomizations=crec["randomizations"],
                  records_dir=str(OUTDIR), meas_type=crec["meas_type"], iq_calibration=tuple(crec["iq_calibration"]),
                  measurement_reason="preregistered DIAGNOSTICS (R161 and R158); nothing about superactivation is "
                                     "tested")


def analyse_job(crec: dict, rec161: dict, rec158: dict, iq_by_name: dict, qubits_by_name: dict) -> tuple:
    """every shot of both parts classified by R161's discriminators (q148 and q149 are in its readout block), the
    counts split by part, R161's analysis and IQ report, and R158's OWN independent analysis on its counts"""
    from oneq.hw import campaign as C
    sys.path.insert(0, str(ROOT / "tools"))
    import analyze_r158_independent as A158
    p0, p1 = crec["iq_calibration"]
    discs = IQ.calibrate(iq_by_name, qubits_by_name, p0, p1)
    counts = IQ.classify_all(iq_by_name, qubits_by_name, discs)
    parts = C.split(crec, counts)
    out = dict(schema="oneq.r161.job_analysis/1",
               r161=analyse(rec161, parts[PART], iq_report(rec161, iq_by_name, qubits_by_name, discs, f"{PART}__")))
    try:
        out["r158"] = A158.primary(rec158, parts["r158"])
    except A158.AnalysisRefused as e:                    # recorded, never hidden: R158's own refusal is a result
        out["r158"] = dict(refused=str(e))
    out["r158_outlier_fractions"] = IQ.outlier_fractions(
        {n: v for n, v in iq_by_name.items() if n.startswith("r158__")},
        {n: v for n, v in qubits_by_name.items() if n.startswith("r158__")}, discs)
    return counts, out


# ------------------------------------------------------------------ modes
def _injected_backend(factor: float):
    """FakeKingston, and a copy with the TARGET edges' CZ error multiplied by `factor` (a positive control)"""
    import copy

    from qiskit_ibm_runtime.fake_provider import FakeKingston
    be = FakeKingston()
    be2 = copy.deepcopy(be)
    for a, b in TARGETS:
        for qa in ((a, b), (b, a)):
            if qa in be2.target["cz"]:
                p = be2.target["cz"][qa]
                p.error = min(0.5, p.error * factor)
    return be, be2


def synthetic_iq(rec: dict, counts: dict, qubits_by_name: dict, leak_per_cz: float, seed: int = 161,
                 prefix: str = "") -> dict:
    """per circuit, (shots, clbits) complex IQ drawn from its simulated histogram: each qubit its own two Gaussian
    clusters; with leak_per_cz > 0, in R161's arms + and +e (names under `prefix`) a fraction leak_per_cz x depth of
    each TARGET edge qubit's shots is moved to a third cluster beyond |1> -- the leakage positive control. Shot order
    is a random permutation."""
    rng = np.random.default_rng(seed)
    targets = {q for e in rec["targets"] for q in e}
    geo = {}
    for q in sorted({q for qs in qubits_by_name.values() for q in qs}):
        ang = rng.uniform(0, 2 * np.pi)
        mu0 = 1e7 * np.exp(1j * ang)
        geo[q] = (mu0, mu0 + 4e7 * np.exp(1j * (ang + 1.0)), 0.6e7)
    meta = {prefix + c["name"]: c for c in rec["circuits"]}
    out = {}
    for name, cnt in counts.items():
        qs = qubits_by_name[name]
        n = len(qs)
        rows = [k.zfill(n) for k, v in cnt.items() for _ in range(v)]
        rng.shuffle(rows)
        bits = np.array([[r[n - 1 - j] == "1" for j in range(n)] for r in rows])
        z = np.empty(bits.shape, complex)
        m = meta.get(name, {})
        for j, q in enumerate(qs):
            mu0, mu1, sig = geo[q]
            mu = np.where(bits[:, j], mu1, mu0)
            if leak_per_cz and m.get("family") == "pb" and m["arm"] in ("+", "+e") and q in targets:
                leak = rng.random(len(mu)) < leak_per_cz * m["depth"]
                mu = np.where(leak, mu1 + 2.5 * (mu1 - mu0), mu)
            z[:, j] = mu + sig * (rng.normal(size=len(mu)) + 1j * rng.normal(size=len(mu)))
        out[name] = z
    return out


def _qubits_by_name(rec, blob) -> dict:
    circs = F.load(rec, blob, check_sources=False)
    return {n: IQ.measured_qubits(qc) for n, qc in circs.items()}


def _summary(out: dict) -> str:
    r = out["r161"]
    keep = dict(contrasts=r["contrasts"], verdicts=r["verdicts"])
    if "iq" in r:
        keep["leakage_per_cz"] = {q: {a: round(v["per_cz"], 6) for a, v in x["arms"].items()}
                                  for q, x in r["iq"]["leakage"].items()}
    keep["r158"] = {k: v for k, v in out.get("r158", {}).items() if k in ("refused", "status", "decision", "verdict")}
    return json.dumps(keep, indent=1, default=str)


def _load_job(folder: pathlib.Path) -> dict:
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    out = dict(crec=load(folder / "job.json"), cblob=(folder / "job.qpy").read_bytes(), rec161=load(folder / "r161.json"),
               blob161=(folder / "r161.qpy").read_bytes(), rec158=load(R158_DIR / "frozen.json"),
               blob158=(R158_DIR / "circuits.qpy").read_bytes())
    from oneq.hw import campaign as C
    C.verify_parts(out["crec"], [C.Part(PART, out["rec161"], out["blob161"], {}, None),
                                 C.Part("r158", out["rec158"], out["blob158"], {}, None)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["freeze", "dryrun", "analyse", "submit", "fetch"])
    ap.add_argument("job", nargs="?", help="the job folder a freeze wrote (job.json, job.qpy, r161.json, r161.qpy)")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--inject", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=0.0)
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
        sha = U.file_sha256(PREREG)
        rec161, blob161 = freeze_r161(be, S.from_backend(be), sha)
        crec, cblob = compose_job(rec161, blob161, sha)
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = OUTDIR / f"job_{stamp}"
        d.mkdir()
        for name, obj, blob in (("r161", rec161, blob161), ("job", crec, cblob)):
            (d / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
            (d / f"{name}.qpy").write_bytes(blob)
        pl = plan_job(crec)
        print(f"frozen ONE job of {crec['n_circuits']} circuits (R161 {rec161['n_circuits']} + R158 "
              f"{crec['n_circuits'] - rec161['n_circuits']}) -> {d.name}; {pl.total_shots} shots, cost estimate "
              f"{pl.cost_s():.1f} s (x{U.QUOTA_MARGIN}: {pl.cost_s() * U.QUOTA_MARGIN:.0f} s); meas_type "
              f"{pl.meas_type}; spectators {rec161['spectators']}; policy refusals {rec161['policy_refusals'][:4]}")
        return 0
    folder = pathlib.Path(a.job).resolve()
    J = _load_job(folder)
    crec, cblob = J["crec"], J["cblob"]
    if a.mode == "dryrun":
        be, be_inj = _injected_backend(a.inject)
        shots = plan_job(crec).shots
        counts = U.local_run(crec, cblob, be, shots)
        if a.inject != 1.0:                              # R161's arms + and +e only, on the injected device
            plus = U.local_run(crec, cblob, be_inj, shots)
            counts.update({k: v for k, v in plus.items() if k.startswith((f"{PART}__sp_", f"{PART}__se_"))})
        qbn = _qubits_by_name(crec, cblob)
        iq = synthetic_iq(J["rec161"], counts, qbn, a.leak, prefix=f"{PART}__")
        _, out = analyse_job(crec, J["rec161"], J["rec158"], iq, qbn)
        out["dryrun"] = dict(inject=a.inject, leak_per_cz=a.leak, simulator="aer NoiseModel.from_backend(FakeKingston)",
                             iq="synthetic, drawn from the simulated bits")
        tag = ("" if a.inject == 1.0 else f"_inject{a.inject:g}") + ("" if not a.leak else f"_leak{a.leak:g}")
        (folder / f"dryrun{tag}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                                 newline="\n")
        print(_summary(out))
        return 0
    jobs = [j for j in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json")))
            if j.get("qpy_sha256") == crec["qpy_sha256"]]
    if a.mode == "analyse":
        if len(jobs) != 1:
            raise SystemExit(f"REFUSED: {len(jobs)} job records for this frozen job; expected one")
        jid = jobs[0]["job_id"]
        index = json.loads((folder / f"flight_{jid}.json").read_text(encoding="utf-8"))
        arrays = dict(np.load(folder / f"flight_{jid}.npz"))
        qbn = _qubits_by_name(crec, cblob)
        iq = {n: IQ.iq_columns(index, arrays, n) for n in crec["circuit_order"]}
        _, out = analyse_job(crec, J["rec161"], J["rec158"], iq, qbn)
        (folder / f"result_{jid}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                                  newline="\n")
        print(_summary(out))
        return 0
    from qiskit_ibm_runtime import QiskitRuntimeService

    from oneq.hw import campaign as C
    svc = QiskitRuntimeService()
    be = svc.backend(crec["backend"])
    pl = plan_job(crec)
    if a.mode == "submit":
        if "fake" in crec["backend"].lower() or folder.name.startswith("job_fake"):
            raise SystemExit("REFUSED: a --fake freeze is a rehearsal; freeze --live first")
        live = S.from_backend(be)
        circs = F.load(crec, cblob)
        health = sorted({r for qc in circs.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(pl, crec, cblob, be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        return 0
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records for this frozen job; expected one")
    jid = jobs[0]["job_id"]
    job = svc.job(jid)
    if "DONE" not in str(job.status()):
        print(f"job {jid}: {job.status()}")
        return 4
    result = job.result()
    # the flight record FIRST: every shot's IQ, the spans, the job's metadata -- whatever the analysis does next
    index, arrays = C.flight_record(result, crec)
    index["job_metrics"] = C._jsonable(job.metrics()) if hasattr(job, "metrics") else None
    np.savez_compressed(folder / f"flight_{jid}.npz", **arrays)
    (folder / f"flight_{jid}.json").write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8",
                                              newline="\n")
    (folder / f"after_{jid}.snapshot.json").write_text(json.dumps(S.from_backend(be).to_json(), indent=1) + "\n",
                                                      encoding="utf-8", newline="\n")
    qbn = _qubits_by_name(crec, cblob)
    iq = {n: IQ.iq_columns(index, arrays, n) for n in crec["circuit_order"]}
    counts, out = analyse_job(crec, J["rec161"], J["rec158"], iq, qbn)
    (folder / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    (folder / f"result_{jid}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                              newline="\n")
    print(_summary(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
