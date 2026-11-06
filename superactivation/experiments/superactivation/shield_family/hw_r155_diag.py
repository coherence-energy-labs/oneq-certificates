"""R155: one diagnostic job that MEASURES what R152 and R153 could only infer (ibm_kingston, free quota).

Families (all twirled CZs, 8 randomizations, measurement twirling and DD off -- the options R152 ran with):
  A  CHIP-WIDE in-circuit calibration, every policy-admissible qubit read at once:
       ro_all0 / ro_all1                 readout as the protocol reads it
       xs_w0 / xs_w1 / xs_w2             Ramsey in |+> at waits 0, W, 2W (W = R152's 816 dt): TWO waits separate an
                                         exponential decay (T2), a Gaussian one (quasi-static noise) and a detuning
                                         (contrast rising again at 2W); and the coupled pairs' R_ij at two waits
     -> an in-circuit T2 map of the chip, so the next layout is chosen on what the qubits do IN a circuit (R153: q149
        reads 3 us in circuit against a vendor 460 us).
  B  MID-CIRCUIT READ SPECTATORS, at SITES sites. Read qubit q in |+>, its coupled neighbours n and a far qubit f
     (graph distance >= 3) in |+>; a barrier; then on q one of
       noread   a delay of q's read duration (the control)
       read     a mid-circuit measurement
       read_ff  a mid-circuit measurement and an if_else on its bit (X on q)
     while n and f wait W/4; a barrier; n and f rotated back and read. The closing barrier makes n and f wait for q's
     block to finish, so f's contrast in read_ff against read measures the FEED-FORWARD LATENCY (through f's own
     in-circuit dephasing, family A), and n's excess over f measures what a read does to its NEIGHBOURS (mechanism B).
  C  CZ SPECTATOR LADDERS, at SITES sites. Edge (a, b) in |0>, a coupled neighbour c of a (not of b) in |+>; k CZs on
     (a, b), k = 0, 16, 48, against the same circuit with each CZ replaced by a delay of the CZ's duration. The
     per-CZ excess decay of c's X contrast IS the spectator error R153 fitted (0.0107) but no ladder measured.
  D  GENUINE EARLY/INLINE TWINS: R147's 16 twin units rebuilt with the barrier that makes the schedules different
     circuits (R153), on R147's own sector-0 layout. R152's V3 tested one circuit twice; this tests the schedules.

  python hw_r155_diag.py freeze --fake | --live B      read-only
  python hw_r155_diag.py dryrun RECORD                  Aer on FakeKingston (MPS for the wide family)
  python hw_r155_diag.py analyse RECORD COUNTS          pure arithmetic
  submit / fetch go through oneq.hw.submit behind every guard (an OWNER act), as every earlier run.
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
import hw_dynamic as D  # noqa: E402
import hw_dynamic_run as R147  # noqa: E402
import hw_faithful as FA  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402
from oneq.hw import submit as U  # noqa: E402

W_DT = 816
ECHO_DT = 2 * 16 * round(W_DT / 32)   # the echo's total wait: two halves, each a whole multiple of 16 dt (832 for W 816)
SITES = 3
LADDER_K = (0, 16, 48)
SHOTS = {"A": 2000, "B": 2000, "C": 2000, "D": 400}
RANDOMIZATIONS = 8
PREREG = ROOT / "docs" / "PREREG_r155_diagnostic.md"
OUTDIR = ROOT / "evidence" / "superactivation" / "r155_diag"
R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
TAG = "r155_diag"


# ------------------------------------------------------------------ the device graph
def graph(snap: S.DeviceSnapshot) -> dict:
    nb: dict = {}
    for a, b in snap.cz:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def admissible(snap: S.DeviceSnapshot, q: int, pol=L.QubitPolicy()) -> bool:
    return (snap.complete(q) and snap.readout_error(q) <= pol.max_readout and snap.t1[q] >= pol.min_t1
            and snap.t2[q] >= pol.min_t2 and snap.sx_error[q] <= pol.max_sx)


def distances(nb: dict, src: int) -> dict:
    out, frontier = {src: 0}, [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in nb.get(u, ()):
                if v not in out:
                    out[v] = out[u] + 1
                    nxt.append(v)
        frontier = nxt
    return out


def pick_sites(snap, n_sites, used=frozenset()):
    """B sites: a read qubit with >= 2 admissible neighbours and an admissible far qubit at distance 3..5; C sites: an
    edge (a, b) inside the CZ policy and a neighbour c of a not adjacent to b. Greedy on readout, disjoint."""
    nb = graph(snap)
    pol = L.QubitPolicy()
    ok = [q for q in range(snap.num_qubits) if admissible(snap, q) and q not in used]
    taken = set(used)
    b_sites, c_sites = [], []
    for q in sorted(ok, key=snap.readout_error):
        ns = [n for n in nb.get(q, ()) if admissible(snap, n) and n not in taken]
        if len(ns) < 2 or q in taken:
            continue
        dist = distances(nb, q)
        far = [f for f in ok if f not in taken and 3 <= dist.get(f, 99) <= 5
               and not any(f in nb.get(n, ()) for n in ns)]
        if not far:
            continue
        f = min(far, key=snap.readout_error)
        b_sites.append(dict(q=q, n=sorted(ns), f=f))
        taken |= {q, f, *ns} | {x for x in nb.get(q, ())} | {x for x in nb.get(f, ())}
        if len(b_sites) == n_sites:
            break
    for (a, b), (err, _) in sorted(snap.cz.items(), key=lambda kv: kv[1][0]):
        if len(c_sites) == n_sites:
            break
        for aa, bb in ((a, b), (b, a)):
            if err > pol.max_cz or aa in taken or bb in taken or not (admissible(snap, aa) and admissible(snap, bb)):
                continue
            cs = [c for c in nb.get(aa, ()) if c != bb and c not in taken and admissible(snap, c)
                  and c not in nb.get(bb, ())]
            if cs:
                c = min(cs, key=snap.readout_error)
                c_sites.append(dict(a=aa, b=bb, c=c))
                taken |= {aa, bb, c} | nb.get(aa, set()) | nb.get(bb, set()) | nb.get(c, set())
                break
    return b_sites, c_sites


# ------------------------------------------------------------------ circuits (on PHYSICAL qubits of the device)
def _wide(nq):
    from qiskit import QuantumCircuit, QuantumRegister
    return QuantumCircuit(QuantumRegister(nq, "q"))


def family_a(be, qubits: list) -> list:
    from qiskit import ClassicalRegister
    out = []
    n = len(qubits)
    for name, prep in (("ro_all0", 0), ("ro_all1", 1)):
        qc = _wide(be.num_qubits)
        qc.add_register(ClassicalRegister(n, "c"))
        qc.name = name
        if prep:
            qc.x(qubits)
        qc.measure(qubits, range(n))
        out.append((name, qc, dict(family="A", kind="ro", prepared=prep, qubits=qubits)))
    for name, wait in (("xs_w0", 0), ("xs_w1", W_DT), ("xs_w2", 2 * W_DT)):
        qc = _wide(be.num_qubits)
        qc.add_register(ClassicalRegister(n, "c"))
        qc.name = name
        qc.h(qubits)
        qc.barrier(qubits)
        if wait:
            qc.delay(wait, qubits, unit="dt")
            qc.barrier(qubits)
        qc.h(qubits)
        qc.measure(qubits, range(n))
        out.append((name, qc, dict(family="A", kind="xs", wait_dt=wait, qubits=qubits)))
    # ECHO (local session, 2026-10-17; docs/R152_R155_COORDINATION.md): H, W/2, X, W/2, H. A static detuning refocuses
    # through the X, white dephasing does not -- so echo against Ramsey at the same total wait separates the two, per
    # qubit, and gives the in-circuit ECHO T2 the transferring model V1s found to be the right idle number. Each half is
    # a whole multiple of 16 dt (Heron delays), so the total is ECHO_DT, compared with xs_w1 as a RATE per unit time.
    qc = _wide(be.num_qubits)
    qc.add_register(ClassicalRegister(n, "c"))
    qc.name = "xs_echo1"
    qc.h(qubits)
    qc.barrier(qubits)
    qc.delay(ECHO_DT // 2, qubits, unit="dt")
    qc.barrier(qubits)
    qc.x(qubits)
    qc.barrier(qubits)
    qc.delay(ECHO_DT // 2, qubits, unit="dt")
    qc.barrier(qubits)
    qc.h(qubits)
    qc.measure(qubits, range(n))
    out.append(("xs_echo1", qc, dict(family="A", kind="echo", wait_dt=ECHO_DT, qubits=qubits)))
    return out


def family_b(be, sites: list) -> list:
    from qiskit import ClassicalRegister
    out = []
    for i, st in enumerate(sites):
        q, ns, f = st["q"], st["n"], st["f"]
        watch = ns + [f]
        for variant in ("noread", "read", "read_ff"):
            qc = _wide(be.num_qubits)
            c = ClassicalRegister(len(watch) + 1, "c")
            qc.add_register(c)
            qc.name = f"B{i}_{variant}"
            qc.h([q] + watch)
            qc.barrier([q] + watch)
            if variant == "noread":
                qc.delay(int(round(be.target["measure"][(q,)].duration / be.dt)), q, unit="dt")
            else:
                qc.measure(q, c[len(watch)])
                if variant == "read_ff":
                    with qc.if_test((c[len(watch)], 1)):
                        qc.x(q)
            qc.delay(W_DT // 4, watch, unit="dt")
            qc.barrier([q] + watch)
            qc.h(watch)
            qc.measure(watch, range(len(watch)))
            if variant == "noread":
                qc.measure(q, c[len(watch)])       # every clbit is written once
            out.append((qc.name, qc, dict(family="B", site=i, variant=variant, q=q, n=ns, f=f, watch=watch)))
    return out


def family_c(be, sites: list) -> list:
    from qiskit import ClassicalRegister
    out = []
    for i, st in enumerate(sites):
        a, b, cq = st["a"], st["b"], st["c"]
        cz_dt = int(round(be.target["cz"][(a, b)].duration / be.dt)) if (a, b) in be.target["cz"] \
            else int(round(be.target["cz"][(b, a)].duration / be.dt))
        for k in LADDER_K:
            for kind in ("cz", "delay"):
                if k == 0 and kind == "delay":
                    continue
                qc = _wide(be.num_qubits)
                qc.add_register(ClassicalRegister(3, "c"))
                qc.name = f"C{i}_{kind}_k{k}"
                qc.h(cq)
                qc.barrier([a, b, cq])
                for _ in range(k):
                    if kind == "cz":
                        qc.cz(a, b)
                    else:
                        qc.delay(cz_dt, [a, b], unit="dt")
                    qc.barrier([a, b, cq])
                qc.h(cq)
                qc.measure([cq, a, b], [0, 1, 2])
                out.append((qc.name, qc, dict(family="C", site=i, kind=kind, k=k, a=a, b=b, c=cq, cz_dt=cz_dt)))
    return out


def family_d(be, snap) -> list:
    rec = json.loads((R152_DIR / "r147.json").read_text(encoding="utf-8"))
    layout0 = rec["layout_sector0"]
    out = []
    twins = [u for u in rec["units"] if u["kind"] == "twin_inline"]
    for k, t in enumerate(twins):
        u = R147.unit_from(t["unit"])
        for early in (True, False):
            qc = D.lean_dynamic(R147.M, u, R147.DECODER, early=early, one_register=True, enforce=True)
            tqc = R147._route(be, snap, qc, layout0, 4000 + 2 * k + early)
            name = f"D{k:02d}_{'early' if early else 'inline'}"
            out.append((name, tqc, dict(family="D", twin=k, early=early, unit=t["unit"],
                                        ideal=FA.lean_ideal(R147.M, u, R147.DECODER))))
    return out


def freeze_r155(be, snap, prereg_sha):
    from qiskit import transpile
    chip = [q for q in range(snap.num_qubits) if admissible(snap, q)]
    rec147 = json.loads((R152_DIR / "r147.json").read_text(encoding="utf-8"))
    b_sites, c_sites = pick_sites(snap, SITES, used=frozenset(rec147["layout_sector0"]))
    named, meta = [], []

    def add(items, phys_fixed=True):
        for name, qc, m in items:
            tqc = qc if not phys_fixed else transpile(qc, be, initial_layout=list(range(be.num_qubits)),
                                                      optimization_level=0, seed_transpiler=1)
            named.append((name, tqc))
            meta.append(dict(name=name, n_bits=tqc.num_clbits, shots=SHOTS[m["family"]], **m))
    add(family_a(be, chip))
    add(family_b(be, b_sites))
    add(family_c(be, c_sites))
    add(family_d(be, snap), phys_fixed=False)
    # every circuit's policy refusals on the freeze's calibration, RECORDED (they were computed and dropped, behind an
    # always-true filter: an unhealthy qubit in the chip-wide family would have left no trace in the record)
    bad = sorted({r for _, tqc in named for r in L.refusals(snap, tqc, L.QubitPolicy())})
    fam_d_bad = sorted({r for n, tqc in named if n.startswith("D") for r in L.refusals(snap, tqc, L.QubitPolicy())})
    rec, blob = F.freeze(named, dict(
        protocol="R155 diagnostic: chip-wide in-circuit calibration, mid-circuit read spectators, CZ spectator "
                 "ladders, enforced early/inline twins", backend=snap.name, snapshot=snap.to_json(),
        snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc, chip_qubits=chip, b_sites=b_sites,
        c_sites=c_sites, w_dt=W_DT, echo_dt=ECHO_DT, ladder_k=list(LADDER_K), circuits=meta,
        randomizations=RANDOMIZATIONS,
        policy_refusals_family_D=fam_d_bad, policy_refusals_all=bad, prereg=PREREG.name, prereg_sha256=prereg_sha,
        sources=[str(HERE / "hw_r155_diag.py"), str(HERE / "hw_dynamic.py"), str(HERE / "hw_dynamic_run.py")]
        + [str(p) for p in sorted((ROOT / "src" / "oneq" / "hw").glob("*.py"))]))
    return rec, blob


def plan(rec) -> U.Plan:
    return U.Plan(tag=TAG, backend=rec["backend"], shots={c["name"]: c["shots"] for c in rec["circuits"]},
                  prereg_path=str(PREREG), prereg_sha256=rec["prereg_sha256"],
                  twirl_randomizations=rec["randomizations"], records_dir=str(OUTDIR),
                  measurement_reason="a preregistered DIAGNOSTIC: every family is a measurement with a stated "
                                     "decision rule; nothing about superactivation is tested")


# ------------------------------------------------------------------ analysis (pure)
def _bits(counts, n):
    keys = list(counts)
    B = np.array([T.bits_of(k, n) for k in keys], dtype=np.int8)
    w = np.array([int(counts[k]) for k in keys], dtype=float)
    return B, w


def _x_contrast(counts, n, i, ro):
    B, w = _bits(counts, n)
    m = float(w @ (1 - 2 * B[:, i])) / w.sum()
    s = 1 - ro["p01"] - ro["p10"]
    c = (m - (ro["p10"] - ro["p01"])) / s
    sd = math.sqrt(max(1 - m * m, 1.0 / w.sum()) / w.sum()) / s
    return c, sd


def analyse(rec, counts):
    chip = rec["chip_qubits"]
    n = len(chip)
    ro = {}
    for name in ("ro_all0", "ro_all1"):
        B, w = _bits(counts[name], n)
        p1 = (w @ B) / w.sum()
        for i, q in enumerate(chip):
            ro.setdefault(q, {})["p01" if name == "ro_all0" else "p10"] = float(p1[i] if name == "ro_all0"
                                                                              else 1 - p1[i])
    dt = rec["snapshot"]["dt"]
    t1w = rec["w_dt"] * dt
    A = {}
    for i, q in enumerate(chip):
        c0, _ = _x_contrast(counts["xs_w0"], n, i, ro[q])
        c1, s1 = _x_contrast(counts["xs_w1"], n, i, ro[q])
        c2, s2 = _x_contrast(counts["xs_w2"], n, i, ro[q])
        r1, r2 = c1 / c0, c2 / c0
        # exponential: r2 = r1^2; Gaussian: r2 = r1^4; detuning: r2 > r1^2 is impossible for pure decay
        shape = math.log(max(r2, 1e-6)) / math.log(max(r1, 1e-6)) if 0 < r1 < 1 else float("nan")
        t2_exp = -t1w / math.log(r1) if 0 < r1 < 1 else float("inf")
        A[q] = dict(ratio_w1=r1, ratio_w2=r2, decay_shape_exponent_ratio=shape, t2_exp_s=t2_exp,
                    t2_vendor_s=rec["snapshot"]["t2"][q], readout=ro[q], sd_ratio_w1=s1 / max(c0, 1e-9))
        if "xs_echo1" in counts and "echo_dt" in rec:
            # echo vs Ramsey as decay RATES (the echo's total wait is ECHO_DT, not W): the fraction of the Ramsey rate
            # the echo removes is the refocusable (quasi-static) share of this qubit's dephasing
            ce, se = _x_contrast(counts["xs_echo1"], n, i, ro[q])
            re_ = ce / c0
            te = rec["echo_dt"] * dt
            g_ramsey = -math.log(r1) / t1w if 0 < r1 < 1 else 0.0
            g_echo = -math.log(re_) / te if 0 < re_ < 1 else 0.0
            A[q].update(ratio_echo=re_, sd_ratio_echo=se / max(c0, 1e-9), t2_echo_s=1 / g_echo if g_echo else
                        float("inf"), refocusable_fraction=(1 - g_echo / g_ramsey) if g_ramsey else float("nan"))
    shapes = np.array([v["decay_shape_exponent_ratio"] for v in A.values()
                       if math.isfinite(v["decay_shape_exponent_ratio"])])
    Bres = {}
    for st_i, st in enumerate(rec["b_sites"]):
        watch = st["n"] + [st["f"]]
        k = len(watch)
        row = {}
        for v in ("noread", "read", "read_ff"):
            cnt = counts[f"B{st_i}_{v}"]
            row[v] = {str(qq): _x_contrast(cnt, k + 1, j, ro.get(qq, dict(p01=0.0, p10=0.0)))
                      for j, qq in enumerate(watch)}
        f = str(st["f"])
        neigh = [str(x) for x in st["n"]]
        Bres[st_i] = dict(
            far_read_over_noread=row["read"][f][0] / row["noread"][f][0],
            far_readff_over_read=row["read_ff"][f][0] / row["read"][f][0],
            neighbour_read_over_noread=float(np.mean([row["read"][x][0] / row["noread"][x][0] for x in neigh])),
            raw=row)
    Cres = {}
    for st_i, st in enumerate(rec["c_sites"]):
        cz = {k: _x_contrast(counts[f"C{st_i}_cz_k{k}"], 3, 0, ro.get(st["c"], dict(p01=0.0, p10=0.0)))
              for k in rec["ladder_k"]}
        dl = {k: _x_contrast(counts[f"C{st_i}_delay_k{k}"], 3, 0, ro.get(st["c"], dict(p01=0.0, p10=0.0)))
              for k in rec["ladder_k"] if k}
        ks = [k for k in rec["ladder_k"] if k]
        excess = [math.log(max(cz[k][0], 1e-6) / max(dl[k][0], 1e-6)) for k in ks]
        slope = float(np.polyfit(ks, excess, 1)[0]) if len(ks) >= 2 else float("nan")
        Cres[st_i] = dict(contrast_cz=cz, contrast_delay=dl, spectator_z_per_cz=max(0.0, -slope / 2))
    Dres = {}
    units = [c for c in rec["circuits"] if c["family"] == "D"]
    diffs = []
    for k in sorted({c["twin"] for c in units}):
        vals = {}
        for c in (x for x in units if x["twin"] == k):
            uu = R147.unit_from(c["unit"])
            sc = FA.score(R147.M, uu, R147.DECODER)
            st = T.tally(counts[c["name"]], sc, T.Unit(c["name"], 1.0, c["shots"], c["n_bits"], 0.0, 2.0))
            vals["early" if c["early"] else "inline"] = st.mean
        diffs.append(vals["early"] - vals["inline"])
        Dres[k] = vals
    return dict(
        schema="oneq.r155_diag.analysis/1",
        A=dict(per_qubit=A, decay_shape_median=float(np.median(shapes)) if len(shapes) else None,
               t2_exp_over_vendor_median=float(np.median([v["t2_exp_s"] / v["t2_vendor_s"] for v in A.values()
                                                          if math.isfinite(v["t2_exp_s"])])),
               # A3: the median refocusable share of the Ramsey rate (1 = all quasi-static, 0 = all white) and the
               # in-circuit echo T2 against the vendor's
               refocusable_fraction_median=float(np.median([v["refocusable_fraction"] for v in A.values()
                                                            if math.isfinite(v.get("refocusable_fraction",
                                                                                   float("nan")))]))
               if any("refocusable_fraction" in v for v in A.values()) else None,
               t2_echo_over_vendor_median=float(np.median([v["t2_echo_s"] / v["t2_vendor_s"] for v in A.values()
                                                           if math.isfinite(v.get("t2_echo_s", float("inf")))]))
               if any(math.isfinite(v.get("t2_echo_s", float("inf"))) for v in A.values()) else None),
        B=Bres, C=Cres,
        D=dict(per_twin=Dres, early_minus_inline=float(np.mean(diffs)),
               early_minus_inline_sd=float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))),
        readout=ro)


# ------------------------------------------------------------------ modes
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["freeze", "dryrun", "analyse", "submit", "fetch"])
    ap.add_argument("record", nargs="?")
    ap.add_argument("counts", nargs="?")
    ap.add_argument("--live", default="ibm_kingston")
    ap.add_argument("--fake", action="store_true")
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
        rec, blob = freeze_r155(be, S.from_backend(be), U.file_sha256(PREREG))
        stamp = ("fake_" if a.fake else "") + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rp = OUTDIR / f"frozen_{stamp}.json"
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
        rp.with_suffix(".qpy").write_bytes(blob)
        pl = plan(rec)
        shown = rp.resolve().relative_to(ROOT).as_posix() if rp.resolve().is_relative_to(ROOT) else str(rp)
        print(f"frozen {rec['n_circuits']} circuits -> {shown}; {pl.total_shots} shots, cost estimate "
              f"{pl.cost_s():.1f} s (x{U.QUOTA_MARGIN}: {pl.cost_s() * U.QUOTA_MARGIN:.0f} s); chip qubits "
              f"{len(rec['chip_qubits'])}; B sites {rec['b_sites']}; C sites {rec['c_sites']}; D policy refusals "
              f"{rec['policy_refusals_family_D'][:3]}")
        return 0
    rp = pathlib.Path(a.record).resolve()
    rec, blob = json.loads(rp.read_text(encoding="utf-8")), rp.with_suffix(".qpy").read_bytes()
    if a.mode == "dryrun":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        counts = U.local_run(rec, blob, FakeKingston(), plan(rec).shots, method="matrix_product_state")
        dest = rp.parent / f"dryrun_counts_{rp.stem}.json"
        dest.write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
        out = analyse(rec, counts)
        (rp.parent / f"dryrun_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n",
                                                        encoding="utf-8", newline="\n")
        print(json.dumps({k: (v if k != "A" else {kk: vv for kk, vv in v.items() if kk != "per_qubit"})
                          for k, v in out.items() if k in ("A", "C", "D")}, indent=1, default=str)[:3000])
        return 0
    if a.mode == "analyse":
        counts = json.loads(pathlib.Path(a.counts).read_text(encoding="utf-8"))
        out = analyse(rec, counts)
        (rp.parent / f"analysis_{rp.stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n",
                                                          encoding="utf-8", newline="\n")
        return 0
    # submit / fetch (local session, 2026-10-17): every oneq.hw.submit guard; the fetch keeps the FLIGHT RECORD (every
    # shot of every circuit in shot order, the execution spans with their (randomization, shot) slices, the
    # calibration after the job), then the counts and the analysis
    from qiskit_ibm_runtime import QiskitRuntimeService

    from oneq.hw import campaign as C
    svc = QiskitRuntimeService()
    be = svc.backend(rec["backend"])
    pl = plan(rec)
    if a.mode == "submit":
        if "fake" in rec["backend"].lower() or rp.name.startswith("frozen_fake"):
            raise SystemExit("REFUSED: a --fake freeze is a rehearsal; freeze --live first")
        live = S.from_backend(be)
        circuits = F.load(rec, blob)
        health = sorted({r for qc in circuits.values() for r in L.refusals(live, qc, L.QubitPolicy())})
        job = U.submit(pl, rec, blob, be, authorized_flag=a.i_understand_this_spends_credits,
                       remaining_s=U.remaining_quota_s(svc), health_refusals=health)
        print(f"submitted job {job['job_id']} ({job['estimated_cost_s']:.0f} s estimated)")
        return 0
    jobs = [j for j in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUTDIR.glob(f"{TAG}_*.job.json")))
            if j.get("qpy_sha256") == rec["qpy_sha256"]]
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: {len(jobs)} job records for this frozen record; expected one")
    jid = jobs[0]["job_id"]
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
    after = S.from_backend(be)
    (rp.parent / f"after_{jid}.snapshot.json").write_text(json.dumps(after.to_json(), indent=1) + "\n",
                                                         encoding="utf-8", newline="\n")
    counts = U.counts_from_result(result, rec["circuit_order"])
    (rp.parent / f"counts_{jid}.json").write_text(json.dumps(counts) + "\n", encoding="utf-8", newline="\n")
    out = analyse(rec, counts)
    (rp.parent / f"result_{jid}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                                 newline="\n")
    print(json.dumps({k: (v if k != "A" else {kk: vv for kk, vv in v.items() if kk != "per_qubit"})
                      for k, v in out.items() if k in ("A", "C", "D")}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
