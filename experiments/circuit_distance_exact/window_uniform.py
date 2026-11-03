r"""window_uniform.py -- the finite, T-independent hypotheses of WINDOW_PROOF.md, checked mechanically, each with a
negative control that must FAIL; the window formula (Theorem 1) against the exact block iteration and against
stim's DEM ALONE at fresh T; the window structure (Corollary 1); determinism (Corollary 2); the lane's certified
models bound by sha256; the witnesses' embedding for every T.

    python -B window_uniform.py                                  # ibm + 3c, memory-Z and memory-X
    python -B window_uniform.py --circuit 3c --basis Z --fresh-T 16

One JSON object per (circuit, basis) on stdout, then a summary. Exit 0 iff every hypothesis holds, every check
agrees and every negative control fails. Single-threaded; peak memory is reported per run.

Names follow WINDOW_PROOF.md: TPL (template), LBL (labelling), FIR(h), END(e), DET (gauges), G (gauge rank =
tableau randomness), FORMULA (Theorem 1), WINDOWS (Corollary 1), PAIRS (P1-P3 on the exact models below T_1),
BIND (recorded certified models), SIZES (recorded model sizes), WITNESS (upper bound for every T).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stim  # noqa: E402

import wu_bind as wb  # noqa: E402
import wu_core as wc  # noqa: E402
import wu_formula as wf  # noqa: E402
import wu_model as wm  # noqa: E402
import wu_stim as ws  # noqa: E402

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # pragma: no cover
    _PROC = None


def peak_mb() -> int | None:
    if _PROC is None:
        return None
    mi = _PROC.memory_info()
    return int(getattr(mi, "peak_wset", mi.rss) // 2**20)


# ------------------------------------------------------------------------------------------- helpers
def circuit_from(instrs: list) -> stim.Circuit:
    c = stim.Circuit()
    for nm, tg, args in instrs:
        if nm in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            c.append(stim.CircuitInstruction(nm, [stim.target_rec(t) for t in tg], list(args)))
        else:
            c.append(stim.CircuitInstruction(nm, list(tg), list(args)))
    return c


def drop_block_noise(kind: str, basis: str, T: int, j: int) -> stim.Circuit:
    """C_T with every noise instruction of body block position j deleted (a circuit that is NOT the template)."""
    per, tail = wc.SHAPE[kind]
    segs = wc.segments(wc.flat(wc.build(kind, basis, T)))
    lo, hi = 1 + per * j, 1 + per * (j + 1)
    ins = [x for s, seg in enumerate(segs) for x in seg if not (lo <= s < hi and x[0] in wc.NOISE)]
    return circuit_from(ins)


def undetectable_logical(mechs) -> bool:
    d = o = 0
    for dm, om in mechs:
        d ^= dm
        o ^= om
    return d == 0 and o != 0


def doctored_blocks(M: wm.Machine, name: str, edit) -> dict:
    blocks = dict(M.blk)
    blocks[name] = wc.Block(edit(list(getattr(M.tpl, name))), M.nq, name)
    return blocks


def data_cx(instrs: list, where: int, a: int, b: int) -> list:
    return instrs[:where] + [("CX", (a, b), ())] + instrs[where:]


# ------------------------------------------------------------------------------------------- one run
def run(kind: str, basis: str, a) -> dict:
    t0 = time.time()
    M = wm.Machine(kind, basis)
    res = dict(circuit=kind, basis=basis)

    # TPL: the builder emits P . F . B^(T-1) . E . R with the closed-form detectors
    res["TPL"] = [wc.template_check(M.tpl, M.code, T) for T in a.template_T]
    res["LBL"] = M.lbl()

    # FIR / END: minimal h, e for the memory-type rows (projected model) and for every row (full model)
    res["FIR"] = {r: M.minimal(M.fir, r) for r in ("mem", "all")}
    res["END"] = {r: M.minimal(M.end, r) for r in ("mem", "all")}
    h = {r: res["FIR"][r]["minimal"] for r in ("mem", "all")}
    e = {r: res["END"][r]["minimal"] for r in ("mem", "all")}
    if None in h.values() or None in e.values():
        res["seconds"] = round(time.time() - t0, 1)
        return res

    # the block machinery against stim at small T (validates this package's frame-rule implementation)
    res["MACHINE_vs_STIM"] = [dict(T=T, projected=pj, equal=M.model(T, pj) == ws.stim_model(kind, basis, T, pj))
                              for T in (1, 2, 3) for pj in (True, False)]

    # DET: every gauge (reset / measurement basis Pauli, initial |0>) flips nothing, for every T
    Fg = wf.Formula(M, False, h["all"], e["all"], comps="gauges")
    direct = {T: sum(1 for x in M.all_symptoms(T, "gauges") if x[3] or x[4]) for T in range(1, Fg.T0 + 1)}
    res["DET"] = dict(direct_nonzero=direct, formula_sets=dict(S=len(Fg.S), I=len(Fg.I), End=len(Fg.End)),
                      T0=Fg.T0, holds=not any(direct.values()) and not (Fg.S or Fg.I or Fg.End))
    res["G"] = [M.gauge_rank(T) for T in (1, 2)]

    # FORMULA, WINDOWS, PAIRS for the projected model (the lane's lower bound) and the full model (witnesses)
    res["FORMULA"], res["WINDOWS"], res["PAIRS"] = {}, {}, {}
    formulas = {}
    for pj in (True, False):
        rows, key = ("mem", "projected") if pj else ("all", "full")
        F = wf.Formula(M, pj, h[rows], e[rows])
        F2 = wf.Formula(M, pj, h[rows], e[rows], T_ref=F.T_ref + 1, i0=F.i0 + 1)
        formulas[key] = F
        rad = wf.radii(F)
        T1 = rad["T1"]
        vs_exact, wins = {}, {}
        for T in range(1, T1 + 2):
            ex = M.model(T, pj)
            if T >= F.T0:
                vs_exact[T] = ex == F.model(T)
            wins[T] = wf.windows(ex, M.W)
        fresh = a.fresh_T if pj else a.fresh_T_full
        vs_stim = {}
        for T in fresh:
            st = ws.stim_model(kind, basis, T, pj)
            vs_stim[T] = dict(stim=len(st), formula=len(F.model(T)), equal=st == F.model(T))
            del st
        res["FORMULA"][key] = dict(h=h[rows], e=e[rows], U=F.U, j_s=F.j_s, T0=F.T0,
                                   T_ref=F.T_ref, i0=F.i0, S=len(F.S), I=len(F.I), End=len(F.End),
                                   stable_across_T_ref=bool(F.S == F2.S and F.I == F2.I and F.End == F2.End),
                                   vs_exact=vs_exact, vs_stim=vs_stim,
                                   holds=bool(all(vs_exact.values()) and all(v["equal"] for v in vs_stim.values())
                                              and F.S == F2.S and F.I == F2.I and F.End == F2.End))
        res["WINDOWS"][key] = dict(radii=rad, structure=wf.structure(F, rad))
        res["PAIRS"][key] = [wf.pair_check(wins[T], wins[T + 1], T) for T in range(1, T1 + 1)]

    # SIZES and BIND: the lane's recorded models
    Fp = formulas["projected"]
    sizes = []
    for c_, b_, T, nm_, nd_, src in wb.recorded_sizes():
        if (c_, b_) != (kind, basis):
            continue
        mech = len(Fp.model(T)) if T >= Fp.T0 else len(M.model(T, True))
        sizes.append(dict(T=T, recorded=[nm_, nd_], predicted=[mech, 72 * (T + 1)], agree=[nm_, nd_] == [mech, 72 * (T + 1)],
                          source=src))
    res["SIZES"] = dict(rows=len(sizes), disagree=sum(not s["agree"] for s in sizes), detail=sizes,
                        holds=bool(sizes) and all(s["agree"] for s in sizes))
    bind = []
    outdir = pathlib.Path(tempfile.mkdtemp(prefix="window_uniform_"))   # deleted below
    for c_, b_, T, to, sh, nm_, nd_, src in wb.recorded_models():
        if (c_, b_) != (kind, basis) or T not in a.bind_T:
            continue
        p, hh = wb.export_and_hash(kind, basis, T, to, outdir)
        fm = wb.read_masks(p)
        model = Fp.model(T) if T >= Fp.T0 else M.model(T, True)
        bind.append(dict(T=T, time_order=to, source=src, sha256_equal=hh == sh, file_equals_model=fm == model,
                         mechanisms=len(fm)))
        p.unlink()
    res["BIND"] = dict(rows=bind, holds=bool(bind) and all(b["sha256_equal"] and b["file_equals_model"] for b in bind))

    # WITNESS: the recorded full-DEM witnesses embed into every T (end type: shifted; start type: unchanged)
    Ff = formulas["full"]
    res["WITNESS"] = []
    for c_, b_, Tw, ids, wt in wb.witnesses():
        if (c_, b_) != (kind, basis):
            continue
        lst = ws.stim_mech_list(kind, basis, Tw)
        Wm = [lst[i] for i in ids]
        ok_w = undetectable_logical(Wm)
        minus_one = all(not undetectable_logical(Wm[:i] + Wm[i + 1:]) for i in range(len(Wm)))
        # classify each mechanism: in the end part (relative to T_w) or in the T-independent start part
        sh_end = (Tw - Ff.D) * M.W
        in_end = all(sh_end >= 0 and (d & ((1 << sh_end) - 1)) == 0 and (d >> sh_end, o) in Ff.End for d, o in Wm)
        need = []
        for d, o in Wm:
            if (d, o) in Ff.S:
                need.append(Ff.T0)
                continue
            hit = None
            for i in range(Ff.j_s, 64):
                s_ = (i - 1) * M.W
                if (d & ((1 << s_) - 1)) == 0 and (d >> s_, o) in Ff.I:
                    hit = i
                    break
            need.append(None if hit is None else max(Ff.T0, hit + 1 + Ff.U))
        start_type = None not in need
        mode = "end" if in_end else "start" if start_type else "neither"
        Tmax_direct = max(wf.radii(Ff)["T1"] + 1, max((n for n in need if n), default=0))
        direct = {}
        for T in range(Tw, Tmax_direct + 1):
            model = M.model(T, False)
            shift = (T - Tw) * M.W if mode == "end" else 0
            Ws = [(d << shift, o) for d, o in Wm]
            direct[T] = all(m in model for m in Ws) and undetectable_logical(Ws)
        res["WITNESS"].append(dict(rounds=Tw, weight=wt, ids=ids, undetectable_logical=ok_w, minus_one_fails=minus_one,
                                   mode=mode, every_T_from=Tw if mode == "end" else max(need) if start_type else None,
                                   direct=direct,
                                   holds=bool(ok_w and minus_one and mode != "neither" and all(direct.values()))))

    if not a.no_controls:
        res["CONTROLS"] = controls(M, h, e, formulas, a, kind, basis, res["WITNESS"])
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_mb"] = peak_mb()
    return res


# ------------------------------------------------------------------------------------------- controls
def controls(M: wm.Machine, h, e, formulas, a, kind, basis, witness_results) -> dict:
    """Each entry is the checked condition evaluated on a doctored input; every one must be False."""
    out = {}
    code, tpl = M.code, M.tpl
    T = 5
    ins = wc.flat(wc.build(kind, basis, T))
    # TPL: one CX pair removed in block 3; one detector record dropped; one observable record dropped
    per, _ = wc.SHAPE[kind]
    segs = wc.segments(ins)
    s3 = next(s for s in range(1 + per * 3, 1 + per * 4) if any(x[0] == "CX" for x in segs[s]))
    cx_at = next(i for i, x in enumerate(segs[s3]) if x[0] == "CX")
    seg3 = list(segs[s3])
    seg3[cx_at] = ("CX", seg3[cx_at][1][2:], seg3[cx_at][2])
    bad = [x for s in segs[:s3] for x in s] + seg3 + [x for s in segs[s3 + 1:] for x in s]
    out["TPL: one CNOT removed in block 3 (stream)"] = wc.template_check(tpl, code, T, circuit_from(bad))["stream"]
    di = next(i for i, x in enumerate(ins) if x[0] == "DETECTOR" and len(x[1]) > 1)
    bad = ins[:di] + [("DETECTOR", ins[di][1][1:], ins[di][2])] + ins[di + 1:]
    out["TPL: a detector missing one record (closed form)"] = wc.template_check(tpl, code, T, circuit_from(bad))["detectors"]
    oi = next(i for i, x in enumerate(ins) if x[0] == "OBSERVABLE_INCLUDE" and len(x[1]) > 1)
    bad = ins[:oi] + [("OBSERVABLE_INCLUDE", ins[oi][1][1:], ins[oi][2])] + ins[oi + 1:]
    out["TPL: an observable missing one record (closed form)"] = wc.template_check(tpl, code, T, circuit_from(bad))["observables"]
    # LBL: the tail (or, with no tail, the body) loses one measurement
    nm_ = "E" if tpl.E else "B"

    def lose_meas(lst):
        i = next(k for k, x in enumerate(lst) if x[0] in ("M", "MX"))
        return lst[:i] + [(lst[i][0], lst[i][1][1:], lst[i][2])] + lst[i + 1:]
    Md = wm.Machine(kind, basis, doctored_blocks(M, nm_, lose_meas))
    out[f"LBL: block {nm_} loses one measurement"] = Md.lbl()["holds"]
    # FIR: a CNOT between two data qubits inside B (a frame that rings forever)
    nb = len(tpl.B)
    Md = wm.Machine(kind, basis, doctored_blocks(M, "B", lambda l: data_cx(l, nb // 2, 0, 1)))
    Md.lbl()
    out["FIR: a data-data CNOT inside the body block (some h <= 8)"] = Md.minimal(Md.fir, "mem")["minimal"] is not None
    if h["mem"] > 0:
        out[f"FIR: one power too few (h = {h['mem'] - 1})"] = M.fir(h["mem"] - 1, "mem")["holds"]
    # END: a CNOT between two data qubits in the readout, before the measurement
    Md = wm.Machine(kind, basis, doctored_blocks(M, "R", lambda l: data_cx(l, 0, 0, 1)))
    Md.lbl()
    out["END: a data-data CNOT in the readout (some e <= 8)"] = Md.minimal(Md.end, "mem")["minimal"] is not None
    if e["mem"] > 0:
        out[f"END: one power too few (e = {e['mem'] - 1})"] = M.end(e["mem"] - 1, "mem")["holds"]
    # DET: an extra, random detector (the other type's round-0 outcome of check 0)
    cd, co = wc.closed_form(code, 2)
    oth = 1 - code.bi
    cd = dict(cd)
    cd[(0, oth, 0)] = frozenset({(code.anc(oth, 0), 0)})
    out["DET: an other-type round-0 detector added (no gauge flips it)"] = not any(
        x[3] or x[4] for x in M.all_symptoms(2, "gauges", cf=(cd, co)))
    # G: the gauge rank without the preparation's gauges
    out["G: gauge rank equals tableau randomness without P's gauges"] = M.gauge_rank(2, drop=lambda p, nm, i: nm == "P")["equal"]
    # FORMULA: the end part one round early; stim of a circuit whose interior block j lost all its noise
    Fp = formulas["projected"]
    Tf = a.fresh_T[0] if a.fresh_T else 16
    st = ws.stim_model(kind, basis, Tf, True)
    shifted = set(Fp.S)
    for i in range(Fp.j_s, Tf - Fp.U):
        shifted.update((d << ((i - 1) * M.W), o) for d, o in Fp.I)
    shifted.update((d << ((Tf - 1 - Fp.D) * M.W), o) for d, o in Fp.End)
    out["FORMULA: end windows placed one round early (vs stim)"] = shifted == st
    st_bad = ws.stim_model(kind, basis, Tf, True, circuit=drop_block_noise(kind, basis, Tf, Tf // 2))
    out["FORMULA: stim of C_T with one interior block's noise deleted"] = st_bad == Fp.model(Tf)
    # WINDOWS: one interior mechanism removed from the exact model at T_1
    rad = wf.radii(Fp)
    T1 = rad["T1"]
    ex = M.model(T1, True)
    mid = rad["R1"]
    victim = next(m for m in sorted(ex) if wf.start_round(m[0], M.W) == mid)
    ex.discard(victim)
    by, p1, nodet = wf.windows(ex, M.W)
    out["WINDOWS: an interior window missing one mechanism (pair check)"] = wf.pair_check(
        (by, p1, nodet), wf.windows(M.model(T1 + 1, True), M.W), T1)["holds"]
    # SIZES: one recorded size off by one
    recs = [r for r in wb.recorded_sizes() if r[0] == kind and r[1] == basis]
    if recs:
        c_, b_, Tr, nm_, nd_, src = recs[-1]
        mech = len(Fp.model(Tr)) if Tr >= Fp.T0 else len(M.model(Tr, True))
        out["SIZES: one recorded model size off by one"] = [nm_ + 1, nd_] == [mech, 72 * (Tr + 1)]
    # WITNESS: one mechanism dropped; the translation one round off what its type requires (into T_w + 2)
    for w in witness_results:
        lst = ws.stim_mech_list(kind, basis, w["rounds"])
        Wm = [lst[i] for i in w["ids"]]
        out[f"WITNESS ({w['mode']} type): one mechanism dropped (undetectable and logical)"] = undetectable_logical(Wm[1:])
        # (translating the WHOLE witness by a wrong amount is not a control: for the IBM cycle the final-window
        # shapes are interior shapes too, so the witness also sits one round earlier -- measured 2026-10-22.)
        model = M.model(w["rounds"] + 2, False)
        sh = 2 if w["mode"] == "end" else 0
        moved = [(d << ((sh + (1 if i == 0 else 0)) * M.W), o) for i, (d, o) in enumerate(Wm)]
        out[f"WITNESS ({w['mode']} type): one mechanism one round off (contained, undetectable, logical)"] = (
            all(m in model for m in moved) and undetectable_logical(moved))
    return out


# ------------------------------------------------------------------------------------------- verdict
def verdict(res: dict) -> tuple[bool, list]:
    why = []
    if not all(t["agrees"] for t in res["TPL"]):
        why.append("TPL")
    if not res["LBL"]["holds"]:
        why.append("LBL")
    for k in ("FIR", "END"):
        if any(v["minimal"] is None for v in res[k].values()):
            why.append(k)
    if "MACHINE_vs_STIM" not in res:
        return False, why
    if not all(x["equal"] for x in res["MACHINE_vs_STIM"]):
        why.append("MACHINE_vs_STIM")
    if not res["DET"]["holds"]:
        why.append("DET")
    if not all(g["equal"] for g in res["G"]):
        why.append("G")
    for key in ("projected", "full"):
        if not res["FORMULA"][key]["holds"]:
            why.append(f"FORMULA[{key}]")
    if not res["WINDOWS"]["projected"]["structure"]["holds"]:
        why.append("WINDOWS[projected]")
    if not all(p["P1"] and p["P2"] and p["P3_start"] and p["P3_final"] and p["time_shift_ok"]
               for p in res["PAIRS"]["projected"]):
        why.append("PAIRS[projected]")
    for k in ("SIZES", "BIND"):
        if not res[k]["holds"]:
            why.append(k)
    if not all(w["holds"] for w in res["WITNESS"]):
        why.append("WITNESS")
    if any(res.get("CONTROLS", {}).values()):
        why.append("CONTROLS: " + ", ".join(k for k, v in res["CONTROLS"].items() if v))
    return not why, why


def summary(res: dict) -> str:
    ok, why = verdict(res)
    L = [f"== {res['circuit']} memory-{res['basis']} -- {res.get('seconds')} s, peak {res.get('peak_mb')} MB"]
    L.append(f"  TPL   {sum(t['agrees'] for t in res['TPL'])}/{len(res['TPL'])} T agree "
             f"({', '.join(str(t['T']) for t in res['TPL'])})")
    L.append(f"  LBL   {'PASS' if res['LBL']['holds'] else 'FAIL'} (delta=0: {res['LBL']['delta0']}, "
             f"delta=-1: {res['LBL']['delta_minus1']})")
    for k in ("FIR", "END"):
        for rows, v in res[k].items():
            below = v["just_below"]
            L.append(f"  {k:<5} rows={rows}: minimal {v['minimal']}"
                     + (f" (fails at {v['minimal'] - 1}: {below.get('violating_basis_vectors', below.get('detector_violations'))} basis vectors)" if below else ""))
    if "MACHINE_vs_STIM" in res:
        L.append(f"  MACHINE = stim at T=1..3, full and projected: {all(x['equal'] for x in res['MACHINE_vs_STIM'])}")
        L.append(f"  DET   {'PASS' if res['DET']['holds'] else 'FAIL'}  G (gauge rank = tableau): "
                 f"{[(g['T'], g['gauge_rank'], g['tableau_random']) for g in res['G']]}")
        for key in ("projected", "full"):
            f = res["FORMULA"][key]
            L.append(f"  FORMULA[{key}] U={f['U']} j_s={f['j_s']} T0={f['T0']} |S|={f['S']} |I|={f['I']} "
                     f"|End|={f['End']} stable={f['stable_across_T_ref']} = exact {f['vs_exact']} "
                     f"= stim {{{', '.join(f'{T}: {v['equal']} ({v['stim']})' for T, v in f['vs_stim'].items())}}}")
            s = res["WINDOWS"][key]["structure"]
            L.append(f"  WINDOWS[{key}] radii {res['WINDOWS'][key]['radii']} structure "
                     f"{'HOLDS' if s['holds'] else 'does not hold'}: interior {s['interior_shapes']} shapes, start "
                     f"{s['start_window']}, last {s['last_window']}, final {s['final_window']}")
        pr = res["PAIRS"]["projected"]
        L.append("  PAIRS[projected] " + " ".join(
            f"T={p['T']}:{'H' if p['holds'] else ('S' if p['time_shift_ok'] else 'F')}" for p in pr)
            + "  (H = P1-P3 hold, S = time-shift preconditions only)")
        L.append(f"  SIZES {res['SIZES']['rows']} recorded rows, {res['SIZES']['disagree']} disagree;  BIND "
                 + ", ".join(f"T={b['T']}{'/to' if b['time_order'] else ''}: sha {b['sha256_equal']} set {b['file_equals_model']}"
                             for b in res["BIND"]["rows"]))
        for w in res["WITNESS"]:
            L.append(f"  WITNESS weight {w['weight']} (T_w={w['rounds']}): mode {w['mode']}, every T >= "
                     f"{w['every_T_from']}, direct {w['direct']}")
        for k, v in res.get("CONTROLS", {}).items():
            L.append(f"  control {'fails as required' if not v else 'DID NOT FAIL'}: {k}")
    L.append(f"  VERDICT {'PASS' if ok else 'FAIL ' + str(why)}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", nargs="*", default=["ibm", "3c"], choices=["ibm", "3c"])
    ap.add_argument("--basis", nargs="*", default=["Z", "X"], choices=["Z", "X"])
    ap.add_argument("--template-T", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 24, 32])
    ap.add_argument("--fresh-T", type=int, nargs="*", default=[16, 24, 32])
    ap.add_argument("--fresh-T-full", type=int, nargs="*", default=[16, 24])
    ap.add_argument("--bind-T", type=int, nargs="*", default=[1, 2, 6, 11, 12])
    ap.add_argument("--no-controls", action="store_true")
    a = ap.parse_args()
    results = []
    for kind in a.circuit:
        for basis in a.basis:
            res = run(kind, basis, a)
            print(json.dumps(res, default=str), flush=True)
            results.append(res)
    print()
    for res in results:
        print(summary(res))
    ok = all(verdict(r)[0] for r in results)
    print(f"\nOVERALL {'PASS' if ok else 'FAIL'}: {len(results)} (circuit, basis) runs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
