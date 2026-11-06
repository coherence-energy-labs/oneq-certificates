"""R171 (item 18): the protocol's own deterministic parities as FREE SYNDROMES (EXPLORATORY: outside every
preregistered verdict).

Every R146 / R147 unit reads bits whose ideal values are partly FIXED: a parity s of the measured bits with
|E_ideal[(-1)^{s.b}]| = 1 is a DETECTOR (the stabilizer of the ideal outcome law, read off the engine's own
Walsh-Hadamard transform of the noiseless run, q^(s) = +-1). Per unit they are spanned by
  ell        the copy bit c0 (Bob's sector bit), fixed by P's sector;
  pair_t     the decode bit of shield pair t when P left it in a Psi state: sector 0, g_t in {X, Y}, beta = 0
             (the pairs P dephased -- g_t in {I, Z}, or read mid-circuit in R147 -- are random bits, NOT detectors);
  logical    the parity the SCORE reads (it contains key0's output bit): held OUT of every fit here.
ell and pair_t are free syndromes: measured on every shot, never scored, and LOCAL -- a pair syndrome on an X/Y
shield sees only its own pair's gates (no key CZ touches it). The DETECTOR ERROR MODEL is the engine's: an error at
a location flips detector s iff its propagated Pauli's X pattern has odd overlap with s (engine.plan's flip
patterns -- exactly a stim DEM for the reachable errors), and oneq.hw.sensitivity gives the exact linear response of
the violation probability to every location (first order for a blocked one), so the per-location loss of a
violation score is the DEM column. ONE_Q's stim tools (oneq.hw.stim_export) cannot be used: these circuits are not
Clifford (CH, ry(pi/4), a T-type key phase) and the model carries sigma, which annotate refuses. The OPERATOR is
ported -- detectors as parities of the outcome law, columns as linear responses -- not the stim application.

What is fitted: per-location excess factors beta (class; qubit; edge) to the measured violation rates by WLS with
Pearson weights (the PREDICTED rate's binomial variance: a detector with no violation in its shots must not take
infinite weight, R164's trap). What is tested, held out:
  split      fit on the even units' syndromes, chi2 on the odd units' syndromes (and back);
  to_score   syndrome-fitted beta on one half -> the SCORE residuals of the OTHER half (the item-17 rows of
             r171_reattribution.py, same model): disjoint units, so no shot is shared between fit and test;
  r146->r147 class factors fitted on R146's syndromes applied to R147's (other qubits, same physics).
And, model-free (counts only): the covariance of syndrome violations on DISJOINT pairs of one unit (a local model
predicts none), and R147's read-shield pairs, whose decode bit must equal the mid-circuit bit (mcm XOR pair = 0):
their violation rate against the two read-outs that floor it (the shield's measured mid-circuit assignment, MCMCAL,
and the pair qubit's in-circuit read-out), split by the bit read (a |1> relaxes, a |0> does not).

Models (--base; r171_reattribution's arms of the same name; R147 on its stand-ins with the hold at R166's measured
2.7 us, no global stall):
  template (default)  V3c + the engine's R166 template (NoiseParams.twirl_template) + dressing x2: the newest model;
                      the read rows play the template BEFORE the exact copy is inserted (the copy is an instrument)
  dressing            V3c + dressing x1.54, with the 1q slot class in every DEM column

  python r171_syndromes.py REPO_ROOT [--traj N] [--limit N] [--base template|dressing]
      -> evidence/superactivation/r171_residual/syndromes_r152[_dressing].json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import r152_attribution as A  # noqa: E402
import r171_reattribution as RA  # noqa: E402

from oneq.hw import engine as E  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import sensitivity as SE  # noqa: E402
from oneq.hw import stats as T  # noqa: E402

TOL = 1e-6


# ------------------------------------------------------------------ detectors (pure)
def detector_group(q_hat: np.ndarray, m: int, tol: float = TOL) -> list:
    """every nonzero parity mask s (engine bit order: bit m-1-j is clbit j) with |q^(s)| = 1, with its sign"""
    return [(s, 1 if q_hat[s] > 0 else -1) for s in range(1, 1 << m) if abs(abs(q_hat[s]) - 1.0) < tol]


def min_weight_basis(group: list) -> list:
    """a basis of the detector group (a GF(2) subspace) of minimum total Hamming weight: greedy by weight, which is
    optimal for a matroid. Each element keeps its sign."""
    basis, red = [], []                                  # red: the XOR basis kept in reduced form for independence
    for s, sg in sorted(group, key=lambda t: (bin(t[0]).count("1"), t[0])):
        v = s
        for b in red:
            v = min(v, v ^ b)
        if v:
            red.append(v)
            red.sort(reverse=True)
            basis.append((s, sg))
    return basis


def clbits_of(mask: int, m: int) -> list:
    return [j for j in range(m) if (mask >> (m - 1 - j)) & 1]


def role(clbits: list, k_pair: int) -> str:
    """the detector's place in the protocol's classical bits: 0 = ell, 1..k_pair = pair decode bits, k_pair + 1 = b1,
    k_pair + 2 = key0's output (the score's)"""
    if k_pair + 2 in clbits:
        return "logical"
    if clbits == [0]:
        return "ell"
    if len(clbits) == 1 and 1 <= clbits[0] <= k_pair:
        return "pair"
    return "other"


def violation_score(clbits: list, sign: int):
    want = 0 if sign > 0 else 1

    def score(bits):
        return 1.0 if sum(bits[j] for j in clbits) % 2 != want else 0.0
    return score


def measured_violations(counts: dict, clbits: list, sign: int, n_bits: int) -> tuple:
    """(violating shots, shots) of one detector in a unit's counts (clbits index the first bits of each key)"""
    sc = violation_score(clbits, sign)
    k = n = 0
    for key, c in counts.items():
        c = int(c)
        n += c
        k += c * int(sc(T.bits_of(key, n_bits)))
    return k, n


def pearson_var(p_pred: float, n: int, se: float = 0.0) -> float:
    """the variance of a measured rate under the model's own rate, floored at half a shot"""
    p = min(max(p_pred, 0.5 / n), 1 - 0.5 / n)
    return p * (1 - p) / n + se ** 2


def disjoint_pair_covariance(counts: dict, dets: list, n_bits: int) -> list:
    """per pair of detectors with DISJOINT bits: the covariance of their violation indicators, and its sd under
    independence. dets: [(label, clbits, sign)]"""
    rows = []
    sc = [(lab, violation_score(cl, sg), set(cl)) for lab, cl, sg in dets]
    tot = sum(int(c) for c in counts.values())
    for (la, fa, ca), (lb, fb, cb) in itertools.combinations(sc, 2):
        if ca & cb:
            continue
        na = nb = nab = 0
        for key, c in counts.items():
            bits = T.bits_of(key, n_bits)
            va, vb = fa(bits), fb(bits)
            c = int(c)
            na += c * int(va)
            nb += c * int(vb)
            nab += c * int(va and vb)
        pa, pb, pab = na / tot, nb / tot, nab / tot
        sd = math.sqrt(max(pa * pb * (1 - pa) * (1 - pb), 0.25 / tot ** 2) / tot)
        rows.append(dict(a=la, b=lb, pa=pa, pb=pb, pab=pab, cov=pab - pa * pb, sd=sd, n=tot))
    return rows


def pooled(rows: list, ref: str | None = None) -> dict:
    """inverse-variance mean of the covariances (minus each row's `ref` -- the model's own covariance -- when given)"""
    if not rows:
        return dict(n=0)
    w = np.array([1 / r["sd"] ** 2 for r in rows])
    c = np.array([r["cov"] - (r[ref] if ref else 0.0) for r in rows])
    mean = float((w * c).sum() / w.sum())
    sd = float(1 / math.sqrt(w.sum()))
    return dict(n=len(rows), cov=mean, sd=sd, z=mean / sd, chi2=float((w * c ** 2).sum()))


def identifiable(rows: list, keys: list, min_norm: float = 1.0) -> list:
    """the keys whose loss column the rows can determine to better than +-1 in beta: weighted norm
    sqrt(sum_r L_k^2 / var_r) >= min_norm. A location a set of detectors barely sees would otherwise take a huge,
    meaningless factor (the per-qubit fit of a half that hardly touches a qubit)"""
    return [k for k in keys if math.sqrt(sum(r["L"].get(k, 0.0) ** 2 / r["var"] for r in rows)) >= min_norm]


# ------------------------------------------------------------------ held-out arithmetic (pure over rows)
def apply_beta(rows: list, beta: dict) -> dict:
    """chi2 of rows' residuals before and after subtracting sum_k beta_k L_k (beta from ANOTHER set of rows)"""
    before = after = 0.0
    for r in rows:
        y = r["measured"] - r["predicted"]
        f = sum(b * r["L"].get(k, 0.0) for k, b in beta.items())
        before += y * y / r["var"]
        after += (y - f) ** 2 / r["var"]
    return dict(n=len(rows), chi2_before=before, chi2_after=after, delta=before - after)


def split(rows: list, key=lambda r: r["unit_index"] % 2) -> tuple:
    a = [r for r in rows if key(r) == 0]
    b = [r for r in rows if key(r) == 1]
    return a, b


def heldout(fit_rows: list, test_rows: list, keys: list) -> dict:
    keys = identifiable(fit_rows, keys)
    if not keys:
        return dict(keys=[], skipped="no location the fit rows see")
    f = A.fit(fit_rows, keys)
    return dict(keys=keys, beta=f["beta"], sd=f["sd"], fit_chi2=[f["chi2_no_excess"], f["chi2_fit"]],
                condition=f["condition"], test=apply_beta(test_rows, f["beta"]))


def to_score(syn_fit: list, score_test: list, keys: list) -> dict:
    """syndrome-fitted beta (one half's units) applied to the OTHER half's score rows: their chi2, and the F_e
    residual the beta predicts for that half's p0 / main units against the one measured there"""
    h = heldout(syn_fit, score_test, keys)
    if "beta" not in h:
        return h
    h["F_e_predicted_shift"] = A.fe_shift(score_test, h["beta"])
    h["F_e_measured_residual"] = 1.5 * sum(r["weight"] * (r["measured"] - r["predicted"]) for r in score_test
                                           if r["kind"] in ("p0", "main"))
    return h


def per_location_ratio(rows: list, by: str) -> dict:
    """model-light: per syndrome location (rows[i][by]), measured over predicted violations, pooled over units"""
    agg: dict = {}
    for r in rows:
        g = agg.setdefault(str(r[by]), dict(k=0, n=0, pred=0.0, units=0))
        g["k"] += r["k"]
        g["n"] += r["n"]
        g["pred"] += r["predicted"] * r["n"]
        g["units"] += 1
    out = {}
    for loc, g in agg.items():
        p, q = g["k"] / g["n"], g["pred"] / g["n"]
        out[loc] = dict(units=g["units"], shots=g["n"], measured=p, sd=math.sqrt(max(q * (1 - q), 1e-12) / g["n"]),
                        predicted=q, ratio=p / q if q > 0 else None, z=(p - q) / math.sqrt(max(q * (1 - q), 1e-12) / g["n"]))
    return out


# ------------------------------------------------------------------ engine rows
def unit_syndromes(qc, model, k_pair: int):
    """the unit's detector basis from the circuit's NOISELESS outcome law, with roles"""
    P0 = E.plan(qc, model, N.NOISELESS)
    m = len(P0.meas)
    q_hat = E.wht(E._true_probs(E.simulator(len(P0.active), "numpy").run(P0.ops, {}), P0.meas))
    basis = min_weight_basis(detector_group(q_hat, m))
    return m, [(clbits_of(s, m), sg, role(clbits_of(s, m), k_pair)) for s, sg in basis]


def phys_of(qc) -> dict:
    return {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index for i in qc.data
            if i.operation.name == "measure"}


def dem_column(qc, score, model, params, with_slots: bool | None = None) -> dict:
    """the detector's DEM column: its first-order violation loss per location (r171_reattribution.class_losses), with
    the 1q slot class only when the model does not already play the template (it would count it twice)"""
    s = SE.sensitivity(qc, score, model, params)
    if not (with_slots if with_slots is not None else not params.twirl_template):
        L = RA.class_losses(s, model, params)
    else:
        active, slots = RA.slot_sensitivity(qc, score, model, params)
        L = RA.class_losses(s, model, params, active, slots)
    L.pop("_value")
    return L


def syndrome_rows(qc, counts, n_bits, unit, k_pair, model, params, traj, seed, kind, index, weight,
                  check_qc=None) -> tuple:
    """one unit's syndrome rows (not the logical): measured violations, the engine's predicted rate, the DEM column
    (per-location first-order loss). check_qc: a second stand-in whose syndrome predictions must agree (R147's c1:
    the wrong-correction Z on the key commutes with every syndrome)"""
    slots = not params.twirl_template                     # a model that plays the template has no slot class
    if check_qc is not None:
        check_qc, _ = RA.played(check_qc, model, params)
    qc, params = RA.played(qc, model, params)             # the template once, not at every engine call
    m, dets = unit_syndromes(qc, model, k_pair)
    ph = phys_of(qc)
    rows, logical, worst = [], [], 0.0
    for j, (cl, sg, rl) in enumerate(dets):
        if rl in ("logical", "other"):
            logical.append(dict(clbits=cl, sign=sg, role=rl))
            continue
        sc = violation_score(cl, sg)
        k, n = measured_violations(counts, cl, sg, n_bits)
        pred = E.expectation(qc, sc, model, params, traj, np.random.default_rng([171, seed, j]))
        if check_qc is not None:
            alt = E.expectation(check_qc, sc, model, params, traj, np.random.default_rng([171, seed, j]))
            worst = max(worst, abs(alt.mean - pred.mean))
        L = dem_column(qc, sc, model, params, with_slots=slots)
        rows.append(dict(unit=unit, unit_index=index, kind=kind, weight=weight, role=rl, clbits=cl, sign=sg,
                         qubit=ph[cl[0]], k=k, n=n, measured=k / n, predicted=pred.mean,
                         var=pearson_var(pred.mean, n, pred.se), L=RA.compact(L)))
    # locality: measured covariance of every DISJOINT pair of syndromes, against the model's own (an error the
    # model knows can flip both -- a routed SWAP chain, a shared qubit -- is not an excess)
    lab = {f"{r['role']}{r['clbits']}": r for r in rows}
    pairs = disjoint_pair_covariance(counts, [(x, r["clbits"], r["sign"]) for x, r in lab.items()], n_bits)
    for j, pr in enumerate(pairs):
        ra, rb = lab[pr["a"]], lab[pr["b"]]
        fa, fb = violation_score(ra["clbits"], ra["sign"]), violation_score(rb["clbits"], rb["sign"])
        joint = E.expectation(qc, lambda b, fa=fa, fb=fb: fa(b) * fb(b), model, params, traj,
                              np.random.default_rng([172, seed, j])).mean
        # Pearson: the sd from the MODEL's rates (a detector with no violation in its shots must not dominate)
        n = pr["n"]
        qa, qb = (min(max(r["predicted"], 0.5 / n), 1 - 0.5 / n) for r in (ra, rb))
        pr.update(unit=unit, model_cov=joint - ra["predicted"] * rb["predicted"],
                  sd=math.sqrt(qa * qb * (1 - qa) * (1 - qb) / n))
    return rows, logical, worst, pairs


def r146_syndromes(art, model, params, traj, limit=None) -> tuple:
    import hw_superact_run as HS  # noqa: F401
    from oneq.hw import freeze as F
    d, r146 = art["dir"], art["parts"]["r146"]
    c146 = F.load(r146, (d / "r146.qpy").read_bytes(), check_sources=False)
    rows, cov, logical = [], [], {}
    for k, u in enumerate(r146["units"]):
        if limit is not None and k >= limit:
            break
        if u["unit"]["branch"] == "erased_erased":
            continue                                        # two bits, both the score's
        n = u["name"]
        cnt = art["counts"]["r146__" + n]
        rr, lg, _, pairs = syndrome_rows(c146[n], cnt, u["n_bits"], n, r146["m"], model, params, traj, k, "p0", k,
                                         u["weight"])
        rows += rr
        logical[n] = lg
        cov += pairs
    return rows, cov, logical


def r147_syndromes(art, rec147, stand_ins, model, params, traj, assign: dict, limit=None) -> tuple:
    import hw_dynamic as D
    from oneq.hw import freeze as F
    d = art["dir"]
    c147 = F.load(rec147, (d / "r147.qpy").read_bytes(), check_sources=False)
    rows, cov, logical, worst = [], [], {}, 0.0
    checked = 0
    lay0 = rec147["layout_sector0"]
    for k, u in enumerate(rec147["units"]):
        if limit is not None and k >= limit:
            break
        if u["unit"]["branch"] == "erased_erased":
            continue
        n, inp = u["name"], u["prediction_inputs"]
        cnt = art["counts"]["r147__" + n]
        if inp["kind"] == "dynamic":
            sn = D.augmented(model, [tuple(e) for e in inp["virt"]])
            chk = stand_ins[inp["c1"]] if (inp["c1"] and checked < 12) else None
            checked += chk is not None
            rr, lg, w, pairs = syndrome_rows(stand_ins[inp["c0"]], cnt, u["n_bits"], n, 5, sn, params, traj, k,
                                             u["kind"], k, u["weight"], check_qc=chk)
            worst = max(worst, w)
            if u["unit"]["beta"] == 0:
                rr += read_rows(stand_ins[inp["c0"]], dict(u, _counts=cnt), k, sn, params, traj, assign, lay0)
        else:
            rr, lg, _, pairs = syndrome_rows(c147[n], cnt, u["n_bits"], n, 5, model, params, traj, k, u["kind"], k,
                                             u["weight"])
        rows += rr
        logical[n] = lg
        cov += pairs
    return rows, cov, logical, dict(c1_checked=checked, c1_max_abs_diff=worst)


# ------------------------------------------------------------------ the read-shield pairs, in the engine
def free_qubit(qc, snap) -> int:
    """the highest-index device qubit the circuit never touches and no calibrated edge joins to a used one"""
    used = {qc.find_bit(b).index for inst in qc.data for b in inst.qubits}
    nb = {q for e in snap.cz for q in e if set(e) & used}
    for f in range(qc.num_qubits - 1, -1, -1):
        if f not in used and f not in nb:
            return f
    raise ValueError("no free qubit to hold the mid-circuit bit")


def with_mcm_copy(qc, shield: int, free: int):
    """the stand-in with the shield's Z value COPIED, at the start of its hold (where the device reads it), onto the
    free qubit: H . CZ(shield, free) . H, as rz(pi/2) sx rz(pi/2) around the CZ; the free qubit read at the end into
    a NEW last classical bit. Nothing else moves. The copy is exact only on a zero-error, zero-duration edge and a
    noiseless free qubit (mcm_snapshot); its read-out then IS the mid-circuit read's assignment."""
    from qiskit import ClassicalRegister, QuantumCircuit
    hold = next((k for k, inst in enumerate(qc.data) if inst.operation.name == "delay"
                 and qc.find_bit(inst.qubits[0]).index == shield), None)
    if hold is None:
        raise ValueError(f"no hold on shield {shield}: not a read shield of this stand-in")
    out = QuantumCircuit(*qc.qregs, *qc.cregs, ClassicalRegister(1, "mcm_copy"), name=qc.name)
    f, s = out.qubits[free], out.qubits[shield]
    for k, inst in enumerate(qc.data):
        if k == hold:
            for gate in ("h1", "cz", "h2"):
                if gate == "cz":
                    out.cz(s, f)
                else:
                    out.rz(math.pi / 2, f)
                    out.sx(f)
                    out.rz(math.pi / 2, f)
        out.append(inst.operation, inst.qubits, inst.clbits)
    out.measure(f, out.clbits[-1])
    return out


def mcm_snapshot(snap, shield: int, free: int, assign: float):
    """the model with a free qubit that cannot err (T1 = T2 = 1e6 s, no sx error or duration, no sigma or detuning),
    read out symmetrically at the shield's measured mid-circuit assignment, and a zero-error zero-duration edge to
    the shield"""
    from oneq.hw import snapshot as S

    def put(t, v):
        t = list(t)
        t[free] = v
        return tuple(t)
    cz = dict(snap.cz)
    cz[(min(shield, free), max(shield, free))] = (0.0, 0.0)
    d = {**snap.__dict__, "cz": cz, "t1": put(snap.t1, 1e6), "t2": put(snap.t2, 1e6), "sx_error": put(snap.sx_error, 0.0),
         "sx_duration": put(snap.sx_duration, 0.0), "p01": put(snap.p01, assign), "p10": put(snap.p10, assign),
         "sigma": {q: v for q, v in snap.sigma.items() if q != free},
         "detuning": {q: v for q, v in snap.detuning.items() if q != free}}
    return S.DeviceSnapshot(**d)


def read_circuits(stand, unit_rec, model, params, assign: dict, layout0: list) -> tuple:
    """(params as played, [(j, t, shield, free qubit, circuit with the copy, its snapshot, detector clbits)]) for
    every READ shield of an R147 dynamic unit, each detector checked FIXED in the noiseless stand-in (refused if
    not). The template is played on the stand-in FIRST, then the copy inserted: the copy is an instrument, and
    resynthesis would give the read shield a spurious slot at the copy's zero-duration CZ."""
    import hw_dynamic as D
    import hw_dynamic_run as R147
    uu = R147.unit_from(unit_rec["unit"])
    stand, params = RA.played(stand, model, params)
    out = []
    for j, t in enumerate(D.measure_type(uu.choice)):
        s_q = layout0[4 + t]
        f = free_qubit(stand, model)
        qc = with_mcm_copy(stand, s_q, f)
        sn = mcm_snapshot(model, s_q, f, assign[str(s_q)])
        m = qc.num_clbits
        cl = [1 + t, m - 1]
        P0 = E.plan(qc, sn, N.NOISELESS)
        q_hat = E.wht(E._true_probs(E.simulator(len(P0.active), "numpy").run(P0.ops, {}), P0.meas))
        if abs(q_hat[sum(1 << (m - 1 - c) for c in cl)] - 1.0) > TOL:
            raise ValueError(f"{unit_rec['name']}: mcm XOR pair on shield {s_q} is not fixed in the stand-in")
        out.append((j, t, s_q, f, qc, sn, cl))
    return params, out


def read_rows(stand, unit_rec, index, model, params, traj, assign: dict, layout0: list) -> list:
    """R147 dynamic unit (beta = 0): one syndrome row per READ shield -- its mid-circuit bit XOR its pair's decode
    bit -- predicted on the stand-in with the exact copy (read_circuits), with its DEM column"""
    slots = not params.twirl_template
    params, circuits = read_circuits(stand, unit_rec, model, params, assign, layout0)
    rows = []
    for j, t, s_q, f, qc, sn, cl in circuits:
        sc = violation_score(cl, 1)
        mcm_bit = 8 + j
        k = n = 0
        for key, c in unit_rec["_counts"].items():
            b = T.bits_of(key, unit_rec["n_bits"])
            n += int(c)
            k += int(c) * int(b[mcm_bit] != b[1 + t])
        pred = E.expectation(qc, sc, sn, params, traj, np.random.default_rng([173, index, j]))
        L = dem_column(qc, sc, sn, params, with_slots=slots)
        v = L.pop(f"readout@q{f}", 0.0)                 # the free qubit's read-out IS the mid-circuit assignment:
        L["readout"] -= v                                 # its own class, on the shield it reads
        L["mcm_assign"] = v
        L[f"q{s_q}"] = L.get(f"q{s_q}", 0.0) + L.pop(f"q{f}", 0.0)
        rows.append(dict(unit=unit_rec["name"], unit_index=index, kind=unit_rec["kind"], weight=unit_rec["weight"],
                         role="read", clbits=cl, sign=1, qubit=s_q, pair_qubit=layout0[9 + t], k=k, n=n,
                         measured=k / n, predicted=pred.mean, var=pearson_var(pred.mean, n, pred.se),
                         L=RA.compact(L)))
    return rows


# ------------------------------------------------------------------ the key's own wait (a mechanism test)
KEY_HOLDS_S = (0.52e-6, 2.7e-6)   # R166's feed-forward at 2.7 us minus R147's 2.18 us read (the most an EARLY key,
                                  # prepared when the read ends, can wait); and the whole 2.7 us (an INLINE key's wait)


def with_key_hold(qc, key: int, virt: list, hold_dt: int):
    """the EARLY stand-in with the key held hold_dt immediately before its first virtual CZ (where the device runs the
    if_else on the key): a delay on the key ALONE -- no barrier, no other qubit moved (R166: a qubit independent of
    the conditional is not stalled). ALAP then prepares the key, and copies it, hold_dt earlier."""
    from qiskit.circuit import Delay
    if type(hold_dt) is not int or hold_dt < 0:
        raise ValueError(f"a hold is a whole number of dt >= 0: {hold_dt!r}")
    pairs = {tuple(sorted(map(int, e))) for e in virt}
    first = next((k for k, inst in enumerate(qc.data) if inst.operation.name == "cz"
                  and tuple(sorted(qc.find_bit(b).index for b in inst.qubits)) in pairs), None)
    if first is None:
        raise ValueError(f"no CZ on the virtual edges {sorted(pairs)}")
    if key not in {qc.find_bit(b).index for b in qc.data[first].qubits}:
        raise ValueError(f"the first virtual CZ does not act on the key {key}")
    out = qc.copy_empty_like()
    for k, inst in enumerate(qc.data):
        if k == first and hold_dt:
            out.append(Delay(hold_dt, unit="dt"), [qc.qubits[key]])
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def feed_forward(virt: list, n_reads: int, name: str = "") -> bool:
    """does the unit run an if_else on the key? A unit whose shields are all X or Y reads none of them mid-circuit:
    no virtual edge and no conditional (hw_dynamic._key_part returns before it), so nothing for the key to wait on --
    held 0 in EVERY key-hold arm by construction, and named (the dynamic design's no-feed-forward control). The
    record's virtual edges and the unit's read count must agree; a disagreement is a wrong lookup, refused."""
    if bool(virt) != bool(n_reads):
        raise ValueError(f"{name}: {n_reads} mid-circuit reads but virtual edges {virt}")
    return bool(virt)


def key_hold_arms(rec147, stand_ins, counts, model, params, traj, holds=KEY_HOLDS_S, limit=None) -> dict:
    """per hold (0 = the stand-in as frozen): every EARLY dynamic unit's predicted score (c0 / c1 mixed as R147
    predicts) and its copy-bit (ell) violation, against the measured -- and the 16 matched twins' early - inline
    gap. NOTHING is fitted: the holds come from R166 / R147's own numbers"""
    import hw_dynamic as D
    import hw_dynamic_run as R147
    lay0 = rec147["layout_sector0"]
    key = lay0[0]
    dt = rec147["snapshot"]["dt"]
    inline = {}                                          # the inline twins as frozen: their key already waits
    for k, u in enumerate(rec147["units"]):
        if u["kind"] != "twin_inline" or (limit is not None and k >= limit):
            continue
        inp = u["prediction_inputs"]
        uu = R147.unit_from(u["unit"])
        sn = D.augmented(model, [tuple(e) for e in inp["virt"]])
        r = 0.5 * (1 - math.prod(1 - 2 * model.readout_error(q) for q in inp["mix_qubits"]))
        sc = R147.FA.score(R147.M, uu, R147.DECODER)
        c0, pp = RA.played(stand_ins[inp["c0"]], sn, params)
        e0 = E.expectation(c0, sc, sn, pp, traj, np.random.default_rng([176, k, 0]))
        e1 = (E.expectation(RA.played(stand_ins[inp["c1"]], sn, params)[0], sc, sn, pp, traj,
                            np.random.default_rng([176, k, 1])) if inp["c1"] else e0)
        stat = T.Unit(u["name"], u["weight"], rec147["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"])
        t = T.tally(counts["r147__" + u["name"]], R147._score(u), stat)
        inline[u["name"].split("__")[1]] = dict(predicted=(1 - r) * e0.mean + r * e1.mean, measured=t.mean)
    out = {}
    for hold in (0.0,) + tuple(holds):
        h_dt = int(round(hold / dt / 16)) * 16
        rows = []
        for k, u in enumerate(rec147["units"]):
            if limit is not None and k >= limit:
                break
            inp = u["prediction_inputs"]
            if inp["kind"] != "dynamic" or u["kind"] != "main":
                continue
            uu = R147.unit_from(u["unit"])
            score = R147.FA.score(R147.M, uu, R147.DECODER)
            sn = D.augmented(model, [tuple(e) for e in inp["virt"]])
            n_reads = len(D.measure_type(uu.choice))
            ff = feed_forward(inp["virt"], n_reads, u["name"])
            hold_c = (lambda qc: with_key_hold(qc, key, inp["virt"], h_dt)) if ff else (lambda qc: qc)
            c0, pp = RA.played(hold_c(stand_ins[inp["c0"]]), sn, params)
            c1 = RA.played(hold_c(stand_ins[inp["c1"]]), sn, params)[0] if inp["c1"] else None
            r = 0.5 * (1 - math.prod(1 - 2 * model.readout_error(q) for q in inp["mix_qubits"]))
            e0 = E.expectation(c0, score, sn, pp, traj, np.random.default_rng([174, k, 0]))
            e1 = E.expectation(c1, score, sn, pp, traj, np.random.default_rng([174, k, 1])) if c1 else e0
            pred = (1 - r) * e0.mean + r * e1.mean
            sd_pred = math.hypot((1 - r) * e0.se, r * e1.se)
            stat = T.Unit(u["name"], u["weight"], rec147["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"])
            t = T.tally(counts["r147__" + u["name"]], R147._score(u), stat)
            row = dict(name=u["name"], weight=u["weight"], measured=t.mean, predicted=pred,
                       var=max(t.s2 / t.n - t.mean ** 2, 1e-6) / t.n + sd_pred ** 2, n_reads=n_reads,
                       no_feed_forward=not ff)
            ell = [(cl, sg) for cl, sg, rl in unit_syndromes(c0, sn, 5)[1] if rl == "ell"]
            if ell:
                cl, sg = ell[0]
                kk, nn = measured_violations(counts["r147__" + u["name"]], cl, sg, u["n_bits"])
                pe = E.expectation(c0, violation_score(cl, sg), sn, pp, traj, np.random.default_rng([175, k]))
                row.update(ell_k=kk, ell_n=nn, ell_pred=pe.mean, ell_se=pe.se)
            rows.append(row)
        res = np.array([x["measured"] - x["predicted"] for x in rows])
        sd = math.sqrt(sum(x["var"] for x in rows)) / len(rows)
        er = [x for x in rows if "ell_k" in x]
        k_tot, n_tot = sum(x["ell_k"] for x in er), sum(x["ell_n"] for x in er)
        p_pred = sum(x["ell_pred"] * x["ell_n"] for x in er) / n_tot
        tw = [(x, inline[x["name"].split("__")[0]]) for x in rows if x["name"].split("__")[0] in inline]
        dm = np.array([a["measured"] - b["measured"] for a, b in tw])
        dp = np.array([a["predicted"] - b["predicted"] for a, b in tw])
        by_reads = {}
        for nr in sorted({x["n_reads"] for x in rows}):
            g = [x for x in rows if x["n_reads"] == nr]
            by_reads[str(nr)] = dict(units=len(g), mean_residual=float(np.mean([x["measured"] - x["predicted"]
                                                                                   for x in g])),
                                     sd=math.sqrt(sum(x["var"] for x in g)) / len(g))
        out[f"{hold * 1e6:g} us"] = dict(
            hold_dt=h_dt, units=len(rows), main_mean_residual=float(res.mean()), sd=sd, by_reads=by_reads,
            no_feed_forward_units=[dict(name=x["name"], reason="no measure-type shield: no mid-circuit read and no "
                                        "if_else on the key, so no feed-forward to wait on; held 0 in every arm")
                                   for x in rows if x["no_feed_forward"]],
            ell=dict(units=len(er), measured=k_tot / n_tot, predicted=p_pred,
                     sd=math.sqrt(max(p_pred * (1 - p_pred), 1e-12) / n_tot)),
            twins_early_minus_inline=dict(n=len(tw), measured=float(dm.mean()) if tw else None,
                                          sd=float(dm.std(ddof=1) / math.sqrt(len(tw))) if len(tw) > 1 else None,
                                          predicted=float(dp.mean()) if tw else None),
            rows=rows)
    return out


# ------------------------------------------------------------------ the read-shield pairs (model-free)
def read_shield_pairs(rec147: dict, counts: dict, model, mcm_assign: dict) -> dict:
    """R147's dynamic units: for every shield read mid-circuit, the detector (its mid-circuit bit) XOR (its pair's
    final decode bit) -- 0 ideally (the read projects the pair onto |00> or |11>, which the decoder maps to 0 / 1).
    Per shield: the violation rate, split by the bit read, against the floor of two measured read-outs: the
    shield's mid-circuit assignment (MCMCAL) + the pair qubit's in-circuit read-out (p01 + p10)/2."""
    import hw_dynamic as D
    import hw_dynamic_run as R147
    lay = rec147["layout_sector0"]
    agg: dict = {}
    for u in rec147["units"]:
        inp = u["prediction_inputs"]
        if inp["kind"] != "dynamic" or u["unit"]["beta"] != 0:
            continue                                       # beta = 1 rotates the decode bit: no longer fixed
        uu = R147.unit_from(u["unit"])
        mt = D.measure_type(uu.choice)
        npair = 5                                          # cz2: one decode bit per pair (a_t)
        for j, t in enumerate(mt):
            pair_bit, mcm_bit = 1 + t, npair + 3 + j
            s_q, a_q = lay[4 + t], lay[4 + 5 + t]
            g = agg.setdefault(str(s_q), dict(pair_qubit=a_q, k=[0, 0], n=[0, 0], kinds={}))
            for key, c in counts["r147__" + u["name"]].items():
                b = T.bits_of(key, u["n_bits"])
                z = b[mcm_bit]
                g["n"][z] += int(c)
                g["k"][z] += int(c) * int(b[mcm_bit] != b[pair_bit])
                kk = g["kinds"].setdefault(u["kind"], [0, 0])
                kk[0] += int(c) * int(b[mcm_bit] != b[pair_bit])
                kk[1] += int(c)
    out = {}
    for s_q, g in agg.items():
        a_q = g["pair_qubit"]
        floor = mcm_assign.get(s_q, float("nan")) + 0.5 * (model.p01[a_q] + model.p10[a_q])
        n, k = sum(g["n"]), sum(g["k"])
        p = k / n
        sd = math.sqrt(max(p * (1 - p), 0.25 / n) / n)
        by = {str(z): dict(shots=g["n"][z], rate=g["k"][z] / g["n"][z] if g["n"][z] else None) for z in (0, 1)}
        out[s_q] = dict(pair_qubit=a_q, shots=n, rate=p, sd=sd, readout_floor=floor, excess=p - floor,
                        excess_z=(p - floor) / sd, by_bit_read=by,
                        one_minus_zero=(by["1"]["rate"] or 0) - (by["0"]["rate"] or 0),
                        by_kind={kd: dict(shots=v[1], rate=v[0] / v[1]) for kd, v in g["kinds"].items()})
    return out


# ------------------------------------------------------------------ main
EXTRA = ("mcm_assign",)                     # the read rows' own class (the free qubit's read-out = the mid-circuit read)
ROLES = ("ell", "pair", "read")


def fit_suite(rows: list) -> dict:
    present = [k for k in RA.CLASSES + EXTRA + ("slot",) if any(abs(r["L"].get(k, 0.0)) > 0 for r in rows)]
    q = sorted({k for r in rows for k in r["L"] if k[0] == "q" and k[1:].isdigit()}, key=lambda s: int(s[1:]))
    e = sorted({k for r in rows for k in r["L"] if k.startswith("cz_")})
    plan = dict(classes=[k for k in present if k != "slot"], classes_slot=present, qubits=identifiable(rows, q),
                edges=identifiable(rows, e) + [k for k in present if k != "cz"])
    plan.update({f"only_{one}": [one] for one in present})
    return {name: A.fit(rows, keys) for name, keys in plan.items() if keys}


# ------------------------------------------------------------------ crash-safe stages
# Each stage is written to its own file the moment it is computed and REUSED on a rerun when its key matches: a hash
# of the source of every function the stage runs (inspect.getsource), of every file it reads or imports for its
# numbers, and of its parameters. A late-stage bug then costs that stage only (2026-10-17: key_hold_arms raised after
# 2 h of syndromes, and nothing had been written).
STAGE_FILES = ("src/oneq/hw/noise.py", "src/oneq/hw/engine.py", "src/oneq/hw/sensitivity.py",
               "src/oneq/hw/snapshot.py", "src/oneq/hw/stats.py",
               "experiments/superactivation/shield_family/hw_dynamic.py",
               "experiments/superactivation/shield_family/hw_dynamic_run.py",
               "experiments/superactivation/shield_family/hw_two_use.py",
               "experiments/superactivation/shield_family/hw_faithful.py",
               "experiments/superactivation/shield_family/hw_superact_run.py",
               "experiments/superactivation/shield_family/r152_posthoc.py",
               "experiments/superactivation/shield_family/r155_heldout_r152.py",
               "evidence/superactivation/campaign_r152/20261017T003152Z/r146.json",
               "evidence/superactivation/campaign_r152/20261017T003152Z/r146.qpy",
               "evidence/superactivation/campaign_r152/20261017T003152Z/r147.json",
               "evidence/superactivation/campaign_r152/20261017T003152Z/r147.qpy",
               "evidence/superactivation/campaign_r152/20261017T003152Z/r147.pred.qpy",
               "evidence/superactivation/campaign_r152/20261017T003152Z/counts_db9c50oc77vs73817rog.json",
               "evidence/superactivation/r155_diag/idle_map_db9dsqpgv2cc73e5c8fg.json")


def file_digests(root: pathlib.Path, files=STAGE_FILES) -> dict:
    import hashlib
    out = {}
    for f in files:
        p = root / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return out


def stage_key(stage: str, funcs, digests: dict, extra: dict) -> str:
    """the hash a stage is reused under: its name, the SOURCE of every function it runs, the digests of every file it
    depends on, and its parameters. Any change to any of them is a different key"""
    import hashlib
    import inspect
    src = {f"{f.__module__}.{f.__qualname__}": hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
           for f in funcs}
    blob = json.dumps(dict(stage=stage, functions=src, files=digests, extra=extra), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_stage(path: pathlib.Path, stage: str, key: str, compute, log=print) -> tuple:
    """(value, reused): the stage's value from `path` when it holds a COMPLETE record under the same key, else
    computed and written there at once (atomically: a temporary file, then a rename). The value comes back in its
    JSON form either way, so a reused stage and a computed one are indistinguishable to what follows."""
    import os
    if path.exists():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            rec = None
        if rec and rec.get("key") == key and rec.get("complete") is True and rec.get("stage") == stage:
            log(f"stage {stage}: REUSED {path.name} (key {key[:12]})")
            return rec["value"], True
    t0 = time.time()
    value = json.loads(json.dumps(compute(), default=float))
    rec = dict(schema="oneq.r171.stage/1", stage=stage, key=key, complete=True, seconds=time.time() - t0,
               value=value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, separators=(",", ":"), default=float) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    log(f"stage {stage}: computed in {rec['seconds']:.0f} s, written to {path.name}")
    return value, False


def stage_funcs():
    common = (unit_syndromes, detector_group, min_weight_basis, clbits_of, role, violation_score, measured_violations,
              pearson_var, disjoint_pair_covariance, phys_of, dem_column, syndrome_rows, RA.played, RA.class_losses,
              RA.slot_sensitivity, RA.slot_derivatives, RA.slots_of, RA._propagate, RA._propagate_one, RA._flip,
              RA.compact)
    return dict(r146_syndromes=common + (r146_syndromes,),
                r147_syndromes=common + (r147_syndromes, read_rows, read_circuits, free_qubit, with_mcm_copy,
                                         mcm_snapshot),
                key_hold=(key_hold_arms, feed_forward, with_key_hold, unit_syndromes, detector_group, min_weight_basis, clbits_of,
                          role, measured_violations, violation_score, RA.played))


def preflight(art, rec147, held, m6, m7, p6, p7, assign, limit=None, log=print) -> dict:
    """every stage's per-unit path on EVERY input before the hours-long run: each unit's detector basis on its
    circuit as played (R146, R147), every read shield's copy circuit with its detector checked fixed, and the
    key-hold arms run whole at 2 trajectories. Raises on the first failure; returns what it exercised"""
    import hw_dynamic as D
    from oneq.hw import freeze as F
    t0 = time.time()
    d = art["dir"]
    c146 = F.load(art["parts"]["r146"], (d / "r146.qpy").read_bytes(), check_sources=False)
    n6 = 0
    for k, u in enumerate(art["parts"]["r146"]["units"]):
        if (limit is not None and k >= limit) or u["unit"]["branch"] == "erased_erased":
            continue
        qc, _ = RA.played(c146[u["name"]], m6, p6)
        n6 += bool(unit_syndromes(qc, m6, 5)[1])
    log(f"preflight: R146 detector bases {n6} ({time.time() - t0:.0f} s)")
    c147 = F.load(rec147, (d / "r147.qpy").read_bytes(), check_sources=False)
    n7 = nr = 0
    for k, u in enumerate(rec147["units"]):
        if (limit is not None and k >= limit) or u["unit"]["branch"] == "erased_erased":
            continue
        inp = u["prediction_inputs"]
        if inp["kind"] == "dynamic":
            sn = D.augmented(m7, [tuple(e) for e in inp["virt"]])
            qc, _ = RA.played(held[inp["c0"]], sn, p7)
            if inp["c1"]:
                RA.played(held[inp["c1"]], sn, p7)
            if u["unit"]["beta"] == 0:
                nr += len(read_circuits(held[inp["c0"]], u, sn, p7, assign, rec147["layout_sector0"])[1])
        else:
            qc, _ = RA.played(c147[u["name"]], m7, p7)
            sn = m7
        n7 += bool(unit_syndromes(qc, sn, 5)[1])
    log(f"preflight: R147 detector bases {n7}, read-shield circuits {nr} ({time.time() - t0:.0f} s)")
    kh = key_hold_arms(rec147, held, art["counts"], m7, p7, 2, limit=limit)
    log(f"preflight: key-hold arms at 2 trajectories, {[v['units'] for v in kh.values()]} units "
        f"({time.time() - t0:.0f} s)")
    return dict(r146_bases=n6, r147_bases=n7, read_circuits=nr, key_hold_units={h: v["units"] for h, v in kh.items()},
                no_feed_forward=[x["name"] for v in kh.values() for x in v["no_feed_forward_units"]],
                seconds=time.time() - t0)


# ------------------------------------------------------------------ error bars that count the twirl's frames
def hardware_detector(row: dict, unit_rec: dict) -> tuple:
    """(clbits, sign) of a syndrome row on the HARDWARE bits: an ell / pair row reads the protocol's own bits; a READ
    row compares the pair's decode bit 1 + t with the shield's mid-circuit bit 8 + j (j its rank among the reads)"""
    if row["role"] != "read":
        return list(row["clbits"]), int(row["sign"])
    import hw_dynamic as D
    import hw_dynamic_run as R147
    t = row["clbits"][0] - 1
    j = D.measure_type(R147.unit_from(unit_rec["unit"]).choice).index(t)
    return [1 + t, 8 + j], 1


def violation_counts(counts: dict, specs: list, cache: dict | None = None) -> list:
    """per spec (pub, clbits, sign, n_bits): (violating shots, shots) in `counts`; cache keeps each outcome's verdict"""
    cache = {} if cache is None else cache
    out = []
    for i, (pub, cl, sg, nb) in enumerate(specs):
        k = n = 0
        for key, c in counts[pub].items():
            v = cache.get((i, key))
            if v is None:
                v = cache[(i, key)] = int(violation_score(cl, sg)(T.bits_of(key, nb)))
            n += c
            k += c * v
        out.append((k, n))
    return out


def syndrome_statistics(rows6: list, rows7: list, kn6: list, kn7: list) -> dict:
    """the item-18 headline numbers with every row's violations replaced by (k, n): rates per role x kind, the class
    factors of both protocols, the read shields' rates"""
    def upd(rows, kn):
        return [dict(r, k=k, n=n, measured=k / n) for r, (k, n) in zip(rows, kn)]
    r6, r7 = upd(rows6, kn6), upd(rows7, kn7)
    out = {}
    for proto, rr in (("r146", r6), ("r147", r7)):
        groups: dict = {}
        for r in rr:
            groups.setdefault(f"{r['role']}|{r['kind']}", []).append(r)
        for g, v in groups.items():
            out[f"{proto}.rate.{g}"] = sum(x["k"] for x in v) / sum(x["n"] for x in v)
        present = [k for k in RA.CLASSES + EXTRA if any(abs(r["L"].get(k, 0.0)) > 0 for r in rr)]
        f = A.fit(rr, present)
        out.update({f"{proto}.classes.{k}": b for k, b in f["beta"].items()})
    shields: dict = {}
    for r in r7:
        if r["role"] == "read":
            shields.setdefault(str(r["qubit"]), []).append(r)
    for q, v in shields.items():
        out[f"r147.read_rate.q{q}"] = sum(x["k"] for x in v) / sum(x["n"] for x in v)
    return out


def bootstrap_syndromes(root: pathlib.Path, tag: str, n_boot: int) -> dict:
    """the frame and shot bootstraps of the stored stages (no engine): syndrome rates and factors, and per key hold the
    main units' mean residual and copy-bit rate"""
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    from oneq.hw import randomization as RZ
    sd = root / RA.OUT / "stages"
    st = {n: json.loads((sd / f"{n}_{tag}.json").read_text(encoding="utf-8")) for n in
          ("r146_syndromes", "r147_syndromes", "key_hold")}
    art = P152.load(root / HO.R152_DIR)
    recs = {u["name"]: u for part in ("r146", "r147") for u in art["parts"][part]["units"]}
    rows6, rows7 = st["r146_syndromes"]["value"]["rows"], st["r147_syndromes"]["value"]["rows"]
    pub = {n: ("r146__" if n in {u["name"] for u in art["parts"]["r146"]["units"]} else "r147__") + n for n in recs}
    spec6 = [(pub[r["unit"]], *hardware_detector(r, recs[r["unit"]]), recs[r["unit"]]["n_bits"]) for r in rows6]
    spec7 = [(pub[r["unit"]], *hardware_detector(r, recs[r["unit"]]), recs[r["unit"]]["n_bits"]) for r in rows7]
    kh = st["key_hold"]["value"]
    specs_u = RA.unit_specs(art)
    kh_units = sorted({x["name"] for v in kh.values() for x in v["rows"]})
    ell_spec = {}
    for v in kh.values():
        for x in v["rows"]:
            if "ell_k" in x and x["name"] not in ell_spec:
                r = next(r for r in rows7 if r["unit"] == x["name"] and r["role"] == "ell")
                ell_spec[x["name"]] = (pub[x["name"]], list(r["clbits"]), int(r["sign"]), recs[x["name"]]["n_bits"])
    names = sorted({s[0] for s in spec6 + spec7} | {pub[n] for n in kh_units})
    index = dict(art["index"], execution_spans=art["spans"])
    shots = RZ.from_flight(index, art["arrays"], names=names)
    c6, c7, ce, cu = {}, {}, {}, {}
    ell_names = sorted(ell_spec)

    def estimator(counts):
        out = syndrome_statistics(rows6, rows7, violation_counts(counts, spec6, c6), violation_counts(counts, spec7, c7))
        means = RA.means_from(counts, specs_u, kh_units, cu)
        ell = dict(zip(ell_names, violation_counts(counts, [ell_spec[n] for n in ell_names], ce)))
        for h, v in kh.items():
            res = [means[x["name"]] - x["predicted"] for x in v["rows"]]
            out[f"key_hold.{h}.main_mean_residual"] = float(np.mean(res))
            er = [x["name"] for x in v["rows"] if "ell_k" in x]
            out[f"key_hold.{h}.ell_rate"] = sum(ell[n][0] for n in er) / sum(ell[n][1] for n in er)
        return out
    return RA.frame_bootstrap(shots, estimator, n_boot=n_boot)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=96, help="blocked-error trajectories per prediction: a violation "
                    "rate of ~2%% needs ~4x the score's 24 for its Monte Carlo sd to sit below the shot noise")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", choices=("template", "dressing"), default="template",
                    help="template: V3c + the engine's R166 template + dressing x2 (the newest model); dressing: "
                         "V3c + dressing x1.54 with the slot class")
    ap.add_argument("--preflight", action="store_true",
                    help="exercise every stage's per-unit path on every input (key-hold at 2 trajectories) and stop")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="frame and shot bootstraps (N resamples) of the stored stages of this --base/--traj/--limit "
                         "(pure; no engine) -> bootstrap_syndromes_r152.json")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    if a.bootstrap:
        t0 = time.time()
        tag = f"{a.base}_traj{a.traj}" + ("" if a.limit is None else f"_limit{a.limit}")
        res = bootstrap_syndromes(root, tag, a.bootstrap)
        rec = dict(schema="oneq.r171.bootstrap/1", label="EXPLORATORY -- outside every preregistered verdict",
                   stages=tag, n_boot=a.bootstrap, provenance=RA.provenance(root), statistics=res,
                   seconds=time.time() - t0)
        (root / RA.OUT / "bootstrap_syndromes_r152.json").write_text(RA.evidence_text(rec), encoding="utf-8",
                                                                      newline="\n")
        for k, v in res.items():
            print(f"{k:58s} {v['estimate']:+.4f}  sd frames {v['sd_frames']:.4f}  shots {v['sd_shots']:.4f}  "
                  f"x{v['inflation']:.2f}")
        return 0
    import psutil
    floor = 1.5e9 if a.preflight else 4e9                 # the preflight holds ~0.3 GB; a full run, hours of it
    if psutil.virtual_memory().available < floor:
        raise SystemExit(f"less than {floor / 1e9:g} GB available: not starting")
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    t0 = time.time()
    prov = RA.provenance(root)
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    m6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    kw = RA.TEMPLATE_KW if a.base == "template" else RA.MODEL_KW
    p6 = N.NoiseParams(**{**P["params146"].describe(), **kw})
    p7 = N.NoiseParams(cz_factor=1.0, schedule=P["params146"].schedule, **kw)
    rec147 = P["record147"]
    held, _ = HO.with_hold(P["stand_ins"], rec147, RA.HOLD_MEASURED_S)
    post = P152.double_read(art)                          # MCMCAL: per shield, prepared 0 and 1
    mcm = {}
    for prep in post.values():
        for q, row in prep["per_qubit"].items():
            mcm.setdefault(q, []).append(row["misassignment_first_read"] + row["state_wrong_at_first_read"])
    mcm_assign = {q: float(np.mean(v)) for q, v in mcm.items()}
    if a.preflight:
        print("preflight OK", json.dumps(preflight(art, rec147, held, m6, m7, p6, p7, mcm_assign, a.limit)), flush=True)
        return 0
    digests = file_digests(root)
    extra = dict(base=a.base, traj=a.traj, limit=a.limit, p6=p6.describe(), p7=p7.describe(),
                 hold_s=RA.HOLD_MEASURED_S, key_holds_s=list(KEY_HOLDS_S), mcm_assign=mcm_assign)
    funcs = stage_funcs()
    tag = f"{a.base}_traj{a.traj}" + ("" if a.limit is None else f"_limit{a.limit}")
    stage_dir = root / RA.OUT / "stages"
    stages: dict = {}

    def stage(name, compute):
        key = stage_key(name, funcs[name], digests, extra)
        path = stage_dir / f"{name}_{tag}.json"
        value, reused = run_stage(path, name, key, compute, log=lambda m: print(m, flush=True))
        import hashlib
        stages[name] = dict(path=path.relative_to(root).as_posix(), key=key, reused=reused,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return value
    st6 = stage("r146_syndromes", lambda: dict(zip(("rows", "cov", "logical"),
                                                   r146_syndromes(art, m6, p6, a.traj, a.limit))))
    s6, cov6, log6 = st6["rows"], st6["cov"], st6["logical"]
    print(f"R146 syndromes: {len(s6)} ({time.time() - t0:.0f} s)", flush=True)
    st7 = stage("r147_syndromes", lambda: dict(zip(("rows", "cov", "logical", "check"),
                                                   r147_syndromes(art, rec147, held, m7, p7, a.traj, mcm_assign,
                                                                  a.limit))))
    s7, cov7, log7, chk = st7["rows"], st7["cov"], st7["logical"], st7["check"]
    print(f"R147 syndromes: {len(s7)} ({time.time() - t0:.0f} s)", flush=True)
    key_hold = stage("key_hold", lambda: key_hold_arms(rec147, held, art["counts"], m7, p7, a.traj, limit=a.limit))
    print(f"key hold arms ({time.time() - t0:.0f} s)", flush=True)
    out: dict = dict(schema="oneq.r171.syndromes/2", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, limit=a.limit, base=a.base, provenance=prov, stages=stages,
                     models=dict(r146=dict(base="V3c", **p6.describe()),
                                 r147=dict(base="V3c", hold_s=RA.HOLD_MEASURED_S, **p7.describe())),
                     counts=dict(r146=dict(rows=len(s6), roles={r: sum(x["role"] == r for x in s6) for r in ROLES}),
                                 r147=dict(rows=len(s7), roles={r: sum(x["role"] == r for x in s7) for r in ROLES})),
                     sanity=dict(max_measured_violation=max(x["measured"] for x in s6 + s7), c1_vs_c0=chk))
    out["fits"] = dict(r146=fit_suite(s6), r147=fit_suite(s7))
    for proto, rows in (("r146", s6), ("r147", s7)):
        for rl in ROLES:
            sub = [r for r in rows if r["role"] == rl]
            if sub:
                out["fits"][f"{proto}_{rl}"] = fit_suite(sub)
    out["per_qubit_ratio"] = {f"{proto}_{rl}": per_location_ratio([r for r in rows if r["role"] == rl], "qubit")
                              for proto, rows in (("r146", s6), ("r147", s7)) for rl in ROLES}
    out["per_role_kind"] = {proto: per_location_ratio([dict(r, rk=f"{r['role']}|{r['kind']}") for r in rows], "rk")
                            for proto, rows in (("r146", s6), ("r147", s7))}
    out["heldout"] = {}
    for proto, rows in (("r146", s6), ("r147", s7)):
        cls = list(RA.CLASSES) + (list(EXTRA) if proto == "r147" else [])
        even, odd = split(rows)
        q = sorted({k for r in rows for k in r["L"] if k[0] == "q" and k[1:].isdigit()}, key=lambda s: int(s[1:]))
        out["heldout"][proto] = dict(
            classes_even_to_odd=heldout(even, odd, cls), classes_odd_to_even=heldout(odd, even, cls),
            classes_slot_even_to_odd=heldout(even, odd, cls + ["slot"]),
            qubits_even_to_odd=heldout(even, odd, q), qubits_odd_to_even=heldout(odd, even, q))
    base_cls = list(RA.CLASSES)
    out["heldout"]["r146_to_r147_classes"] = heldout(s6, s7, base_cls)
    out["heldout"]["r146_to_r147_classes_slot"] = heldout(s6, s7, base_cls + ["slot"])
    re = root / RA.OUT / "reattribution_r152.json"
    if re.exists() and a.limit is None:                    # the item-17 score rows, same model, same units
        sc = json.loads(re.read_text(encoding="utf-8"))["rows"]
        score_arm6, score_arm7 = ("r146_template", "r147_template") if a.base == "template" else ("r146", "r147_physical")
        idx6 = {u["name"]: k for k, u in enumerate(art["parts"]["r146"]["units"])}
        idx7 = {u["name"]: k for k, u in enumerate(rec147["units"])}
        for proto, srows, rows, idx in (("r146", sc[score_arm6], s6, idx6), ("r147", sc[score_arm7], s7, idx7)):
            for r in srows:
                r["unit_index"] = idx[r["name"]]
            se, so = split(srows)
            ye, yo = split(rows)
            cls = base_cls
            q = sorted({k for r in rows for k in r["L"] if k[0] == "q" and k[1:].isdigit()}, key=lambda s: int(s[1:]))
            out["heldout"][f"{proto}_syndromes_to_score"] = dict(
                classes_even_to_odd=to_score(ye, so, cls), classes_odd_to_even=to_score(yo, se, cls),
                classes_slot_even_to_odd=to_score(ye, so, cls + ["slot"]),
                classes_slot_odd_to_even=to_score(yo, se, cls + ["slot"]),
                qubits_even_to_odd=to_score(ye, so, q), qubits_odd_to_even=to_score(yo, se, q))
    out["locality"] = dict(r146=dict(measured=pooled(cov6), excess_over_model=pooled(cov6, "model_cov"),
                                     model=pooled([dict(r, cov=r["model_cov"]) for r in cov6])),
                           r147=dict(measured=pooled(cov7), excess_over_model=pooled(cov7, "model_cov"),
                                     model=pooled([dict(r, cov=r["model_cov"]) for r in cov7])))
    out["read_shield_pairs"] = read_shield_pairs(rec147, art["counts"], m7, mcm_assign)
    out["key_hold"] = {h: {k: x for k, x in v.items() if k != "rows"} for h, v in key_hold.items()}   # rows: stage
    for h, v in out["key_hold"].items():
        print("key hold", h, json.dumps(v, default=float)[:900])
    out["mcm_assignment"] = mcm_assign
    out["seconds"] = time.time() - t0                    # rows, pairs and logical detectors live in the stage files
    d = root / RA.OUT
    d.mkdir(parents=True, exist_ok=True)
    name = ("syndromes_r152" + ("" if a.base == "template" else "_dressing")
            + ("" if a.limit is None else f"_smoke{a.limit}") + ".json")
    (d / name).write_text(RA.evidence_text(out), encoding="utf-8", newline="\n")
    for proto in ("r146", "r147"):
        for fname in ("classes", "classes_slot"):
            f = out["fits"][proto][fname]
            print(proto, fname, {k: f"{f['beta'][k]:+.2f}+-{f['sd'][k]:.2f}" for k in f["keys"]},
                  f"chi2 {f['chi2_no_excess']:.0f} -> {f['chi2_fit']:.0f} (n {f['n']}, cond {f['condition']:.0f})")
    for p, h in out["heldout"].items():
        items = h.items() if "test" not in h else [("", h)]
        for k, v in items:
            t = v.get("test", {})
            print("heldout", p, k, {x: round(y, 2) for x, y in t.items() if isinstance(y, float)},
                  {x: round(v[x], 4) for x in ("F_e_predicted_shift", "F_e_measured_residual") if x in v})
    print("locality", json.dumps({p: out["locality"][p] for p in ("r146", "r147")}))
    print("read-shield pairs", json.dumps({q: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                               if k in ("rate", "sd", "readout_floor", "excess", "excess_z",
                                                        "one_minus_zero")}
                                           for q, r in out["read_shield_pairs"].items()}))
    print("wrote", name, f"({out['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
