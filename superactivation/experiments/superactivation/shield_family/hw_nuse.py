"""The protocol with n >= 2 channel uses at the SAME quantum-decoding cost: code |0_L> = |+>^n, |1_L> = |->^n (R147).

THE BAR IS THE SAME FOR EVERY n. Neither P_d nor A_{2d} alone exceeds 3/4 with ANY number of uses (R142, point 3: P_d^n
is PPT, so F <= 1/2; A^n is antidegradable, so F <= 3/4). So any n-use code above 3/4 is a superactivation witness
of the same standing as the two-use one.

THE DECODER. Bob quantum-decodes ONE kept use (the first, by the code's permutation symmetry relabelled use 0) exactly
as before, and reads ONE Z bit from every other use: the copy c_u through A if that use was kept (exact), P's key in Z
if it was erased (flipped with probability p). The Z bits fix the decoded key's frame X^(XOR of the bits); the logical
operators are single-qubit products (Z_L = X_0, X_L = Z_0 Z_1 ... Z_{n-1}). If every use was erased, Bob holds n
classical bits and outputs |+> or |-> by their parity (entanglement fidelity at most 1/2, the PPT ceiling).

THE FIDELITY (exact; checked against density matrices for n = 2, 3, 4 at three p and against these circuits here):
    F_n = sum_{e=0}^{n-1} C(n,e)/2^n (1 - P_e) + 2^-n (1 - P_n)/2,      P_e = (1 - (1 - 2p)^e)/2,
the probability that e erased bits carry an odd number of flips. n = 2 is (7 - 6p + 2p^2)/8. More uses shrink the
all-erased branch (2^-n, capped at 1/2) and add erased bits (each a chance of a frame flip): at d = 256, n = 3 gives
0.8641 and n = 4 0.8672 against 0.8317 for n = 2.

THE COST of each extra use: one code CNOT, a copy CNOT (kept) and one read-out -- no shields decoded. On heavy-hex,
n = 3 embeds with no routing: key0 on a degree-3 qubit whose neighbours are c0, key1, key2, and c1, c2 hang off key1,
key2.

Qubits (lean): keys 0..n-1, copies n..2n-1 (c0 = n; c_u = n + u for kept others), s[t] = 2n + t, a[t] = 2n + m + t.
Kept branch "kept_e<E>": use 0 decoded, uses n-1-E+1 .. n-1 erased (the LAST E of the others), flips: their parity.
Classical bits: l (c0), the pair bits, b_1 .. b_{n-1}, key0. All erased: key_0 .. key_{n-1}.

  python hw_nuse.py check              circuits' exact ideal F_e = formula, n = 2..4, m = 1..3, both decoders
  python hw_nuse.py hh  N M DEC UNITS TRAJ     heavy-hex (R145 calibration), dynamic P, early schedule
  python hw_nuse.py ion N M DEC UNITS TRAJ [--cz-factor X] [--device helios_typ|helios_max|ionq_aria]
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hardware_design as hd  # noqa: E402
import hw_dynamic as D  # noqa: E402
import hw_faithful as F  # noqa: E402
import hw_levers as V  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import layout as LY  # noqa: E402
from oneq.hw import noise as N  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "shield_family" / "nuse"
INPUTS = H.INPUTS
BOB_BASIS = H.BOB_BASIS


def p_odd(e, p):
    return (1 - (1 - 2 * p) ** e) / 2


def f_closed(n, m):
    p = H.p_flip(m)
    return (sum(math.comb(n, e) / 2 ** n * (1 - p_odd(e, p)) for e in range(n))
            + 0.5 ** n * 0.5 * (1 - p_odd(n, p)))


@dataclass(frozen=True)
class NUnit:
    n: int
    e: int                          # erased among the n-1 others; e = n means every use erased
    choice: tuple | None            # P's unitary on the decoded use
    beta: int
    inp: tuple
    parity: int                     # parity of the sector-1 flips of the erased others (lean)

    @property
    def name(self):
        ch = "none" if self.choice is None else f"{self.choice[0]}_{''.join(map(str, self.choice[1]))}"
        return f"n{self.n}_e{self.e}__{ch}__b{self.beta}__{self.inp[0]}{'p' if self.inp[1] > 0 else 'm'}__f{self.parity}"


def _code(qc, n, inp):
    H._prep(qc, 0, inp)
    for u in range(1, n):
        qc.cx(0, u)
    for u in range(n):
        qc.h(u)


def circuit(m, u: NUnit, decoder="cz2", dynamic=False, early=True):
    """the lean n-use circuit; dynamic: P's sector 0 by mid-circuit read + feed-forward (hw_dynamic)"""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    n = u.n
    s = [2 * n + t for t in range(m)]
    a = [2 * n + m + t for t in range(m)]
    q = QuantumRegister(2 * n + 2 * m, "q")
    if u.e == n:
        c = ClassicalRegister(n, "c")
        qc = QuantumCircuit(q, c, name=u.name)
        _code(qc, n, u.inp)
        if u.parity:
            qc.x(n - 1)
        qc.measure(list(range(n)), list(range(n)))
        return qc
    kept_others = list(range(1, n - u.e))
    erased_others = list(range(n - u.e, n))
    npair = F.n_pair_bits(m, u, decoder)
    c = ClassicalRegister(npair + n + 1, "c")
    regs = [q, c]
    mt = D.measure_type(u.choice) if dynamic else []
    mc = None
    if mt:
        mc = ClassicalRegister(len(mt), "mcm")
        regs.append(mc)
    qc = QuantumCircuit(*regs, name=u.name)
    mcm = list(mc) if mc is not None else []
    for t in range(m):
        qc.h(a[t])
        qc.cx(a[t], s[t])
    if dynamic and early:
        D.apply_p_dynamic(qc, u.choice, 0, s, mcm, part="shield")
    _code(qc, n, u.inp)
    qc.cx(0, n)
    if dynamic:
        D.apply_p_dynamic(qc, u.choice, 0, s, mcm, part="key" if early else "both")
    else:
        H._apply_p(qc, u.choice, key=0, shields=s)
    for k in kept_others:
        qc.cx(k, n + k)
    if u.parity and erased_others:
        qc.x(erased_others[-1])
    qc.cx(0, n)
    bits = []
    for t in range(m):
        bits += F.pair_decode(qc, a[t], s[t], u.beta, decoder)
    H._measure_in(qc, 0, BOB_BASIS[u.inp[0]][0])
    order = [n] + bits + [n + k for k in kept_others] + erased_others + [0]
    for j, qb in enumerate(order):
        qc.measure(qb, c[j])
    return qc


def score(m, u: NUnit, decoder="cz2"):
    basis, e = u.inp
    n = u.n
    if u.e == n:
        def ee(bits):
            if basis != "X":
                return 0.5
            return 1.0 if (1 - 2 * (sum(bits[:n]) % 2)) == e else 0.0
        return ee
    k = F.n_pair_bits(m, u, decoder)
    meas_basis, sign = BOB_BASIS[basis]
    e_prime = sign * e
    z_flips = meas_basis in "XY"
    x_flips = meas_basis in "ZY"

    def kept(bits):
        ell, par = bits[0], sum(bits[1:1 + k]) % 2
        bpar = sum(bits[1 + k:k + n]) % 2
        o = bits[k + n]
        if ell != u.beta:
            return 0.0
        want = e_prime * (-1) ** ((par if z_flips else 0) + ((ell ^ bpar) if x_flips else 0))
        return 2.0 if (1 - 2 * o) == want else 0.0
    return kept


def ideal(m, u, decoder="cz2"):
    from qiskit.quantum_info import Statevector
    qc = circuit(m, u, decoder)
    nb = qc.num_clbits
    probs = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
    qubit_of = {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index for i in qc.data
                if i.operation.name == "measure"}
    sc = score(m, u, decoder)
    return float(sum(probs[i] * sc(tuple((int(i) >> qubit_of[j]) & 1 for j in range(nb)))
                     for i in np.flatnonzero(probs > 1e-15)))


def strata(n, m):
    """kept strata: (e, sector, beta, parity) with exact weights in F_avg; plus the all-erased weight"""
    p = H.p_flip(m)
    out = []
    for e in range(n):
        we = math.comb(n, e) / 2 ** n
        for sector, ps in (("s0", 1 - p), ("s1", p)):
            for beta in (0, 1):
                for par in ((0,) if e == 0 else (0, 1)):
                    pp = 1.0 if e == 0 else (p_odd(e, p) if par else 1 - p_odd(e, p))
                    out.append(((e, sector, beta, par), we * ps * 0.5 * pp))
    return out


def sample_choice(m, sector, rng):
    return (sector, tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m)) if sector == "s0"
            else tuple(int(x) for x in rng.integers(2, size=m)))


_EX: dict = {}


def stratum_exact(n, m, key):
    if (n, m, key) not in _EX:
        e, sector, beta, par = key
        rng = np.random.default_rng(0)
        ch = sample_choice(m, sector, rng)
        _EX[(n, m, key)] = float(np.mean([ideal(m, NUnit(n, e, ch, beta, inp, par)) for inp in INPUTS]))
    return _EX[(n, m, key)]


def all_erased_value(n, m, values=None):
    """F_avg contribution of the all-erased branch from per-(input, parity) values (ideal if values is None)"""
    p = H.p_flip(m)
    tot = 0.0
    for inp in INPUTS:
        for par in (0, 1):
            v = values[(inp, par)] if values is not None else ideal(m, NUnit(n, n, None, 0, inp, par))
            tot += 0.5 ** n / 6 * (p_odd(n, p) if par else 1 - p_odd(n, p)) * v
    return tot


def check(ns=(2, 3, 4), ms=(1, 2, 3)):
    worst = 0.0
    for n in ns:
        for m in ms:
            if 2 * n + 2 * m > 20:
                continue
            fa = sum(W * stratum_exact(n, m, k) for k, W in strata(n, m)) + all_erased_value(n, m)
            fe = (3 * fa - 1) / 2
            worst = max(worst, abs(fe - f_closed(n, m)))
            print(f"n = {n}, m = {m}: circuits' exact F_e {fe:.12f}   formula {f_closed(n, m):.12f}", flush=True)
    # the dynamic circuits (real mid-circuit reads + feed-forward) by Aer shots, n = 3
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    rng = np.random.default_rng(4)
    z = 0.0
    for _ in range(8):
        key = strata(3, 2)[int(rng.integers(len(strata(3, 2))))][0]
        uu = NUnit(3, key[0], sample_choice(2, "s0", rng), key[2], INPUTS[int(rng.integers(6))], key[3])
        qc = circuit(2, uu, "cz2", dynamic=True)
        nb = F.n_pair_bits(2, uu, "cz2") + 3 + 1
        cnt = sim.run(transpile(qc, sim), shots=4000, seed_simulator=int(rng.integers(1 << 30))).result().get_counts()
        sc = score(2, uu)
        vals = []
        for k_, nshot in cnt.items():
            creg = k_.split()[-1][::-1]
            vals += [sc(tuple(int(creg[j]) for j in range(nb)))] * nshot
        se = max(np.std(vals) / math.sqrt(len(vals)), 1e-9)
        z = max(z, abs(np.mean(vals) - ideal(2, uu)) / se)
    print(f"dynamic n = 3 circuits vs exact ideal (Aer shots): worst z = {z:.2f}")
    return worst


# ---------------------------------------------------------------- predictions

def finish(n, m, key, W, ideals, noisy, czs):
    loss = np.array(ideals) - np.array(noisy)
    ex = stratum_exact(n, m, key)
    return dict(key=list(key), weight=W, exact_ideal=ex, loss=float(loss.mean()),
                loss_se=float(loss.std(ddof=1) / math.sqrt(len(loss))), value=ex - float(loss.mean()),
                mean_cz=float(np.mean(czs)))


def compose(n, m, rows, all_erased, q):
    """F_e lean and faithful. Faithful: each erased other's key flips with p_eff = (1-p) q0 + p (1 - q1) instead of p
    (exact in the engine's model, as hw_levers.compose for n = 2); kept others are read through A before P acts."""
    p = H.p_flip(m)
    p_eff = (1 - p) * q[0] + p * (1 - q[1])
    lean = faith = 0.0
    for r in rows.values():
        e, sector, beta, par = r["key"]
        lean += r["weight"] * r["value"]
        if e == 0:
            faith += r["weight"] * r["value"]
            continue
        base = r["weight"] / (p_odd(e, p) if par else 1 - p_odd(e, p))
        faith += base * (p_odd(e, p_eff) if par else 1 - p_odd(e, p_eff)) * r["value"]
    ae_lean = ae_faith = 0.0
    for (inp, par), v in all_erased.items():
        ae_lean += 0.5 ** n / 6 * (p_odd(n, p) if par else 1 - p_odd(n, p)) * v
        ae_faith += 0.5 ** n / 6 * (p_odd(n, p_eff) if par else 1 - p_odd(n, p_eff)) * v
    to = lambda fa: (3 * fa - 1) / 2  # noqa: E731
    return to(lean + ae_lean), to(faith + ae_faith)


def hh_layout(snap, be, n, m, policy=LY.QubitPolicy()):
    """heavy-hex, dynamic sector 0, n <= 3: key0 on a degree-3 qubit with c0 and the other keys as neighbours; each
    other key's copy on one of its free neighbours; m disjoint shield pairs on the best remaining edges"""
    edges = {tuple(sorted(e_)) for e_ in be.coupling_map.get_edges()}
    nb = {q: set() for q in range(snap.num_qubits)}
    for a_, b_ in edges:
        nb[a_].add(b_)
        nb[b_].add(a_)
    ro = snap.readout_error
    f = 1.25 * V.CZ_FACTOR_V4

    def okq(q):
        return (snap.complete(q) and ro(q) <= policy.max_readout and snap.t1[q] >= policy.min_t1
                and snap.t2[q] >= policy.min_t2 and snap.sx_error[q] <= policy.max_sx)

    def oke(a_, b_):
        e_ = (min(a_, b_), max(a_, b_))
        return e_ in snap.cz and snap.cz[e_][0] <= policy.max_cz and okq(a_) and okq(b_)

    def ecost(a_, b_, uses):
        return uses * f * snap.cz[(min(a_, b_), max(a_, b_))][0]
    import itertools as it
    best = None
    for k0 in range(snap.num_qubits):
        if len(nb[k0]) < n or not okq(k0):
            continue
        for c0, *others in it.permutations(sorted(nb[k0]), n):
            if not oke(k0, c0) or not all(oke(k0, o) for o in others):
                continue
            used = {k0, c0, *others}
            cost = ecost(k0, c0, 2) + ro(k0) + ro(c0) + sum(ecost(k0, o, 1) for o in others)
            copies, okall = [], True
            for o in others:
                opts = sorted((x for x in nb[o] - used if oke(o, x)), key=lambda x: ecost(o, x, 1) + 2 * ro(x))
                if not opts:
                    okall = False
                    break
                copies.append(opts[0])
                used.add(opts[0])
                cost += ecost(o, opts[0], 1) + ro(opts[0]) + ro(o)
            if okall and (best is None or cost < best[0]):
                best = (cost, k0, c0, others, copies)
    _, k0, c0, others, copies = best
    used = {k0, c0, *others, *copies}
    cand = sorted((e_ for e_ in edges if oke(*e_)), key=lambda e_: 3 * f * snap.cz[e_][0] + ro(e_[0]) + ro(e_[1]))
    pairs = []
    for a_, b_ in cand:
        if a_ in used or b_ in used:
            continue
        aa, ss = (a_, b_) if ro(a_) <= ro(b_) else (b_, a_)   # cz2 reads a at the end; s is read mid-circuit
        pairs.append((aa, ss))
        used.update((a_, b_))
        if len(pairs) == m:
            break
    keys = [k0] + list(others)
    cps = [c0] + list(copies)
    layout = keys + cps + [s_ for _, s_ in pairs] + [a_ for a_, _ in pairs]
    return layout, best[0]


def engine_pair(m, u, decoder, layout, be, tau):
    """sector-0 dynamic unit for the engine: measure-type CZs on virtual edges, the read-out hold on those shields
    (early schedule), and the same circuit with the wrong correction (Z on key0)"""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    n = u.n
    mt = D.measure_type(u.choice)
    out = []
    for extra_z in (0, 1):
        qc = circuit(m, u, decoder, dynamic=False)
        # insert the hold and the wrong correction: rebuild with the hold after the Bell pairs
        from qiskit import QuantumCircuit
        new = QuantumCircuit(*qc.qregs, *qc.cregs)
        s = [2 * n + t for t in range(m)]
        bell_done = False
        cz_seen = 0
        for ins in qc.data:
            new.append(ins.operation, ins.qubits, ins.clbits)
            qs = [qc.find_bit(x).index for x in ins.qubits]
            if not bell_done and ins.operation.name == "cx" and qs[1] == s[-1]:
                for t in mt:
                    new.delay(tau, s[t], unit="s")
                bell_done = True
            if ins.operation.name == "cz" and 0 in qs and any(x in s for x in qs):
                cz_seen += 1
                if extra_z and cz_seen == len(mt):
                    new.z(0)
        virt = [(layout[0], layout[2 * n + t]) for t in mt]
        cm = CouplingMap([list(e_) for e_ in be.coupling_map.get_edges()] + [list(v) for v in virt]
                         + [[b_, a_] for a_, b_ in virt])
        tqc = transpile(new, basis_gates=["cz", "rz", "sx", "x", "measure", "delay"], coupling_map=cm,
                        initial_layout=layout, optimization_level=1, seed_transpiler=1, routing_method="none")
        out.append((tqc, virt))
    return out


def predict_hh(n, m, decoder, units, traj, seed=7, tau_ff=D.TAU_FF):
    t0 = time.time()
    be, snap = V.backend(), V.snapshot()
    params = N.NoiseParams(cz_factor=V.CZ_FACTOR_V4)
    lay0, cost0 = hh_layout(snap, be, n, m)
    tau = float(np.median([x for x in snap.meas_duration if math.isfinite(x)])) + tau_ff
    rng0 = np.random.default_rng(1)
    reps = [circuit(m, NUnit(n, e, sample_choice(m, "s1", rng0), 1, INPUTS[0], 0), decoder) for e in range(n)]
    lay1 = LY.choose_layout(reps, be, snap, params).initial_layout
    urng, rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    rows, i = {}, 0
    for key, W in strata(n, m):
        e, sector, beta, par = key
        idl, noisy, czs = [], [], []
        for _ in range(units):
            u = NUnit(n, e, sample_choice(m, sector, urng), beta, INPUTS[int(urng.integers(6))], par)
            if sector == "s0":
                (c0, virt), (c1, _) = engine_pair(m, u, decoder, lay0, be, tau)
                sn = D.augmented(snap, virt)
                mt = D.measure_type(u.choice)
                r = 0.5 * (1 - math.prod(1 - 2 * snap.readout_error(lay0[2 * n + t]) for t in mt))
                v0 = E.expectation(c0, score(m, u, decoder), sn, params, traj, rng).mean
                v1 = E.expectation(c1, score(m, u, decoder), sn, params, traj, rng).mean if mt else v0
                val = (1 - r) * v0 + r * v1
                vs = {tuple(sorted(x)) for x in virt}
                ncz = sum(1 for ins in c0.data if ins.operation.name == "cz"
                          and tuple(sorted(c0.find_bit(x).index for x in ins.qubits)) not in vs)
            else:
                tqc = V.isa(be, circuit(m, u, decoder), lay1, 1000 + i)
                val = E.expectation(tqc, score(m, u, decoder), snap, params, traj, rng).mean
                ncz = tqc.count_ops().get("cz", 0)
            i += 1
            idl.append(ideal(m, u, decoder))
            noisy.append(val)
            czs.append(ncz)
        rows["|".join(map(str, key))] = row = finish(n, m, key, W, idl, noisy, czs)
        print(f"  {'|'.join(map(str, key)):14s} W={W:.4f} exact {row['exact_ideal']:.3f} loss {row['loss']:+.4f}"
              f"+-{row['loss_se']:.4f} CZ {row['mean_cz']:.1f}", flush=True)
    ae = {}
    for inp in INPUTS:
        for par in (0, 1):
            u = NUnit(n, n, None, 0, inp, par)
            tqc = V.isa(be, circuit(m, u), lay0, 1)
            ae[(inp, par)] = E.expectation(tqc, score(m, u), snap, params, traj, rng).mean
    q = {0: D.q_sector0(snap, be, [lay0[1], lay0[1], lay0[n + 1], lay0[n + 1]], tau, params, rng)
         if n > 1 else 0.0, 1: V.q_flip(m, 1, params, be, snap)}
    lean, faith = compose(n, m, rows, ae, q)
    se = 1.5 * math.sqrt(sum((r["weight"] * r["loss_se"]) ** 2 for r in rows.values()))
    rec = dict(n=n, m=m, decoder=decoder, device="ibm_kingston R145 calibration (heavy-hex)", design="dynamic, early",
               tau_s=tau, layout_sector0=lay0, layout_sector1=lay1, F_e_ideal=f_closed(n, m), F_e_lean=lean,
               F_e_faithful=faith, se=se, q=q, strata=rows, units=units, traj=traj, seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"hh_n{n}_m{m}_{decoder}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"HEAVY-HEX n={n} m={m} {decoder} dynamic: lean {lean:.4f} faithful {faith:.4f} +- {se:.4f} "
          f"(ideal {f_closed(n, m):.4f}; q0 {q[0]:.4f} q1 {q[1]:.4f})", flush=True)
    return rec


DEVICES = {
    "helios_typ": dict(V.HELIOS),
    "helios_max": dict(e2=2e-3, e1=2e-3, ro=2e-3, mem=1e-3),        # Helios data sheet v1.00, Table 1, max column
    "ionq_aria": dict(e2=4e-3, e1=5e-4, ro=3.9e-3, mem=3e-4),       # Azure provider page: 2q 99.6%, 1q 99.95%,
}                                                                   # SPAM 99.61%; memory 3e-4 per 2q layer, i.e.
                                                                    # one 600 us gate against T2 = 1 s (all_to_all's
                                                                    # 200 us layer then implies an effective T2 0.33 s)


def predict_ion(n, m, decoder, units, traj, cz_factor=1.0, seed=7, device="helios_typ"):
    t0 = time.time()
    nq = 2 * n + 2 * m
    dev = DEVICES[device]
    be, snap = V.all_to_all(nq, **dev)
    params = N.NoiseParams(cz_factor=cz_factor)
    lay = list(range(nq))
    urng, rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    rows = {}
    for key, W in strata(n, m):
        e, sector, beta, par = key
        idl, noisy, czs = [], [], []
        for _ in range(units):
            u = NUnit(n, e, sample_choice(m, sector, urng), beta, INPUTS[int(urng.integers(6))], par)
            tqc = V.isa(be, circuit(m, u, decoder), lay, 1)
            noisy.append(E.expectation(tqc, score(m, u, decoder), snap, params, traj, rng).mean)
            idl.append(ideal(m, u, decoder))
            czs.append(tqc.count_ops().get("cz", 0))
        rows["|".join(map(str, key))] = finish(n, m, key, W, idl, noisy, czs)
    ae = {}
    for inp in INPUTS:
        for par in (0, 1):
            u = NUnit(n, n, None, 0, inp, par)
            ae[(inp, par)] = E.expectation(V.isa(be, circuit(m, u), lay, 1), score(m, u), snap, params, traj,
                                           rng).mean
    be1, snap1 = V.all_to_all(2 + 2 * m, **dev)
    q = {}
    for sec in (0, 1):
        q[sec] = float(np.mean([V.q_single(m, sample_choice(m, "s0" if sec == 0 else "s1", urng), params, be1, snap1,
                                           rng) for _ in range(4)]))
    lean, faith = compose(n, m, rows, ae, q)
    se = 1.5 * math.sqrt(sum((r["weight"] * r["loss_se"]) ** 2 for r in rows.values()))
    rec = dict(n=n, m=m, decoder=decoder, device=device, device_params=dev, physical_qubits_faithful=n * (2 + 2 * m),
               cz_factor=cz_factor, F_e_ideal=f_closed(n, m), F_e_lean=lean, F_e_faithful=faith, se=se, q=q,
               strata=rows, units=units, traj=traj, seconds=time.time() - t0,
               mean_cz=float(np.mean([r["mean_cz"] for r in rows.values()])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ion_{device}_n{n}_m{m}_{decoder}_cz{cz_factor:g}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n",
                                                                      encoding="utf-8")
    print(f"ION[{device}] n={n} m={m} {decoder} cz x{cz_factor:g}: lean {lean:.4f} faithful {faith:.4f} +- {se:.4f} "
          f"(ideal {f_closed(n, m):.4f}; CZ {rec['mean_cz']:.1f})", flush=True)
    return rec


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 1 and a[1] == "check":
        print(f"worst |circuits - formula| = {check():.2e}")
        sys.exit(0)
    if len(a) > 1 and a[1] == "hh":
        predict_hh(int(a[2]), int(a[3]), a[4], int(a[5]), int(a[6]))
        sys.exit(0)
    if len(a) > 1 and a[1] == "ion":
        cz = float(a[a.index("--cz-factor") + 1]) if "--cz-factor" in a else 1.0
        dv = a[a.index("--device") + 1] if "--device" in a else "helios_typ"
        predict_ion(int(a[2]), int(a[3]), a[4], int(a[5]), int(a[6]), cz_factor=cz, device=dv)
        sys.exit(0)
    raise SystemExit(__doc__)
