"""P_d with its dominant sector realized by MID-CIRCUIT MEASUREMENT and feed-forward: no key-shield gates (R147).

THE IDENTITY (exact; Choi checked to 1e-17 at m = 1-3 by locc_choi below). P_d's sector 0 (probability 1 - p) is a
uniform mixture over shield Paulis g of V_g = prod_{g_t in {I,Z}} CZ(key, s_t) . Z_key^{#Y} . W_g, which factorizes
over shields into commuting per-shield channels
    Lam_t = 1/2 [CZ o Delta_s] + 1/4 [X_s] + 1/4 [Z_key Y_s],        Delta_s = dephasing of s_t in Z,
because the pair {CZ, CZ Z_s} averages to CZ o Delta_s. And CZ o Delta_s IS "measure s_t in Z (outcome b), apply
Z^b to the key". So in sector 0 the key never interacts with a shield: the shields drawn as measure-type are read
mid-circuit and ONE conditional Z, on the XOR of their outcomes, is applied to the key by the channel's controller.
Sector 1 keeps its gates: its per-shield factor preserves joint key-shield coherence (|0,0><1,1| survives with
factor -1/sqrt2), which no measure-and-correct map can.
No local-unitary decomposition of sector 0 exists either: every local-unitary mixture maps X_key (x) 1 to
(something) (x) 1, while Lam_t maps it to X_key Z_s / 2. One-way classical communication is the minimum.

WHY IT MATTERS ON HEAVY-HEX. The key-to-shield CZ star (degree m + 2 at the key, against heavy-hex's 3) is what
routing tripled. With it gone, a sector-0 circuit needs a 4-qubit path (c1 - key1 - key0 - c0) and m DISJOINT
shield pairs (a_t, s_t) anywhere on the chip: zero routing, and every pair can sit on a well-calibrated edge.

MODEL FOR THE ENGINE (oneq.hw.engine has no mid-circuit measurement). The engine circuit keeps the original sector-0
unitaries (whose average over g in {I, Z} is exactly CZ o Delta_s) but places the measure-type CZs on VIRTUAL edges
with zero error and zero duration, then holds the key and the measured shields for TAU (measurement + feed-forward)
-- an idle the engine charges on the ALAP schedule. A mid-circuit readout error on s_t applies the wrong correction,
a Z on the key: the expectation is mixed exactly, E = (1 - r) E[no Z] + r E[Z], r = P(odd number of flips).
Unmodelled, and stated: measurement back-action beyond assignment error, and measurement-induced dephasing of
neighbours.

  python hw_dynamic.py check          Choi identity (m = 1-3); dynamic hardware circuits vs exact ideal by Aer shots
  python hw_dynamic.py predict M DEC UNITS TRAJ [--tau-ff S] [--idle-factor X] [--inline] [--tag T]
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hardware_design as hd  # noqa: E402
import hw_faithful as F  # noqa: E402
import hw_levers as V  # noqa: E402
import hw_two_use as H  # noqa: E402
import ppt_family as pf  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import layout as LY  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "shield_family" / "dynamic"
TAU_FF = 0.65e-6            # feed-forward decision latency after the read-out ends (ASSUMPTION; vary it)


# ---------------------------------------------------------------- the identity

def locc_choi_error(m: int) -> float:
    I2, X2, Y2, Z2 = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1])
    P0, P1 = np.diag([1.0, 0]), np.diag([0.0, 1])
    d = 2 ** m
    D = 2 * d
    p = 1 / (math.sqrt(d) + 1)
    per = [math.sqrt(.5) * np.kron(I2, P0), math.sqrt(.5) * np.kron(Z2, P1), .5 * np.kron(I2, X2), .5 * np.kron(Z2, Y2)]

    def embed(op, t):
        A = op.reshape(2, 2, 2, 2)
        full = np.zeros((D, D), complex)
        for i in range(D):
            bits = [(i >> (m - j)) & 1 for j in range(m + 1)]
            for ko, so in itertools.product((0, 1), (0, 1)):
                amp = A[ko, so, bits[0], bits[1 + t]]
                if amp:
                    ob = list(bits)
                    ob[0], ob[1 + t] = ko, so
                    full[sum(b << (m - j) for j, b in enumerate(ob)), i] += amp
        return full
    emb = [[embed(op, t) for op in per] for t in range(m)]
    Ks = []
    for combo in itertools.product(range(4), repeat=m):
        K = np.eye(D, dtype=complex)
        for t, c in enumerate(combo):
            K = emb[t][c] @ K
        Ks.append(math.sqrt(1 - p) * K)
    Ks += [math.sqrt(w) * U for w, lab, U in hd.p_unitaries(m) if lab[0] == "s1"]
    J = 0
    for K in Ks:
        vec = np.zeros(D * D, complex)
        for i in range(D):
            vec[i * D:(i + 1) * D] = K[:, i]
        J = J + np.outer(vec, vec.conj())
    return float(np.abs(J / D - pf.choi_state(pf.sylvester(m), p)).max())


# ---------------------------------------------------------------- hardware circuits (dynamic)

def measure_type(choice):
    """the shields P reads mid-circuit: sector 0 with g_t in {I, Z}"""
    kind, lab = choice
    return [t for t, g in enumerate(lab) if g in "IZ"] if kind == "s0" else []


def apply_p_dynamic(qc, choice, key, shields, mcm_bits, part="both"):
    """P_d's sampled unitary, sector 0 by measure-and-correct. mcm_bits: classical bits for the measure-type shields.
    part: 'shield' (the Paulis and the mid-circuit reads), 'key' (the key's static Z and the conditional Z), 'both'.
    The two parts act on disjoint registers and are joined only by the classical bits, so the shield part may run as
    soon as Alice releases the shields and the key part when she releases the key (the EARLY schedule)."""
    from qiskit.circuit.classical import expr
    kind, lab = choice
    if kind != "s0":
        if part in ("both", "key"):
            H._apply_p(qc, choice, key, shields)
        return
    if part in ("both", "shield"):
        _shield_part(qc, lab, shields, choice, mcm_bits)
    if part in ("both", "key"):
        _key_part(qc, lab, key, choice, mcm_bits)


def _shield_part(qc, lab, shields, choice, mcm_bits):
    for t, g in enumerate(lab):
        if g == "X":
            qc.x(shields[t])
        elif g == "Y":
            qc.y(shields[t])
    for t, b in zip(measure_type(choice), mcm_bits):
        qc.measure(shields[t], b)                      # g = Z before a Z read-out does nothing: dropped


def _key_part(qc, lab, key, choice, mcm_bits):
    from qiskit.circuit.classical import expr
    if lab.count("Y") % 2:
        qc.z(key)
    if not mcm_bits:
        return
    cond = mcm_bits[0]
    for b in mcm_bits[1:]:
        cond = expr.bit_xor(cond, b)
    with qc.if_test(cond if len(mcm_bits) > 1 else (mcm_bits[0], 1)):
        qc.z(key)


def lean_dynamic(m, u, decoder="cz2", early=True, one_register=False, enforce=True):
    """hw_two_use.circuit / hw_faithful.lean_circuit with P dynamic. Classical bits: the protocol's bits first (same
    order and meaning, so hw_faithful.score applies), then the mid-circuit bits (ignored by the score)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    if u.branch == "erased_erased":
        return H.circuit(m, u)
    s = [4 + t for t in range(m)]
    a = [4 + m + t for t in range(m)]
    q = QuantumRegister(4 + 2 * m, "q")
    nmt = len(measure_type(u.choice))
    npair = F.n_pair_bits(m, u, decoder)
    if one_register:                                   # hardware: one register (oneq.hw.submit reads exactly one);
        c = ClassicalRegister(npair + 3 + nmt, "c")    # the mid-circuit bits trail the protocol's, which the score
        qc = QuantumCircuit(q, c, name=u.name)         # never reads
        mcm = [c[npair + 3 + j] for j in range(nmt)]
    else:
        c = ClassicalRegister(npair + 3, "c")
        regs = [q, c]
        mc = None
        if nmt:
            mc = ClassicalRegister(nmt, "mcm")
            regs.append(mc)
        qc = QuantumCircuit(*regs, name=u.name)
        mcm = list(mc) if mc is not None else []
    for t in range(m):                                 # Alice's shield encoding
        qc.h(a[t])
        qc.cx(a[t], s[t])
    # THE SCHEDULE IS ENFORCED BY A BARRIER, not by instruction order (R153). The key's preparation and the shields'
    # mid-circuit reads act on disjoint qubits with no dependency, so without a barrier "early" and "inline" are the
    # SAME dependency graph and any scheduler runs them identically: R152's 16 early/inline twins were identical
    # circuits (checked by DAG equality), and its V3 tested nothing.
    sync = [0, 1, 2, 3] + [s[t] for t in measure_type(u.choice)]
    if early:
        apply_p_dynamic(qc, u.choice, 0, s, mcm, part="shield")
        if enforce:
            qc.barrier(sync)                           # the key is prepared only after the reads have finished
    H._prep(qc, 0, u.inp)                              # Alice's key encoding
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 2)
    if not early and enforce:
        qc.barrier(sync)                               # the reads start only once the key is prepared: it waits
    apply_p_dynamic(qc, u.choice, 0, s, mcm, part="key" if early else "both")
    if u.branch == "kept_kept":
        qc.cx(1, 3)
    elif u.flips[1]:
        qc.x(1)
    qc.cx(0, 2)
    bits = []
    for t in range(m):
        bits += F.pair_decode(qc, a[t], s[t], u.beta, decoder)
    H._measure_in(qc, 0, H.BOB_BASIS[u.inp[0]][0])
    order = [2] + bits + [3 if u.branch == "kept_kept" else 1, 0]
    for j, qb in enumerate(order):
        qc.measure(qb, c[j])
    return qc


def aer_check(m=2, units=10, shots=20000, seed=2, decoder="cz2", early=True):
    """mean score of dynamic circuits by Aer shots vs the exact ideal of the unitary circuits: max z-score"""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(units):
        key = H.strata(m)[int(rng.integers(len(H.strata(m))))][0]
        u = H.sample_in_stratum(m, ("kept_kept" if key[0] == "kept_kept" else "kept_erased", "s0", key[2], key[3]),
                                rng)
        qc = lean_dynamic(m, u, decoder, early=early)
        ncl = F.n_pair_bits(m, u, decoder) + 3
        counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=int(rng.integers(1 << 30))).result().get_counts()
        sc = F.score(m, u, decoder)
        vals = []
        for k, n in counts.items():
            creg = k.split()[-1]                      # the protocol register is listed last (registers: c, mcm)
            bits = tuple(int(creg[::-1][j]) for j in range(ncl))
            vals += [sc(bits)] * n
        mean, se = float(np.mean(vals)), float(np.std(vals) / math.sqrt(shots))
        ideal = F.lean_ideal(m, u, decoder)
        z = abs(mean - ideal) / max(se, 1e-9)
        worst = max(worst, z)
        print(f"  {u.name[:60]:60s} aer {mean:.4f} +- {se:.4f}  exact {ideal:.4f}  z {z:.2f}", flush=True)
    return worst


# ---------------------------------------------------------------- engine model and layout

def augmented(snap: S.DeviceSnapshot, pairs) -> S.DeviceSnapshot:
    cz = dict(snap.cz)
    for a, b in pairs:
        cz[(min(a, b), max(a, b))] = (0.0, 0.0)
    return S.DeviceSnapshot(**{**snap.__dict__, "cz": cz, "name": snap.name + "+virtual"})


def hand_layout(snap, be, m, policy=LY.QubitPolicy()):
    """sector-0 layout: a 4-path (c1, key1, key0, c0) and m disjoint shield pairs (a, s), greedy on the calibration.
    Edge cost = (2 or 3 uses) x the in-circuit CZ rate + read-out of the qubits measured at the end and mid-circuit."""
    cmap = be.coupling_map
    edges = {tuple(sorted(e)) for e in cmap.get_edges()}
    ro = snap.readout_error

    def ok_q(q):
        return (snap.complete(q) and ro(q) <= policy.max_readout and snap.t1[q] >= policy.min_t1
                and snap.t2[q] >= policy.min_t2 and snap.sx_error[q] <= policy.max_sx)

    def ok_e(e):
        return e in snap.cz and snap.cz[e][0] <= policy.max_cz and ok_q(e[0]) and ok_q(e[1])
    nb = {q: set() for q in range(snap.num_qubits)}
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    best = None
    for k0 in range(snap.num_qubits):                         # key0 with neighbours key1 and c0; key1 with c1
        for k1 in nb[k0]:
            for c0 in nb[k0] - {k1}:
                for c1 in nb[k1] - {k0, c0}:
                    path = [(k1, k0), (k0, c0), (k1, c1)]
                    es = [tuple(sorted(e)) for e in path]
                    if not all(ok_e(e) for e in es):
                        continue
                    cost = (snap.cz[es[0]][0] + 2 * snap.cz[es[1]][0] + snap.cz[es[2]][0]) * 1.25 * V.CZ_FACTOR_V4 \
                        + ro(k0) + ro(k1) + ro(c0) + ro(c1)
                    if best is None or cost < best[0]:
                        best = (cost, (k0, k1, c0, c1))
    used = set(best[1])
    pairs = []
    cand = sorted((e for e in edges if ok_e(e)),
                  key=lambda e: 3 * 1.25 * V.CZ_FACTOR_V4 * snap.cz[e][0] + ro(e[0]) + 2 * ro(e[1]))
    for a, s in cand:
        if a in used or s in used:
            continue
        # orient: the qubit with the better read-out is the shield (read mid-circuit AND at the end)
        aa, ss = (a, s) if ro(a) >= ro(s) else (s, a)
        pairs.append((aa, ss))
        used.update((a, s))
        if len(pairs) == m:
            break
    k0, k1, c0, c1 = best[1]
    layout = [k0, k1, c0, c1] + [s for _, s in pairs] + [a for a, _ in pairs]
    return layout, best[0]


def engine_circuit(m, u, decoder, layout, be, tau, early=True):
    """the ISA circuit the engine scores for a sector-0 kept unit: original unitaries, measure-type CZs on virtual
    edges, a hold of tau on the key and the measured shields; plus the same circuit with a Z on the key after the hold
    (the wrong correction a mid-circuit read-out error causes)"""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.transpiler import CouplingMap
    s = [4 + t for t in range(m)]
    a = [4 + m + t for t in range(m)]
    mt = measure_type(u.choice)
    out = []
    for extra_z in (0, 1):
        q = QuantumRegister(4 + 2 * m, "q")
        npair = F.n_pair_bits(m, u, decoder)
        c = ClassicalRegister(npair + 3, "c")
        qc = QuantumCircuit(q, c)
        for t in range(m):
            qc.h(a[t])
            qc.cx(a[t], s[t])
        if early and mt:                               # the shields are read before Alice touches the key
            for t in mt:
                qc.delay(tau, s[t], unit="s")
        H._prep(qc, 0, u.inp)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)
        H._apply_p(qc, u.choice, key=0, shields=s)     # sector 0: CZ on measure-type shields only (virtual edges)
        if mt:
            if not early:
                for qb in [0] + [s[t] for t in mt]:
                    qc.delay(tau, qb, unit="s")
            if extra_z:
                qc.z(0)
        if u.branch == "kept_kept":
            qc.cx(1, 3)
        elif u.flips[1]:
            qc.x(1)
        qc.cx(0, 2)
        bits = []
        for t in range(m):
            bits += F.pair_decode(qc, a[t], s[t], u.beta, decoder)
        H._measure_in(qc, 0, H.BOB_BASIS[u.inp[0]][0])
        order = [2] + bits + [3 if u.branch == "kept_kept" else 1, 0]
        qc.measure(order, list(range(len(order))))
        virt = [(layout[0], layout[4 + t]) for t in mt]
        cm = CouplingMap([list(e) for e in be.coupling_map.get_edges()] + [list(e) for e in virt]
                         + [[b, a_] for a_, b in virt])
        tqc = transpile(qc, basis_gates=["cz", "rz", "sx", "x", "measure", "delay"], coupling_map=cm,
                        initial_layout=layout, optimization_level=1, seed_transpiler=1, routing_method="none")
        out.append((tqc, virt))
    return out


def predict(m, decoder, units, traj, tau_ff=TAU_FF, idle_factor=1.0, seed=7, tag="", early=True):
    t0 = time.time()
    be, snap = V.backend(), V.snapshot()
    params = N.NoiseParams(cz_factor=V.CZ_FACTOR_V4, idle_factor=idle_factor)
    layout0, cost0 = hand_layout(snap, be, m)
    tau = float(np.median([x for x in snap.meas_duration if math.isfinite(x)])) + tau_ff
    # sector 1 (and both-erased) keep routed gates: their own Sabre layout, as hw_levers
    choice1 = LY.choose_layout(V.representatives(m, decoder), be, snap, params)
    layout1 = choice1.initial_layout
    unit_rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    rows, i = {}, 0
    for key, W in H.strata(m):
        noisy, ideal, czs = [], [], []
        for _ in range(units):
            u = H.sample_in_stratum(m, key, unit_rng)
            if key[1] == "s0":
                (t0c, virt), (t1c, _) = engine_circuit(m, u, decoder, layout0, be, tau, early=early)
                sn = augmented(snap, virt)
                mt = measure_type(u.choice)
                r_odd = 0.5 * (1 - math.prod(1 - 2 * snap.readout_error(layout0[4 + t]) for t in mt))
                e0 = E.expectation(t0c, F.score(m, u, decoder), sn, params, traj, rng).mean
                e1 = E.expectation(t1c, F.score(m, u, decoder), sn, params, traj, rng).mean if mt else e0
                val = (1 - r_odd) * e0 + r_odd * e1
                ncz = sum(1 for ins in t0c.data if ins.operation.name == "cz"
                          and tuple(sorted(t0c.find_bit(x).index for x in ins.qubits)) not in
                          {tuple(sorted(v)) for v in virt})
            else:
                tqc = V.isa(be, F.lean_circuit(m, u, decoder), layout1, 1000 + i)
                val = E.expectation(tqc, F.score(m, u, decoder), snap, params, traj, rng).mean
                ncz = tqc.count_ops().get("cz", 0)
            i += 1
            noisy.append(val)
            ideal.append(F.lean_ideal(m, u, decoder))
            czs.append(ncz)
        rows["|".join(map(str, key))] = row = V.finish_row(m, key, W, ideal, noisy, czs)
        print(f"  {'|'.join(map(str, key)):26s} W={W:.4f} exact {row['exact_ideal']:.4f} loss {row['loss']:+.4f}"
              f"+-{row['loss_se']:.4f} CZ {np.mean(czs):.1f}", flush=True)
    ee = {}
    for w, u in H.erased_erased_units(m):
        tqc = V.isa(be, H.circuit(m, u), layout0, 1)
        ee[(u.inp, u.flips)] = E.expectation(tqc, H.score(m, u), snap, params, traj, rng).mean
    # faithful: use 1's flip law. Sector 0: copy CNOT + the hold (no key-shield gate); sector 1: routed gates
    q = {0: q_sector0(snap, be, layout0, tau, params, rng), 1: V.q_flip(m, 1, params, be, snap)}
    lean, faith = V.compose(m, rows, ee, q, "noisy")
    se = 1.5 * math.sqrt(sum((r["weight"] * r["loss_se"]) ** 2 for r in rows.values()))
    rec = dict(m=m, d=2 ** m, decoder=decoder, schedule="early" if early else "inline", tau_s=tau, tau_ff_assumed_s=tau_ff, idle_factor=idle_factor,
               snapshot_sha256=snap.sha256(), cz_factor=V.CZ_FACTOR_V4, layout_sector0=layout0,
               layout_sector0_cost=cost0, layout_sector1=layout1, units_per_stratum=units, trajectories=traj,
               F_e_ideal=H.f_e_closed(m), F_e_lean=lean, F_e_faithful=faith, se=se, q=q, strata=rows,
               seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    sched = "early" if early else "inline"
    (OUT / f"dynamic_m{m}_{decoder}_{sched}_idle{idle_factor:g}{('_' + tag) if tag else ''}.json").write_text(
        json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"m={m} {decoder} DYNAMIC {sched} (tau {tau * 1e6:.2f} us, idle x{idle_factor:g}): lean {lean:.4f} faithful "
          f"{faith:.4f} +- {se:.4f}  (ideal {H.f_e_closed(m):.4f}; q0 {q[0]:.4f} q1 {q[1]:.4f})", flush=True)
    return rec


def q_sector0(snap, be, layout0, tau, params, rng):
    """an erased use in sector 0: its key sees the copy CNOT and the hold, nothing else that is not diagonal"""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(1, "c"))
    qc.x(0)                                            # key in |1>: amplitude damping during the hold can flip it
    qc.cx(0, 1)
    qc.delay(tau, 0, unit="s")
    qc.measure(0, 0)
    k1, c1 = layout0[1], layout0[3]
    tqc = transpile(qc, be, initial_layout=[k1, c1], optimization_level=1)
    r = E.expectation(tqc, lambda b: 1.0 if b[0] != 1 else 0.0, snap,
                      N.NoiseParams(**{**params.describe(), "readout": False}), 20, rng)
    return r.mean


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        for mm in (1, 2, 3):
            print(f"m = {mm}: |Choi(measure-and-correct sector 0 + sector 1) - Choi(P_d)| = {locc_choi_error(mm):.2e}")
        for dec in ("cz2", "cz1"):
            for early in (True, False):
                print(f"dynamic circuits (Aer shots) vs exact ideal, m = 3, decoder {dec}, "
                      f"{'early' if early else 'inline'} schedule: worst z = "
                      f"{aer_check(3, units=8, decoder=dec, early=early):.2f}")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        a = sys.argv
        kw = {}
        if "--tau-ff" in a:
            kw["tau_ff"] = float(a[a.index("--tau-ff") + 1])
        if "--idle-factor" in a:
            kw["idle_factor"] = float(a[a.index("--idle-factor") + 1])
        if "--tag" in a:
            kw["tag"] = a[a.index("--tag") + 1]
        if "--inline" in a:
            kw["early"] = False
        predict(int(a[2]), a[3], int(a[4]), int(a[5]), **kw)
        sys.exit(0)
    raise SystemExit(__doc__)
