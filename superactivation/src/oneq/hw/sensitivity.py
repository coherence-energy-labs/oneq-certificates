"""Linear response of a predicted expectation to EVERY error location of a circuit -- the primitive the device posterior,
decision-value scheduling, loss attribution and posterior-robust certificates are built on.

The engine writes the expectation's reachable part as S = 2^-m sum_s chi(s) q(s) g(s) with chi = prod_i f_i and
f_i(s) = 1 + sum_{e at op i} w_e (sigma_e(s) - 1) / (1 - pB_i), sigma_e(s) = (-1)^{s . flip_e}. An error enters only
through its own op's factor, so

    dS / dw_e = 2^-m sum_s chi_{not i}(s) q(s) g(s) (sigma_e(s) - 1) / (1 - pB_i)            (reachable e: EXACT)

one matrix-vector product per op. An error a later non-Clifford gate BLOCKS is not a Pauli at the readout; its effect is
one single-error simulation, value(e fired) - value(no blocked error): first order in the blocked mass. Readout enters
only through the smoothed score g, linearly in each p01 / p10, so a central difference there is exact.

Aggregated to device parameters:
  ("cz", a, b, "XZ")   dS/dp for that Pauli (a<b order) per unit probability at EVERY instance of the edge
  ("sx", q)            dS/dp_sx, p_sx the engine's total sx error probability (X, Y, Z at p_sx/3 each)
  ("gamma1", q), ("gamma2", q)   dS/d(1/T1), dS/d(1/T2) through the idle relaxation of every wait on q
  ("p01", q), ("p10", q)          readout
  ("quasistatic", q), ("spectator", q)   dS/dp per unit Z probability at every such error on q (R155, R153)
and a first-order LOSS BUDGET: each component's sum_e w_e dS/dw_e, its share of the prediction's departure (crosstalk
pairs as ("crosstalk", a, b)). An error after an op with no budget key is refused, never dropped from the budget."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from oneq.hw import engine as E
from oneq.hw.noise import PAULIS2


@dataclass
class Sensitivity:
    value: float                               # the prediction's reachable part (the full expectation when exact)
    exact: bool                                # no blocked error: every derivative is exact
    blocked_mass: float
    components: dict = field(default_factory=dict)      # parameter key -> dS/dparam
    loss_budget: dict = field(default_factory=dict)     # component key -> first-order contribution to S - S_ideal
    per_error: list = field(default_factory=list)       # (op index, op name, physical qubits, paulis as
                                                        # [(physical qubit, letter)], w, dS/dw)


def _label(paulis: list, op_qubits: tuple) -> str:
    """the two-letter Pauli of a CZ error in the op's own qubit order"""
    d = dict(paulis)
    return "".join(d.get(i, "I") for i in op_qubits)


def sensitivity(qc, score, snap, params, prefer: str = "numpy") -> Sensitivity:
    P = E.plan(qc, snap, params)
    k, m, n_ops = len(P.active), len(P.meas), len(P.ops)
    if k > E.MAX_ACTIVE or m > E.MAX_MEASURED:
        raise E.EngineError("circuit beyond the engine's limits")
    sv = E.simulator(k, prefer)
    conf = E._confusions(P.meas, snap, P.active, params)
    svec = E.score_vector(score, m)
    g_hat = E.wht(E._smoothed_score(svec, conf))
    factors, pB = E._op_factors(P, n_ops, m)
    q_hat = E.wht(E._true_probs(sv.run(P.ops, {}), P.meas))
    A = q_hat * g_hat / (1 << m)
    order = sorted(factors)
    # products of every factor but one, without dividing (a factor can approach zero at large error)
    pre = [np.ones(1 << m)]
    for i in order:
        pre.append(pre[-1] * factors[i])
    suf = [np.ones(1 << m)]
    for i in reversed(order):
        suf.append(suf[-1] * factors[i])
    suf = suf[::-1]
    excl = {i: pre[j] * suf[j + 1] for j, i in enumerate(order)}
    chi_all = pre[-1]
    s0 = float(chi_all @ A)
    dS = np.zeros(len(P.e_op))
    reach = ~P.e_blocked & (P.e_flip != 0)
    for i in order:
        sel = np.nonzero(reach & (P.e_op == i))[0]
        signs = E._parity_signs(P.e_flip[sel], m)
        c = excl[i] * A / (1.0 - pB[i])
        dS[sel] = signs @ c - c.sum()
    # blocked errors: first order, one simulation each (the reachable factors of every OTHER op kept)
    blocked = np.nonzero(P.e_blocked)[0]
    for e in blocked:
        i = int(P.e_op[e])
        chi = excl[i] if i in excl else chi_all
        qh = E.wht(E._true_probs(sv.run(P.ops, {i: P.e_paulis[e]}), P.meas))
        dS[e] = float(chi @ (qh * g_hat)) / (1 << m) - float(chi @ A)
    out = Sensitivity(value=s0, exact=blocked.size == 0, blocked_mass=float(pB.sum()))
    comp: dict = {}
    loss: dict = {}

    def add(d: dict, key, v: float) -> None:
        d[key] = d.get(key, 0.0) + v
    phys = P.active
    for e in range(len(P.e_op)):
        i = int(P.e_op[e])
        op = P.ops[i]
        qs = tuple(phys[a] for a in op.qubits)
        w = float(P.e_prob[e])
        out.per_error.append((i, op.name, qs, [(phys[a], g) for a, g in P.e_paulis[e]], w, float(dS[e])))
        if op.name == "cz":
            lab = _label([(phys[a], g) for a, g in P.e_paulis[e]], qs)
            a, b = qs
            if a > b:
                a, b, lab = b, a, lab[::-1]
            add(comp, ("cz", a, b, lab), float(dS[e]))
            add(loss, ("cz", a, b), w * float(dS[e]))
        elif op.name in ("sx", "x"):
            add(comp, ("sx", qs[0]), float(dS[e]) / 3)
            add(loss, ("sx", qs[0]), w * float(dS[e]))
        elif op.name == "idle":
            q = qs[0]
            g = P.e_paulis[e][0][1]
            add(comp, ("idle", q, g), float(dS[e]))
            add(loss, ("idle", q), w * float(dS[e]))
            px, _, pz = op.pauli_probs
            f_idle = params.idle_factor
            t1, t2 = snap.t1[q], min(snap.t2[q], 2 * snap.t1[q])
            # recover the wait from px = f a / 4, a = 1 - exp(-t/T1)
            a_ = min(4 * px / f_idle, 1 - 1e-15) if f_idle > 0 else 0.0
            t = -t1 * math.log(1 - a_) if a_ > 0 else 0.0
            d_g1 = f_idle * t * math.exp(-t / t1) / 4               # d(px)/d(1/T1) = d(py)/d(1/T1) = -d(pz)/d(1/T1)
            d_g2 = f_idle * t * math.exp(-t / t2) / 2               # d(pz)/d(1/T2)
            if g in "XY":
                add(comp, ("gamma1", q), float(dS[e]) * d_g1)
            else:
                add(comp, ("gamma1", q), -float(dS[e]) * d_g1)
                add(comp, ("gamma2", q), float(dS[e]) * d_g2)
        elif op.name == "dressing":                        # the twirl's pulses on an idle qubit: X, Y, Z at p/3 each
            add(comp, ("dressing", qs[0]), float(dS[e]) / 3)
            add(loss, ("dressing", qs[0]), w * float(dS[e]))
        elif op.name in ("quasistatic", "spectator"):      # one Z error per op (R155 sigma windows; R153 spectators)
            add(comp, (op.name, qs[0]), float(dS[e]))
            add(loss, (op.name, qs[0]), w * float(dS[e]))
        elif op.name == "crosstalk":                       # R153's twirled conditional phase on a coupled pair
            a, b = sorted(qs)
            add(loss, ("crosstalk", a, b), w * float(dS[e]))
        else:
            raise E.EngineError(f"sensitivity: no budget key for an error after a {op.name!r} op")
    # readout: linear in each entry, so a central difference over the smoothed score alone is exact
    if params.readout:
        h = 1e-4
        for j, (a, _) in enumerate(P.meas):
            q = phys[a]
            for key, (r, c) in (("p01", (1, 0)), ("p10", (0, 1))):
                vals = []
                for sgn in (1, -1):
                    cf = [x.copy() for x in conf]
                    cf[j][r, c] += sgn * h * params.readout_factor
                    cf[j][1 - r, c] -= sgn * h * params.readout_factor
                    gh = E.wht(E._smoothed_score(svec, cf))
                    vals.append(float(chi_all @ (q_hat * gh)) / (1 << m))
                add(comp, (key, q), (vals[0] - vals[1]) / (2 * h))
    out.components = comp
    out.loss_budget = loss
    return out


__all__ = ["Sensitivity", "sensitivity", "PAULIS2"]
