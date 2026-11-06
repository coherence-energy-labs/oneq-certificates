"""Circuits for P (private Horodecki PPT channel), A (50% erasure), and Prop. 1's E~ / D~.

Two implementation profiles for the channels themselves:
  COHERENT  -- Stinespring dilation: system + environment qubits, one unitary, environment discarded.
  RANDOMIZED-- the environment is a classical random seed drawn by the controller each shot (exact for
               mixed-unitary channels: P is a mixture of six 2-qubit unitaries, A of "keep"/"erase").
Both are exact implementations of the CHANNEL (verified Choi); RANDOMIZED is the cheaper one and is
how synthetic channels are normally run on hardware.

Qubit roles for one use of P (x) A inside Prop. 1:
  k1 = A1' -> B1'  (key, carries the logical qubit)       s1 = A1'' -> B1'' (shield)
  k2 = A2'          (key copy, through A)                  s2 = A2''         (shield copy, through A)
  P acts on (k1, s1) with key FIRST in the paper's tensor order; A acts on (k2, s2).
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate

import sclib as L

p = L.P_FLIP
C8 = math.sqrt(2 + L.SQ2) / 2   # cos(pi/8)
S8 = math.sqrt(2 - L.SQ2) / 2   # sin(pi/8)


# ----------------------------------------------------------------------------------------------
# P: mixed-unitary decomposition  P = sum_j w_j U_j . U_j^dag   (verified in check_paper_objects.py)
# ----------------------------------------------------------------------------------------------
P_UNITARY_WEIGHTS = [(1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, p / 2, p / 2]


def p_unitary_circuit(j: int) -> QuantumCircuit:
    """U_j on qubits (k, s) = (0, 1). Up to a global phase:
       U1 = CZ, U2 = CZ Z_s, U3 = X_s, U4 = Z_k Y_s, U5 = X_k D+, U6 = X_k D-,
       D+ = P(-pi/4)_k S^dag_s CZ, D- = P(+pi/4)_k S_s CZ."""
    qc = QuantumCircuit(2, name=f"P_U{j + 1}")
    k, s = 0, 1
    if j == 0:
        qc.cz(k, s)
    elif j == 1:
        qc.cz(k, s); qc.z(s)
    elif j == 2:
        qc.x(s)
    elif j == 3:
        qc.z(k); qc.y(s)
    elif j == 4:
        qc.cz(k, s); qc.p(-math.pi / 4, k); qc.sdg(s); qc.x(k)
    elif j == 5:
        qc.cz(k, s); qc.p(math.pi / 4, k); qc.s(s); qc.x(k)
    return qc


def p_coherent_circuit_pp() -> QuantumCircuit:
    """Same Stinespring as p_coherent_circuit, select phase synthesized as one phase polynomial
    (pp_synth.py): 20 CZ instead of 26 after compilation."""
    import pp_synth
    qc = QuantumCircuit(5, name="P_coh_pp")
    k, s, ea, eb, ec = 0, 1, 2, 3, 4
    th_a = 2 * math.asin(math.sqrt((1 + p) / 2))
    th_c = 2 * math.acos(math.sqrt(2 * p / (1 + p)))
    qc.ry(th_a, ea)
    qc.h(eb)
    qc.ry(-th_c / 2, ec); qc.cx(ea, ec); qc.ry(th_c / 2, ec)
    qc.compose(pp_synth.select_phase_circuit(), inplace=True)
    qc.x(s); qc.cx(ea, s)
    qc.cx(ec, k)
    return qc


def p_coherent_circuit() -> QuantumCircuit:
    """Stinespring dilation of P with 3 environment qubits (ea, eb, ec) -- the minimum, Choi rank 6.

    Env coin: ea = 1 w.p. (1+p)/2 ("branch contains CZ"), eb uniform, ec = 1 w.p. 2p/(1+p) given ea=1.
    Select: U = X_s^(1-ea) X_k^(ec) . exp(i phi), with the phase polynomial (derived in RESULTS.md)
      phi = pi eb s + pi ea k s + pi eb k - pi ea eb k + (pi/2) eb ec k - (pi/4) ec k - (pi/2) ec s
    (the env value ea=0, ec=1 never occurs, which is what makes phi this sparse)."""
    q = QuantumRegister(5, "q")
    qc = QuantumCircuit(q, name="P_coh")
    k, s, ea, eb, ec = 0, 1, 2, 3, 4
    th_a = 2 * math.asin(math.sqrt((1 + p) / 2))
    th_c = 2 * math.acos(math.sqrt(2 * p / (1 + p)))
    qc.ry(th_a, ea)
    qc.h(eb)
    # controlled split of ec on a |0> target with ONE CNOT: ea=1 -> Ry(th_c)|1> (sign is a Kraus phase)
    qc.ry(-th_c / 2, ec); qc.cx(ea, ec); qc.ry(th_c / 2, ec)
    # phase polynomial
    qc.cz(eb, s)
    qc.ccz(ea, k, s)
    qc.cz(eb, k)
    qc.ccz(ea, eb, k)
    qc.mcp(math.pi / 2, [eb, ec], k)
    qc.cp(-math.pi / 4, ec, k)
    qc.cp(-math.pi / 2, ec, s)
    # non-diagonal part
    qc.x(s); qc.cx(ea, s)
    qc.cx(ec, k)
    return qc


# ----------------------------------------------------------------------------------------------
# A: 50% erasure on C^4 = (k2, s2); output (f, k2, s2) with |e> := |1>_f |00>
# ----------------------------------------------------------------------------------------------
def move_gate(qc, f, x, e, impl="toffoli"):
    """|f, x, 0_e> -> f=0: |0, x, 0>,  f=1: |1, 0, x>  (correct on e=|0> inputs)."""
    if impl == "toffoli":
        qc.ccx(f, x, e)
        qc.cx(e, x)
    elif impl == "rccx":   # relative-phase Toffoli is exact here because e enters in |0>
        qc.rccx(f, x, e)
        qc.cx(e, x)
    else:
        raise ValueError(impl)


def a_coherent_circuit(move_impl="toffoli") -> QuantumCircuit:
    """Stinespring of A: qubits (k2, s2, f, fE, e1, e2); f is the flag (output), fE/e1/e2 environment.
    V|psi> = ( |0>_f |psi>_{k2 s2} |0>_fE |00>_e  +  |1>_f |00>_{k2 s2} |1>_fE |psi>_e ) / sqrt 2,
    the paper's V = (|psi>_B |e>_E + |e>_B |psi>_E)/sqrt 2 (l.317) with |e> embedded as |1>|00>."""
    qc = QuantumCircuit(6, name="A_coh")
    k2, s2, f, fE, e1, e2 = range(6)
    qc.ry(math.pi / 2, f)
    qc.cx(f, fE)
    move_gate(qc, f, k2, e1, move_impl)
    move_gate(qc, f, s2, e2, move_impl)
    return qc


def a_target_choi() -> np.ndarray:
    """J(A) with the 5-dim output embedded into (f, k2, s2)."""
    JA = L.choi_from_kraus(L.erasure_kraus())
    Emb = np.zeros((8, 5))
    for i in range(4):
        Emb[i, i] = 1        # |0>_f |i>
    Emb[4, 4] = 1            # |e> -> |1>_f |00>  (index f*4 + 0)
    M = np.kron(np.eye(4), Emb)
    return M @ JA @ M.T


# ----------------------------------------------------------------------------------------------
# E~ and D~ (supplement S3, Theorem effective_channel_general, K = d_s = 2)
# ----------------------------------------------------------------------------------------------
def e_tilde(qc, k1, s1, k2, s2):
    """Append Phi^{d_s} on (A1'', A2'') and |0> on A2', then CNOT A1' -> A2' (eq. encoder_protocol)."""
    qc.h(s1)
    qc.cx(s1, s2)
    qc.cx(k1, k2)


def untwist_unitary(order="s2_first") -> np.ndarray:
    """V^dag of eq. (untwisting_unitary) on qiskit qargs (k1, k2, s1, s2) [B1', A2', B1'', A2''].
    The projector printed on A2'' is read as A2' (the key copy) -- see RESULTS.md, ambiguity A1.
    U^{i,l} acts on (A2'', B1'') in that tensor order (s2_first) or the reverse (s1_first)."""
    h = 1 / L.SQ2
    U = {(0, 0): np.eye(4), (0, 1): np.eye(4),
         (1, 0): np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=complex),
         (1, 1): np.array([[h, 0, 0, h], [0, 0, 1, 0], [0, 1, 0, 0], [h, 0, 0, -h]], dtype=complex)}
    # build on big-endian order (k1, k2, sA, sB) then convert to qiskit little-endian over (k1,k2,s1,s2)
    V = np.zeros((16, 16), dtype=complex)
    for l in range(2):
        for i in range(2):
            b1 = (i + l) % 2
            Pk = np.zeros((4, 4))
            Pk[b1 * 2 + i, b1 * 2 + i] = 1            # |i+l><i+l|_{k1} (x) |i><i|_{k2}
            V += np.kron(Pk, U[(i, l)].conj().T)      # shield factor on (first, second)
    # V is on (k1, k2, first, second) big-endian. first/second = (s2, s1) or (s1, s2)
    T = V.reshape([2] * 8)
    if order == "s2_first":
        T = T.transpose(0, 1, 3, 2, 4, 5, 7, 6)       # -> (k1, k2, s1, s2)
    # big-endian (k1,k2,s1,s2) -> qiskit little-endian over qargs [k1,k2,s1,s2]: reverse axes
    T = T.transpose(3, 2, 1, 0, 7, 6, 5, 4)
    return T.reshape(16, 16)


def untwist_structured(qc, k1, s1, k2, s2):
    """The paper's V^dag with its structure exposed: U^{1,0} and U^{1,1} both act as SWAP on the odd
    shield-parity subspace and as Z resp. H on the even one, so after CX(s2 -> s1) (s1 = parity)
    V^dag is ONE uniformly controlled single-qubit gate on s2 (controls s1, k1, k2), then CX back.
    Checked equal to untwist_unitary() as an operator in verify_channels.py."""
    from qiskit.circuit.library import UCGate
    I = np.eye(2, dtype=complex)
    Xm = np.array([[0, 1], [1, 0]], dtype=complex)
    Zm = np.diag([1, -1]).astype(complex)
    Hm = np.array([[1, 1], [1, -1]], dtype=complex) / L.SQ2
    gates = []
    for i in range(8):                     # i = s1 + 2 k1 + 4 k2
        s1v, k1v, k2v = i & 1, (i >> 1) & 1, (i >> 2) & 1
        if k2v == 0:
            gates.append(I)
        elif s1v == 1:
            gates.append(Xm)
        elif k1v == 1:
            gates.append(Zm)
        else:
            gates.append(Hm)
    qc.cx(s2, s1)
    qc.append(UCGate(gates), [s2, s1, k1, k2])
    qc.cx(s2, s1)


def d0_paper(qc, k1, s1, k2, s2, l_bit, order="s2_first"):
    """Paper's D^0 (no erasure, Fig. decoder_noerasure, eq. decoder_D0): untwist V^dag (coherent),
    discard the shields B1'' A2'', parity measurement on B1' A2' (CNOT B1'->A2', read A2'), X^l
    correction on B1', inverse CNOT + discard A2'. [After the destructive parity read A2' is the
    product state |l>, so 'inverse CNOT then discard A2'' equals 'discard A2''.]"""
    if order == "generic":
        qc.append(UnitaryGate(untwist_unitary("s2_first"), label="Vdag"), [k1, k2, s1, s2])
    else:   # structured form, operator-equal to untwist_unitary (check_untwist.py)
        untwist_structured(qc, k1, s1, k2, s2)
    qc.reset(s1)
    qc.reset(s2)
    qc.cx(k1, k2)
    qc.measure(k2, l_bit)
    with qc.if_test((l_bit, 1)):
        qc.x(k1)
    qc.reset(k2)


def d0_opt(qc, k1, s1, k2, s2, l_bit, m_bit):
    """Equivalent D^0 with MEASUREMENT-BASED untwisting (verified equal as a channel): the shields
    are about to be discarded, so instead of the doubly-controlled V^dag Bob measures the +-1 twist
    observable U^{1,l} on the shields; its eigenvalue decides a Z on the key.
      parity: CNOT k1->k2, read k2 -> l ; X^l on k1
      twist : CNOT s2->s1 ; CH s1->s2 ; [l=1: Ry(-pi/4) s2] ; read s2 -> m ; Z^m on k1
    3 CZ, 2 mid-circuit measurements, 3 feed-forward single-qubit gates."""
    qc.cx(k1, k2)
    qc.measure(k2, l_bit)
    qc.cx(s2, s1)
    qc.ch(s1, s2)
    with qc.if_test((l_bit, 1)):
        qc.ry(-math.pi / 4, s2)
        qc.x(k1)
    qc.measure(s2, m_bit)
    with qc.if_test((m_bit, 1)):
        qc.z(k1)
    qc.reset(s1)
    qc.reset(k2)
    qc.reset(s2)


def d0_coherent(qc, k1, s1, k2, s2):
    """Measurement-free D^0 (deferred-measurement form of d0_opt): no mid-circuit reads."""
    qc.cx(k1, k2)
    qc.cx(k2, k1)                 # k1 <- i, k2 holds l
    qc.cx(s2, s1)
    qc.ch(s1, s2)
    qc.cry(-math.pi / 4, k2, s2)  # Ry(-pi/4) on s2 iff l = 1
    qc.cz(s2, k1)
    qc.reset(s1)
    qc.reset(k2)
    qc.reset(s2)


def d1(qc, k1, s1, k2, s2):
    """Paper's D^1 (erasure): discard B1'', A2', A2'' (here: reset, so the qubits can be reused)."""
    qc.reset(s1)
    qc.reset(k2)
    qc.reset(s2)


def d0(qc, k1, s1, k2, s2, l_bit, m_bit, decoder, order="s2_first"):
    if decoder == "paper":
        d0_paper(qc, k1, s1, k2, s2, l_bit, order)
    elif decoder == "opt":
        d0_opt(qc, k1, s1, k2, s2, l_bit, m_bit)
    elif decoder == "coherent":
        d0_coherent(qc, k1, s1, k2, s2)
    else:
        raise ValueError(decoder)


def d_tilde_dynamic(qc, k1, s1, k2, s2, z, l_bit, m_bit, decoder, order="s2_first"):
    """D~ with the erasure flag already read into clbit z: if z == 0 run D^0 else D^1."""
    with qc.if_test((z, 0)) as else_:
        d0(qc, k1, s1, k2, s2, l_bit, m_bit, decoder, order)
    with else_:
        d1(qc, k1, s1, k2, s2)


# ----------------------------------------------------------------------------------------------
# full single-use blocks: logical qubit in on k1 -> logical qubit out on k1 + classical flag z
# ----------------------------------------------------------------------------------------------
def block_coherent(decoder="opt", move_impl="toffoli", order="s2_first") -> QuantumCircuit:
    """E~ -> (P coherent) (x) (A coherent) -> measure flag -> D~. Qubits:
    0 k1, 1 s1, 2 k2, 3 s2, 4-6 P env, 7 f, 8 fE, 9-10 A env.  clbits: z, l, m."""
    qc = QuantumCircuit(11, 3, name="block_coh")
    k1, s1, k2, s2 = 0, 1, 2, 3
    e_tilde(qc, k1, s1, k2, s2)
    qc.compose(p_coherent_circuit_pp(), qubits=[k1, s1, 4, 5, 6], inplace=True)
    qc.compose(a_coherent_circuit(move_impl), qubits=[k2, s2, 7, 8, 9, 10], inplace=True)
    z, lb, mb = qc.clbits
    qc.measure(7, z)          # the erasure measurement Pi_{B2} (flag qubit)
    d_tilde_dynamic(qc, k1, s1, k2, s2, z, lb, mb, decoder, order)
    return qc


def block_randomized(jP: int, erased: int, decoder="opt", order="s2_first") -> QuantumCircuit:
    """One shot of the RANDOMIZED profile with the controller's draws (jP, erased) fixed: P is the
    unitary U_jP, A erases iff `erased`. The flag is the controller's classical draw, so Bob's branch
    is selected at compile time (no flag measurement, no flag qubit). On erasure the A2 qubits are
    discarded (reset) and nothing of A2 reaches Bob. Qubits 0 k1, 1 s1, 2 k2, 3 s2; clbits l, m."""
    qc = QuantumCircuit(4, 2, name=f"block_rand_P{jP + 1}_e{erased}")
    k1, s1, k2, s2 = 0, 1, 2, 3
    e_tilde(qc, k1, s1, k2, s2)
    qc.compose(p_unitary_circuit(jP), qubits=[k1, s1], inplace=True)
    lb, mb = qc.clbits
    if erased:
        d1(qc, k1, s1, k2, s2)
    else:
        d0(qc, k1, s1, k2, s2, lb, mb, decoder, order)
    return qc
