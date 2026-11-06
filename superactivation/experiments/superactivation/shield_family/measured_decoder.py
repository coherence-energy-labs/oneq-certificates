"""Can Bob replace the coherent untwisting by MEASURING the shields and correcting the key's phase classically?

Kept branch of one use of P_d (x) A_{2d}, Sylvester u (d = 2^m), encoder V|r> = (1/sqrt d) sum_s |r,s>_P |r,s>_A.
Bob holds key-out k, shield-out B'' (from P) and key-copy c, shield-copy R'' (from A). The paper's decoder measures the
key parity l = k + c and applies the untwisting U^{i,l dag} on (R'', B''), controlled by the key: m Fredkins + m CCZ for
l = 0. The proposed replacement, per shield-qubit pair t = (R''_t, B''_t):
  l = 0: U^{1,0} = D_u F = prod_t CZ_t SWAP_t, whose joint eigenbasis per pair is {|00>, |11>, Psi+, Psi-} with
         eigenvalues +1, -1, +1, -1. W = CH(b -> a) CNOT(a -> b) maps it to the computational basis, and the eigenvalue
         is (-1)^a after W, whatever b is.
  l = 1: U^{1,1} acts as u/sqrt d = H^{(x)m} on span{|aa>}, i.e. as H on each pair's span{|00>, |11>}: CNOT(a -> b) puts the pair on
         a, then measure a in H's eigenbasis.
The product of the per-pair eigenvalues is the phase between the key's two branches: Bob applies Z^{[product = -1]}
to the key (a Pauli-frame update). Claim to test: for EVERY outcome the key is the logical qubit exactly.

  python measured_decoder.py      (m = 1, 2, 3)
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import effective_channel_check as ec  # noqa: E402
import ppt_family as pf  # noqa: E402

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)


def kept_branch_state(m: int, psi: np.ndarray) -> np.ndarray:
    """rho on (k, B''_1..m, c, R''_1..m) for logical input psi; P_d acts on (k, B''); A keeps (c, R'')."""
    d = 2 ** m
    D = 2 * d
    p = 1 / (math.sqrt(d) + 1)
    Ks = ec.kraus_from_choi_state(pf.choi_state(pf.sylvester(m), p), D, D)
    V = ec.encoder(d)                                   # (P-in (x) A-in) x 2, P-in = (key, shield), key first
    v = V @ psi
    rho = 0
    for K in Ks:
        w = np.kron(K, np.eye(D)) @ v
        rho = rho + np.outer(w, w.conj())
    return rho                                          # order: (key, shield-out) (x) (key-copy, shield-copy)


def project(rho, ops, dims):
    """apply a list of (index, operator) single-site operators as a product, rho -> O rho O^dag (O on listed sites)"""
    Op = np.array([[1.0]])
    for s, dsz in enumerate(dims):
        Op = np.kron(Op, ops.get(s, np.eye(dsz)))
    return Op @ rho @ Op.conj().T


def decode_all_outcomes(m: int, psi: np.ndarray, with_x: bool = True):
    """worst fidelity, over every measurement outcome with nonzero probability, of the key after the frame rule
    X^l Z^[product of pair eigenvalues = -1]"""
    rho = kept_branch_state(m, psi)
    # sites: 0 key, 1..m shield-out bits, m+1 key-copy, m+2..2m+1 shield-copy bits (qubits, big-endian)
    n = 2 * m + 2
    dims = [2] * n

    def U2(op, a, b):
        """two-qubit operator on sites a, b as a full matrix"""
        full = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for idx in range(2 ** n):
            bits = [(idx >> (n - 1 - s)) & 1 for s in range(n)]
            col = bits[a] * 2 + bits[b]
            for row in range(4):
                amp = op[row, col]
                if amp == 0:
                    continue
                nb = list(bits)
                nb[a], nb[b] = row >> 1, row & 1
                j = sum(bit << (n - 1 - s) for s, bit in enumerate(nb))
                full[j, idx] += amp
        return full
    CNOT =np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CH = np.zeros((4, 4), dtype=complex)                # control = first site (b), target = second site (a)
    CH[0, 0] = CH[1, 1] = 1
    CH[2:, 2:] = H
    worst, total_p = 1.0, 0.0
    # key parity: CNOT key -> key-copy, measure key-copy
    rho = U2(CNOT, 0, m + 1) @ rho @ U2(CNOT, 0, m + 1).conj().T
    Hev = np.linalg.eigh(H)[1]                          # H's eigenbasis (columns: eigenvalue -1, +1)
    for ell in (0, 1):
        Pc = {m + 1: np.diag([1.0 - ell, float(ell)])}
        r_l = project(rho, Pc, dims)
        if np.real(np.trace(r_l)) < 1e-12:
            continue
        # per pair t: a = shield-copy bit (site m+2+t), b = shield-out bit (site 1+t)
        for t in range(m):
            a, b = m + 2 + t, 1 + t
            if ell == 0:
                Wt = U2(CH, b, a) @ U2(CNOT, a, b)
            else:
                Wt = U2(CNOT, a, b)
            r_l = Wt @ r_l @ Wt.conj().T
        for outs in itertools.product((0, 1), repeat=2 * m):
            ops = dict(Pc)
            sign = 1
            for t in range(m):
                a, b = m + 2 + t, 1 + t
                oa, ob = outs[2 * t], outs[2 * t + 1]
                if ell == 0:
                    ops[a] = np.diag([1.0 - oa, float(oa)])
                    ops[b] = np.diag([1.0 - ob, float(ob)])
                    sign *= (-1) ** oa
                else:
                    vec = Hev[:, oa]                    # outcome oa -> eigenvector with eigenvalue -1 (0) / +1 (1)
                    ops[a] = np.outer(vec, vec.conj())
                    ops[b] = np.diag([1.0 - ob, float(ob)])
                    sign *= (1 if oa == 1 else -1)
            r = project(r_l, ops, dims)
            pr = float(np.real(np.trace(r)))
            if pr < 1e-12:
                continue
            total_p += pr
            # reduce to the key and apply the RULE: frame X^l Z^[product = -1] (with_x=False drops X^l: the first
            # draft's bug, kept as the negative control); also record which Pauli would have been best
            key = np.einsum("ijkj->ik", r.reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))) / pr
            fs = {name: float(np.real(psi.conj() @ (P @ key @ P.conj().T) @ psi)) for name, P in PAULIS.items()}
            rule = ("X" if (ell and with_x) else "") + ("Z" if sign < 0 else "")
            rule = {"": "I", "XZ": "XZ", "X": "X", "Z": "Z"}[rule]
            table[(ell, outs)] = (max(fs, key=fs.get), sign)
            worst = min(worst, fs[rule])
    return worst, total_p


PAULIS = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1, -1]),
          "XZ": np.array([[0, 1], [1, 0]]) @ np.diag([1, -1])}
table: dict = {}


def main() -> int:
    rng = np.random.default_rng(4)
    ok = True
    for m in (1, 2):
        tables = []
        for _ in range(3):
            psi = rng.normal(size=2) + 1j * rng.normal(size=2)
            psi /= np.linalg.norm(psi)
            table.clear()
            worst, tp = decode_all_outcomes(m, psi)
            tables.append(dict(table))
            print(f"m = {m} (d = {2 ** m}): worst fidelity under the frame rule X^l Z^[prod=-1]: {worst:.12f}, "
                  f"total probability {tp:.12f}")
            ok &= worst > 1 - 1e-10 and abs(tp - 1) < 1e-10
        same = all(t == tables[0] for t in tables)
        print(f"   Pauli frame table independent of the input state: {same}")
        ok &= same
        for (ell, outs), (name, sign) in sorted(tables[0].items())[:8]:
            print(f"   l = {ell}, outcomes {outs}: frame {name}, my eigenvalue product {sign:+d}")
    print("MEASURE-AND-CORRECT DECODER", "EXACT" if ok else "NOT exact")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
