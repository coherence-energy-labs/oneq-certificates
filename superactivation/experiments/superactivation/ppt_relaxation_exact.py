"""The PPT-assisted relaxation's value, exactly: F_PPT(N^{(x)n}, 2) = 1 - 2^-(n+1) for the flagged channel N~_p and for
the 50% erasure channel (paper 4, referee finding N2; it settles R137's measured values and its 1 - 2^-(n+1) hypothesis).

The SDP is ppt_relaxation_probe.py's (Leung-Matthews): F_PPT(N, M) = max Tr(J_N W) over 0 <= W <= rho (x) 1_B,
Tr rho = 1, -rho (x) 1_B <= M W^{T_B} <= rho (x) 1_B, with J_N = sum_ij |i><j| (x) N(|i><j|) (so Tr_B J_N = 1_A), and,
the flag being classical, one block W_z per flag string z with a common rho. The paper proves the UPPER bound by hand
(each block's value is at most 1, and at most 1/M when every use is erased, because that block's channel is
entanglement-breaking). This script checks the matching FEASIBLE POINT exactly, for n = 1, 2 and both channels:

    rho = 1 / 2^n;   W_z = (Phi_i / 2) (x) rho_rest (x) 1_rest  if use i is the first kept use of z,
                     W_z = rho (x) 1 / 2                         if every use of z is erased,

every constraint in exact rational arithmetic (sympy: positive semidefiniteness of W_z, rho (x) 1 - W_z and
rho (x) 1 -+ M W_z^{T_B}), and the objective sum_z 2^-n Tr(J_z W_z) symbolically in p, which must equal 1 - 2^-(n+1) for
every p. Control that must FAIL: the kept block without its factor 1/2 (W_z = Phi_i (x) ...) violates W_z <= rho (x) 1.
The probe's measured optima are compared with the exact values.

  python ppt_relaxation_exact.py OUT.json        (seconds)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import platform
import sys

import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = ROOT / "evidence" / "superactivation" / "ppt_relaxation_probe.json"
p = sp.Symbol("p", positive=True)
M = 2


def choi_id():
    J = sp.zeros(4, 4)
    for i in range(2):
        for j in range(2):
            J[2 * i + i, 2 * j + j] = 1                         # sum |ii><jj|, A first
    return J


def choi_noisy():                                               # X^p o full dephasing (Z basis): entanglement-breaking
    J = sp.zeros(4, 4)
    for i in range(2):
        for o in range(2):
            J[2 * i + o, 2 * i + o] = (1 - p) if o == i else p
    return J


def choi_erased():                                              # the erasure symbol, as the probe models it
    return sp.kronecker_product(sp.eye(2), sp.diag(1, 0))


def to_AQ(blocks: list[sp.Matrix]) -> sp.Matrix:
    """tensor product of per-use operators given in (A_i, Q_i) order, returned in (A_1..A_n, Q_1..Q_n) order"""
    n = len(blocks)
    T = blocks[0]
    for B in blocks[1:]:
        T = sp.kronecker_product(T, B)                          # order A1 Q1 A2 Q2 ...
    dim = 4 ** n
    out = sp.zeros(dim, dim)

    def perm(idx):
        bits = [(idx >> (2 * n - 1 - q)) & 1 for q in range(2 * n)]   # A1 Q1 A2 Q2 ...
        new = [bits[2 * i] for i in range(n)] + [bits[2 * i + 1] for i in range(n)]
        return sum(b << (2 * n - 1 - q) for q, b in enumerate(new))
    for r in range(dim):
        for c in range(dim):
            if T[r, c] != 0:
                out[perm(r), perm(c)] = T[r, c]
    return out


def ptrans_B(W: sp.Matrix, dA: int, dB: int) -> sp.Matrix:
    out = sp.zeros(dA * dB, dA * dB)
    for a in range(dA):
        for b in range(dB):
            for a2 in range(dA):
                for b2 in range(dB):
                    out[a * dB + b2, a2 * dB + b] = W[a * dB + b, a2 * dB + b2]
    return out


def psd(A: sp.Matrix) -> bool:
    return bool(A.is_positive_semidefinite)


def check(n: int, erased_choi, halve: bool = True) -> dict:
    phi = sp.zeros(4, 4)                                        # Phi on (A_i, Q_i), normalized: (|00> + |11>)/sqrt 2
    for i in range(2):
        for j in range(2):
            phi[2 * i + i, 2 * j + j] = sp.Rational(1, 2)
    half = sp.eye(2) / 2
    rho = sp.eye(2 ** n) / 2 ** n
    RI = sp.kronecker_product(rho, sp.eye(2 ** n))
    value, feasible, worst = 0, True, None
    for z in itertools.product((0, 1), repeat=n):               # 0 = kept, 1 = erased
        Jz = to_AQ([choi_id() if zi == 0 else erased_choi() for zi in z])
        if 0 in z:
            i = z.index(0)
            per_use = [(phi / 2 if halve else phi) if q == i else sp.kronecker_product(half, sp.eye(2))
                       for q in range(n)]
            Wz = to_AQ(per_use)
        else:
            Wz = RI / 2
        PT = ptrans_B(Wz, 2 ** n, 2 ** n)
        conds = {"W>=0": psd(Wz), "rho1-W>=0": psd(RI - Wz), "rho1-MW^T>=0": psd(RI - M * PT),
                 "rho1+MW^T>=0": psd(RI + M * PT)}
        if not all(conds.values()):
            feasible = False
            worst = {"pattern": "".join(map(str, z)), **{k: v for k, v in conds.items() if not v}}
        value += sp.Rational(1, 2 ** n) * (Jz * Wz).trace()
    value = sp.simplify(value)
    return {"n": n, "feasible": feasible, "first_violation": worst, "value": str(value),
            "value_independent_of_p": not value.has(p),
            "equals_1_minus_2^-(n+1)": sp.simplify(value - (1 - sp.Rational(1, 2 ** (n + 1)))) == 0}


def git_blob(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def main(argv) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    measured = {r["n"]: r for r in probe["rows"]}
    rows = []
    for n in (1, 2):
        for name, choi, key in (("flagged channel N~_p", choi_noisy, "F_PPT_flagged"),
                                ("50% erasure channel", choi_erased, "F_PPT_erasure")):
            r = check(n, choi)
            r["channel"] = name
            r["probe_measured"] = measured[n][key]
            r["probe_minus_exact"] = measured[n][key] - float(sp.Rational(r["value"]))
            rows.append(r)
    control = check(2, choi_noisy, halve=False)
    ok = (all(r["feasible"] and r["value_independent_of_p"] and r["equals_1_minus_2^-(n+1)"]
              and abs(r["probe_minus_exact"]) < 1e-6 for r in rows) and not control["feasible"])
    rec = {"schema": "oneq.superactivation.ppt-relaxation-exact/1",
           "sdp": "F_PPT(N, 2) of ppt_relaxation_probe.py (Leung-Matthews), flag-block form",
           "claim": "an exact feasible point with value 1 - 2^-(n+1) for every p, n = 1, 2, both channels; the matching "
                    "upper bound is proved in the paper",
           "rows": rows, "control_kept_block_without_one_half": {"feasible": control["feasible"],
                                                                  "first_violation": control["first_violation"]},
           "verdict": "EXACT" if ok else "FAILED",
           "provenance": {"code_git_blobs": {f: git_blob(ROOT / f) for f in (
               "experiments/superactivation/ppt_relaxation_exact.py", "experiments/superactivation/ppt_relaxation_probe.py")},
               "python": platform.python_version(), "sympy": sp.__version__}}
    pathlib.Path(argv[0]).write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print(json.dumps([{k: r[k] for k in ("n", "channel", "feasible", "value", "probe_minus_exact")} for r in rows]),
          "control feasible:", control["feasible"], control["first_violation"], "VERDICT", rec["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
