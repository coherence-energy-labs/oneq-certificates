"""Is the d = 2 member of the shield family (ppt_family.choi_state with u = [[1, 1], [1, -1]]) Smith and Yard's channel,
entry by entry? (paper 4, referee finding F10)

The earlier check (test_the_d2_member_is_the_papers_channel) compared spectra of the Choi state and of its partial
transpose: necessary, not sufficient. Here the channel of arXiv:2604.27042 (v2 source, sha256 a5c80fa2..., read
2026-11-03) is transcribed INDEPENDENTLY of the repository's other transcriptions, from its supplement:

  Eqs. S2.20-S2.25  K_1 = sqrt((1-p)/2) 1 (x) |0><0|,   K_2 = sqrt((1-p)/4) 1 (x) X,
                    K_3 = sqrt((1-p)/2) Z (x) |1><1|,   K_4 = sqrt((1-p)/4) Z (x) Y,
                    K_5 = sqrt(p) X (x) M_0,            K_6 = sqrt(p) Y (x) M_1,
  Eqs. S2.26-S2.27  M_0 = (sqrt(2 + sqrt 2) |0><0| + sqrt(2 - sqrt 2) |1><1|)/2,
                    M_1 = (sqrt(2 - sqrt 2) |0><0| - sqrt(2 + sqrt 2) |1><1|)/2,       p = 1/(1 + sqrt 2),
  Eqs. S2.4-S2.17   Phi^P = sum_i q_i psi^i_{R'B'} (x) rho^(i)_{R''B''} (the Choi state the Kraus operators define),

the first tensor factor being the key and the second the shield. Both channels are compared through their normalized
Choi matrices on R'R''B'B'' (equal Choi matrices <=> equal channels), with no local unitary applied, in floating point
and EXACTLY (sympy), the three pairs family / Kraus / Choi formula each. Controls that must FAIL: Smith and Yard's
original M_1, whose |1><1| coefficient has the sign typo the supplement's footnote reports (float and exact); the family
member at 0.97 p; and, exactly, the Choi formula with its weights q_0, q_1 and q_2, q_3 exchanged.

  python p2_kraus_comparison.py OUT.json        (schema 2 since 2026-11-03: p2_kraus_comparison_v2.json; seconds)
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import platform
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import ppt_family as pf  # noqa: E402

SOURCE_SHA256 = "a5c80fa27c47f8a7a237defd485ddff1b1cd76ea5b0569716135f7f48ac58e3d"
S2 = math.sqrt(2)
P = 1 / (1 + S2)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
E0 = np.diag([1, 0]).astype(complex)
E1 = np.diag([0, 1]).astype(complex)


def kraus(p: float = P, typo: bool = False) -> list[np.ndarray]:
    m0 = np.diag([math.sqrt(2 + S2), math.sqrt(2 - S2)]) / 2
    m1 = np.diag([math.sqrt(2 - S2), (1 if typo else -1) * math.sqrt(2 + S2)]) / 2
    return [math.sqrt((1 - p) / 2) * np.kron(I2, E0), math.sqrt((1 - p) / 4) * np.kron(I2, X),
            math.sqrt((1 - p) / 2) * np.kron(Z, E1), math.sqrt((1 - p) / 4) * np.kron(Z, Y),
            math.sqrt(p) * np.kron(X, m0), math.sqrt(p) * np.kron(Y, m1)]


def choi_of(Ks: list[np.ndarray]) -> np.ndarray:
    """normalized Choi matrix (id_R (x) P)(Phi^4) on R'R''B'B'', R first"""
    phi = np.zeros(16, dtype=complex)
    for i in range(4):
        phi[5 * i] = 0.5
    Phi = np.outer(phi, phi.conj())
    return sum(np.kron(np.eye(4), K) @ Phi @ np.kron(np.eye(4), K).conj().T for K in Ks)


def choi_formula(p: float = P) -> np.ndarray:
    """Eqs. S2.4-S2.17: sum_i q_i psi^i_{R'B'} (x) rho^(i)_{R''B''}, permuted to R'R''B'B''"""
    def v(x):
        return np.array(x, dtype=complex)
    psi = [v([1, 0, 0, 1]) / S2, v([1, 0, 0, -1]) / S2, v([0, 1, 1, 0]) / S2, v([0, 1, -1, 0]) / S2]
    e00, e11 = v([1, 0, 0, 0]), v([0, 0, 0, 1])
    chip = (math.sqrt(2 + S2) * e00 + math.sqrt(2 - S2) * e11) / 2
    chim = (math.sqrt(2 - S2) * e00 - math.sqrt(2 + S2) * e11) / 2
    rho = [(np.outer(e00, e00) + np.outer(psi[2], psi[2].conj())) / 2,
           (np.outer(e11, e11) + np.outer(psi[3], psi[3].conj())) / 2,
           np.outer(chip, chip.conj()), np.outer(chim, chim.conj())]
    q = [(1 - p) / 2, (1 - p) / 2, p / 2, p / 2]
    S = sum(q[i] * np.kron(np.outer(psi[i], psi[i].conj()), rho[i]) for i in range(4))     # (R', B', R'', B'')
    return S.reshape([2] * 8).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(16, 16)


def exact_comparison(typo: bool = False, swap_weights: bool = False) -> dict:
    """the same comparison in exact arithmetic (sympy): the family's d = 2 Choi state, built from its DEFINITION (the
    paper's Eq. for gamma^l with u = [[1, 1], [1, -1]], p1 = 1/(sqrt 2 + 1)), against BOTH the Choi matrix of the Kraus
    operators of Eqs. S2.20-S2.27 and the Choi-state formula of Eqs. S2.4-S2.17, and those two against each other; an
    entry difference is zero iff its minimal polynomial is x. Controls: `typo` puts Smith and Yard's original sign
    error into M_1 (Kraus side); `swap_weights` exchanges the formula's weights q_0, q_1 with q_2, q_3 (formula side)."""
    import sympy as sp
    s2 = sp.sqrt(2)
    p = 1 / (1 + s2)
    x = sp.Symbol("x")
    eye2 = sp.eye(2)
    Xs, Zs = sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[1, 0], [0, -1]])
    Ys = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    e0, e1 = sp.Matrix([[1, 0], [0, 0]]), sp.Matrix([[0, 0], [0, 1]])
    m0 = sp.diag(sp.sqrt(2 + s2), sp.sqrt(2 - s2)) / 2
    m1 = sp.diag(sp.sqrt(2 - s2), (1 if typo else -1) * sp.sqrt(2 + s2)) / 2
    kp = sp.kronecker_product
    Ks = [sp.sqrt((1 - p) / 2) * kp(eye2, e0), sp.sqrt((1 - p) / 4) * kp(eye2, Xs),
          sp.sqrt((1 - p) / 2) * kp(Zs, e1), sp.sqrt((1 - p) / 4) * kp(Zs, Ys),
          sp.sqrt(p) * kp(Xs, m0), sp.sqrt(p) * kp(Ys, m1)]
    phi = sp.zeros(16, 1)
    for i in range(4):
        phi[5 * i] = sp.Rational(1, 2)
    CK = sp.zeros(16, 16)
    for K in Ks:
        v = kp(sp.eye(4), K) * phi
        CK += v * v.H
    # the family's definition at d = 2: gamma^l on (R', B', R'', B''), then permuted to (R', R'', B', B'')
    u = sp.Matrix([[1, 1], [1, -1]])
    Du = sp.diag(*[u[a, b] for a in range(2) for b in range(2)])
    F = sp.zeros(4, 4)
    for a in range(2):
        for b in range(2):
            F[b * 2 + a, a * 2 + b] = 1
    U10 = Du * F
    U11 = sp.eye(4)
    for i, a in enumerate((0, 3)):
        for j, b in enumerate((0, 3)):
            U11[b, a] = u[i, j] / s2
    sig = {0: sp.eye(4) / 4, 1: sp.diag(sp.Rational(1, 2), 0, 0, sp.Rational(1, 2))}
    Us = {0: [sp.eye(4), U10], 1: [sp.eye(4), U11]}
    rho = sp.zeros(16, 16)
    for ell, w in ((0, s2 / (s2 + 1)), (1, 1 / (s2 + 1))):
        for i in range(2):
            for j in range(2):
                key = sp.zeros(4, 4)
                key[i * 2 + (i + ell) % 2, j * 2 + (j + ell) % 2] = sp.Rational(1, 2)
                rho += w * kp(key, Us[ell][i] * sig[ell] * Us[ell][j].H)
    perm = [0, 2, 1, 3]                                  # (R', B', R'', B'') -> (R', R'', B', B'')

    def idx(t):
        return sum(t[q] * 2 ** (3 - q) for q in range(4))

    def to_RRBB(S):
        out = sp.zeros(16, 16)
        for r in range(16):
            for c in range(16):
                tr = [(r >> (3 - q)) & 1 for q in range(4)]
                tc = [(c >> (3 - q)) & 1 for q in range(4)]
                out[idx([tr[perm[q]] for q in range(4)]), idx([tc[perm[q]] for q in range(4)])] = S[r, c]
        return out
    ours = to_RRBB(rho)
    # Eqs. S2.4-S2.17: sum_i q_i psi^i_{R'B'} (x) rho^(i)_{R''B''}, built on (R', B', R'', B'')
    psi = [sp.Matrix([1, 0, 0, 1]) / s2, sp.Matrix([1, 0, 0, -1]) / s2,
           sp.Matrix([0, 1, 1, 0]) / s2, sp.Matrix([0, 1, -1, 0]) / s2]
    e00, e11 = sp.Matrix([1, 0, 0, 0]), sp.Matrix([0, 0, 0, 1])
    chip = (sp.sqrt(2 + s2) * e00 + sp.sqrt(2 - s2) * e11) / 2
    chim = (sp.sqrt(2 - s2) * e00 - sp.sqrt(2 + s2) * e11) / 2
    rhos = [(e00 * e00.T + psi[2] * psi[2].T) / 2, (e11 * e11.T + psi[3] * psi[3].T) / 2, chip * chip.T, chim * chim.T]
    q = [(1 - p) / 2, (1 - p) / 2, p / 2, p / 2]
    if swap_weights:
        q = [q[2], q[3], q[0], q[1]]
    CF = to_RRBB(sum((q[i] * kp(psi[i] * psi[i].T, rhos[i]) for i in range(4)), sp.zeros(16, 16)))

    def equal_entries(A, B):
        count = 0
        for r in range(16):
            for c in range(16):
                diff = A[r, c] - B[r, c]
                if diff == 0 or sp.minimal_polynomial(diff, x) == x:
                    count += 1
        return count
    return {"entries": 256, "family_vs_kraus_S2.20-S2.27": equal_entries(ours, CK),
            "family_vs_choi_formula_S2.4-S2.17": equal_entries(ours, CF),
            "kraus_vs_choi_formula": equal_entries(CK, CF),
            "method": "sympy: each Choi-entry difference is an algebraic number, and it is zero iff its minimal "
                      "polynomial is x", "sympy": sp.__version__}


def git_blob(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def main(argv) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    Ks = kraus()
    tp = float(np.abs(sum(K.conj().T @ K for K in Ks) - np.eye(4)).max())
    CK, CF = choi_of(Ks), choi_formula()
    ours = pf.choi_state(pf.hadamard2(), P)
    diffs = {"family_d2_vs_kraus_S2.20-S2.27": float(np.abs(ours - CK).max()),
             "family_d2_vs_choi_formula_S2.4-S2.17": float(np.abs(ours - CF).max()),
             "kraus_vs_choi_formula": float(np.abs(CK - CF).max())}
    controls = {"smith_yard_original_M1_sign_typo_vs_family_d2": float(np.abs(ours - choi_of(kraus(typo=True))).max()),
                "family_at_0.97p_vs_kraus": float(np.abs(pf.choi_state(pf.hadamard2(), 0.97 * P) - CK).max())}
    tol = 1e-14
    exact = exact_comparison()
    pairs = ("family_vs_kraus_S2.20-S2.27", "family_vs_choi_formula_S2.4-S2.17", "kraus_vs_choi_formula")
    controls["exact_check_with_the_M1_sign_typo_family_vs_kraus_entries_equal"] = exact_comparison(typo=True)[pairs[0]]
    controls["exact_check_with_swapped_formula_weights_family_vs_choi_formula_entries_equal"] = exact_comparison(
        swap_weights=True)[pairs[1]]
    ok = (tp < tol and all(x < tol for x in diffs.values())
          and all(controls[k] > 1e-3 for k in ("smith_yard_original_M1_sign_typo_vs_family_d2",
                                               "family_at_0.97p_vs_kraus"))
          and all(exact[k] == exact["entries"] for k in pairs)
          and controls["exact_check_with_the_M1_sign_typo_family_vs_kraus_entries_equal"] < exact["entries"]
          and controls["exact_check_with_swapped_formula_weights_family_vs_choi_formula_entries_equal"]
          < exact["entries"])
    rec = {"schema": "oneq.superactivation.p2-kraus-comparison/2",
           "supersedes": "p2_kraus_comparison.json (schema 1: exact against the Kraus operators only)",
           "source": "arXiv:2604.27042v2 source archive, supplemental.tex, Eqs. S2.4-S2.27",
           "source_sha256": SOURCE_SHA256, "p": P, "twist": "u = [[1, 1], [1, -1]]",
           "comparison": "normalized Choi matrices on R'R''B'B'', entrywise max |difference|; no local unitary applied",
           "kraus_trace_preservation_deviation": tp, "max_abs_differences": diffs, "controls_must_differ": controls,
           "tolerance": tol, "exact_symbolic": exact, "verdict": "IDENTICAL" if ok else "DIFFERENT",
           "provenance": {"code_git_blobs": {f: git_blob(ROOT / f) for f in (
               "experiments/superactivation/shield_family/p2_kraus_comparison.py",
               "experiments/superactivation/shield_family/ppt_family.py")},
               "python": platform.python_version(), "numpy": np.__version__}}
    pathlib.Path(argv[0]).write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(rec["max_abs_differences"]), json.dumps(controls), json.dumps(exact), "VERDICT", rec["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
