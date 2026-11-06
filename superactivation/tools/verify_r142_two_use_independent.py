#!/usr/bin/env python3
"""Independent small-matrix physical recovery witness for ONE_Q's R142 two-use codes.

No ONE_Q producer, flint, optimizer, or certificate generator import.
Uses 4x4 real quaternion and 2x2 Hermitian eigenproblems to construct
branch-conditioned, physically implementable decoder operations.
This is a numerical achievable-fidelity witness, NOT a rigorous interval proof.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

I = np.eye(2, dtype=complex)  # noqa: E741  (the Pauli identity, beside X, Y, Z)
PAULIS = (I, np.array([[0, 1], [1, 0]], complex),
          np.array([[0, -1j], [1j, 0]], complex),
          np.diag([1, -1]).astype(complex))


class WitnessRefused(ValueError):
    pass


def _matrix(cert: dict):
    data = cert.get("code_columns")
    if not isinstance(data, list) or len(data) != 4 or any(len(row) != 2 for row in data):
        raise WitnessRefused("missing 4 x 2 encoder witness")
    try:
        V = np.array([[complex(float(pair[0]), float(pair[1])) for pair in row]
                      for row in data], dtype=complex)
    except (TypeError, ValueError, IndexError) as exc:
        raise WitnessRefused("invalid complex encoder entries") from exc
    if not np.isfinite(V).all():
        raise WitnessRefused("nonfinite encoder")
    orth = float(np.linalg.norm(V.conj().T @ V - np.eye(2), ord=2))
    if orth > 1e-10:
        raise WitnessRefused("encoder is not sufficiently close to an isometry")
    # Polar correction gives an isometry realizable by a physical encoder.
    # This numerical correction is tiny, but is not a formal error bound.
    U, _, Vh = np.linalg.svd(V, full_matrices=False)
    return U @ Vh, orth


def _conditional_unitary_decode(V, p, flagged):
    """For observed Z bit c on a flagged qubit, maximize a physical SU(2)
    correction on the untouched qubit.

    U = a I + i(b X + c Y + d Z), where a,b,c,d are REAL, norm 1.
    Fidelity is quadratic in these four coefficients, so the largest
    eigenvector of a 4x4 real symmetric matrix defines an actual decoder.
    """
    total, witnesses = 0.0, []
    source = V.reshape(2, 2, 2)  # physical a1, a2, logical
    for observed in (0, 1):
        K = np.zeros((4, 4), dtype=float)
        for actual in (0, 1):
            M = source[actual, :, :] if flagged == 0 else source[:, actual, :]
            amps = np.array([np.trace(op @ M) if j == 0 else
                             1j * np.trace(op @ M) for j, op in enumerate(PAULIS)])
            K += (1-p if observed == actual else p) * np.outer(amps, amps.conj()).real
        K = 0.5 * (K + K.T)
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        q = eigenvectors[:, -1]
        rotation = q[0]*I + 1j * sum((q[j]*PAULIS[j] for j in range(1, 4)),
                                     start=np.zeros_like(I))
        if not np.allclose(rotation.conj().T @ rotation, I, rtol=0, atol=1e-12):
            raise WitnessRefused("conditional decoder not unitary")
        total += float(q @ K @ q)
        witnesses.append({"observed_bit": observed, "su2_quaternion": q.tolist()})
    return total / 4, witnesses


def _both_flagged_decode(V, p):
    """If both qubits were Z-measured, Bob performs measure-and-prepare.
    Conditioned on observed bits c0,c1, the largest eigenvector of the
    reference's 2x2 conditional density matrix specifies Bob's pure output.
    """
    total, witnesses = 0.0, []
    for c0 in (0, 1):
        for c1 in (0, 1):
            S = np.zeros((2, 2), dtype=complex)
            for m0 in (0, 1):
                for m1 in (0, 1):
                    v = V[2*m0 + m1, :]
                    weight = (1-p if c0 == m0 else p) * (1-p if c1 == m1 else p)
                    S += weight * np.outer(v, v.conj())
            vals, vecs = np.linalg.eigh((S + S.conj().T) / 2)
            total += float(vals[-1])
            w = vecs[:, -1]
            witnesses.append({"observed_bits": [c0, c1],
                              "prepared_conjugate_reference": [[float(z.real), float(-z.imag)]
                                                               for z in w]})
    return total / 4, witnesses


def verify(cert: dict) -> dict:
    d = cert.get("d")
    if type(d) is not int or d < 2:
        raise WitnessRefused("invalid shield dimension")
    V, deviation = _matrix(cert)
    p = 1 / (1 + math.sqrt(d))
    recorded_p = cert.get("p")
    if recorded_p is not None and (type(recorded_p) not in (float, int) or
                                   not math.isclose(recorded_p, p, rel_tol=0, abs_tol=1e-12)):
        raise WitnessRefused("certificate uses an inconsistent flip probability")
    one_a1, unitary_a1 = _conditional_unitary_decode(V, p, 0)
    one_a2, unitary_a2 = _conditional_unitary_decode(V, p, 1)
    both, measure_prepare = _both_flagged_decode(V, p)
    F = (1 + one_a1 + one_a2 + both) / 4
    reference = cert.get("certified_lower_bound_float")
    if type(reference) not in (int, float) or not math.isfinite(reference):
        raise WitnessRefused("missing finite original comparison number")
    if d in (20, 21, 22) and abs(F-reference) > 1e-8:
        raise WitnessRefused("independent physical decoder differs from recorded R142 witness")
    if "exceeds_three_quarters" in cert and cert["exceeds_three_quarters"] is not (F > .75):
        raise WitnessRefused("certificate crossing label disagrees with physical decoder")
    if "certified_margin_lower_bound" in cert and not math.isclose(
            cert["certified_margin_lower_bound"], reference-.75, rel_tol=0, abs_tol=1e-10):
        raise WitnessRefused("certificate margin differs from recorded lower bound minus 3/4")
    return {"shield_dimension": d, "flip_probability": p,
            "achievable_fidelity_double_precision": F,
            "branch_fidelities": {"both_kept": 1, "first_flagged": one_a1,
                                  "second_flagged": one_a2, "both_flagged": both},
            "reported_certificate_lower_bound": reference,
            "difference_from_recorded": F-reference,
            "crosses_three_quarters_numerically": F > .75,
            "encoder_gram_deviation_before_polar": deviation,
            "recovery_witnesses": {"first_flagged": unitary_a1, "second_flagged": unitary_a2,
                                   "both_flagged": measure_prepare},
            "status": "INDEPENDENT_NUMERICAL_PHYSICAL_DECODER_WITNESS_NOT_INTERVAL_PROOF"}


def main(args):
    if not args:
        raise SystemExit("usage: python tools/verify_r142_two_use_independent.py CERT_D20.json [CERT_D21.json ...]")
    for path in args:
        cert = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        result = verify(cert)
        print(json.dumps({"source": path, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
