"""R178: deterministic symmetric-subspace logical-qubit code search on N~_p^n.

This is a NUMERICAL proposer only. Codes act on n flagged channel USES
and use a non-physical reduced channel. It is not an IBM protocol or an
exact rational certificate. Candidate values have no superactivation claim
until an independent decoder trace, all-component-channel capacity bound
and a rigorous fidelity certificate are supplied.

The two-use code in existing ONE_Q is not assumed optimal at higher n.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation as S  # noqa: E402


def polar(z):
    u, _, vh = np.linalg.svd(z, full_matrices=False)
    return u @ vh


def closed_v(n):
    """Dicke-coordinate encoding |0_L> = |+>^n, |1_L> = |->^n."""
    v = np.zeros((n + 1, 2), dtype=complex)
    for w in range(n + 1):
        amp = math.sqrt(math.comb(n, w) / (2**n))
        v[w, 0] = amp
        v[w, 1] = (-1) ** w * amp
    assert np.max(np.abs(v.conj().T @ v - np.eye(2))) < 1e-12
    return v


def x_of(v):
    n = v.shape[0] - 1
    if np.max(np.abs(v.conj().T @ v - np.eye(2))) > 1e-9:
        raise ValueError("encoder isometry not valid")
    phi = np.concatenate([v[:, 0], v[:, 1]])
    X = np.outer(phi, phi.conj())
    S.check_encoder(X, n)
    return X


def objective(v, n, p, iters=160):
    decs, f, fk = S.decode_power(x_of(v), n, iters=iters, p=p)
    return decs, f, fk


def proposal(n, p, seed, steps, decoder_steps=75):
    rng = np.random.default_rng(seed)
    v = (
        closed_v(n)
        if seed == 0
        else polar(rng.normal(size=(n + 1, 2)) + 1j * rng.normal(size=(n + 1, 2)))
    )
    decs, f, _ = objective(v, n, p, iters=decoder_steps)
    values = [f]
    for k in range(steps):
        G = S.encoder_gradient(decs, n, p=p)
        y = (G @ np.concatenate([v[:, 0], v[:, 1]])).reshape(2, n + 1).T
        candidate = polar(y)
        cdecs, nf, _ = objective(candidate, n, p, iters=decoder_steps)
        if nf + 2e-9 < f:
            raise RuntimeError(
                f"non-monotone proposal n={n} seed={seed} k={k}: {nf} < {f}"
            )
        v, decs, f = candidate, cdecs, nf
        values.append(f)
    # final independent recomputation with a different (longer) decoder pass
    _, replay, fk = objective(v, n, p, iters=500)
    return v, replay, fk, values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    if (
        not 2 <= args.n <= 12
        or not 1 <= args.m <= 18
        or not 1 <= args.steps <= 1000
        or not 1 <= args.seeds <= 1000
    ):
        raise SystemExit("invalid bounded search values")
    p = 1 / (1 + math.sqrt(2**args.m))
    scores = []
    start = time.monotonic()
    for seed in range(args.seeds):
        v, f, fk, vals = proposal(args.n, p, seed, args.steps)
        scores.append((f, seed, v, fk, vals))
        print(
            f"n={args.n} m={args.m} seed={seed} "
            f"F={f:.12f} F_k={[round(float(x), 8) for x in fk]} "
            f"start={vals[0]:.8f} t={time.monotonic() - start:.2f}s",
            flush=True,
        )
    best, seed, v, fk, vals = max(scores, key=lambda o: o[0])
    rec = {
        "schema": "oneq.r178.symmetric_code_candidate/1",
        "evidence": "NUMERICAL / EFFECTIVE FLAGGED CHANNEL ONLY",
        "n_uses": args.n,
        "shield_m": args.m,
        "p": p,
        "seed": seed,
        "restart_count": args.seeds,
        "optimization_steps": args.steps,
        "F_e_candidate_float": best,
        "F_e_by_k_flagged": [float(x) for x in fk],
        "control_closed_n_code_float": objective(
            closed_v(args.n), args.n, p, iters=500
        )[1],
        "candidate_isometry_trace_error": float(
            np.max(np.abs(v.conj().T @ v - np.eye(2)))
        ),
        "margin_above_three_quarters": best - 0.75,
        "numerical_optimization_monotone_witness": bool(
            all(x <= y + 2e-9 for x, y in zip(vals, vals[1:]))
        ),
        "not_a_rigorous_certificate": True,
        "not_a_physical_Ibm_run": True,
    }
    if args.output_dir:
        folder = Path(args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        npy = folder / f"candidate_n{args.n}_m{args.m}.npy"
        np.save(npy, v, allow_pickle=False)
        rec["isometry_numpy_sha256"] = hashlib.sha256(npy.read_bytes()).hexdigest()
        path = folder / f"candidate_n{args.n}_m{args.m}.json"
        path.write_bytes((json.dumps(rec, indent=2) + "\n").encode())
        print("saved", path)
    print("BEST", json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    main()
