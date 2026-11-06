r"""PRODUCER lane for finite-blocklength superactivation codes on Sym^n (oneq.superactivation).

See-saw: batched power-iteration decoders (warm-started, a fast PROPOSER) alternating with the
EXACT encoder SDP (one small cvxpy/Clarabel problem: max Tr(X G), X >= 0, Tr_Sym X = I_R). A final
witness gets EXACT decoders from per-branch SDPs, and is then certified by
oneq.superactivation_certificate, which never trusts this lane.

  python seesaw_lane.py seesaw FILE N ITERS OUT          continue from an encoder (.npy)
  python seesaw_lane.py random N SEED RESTARTS ITERS OUT
  python seesaw_lane.py curve FILE N0 N1 ITERS OUTDIR    chain upward, each n seeded from n-1
  python seesaw_lane.py witness FILE N OUT_NPZ OUT_JSON  exact decoders + witness + certificate
  python seesaw_lane.py score OUT_JSON FILE...           exact-decoder fidelity of each encoder (float)
Needs the `superactivation` extra (cvxpy, clarabel) and, for `witness`, the `certify` extra.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation as sa  # noqa: E402


class EncoderSDP:
    def __init__(self, n: int):
        import cvxpy as cp
        self.cp = cp
        m = n + 1
        self.m = m
        self.G = cp.Parameter((2 * m, 2 * m), hermitian=True)
        self.X = cp.Variable((2 * m, 2 * m), hermitian=True)
        cons = [self.X >> 0, cp.partial_trace(self.X, dims=[2, m], axis=1) == np.eye(2)]
        self.prob = cp.Problem(cp.Maximize(cp.real(cp.trace(self.G @ self.X))), cons)

    def solve(self, G: np.ndarray, n: int) -> np.ndarray:
        self.G.value = sa.herm(G)
        self.prob.solve(solver=self.cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                        tol_feas=1e-11, max_iter=400)
        X = sa.herm(self.X.value)
        w, v = np.linalg.eigh(X)
        return sa.normalize_encoder((v * np.clip(w, 0, None)) @ v.conj().T, n)


def exact_decoder(rho: np.ndarray, d: int):
    import cvxpy as cp
    D = cp.Variable((2 * d, 2 * d), hermitian=True)
    cons = [D >> 0, cp.partial_trace(D, dims=[2, d], axis=0) == np.eye(d)]
    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(sa.herm(rho) @ D))), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
    Dv = sa.herm(D.value)
    w, v = np.linalg.eigh(Dv)
    Dv = (v * np.clip(w, 0, None)) @ v.conj().T
    S = sa.ptrace_ref(Dv, d)
    ws, vs = np.linalg.eigh(sa.herm(S))
    Si = np.kron(np.eye(2), (vs / np.sqrt(ws)) @ vs.conj().T)
    return sa.herm(Si @ Dv @ Si)


def seesaw(n: int, X0: np.ndarray, iters: int, log=None, every: int = 50):
    enc = EncoderSDP(n)
    X = X0
    decs, F, _ = sa.decode_power(X, n, iters=200)
    hist = [F]
    for it in range(iters):
        X = enc.solve(sa.encoder_gradient(decs, n), n)
        decs, F, _ = sa.decode_power(X, n, warm=decs, iters=40)
        hist.append(F)
        if log and (it + 1) % every == 0:
            log(f"  n={n} it={it + 1} F={F:.12f}")
        if it > 20 and hist[-1] - hist[-11] < 1e-11:
            break
    return X, F, hist


def exact_fidelity(X: np.ndarray, n: int):
    """The encoder's fidelity under EXACT per-branch decoders (the power iteration plateaus ~2e-7 low)."""
    decs = {}
    for k in range(n + 1):
        d = n - k + 1
        decs[k] = [exact_decoder(r, d) for r in sa.bob_states(X, n, k)]
    return sa.fidelity(X, {k: np.stack(v) for k, v in decs.items()}, n), decs


def encoder_blocklength(X: np.ndarray) -> int:
    if X.ndim != 2 or X.shape[0] != X.shape[1] or X.shape[0] % 2 or X.shape[0] < 4:
        raise ValueError(f"not an encoder on R (x) Sym^n: shape {X.shape}")
    return X.shape[0] // 2 - 1


def make_witness(X: np.ndarray, n: int, out_npz: pathlib.Path, out_json: pathlib.Path, log=print):
    from oneq import superactivation_certificate as sc
    t0 = time.time()
    Ffloat, decs = exact_fidelity(X, n)
    log(f"exact decoders: F = {Ffloat:.15f} ({time.time() - t0:.0f}s)")
    sc.save_witness(out_npz, n, X, decs)
    res = sc.certify(n, X, decs)
    res["float_fidelity"] = Ffloat
    res["witness"] = out_npz.name
    res["witness_sha256"] = __import__("hashlib").sha256(out_npz.read_bytes()).hexdigest()
    out_json.write_bytes((json.dumps(res, indent=2) + "\n").encode("utf-8"))
    log(json.dumps({k: res[k] for k in ("certified_lower_bound_float", "exceeds_three_quarters",
                                        "certified_margin_lower_bound", "seconds")}))
    return res


def main() -> int:
    mode = sys.argv[1]
    log = lambda m: print(m, flush=True)  # noqa: E731
    if mode == "seesaw":
        path, n, iters, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        X, F, hist = seesaw(n, np.load(path), iters, log=log)
        np.save(out, X)
        log(f"n={n}: {hist[0]:.12f} -> {F:.12f} ({len(hist) - 1} steps) -> {out}")
    elif mode == "random":
        n, seed, restarts, iters, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
        rng = np.random.default_rng(seed)
        best = (-1.0, None)
        for r in range(restarts):
            X, F, hist = seesaw(n, sa.random_encoder(n, rng, rank=int(rng.integers(1, 4))), iters)
            log(f"n={n} seed={seed} restart={r} F={F:.12f}")
            if F > best[0]:
                best = (F, X)
        np.save(out, best[1])
    elif mode == "curve":
        path, n0, n1, iters, outdir = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), pathlib.Path(sys.argv[6])
        outdir.mkdir(parents=True, exist_ok=True)
        X, prev = np.load(path), n0
        rows = []
        for n in range(n0 + 1, n1 + 1):
            X, F, hist = seesaw(n, sa.resample_encoder(X, prev, n), iters)
            np.save(outdir / f"encoder_n{n}.npy", X)
            rows.append({"n": n, "F_power_decoders": F, "steps": len(hist) - 1})
            log(f"curve n={n}: {F:.12f}")
            prev = n
        (outdir / "curve.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif mode == "witness":
        path, n, out_npz, out_json = sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])
        make_witness(np.load(path), n, out_npz, out_json)
    elif mode == "score":
        out, rows = pathlib.Path(sys.argv[2]), []
        for path in sys.argv[3:]:
            raw = pathlib.Path(path).read_bytes()
            X = np.load(path)
            n = encoder_blocklength(X)
            t0 = time.time()
            F, _ = exact_fidelity(X, n)
            rows.append({"n": n, "encoder": pathlib.Path(path).name,
                         "encoder_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                         "F_exact_decoders": F, "margin": F - sa.THRESHOLD,
                         "seconds": round(time.time() - t0, 1)})
            log(f"score n={n}: F = {F:.12f} ({rows[-1]['seconds']}s)")
        out.write_bytes((json.dumps(rows, indent=2) + "\n").encode("utf-8"))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
