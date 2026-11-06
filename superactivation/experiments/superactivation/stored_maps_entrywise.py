r"""Entry-by-entry check of the published intermediate maps M_k = N_k o E (arXiv 2604.27042v2, anc/).

The authors' verifier contracts their STORED intermediate maps (bob_pov_{k}_block_{i}) with their
stored decoders; nothing in it checks that those maps are the channel composed with the encoder.
This lane predicts every stored block from the stored ENCODER alone, through the exact reduction
in oneq.superactivation, and compares entry by entry.

Their block layout (anc/verify_results.py): for k flagged uses (placed first) the S_k x S_{n-k}
Schur-Weyl blocks (j_k, j_nk), R first, then V_{j_k} (x) V_{j_nk}; j runs 1..floor(m/2), then 0
(the symmetric irrep last); j_k is the major index. Prediction:
  - every block with j_nk != 0 is ZERO (an encoder on Sym^n leaves the unflagged uses symmetric);
  - for j_nk = 0, J = k/2 - j_k:  M = sum_{b=j_k}^{k-j_k} |t_b><t_b|_{V_J} (x) rho_{k,b} / C(k,b),
    rho_{k,b} = oneq.superactivation.bob_states(X, n, k)[b].
The basis order inside V_J and inside Sym^{n-k} is the authors' convention, unknown a priori, so all
four orders are evaluated and reported; the check is decided by the best one, and the other three
are kept in the record as the witness that the comparison can fail.

  python stored_maps_entrywise.py --source 2604.27042v2.tar.gz [--out evidence/superactivation/stored_maps_entrywise.json]

The source archive is read in place (no extraction, allow_pickle=False) and must hash to the pinned
SHA-256; the authors' arrays are not redistributed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import pathlib
import sys
import tarfile

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation as sa  # noqa: E402

SOURCE_SHA256 = "a5c80fa27c47f8a7a237defd485ddff1b1cd76ea5b0569716135f7f48ac58e3d"
TOL = 1e-12


def irreps(r: int) -> list[int]:
    return list(range(1, r // 2 + 1)) + [0]


def blocks(n: int, k: int) -> list[tuple[int, int]]:
    if k == 0:
        return [(0, j) for j in irreps(n)]
    if k == n:
        return [(j, 0) for j in irreps(n)]
    return [(jk, jnk) for jk in irreps(k) for jnk in irreps(n - k)]


def predicted(X: np.ndarray, n: int, k: int, jk: int, jnk: int, rho, rev_flag: bool, rev_sym: bool):
    dk, dnk = k - 2 * jk + 1, (n - k) - 2 * jnk + 1
    M = np.zeros((2, dk, dnk, 2, dk, dnk), dtype=complex)
    if jnk == 0:
        for b in range(jk, k - jk + 1):
            t = (k - jk - b) if rev_flag else (b - jk)
            r = rho[b].reshape(2, n - k + 1, 2, n - k + 1) / math.comb(k, b)
            if rev_sym:
                r = r[:, ::-1, :, ::-1]
            M[:, t, :, :, t, :] = r
    return M.reshape(2 * dk * dnk, 2 * dk * dnk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "evidence" / "superactivation" / "stored_maps_entrywise.json")
    a = ap.parse_args()
    raw = a.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"REFUSING: source sha256 {digest} is not the pinned {SOURCE_SHA256}")
    tar = tarfile.open(fileobj=io.BytesIO(raw))

    def load(name: str) -> np.ndarray:
        f = tar.extractfile(f"anc/{name}.npy")
        if f is None:
            raise SystemExit(f"REFUSING: anc/{name}.npy missing from the source")
        return np.load(io.BytesIO(f.read()), allow_pickle=False)

    n = int(load("n"))
    X = load(f"encoder_block_{len(irreps(n)) - 1}")                # j = 0 (symmetric) block, R first
    sa.check_encoder(X, n)
    rhos = {k: sa.bob_states(X, n, k) for k in range(n + 1)}
    stored = {(k, i): load(f"bob_pov_{k}_block_{i}") for k in range(n + 1) for i in range(len(blocks(n, k)))}

    conventions = []
    for rev_flag, rev_sym in itertools.product([False, True], repeat=2):
        worst, at, zero_max, per_k = 0.0, None, 0.0, []
        for k in range(n + 1):
            wk = 0.0
            for i, (jk, jnk) in enumerate(blocks(n, k)):
                S = stored[(k, i)]
                P = predicted(X, n, k, jk, jnk, rhos[k], rev_flag, rev_sym)
                if S.shape != P.shape:
                    raise SystemExit(f"REFUSING: block k={k} i={i} has shape {S.shape}, predicted {P.shape}")
                d = float(np.abs(S - P).max())
                if not np.any(P):
                    zero_max = max(zero_max, d)
                wk = max(wk, d)
                if d > worst:
                    worst, at = d, {"k": k, "block": i, "j_k": jk, "j_nk": jnk}
            per_k.append(wk)
        conventions.append({"flag_order_reversed": rev_flag, "sym_order_reversed": rev_sym,
                            "max_abs_residual": worst, "at": at,
                            "max_abs_on_blocks_predicted_zero": zero_max, "per_k_max_abs_residual": per_k})
    best = min(conventions, key=lambda c: c["max_abs_residual"])
    others = sorted(c["max_abs_residual"] for c in conventions if c is not best)
    res = {
        "source": "arXiv 2604.27042v2 source archive, anc/", "source_sha256": digest, "n": n,
        "blocks_compared": len(stored),
        "nonzero_predicted_blocks": sum(1 for (k, i) in stored if blocks(n, k)[i][1] == 0),
        "tolerance": TOL,
        "verdict": "M_k = N_k o E entry by entry" if best["max_abs_residual"] < TOL else "MISMATCH",
        "best_convention": {k: best[k] for k in ("flag_order_reversed", "sym_order_reversed",
                                                 "max_abs_residual", "at", "max_abs_on_blocks_predicted_zero")},
        "other_conventions_max_abs_residual": others,
        "conventions": conventions,
    }
    a.out.write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(f"{res['verdict']}: {len(stored)} blocks, best residual {best['max_abs_residual']:.3e}, "
          f"zero-blocks max {best['max_abs_on_blocks_predicted_zero']:.3e}, other conventions {others}")
    return 0 if res["verdict"].startswith("M_k") else 1


if __name__ == "__main__":
    sys.exit(main())
