"""R178 (a): a three-group see-saw at n = 14 -- does any three-group code cross 3/4 at n = 14 (is 15 the floor)?

R140 left: 7+7 two-group codes reach 0.7497607 after 400 exact steps; "a three-group (5,5,4) see-saw at n = 14 grown
from the 7+7 code (about 5 h, under 1 GB)" decides whether 15 is the floor in that richer family.

TWO FACTS this lane had to face first (both checked here):
  1. The 7+7 encoder R140 reached is NOT on disk (twogroup_n14_exact.log records only its trajectory; the run saved
     twogroup_X_a7_b7_exact.npy into the working directory, which was not committed). The committed n = 15 witness
     (twogroup_witness_a7_b8.npy, sha256 83b00e93...) is the closest committed code. Two exact maps take it to n = 14:
       drop    trace out one use of the 8-group: Tr_1 maps operators on Sym^8 into operators on Sym^7 EXACTLY, so
               the result is a valid (mixed) 7+7 encoder -- Alice simply never sends that use;
       resample  R140's Dicke-profile resampling 8 -> 7 (twogroup.resample_twogroup).
     The better of the two seeds a 7+7 exact see-saw (twogroup's encoder_exact + power decoders) for a few steps.
  2. The (5,5,4) family does NOT contain the 7+7 family (Sym^7 (x) Sym^7 is not inside Sym^5 (x) Sym^5 (x) Sym^4 for
     any assignment of the 14 uses), so "grown from the 7+7 code" can only be a projection, which loses fidelity.
     The (7,4,3) family DOES contain it exactly (Sym^7 sits inside Sym^4 (x) Sym^3), so a (7,4,3) see-saw started
     there is monotone from the 7+7 value: every gain it records is a three-group gain over the two-group code.
     Both are run (budget permitting); (5,5,4) is the family R140 named.

Embedding: every family is a set of qubit groups of the 14 uses; a code is moved by the overlap of the product-Dicke
bases, computed on the full 2^14 space (O = E_tgt^T E_src); exact (an isometry) when the source family is contained.

Checkpoints after every step: kgroup_X_<tag>.npy and the trajectory JSON. Resume:
  python r178_n14_threegroup.py run FAMILY --resume [--minutes M]
  python r178_n14_threegroup.py run 5,5,4 --minutes 20
  python r178_n14_threegroup.py run 7,4,3 --minutes 20
  python r178_n14_threegroup.py seed --minutes 8          (builds the 7+7 seed; first)
Evidence: evidence/superactivation/r178_n14/ (trajectories, seed record, encoders).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from oneq import superactivation as sa  # noqa: E402

import kgroup as kg  # noqa: E402
import twogroup as tg  # noqa: E402

EV = ROOT / "evidence" / "superactivation"
OUTD = EV / "r178_n14"
WITNESS15 = EV / "general_codes" / "twogroup_witness_a7_b8.npy"
R140_77 = 0.7497606777          # twogroup_n14_exact.log, step 400
N = 14


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ pure pieces
def drop_one_use(X: np.ndarray, a: int, b: int) -> np.ndarray:
    """encoder on R (x) Sym^a (x) Sym^b -> R (x) Sym^a (x) Sym^(b-1) by tracing out ONE qubit of the b-group.
    Tr_1 |D^b_w><D^b_w'| = sqrt(C(b-1,w)C(b-1,w')/(C(b,w)C(b,w'))) |D^{b-1}_w><D^{b-1}_w'| (qubit kept 0)
                          + sqrt(C(b-1,w-1)C(b-1,w'-1)/(C(b,w)C(b,w'))) |D^{b-1}_{w-1}><D^{b-1}_{w'-1}| (qubit 1)."""
    K0 = np.zeros((b, b + 1))
    K1 = np.zeros((b, b + 1))
    for w in range(b + 1):
        if w <= b - 1:
            K0[w, w] = math.sqrt(math.comb(b - 1, w) / math.comb(b, w))
        if w >= 1:
            K1[w - 1, w] = math.sqrt(math.comb(b - 1, w - 1) / math.comb(b, w))
    Ia = np.eye(2 * (a + 1))
    A0, A1 = np.kron(Ia, K0), np.kron(Ia, K1)
    return sa.herm(A0 @ X @ A0.T + A1 @ X @ A1.T)


def product_dicke_basis(groups: list, n: int) -> np.ndarray:
    """(2^n, prod(len+1)) columns |D_{w1}>_{G1} ... |D_{wK}>_{GK}, groups' basis order lexicographic (G1 outermost),
    qubit q is bit (n-1-q) of the computational index"""
    idx = np.arange(1 << n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    cols = []
    shapes = [len(g) + 1 for g in groups]
    for flat in range(math.prod(shapes)):
        ws, r = [], flat
        for s in reversed(shapes):
            ws.append(r % s)
            r //= s
        ws = ws[::-1]
        ok = np.ones(1 << n, bool)
        norm = 1.0
        for g, w in zip(groups, ws, strict=True):
            ok &= bits[:, g].sum(axis=1) == w
            norm *= math.comb(len(g), w)
        cols.append(ok / math.sqrt(norm))
    return np.stack(cols, axis=1)


def overlap(src_groups: list, tgt_groups: list, n: int) -> np.ndarray:
    return product_dicke_basis(tgt_groups, n).T @ product_dicke_basis(src_groups, n)


def move(X: np.ndarray, O: np.ndarray) -> np.ndarray:
    """X on R (x) src -> R (x) tgt by the overlap O, renormalised so Tr_tgt X = 1_R"""
    big = np.kron(np.eye(2), O)
    Y = sa.herm(big @ X @ big.T)
    m = O.shape[0]
    T = Y.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m))
    return sa.herm(Ti @ Y @ Ti)


def groups_of(ms) -> list:
    out, s = [], 0
    for m in ms:
        out.append(list(range(s, s + m)))
        s += m
    return out


# ------------------------------------------------------------------------------------------------ runs
def _save(tag, X, traj, extra):
    OUTD.mkdir(parents=True, exist_ok=True)
    np.save(OUTD / f"X_{tag}.npy", X)
    rec = {"tag": tag, "trajectory": traj, **extra}
    (OUTD / f"trajectory_{tag}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")


def run_seed(minutes: float) -> None:
    X15 = np.load(WITNESS15)
    t0 = time.time()
    _, F15 = tg.decode(X15, 7, 8, iters=150)
    Xd = drop_one_use(X15, 7, 8)
    Xr = tg.resample_twogroup(X15, 7, 8, 7, 7)
    decd, Fd = tg.decode(Xd, 7, 7, iters=200)
    decr, Fr = tg.decode(Xr, 7, 7, iters=200)
    print(f"seed: n=15 witness F={F15:.10f}; drop-one-use F={Fd:.10f}; resample F={Fr:.10f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    X, decs, F = (Xd, decd, Fd) if Fd >= Fr else (Xr, decr, Fr)
    how = "drop" if Fd >= Fr else "resample"
    traj = [{"step": 0, "F": F, "t": time.time() - t0}]
    z = None
    s = 0
    while time.time() - t0 < 60 * minutes:
        X, z, _ = tg.encoder_exact(tg.gradient(decs, 7, 7), 64, z0=z)
        decs, F = tg.decode(X, 7, 7, warm=decs, iters=60)
        s += 1
        traj.append({"step": s, "F": F, "t": time.time() - t0})
        print(f"   7+7 step {s}: F = {F:.10f}  ({time.time() - t0:.0f}s)", flush=True)
        _save("7_7", X, traj, {"family": [7, 7], "seed_from": how, "F15_witness": F15, "F_drop": Fd, "F_resample": Fr,
                               "witness_sha256": sha(WITNESS15)})


def run_family(ms, minutes: float, resume: bool) -> None:
    tag = "_".join(map(str, ms))
    m = kg.dim_sym(ms)
    t0 = time.time()
    if resume and (OUTD / f"X_{tag}.npy").exists():
        X = np.load(OUTD / f"X_{tag}.npy")
        prev = json.loads((OUTD / f"trajectory_{tag}.json").read_text(encoding="utf-8"))
        traj, extra = prev["trajectory"], {k: v for k, v in prev.items() if k not in ("tag", "trajectory")}
        t_off = traj[-1]["t"]
        print(f"resume {tag} at step {traj[-1]['step']} F={traj[-1]['F']:.10f}", flush=True)
    else:
        X77 = np.load(OUTD / "X_7_7.npy")
        O = overlap(groups_of((7, 7)), groups_of(ms), N)
        sv = np.linalg.svd(O, compute_uv=False)
        X = move(X77, O)
        _, F77 = tg.decode(X77, 7, 7, iters=150)
        traj, t_off = [], 0.0
        extra = {"family": list(ms), "seed": "X_7_7.npy moved by the product-Dicke overlap",
                 "seed_F_7_7": F77, "overlap_singular_values_min_max": [float(sv.min()), float(sv.max())],
                 "contains_7_7_exactly": bool(sv.min() > 1 - 1e-12),
                 "seed_sha256": sha(OUTD / "X_7_7.npy")}
        print(f"{tag}: overlap singular values [{sv.min():.6f}, {sv.max():.6f}] "
              f"(exact containment: {extra['contains_7_7_exactly']}); 7+7 seed F={F77:.10f}", flush=True)
    decs, F = kg.decode(X, ms, iters=200 if not traj else 120)
    if not traj:
        traj.append({"step": 0, "F": F, "t": time.time() - t0})
        extra["F_after_move"] = F
        print(f"   {tag} step 0: F = {F:.10f}", flush=True)
    z = None
    s = traj[-1]["step"]
    while time.time() - t0 < 60 * minutes:
        X, z, _ = tg.encoder_exact(kg.gradient(decs, ms), m, z0=z)
        decs, F = kg.decode(X, ms, warm=decs, iters=60)
        s += 1
        traj.append({"step": s, "F": F, "t": t_off + time.time() - t0})
        print(f"   {tag} step {s}: F = {F:.10f}  ({time.time() - t0:.0f}s)", flush=True)
        _save(tag, X, traj, extra)
    _save(tag, X, traj, extra)


def main(argv) -> int:
    if argv and argv[0] == "seed":
        run_seed(float(argv[argv.index("--minutes") + 1]) if "--minutes" in argv else 8.0)
        return 0
    if argv and argv[0] == "run":
        ms = tuple(int(x) for x in argv[1].split(","))
        if sum(ms) != N:
            raise SystemExit("family must sum to 14")
        run_family(ms, float(argv[argv.index("--minutes") + 1]) if "--minutes" in argv else 20.0, "--resume" in argv)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
