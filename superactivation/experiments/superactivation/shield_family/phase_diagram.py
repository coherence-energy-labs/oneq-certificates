"""Onset of superactivation vs shield dimension: for P_d (x) A_{2d}, p = 1/(sqrt d + 1), the smallest n at which an
explicit code on N~_p^n exceeds 3/4. Each entry is an UPPER bound on n*(d) (a code that works); n = 1 never works.

  n <= 4 : dense general codes (all encoders; onset_vs_p.best)
  n >= 5 : permutation-symmetric codes on Sym^n (oneq.superactivation with flip probability p), power see-saw

  python phase_diagram.py D1,D2,...        the ORIGINAL full scan (n = 2..17, stops at the first crossing): one line
                                           per d, and phase_diagram.json next to this file (not the evidence copy)

The n <= 4 rows of the evidence table (evidence/superactivation/shield_family/phase_diagram.json) were first made
with onset_vs_p.best while its encoder gradient was wrong (apply_dual computed K^dag Y K^T; fixed 2026-10-19). The
general mode recomputes ONLY those rows; the symmetric n >= 5 rows are NOT recomputed (a different code path,
oneq.superactivation, untouched by that bug) and are carried forward with their provenance stated:

  python phase_diagram.py general D1,D2,... OUTDIR [--restarts R] [--steps S] [--seeds A,B,..] [--polish P]
        per d and n = 2, 3, 4: onset_vs_p.best(n, p, R, S, seed) for every seed (independent restart batches), each
        seed's winner polished by P more see-saw steps; writes OUTDIR/general_d{d}.json and the best code as a
        2^n x 2 isometry OUTDIR/general_code_d{d}_n{n}.npy (columns |0_L>, |1_L>)
  python phase_diagram.py merge OUTDIR [--json PATH]
        rewrites the evidence table: n <= 4 from OUTDIR (the best polished seed, with the seed spread), n >= 5 kept as
        recorded, every value tagged with its search and evidence class; an onset at n <= 4 is CERTIFIED only when a
        nuse_certificate.py record for that (d, n) exists beside the table and says so. A ONE-SHOT migration: run on
        its own output it would record post-fix values as "recorded_before_fix", so it is not re-run.

The original scan never reached n >= 5 at d = 5 (the row was added by the general mode, n <= 4 only). Two modes fill
such a row with the SAME function and parameters as the carried-forward symmetric rows (paper 4, Table III):

  python phase_diagram.py symmetric D OUTDIR [--nmax N] [--check D1,D2,..]
        first re-runs symmetric_best on every symmetric n >= 5 entry the table carries forward at the d in --check
        (default: every row that has one, d = 2, 3, 4, 6, 9 today), which must reproduce it to REPRODUCE_TOL, and a
        record that skipped a row does not count as reproducing; so that the new values come from the procedure
        that made the old ones; then symmetric_best(n, p) for n = 5, 6, ... at shield dimension D, stopping at the first
        n above 3/4 (or at --nmax, default 17, as the original scan); writes OUTDIR/symmetric_d{D}.json
  python phase_diagram.py add-symmetric D OUTDIR [--json PATH]
        writes that record's n >= 5 values into row D of the table, each entry naming the record and its sha256, and
        the row's onset by the merge rule; refuses when the record did not reproduce the carried-forward rows, or when
        row D already holds n >= 5 values or an onset. Every other row is left byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "general_codes"))
from oneq import superactivation as sa  # noqa: E402

import onset_vs_p as ovp  # noqa: E402
import twogroup as tg  # noqa: E402

EV = ROOT / "evidence" / "superactivation" / "shield_family"
GENERAL_NS = (2, 3, 4)
CODE_FILES = ("experiments/superactivation/shield_family/phase_diagram.py",
              "experiments/superactivation/shield_family/onset_vs_p.py",
              "experiments/superactivation/general_codes/general_codes.py",
              "experiments/superactivation/general_codes/twogroup.py",
              "src/oneq/superactivation.py")


def symmetric_best(n, p, restarts=3, steps=300, seed=13):
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(restarts):
        X = sa.random_encoder(n, rng)
        decs, F, _ = sa.decode_power(X, n, iters=200, p=p)
        for _ in range(steps):
            X = tg.encoder_power(sa.encoder_gradient(decs, n, p), X, n + 1, iters=10)
            decs, F, _ = sa.decode_power(X, n, warm=decs, iters=30, p=p)
        decs, F, _ = sa.decode_power(X, n, warm=decs, iters=1000, p=p)
        best = max(best, F)
    return best


def onset(d, nmax=17):
    p = 1 / (math.sqrt(d) + 1)
    row = {"d": d, "p": p, "F": {}}
    for n in range(2, nmax + 1):
        F = ovp.best(n, p, 4, 200)[0] if n <= 4 else symmetric_best(n, p)
        row["F"][n] = F
        if F > 0.75:
            row["onset_upper_bound"] = n
            row["family"] = "general" if n <= 4 else "symmetric"
            return row
    row["onset_upper_bound"] = None
    return row


# ------------------------------------------------------------------------------------ general mode (n <= 4 only)

def git_blob_id(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def code_version() -> dict:
    """the exact bytes of every file the search ran (git blob ids: a committed copy has the same id) and HEAD"""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30, stdin=subprocess.DEVNULL).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        head = None
    return {"git_head": head, "code_git_blobs": {f: git_blob_id((ROOT / f).read_bytes()) for f in CODE_FILES},
            "python": sys.version.split()[0], "numpy": np.__version__}


def polish(n, p, X, steps, tol=1e-11):
    """continue best()'s see-saw (same two half-steps) from X until F gains < tol over 50 steps"""
    decs, F = ovp.decode(X, n, p, iters=2000)
    hist = [F]
    for _ in range(steps):
        X = tg.encoder_power(ovp.adjoint_states(decs, n, p), X, 2 ** n, iters=10)
        decs, F = ovp.decode(X, n, p, warm=decs, iters=30)
        hist.append(F)
        if len(hist) > 60 and hist[-1] - hist[-51] < tol:
            break
    decs, F = ovp.decode(X, n, p, warm=decs, iters=5000)
    return X, decs, F, len(hist) - 1, hist[-1] - hist[max(0, len(hist) - 51)]


def isometry_of(X, n):
    """the code of a rank-1 encoder Choi X = |w><w| (w = sum_r |r> (x) V|r>, R first): V, and the rank gap"""
    m = 2 ** n
    w, v = np.linalg.eigh(sa.herm(X))
    top = v[:, -1] * math.sqrt(max(w[-1], 0.0))
    V = np.stack([top[:m], top[m:]], axis=1)
    U, _, Vh = np.linalg.svd(V, full_matrices=False)
    return U @ Vh, float(w[-2])


def code_fidelity(V, n, p, iters=20000):
    wv = np.concatenate([V[:, 0], V[:, 1]])
    return ovp.decode(np.outer(wv, wv.conj()), n, p, iters=iters)


def general_search(d, n, restarts, steps, seeds, polish_steps, outdir):
    p = 1 / (math.sqrt(d) + 1)
    per_seed, top = [], None
    for seed in seeds:
        t0 = time.time()
        F0, X = ovp.best(n, p, restarts, steps, seed=seed)
        Xp, _, Fp, used, tail = polish(n, p, X, polish_steps)
        per_seed.append({"seed": seed, "F_best": F0, "F_polished": Fp, "polish_steps": used,
                         "polish_last50_gain": tail, "seconds": round(time.time() - t0, 1)})
        print(f"  d = {d} n = {n} seed {seed}: best {F0:.10f} -> polished {Fp:.10f} ({used} steps, last-50 gain "
              f"{tail:.1e}, {time.time() - t0:.0f}s)", flush=True)
        if top is None or Fp > top[0]:
            top = (Fp, Xp)
    V, gap = isometry_of(top[1], n)
    _, Fcode = code_fidelity(V, n, p)
    path = outdir / f"general_code_d{d}_n{n}.npy"
    np.save(path, V)
    pol = [s["F_polished"] for s in per_seed]
    return {"F": Fcode, "F_choi": top[0], "encoder_second_eigenvalue": gap, "seed_spread": max(pol) - min(pol),
            "seeds": per_seed, "restarts_per_seed": restarts, "steps": steps, "polish_cap": polish_steps,
            "code_file": path.name, "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "exceeds_three_quarters_float": bool(Fcode > 0.75)}


def run_general(ds, outdir, restarts, steps, seeds, polish_steps, ns=GENERAL_NS):
    """per d, the n in `ns`; an existing general_d{d}.json keeps its other n entries (each entry records the code
    version that produced it, so a file updated at another revision says so per entry)"""
    outdir.mkdir(parents=True, exist_ok=True)
    version = code_version()
    for d in ds:
        path = outdir / f"general_d{d}.json"
        res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "d": d, "p": 1 / (math.sqrt(d) + 1), "search": "onset_vs_p.best (dense, every encoder; power see-saw)",
            "code_version": version, "n": {}}
        for n in ns:
            res["n"][str(n)] = dict(general_search(d, n, restarts, steps, seeds, polish_steps, outdir),
                                    code_version=version)
            r = res["n"][str(n)]
            print(f"d = {d} n = {n}: F = {r['F']:.10f} (seed spread {r['seed_spread']:.1e}, encoder 2nd eigenvalue "
                  f"{r['encoder_second_eigenvalue']:.1e})", flush=True)
            (outdir / f"general_d{d}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def merge(outdir, table):
    """the evidence table: n <= 4 recomputed (general mode), n >= 5 symmetric carried forward as recorded"""
    old = json.loads(table.read_text(encoding="utf-8")) if table.exists() else {}
    rows = {}
    ds = sorted({int(k) for k in old} | {int(f.stem[len("general_d"):]) for f in outdir.glob("general_d*.json")})
    for d in ds:
        rec = old.get(str(d), {})
        gpath = outdir / f"general_d{d}.json"
        gen = json.loads(gpath.read_text(encoding="utf-8")) if gpath.exists() else None
        row = {"d": d, "p": 1 / (math.sqrt(d) + 1), "F": {}, "entries": {}}
        for n_str, F in sorted(rec.get("F", {}).items(), key=lambda kv: int(kv[0])):
            n = int(n_str)
            if n >= 5:
                row["F"][n_str] = F
                row["entries"][n_str] = {
                    "search": "phase_diagram.symmetric_best (Sym^n codes, power see-saw, 3 restarts x 300 steps, "
                              "seed 13)", "provenance": "carried forward as recorded at R142 (commit 641b20fd); "
                              "not recomputed: oneq.superactivation is not touched by the apply_dual fix",
                    "class": "MEASURED-float"}
            elif gen is None:
                row["F"][n_str] = F
                row["entries"][n_str] = {"search": "onset_vs_p.best BEFORE the apply_dual fix",
                                         "class": "SUPERSEDED-not-recomputed"}
        if gen is not None:
            for n_str, g in gen["n"].items():
                if str(int(n_str)) in rec.get("F", {}):
                    g = dict(g, recorded_before_fix=rec["F"][n_str])
                row["F"][n_str] = g["F"]
                cert = EV / f"certificate_nuse_d{d}_n{n_str}.json"
                entry = {"search": gen["search"], "class": "MEASURED-float", **{k: g[k] for k in (
                    "seed_spread", "restarts_per_seed", "steps", "polish_cap", "code_file", "code_sha256",
                    "encoder_second_eigenvalue")}, "seeds": [s["seed"] for s in g["seeds"]],
                    "code_version": gen["code_version"]}
                if "recorded_before_fix" in g:
                    entry["recorded_before_fix"] = g["recorded_before_fix"]
                if cert.exists():
                    c = json.loads(cert.read_text(encoding="utf-8"))
                    entry["certificate"] = cert.name
                    entry["certified_lower_bound"] = c["certified_lower_bound_float"]
                    entry["certified_exceeds_three_quarters"] = c["exceeds_three_quarters"]
                    if c["exceeds_three_quarters"]:
                        entry["class"] = "CERTIFIED"
                row["entries"][n_str] = entry
        row["F"] = dict(sorted(row["F"].items(), key=lambda kv: int(kv[0])))
        cross = [int(n) for n, F in row["F"].items() if F > 0.75]
        n_star = min(cross) if cross else None
        row["onset_upper_bound"] = n_star
        if n_star is not None:
            e = row["entries"][str(n_star)]
            row["family"] = "general" if n_star <= 4 else "symmetric"
            row["onset_class"] = e["class"]
            if "certified_lower_bound" in e:
                row["onset_certified_lower_bound"] = e["certified_lower_bound"]
        if rec.get("onset_upper_bound") is not None:
            row["onset_recorded_before_fix"] = rec["onset_upper_bound"]
        rows[str(d)] = row
    table.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    for d in ds:
        r = rows[str(d)]
        print(f"d = {d:3d}: n* <= {r['onset_upper_bound']} ({r.get('onset_class')}; before the fix "
              f"{r.get('onset_recorded_before_fix')})")
    return rows


# ------------------------------------------------------------------------------------ symmetric mode (n >= 5, one d)

#: the search the original scan ran for n >= 5 (onset() above), as the table names it
SYMMETRIC_SEARCH = "phase_diagram.symmetric_best (Sym^n codes, power see-saw, 3 restarts x 300 steps, seed 13)"
#: a re-run must give the carried-forward value to this (absolute): same seed, same code, so only BLAS rounding differs
REPRODUCE_TOL = 1e-9


def carried_symmetric(rows: dict, d: int) -> dict[int, float]:
    """the n >= 5 values of row d that the table carries forward from the original scan"""
    row = rows.get(str(d), {})
    return {int(n): row["F"][n] for n, e in row.get("entries", {}).items()
            if int(n) >= 5 and e.get("search") == SYMMETRIC_SEARCH
            and e.get("provenance", "").startswith("carried forward as recorded")}


def run_symmetric(d: int, outdir: pathlib.Path, table: pathlib.Path, nmax: int = 17, check=None) -> dict:
    """check=None (the default): EVERY row the table carries symmetric entries for, so the reproduction cannot leave
    one out (a hand-typed list (2, 3, 4, 6) missed d = 9's n = 5 entry, R142's old onset there, on the first run)"""
    rows = json.loads(table.read_text(encoding="utf-8"))
    every = tuple(sorted(int(dk) for dk in rows if carried_symmetric(rows, int(dk))))
    check = every if check is None else tuple(check)
    t_all = time.time()
    repro, worst = {}, 0.0
    for dc in check:
        old = carried_symmetric(rows, dc)
        if not old:
            raise SystemExit(f"REFUSING: the table carries no symmetric n >= 5 entry at d = {dc} to reproduce")
        pc = 1 / (math.sqrt(dc) + 1)
        repro[str(dc)] = {}
        for n, F_old in sorted(old.items()):
            t0 = time.time()
            F_now = symmetric_best(n, pc)
            diff = abs(F_now - F_old)
            worst = max(worst, diff)
            repro[str(dc)][str(n)] = {"recorded": F_old, "now": F_now, "abs_difference": diff,
                                      "seconds": round(time.time() - t0, 1)}
            print(f"  reproduce d = {dc} n = {n}: recorded {F_old:.13f} now {F_now:.13f} (|diff| {diff:.1e})",
                  flush=True)
    p = 1 / (math.sqrt(d) + 1)
    F, secs, n_star = {}, {}, None
    for n in range(5, nmax + 1):
        t0 = time.time()
        F[str(n)] = symmetric_best(n, p)
        secs[str(n)] = round(time.time() - t0, 1)
        print(f"d = {d} n = {n}: F = {F[str(n)]:.13f} ({secs[str(n)]} s)", flush=True)
        if F[str(n)] > 0.75:
            n_star = n
            break
    rec = {"schema": "oneq.superactivation.symmetric-onset/1", "d": d, "p": p, "search": SYMMETRIC_SEARCH,
           "parameters": {"restarts": 3, "steps": 300, "seed": 13, "nmin": 5, "nmax": nmax,
                          "stops_at_first_crossing": True},
           "F": F, "seconds": secs, "onset_upper_bound": n_star, "class": "MEASURED-float",
           "reproduction": {"of": "every symmetric n >= 5 entry the table carries forward at d = "
                                  + ", ".join(map(str, check)), "rows": repro, "max_abs_difference": worst,
                            "covers_every_carried_row": set(check) >= set(every),
                            "tolerance": REPRODUCE_TOL,
                            "reproduces": worst <= REPRODUCE_TOL and set(check) >= set(every)},
           "code_version": code_version(), "seconds_total": round(time.time() - t_all, 1)}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"symmetric_d{d}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"d = {d}: n* <= {n_star}; reproduction max |diff| {worst:.1e} "
          f"({'reproduces' if rec['reproduction']['reproduces'] else 'DOES NOT REPRODUCE'})", flush=True)
    return rec


def add_symmetric(d: int, outdir: pathlib.Path, table: pathlib.Path) -> dict:
    path = outdir / f"symmetric_d{d}.json"
    raw = path.read_bytes()
    rec = json.loads(raw)
    if rec.get("schema") != "oneq.superactivation.symmetric-onset/1" or rec.get("d") != d:
        raise SystemExit(f"REFUSING: {path.name} is not a symmetric-onset record for d = {d}")
    if rec["search"] != SYMMETRIC_SEARCH or not rec["reproduction"]["reproduces"]:
        raise SystemExit(f"REFUSING: {path.name} did not reproduce the carried-forward rows with the same search")
    raw_table = table.read_bytes()
    rows = json.loads(raw_table.decode("utf-8"))
    # evidence/ is stored byte for byte (.gitattributes: -text), and merge() wrote the table through Windows text mode,
    # so it ends its lines in CRLF: keep whichever the file has, or every other row would change in its bytes
    nl = "\r\n" if b"\r\n" in raw_table else "\n"
    row = rows[str(d)]
    if any(int(n) >= 5 for n in row["F"]) or row.get("onset_upper_bound") is not None:
        raise SystemExit(f"REFUSING: row d = {d} already holds n >= 5 values or an onset")
    sha = hashlib.sha256(raw).hexdigest()
    for n_str, F in rec["F"].items():
        row["F"][n_str] = F
        row["entries"][n_str] = {
            "search": SYMMETRIC_SEARCH, "provenance": f"computed by phase_diagram.py symmetric, recorded in "
            f"{path.name} with the function and parameters of the rows carried forward from R142, which that run "
            f"reproduces", "class": "MEASURED-float", "record": path.name, "record_sha256": sha}
    row["F"] = dict(sorted(row["F"].items(), key=lambda kv: int(kv[0])))
    cross = [int(n) for n, F in row["F"].items() if F > 0.75]
    row["onset_upper_bound"] = min(cross) if cross else None
    if cross:
        row["family"] = "general" if min(cross) <= 4 else "symmetric"
        row["onset_class"] = row["entries"][str(min(cross))]["class"]
    table.write_bytes((json.dumps(rows, indent=1) + "\n").replace("\n", nl).encode("utf-8"))
    print(f"d = {d}: n* <= {row['onset_upper_bound']} ({row.get('onset_class')}), from {path.name} ({sha[:12]})")
    return row


def main(argv) -> int:
    if argv and argv[0] in ("symmetric", "add-symmetric"):
        ap = argparse.ArgumentParser(prog="phase_diagram.py")
        ap.add_argument("mode")
        ap.add_argument("d", type=int)
        ap.add_argument("outdir", type=pathlib.Path)
        ap.add_argument("--nmax", type=int, default=17)
        ap.add_argument("--check", default=None, help="the rows to reproduce (default: every row with carried "
                                                      "symmetric entries; a narrower list is recorded as not covering)")
        ap.add_argument("--json", type=pathlib.Path, default=EV / "phase_diagram.json")
        a = ap.parse_args(argv)
        if a.mode == "symmetric":
            run_symmetric(a.d, a.outdir, a.json, a.nmax,
                          None if a.check is None else tuple(int(x) for x in a.check.split(",")))
        else:
            add_symmetric(a.d, a.outdir, a.json)
        return 0
    if argv and argv[0] in ("general", "merge"):
        ap = argparse.ArgumentParser(prog="phase_diagram.py")
        ap.add_argument("mode")
        if argv[0] == "general":
            ap.add_argument("ds")
        ap.add_argument("outdir", type=pathlib.Path)
        ap.add_argument("--restarts", type=int, default=6)
        ap.add_argument("--steps", type=int, default=200)
        ap.add_argument("--seeds", default="5,6,7,8")
        ap.add_argument("--polish", type=int, default=2000)
        ap.add_argument("--ns", default=",".join(map(str, GENERAL_NS)), help="the n to (re)compute, a subset of 2,3,4")
        ap.add_argument("--json", type=pathlib.Path, default=EV / "phase_diagram.json")
        a = ap.parse_args(argv)
        if a.mode == "general":
            ns = tuple(int(x) for x in a.ns.split(","))
            if not set(ns) <= set(GENERAL_NS):
                raise SystemExit(f"--ns {a.ns}: general mode covers only n in {GENERAL_NS}")
            run_general([int(x) for x in a.ds.split(",")], a.outdir, a.restarts, a.steps,
                        [int(x) for x in a.seeds.split(",")], a.polish, ns)
        else:
            merge(a.outdir, a.json)
        return 0
    ds = [int(x) for x in argv[0].split(",")]
    out = HERE / "phase_diagram.json"
    rows = json.loads(out.read_text()) if out.exists() else {}
    for d in ds:
        r = onset(d)
        rows[str(d)] = r
        F = r["F"].get(r["onset_upper_bound"], float("nan")) if r["onset_upper_bound"] else float("nan")
        print(f"d = {d:3d} (p = {r['p']:.4f}): onset n* <= {r['onset_upper_bound']} ({r.get('family')}, F = {F:.6f})",
              flush=True)
        out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
