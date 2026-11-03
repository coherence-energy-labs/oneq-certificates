r"""Why some re-certified proofs differ from their records: regenerate the RECORDED proof with the prover of its day.

    python recert_old_prover.py --commit 72d03874 --cells morph72 [--work C:/lq/oldprover] [--threads 8]
    python recert_old_prover.py --commit 72d03874 --cells 3c:Z:10 morph144:Z:1 ...

WHY (paper 3, 2026-11-02). recert_campaign.py re-proved every recorded certificate with the committed prover; most
proofs came back byte-identical, some did not. Those records were written by an earlier prover, before the proof
language advanced to CDP5 (ce5e5b02), which omits trace bits for subtrees the checker decides itself. A record's file
date does not say which prover wrote a row (the three-colour campaign ran across that change, from another checkout),
so the explanation is TESTED instead: the prover source of a named commit (`git show`, never the working tree) is
compiled, run on the model the campaign re-exported -- byte-identical to the one the record names -- with the record's
weight and anchoring, and the resulting proof's sha256 is compared with the record's. Equal means the record is that
prover's proof, bit for bit. Rows go to records/recert_old_prover.jsonl (appended, one per cell and commit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recert_campaign as rc  # noqa: E402

OUT = ROOT / "evidence" / "circuit_distance_exact" / "records" / "recert_old_prover.jsonl"
RUNS = [(pathlib.Path("C:/lq/recert/records.jsonl"), pathlib.Path("C:/lq/recert")),
        (pathlib.Path("C:/lq/recert_z/records.jsonl"), pathlib.Path("C:/lq/recert_z")),
        (pathlib.Path("C:/lq/recert_x/records.jsonl"), pathlib.Path("C:/lq/recert_x"))]


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="family, or family:basis:rounds")
    ap.add_argument("--work", default="C:/lq/oldprover")
    ap.add_argument("--threads", type=int, default=8)
    # a tree without the development history (the public release) reads the commit's prover from the shipped copy,
    # old_provers/cdist_prove_<commit>.c, whose sha256 every row records as prover_source_sha256_git
    ap.add_argument("--source", default=None, help="the prover source of --commit as a file, instead of git show")
    # one cell, before its campaign row exists: the model the campaign has just exported (its sha256 must still be the
    # record's), so a reproduction can run beside the re-certification instead of after it (2026-11-03, T = 2 cells)
    ap.add_argument("--model", default=None, help="with ONE family:basis:rounds cell: use this exported model")
    a = ap.parse_args()
    work = pathlib.Path(a.work) / a.commit
    work.mkdir(parents=True, exist_ok=True)
    if a.source:
        src = pathlib.Path(a.source).read_bytes()
    else:
        src = subprocess.run(["git", "show", f"{a.commit}:experiments/circuit_distance_exact/cdist_prove.c"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    (work / "cdist_prove.c").write_bytes(src)
    exe = work / "cdist_prove.exe"
    gcc = shutil.which("gcc")
    r = subprocess.run([gcc, "-O3", "-fopenmp", "-o", str(exe), str(work / "cdist_prove.c")], capture_output=True)
    if r.returncode:
        raise SystemExit(f"the prover of {a.commit} does not compile: {r.stderr[-400:]!r}")
    done = {}
    for records, wd in RUNS:
        if records.is_file():
            for x in records.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    row = json.loads(x)
                    # a campaign row with no proof and no prover output is a STOPPED attempt, not a re-certified
                    # cell (recert_package.py lists such rows as aborted); it has nothing to explain
                    if row["proof_sha256"] is None and not row["prover"] and not row["checker"]:
                        continue
                    done.setdefault(rc.key(row), (row, wd))
    want = []
    if a.model:
        parts = a.cells[0].split(":") if len(a.cells) == 1 else []
        if len(parts) != 3:
            raise SystemExit("--model needs exactly one family:basis:rounds cell")
        cell = [c for c in rc.cells([parts[0]]) if c["basis"] == parts[1] and c["rounds"] == int(parts[2])]
        if len(cell) != 1:
            raise SystemExit(f"--cells {a.cells[0]}: not exactly one recorded cell")
        c = cell[0]
        row = dict(c, model_file=pathlib.Path(a.model).name, proof_sha256=None,
                   proof_file=f"proof_{c['circuit']}_{c['basis']}{c['rounds']}_w{c['w']}.bin")
        want.append((row, pathlib.Path(a.model).resolve().parents[2]))
    for spec in ([] if a.model else a.cells):
        parts = spec.split(":")
        for k, (row, wd) in sorted(done.items(), key=lambda kv: [str(x) for x in kv[0]]):
            if row["family"] == parts[0] and (len(parts) == 1 or (row["basis"] == parts[1]
                                                                   and row["rounds"] == int(parts[2]))):
                want.append((row, wd))
    if not want:
        raise SystemExit("no re-certified cell matches --cells")
    for row, wd in want:
        model = pathlib.Path(a.model) if a.model else wd / row["family"] / row["kind"] / row["model_file"]
        if sha(model) != row["model_sha256_record"]:
            raise SystemExit(f"{model}: not the model the record names")
        proof = work / f"old_{row['family']}_{row['proof_file']}"
        shift = [str(row["maxround"])] if row["maxround"] >= 0 else []
        t0 = time.time()
        p = subprocess.run([str(exe), str(model), str(row["w"]), str(proof), *shift], capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS=str(a.threads)), stdin=subprocess.DEVNULL)
        got = sha(proof) if proof.is_file() else None
        out = {"family": row["family"], "kind": row["kind"], "circuit": row["circuit"], "basis": row["basis"],
               "rounds": row["rounds"], "w": row["w"], "maxround": row["maxround"], "commit": a.commit,
               "prover_source_sha256_git": hashlib.sha256(src).hexdigest(), "prover_exit": p.returncode,
               "prover": p.stdout.strip().replace(str(work), "<work>"), "prover_s": round(time.time() - t0, 1),
               "proof_sha256": got, "proof_bytes": proof.stat().st_size if proof.is_file() else None,
               "proof_magic": proof.read_bytes()[:4].decode("ascii", "replace") if proof.is_file() else None,
               "recorded_proof_sha256": row["proof_sha256_record"], "reproduces_record": got == row["proof_sha256_record"],
               "committed_prover_proof_sha256": row["proof_sha256"]}
        with open(OUT, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(out) + "\n")
        print(f"{row['family']} {row['basis']}{row['rounds']} w{row['w']} @ {a.commit}: "
              f"{'REPRODUCES the record' if out['reproduces_record'] else 'differs'} "
              f"({out['proof_magic']}, {out['proof_bytes']} bytes, {out['prover_s']} s)", flush=True)
        if proof.is_file() and out["proof_bytes"] > 50_000_000:
            proof.unlink()                       # the hash is the record; a 370 MB scratch file is not kept
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
