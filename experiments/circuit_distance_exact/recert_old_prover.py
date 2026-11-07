r"""Why some re-certified proofs differ from their records: regenerate the RECORDED proof with the prover of its day.

    python recert_old_prover.py --commit 72d03874 --cells morph72 [--work DIR] [--threads 8]
    python recert_old_prover.py --commit 72d03874 --cells 3c:Z:10 morph144:Z:1 ...

THE PROVER SOURCE, and where it comes from (external audit 4, R8, 2026-11-07). The commit names a revision of the
development repository, whose history is not public; the public repository's history is squashed, so a public clone
has no such commit. The source is therefore taken, in this order: --source FILE if given; else `git show
<commit>:experiments/circuit_distance_exact/cdist_prove.c` when this tree's git knows the commit; else the shipped
copy old_provers/cdist_prove_<commit>.c. Whichever route, its sha256 must equal a value records/recert_old_prover.jsonl
pins for that commit (prover_source_sha256_git; the shipped copy is held to it with CRLF read as LF), and the shipped
copy is refused when no row pins one. The route taken is printed and recorded in each new row (prover_source_route).
Neither source found is a refusal that says so. --work defaults to a fresh temporary directory.

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
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recert_campaign as rc  # noqa: E402

OUT = ROOT / "evidence" / "circuit_distance_exact" / "records" / "recert_old_prover.jsonl"
SHIPPED = HERE / "old_provers"
GIT_PATH = "experiments/circuit_distance_exact/cdist_prove.c"


def pinned(commit: str, records: pathlib.Path = OUT) -> set[str]:
    """The prover-source sha256 values the records pin for `commit` (prover_source_sha256_git)."""
    if not records.is_file():
        return set()
    rows = (json.loads(x) for x in records.read_text(encoding="utf-8").splitlines() if x.strip())
    return {r["prover_source_sha256_git"] for r in rows if r.get("commit") == commit and r.get("prover_source_sha256_git")}


def prover_source(commit: str, source: str | None = None, root: pathlib.Path = ROOT,
                  records: pathlib.Path = OUT, shipped: pathlib.Path = SHIPPED) -> tuple[bytes, str]:
    """(source bytes, route) of the prover of `commit`: --source, else `git show`, else the shipped copy; each held
    to the sha256 the records pin for the commit. Refuses (SystemExit) when no route yields bytes it can vouch for."""
    pins = pinned(commit, records)

    def vouched(src: bytes, route: str) -> tuple[bytes, str]:
        h = hashlib.sha256(src).hexdigest()
        if pins and h not in pins:
            raise SystemExit(f"REFUSED: the prover source from {route} has sha256 {h}, not the one "
                             f"{records.name} pins for commit {commit} ({', '.join(sorted(pins))})")
        return src, route

    if source:
        return vouched(pathlib.Path(source).read_bytes(), f"--source {source}")
    try:
        src = subprocess.run(["git", "show", f"{commit}:{GIT_PATH}"], cwd=root, capture_output=True, check=True,
                             stdin=subprocess.DEVNULL).stdout
        return vouched(src, f"git show {commit}")
    except (OSError, subprocess.CalledProcessError):
        pass                                       # no git, no repository here, or a history without the commit
    copy = shipped / f"cdist_prove_{commit}.c"
    if not copy.is_file():
        raise SystemExit(f"REFUSED: the prover of commit {commit} is neither in this tree's git history nor shipped "
                         f"as {copy}; give it with --source")
    if not pins:
        raise SystemExit(f"REFUSED: {copy} is shipped, but {records.name} pins no source sha256 for commit {commit}, "
                         f"so nothing vouches that it is that commit's prover")
    return vouched(copy.read_bytes().replace(b"\r\n", b"\n"), f"shipped {copy.relative_to(root).as_posix()}"
                   if copy.is_relative_to(root) else f"shipped {copy}")
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
    ap.add_argument("--work", default=None, help="scratch directory (default: a fresh temporary one)")
    ap.add_argument("--threads", type=int, default=8)
    # a tree without the development history (the public release) reads the commit's prover from the shipped copy,
    # old_provers/cdist_prove_<commit>.c, whose sha256 every row records as prover_source_sha256_git (prover_source)
    ap.add_argument("--source", default=None, help="the prover source of --commit as a file, instead of git show")
    # one cell, before its campaign row exists: the model the campaign has just exported (its sha256 must still be the
    # record's), so a reproduction can run beside the re-certification instead of after it (2026-11-03, T = 2 cells)
    ap.add_argument("--model", default=None, help="with ONE family:basis:rounds cell: use this exported model")
    a = ap.parse_args()
    work = pathlib.Path(a.work or tempfile.mkdtemp(prefix="recert_old_prover_")) / a.commit
    work.mkdir(parents=True, exist_ok=True)
    src, route = prover_source(a.commit, a.source)
    print(f"prover source of {a.commit}: {route} (sha256 {hashlib.sha256(src).hexdigest()}); work {work}", flush=True)
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
               "prover_source_sha256_git": hashlib.sha256(src).hexdigest(), "prover_source_route": route,
               "prover_exit": p.returncode,
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
