r"""Is the committed prover deterministic? Regenerate re-certified proofs at ANOTHER thread count, on cells where it
matters (proofs with trace bits), and compare byte for byte.

    python recert_determinism.py --threads 4 --cells morph144:Z:1 morph144:X:1 3c:X:1 3c:Z:11

WHY (paper 3, second referee of 2026-11-02, M1). The first determinism check re-ran 14 morph72 cells whose CDP5 proofs
carry ZERO trace bits: a header and task lengths whose bytes no thread schedule can change, so the check could not
fail. This one runs the committed prover (compiled here from the committed source; its binary hash recorded) on the
model the campaign re-exported, with OMP_NUM_THREADS different from the campaign's, and compares the proof's sha256
with the campaign's for the same cell. Rows record the thread count, the trace bits and the binary hash; a cell whose
proof has no trace bits is recorded but marked as unable to test determinism. Appends to
records/recert_determinism.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recert_campaign as rc  # noqa: E402

OUT = ROOT / "evidence" / "circuit_distance_exact" / "records" / "recert_determinism.jsonl"
RUNS = [(pathlib.Path("C:/lq/recert/records.jsonl"), pathlib.Path("C:/lq/recert")),
        (pathlib.Path("C:/lq/recert_z/records.jsonl"), pathlib.Path("C:/lq/recert_z")),
        (pathlib.Path("C:/lq/recert_x/records.jsonl"), pathlib.Path("C:/lq/recert_x")),
        (pathlib.Path("C:/lq/recert_t2z/records.jsonl"), pathlib.Path("C:/lq/recert_t2z")),
        (pathlib.Path("C:/lq/recert_t2x/records.jsonl"), pathlib.Path("C:/lq/recert_t2x"))]
CAMPAIGN_THREADS = 8


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cells", nargs="+", required=True, help="family:basis:rounds")
    ap.add_argument("--work", default="C:/lq/recert_determinism")
    a = ap.parse_args()
    if a.threads == CAMPAIGN_THREADS:
        raise SystemExit("use a thread count other than the campaign's, or the check tests nothing")
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    src = HERE / "cdist_prove.c"
    exe = work / "cdist_prove.exe"
    subprocess.run([shutil.which("gcc"), "-O3", "-fopenmp", "-o", str(exe), str(src)], check=True, capture_output=True)
    done = {}
    for records, wd in RUNS:
        if records.is_file():
            for x in records.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    row = json.loads(x)
                    if row["proof_sha256"] is None and not row["prover"] and not row["checker"]:
                        continue                  # an aborted attempt (recert_package.py), not a certified proof
                    done.setdefault((row["family"], row["basis"], row["rounds"]), (row, wd))
    for spec in a.cells:
        fam, b, n = spec.split(":")
        row, wd = done[(fam, b, int(n))]
        model = wd / row["family"] / row["kind"] / row["model_file"]
        if sha(model) != row["model_sha256"]:
            raise SystemExit(f"{model}: not the model the campaign certified")
        proof = work / f"det_{row['family']}_{row['proof_file']}"
        shift = [str(row["maxround"])] if row["maxround"] >= 0 else []
        t0 = time.time()
        p = subprocess.run([str(exe), str(model), str(row["w"]), str(proof), *shift], capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS=str(a.threads)), stdin=subprocess.DEVNULL)
        bits = re.search(r"\((\d+) trace bits", p.stdout)
        got = sha(proof)
        out = {"family": row["family"], "kind": row["kind"], "circuit": row["circuit"], "basis": row["basis"],
               "rounds": row["rounds"], "w": row["w"], "maxround": row["maxround"], "threads": a.threads,
               "campaign_threads": CAMPAIGN_THREADS, "prover_source_sha256_git": rc.lf_sha(src),
               "binary_sha256": sha(exe), "prover": p.stdout.strip().replace(str(work), "<work>"),
               "prover_s": round(time.time() - t0, 1), "trace_bits": int(bits.group(1)) if bits else None,
               "proof_bytes": proof.stat().st_size, "proof_sha256": got, "campaign_proof_sha256": row["proof_sha256"],
               "identical": got == row["proof_sha256"],
               "can_test_determinism": bool(bits and int(bits.group(1)) > 0)}
        with open(OUT, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(out) + "\n")
        print(f"{fam} {b}{n}: {'IDENTICAL' if out['identical'] else 'DIFFERS'} at {a.threads} threads "
              f"({out['trace_bits']} trace bits, {out['proof_bytes']} bytes, {out['prover_s']} s)", flush=True)
        proof.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
