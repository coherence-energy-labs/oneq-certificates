r"""Where the frozen producer and the development rerun's producer differ, shot by shot.

    python experiments/m0_prior_art/producer_identity_diff.py [--workers 3]

producer_identity_willow.py re-ran every development and held-out chunk of paper 1's held-out placements through the
unmodified chunk runner: every held-out chunk was reproduced byte for byte, and all but a few development chunks were.
This re-runs exactly the chunks it found different, keeps their receipts, and compares each shot with the receipt the
2026-09-14 rerun committed (data/epsilon_receipts/<config>/chunk_*.jsonl.gz, pinned by the committed chunk index):
status, slack and lower bound, the correction and the packing. It also writes, for every development configuration,
the status histogram under TODAY's producer -- the committed index's histogram for a byte-identical chunk, this run's
for a differing one -- so the development rates of the same producer as the held-out sample can be stated exactly.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
IDENT = ROOT / "evidence" / "producer_identity_willow.json"
OUT = ROOT / "evidence" / "producer_identity_willow_diff.json"


def _worker_init(scratch: str) -> None:
    import epsilon_rerun as er
    er.RECEIPT_DIR = pathlib.Path(scratch)


def _job(job):
    import epsilon_rerun as er
    return job, er.run_chunk(job)


def _shots(path: pathlib.Path) -> dict:
    return {r["shot"]: r for r in map(json.loads, gzip.decompress(path.read_bytes()).decode().splitlines()) if r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import epsilon_rerun as er
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    for p, h in ident["head"]["sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != h:
            raise SystemExit(f"{p} changed since the identity run; rerun producer_identity_willow.py first")
    jobs = [(name, seed) for name, c in ident["configs"].items() for seed in c["differ"]]
    out = {"schema": "oneq-producer-identity-diff/1", "identity_head": ident["head"]["commit"], "chunks": {},
           "development_status_today": {}}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(scratch,)) as ex:
            res = dict(ex.map(_job, jobs, chunksize=1))
        for (name, seed), r in sorted(res.items()):
            index = {x["seed"]: x for x in map(json.loads, (er.INDEX_DIR / name / "chunks.jsonl")
                                               .read_text(encoding="utf-8").splitlines())}
            old_p = ROOT / index[seed]["receipts_file"]
            if hashlib.sha256(gzip.decompress(old_p.read_bytes())).hexdigest() != index[seed]["receipts_sha256"]:
                raise SystemExit(f"{old_p}: the committed receipt file is not the bytes its index pins")
            old, new = _shots(old_p), _shots(pathlib.Path(scratch) / name / f"chunk_{seed:06d}.jsonl.gz")
            if set(old) != set(new):
                raise SystemExit(f"{name} chunk {seed}: the two runs certified different shots")
            diffs = []
            for s in sorted(old):
                o, n = old[s], new[s]
                fields = [k for k in ("status", "epsilon", "lower_bound", "correction", "duals") if o[k] != n[k]]
                if fields:
                    diffs.append({"shot": s, "fields": fields, "status_then": o["status"], "status_now": n["status"],
                                  "epsilon_then": o["epsilon"], "epsilon_now": n["epsilon"]})
            out["chunks"][f"{name}:{seed}"] = {"shots_differing": len(diffs), "diffs": diffs,
                                                "status_now": r["status"], "status_then": index[seed]["status"]}
            print(f"{name} chunk {seed}: {len(diffs)} shots differ "
                  f"({collections.Counter((d['status_then'], d['status_now']) for d in diffs)})", flush=True)
    for name, c in ident["configs"].items():
        if c["sample"] != "development":
            continue
        tot = collections.Counter()
        for x in map(json.loads, (er.INDEX_DIR / name / "chunks.jsonl").read_text(encoding="utf-8").splitlines()):
            k = f"{name}:{x['seed']}"
            tot.update(out["chunks"][k]["status_now"] if k in out["chunks"] else x["status"])
        out["development_status_today"][name] = dict(tot)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
