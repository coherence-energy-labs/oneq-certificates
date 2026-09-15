r"""Pack each completed epsilon rerun's receipts into one deterministic archive.

The receipts are too large for git, so they travel as a release asset. For a
stranger to trust that asset, its identity must be pinned by something that IS
in git. This tool

  1. takes exactly the receipt files the committed chunk index names, in seed
     order, and refuses if any is missing, fails its index hash, or if the
     receipt directory holds a file the index does not name;
  2. writes an uncompressed tar whose bytes depend only on those files (fixed
     member order, mtime 0, uid/gid 0, empty owner names, mode 0644), so packing
     twice gives the same sha256;
  3. records the archive's sha256, size and member count, and the sha256 of the
     chunk index it was packed against, in evidence/epsilon_rerun/<config>/archive.json.

`--verify` re-hashes an archive against archive.json, extracts it to a
temporary directory and re-checks every member against the chunk index.

    python tools/pack_epsilon_receipts.py                    # every completed config
    python tools/pack_epsilon_receipts.py --verify --config sim_d5_r25
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "evidence" / "epsilon_rerun"
RECEIPTS = ROOT / "data" / "epsilon_receipts"
ARCHIVES = ROOT / "data" / "epsilon_receipts_archive"


class PackError(Exception):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _complete(name: str) -> dict | None:
    p = INDEX / name / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    lo, hi = s["spec"]["seeds"]
    return s if s["chunks"] == hi - lo + 1 else None


def _bound_rows(name: str, summ: dict) -> list[dict]:
    """The one index row per planned chunk whose receipt file carries its hash."""
    lo, hi = summ["spec"]["seeds"]
    rows = [json.loads(x) for x in (INDEX / name / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if x]
    by_seed: dict[int, dict] = {}
    for r in rows:
        rp = ROOT / r["receipts_file"]
        if rp.exists() and _sha(gzip.decompress(rp.read_bytes())) == r["receipts_sha256"]:
            if r["seed"] in by_seed:
                raise PackError(f"{name}: chunk {r['seed']} indexed twice")
            by_seed[r["seed"]] = r
    missing = [s for s in range(lo, hi + 1) if s not in by_seed]
    if missing:
        raise PackError(f"{name}: {len(missing)} planned chunks have no valid receipt file (first {missing[:5]})")
    return [by_seed[s] for s in range(lo, hi + 1)]


def pack(name: str) -> dict:
    summ = _complete(name)
    if summ is None:
        raise PackError(f"{name}: no complete merged run to pack")
    rows = _bound_rows(name, summ)
    named = {pathlib.PurePosixPath(r["receipts_file"]).name for r in rows}
    present = {p.name for p in (RECEIPTS / name).iterdir() if p.is_file()}
    stray = sorted(present - named)
    if stray:
        raise PackError(f"{name}: the receipt directory holds files the index does not name: {stray[:5]}")
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVES / f"{name}.tar"
    tmp = dest.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
        for r in rows:
            data = (ROOT / r["receipts_file"]).read_bytes()
            info = tarfile.TarInfo(r["receipts_file"])
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    tmp.replace(dest)
    doc = {"schema": "oneq-epsilon-receipt-archive/1", "config": name,
           "archive": f"{name}.tar", "sha256": h.hexdigest(), "bytes": dest.stat().st_size,
           "members": len(rows),
           "chunks_index_sha256": _sha((INDEX / name / "chunks.jsonl").read_bytes()),
           "extract_to": "the repository root (members are data/epsilon_receipts/<config>/chunk_*.jsonl.gz)",
           "then": f"python tools/replay_epsilon_receipts.py --config {name} --jobs 8"}
    (INDEX / name / "archive.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return doc


def verify(name: str, archive_dir: pathlib.Path | None = None) -> dict:
    """archive_dir defaults to ARCHIVES read at CALL time: a default bound at definition
    time ignored every redirection, including a reviewer's download directory."""
    doc = json.loads((INDEX / name / "archive.json").read_text(encoding="utf-8"))
    tar_path = (archive_dir or ARCHIVES) / doc["archive"]
    h = hashlib.sha256()
    with open(tar_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != doc["sha256"]:
        raise PackError(f"{name}: archive sha256 does not match archive.json")
    if _sha((INDEX / name / "chunks.jsonl").read_bytes()) != doc["chunks_index_sha256"]:
        raise PackError(f"{name}: the chunk index changed since the archive was packed")
    rows = {r["receipts_file"]: r for r in
            (json.loads(x) for x in (INDEX / name / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if x)}
    seen = 0
    with tempfile.TemporaryDirectory() as tmpd, tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            p = pathlib.PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts or not m.isfile() or m.name not in rows:
                raise PackError(f"{name}: unexpected archive member {m.name!r}")
            data = tar.extractfile(m).read()
            if _sha(gzip.decompress(data)) != rows[m.name]["receipts_sha256"]:
                raise PackError(f"{name}: member {m.name} does not match the chunk index")
            seen += 1
        _ = tmpd
    if seen != doc["members"]:
        raise PackError(f"{name}: archive holds {seen} members, archive.json says {doc['members']}")
    return {"config": name, "members": seen, "sha256": doc["sha256"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--archive-dir", default=None,
                    help="where the downloaded archives are (default: data/epsilon_receipts_archive)")
    a = ap.parse_args()
    names = a.config or sorted(p.parent.name for p in INDEX.glob("*/summary.json") if _complete(p.parent.name))
    try:
        for n in names:
            res = verify(n, pathlib.Path(a.archive_dir) if a.archive_dir else None) if a.verify else pack(n)
            print(json.dumps({k: res[k] for k in ("config", "members", "sha256")}), flush=True)
    except PackError as exc:
        print(f"PACK FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
