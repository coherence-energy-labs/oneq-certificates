r"""Paper 3's large re-certified proofs as release assets: record them, verify a download, put the proofs in place,
and run the committed checker on them.

    python recert_assets.py record --archives DIR        # maintainer: write recert_20261102/release_assets.json
    python recert_assets.py verify --archives DIR        # reader: every archive and proof against the hashes
    python recert_assets.py verify --archives DIR --extract   # ... and put the proofs where the manifest's cells are
    python recert_assets.py check [--threads 8] [--family 3c|3c_t2|morph144] [--limit N]

WHY THIS EXISTS (2026-11-06). The thirty proofs of the three-colour cycle (families 3c, 3c_t2) and of the [[144,12,12]]
morphing circuit (morph144), 732 MB raw, are too large for git. Release v2026.11.05 attached them as two archives,
but no tracked file named the archives, and paper 3 still said those proofs "ship as their models and the hashes"
(external audit 3, docs/reviews/EXTERNAL3_2026-11-06.md, Paper C). This tool is the one place that knows the
archives: `record` writes release_assets.json, which the release ships, the paper's generator reads and
paper/tools/build.py accepts as the list of names that are release assets rather than repository files.

WHAT EACH STEP PROVES. `verify` shows that each archive is the recorded bytes and that each member is exactly the proof
a manifest cell names (path, size and sha256 of the regenerated CDP5 proof), with no member missing, extra or out of
place; it extracts only regular files to the paths the record names, never outside the evidence directory, and refuses
to overwrite a different file. Hash agreement shows a file IS the declared proof; it does not decide the proof. Only
`check`, the committed checker run on the proof and its shipped model, does that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tarfile
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EV = ROOT / "evidence" / "circuit_distance_exact"
PKG = EV / "recert_20261102"
MANIFEST = PKG / "manifest.json"
RECORD = PKG / "release_assets.json"
SCHEMA = "oneq-recert-release-assets/1"

#: Which families each archive carries. The archive names are the release's asset names.
ARCHIVES = {"oneq-recert-proofs-3c.tar.gz": ("3c", "3c_t2"), "oneq-recert-proofs-morph144.tar.gz": ("morph144",)}
#: A member's path inside an archive, relative to the evidence directory it is extracted into.
MEMBER = re.compile(r"recert_20261102/proofs/(?P<family>[a-z0-9_]+)/(?P<kind>[a-z_]+)/"
                    r"proof_(?P<circuit>[a-z0-9]+)_(?P<basis>[XZ])(?P<rounds>\d+)_w(?P<w>\d+)\.bin")


class AssetError(Exception):
    """A release asset that is not the recorded bytes, or not the proof its manifest cell names."""


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def member_path(cell: dict) -> str:
    """Where a cell's proof lives, relative to the evidence directory (the archives' member names)."""
    return (f"recert_20261102/proofs/{cell['family']}/{cell['kind']}/"
            f"proof_{cell['circuit']}_{cell['basis']}{cell['rounds']}_w{cell['w']}.bin")


def unshipped(manifest: dict) -> dict[str, dict]:
    """member path -> manifest cell, for every proof the repository does not carry."""
    out = {}
    for c in manifest["cells"]:
        if c["shipped"]:
            continue
        p = member_path(c)
        if p in out:
            raise AssetError(f"two unshipped cells share the proof path {p}")
        out[p] = c
    return out


def read_archive(archive: pathlib.Path) -> list[dict]:
    """Every member of one archive with its size and sha256, refusing anything that is not a plain file."""
    out = []
    with tarfile.open(archive, "r:gz") as tf:
        for ti in tf:
            if ti.isdir():
                continue
            if not ti.isfile():
                raise AssetError(f"{archive.name}: {ti.name} is not a regular file")
            if ti.name.startswith(("/", "\\")) or ".." in pathlib.PurePosixPath(ti.name).parts:
                raise AssetError(f"{archive.name}: {ti.name} would be written outside the evidence directory")
            h = hashlib.sha256()
            f = tf.extractfile(ti)
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
            out.append({"path": ti.name, "bytes": ti.size, "sha256": h.hexdigest()})
    return out


def check_members(name: str, members: list[dict], cells: dict[str, dict]) -> None:
    """The members of archive `name` are exactly the unshipped proofs of its families, each the manifest's bytes."""
    fams = ARCHIVES[name]
    want = {p for p, c in cells.items() if c["family"] in fams}
    seen = [m["path"] for m in members]
    if len(seen) != len(set(seen)):
        raise AssetError(f"{name}: a member appears twice")
    if set(seen) != want:
        raise AssetError(f"{name}: members {sorted(set(seen) - want)} are extra and {sorted(want - set(seen))} "
                         f"are missing")
    for m in members:
        if not MEMBER.fullmatch(m["path"]):
            raise AssetError(f"{name}: {m['path']} is not a proof path")
        c = cells[m["path"]]
        if (m["bytes"], m["sha256"]) != (c["proof_bytes"], c["proof_sha256"]):
            raise AssetError(f"{name}: {m['path']} is not the proof its manifest cell names "
                             f"({m['bytes']} bytes, {m['sha256'][:12]}... against {c['proof_bytes']}, "
                             f"{c['proof_sha256'][:12]}...)")


def lf_sha256(p: pathlib.Path) -> str:
    """sha256 of the bytes git stores (LF), so a Windows checkout and a reader's Linux one agree."""
    return hashlib.sha256(p.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def build_record(archives: dict[str, pathlib.Path], manifest: dict, manifest_sha256: str) -> dict:
    cells = unshipped(manifest)
    if set(archives) != set(ARCHIVES):
        raise AssetError(f"expected the archives {sorted(ARCHIVES)}, got {sorted(archives)}")
    rec = {"schema": SCHEMA,
           "what": "paper 3's re-certified proofs too large for git, attached to the release as these archives",
           "manifest": "evidence/circuit_distance_exact/recert_20261102/manifest.json",
           "manifest_sha256_lf": manifest_sha256,
           "extract_into": "evidence/circuit_distance_exact/",
           "verify": "python experiments/circuit_distance_exact/recert_assets.py verify --archives DIR --extract",
           "check": "python experiments/circuit_distance_exact/recert_assets.py check --threads 8",
           "archives": []}
    covered = set()
    for name in sorted(archives):
        members = read_archive(archives[name])
        check_members(name, members, cells)
        covered |= {m["path"] for m in members}
        rec["archives"].append({"name": name, "bytes": archives[name].stat().st_size,
                                "sha256": sha256_file(archives[name]), "families": list(ARCHIVES[name]),
                                "members": sorted(members, key=lambda m: m["path"])})
    if covered != set(cells):
        raise AssetError(f"no archive carries {sorted(set(cells) - covered)}")
    rec["totals"] = {"archives": len(rec["archives"]), "proofs": len(covered),
                     "proof_bytes": sum(cells[p]["proof_bytes"] for p in covered),
                     "archive_bytes": sum(a["bytes"] for a in rec["archives"])}
    return rec


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify(archive_dir: pathlib.Path, record: dict, manifest: dict, *, manifest_sha256: str | None = None,
           extract_into: pathlib.Path | None = None) -> dict:
    """A reader's downloaded archives against the record and the manifest; optionally put the proofs in place."""
    if record.get("schema") != SCHEMA:
        raise AssetError(f"not a {SCHEMA} record")
    if manifest_sha256 is not None and record.get("manifest_sha256_lf") != manifest_sha256:
        raise AssetError("the record was written against another manifest")
    cells = unshipped(manifest)
    out = {"archives": 0, "proofs": 0, "extracted": 0, "already_in_place": 0}
    if sorted(a["name"] for a in record["archives"]) != sorted(ARCHIVES):
        raise AssetError(f"the record's archives are not {sorted(ARCHIVES)}")
    for a in record["archives"]:
        p = archive_dir / a["name"]
        if not p.is_file():
            raise AssetError(f"{a['name']} is not in {archive_dir}: download it from the release")
        if (p.stat().st_size, sha256_file(p)) != (a["bytes"], a["sha256"]):
            raise AssetError(f"{a['name']} is not the recorded archive (sha256 or size differs)")
        members = read_archive(p)
        if sorted(members, key=lambda m: m["path"]) != a["members"]:
            raise AssetError(f"{a['name']}: its members are not the recorded ones")
        check_members(a["name"], members, cells)
        out["archives"] += 1
        out["proofs"] += len(members)
        if extract_into is not None:
            root = extract_into.resolve()
            with tarfile.open(p, "r:gz") as tf:
                for m in members:
                    dest = (root / m["path"]).resolve()
                    if not dest.is_relative_to(root):
                        raise AssetError(f"{m['path']} would be written outside {root}")
                    if dest.exists():
                        if sha256_file(dest) != m["sha256"]:
                            raise AssetError(f"{dest} exists with other bytes; not overwritten")
                        out["already_in_place"] += 1
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".part")
                    with tf.extractfile(m["path"]) as src, tmp.open("wb") as dst:
                        for chunk in iter(lambda: src.read(1 << 22), b""):
                            dst.write(chunk)
                    if sha256_file(tmp) != m["sha256"]:
                        tmp.unlink()
                        raise AssetError(f"{m['path']}: the extracted bytes are not the recorded proof")
                    tmp.replace(dest)
                    out["extracted"] += 1
    if out["proofs"] != len(cells):
        raise AssetError(f"the record's archives carry {out['proofs']} proofs, the manifest names {len(cells)}")
    return out


def _verdict(text: str) -> str:
    """A checker verdict without its timing, which differs between runs."""
    return re.sub(r";\s*[\d.]+s\)\s*$", ")", text.strip().splitlines()[0] if text.strip() else "")


def check(manifest: dict, *, threads: int, family: str | None, limit: int | None,
          root: pathlib.Path = EV) -> list[dict]:
    """Run the committed checker on every unshipped proof in place under `root` (the evidence directory unless the
    proofs were extracted elsewhere); each must be VERIFIED with the recorded verdict. Models are the shipped ones."""
    sys.path.insert(0, str(HERE))
    import bb_circuits as bc
    import tools_cde as tc
    cells = [(p, c) for p, c in sorted(unshipped(manifest).items()) if family in (None, c["family"])]
    present = [(p, c) for p, c in cells if (root / p).is_file()]
    if not present:
        raise AssetError(f"no unshipped proof is in place under {root}: run `verify --archives DIR --extract` first")
    if limit is not None:
        present = present[:limit]
    with tempfile.TemporaryDirectory() as td:
        exe = tc.build_c(pathlib.Path(td))
        if "ccheck" not in exe:
            raise AssetError("gcc is needed to compile the committed checker (experiments/circuit_distance_exact/ccheck.c)")
        rows = []
        for p, c in present:
            proof, model = root / p, PKG / c["model"]
            if sha256_file(proof) != c["proof_sha256"] or sha256_file(model) != c["model_sha256"]:
                raise AssetError(f"{p} or its model is not the bytes the manifest names")
            t0 = time.monotonic()
            r = tc.run(exe["ccheck"], model, proof, bc.group_kind(c["circuit"]), threads=threads)
            secs = time.monotonic() - t0
            ok = r.returncode == 0 and r.stdout.startswith("VERIFIED") and _verdict(r.stdout) == _verdict(c["checker"])
            rows.append({"proof": p, "verified": ok, "seconds": round(secs, 1), "recorded_seconds": c["checker_s"],
                         "verdict": r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr.strip()[:200]})
            print(f"{'VERIFIED' if ok else 'REFUSED '} {p}  {secs:.1f} s (recorded {c['checker_s']} s)", flush=True)
            if not ok:
                raise AssetError(f"{p}: the committed checker did not reproduce the recorded verdict: {rows[-1]['verdict']}")
        return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--archives", required=True, type=pathlib.Path)
    v = sub.add_parser("verify")
    v.add_argument("--archives", required=True, type=pathlib.Path)
    v.add_argument("--extract", action="store_true")
    v.add_argument("--root", type=pathlib.Path, default=EV,
                   help="where to extract (default: the evidence directory, where the manifest's cells are)")
    c = sub.add_parser("check")
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--family", choices=sorted({f for fams in ARCHIVES.values() for f in fams}))
    c.add_argument("--limit", type=int)
    c.add_argument("--root", type=pathlib.Path, default=EV, help="where the proofs were extracted")
    c.add_argument("--out", type=pathlib.Path, help="write the run's record (JSON) here")
    a = ap.parse_args(argv)
    manifest = load_manifest()
    try:
        if a.cmd == "record":
            rec = build_record({n: a.archives / n for n in ARCHIVES}, manifest, lf_sha256(MANIFEST))
            RECORD.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            shown = RECORD.relative_to(ROOT).as_posix() if RECORD.is_relative_to(ROOT) else str(RECORD)
            print(f"wrote {shown}: {rec['totals']}")
        elif a.cmd == "verify":
            record = json.loads(RECORD.read_text(encoding="utf-8"))
            out = verify(a.archives, record, manifest, manifest_sha256=lf_sha256(MANIFEST),
                         extract_into=a.root if a.extract else None)
            print(f"RELEASE ASSETS VERIFIED: {out['archives']} archives, {out['proofs']} proofs, each the bytes its "
                  f"manifest cell names; extracted {out['extracted']}, already in place {out['already_in_place']}")
        else:
            rows = check(manifest, threads=a.threads, family=a.family, limit=a.limit, root=a.root)
            print(f"CHECKED: {len(rows)} release-asset proofs VERIFIED by the committed checker, "
                  f"{sum(x['seconds'] for x in rows):.0f} s")
            if a.out:
                import platform
                doc = {"schema": "oneq-recert-asset-check/1", "threads": a.threads, "family": a.family,
                       "limit": a.limit, "checker_source_sha256_lf": lf_sha256(HERE / "ccheck.c"),
                       "manifest_sha256_lf": lf_sha256(MANIFEST), "platform": platform.platform(),
                       "python": platform.python_version(), "rows": rows,
                       "verified": sum(r["verified"] for r in rows), "of": len(rows)}
                a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
                print(f"-> {a.out}")
    except AssetError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
