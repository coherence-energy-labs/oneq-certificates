r"""Source snapshots of the revisions that produced the papers' historical results.

External audit 2026-09-14, R-02: the public release lets a reader replay every
receipt, but re-running the original sampling, decoding and production needs the
code as it was at the frozen and producing revisions, which live only in the
private development history. This publishes exactly that code and nothing more:

  for each revision, the producing scripts named below and every module they
  import, followed recursively -- experiment-local imports (`from google_corpus_gate
  import ...`), imports resolved on the directories the repository's scripts put on
  sys.path, and `oneq` imports, absolute and relative -- read from the git objects
  of THAT revision, never from the working tree.

Each revision becomes one deterministic tar (members at their repository paths
under oneq-<commit>/, mtime 0, uid/gid 0, mode 0644, sorted). The manifest,
evidence/revision_snapshots.json, is committed and shipped in the bundle: it names
every file with its sha256, the full commit id, the archive's sha256, and what the
revision produced, so git pins the published archives.

    python tools/revision_snapshots.py --out data/revision_snapshots
    python tools/revision_snapshots.py --out data/revision_snapshots --verify
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "revision_snapshots.json"

REVISIONS = {
    "aba2aec": {"produced": "paper 2: the frozen held-out campaign (seeds derive from this commit)",
                "entry": ["experiments/m0_prior_art/qldpc_heldout_gate.py",
                          "experiments/m0_prior_art/qldpc_bb_refsched_gate.py"]},
    "d4a81ee": {"produced": "paper 1: the original 10^6-shot simulation (million_shot_gate_d5_*)",
                "entry": ["experiments/m0_prior_art/million_shot_gate.py"]},
    "cce1e91": {"produced": "paper 1: the gap distribution",
                "entry": ["experiments/m0_prior_art/gap_distribution.py"]},
    "06344ae": {"produced": "paper 1: Willow generation 4",
                "entry": ["experiments/m0_prior_art/google_corpus_gate.py"]},
    "f57e87d": {"produced": "paper 1: Willow generation 6 (distance 7, 30 rounds)",
                "entry": ["experiments/m0_prior_art/google_corpus_gate.py",
                          "experiments/m0_prior_art/corpus_chunk_runner.py",
                          "tools/corpus_chunk_driver.py"]},
}


class SnapshotError(Exception):
    pass


def _git(*args: str) -> bytes:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    if r.returncode != 0:
        raise SnapshotError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def _tree(commit: str) -> set[str]:
    return set(_git("ls-tree", "-r", "--name-only", commit).decode().splitlines())


def _imports(path: str, source: bytes, tree: set[str]) -> set[str]:
    """Repository files this module imports, resolved against the revision's own tree."""
    try:
        mod = ast.parse(source)
    except SyntaxError as exc:
        raise SnapshotError(f"{path} does not parse at this revision: {exc}") from exc
    here = pathlib.PurePosixPath(path).parent
    found: set[str] = set()

    def module_file(dotted: str) -> list[str]:
        parts = dotted.split(".")
        cands = []
        if parts[0] == "oneq":
            base = "src/" + "/".join(parts)
            cands += [base + ".py", base + "/__init__.py"]
        else:
            # beside the importing file, and in the directories the repository's scripts put on
            # sys.path themselves (tools import experiment modules and vice versa)
            for root_dir in (here.as_posix(), "experiments/m0_prior_art", "tools"):
                base = f"{root_dir}/{'/'.join(parts)}"
                cands += [base + ".py", base + "/__init__.py"]
        return [c for c in cands if c in tree][:1]

    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.update(module_file(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:                          # relative import inside a package
                pkg = here
                for _ in range(node.level - 1):
                    pkg = pkg.parent
                target = pkg / node.module.replace(".", "/") if node.module else pkg
                names = [target.as_posix() + ".py", target.as_posix() + "/__init__.py"]
                names += [(target / a.name).as_posix() + ".py" for a in node.names]
                found.update(n for n in names if n in tree)
            elif node.module:
                found.update(module_file(node.module))
                found.update(f for a in node.names for f in module_file(f"{node.module}.{a.name}"))
    return found


def closure(commit: str, entry: list[str]) -> list[str]:
    tree = _tree(commit)
    missing = [e for e in entry if e not in tree]
    if missing:
        raise SnapshotError(f"{commit}: entry scripts absent at this revision: {missing}")
    todo, seen = list(entry), set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        deps = set(_imports(p, _git("show", f"{commit}:{p}"), tree))
        if p.startswith("src/oneq/") and "src/oneq/__init__.py" in tree:
            deps.add("src/oneq/__init__.py")        # importing a submodule runs the package __init__
        todo.extend(d for d in deps if d not in seen)
    return sorted(seen)


def build(out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "oneq-revision-snapshots/1",
                "what": ("source of the revisions that produced the papers' historical results: the producing "
                         "scripts and every module they import, read from each revision's git objects"),
                "revisions": {}}
    for short, spec in REVISIONS.items():
        full = _git("rev-parse", f"{short}^{{commit}}").decode().strip()
        files = closure(full, spec["entry"])
        buf = io.BytesIO()
        entries = []
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for f in files:
                data = _git("show", f"{full}:{f}")
                info = tarfile.TarInfo(f"oneq-{short}/{f}")
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
                # Each entry names the commit and the blob it binds. These are bytes of a past
                # revision, and a pin that does not say so is audited as a claim about HEAD:
                # tools/hash_pin_audit.py reads `commit`/`blob` on the pin and checks it there.
                entries.append({"path": f, "commit": full,
                                "blob": _git("rev-parse", f"{full}:{f}").decode().strip(),
                                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        blob = buf.getvalue()
        name = f"oneq-source-{short}.tar"
        (out / name).write_bytes(blob)
        manifest["revisions"][short] = {"commit": full, "produced": spec["produced"], "entry": spec["entry"],
                                        "archive": name, "archive_sha256": hashlib.sha256(blob).hexdigest(),
                                        "files": entries}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify(out: pathlib.Path, *, against_git: bool = True) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for short, rev in manifest["revisions"].items():
        blob = (out / rev["archive"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != rev["archive_sha256"]:
            raise SnapshotError(f"{short}: archive hash differs from the manifest")
        want = {e["path"]: e["sha256"] for e in rev["files"]}
        got = {}
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for m in tar.getmembers():
                prefix, _, rel = m.name.partition("/")
                if prefix != f"oneq-{short}" or not m.isfile() or ".." in pathlib.PurePosixPath(rel).parts:
                    raise SnapshotError(f"{short}: unexpected archive member {m.name!r}")
                got[rel] = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
        if got != want:
            raise SnapshotError(f"{short}: archive members differ from the manifest")
        if against_git:
            for p, h in want.items():
                if hashlib.sha256(_git("show", f"{rev['commit']}:{p}")).hexdigest() != h:
                    raise SnapshotError(f"{short}: {p} differs from the revision's git object")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "revision_snapshots"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-git", action="store_true",
                    help="verify archives against the manifest only (a public clone has no history)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    try:
        if a.verify:
            verify(out, against_git=not a.no_git)
            print("REVISION SNAPSHOTS VERIFIED against the manifest"
                  + ("" if a.no_git else " and the git objects"))
        else:
            m = build(out)
            for short, rev in m["revisions"].items():
                print(f"{short}  {len(rev['files']):3d} files  {rev['archive_sha256'][:16]}  {rev['produced']}")
    except SnapshotError as exc:
        print(f"SNAPSHOT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
