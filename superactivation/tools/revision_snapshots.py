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

CITED REVISIONS (2026-11-07). The papers also NAME development commits for what they
changed or ran -- the first JavaScript checker, the dual-repair commits written into both
languages, the held-out distance-7 producer and driver, the gap run, paper 3's earlier
prover -- and a reader could open none of them (external audit 4, X3/R1). Rather than
publish the private history, each is published here as exactly the files its citation is
about: the files the run's record pins, or the checker files the sentence describes. Where
the citation is about a CHANGE, the archive also carries CHANGE.patch, that commit's diff of
those files against its first parent (myers, full blob ids, three lines of context), so the
reader sees what the commit did; the patch's post-image blob ids are checked against the
files it ships with. Like every snapshot, these are pinned by sha256 in the manifest, not
proven against the commit id (that needs the history).

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
    # ---- CITED revisions: the files a citation is about, no import closure (see the module docstring)
    "7a5a3cf": {"cited": "paper 1, appendix, 'What of the first JavaScript checker is independent': the first JavaScript checker, written from the specification",
                "files": ["verifier-js/verify-certificate.mjs"], "patch": True},
    "888b745": {"cited": "paper 1, appendix, 'What of the first JavaScript checker is independent': the exact lattice rule ported to the JavaScript checker from the "
                         "Python checker (both shipped as they stood at this commit)",
                "files": ["src/oneq/matching_cert.py", "verifier-js/verify-certificate.mjs"], "patch": True},
    "6b3a146": {"cited": "paper 1, appendix, 'What of the first JavaScript checker is independent': the dual repair, written into both languages in one commit",
                "files": ["src/oneq/matching_cert.py", "verifier-js/verify-certificate.mjs"], "patch": True},
    "e0767dc": {"cited": "paper 1, appendix, 'What of the first JavaScript checker is independent': the spread-absorb variant, in both languages in one commit",
                "files": ["src/oneq/matching_cert.py", "verifier-js/verify-certificate.mjs"], "patch": True},
    "900d647": {"cited": "paper 1, appendix, 'What of the first JavaScript checker is independent': the four-order variant, in both languages in one commit",
                "files": ["src/oneq/matching_cert.py", "verifier-js/verify-certificate.mjs"], "patch": True},
    "56dce45": {"cited": "paper 1, appendix, 'What of the first JavaScript checker is independent': boundary-first edge keys canonicalised, in both languages",
                "files": ["src/oneq/matching_cert.py", "src/oneq/matching_cert_strict.py",
                          "verifier-js/verify-certificate.mjs"], "patch": True},
    "617da3a": {"cited": "paper 1, appendix, 'A defect in the exact checkers, found and fixed': the exact and the stricter checker fixed against the three "
                         "packing forgeries",
                "files": ["src/oneq/matching_cert.py", "src/oneq/matching_cert_strict.py"], "patch": True},
    "2008c81": {"cited": "paper 1, appendix, 'The held-out distance-7 stop': the held-out distance-7 run's producer, the eleven files its record "
                         "pins (evidence/epsilon_rerun/heldout_d7_partial.json, head.producer)",
                "files": ["experiments/m0_prior_art/epsilon_rerun.py",
                          "experiments/m0_prior_art/google_corpus_gate.py",
                          "experiments/m0_prior_art/heldout_willow_d7.py", "src/oneq/certifying_decoder.py",
                          "src/oneq/exact_lp.py", "src/oneq/forgeries.py", "src/oneq/matching_cert.py",
                          "src/oneq/matching_cert_epsilon.py", "src/oneq/moat_dual.py", "src/oneq/moat_growth.py",
                          "src/oneq/repair_impossible.py"]},
    "4030f88": {"cited": "paper 1, appendix, 'The held-out distance-7 stop': the deviation's driver and watchdog the held-out run restarted under "
                         "(the record's head.deviation)",
                "files": ["experiments/m0_prior_art/heldout_d7_watchdog.py",
                          "experiments/m0_prior_art/heldout_willow_d7_interleaved.py"]},
    "70b3c4d": {"cited": "paper 1, appendix, 'The gap experiment's scorer': the preregistered gap run's judged files "
                         "(evidence/gap_distribution_v4/prereg_2026-11-03/summary.json, head)",
                "files": ["experiments/m0_prior_art/gap_distribution.py", "src/oneq/certifying_decoder.py",
                          "src/oneq/matching_cert.py", "src/oneq/matching_cert_epsilon.py"]},
    "72d0387": {"cited": "paper 3: the earlier prover that reproduced the original records "
                         "(also shipped as experiments/circuit_distance_exact/old_provers/)",
                "files": ["experiments/circuit_distance_exact/cdist_prove.c"]},
}

#: The archive member that carries a cited revision's diff. Not a repository path, so it cannot collide with one.
PATCH_MEMBER = "CHANGE.patch"
#: Fixed so that anyone holding the history regenerates the same bytes whatever their git configuration says.
_DIFF_ARGS = ("-c", "core.quotepath=false", "diff", "--no-color", "--no-ext-diff", "--no-renames", "--full-index",
              "--diff-algorithm=myers", "-U3")


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


def named_files(commit: str, files: list[str]) -> list[str]:
    """A cited revision's files, exactly as named: every one must exist at that revision."""
    tree = _tree(commit)
    missing = [f for f in files if f not in tree]
    if missing:
        raise SnapshotError(f"{commit}: cited files absent at this revision: {missing}")
    if len(set(files)) != len(files):
        raise SnapshotError(f"{commit}: a cited file is named twice")
    return sorted(files)


def change_patch(commit: str, files: list[str]) -> tuple[str, bytes]:
    """(first parent, the commit's diff of `files` against it). Refuses an empty diff: a revision cited for a
    change it did not make to the named files names the wrong files."""
    parents = _git("rev-list", "--parents", "-n", "1", commit).decode().split()[1:]
    if not parents:
        raise SnapshotError(f"{commit}: a root commit has no parent to diff against")
    patch = _git(*_DIFF_ARGS, parents[0], commit, "--", *files)
    if not patch.strip():
        raise SnapshotError(f"{commit}: changed none of the files it is cited for: {files}")
    return parents[0], patch


def patch_post_images(patch: bytes) -> dict[str, str]:
    """{path: post-image blob id} for every file a full-index git patch touches (deletions excluded)."""
    out: dict[str, str] = {}
    path = None
    for line in patch.decode("utf-8", errors="replace").splitlines():
        if line.startswith("diff --git a/"):
            path = line.rsplit(" b/", 1)[1]
        elif line.startswith("index ") and path is not None:
            post = line.split()[1].split("..")[1]
            if set(post) != {"0"}:
                out[path] = post
            path = None
    return out


def build(out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "oneq-revision-snapshots/1",
                "what": ("source of the revisions that produced the papers' historical results: the producing "
                         "scripts and every module they import, read from each revision's git objects; and of "
                         "the revisions the papers cite, the files each citation is about, with the cited "
                         "commit's diff of them where the citation is about a change"),
                "revisions": {}}
    for short, spec in REVISIONS.items():
        full = _git("rev-parse", f"{short}^{{commit}}").decode().strip()
        files = closure(full, spec["entry"]) if "entry" in spec else named_files(full, spec["files"])
        parent, patch = change_patch(full, spec["files"]) if spec.get("patch") else (None, None)
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
            if patch is not None:
                if PATCH_MEMBER in files:
                    raise SnapshotError(f"{short}: a repository file is named {PATCH_MEMBER}")
                info = tarfile.TarInfo(f"oneq-{short}/{PATCH_MEMBER}")
                info.size, info.mtime, info.mode = len(patch), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(patch))
        blob = buf.getvalue()
        name = f"oneq-source-{short}.tar"
        (out / name).write_bytes(blob)
        if "entry" in spec:
            rec = {"commit": full, "produced": spec["produced"], "entry": spec["entry"]}
        else:
            rec = {"commit": full, "cited": spec["cited"], "named": spec["files"]}
        rec.update({"archive": name, "archive_sha256": hashlib.sha256(blob).hexdigest(), "files": entries})
        if patch is not None:
            # no `path` key: the patch is not a file of any revision, and a pin audit must not read it as one
            rec["change"] = {"member": PATCH_MEMBER, "parent": parent, "diff": " ".join(_DIFF_ARGS[2:]),
                             "sha256": hashlib.sha256(patch).hexdigest(), "bytes": len(patch)}
        manifest["revisions"][short] = rec
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify(out: pathlib.Path, *, against_git: bool = True) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for short, rev in manifest["revisions"].items():
        blob = (out / rev["archive"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != rev["archive_sha256"]:
            raise SnapshotError(f"{short}: archive hash differs from the manifest")
        want = {e["path"]: e["sha256"] for e in rev["files"]}
        change = rev.get("change")
        if change is not None:
            if change.get("member") != PATCH_MEMBER or PATCH_MEMBER in want:
                raise SnapshotError(f"{short}: the change record names another member or collides with a file")
            want[PATCH_MEMBER] = change["sha256"]
        got, patch = {}, None
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for m in tar.getmembers():
                prefix, _, rel = m.name.partition("/")
                if prefix != f"oneq-{short}" or not m.isfile() or ".." in pathlib.PurePosixPath(rel).parts:
                    raise SnapshotError(f"{short}: unexpected archive member {m.name!r}")
                data = tar.extractfile(m).read()
                got[rel] = hashlib.sha256(data).hexdigest()
                if change is not None and rel == PATCH_MEMBER:
                    patch = data
        if got != want:
            raise SnapshotError(f"{short}: archive members differ from the manifest")
        if change is not None:
            # The patch and the files it ships with must describe ONE commit: every file the patch leaves behind
            # is in the archive with exactly the blob id the patch's index line names. Checkable with no history.
            blobs = {e["path"]: e["blob"] for e in rev["files"]}
            post = patch_post_images(patch)
            if not post:
                raise SnapshotError(f"{short}: the patch changes none of the files it ships with")
            wrong = sorted(p for p, b in post.items() if blobs.get(p) != b)
            if wrong:
                raise SnapshotError(f"{short}: the patch's post-images are not the shipped files: {wrong}")
        if against_git:
            for p, h in want.items():
                if p == PATCH_MEMBER:
                    parent_again, again = change_patch(rev["commit"], rev["named"])
                    if change["parent"] != parent_again or hashlib.sha256(again).hexdigest() != h:
                        raise SnapshotError(f"{short}: the patch is not the commit's diff of the files it names")
                elif hashlib.sha256(_git("show", f"{rev['commit']}:{p}")).hexdigest() != h:
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
                what = rev.get("produced") or rev["cited"]
                print(f"{short}  {len(rev['files']):3d} files{' +patch' if 'change' in rev else '       '}  "
                      f"{rev['archive_sha256'][:16]}  {what[:90]}")
    except SnapshotError as exc:
        print(f"SNAPSHOT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
