#!/usr/bin/env python3
r"""Mint an evidence release: ONE rule for what is in it, applied everywhere.

WHY THIS EXISTS. Four releases in a row shipped an artifact that contradicted
its own description, and every one traced to the same shape: different parts of
the pipeline disagreed about WHICH BYTES the release consists of.

  v1.0.0  absolute paths -> crashed on other machines
  v1.0.1  verifier fixed in the directory, not in the published archive
  v1.0.2  manifest predated the verifier fix; tree pin computed over CRLF
          bytes that git never stores, so no clean clone could match
  v1.0.3  manifest regenerated over __pycache__/*.pyc left by a local test
          run -- files a clean clone does not have, so sha256sum reported
          "No such file" for six entries

The tarball builder excluded __pycache__. The manifest generator did not. That
one-line disagreement is the whole bug, and it recurred because the exclusion
lived in two places. Here it lives in ONE, and the manifest, the archive and
the tree pin are all derived from the same file list.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import tarfile

TEXT = {".py", ".md", ".json", ".txt", ".cff", ".sh", ".ps1"}


def payload(root: pathlib.Path) -> list[pathlib.Path]:
    """THE file list. Everything else derives from this."""
    out = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if "__pycache__" in f.parts or f.suffix == ".pyc":
            continue                      # never in git, never in a clone
        if f.name == "SHA256SUMS":
            continue                      # the manifest is not in itself
        out.append(f)
    return out


def normalize(root: pathlib.Path, keep: set[pathlib.Path] | None = None) -> int:
    """LF everywhere, so a hash pin survives the trip through git -- except the files in `keep`.

    `keep` names files another shipped file pins by their current digest: rewriting them would
    break that pin for every reader (build_certificate_bundle decides which; 2026-09-15).
    """
    n = 0
    for f in payload(root):
        if f.suffix.lower() in TEXT and not (keep and f in keep):
            b = f.read_bytes()
            nb = b.replace(bytes([13, 10]), bytes([10]))
            if nb != b:
                f.write_bytes(nb)
                n += 1
    return n


def write_manifest(root: pathlib.Path) -> int:
    files = payload(root)
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  "
             f"{f.relative_to(root).as_posix()}" for f in files]
    (root / "SHA256SUMS").write_bytes(("\n".join(lines) + "\n").encode())
    return len(files)


def _reproducible(info: "tarfile.TarInfo") -> "tarfile.TarInfo":
    """Strip everything that records WHERE and WHEN a build happened.

    mtime, uid, gid, uname and gname come from the builder's filesystem
    and have nothing to do with the content being signed. Left in, two
    builds of byte-identical content produce different archives, so the
    digest a signature binds can never be re-derived by a reader -- and
    every CI run that rebuilds the bundle silently invalidates the
    signature beside it.
    """
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    # normalise the permission bits too: a umask difference between two
    # machines is not a content difference
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def build_archive(root: pathlib.Path, dest: pathlib.Path) -> str:
    """A byte-identical archive for byte-identical content.

    gzip stamps the current time into its own header, which tarfile's
    "w:gz" mode gives no way to suppress -- so the gzip stream is opened
    explicitly with mtime=0 and an empty filename, and tar is layered on
    top of it. Members go through `_reproducible` for the same reason.

    Verified by tools/verify_release.py, which builds twice and requires
    the two digests to match. Determinism that nothing checks does not
    stay true.
    """
    import gzip

    keep = {f.relative_to(root).as_posix() for f in payload(root)}
    keep.add("SHA256SUMS")
    if dest.exists():
        dest.unlink()
    with open(dest, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w",
                              format=tarfile.GNU_FORMAT) as t:
                for f in sorted(root.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(root).as_posix()
                    if rel in keep:
                        t.add(f, arcname=f"{root.name}/{rel}",
                              filter=_reproducible)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    dest = pathlib.Path(sys.argv[2])
    print(f"normalized {normalize(root)} files to LF")
    print(f"manifest over {write_manifest(root)} files")
    sha = build_archive(root, dest)
    print(f"archive {dest.name}: {sha}")
    print(sha, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
