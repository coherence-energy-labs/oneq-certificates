r"""Sign the release archive with the issuer key -- locally, nothing published.

Binds the archive bytes (by sha256) and the in-toto subject list to the
issuer identity via passport.mint, which refuses the all-zeros seed and
every registry-listed weak key at mint time. The signature makes no number
true -- it answers "who built this archive", beside the witnesses, and its
verification needs only the public key and the archive bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.passport import mint, verify  # noqa: E402

KEY_PATH = pathlib.Path(os.environ.get(
    "ONEQ_ISSUER_KEY", pathlib.Path.home() / ".oneq" / "oneq-issuer.key"))
#: The release's bundles, by name -> the directory each is minted to and archived under. ONE release tag carries both
#: (2026-11-03): the decoder bundle and paper 4's superactivation bundle, each signed on its own. Held equal to the
#: builder's BUNDLES by tests/test_two_bundle_release.py (this file ships in both bundles; the builder in one).
BUNDLE_DIRS = {"certifying-decoder": "oneq-certifying-decoder", "superactivation": "oneq-superactivation"}


def release_paths(bundle: str = "certifying-decoder", root: pathlib.Path = ROOT) -> dict[str, pathlib.Path]:
    """The archive, its minted statement and its signature document, for one bundle."""
    if bundle not in BUNDLE_DIRS:
        raise ValueError(f"no bundle named {bundle!r}; the bundles are {sorted(BUNDLE_DIRS)}")
    d = BUNDLE_DIRS[bundle]
    return {"dir": d, "archive": root / "release" / f"{d}.tar.gz",
            "statement": root / "release" / d / "in-toto-statement.json",
            "signature": root / "release" / f"{d}.tar.gz.sig.json"}


ARCHIVE = release_paths()["archive"]
STATEMENT = release_paths()["statement"]


def release_core(commit: str, archive_name: str, archive_sha: str,
                 subjects: int, statement_sha: str) -> dict:
    """THE SIGNED CORE, in ONE place.

    Both the signer and the verifier must build this identically or the
    recomputed digest will never match and the check becomes untestable
    noise. An external audit found the verifier not recomputing it at
    all -- it searched the document for any string equal to the archive
    digest, which accepts a digest written into an UNSIGNED field.

    Keeping the construction here means the two sides cannot drift: a
    field added to the signature is added to the verification by the
    same edit.
    """
    return {"schema": "oneq-release-signature/1",
            "provenance": {"provider": "oneq-release",
                           "backend": f"git:{commit}"},
            "archive": archive_name,
            "archive_sha256": archive_sha,
            "subjects": int(subjects),
            "statement_sha256": statement_sha}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="sign one bundle of the release")
    ap.add_argument("--bundle", choices=sorted(BUNDLE_DIRS), default="certifying-decoder")
    a = ap.parse_args(argv)
    paths = release_paths(a.bundle)
    archive, statement = paths["archive"], paths["statement"]
    raw = KEY_PATH.read_bytes().strip()
    seed = bytes.fromhex(raw.decode()) if len(raw) == 64 else raw
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    stmt = json.loads(statement.read_text(encoding="utf-8"))
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, timeout=30
                            ).stdout.decode().strip()
    core = release_core(commit, archive.name, sha,
                        len(stmt.get("subject", [])),
                        hashlib.sha256(statement.read_bytes()).hexdigest())
    p = mint(core, private_key_bytes=seed)
    out = {**core, "digest": p["digest"], "pubkey": p["pubkey"],
           "sig_alg": p["sig_alg"], "signature": p["signature"],
           "note": ("binds the archive to the issuer; does NOT make any "
                    "number true -- authenticity beside the witnesses")}
    dest = paths["signature"]
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    v = verify(p)                       # Verdict dataclass (read from source)
    checks = v.checks
    assert checks.get("digest") and checks.get("signature")         and checks.get("key_not_weak") and checks.get("provenance"), (
        f"release signature failed its own verification: {v} {checks}")
    print(f"archive sha256 {sha[:16]}...  pubkey {p['pubkey'][:16]}...  "
          f"signature valid, key not weak, bound to git:{commit[:12]}")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
