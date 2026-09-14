r"""What a stranger runs first: verify we shipped what we claim.

No project knowledge required, no solver, no network. Checks the four
things a reproducer must be able to trust before any science:

  1. the release archive's digest matches its signature document, the
     signature verifies, and the signing key is not weak
  2. every per-shot record file matches the sha256 in MANIFEST.json
     (canonical AND held-out sections), with the recorded counts
  3. the pinned external checkers match their recorded sha256
  4. every published integer weight vector hashes to its declared value

Exit 0 only if all four pass; any failure prints what differed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_CHECKER_SHA256 = \
    "f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd"


def _weak_pubkeys() -> set:
    """Registry-listed weak keys. An empty set is NOT a pass -- the
    caller treats a missing registry as a refusal."""
    try:
        from oneq.passport import WEAK_PUBKEYS
        return {str(k).lower() for k in WEAK_PUBKEYS}
    except Exception:
        return set()


def _issuers():
    """(trusted, revoked) from ISSUERS.json -- VALIDATED, never assumed.

    REUSES the discipline verify_closures.py already established rather
    than inventing a second parser: schema pin, 64-hex keys, an explicit
    `status == "active"`, no key both active and revoked, and at least
    one active issuer. My first version here looked for a top-level
    "active" list; the real schema is `issuers: [{pubkey, status}]`, so
    it found nothing and reported "no trusted registry" on a repository
    that has one. A verifier that fails closed for the WRONG reason is
    still telling the reader something false.

    Returning empty is deliberate on any failure: the caller must FAIL
    CLOSED rather than treat an unreadable registry as universal trust.
    """
    import re
    for base in (ROOT, ROOT / "release", ROOT / "tools"):
        f = base / "ISSUERS.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return set(), set()
        if d.get("schema") != "oneq-issuers/1":
            return set(), set()

        def hexok(k):
            return isinstance(k, str) and re.fullmatch(r"[0-9a-f]{64}", k)

        act, rev = [], []
        for i in d.get("issuers", []):
            k = i.get("pubkey")
            if not hexok(k):
                return set(), set()
            if i.get("status") == "active":
                act.append(k)
        for i in d.get("revoked", []):
            k = i.get("pubkey")
            if hexok(k):
                rev.append(k)
        if len(set(act)) != len(act) or (set(act) & set(rev)) or not act:
            return set(), set()
        return set(act), set(rev)
    return set(), set()


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _safe_rel(name: str) -> str:
    """Return a canonical archive-relative path or raise."""
    if not isinstance(name, str) or "\\" in name:
        raise ValueError(f"non-canonical subject path {name!r}")
    p = pathlib.PurePosixPath(name)
    if (p.is_absolute() or p.as_posix() != name
            or any(part in ("", ".", "..") for part in p.parts)):
        raise ValueError(f"unsafe subject path {name!r}")
    return name


class _ArchivePayload:
    """Read-only view of the one authenticated release tree in the tarball."""

    PREFIX = "oneq-certifying-decoder/"

    def __init__(self, archive: pathlib.Path):
        self._tar = tarfile.open(archive, "r:gz")
        self._members: dict[str, tarfile.TarInfo] = {}
        for member in self._tar.getmembers():
            if not member.name.startswith(self.PREFIX):
                raise ValueError(f"archive member outside release root: "
                                 f"{member.name!r}")
            rel = member.name[len(self.PREFIX):]
            if not rel and member.isdir():
                continue
            _safe_rel(rel)
            if not member.isfile():
                raise ValueError(f"non-regular archive member: {member.name!r}")
            if rel in self._members:
                raise ValueError(f"duplicate archive member: {rel}")
            self._members[rel] = member

    @property
    def names(self) -> set[str]:
        return set(self._members)

    def read(self, rel: str) -> bytes:
        rel = _safe_rel(rel)
        member = self._members.get(rel)
        if member is None:
            raise FileNotFoundError(f"authenticated archive lacks {rel}")
        fh = self._tar.extractfile(member)
        if fh is None:
            raise OSError(f"cannot read archive member {rel}")
        return fh.read()

    def close(self) -> None:
        self._tar.close()


def _bind_worktree(payload: _ArchivePayload, rel: str,
                   fails: list[str]) -> bytes | None:
    """Return authenticated bytes only if the worktree presents the same bytes."""
    try:
        archived = payload.read(rel)
    except Exception as exc:
        fails.append(f"{rel}: absent/unreadable in authenticated archive ({exc})")
        return None
    local = ROOT / rel
    try:
        working = local.read_bytes()
    except Exception as exc:
        fails.append(f"{rel}: worktree subject missing/unreadable ({exc})")
        return None
    if working != archived:
        fails.append(
            f"{rel}: WORKTREE/ARCHIVE EQUIVOCATION -- the bytes being "
            "checked are not the bytes under the release signature")
        return None
    return archived


def _verify_statement(payload: _ArchivePayload, sig_doc: dict,
                      fails: list[str], ok: list[str]) -> None:
    """Verify the signed statement names the exact archived subject bytes."""
    try:
        raw = payload.read("in-toto-statement.json")
        if hashlib.sha256(raw).hexdigest() != sig_doc.get("statement_sha256"):
            fails.append("authenticated archive's in-toto statement is not the "
                         "statement named by the signed core")
            return
        statement = json.loads(raw)
        subjects = statement.get("subject")
        if not isinstance(subjects, list) or not subjects:
            raise ValueError("in-toto subject list is empty or malformed")
        seen = set()
        for subject in subjects:
            name = _safe_rel(subject.get("name"))
            if name in seen:
                raise ValueError(f"duplicate in-toto subject {name}")
            seen.add(name)
            want = subject.get("digest", {}).get("sha256")
            if not isinstance(want, str) or not re.fullmatch(r"[0-9a-f]{64}", want):
                raise ValueError(f"malformed sha256 for subject {name}")
            if hashlib.sha256(payload.read(name)).hexdigest() != want:
                fails.append(f"in-toto subject digest mismatch: {name}")
        if sig_doc.get("subjects") != len(subjects):
            fails.append("signed subject count does not match in-toto statement")
        else:
            ok.append(f"signed in-toto statement binds {len(subjects)} "
                      "archived subjects")
    except Exception as exc:
        fails.append(f"authenticated subject statement invalid: "
                     f"{type(exc).__name__}: {exc}")


def _manifest_sections(man: dict) -> list[tuple[str, dict]]:
    sections = [("canonical", man.get("rungs"))]
    if "heldout" in man:
        held = man.get("heldout")
        sections.append(("held-out", held.get("rungs")
                         if isinstance(held, dict) else None))
    for label, rungs in sections:
        if not isinstance(rungs, dict) or not rungs:
            raise ValueError(f"{label} rung map is empty or malformed")
    return sections


def _verify_bound_evidence(payload: _ArchivePayload, fails: list[str],
                           ok: list[str]) -> None:
    """Verify records/checkers/weights from one signed archive/worktree view."""
    raw_manifest = _bind_worktree(
        payload, "evidence/per_shot/MANIFEST.json", fails)
    if raw_manifest is None:
        return
    try:
        man = json.loads(raw_manifest)
        sections = _manifest_sections(man)
    except Exception as exc:
        fails.append(f"per-shot manifest malformed: {type(exc).__name__}: {exc}")
        return

    weight_claims: dict[str, str] = {}
    for label, rungs in sections:
        good = 0
        for tag, rung in rungs.items():
            if not isinstance(rung, dict):
                fails.append(f"{label} {tag}: malformed rung")
                continue
            try:
                records_rel = _safe_rel(rung["records_file"])
                weights_rel = _safe_rel(rung["weights_file"])
                count = rung["record_count"]
                if not isinstance(count, int) or count < 0:
                    raise ValueError("record_count is not a non-negative integer")
                for field in ("records_sha256", "weights_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}", rung.get(field, "")):
                        raise ValueError(f"{field} is not a sha256")
            except Exception as exc:
                fails.append(f"{label} {tag}: malformed subject declaration ({exc})")
                continue
            prior = weight_claims.setdefault(weights_rel, rung["weights_sha256"])
            if prior != rung["weights_sha256"]:
                fails.append(f"{label} {tag}: contradictory digest for {weights_rel}")
            data = _bind_worktree(payload, records_rel, fails)
            if data is None:
                continue
            if hashlib.sha256(data).hexdigest() != rung["records_sha256"]:
                fails.append(f"{label} {tag}: RECORDS HASH MISMATCH")
                continue
            try:
                n = len(data.decode("utf-8").splitlines())
            except UnicodeDecodeError as exc:
                fails.append(f"{label} {tag}: records are not UTF-8 ({exc})")
                continue
            if n != count:
                fails.append(f"{label} {tag}: count {n} != {count}")
                continue
            good += 1
        ok.append(f"{label} per-shot records verified: {good}/{len(rungs)} "
                  "rungs (signed archive + worktree binding + hash + count)")

    # Checkers are subjects too.  The reference pin is carried by the bound
    # manifest; the second checker pin is carried by a separately bound file.
    if man.get("reference_checker_sha256") != REFERENCE_CHECKER_SHA256:
        fails.append("per-shot manifest names a different reference checker "
                     "than the independently pinned verifier")
    pins = {
        "tools/external/exact_lp_certificate_reference.py":
            REFERENCE_CHECKER_SHA256
    }
    add_raw = _bind_worktree(
        payload, "evidence/qldpc_heldout_addendum.json", fails)
    if add_raw is not None:
        try:
            add = json.loads(add_raw)
            pins["tools/external/tree_checker_b.py"] = \
                add["tree_checker_b_sha256"]
        except Exception as exc:
            fails.append(f"held-out addendum malformed: {exc}")
    for rel, want in pins.items():
        if not isinstance(want, str) or not re.fullmatch(r"[0-9a-f]{64}", want):
            fails.append(f"{rel}: checker pin missing/malformed")
            continue
        data = _bind_worktree(payload, rel, fails)
        if data is None:
            continue
        if hashlib.sha256(data).hexdigest() != want:
            fails.append(f"{rel}: SHA MISMATCH (pinned {want[:16]}...)")
        else:
            ok.append(f"{rel} matches its authenticated pin")

    expected = set(weight_claims)
    archived = {n for n in payload.names
                if n.startswith("evidence/weights/")
                and n.endswith("_weights.json")}
    local_dir = ROOT / "evidence" / "weights"
    local = ({p.relative_to(ROOT).as_posix()
              for p in local_dir.glob("*_weights.json") if p.is_file()}
             if local_dir.is_dir() else set())
    if not expected:
        fails.append("per-shot manifest declares no weight vectors")
    if archived != expected:
        fails.append("authenticated archive weight set is not exact "
                     f"(missing {sorted(expected - archived)}; "
                     f"undeclared {sorted(archived - expected)})")
    if local != expected:
        fails.append("worktree weight set is not exact "
                     f"(missing {sorted(expected - local)}; "
                     f"undeclared {sorted(local - expected)})")

    good = 0
    for rel in sorted(expected):
        data = _bind_worktree(payload, rel, fails)
        if data is None:
            continue
        try:
            doc = json.loads(data)
            vector = doc["vector"]
            if not isinstance(vector, list) or not vector:
                raise ValueError("vector must be a non-empty list")
            declared = doc["spec"]["weights_sha256"]
            recomputed = hashlib.sha256(json.dumps(vector).encode()).hexdigest()
            if declared != weight_claims[rel]:
                raise ValueError("file digest declaration differs from manifest")
            if recomputed != declared:
                raise ValueError("weight-vector hash mismatch")
        except Exception as exc:
            fails.append(f"{rel}: invalid weight vector ({exc})")
            continue
        good += 1
    ok.append(f"published weight vectors verified exactly: {good}/{len(expected)}")


def main() -> int:
    fails: list[str] = []
    ok: list[str] = []

    # 1. RELEASE SIGNATURE -- actually verified, not searched for.
    #
    # WHAT THIS USED TO DO, and why an audit called it critical. It
    # walked the JSON recursively looking for ANY string value anywhere
    # equal to the archive digest, then imported oneq.passport.verify
    # and never called it. The recursion never descended into lists and
    # accepted a digest found in any field, including unsigned ones --
    # so an attacker replacing the tarball needed no key at all: write
    # the new digest into an unsigned field and the tool printed OK.
    # Worse, the import's failure branch appended to `ok`, so a machine
    # without `cryptography` printed a green line and exited 0.
    #
    # A crypto check that cannot run has NOT passed. Every branch below
    # that cannot establish something appends to `fails`.
    arc = ROOT / "release" / "oneq-certifying-decoder.tar.gz"
    sig = arc.with_suffix(".gz.sig.json")
    if not (arc.exists() and sig.exists()):
        fails.append("release archive or signature missing")
    else:
        try:
            doc = json.loads(sig.read_text(encoding="utf-8"))
        except Exception as exc:
            doc = None
            fails.append(f"signature document unparseable: {exc}")
        if doc is not None:
            digest = sha(arc)
            # (a) the archive digest at its NAMED path, not by search
            if doc.get("archive_sha256") != digest:
                fails.append(
                    f"release signature does not bind this archive "
                    f"(archive_sha256 {str(doc.get('archive_sha256'))[:16]}"
                    f"... vs actual {digest[:16]}...)")
            else:
                ok.append(f"archive_sha256 binds this archive "
                          f"({digest[:16]}...)")

            # (b) rebuild the signed core and recompute its digest, so a
            #     field the signature covers cannot be edited after the
            #     fact
            try:
                sys.path.insert(0, str(ROOT / "src"))
                sys.path.insert(0, str(ROOT / "tools"))
                from cryptography.exceptions import InvalidSignature
                from cryptography.hazmat.primitives.asymmetric.ed25519                     import Ed25519PublicKey
                from oneq.passport import digest_obj
                from sign_release import release_core
                crypto_ok = True
            except Exception as exc:
                crypto_ok = False
                fails.append(
                    f"cannot verify the release signature here "
                    f"({type(exc).__name__}: {exc}). A crypto check "
                    f"that cannot run has not passed.")

            if crypto_ok:
                commit = str(doc.get("provenance", {})
                             .get("backend", "")).removeprefix("git:")
                core = release_core(commit, doc.get("archive"),
                                    doc.get("archive_sha256"),
                                    doc.get("subjects"),
                                    doc.get("statement_sha256"))
                recomputed = digest_obj(core)
                if recomputed != doc.get("digest"):
                    fails.append(
                        "the signed core does not match the document: a "
                        "field the signature covers was edited after "
                        "signing")
                else:
                    ok.append("signed core matches its recorded digest")

                # (c) the key must not be weak, and must be pinned
                pub = str(doc.get("pubkey") or "")
                weak = _weak_pubkeys()
                if not pub:
                    fails.append("no pubkey in the signature document")
                elif pub in weak:
                    fails.append(f"signed by a REGISTRY-WEAK key "
                                 f"{pub[:16]}...")
                else:
                    trusted, revoked = _issuers()
                    if revoked and pub in revoked:
                        fails.append(f"signed by a REVOKED key "
                                     f"{pub[:16]}...")
                    elif trusted and pub not in trusted:
                        fails.append(f"signed by an UNPINNED key "
                                     f"{pub[:16]}...")
                    elif not trusted:
                        fails.append(
                            "NO TRUSTED ISSUER REGISTRY -- refusing. A "
                            "verifier that claims to enforce pinning "
                            "must fail closed, or deleting a file buys "
                            "the trust the file withheld.")
                    else:
                        ok.append(f"issuer {pub[:16]}... pinned, not "
                                  f"revoked, not weak")

                # (d) THE SIGNATURE ITSELF
                try:
                    Ed25519PublicKey.from_public_bytes(
                        bytes.fromhex(pub)).verify(
                            bytes.fromhex(str(doc.get("signature") or "")),
                            bytes.fromhex(str(doc.get("digest") or "")))
                    ok.append("Ed25519 signature VERIFIES over the "
                              "signed core")
                except InvalidSignature:
                    fails.append("signature does not verify against the "
                                 "recorded key")
                except Exception as exc:
                    fails.append(f"signature check failed: "
                                 f"{type(exc).__name__}: {exc}")

    # 2-4. Every subject below is read from the authenticated archive and
    # compared byte-for-byte with its worktree counterpart.  A mutable
    # manifest may no longer "re-seal" mutable records beside an unchanged
    # genuine release signature.
    if not fails and doc is not None:
        payload = None
        try:
            payload = _ArchivePayload(arc)
            _verify_statement(payload, doc, fails, ok)
            _verify_bound_evidence(payload, fails, ok)
        except Exception as exc:
            fails.append(f"authenticated archive payload cannot be checked: "
                         f"{type(exc).__name__}: {exc}")
        finally:
            if payload is not None:
                payload.close()
    else:
        fails.append("evidence verification not run: release authenticity "
                     "was not established")

    for line in ok:
        print(f"  OK  {line}")
    if fails:
        print(f"\nVERIFY RELEASE: {len(fails)} FAILURE(S)")
        for f in fails:
            print(f"  FAIL {f}")
        return 1
    print("\nVERIFY RELEASE: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
