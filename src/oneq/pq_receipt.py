r"""Hybrid Ed25519 + ML-DSA-65 envelopes for certificate documents.

QUANTUM-SAFE RECEIPTS FOR A QUANTUM-COMPUTING INSTRUMENT. The atlas's
canonical hybrid-seal design (LumOne_Platform capsule seal, proven
cross-language against liboqs) transplanted to ONE_Q's certificate
documents, with its three laws kept intact because each one closes a real
attack:

  BOTH OR NOTHING -- verify requires BOTH signatures. A missing or empty
  PQ signature is treated as a DOWNGRADE ATTACK and refused, never as
  "classical-only mode": the attacker who can strip a field must not be
  able to buy themselves the weaker game.

  THE PQ KEY IS BOUND INSIDE THE CLASSICALLY-SIGNED CORE -- a quantum
  forger of the Ed25519 signature cannot swap in their own ML-DSA key,
  because the key they must displace is under the signature they are
  trying to survive without.

  CANONICAL BYTES OR NO BYTES -- the signed core is the canonical JSON of
  the certificate document (sorted keys, exact-hex floats via
  cert_format), so two verifiers can disagree about nothing.

*** THE SECOND LAW IS CIRCULAR AS WRITTEN. AN AUDIT CAUGHT IT. ***
A forger who CAN produce the Ed25519 signature simply re-signs a core
containing their own ML-DSA public key. Binding the PQ key under the
classical signature only helps against an adversary who cannot forge
Ed25519 -- precisely the adversary the PQ leg is not needed for.

The root cause was that the PQ leg carried NO IDENTITY: a fresh
ML-DSA-65 keypair per seal, with the stored-key path raising
NotImplementedError, so there was nothing long-lived to pin. That block
turned out to be an UNIMPLEMENTED capability rather than a missing one
-- liboqs accepts `oqs.Signature(alg, secret_key)` and round-trips --
and `seal_document` now supports a stored secret, so a stable PQ
identity is possible.

WHAT REMAINS BEFORE THE LAW HOLDS: ISSUERS.json has no `mldsa_public`
field, so a verifier still has nowhere to pin the PQ key even though one
can now be kept. Until that schema change lands, this envelope gives
post-quantum INTEGRITY -- the document has not changed since sealing,
and a quantum adversary cannot forge that -- but NOT post-quantum
AUTHENTICITY. The classical leg carries authenticity and IS pinned, by
`_issuer_is_trusted`, against both the weak-key registry and the active
issuer list.

Ed25519 comes from oneq.passport (reused, not rebuilt -- and NEVER the
all-zeros seed: passport's weak-key registry refuses it at mint, the
defect the estate's quantum-tournament receipts still carry). ML-DSA-65
comes from liboqs.
"""

from __future__ import annotations

from typing import Any

ALG_CLASSICAL = "Ed25519"
ALG_PQ = "ML-DSA-65"


def _issuer_is_trusted(pubkey_hex: str) -> tuple[bool, str]:
    """(ok, why) -- weak-key registry AND issuer pinning, fail closed.

    Mirrors what verify_closures.py already enforces rather than
    inventing a second policy. A missing or unreadable registry is a
    REFUSAL: deleting a file must not buy the trust the file withheld.
    """
    import json as _json
    import pathlib as _pl
    import re as _re

    key = (pubkey_hex or "").lower()
    if not _re.fullmatch(r"[0-9a-f]{64}", key):
        return False, "ed25519_public is not a 64-hex key"
    try:
        from .passport import WEAK_PUBKEYS
        if key in {str(k).lower() for k in WEAK_PUBKEYS}:
            return False, (f"signed by a REGISTRY-WEAK key "
                           f"{key[:16]}... -- refused")
    except Exception:
        return False, ("cannot load the weak-key registry; refusing "
                       "rather than trusting an unchecked key")

    here = _pl.Path(__file__).resolve()
    for base in (here.parents[2], here.parents[2] / "release",
                 here.parent):
        f = base / "ISSUERS.json"
        if not f.exists():
            continue
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return False, "ISSUERS.json is unreadable; refusing"
        if d.get("schema") != "oneq-issuers/1":
            return False, "ISSUERS.json schema is not oneq-issuers/1"
        active = {str(i.get("pubkey", "")).lower()
                  for i in d.get("issuers", [])
                  if i.get("status") == "active"}
        revoked = {str(i.get("pubkey", "")).lower()
                   for i in d.get("revoked", [])}
        if key in revoked:
            return False, f"signed by a REVOKED key {key[:16]}..."
        if key not in active:
            return False, f"signed by an UNPINNED key {key[:16]}..."
        return True, f"issuer {key[:16]}... pinned and not revoked"
    return False, ("NO TRUSTED ISSUER REGISTRY found -- refusing, "
                   "because deleting a file must not buy trust")


def _canon(doc: dict) -> bytes:
    """The ONE canonicaliser, imported -- never re-implemented.

    This used to be its own json.dumps and omitted allow_nan=False,
    which passport.canonical_bytes treats as load-bearing: NaN and
    Infinity are not valid JSON, so `{"x":NaN}` is an ambiguity two
    parsers may resolve differently, and an ambiguity is a forgery
    surface. cert_format already states the rule this file broke --
    "signing is reused wholesale from passport.py, because a second
    crypto path is a second thing to get wrong."
    """
    from .passport import canonical_bytes
    return canonical_bytes(doc)


def seal_document(doc: dict, *, ed25519_private: bytes,
                  mldsa_secret: bytes | None = None,
                  mldsa_public: bytes | None = None) -> dict[str, Any]:
    """Seal `doc` under the hybrid envelope. Returns the sealed wrapper.

    If `mldsa_secret` is None a fresh ML-DSA-65 keypair is generated and
    the secret is RETURNED inside the wrapper under '_mldsa_secret' for the
    caller to store -- it is stripped from the signed content and must
    never be published.

    PASSING A STORED SECRET IS NOW SUPPORTED, and it is what a
    long-lived PQ identity requires. This used to raise
    NotImplementedError claiming liboqs secret import was unavailable;
    it is available -- `oqs.Signature(alg, secret_key)` round-trips --
    so the PQ leg was generating a throwaway keypair per seal for no
    reason, leaving nothing for a verifier to pin.

    `mldsa_public` must accompany a stored secret: liboqs returns the
    public key only from generate_keypair(), so a re-imported secret
    cannot re-derive it. A mismatched pair is caught here rather than
    producing a seal that verifies against nothing.
    """
    import oqs
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey)

    sk = Ed25519PrivateKey.from_private_bytes(ed25519_private)
    if mldsa_secret is not None:
        if mldsa_public is None:
            raise ValueError(
                "a stored ML-DSA secret needs its matching public key: "
                "liboqs returns the public key only from "
                "generate_keypair(), so it cannot be re-derived from the "
                "secret. Passing one without the other would produce a "
                "seal that verifies against nothing.")
        sig = oqs.Signature(ALG_PQ, mldsa_secret)
        mldsa_secret_out = mldsa_secret
        # SELF-CHECK THE PAIR. A mismatched secret/public pair seals
        # happily and fails only at verification, on someone else's
        # machine, with no way to tell which half was wrong.
        probe = sig.sign(b"pq_receipt keypair self-check")
        if not oqs.Signature(ALG_PQ).verify(
                b"pq_receipt keypair self-check", probe, mldsa_public):
            raise ValueError(
                "the supplied ML-DSA secret does not match the supplied "
                "public key -- refusing to seal a document nobody could "
                "verify")
    else:
        sig = oqs.Signature(ALG_PQ)
        mldsa_public = sig.generate_keypair()
        mldsa_secret_out = sig.export_secret_key()

    core = dict(doc)
    core["sig_algs"] = [ALG_CLASSICAL, ALG_PQ]
    core["mldsa_public"] = mldsa_public.hex()   # BOUND inside classical sig
    core_bytes = _canon(core)

    ed_sig = sk.sign(core_bytes)
    pq_sig = sig.sign(core_bytes)
    pub = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    return {"core": core,
            "ed25519_public": pub.hex(),
            "ed25519_signature": ed_sig.hex(),
            "mldsa_signature": pq_sig.hex(),
            "_mldsa_secret": mldsa_secret_out.hex()}


def verify_sealed(sealed: dict) -> tuple[bool, str]:
    """BOTH signatures verify over the SAME canonical core, or refusal."""
    import oqs
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey)

    core = sealed.get("core")
    if not isinstance(core, dict):
        return False, "no core"
    if core.get("sig_algs") != [ALG_CLASSICAL, ALG_PQ]:
        return False, ("sig_algs != [Ed25519, ML-DSA-65]: a weakened "
                       "algorithm list is a downgrade attack, refused")
    pq_sig_hex = sealed.get("mldsa_signature")
    if not pq_sig_hex:
        return False, ("missing/empty ML-DSA signature: DOWNGRADE ATTACK "
                       "refused -- classical-only is not a fallback")
    ed_sig_hex = sealed.get("ed25519_signature")
    if not ed_sig_hex:
        return False, "missing Ed25519 signature"
    # PIN THE CLASSICAL KEY. This read ed25519_public straight from the
    # envelope with no trusted set at all, so the all-zeros seed -- this
    # estate's own revoked, permanent Genesis Test 0 mutant -- verified
    # here while verify_closures.py refused it. A signature checked
    # against a key the document supplies proves only that the document
    # is self-consistent.
    ed_pub = str(sealed.get("ed25519_public") or "")
    ok_key, why_key = _issuer_is_trusted(ed_pub)
    if not ok_key:
        return False, why_key
    try:
        core_bytes = _canon(core)
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(ed_pub)).verify(
            bytes.fromhex(ed_sig_hex), core_bytes)
    except InvalidSignature:
        return False, "Ed25519 signature invalid"
    except Exception as e:
        return False, f"Ed25519 verification failed: {e}"
    try:
        mldsa_public = bytes.fromhex(core["mldsa_public"])
        ok = oqs.Signature(ALG_PQ).verify(core_bytes,
                                          bytes.fromhex(pq_sig_hex),
                                          mldsa_public)
    except Exception as e:
        return False, f"ML-DSA verification failed: {e}"
    if not ok:
        return False, "ML-DSA-65 signature invalid"
    return True, "hybrid seal verified: Ed25519 AND ML-DSA-65 over the " \
                 "same canonical core, PQ key bound under the classical sig"
