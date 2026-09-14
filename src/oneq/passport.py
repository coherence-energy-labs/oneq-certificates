"""The Fault-Tolerance Passport — ONE-Q's evidence object.

WHY THIS MODULE EXISTS
----------------------
The estate's flagship quantum receipts were signed with the all-zeros Ed25519 seed
(`scripts/quantum_tournament.py:59` passes `receipt_seed=bytes(32)`), making every
signature forgeable by anyone, and their signed core carried no `backend`/`job_id`,
so a receipt could not attest *which machine* produced the result.

The root cause is NOT carelessness. It is a real design tension:

    REPLAY wants determinism      -- re-running must reproduce the artifact byte-for-byte.
    PROVENANCE wants a secret     -- a signature only means something if others cannot forge it.

Fixing the signing key to a constant buys the first by destroying the second. The
correct resolution is that these are two INDEPENDENT verification axes and must never
share a mechanism:

    L1 REPLAY      the CORE re-derives byte-identically from its inputs.  Keyless.
                   Anyone can check it. Determinism lives here.
    L0 ASSERTED    the core is signed by a real, secret key identifying the issuer.
                   Unforgeability lives here.

A passport therefore reproduces its *digest* on re-run, never its *signature*.

The trust lattice's cardinal rule is enforced on every verify:

    ACHIEVED = min(CLAIMED, INDEPENDENTLY_REVERIFIABLE)

CANONICAL HOMES (reused, not reimplemented -- see CAPABILITY_ATLAS.md)
    canonical serialization + digest : coherence_covenant/covenant/canon.py
    the L0-L5 soundness lattice      : coherence_covenant/covenant/lattice.py
A byte-identical fallback is vendored below so a stranger can verify a passport with
this file, `cryptography`, and nothing else from the estate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Mapping

SCHEMA = "oneq-passport/1"

# ---------------------------------------------------------------------------
# Canonical bytes. Identical rules to covenant/canon.py; vendored so the verifier
# is standalone. allow_nan=False is load-bearing: NaN/Infinity serialize to tokens
# that are not valid JSON, so two verifiers could disagree -- an ambiguity is a
# forgery surface. Non-finite input fails closed at mint time.
# ---------------------------------------------------------------------------

def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_obj(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))


class L(IntEnum):
    """Soundness levels. A level is a claim about how a stranger can check you."""
    ASSERTED = 0   # a signature exists; trust reduces to the signer's word
    REPLAY = 1     # re-running reproduces the core byte-for-byte
    ATTESTED = 2   # ran inside measured hardware (TPM/TEE quote chained in)
    LOGGED = 3     # committed to a witnessed append-only log
    PROVEN = 4     # a succinct proof verifies it without re-execution
    FOLDED = 5     # the whole history folds via IVC into one O(1) proof


LEVEL_NAME = {
    L.ASSERTED: "L0-ASSERTED", L.REPLAY: "L1-REPLAY", L.ATTESTED: "L2-ATTESTED",
    L.LOGGED: "L3-LOGGED", L.PROVEN: "L4-PROVEN", L.FOLDED: "L5-FOLDED",
}

# ---------------------------------------------------------------------------
# Weak-key registry. Derived at import, never hardcoded, so it cannot drift from
# the seeds it claims to cover. The all-zeros seed is the estate's own historical
# defect and is a PERMANENT test mutant: Genesis Test 0 asserts the verifier
# rejects it. A gate that cannot fail is not a gate.
# ---------------------------------------------------------------------------

_WEAK_SEEDS: tuple[bytes, ...] = (
    bytes(32),                      # all zeros -- the historical defect
    b"\xff" * 32,                   # all ones
    bytes(range(32)),               # 00 01 02 ... 1f
    b"\x01" + bytes(31),            # low-entropy sentinel
    hashlib.sha256(b"test").digest(),
    hashlib.sha256(b"").digest(),
)


def _derive_weak_pubkeys() -> frozenset[str]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:  # pragma: no cover - verification still works without minting
        return frozenset()
    out = set()
    for seed in _WEAK_SEEDS:
        pub = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        out.add(pub.hex())
    return frozenset(out)


WEAK_PUBKEYS: frozenset[str] = _derive_weak_pubkeys()


class WeakKeyError(ValueError):
    """Raised when a passport is minted with a publicly-derivable key."""


# ---------------------------------------------------------------------------
# Provenance -- the fields whose absence made the old receipts unable to attest
# which machine produced a result. These live INSIDE the signed core.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    provider: str                      # "ibm_quantum" | "aws_braket" | "simulator" | ...
    backend: str                       # "ibm_marrakesh" | "stim" | ...
    job_id: str | None = None          # the vendor's job identifier
    calibration_hash: str | None = None  # sha256 of the calibration snapshot in force
    shots: int | None = None
    seed: int | None = None            # RNG seed -- a run is not reproducible without it
    software: Mapping[str, str] = field(default_factory=dict)  # {"stim": "1.13.0", ...}

    # THE SIX FIELDS THE PROGRAM SPEC REQUIRES IN THE SIGNED CORE
    # (FTQC_PROGRAM.md milestone 0.1). Named here so completeness is checked
    # against the spec rather than against whatever happened to be filled in.
    REQUIRED = ("backend", "job_id", "calibration_hash", "software", "shots", "seed")

    def to_core(self) -> dict[str, Any]:
        d: dict[str, Any] = {"provider": self.provider, "backend": self.backend}
        if self.job_id is not None:
            d["job_id"] = self.job_id
        if self.calibration_hash is not None:
            d["calibration_hash"] = self.calibration_hash
        if self.shots is not None:
            d["shots"] = int(self.shots)
        if self.seed is not None:
            d["seed"] = int(self.seed)
        if self.software:
            d["software"] = dict(sorted(self.software.items()))
        # DECLARE INCOMPLETENESS RATHER THAN HIDE IT. A field left None is
        # invisible in the signed core, so a reader cannot tell "not captured"
        # from "not applicable". The reissued 2026-07-12 receipts are the case
        # that forced this: their calibration snapshot was never captured at
        # run time and CANNOT be recovered -- fabricating a hash would be
        # worse than admitting the gap, so the gap is signed too.
        missing = [k for k in self.REQUIRED if not d.get(k)]
        if missing:
            d["provenance_incomplete"] = missing
        return d


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------

def build_core(
    *,
    claim: Mapping[str, Any],
    provenance: Provenance,
    prereg_hash: str | None = None,
    inputs: Mapping[str, str] | None = None,
    claimed_level: L = L.ASSERTED,
    expires_after_calibration_change: bool = True,
    soundness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The deterministic core. Two honest parties re-running the same computation
    build a byte-identical core and therefore an identical digest -- with different
    signatures. That separation is the whole point."""
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "claim": _freeze(claim),
        "provenance": provenance.to_core(),
        "claimed_level": int(claimed_level),
        "expires_on_calibration_change": bool(expires_after_calibration_change),
    }
    if soundness is not None:
        # THE COMPOSED SOUNDNESS BUDGET, INSIDE THE SIGNATURE. Built by
        # `soundness_budget.passport_block`, which refuses to produce one
        # for a claim the evidence does not support -- so the check
        # happens before minting rather than as a note beside it.
        #
        # It is checked here TOO, because a caller can hand-build the
        # block. The signed core is the last place a claim can be
        # inflated, and "the producer already checked" is exactly the
        # trust this repository does not extend to producers.
        claimed_in_block = int(soundness.get("claimed_level", -1))
        if claimed_in_block != int(claimed_level):
            raise ValueError(
                f"the soundness block claims level {claimed_in_block} and "
                f"the passport claims {int(claimed_level)}; two numbers for "
                f"one claim is how a claim gets inflated between them")
        supported = int(soundness.get("supported_level", -1))
        if int(claimed_level) > supported:
            raise ValueError(
                f"claimed_level {int(claimed_level)} exceeds the "
                f"supported_level {supported} the evidence carries. "
                f"Understating is free; overstating is the forgery this "
                f"repository was audited for")
        core["soundness"] = _freeze(soundness)
    if prereg_hash is not None:
        core["prereg_hash"] = prereg_hash
    if inputs:
        core["inputs"] = dict(sorted(inputs.items()))
    return core


def _freeze(obj: Any) -> Any:
    """Reject floats in the signed core. Floats are the classic cross-language
    divergence: two verifiers can format the same value differently and disagree
    about a signature. Encode a measured quantity as an exact ratio or a string."""
    if isinstance(obj, bool) or isinstance(obj, int) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        raise TypeError(
            "floats are not permitted in a passport core (cross-language formatting "
            "is a forgery surface); encode as an exact ratio {'num':int,'den':int} or a string"
        )
    if isinstance(obj, Mapping):
        return {str(k): _freeze(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_freeze(v) for v in obj]
    raise TypeError(f"unserializable type in passport core: {type(obj).__name__}")


def mint(
    core: Mapping[str, Any],
    *,
    private_key_bytes: bytes,
    allow_weak_key: bool = False,
) -> dict[str, Any]:
    """Sign a core. Refuses publicly-derivable keys unless explicitly overridden
    (the override exists only so the test-suite can construct the mutant)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if len(private_key_bytes) != 32:
        raise ValueError("Ed25519 seed must be exactly 32 bytes")

    key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    pub_hex = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    ).hex()

    if pub_hex in WEAK_PUBKEYS and not allow_weak_key:
        raise WeakKeyError(
            f"refusing to mint with a publicly-derivable key (pubkey {pub_hex[:16]}...). "
            "This is the estate's historical defect: determinism was bought by fixing the "
            "signing key, which destroys unforgeability. Reproducibility belongs to the "
            "CORE DIGEST (L1 REPLAY), never to the signature."
        )

    frozen = _freeze(core)
    digest = digest_obj(frozen)
    sig = key.sign(bytes.fromhex(digest))
    return {
        "core": frozen,
        "digest": digest,
        "pubkey": pub_hex,
        "sig_alg": "ed25519",
        "signature": sig.hex(),
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    ok: bool
    claimed: int
    achieved: int
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def achieved_name(self) -> str:
        return LEVEL_NAME[L(self.achieved)]

    def __str__(self) -> str:
        head = "VALID" if self.ok else "REJECTED"
        return f"{head} claimed={LEVEL_NAME[L(self.claimed)]} achieved={self.achieved_name}"


def verify(
    passport: Mapping[str, Any],
    *,
    trusted_pubkeys: Iterable[str] | None = None,
    replay_core: Mapping[str, Any] | None = None,
    attested: bool = False,
    log_inclusion_proof: Any = None,
    current_calibration_hash: str | None = None,
) -> Verdict:
    """Verify a passport and compute the level it ACTUALLY achieves.

    ACHIEVED = min(CLAIMED, INDEPENDENTLY_REVERIFIABLE). A passport may claim L4;
    if this verifier cannot itself re-derive L4 evidence, the achieved level is
    lower and the passport is still *valid* -- it just proves less than it claimed.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    for f in ("core", "digest", "pubkey", "sig_alg", "signature"):
        if f not in passport:
            return Verdict(False, 0, 0, [f"missing field: {f}"], {"schema": False})
    checks["schema"] = True

    if passport["sig_alg"] != "ed25519":
        return Verdict(False, 0, 0, [f"unsupported sig_alg {passport['sig_alg']!r}"], checks)

    core = passport["core"]
    claimed = int(core.get("claimed_level", 0))

    # 1. digest binds the core
    recomputed = digest_obj(core)
    checks["digest"] = recomputed == passport["digest"]
    if not checks["digest"]:
        reasons.append("digest does not match the core (content was altered)")
        return Verdict(False, claimed, 0, reasons, checks)

    # 2. the key is not publicly derivable  -- GENESIS TEST 0
    pub = passport["pubkey"]
    checks["key_not_weak"] = pub not in WEAK_PUBKEYS
    if not checks["key_not_weak"]:
        reasons.append(
            f"signed with a publicly-derivable key ({pub[:16]}...): the signature "
            "proves nothing about origin"
        )

    # 3. signature
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(
            bytes.fromhex(passport["signature"]), bytes.fromhex(passport["digest"])
        )
        checks["signature"] = True
    except (InvalidSignature, ValueError):
        checks["signature"] = False
        reasons.append("signature does not verify")

    # 4. issuer is trusted (optional)
    if trusted_pubkeys is not None:
        checks["trusted_issuer"] = pub in set(trusted_pubkeys)
        if not checks["trusted_issuer"]:
            reasons.append("issuer is not in the trusted set")

    # 5. provenance is present and non-empty -- a passport must say WHICH machine
    prov = core.get("provenance") or {}
    checks["provenance"] = bool(prov.get("provider")) and bool(prov.get("backend"))
    if not checks["provenance"]:
        reasons.append("provenance is missing provider/backend (cannot attest origin)")
    # Completeness is REPORTED, never silently assumed. An incomplete passport
    # is still verifiable -- it just cannot claim full reproducibility, and the
    # verifier says which fields are absent instead of leaving a reader to
    # guess from what is not printed.
    checks["provenance_complete"] = not prov.get("provenance_incomplete")
    if prov.get("provenance_incomplete"):
        reasons.append("provenance INCOMPLETE, missing: "
                       + ", ".join(prov["provenance_incomplete"])
                       + " (verifiable, but not fully reproducible)")

    # 6. calibration expiry
    if core.get("expires_on_calibration_change") and current_calibration_hash is not None:
        bound = prov.get("calibration_hash")
        checks["calibration_current"] = bound == current_calibration_hash
        if not checks["calibration_current"]:
            reasons.append("calibration snapshot has changed since minting (passport expired)")

    valid = checks["digest"] and checks.get("signature", False) and checks["key_not_weak"] \
        and checks["provenance"] and checks.get("trusted_issuer", True) \
        and checks.get("calibration_current", True)

    # ---- the achieved level -------------------------------------------------
    achieved = L.ASSERTED if valid else -1
    if valid:
        if replay_core is not None:
            replayed = digest_obj(_freeze(replay_core))
            checks["replay"] = replayed == passport["digest"]
            if checks["replay"]:
                achieved = L.REPLAY
            else:
                reasons.append("replay produced a different digest (result is not reproducible)")
        if achieved >= L.REPLAY and attested:
            achieved = L.ATTESTED
        if achieved >= L.ATTESTED and log_inclusion_proof is not None:
            achieved = L.LOGGED

    achieved_final = max(0, min(int(claimed), int(achieved))) if valid else 0
    if valid and achieved_final < claimed:
        reasons.append(
            f"claimed {LEVEL_NAME[L(claimed)]} but this verifier could only re-derive "
            f"{LEVEL_NAME[L(achieved_final)]} (ACHIEVED = min(CLAIMED, REVERIFIABLE))"
        )
    return Verdict(valid, claimed, achieved_final, reasons, checks)


def seq_compose(levels: Iterable[int]) -> int:
    """A pipeline is as sound as its weakest step."""
    ls = list(levels)
    return int(min(ls)) if ls else 0
