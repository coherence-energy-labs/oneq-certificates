r"""OpenTimestamps proofs, executed: the digest each attestation commits to, with no client and no network.

WHY THIS EXISTS. `tools/verify_release.py` reads which Bitcoin blocks a proof NAMES (ledger R231), and says plainly
that it does not check them: "confirming that the block's merkle root is the proof's final digest needs a Bitcoin
node (`ots verify`) or a block explorer". Checking a release that way was done by hand, with a scratch script piping
`ots -v info` (R237). An anchored preregistration (`tools/prereg_anchor.py`, docs/PREREG_ANCHORING.md) needs the same
check as a step a stranger runs, so the proof is executed here: every operation from the file's digest down to each
attestation, giving the 32 bytes that must equal that block's merkle root. Comparing them with a block header is the
caller's job (an explorer, or a node); this module is pure -- no network, clock or filesystem.

THE FORMAT (python-opentimestamps, core/serialize.py, core/timestamp.py, core/op.py, core/notary.py), as written by
every proof this repository has made:

    detached file   = MAGIC, varuint major version (1), file-hash op tag, digest, timestamp
    timestamp       = (0xff item)* item              -- 0xff marks "another branch follows"
    item            = 0x00 attestation | op-tag [varbytes argument] timestamp
    attestation     = 8-byte notary tag, varbytes payload
    Bitcoin payload = varuint block height;  pending (calendar) payload = varbytes URI

Operations: 0x08 sha256, 0x02 sha1, 0x03 ripemd160, 0x67 keccak256 (refused: hashlib has no keccak, and no proof
here uses it), 0xf0 append, 0xf1 prepend, 0xf2 reverse, 0xf3 hexlify. A Bitcoin attestation's digest is the merkle
root in the block header's INTERNAL byte order, which block explorers print reversed.

CEILING, NAMED: this proves that the proof's operations lead from the file's digest to these 32 bytes. That those
bytes are the merkle root of the block at that height, on the chain with the most work, is what the comparison
with a block source establishes, and only to the extent that source is honest; two independent explorers agreeing
is the check `tools/prereg_anchor.py` makes, a full node is better.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
BITCOIN_TAG = bytes.fromhex("0588960d73d71901")
PENDING_TAG = bytes.fromhex("83dfe30d2ef90c8e")
_MAX_MSG = 4096            # python-opentimestamps refuses longer intermediate results; so do we
_MAX_DEPTH = 256           # and deeper recursion


class ProofError(ValueError):
    """The bytes are not an OpenTimestamps proof this module can execute."""


@dataclass(frozen=True)
class Attestation:
    kind: str              # "bitcoin", "pending" or "unknown:<tag hex>"
    height: int | None     # Bitcoin block height, for kind == "bitcoin"
    uri: str | None        # calendar URI, for kind == "pending"
    digest: bytes          # the message this attestation commits to

    @property
    def merkle_root_display(self) -> str:
        """The digest as a block explorer prints a merkle root (byte-reversed hex)."""
        return self.digest[::-1].hex()


class _Reader:
    def __init__(self, buf: bytes):
        self.buf, self.i = buf, 0

    def byte(self) -> int:
        if self.i >= len(self.buf):
            raise ProofError("proof truncated")
        b = self.buf[self.i]
        self.i += 1
        return b

    def take(self, n: int) -> bytes:
        if n < 0 or self.i + n > len(self.buf):
            raise ProofError("proof truncated")
        out = self.buf[self.i:self.i + n]
        self.i += n
        return out

    def varuint(self) -> int:
        value, shift = 0, 0
        while True:
            b = self.byte()
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                return value
            shift += 7
            if shift > 63:
                raise ProofError("varuint too long")

    def varbytes(self, limit: int = _MAX_MSG) -> bytes:
        n = self.varuint()
        if n > limit:
            raise ProofError(f"varbytes of {n} bytes exceeds {limit}")
        return self.take(n)


def _apply(tag: int, msg: bytes, r: _Reader) -> bytes:
    if tag == 0x08:
        out = hashlib.sha256(msg).digest()
    elif tag == 0x02:
        out = hashlib.sha1(msg).digest()
    elif tag == 0x03:
        try:
            out = hashlib.new("ripemd160", msg).digest()
        except ValueError as exc:            # OpenSSL 3 builds may omit it; say so rather than guess
            raise ProofError("ripemd160 is not available in this Python's hashlib") from exc
    elif tag == 0x67:
        raise ProofError("keccak256 operations are not supported (hashlib has no keccak)")
    elif tag == 0xF0:
        out = msg + r.varbytes()
    elif tag == 0xF1:
        out = r.varbytes() + msg
    elif tag == 0xF2:
        out = msg[::-1]
    elif tag == 0xF3:
        out = msg.hex().encode()
    else:
        raise ProofError(f"unknown operation tag 0x{tag:02x}")
    if len(out) > _MAX_MSG:
        raise ProofError("an intermediate message exceeds 4096 bytes")
    return out


def _attestation(r: _Reader, msg: bytes) -> Attestation:
    tag = r.take(8)
    payload = _Reader(r.varbytes(limit=8192))
    if tag == BITCOIN_TAG:
        height = payload.varuint()
        kind, uri = "bitcoin", None
    elif tag == PENDING_TAG:
        height, kind = None, "pending"
        uri = payload.varbytes(limit=1000).decode("utf-8", errors="replace")
    else:
        return Attestation(f"unknown:{tag.hex()}", None, None, msg)
    if payload.i != len(payload.buf):
        raise ProofError(f"{kind} attestation payload has trailing bytes")
    return Attestation(kind, height, uri, msg)


def _timestamp(r: _Reader, msg: bytes, depth: int, out: list[Attestation]) -> None:
    if depth > _MAX_DEPTH:
        raise ProofError("proof nests deeper than 256 levels")
    tag = r.byte()
    while tag == 0xFF:
        _item(r, r.byte(), msg, depth, out)
        tag = r.byte()
    _item(r, tag, msg, depth, out)


def _item(r: _Reader, tag: int, msg: bytes, depth: int, out: list[Attestation]) -> None:
    if tag == 0x00:
        out.append(_attestation(r, msg))
    else:
        _timestamp(r, _apply(tag, msg, r), depth + 1, out)


def parse_detached(proof: bytes) -> tuple[bytes, list[Attestation]]:
    """(the sha256 the proof is about, every attestation with the digest it commits to).

    Refuses anything but a complete sha256 detached proof: trailing bytes, an unknown operation, a truncation. A
    proof that parses only partly would report fewer attestations than it carries, which reads as "not yet
    attested" -- the silent failure R231 recorded.
    """
    if not proof.startswith(MAGIC):
        raise ProofError("not an OpenTimestamps proof (magic bytes differ)")
    r = _Reader(proof)
    r.i = len(MAGIC)
    if r.varuint() != 1:
        raise ProofError("unsupported proof version")
    if r.byte() != 0x08:
        raise ProofError("not a sha256 detached proof")
    digest = r.take(32)
    out: list[Attestation] = []
    _timestamp(r, digest, 0, out)
    if r.i != len(proof):
        raise ProofError(f"{len(proof) - r.i} trailing bytes after the timestamp")
    return digest, out


def bitcoin_attestations(proof: bytes) -> list[Attestation]:
    """The Bitcoin attestations of a detached proof, lowest block first."""
    return sorted((a for a in parse_detached(proof)[1] if a.kind == "bitcoin"), key=lambda a: a.height)
