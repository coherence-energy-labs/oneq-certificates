r"""On-the-wire form for a decoding certificate: exact, bound, and portable.

A certificate is only worth transmitting if a stranger can re-check it without
the machinery that produced it. Three properties make that possible, and all
three are easy to lose:

  1. IT IS BOUND TO ITS GRAPH. A packing that is feasible for one weighted
     decoding graph says nothing about another. Without the graph's fingerprint
     in the signed body, a genuine certificate for an easy graph can be
     replayed as a certificate for a hard one and every arithmetic check still
     passes. `graph_sha256` is therefore part of the document, and re-checking
     recomputes it rather than trusting the field.

  2. ITS NUMBERS ARE EXACT. Decoding weights are IEEE-754 doubles, and a
     decimal round-trip through JSON is not guaranteed to be bit-identical
     across languages -- Python writes `1e-05` where JavaScript writes
     `0.00001`, and a checker that re-parses a slightly different weight can
     refuse a valid certificate or, worse, accept a marginal one. Every float
     here is carried as its 64-bit pattern in hex, which every language reads
     the same way (`f64::from_bits`, `DataView.getFloat64`, `struct.unpack`).
     This is what makes the second-language checker a conformance test rather
     than a tolerance negotiation.

  3. IT SAYS WHAT IT CLAIMS. A refused shot still gets a document -- the
     degraded receipt -- and it is labelled `proven_optimal: false` with its
     gap. A format that could only express success would make silence and
     failure look identical, which is the habit this whole line of work exists
     to break.

WHAT THE SIGNATURE DOES AND DOES NOT DO. It attests who emitted the document,
nothing more. A signature over a false certificate is a signed false
certificate: the mathematics is checked by re-running the checker, and no key
can substitute for that. Signing is reused wholesale from `passport.py` --
same canonical digest, same issuer registry, same refusal to mint under a
publicly-derivable key -- because a second crypto path is a second thing to
get wrong.
"""

from __future__ import annotations

import struct
from typing import Any

from .matching_cert import BOUNDARY, MatchingCertificate, check
from .passport import canonical_bytes, sha256_hex

SCHEMA = "oneq-decoding-certificate/1"


def f2h(x: float) -> str:
    """A double as its exact 64-bit pattern: '0x3ff0000000000000'."""
    return "0x" + struct.pack(">d", float(x)).hex()


def h2f(s: str) -> float:
    if not isinstance(s, str) or not s.startswith("0x") or len(s) != 18:
        raise ValueError(f"not a 64-bit float pattern: {s!r}")
    return struct.unpack(">d", bytes.fromhex(s[2:]))[0]


def canon_edge(e) -> tuple[int, int]:
    u, v = int(e[0]), int(e[1])
    # BOUNDARY IS ALWAYS SECOND. Written boundary-first,
    # u = -1 makes `u <= v` already true, so the swap never
    # fired: (-1, 3) and (3, -1) both survived as distinct
    # keys, the same edge was counted twice, and a valid
    # document was refused with GRAPH MISMATCH -- the exact
    # outcome graph_fingerprint exists to avoid.
    if u == BOUNDARY:
        return (v, u)
    if v == BOUNDARY:
        return (u, v)
    return (u, v) if u <= v else (v, u)


def graph_fingerprint(edges) -> str:
    """sha256 over the decoding graph, canonicalised so it does not depend on
    the order PyMatching happened to enumerate edges in.

    PARALLEL EDGES COLLAPSE TO THE CHEAPEST, matching what both the checker and
    the producer use. If the fingerprint kept duplicates but the checker read
    the minimum, two graphs that behave identically would fingerprint
    differently and valid certificates would be refused for a bookkeeping
    reason.
    """
    best: dict[tuple[int, int], float] = {}
    for u, v, w in edges:
        k = canon_edge((u, v))
        w = float(w)
        if k not in best or w < best[k]:
            best[k] = w
    rows = [[k[0], k[1], f2h(best[k])] for k in sorted(best)]
    return sha256_hex(canonical_bytes({"schema": "oneq-decoding-graph/1",
                                       "edges": rows}))


def to_document(*, edges, syndrome, cert: MatchingCertificate | None,
                proven_optimal: bool, primal: float, bound: float,
                reason: str, graph_sha256: str | None = None,
                context: dict | None = None) -> dict[str, Any]:
    """Serialise one shot's outcome -- certified or degraded."""
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "graph_sha256": graph_sha256 or graph_fingerprint(edges),
        "syndrome": sorted(int(s) for s in syndrome),
        "proven_optimal": bool(proven_optimal),
        "weight": f2h(primal),
        "certified_lower_bound": f2h(bound),
        "optimality_gap_at_most": f2h(max(0.0, primal - bound)),
        "reason": reason,
    }
    if cert is not None:
        doc["correction"] = [list(canon_edge(e)) for e in cert.matched_edges]
        # sorted for determinism: two runs that find the same packing must
        # produce byte-identical documents or the digest is meaningless
        doc["dual"] = [[sorted(int(v) for v in S), f2h(val)]
                       for S, val in sorted(cert.blossom_duals.items(),
                                            key=lambda kv: sorted(kv[0]))]
    if context:
        doc["context"] = context
    return doc


class AmbiguousDocument(ValueError):
    """The document does not denote exactly one certificate."""


def from_document(doc: dict) -> tuple[MatchingCertificate, list[int]]:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"unknown schema {doc.get('schema')!r}")
    if "correction" not in doc or "dual" not in doc:
        raise ValueError("degraded receipt: there is no certificate to re-check")

    # A REPEATED MEMBER SET MAKES THE DOCUMENT AMBIGUOUS, AND THE TWO
    # IMPLEMENTATIONS DISAGREED ABOUT IT.
    #
    # A dual is a MAP from vertex sets to values. A list is only a faithful
    # encoding of one while the keys are distinct. Building a dict from a list
    # with repeats silently keeps the LAST value and drops the rest; summing
    # the list adds them all. On the document
    #
    #     dual = [ [{0}, 0.4], [{0}, 1.0], [{2}, 1.0] ]
    #
    # Python read an objective of 2.0 and ACCEPTED; the JavaScript read 2.4 and
    # refused with DUAL INFEASIBLE. Same bytes, opposite verdicts -- exactly
    # the failure a second implementation exists to surface, and invisible to
    # any number of tests written against one of them.
    #
    # Neither reading is right, because the document does not denote one
    # certificate. So it is refused rather than resolved: a rule that picks a
    # winner would still let a forger publish a dual whose printed entries do
    # not sum to the bound the checker used.
    seen: set[frozenset] = set()
    duals: dict[frozenset, float] = {}
    for S, v in doc["dual"]:
        key = frozenset(int(x) for x in S)
        if key in seen:
            raise AmbiguousDocument(
                f"member set {sorted(key)} appears more than once in the dual; "
                f"a dual is a map from sets to values and this document does "
                f"not denote one")
        seen.add(key)
        duals[key] = h2f(v)

    cert = MatchingCertificate(
        matched_edges=tuple(canon_edge(e) for e in doc["correction"]),
        node_potentials={}, blossom_duals=duals)
    return cert, [int(s) for s in doc["syndrome"]]


def recheck(doc: dict, edges) -> tuple[bool, str]:
    """Re-verify a document against a graph the verifier supplies itself.

    THE GRAPH IS NOT TAKEN FROM THE DOCUMENT. A certificate carries its graph's
    fingerprint so that a mismatch is detectable, but the edges must come from
    the verifier -- if the document could supply both the claim and the graph
    it is checked against, it could supply a graph on which any claim is true.
    """
    fp = graph_fingerprint(edges)
    if doc.get("graph_sha256") != fp:
        return False, (f"GRAPH MISMATCH: document is for {doc.get('graph_sha256')}, "
                       f"this graph is {fp}")
    if not doc.get("proven_optimal"):
        return False, "document does not claim optimality (degraded receipt)"
    try:
        cert, syndrome = from_document(doc)
    except AmbiguousDocument as e:
        return False, f"AMBIGUOUS DOCUMENT: {e}"
    r = check(edges=edges, syndrome=syndrome, cert=cert)
    if not r.accepted:
        return False, r.reason
    # The document's stated weight must be the one that was actually proven,
    # and EXACTLY so. `f2h` stores a double as its 64-bit pattern precisely
    # so the number survives the round trip unchanged; comparing it back with
    # a 1e-9 tolerance threw away the bit-exactness the encoding exists to
    # provide, and let a receipt advertise a weight that is not the weight
    # anything was proven about. The mathematics above is decided exactly, so
    # a slack comparison here was the last place a receipt could drift from
    # its own proof.
    if r.primal_weight is None:
        return False, "checker returned no primal weight to compare against"
    if h2f(doc["weight"]) != r.primal_weight:
        return False, (f"document states weight {h2f(doc['weight'])} but the "
                       f"correction weighs {r.primal_weight}")
    return True, r.reason


def sign_document(doc: dict, private_key_bytes: bytes) -> dict:
    """Attest authorship. Reuses passport.py's signer verbatim."""
    from .passport import mint
    return mint(doc, private_key_bytes=private_key_bytes)


def verify_signature(signed: dict) -> tuple[bool, str]:
    """Check the signature ONLY. The mathematics is `recheck`'s job.

    Kept as a separate call on purpose: a caller that wants "is this true"
    must ask for it, and cannot get a true-looking answer by asking "is this
    signed". Conflating the two is how a signed wrong answer gets deployed.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from .passport import WEAK_PUBKEYS, digest_obj
    for f in ("core", "digest", "pubkey", "sig_alg", "signature"):
        if f not in signed:
            return False, f"missing field: {f}"
    if signed["sig_alg"] != "ed25519":
        return False, f"unsupported sig_alg {signed['sig_alg']!r}"
    if digest_obj(signed["core"]) != signed["digest"]:
        return False, "digest does not match the document (content was altered)"
    if signed["pubkey"] in WEAK_PUBKEYS:
        return False, "signed with a publicly-derivable key: proves nothing"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(signed["pubkey"])).verify(
            bytes.fromhex(signed["signature"]), bytes.fromhex(signed["digest"]))
    except (InvalidSignature, ValueError):
        return False, "signature does not verify"
    return True, "signature verifies"


def graph_to_json(edges) -> dict:
    """The graph as it travels between implementations.

    Weights are carried in the SAME exact 64-bit encoding as the certificate's.
    A decimal number here would reintroduce precisely the round-trip the format
    exists to eliminate, and it would do so where a discrepancy is hardest to
    read: a fingerprint mismatch looks like "wrong graph", not "wrong parser".
    """
    best: dict[tuple[int, int], float] = {}
    for u, v, w in edges:
        k = canon_edge((u, v))
        if k not in best or float(w) < best[k]:
            best[k] = float(w)
    return {"schema": "oneq-decoding-graph/1",
            "edges": [[k[0], k[1], f2h(best[k])] for k in sorted(best)]}


def graph_from_json(doc) -> list[tuple[int, int, float]]:
    rows = doc["edges"] if isinstance(doc, dict) else doc
    return [(int(u), int(v), h2f(w)) for u, v, w in rows]
