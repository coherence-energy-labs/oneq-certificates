r"""L0 -- configuration integrity at CYCLE granularity. The real ABFT.

ONE_Q_MAX Sec 3.7.4. The field has thought hard about whether the decoder
is fast enough and never about whether it is RIGHT, and the fault class
that matters most is the dullest one: the decoder's own configuration.

    precompute, in separately-addressed, separately-ECC'd memory:
        tag[e] = H(e || w_e || mask_e || endpoints_e)
    during decode, every edge read already touches its entry:
        h ^= tag_recomputed_from_what_was_read
    at commit:
        h == XOR over the same active set of tag_ref[e] ?

WHY P1 IS THE HEADLINE. One bit flipped in one edge's OBSERVABLE MASK at
d=5 takes p_L from 1.4e-4 to 2.5e-3 -- 17.9x -- and it is
DISTANCE-INDEPENDENT, because it is a systematic mislabeling of WHICH
LOGICAL CLASS an error belongs to rather than a distance-limited event.
`H x_hat = s` catches it in 0 of 16,000 shots, because a mask flip
changes no edge weight, so the correction is unchanged, so a check on the
correction is looking at an object the fault never touched.

WHY 1 CYCLE MATTERS. The industrial baseline is CRAM scrubbing at
10-100 ms detection latency: 9,091-90,909 exposed cycles at a 1.1 us QEC
cycle, every one of them carrying the 17.9x multiplier. **The exposure
window is the problem; area is not.** This does not replace scrubbing --
it closes scrubbing's window.

*** THE TWO WAYS TO BUILD THIS SO IT DETECTS NOTHING. ***

  ACCUMULATE THE STORED TAG INSTEAD OF RECOMPUTING IT. If the decode
  loop does `h ^= tag_ref[e]`, the commit check compares the tag table
  against itself and passes for every corruption of the working table.
  It looks identical, it costs the same, and it is a no-op. The tag must
  be recomputed FROM THE BYTES THE DECODER ACTUALLY READ. Sec 3.7.4's
  own pseudocode reads `h <- h XOR tag[e]`, which is this trap; the
  departure is deliberate and recorded.

  USE A PARITY ACCUMULATOR AT ALL. The first version of this module XORed
  the recomputed tag into a register and compared at commit. That is
  cancellation-prone BY CONSTRUCTION: a persistently corrupted edge read
  TWICE gives t' XOR t' = 0 against a reference of t XOR t = 0, and the
  corruption commits silently. Measured -- detected at one read, missed
  at two -- and invisible to an exhaustive 3,840-corruption sweep because
  every active list in that corpus held unique edge ids. Deduplicating
  would have fixed the instance and left the class. The accumulator is
  now a STICKY COMPARATOR, which cannot cancel and which knows at the
  READ rather than at the commit.

  LET A REQUESTED CHECK SILENTLY NOT RUN. `check_mask_replica=True` with
  `masks_read=None` used to skip the replica comparison and still report
  "verified against the provisioned table" -- and both are DEFAULTS, so
  the default call did no mask check while saying it had. A requested
  check that cannot run is a refusal.

  LET THE MASK RIDE ON THE PER-EDGE TAG ALONE. A matching decoder reads
  weights and endpoints while searching; it reads the OBSERVABLE MASK at
  commit, to compute kappa(x_hat) = L . x_hat. So "every edge read
  already touches its entry" does not cover the mask on the edges the
  search never expanded -- and the mask is the highest-consequence field
  in the table. It therefore gets a DEDICATED, INDEPENDENTLY STORED,
  INDEPENDENTLY HASHED replica, checked in full at commit. ~1.25 kB at
  d=11: the highest consequence and the cheapest thing in the table to
  duplicate. Sec 3.7.4 states it plainly -- without the replica the check
  is vacuous against exactly the class it targets -- and
  `tests/test_abft_l0.py` measures that vacuity rather than quoting it.

WHAT THIS DOES NOT COVER. P2 (transient algorithm state) and P3 (the
output register after commit) are other layers' domains; L0 is
configuration and binary integrity, P1 and P6. Naming the domain is not
modesty, it is the point: a check sold outside its proven domain is how
`H x_hat = s` came to be trusted against a fault class it provably cannot
see.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

#: Tag width. 32 bits of state and one 32-bit XOR per edge already being
#: read -- the honest cost, quoted from the section rather than rounded
#: down. A wider tag buys a smaller collision probability and nothing
#: else; the failure mode this guards is a bit flip, not an adversary.
TAG_BITS = 32
_TAG_MASK = (1 << TAG_BITS) - 1


class ConfigurationFault(Exception):
    """The decode path read bytes the provisioned table does not contain."""


@dataclass(frozen=True)
class Edge:
    """One decoding-graph edge as the decoder reads it."""
    eid: int
    weight: int                    # integer: a float in a hashed tag is
    mask: int                      # a value two implementations dispute
    endpoints: tuple[int, int]

    def tag(self) -> int:
        """H(e || w_e || mask_e || endpoints_e), truncated to TAG_BITS.

        Recomputed from the FIELDS AS READ every time. That is the whole
        mechanism: a stored tag XORed into the accumulator would compare
        the tag table with itself.
        """
        body = (f"{self.eid}|{self.weight}|{self.mask}|"
                f"{self.endpoints[0]},{self.endpoints[1]}").encode("ascii")
        return int.from_bytes(hashlib.sha256(body).digest()[:4], "big") & _TAG_MASK


@dataclass
class ProvisionedTable:
    """The reference: tags and the mask replica, separately addressed.

    "Separately addressed, separately ECC'd" is modelled here as a
    distinct object holding DERIVED values -- the tags and a copy of the
    masks. It is never mutated by the corruption helpers, which is the
    modelling equivalent of the memory being in a different device.
    """
    tags: dict[int, int]
    #: THE DEDICATED REPLICA. Masks only: highest consequence, cheapest
    #: to duplicate, and the one field the decode path may not read on
    #: every edge it touches.
    mask_replica: dict[int, int]

    @classmethod
    def provision(cls, edges) -> "ProvisionedTable":
        edges = list(edges)
        if not edges:
            raise ConfigurationFault(
                "provisioning an empty table would make every commit check "
                "pass over nothing")
        return cls(tags={e.eid: e.tag() for e in edges},
                   mask_replica={e.eid: e.mask for e in edges})


@dataclass
class Accumulator:
    """One sticky bit, set at the READ that first disagrees.

    NOT a parity accumulator. The XOR form this replaced cancelled on a
    persistently corrupted edge read twice -- t' XOR t' = 0 on the
    accumulator and t XOR t = 0 on the reference -- so the corruption
    committed silently. A sticky OR cannot cancel: a bit that is set is
    never cleared by a later read, whatever its multiplicity and whatever
    the other deltas are.

    It also detects at the read rather than at the commit, which is the
    one-cycle latency this layer claims. The parity form could only ever
    have known at commit.
    """
    table: "ProvisionedTable"
    fault_seen: bool = False
    first_fault: int | None = None
    touched: list[int] = field(default_factory=list)

    def read(self, edge: Edge) -> Edge:
        """What the decoder calls instead of a bare table read.

        Returns the edge so the call site cannot skip the check and still
        get its data -- the failure that would leave a subset of reads
        uncovered while every test still passed.
        """
        self.touched.append(edge.eid)
        ref = self.table.tags.get(edge.eid)
        # An edge the provisioned table does not describe is P6: a stale
        # binary reading a graph it was not built for. Caught HERE, at the
        # read, rather than inferred later from a reference that silently
        # omitted it.
        if ref is None or edge.tag() != ref:
            if not self.fault_seen:
                self.first_fault = edge.eid
            self.fault_seen = True
        return edge


def commit_check(acc: Accumulator, table: ProvisionedTable,
                 masks_read: dict[int, int] | None = None,
                 check_mask_replica: bool = True) -> tuple[bool, str]:
    """1-cycle verdict at commit. (ok, why).

    `masks_read` is what the commit stage actually used to compute
    kappa(x_hat) = L . x_hat. It is compared against the dedicated
    replica IN FULL, because those reads are not covered by the per-edge
    accumulation.

    `check_mask_replica=False` exists to MEASURE the vacuity the section
    asserts, not as an option anyone should ship. It is named in the
    verdict string so a run that used it cannot be quoted as if it had
    not.
    """
    if not acc.touched:
        return False, ("no edge was read, so nothing was compared and the "
                       "check would pass vacuously")
    if acc.table is not table:
        return False, ("the accumulator was built against a DIFFERENT "
                       "provisioned table than the one presented at commit; "
                       "comparing the two would certify neither")
    if acc.fault_seen:
        return False, (f"configuration fault at edge {acc.first_fault}: the "
                       f"bytes read do not hash to the provisioned tag "
                       f"(detected at the read, {len(acc.touched)} reads in "
                       f"this commit)")
    # A REQUESTED CHECK THAT CANNOT RUN IS A REFUSAL, NOT A PASS. Asking
    # for the replica check and supplying no masks used to skip it in
    # silence and still report "verified" -- and both arguments defaulted
    # that way, so the DEFAULT call performed no mask check at all while
    # saying it had.
    if check_mask_replica and masks_read is None:
        return False, ("the observable-mask replica check was requested and "
                       "no masks were supplied, so it cannot run. Pass the "
                       "masks the commit stage actually read, or set "
                       "check_mask_replica=False and accept that the P1 "
                       "class is uncovered")
    if check_mask_replica and masks_read is not None:
        for eid, mask in masks_read.items():
            if eid not in table.mask_replica:
                return False, f"edge {eid} has no mask replica"
            if table.mask_replica[eid] != mask:
                return False, (
                    f"observable mask for edge {eid} disagrees with its "
                    f"dedicated replica ({mask} vs "
                    f"{table.mask_replica[eid]}) -- this is the P1 class, "
                    f"17.9x at d=5 and distance-independent")
    suffix = "" if check_mask_replica else " (MASK REPLICA CHECK DISABLED)"
    return True, (f"{len(acc.touched)} edge reads verified against the "
                  f"provisioned table{suffix}")


#: The declared fixed-point scale for weights entering a tag.
#:
#: `Edge.weight` is an int on purpose -- "a float in a hashed tag is a
#: value two implementations dispute". Real decoding graphs carry floats,
#: so the conversion has to happen somewhere, and somewhere is HERE,
#: once, so the provisioning side and the reading side cannot drift
#: apart. A repository that already paid for one shared predicate living
#: in four copies does not need a fifth.
WEIGHT_SCALE = 10 ** 6


def edges_for(edges, masks, *, scale: int = WEIGHT_SCALE):
    r"""`(u, v, w)` rows + a mask map -> the `Edge` list L0 hashes.

    ONE function for both sides of the comparison. Provisioning calls it
    on the trusted configuration; the decode path calls it on the bytes
    it actually read. If the quantisation lived at two call sites, a
    change to one would look exactly like a corruption -- and the whole
    mechanism is a comparison of two hashes.

    `masks[key]` is the OBSERVABLE MASK: whether this edge flips the
    logical class. It is the P1 field -- one bit of it at d=5 is 17.9x
    and distance-independent -- and it is REQUIRED rather than defaulted
    to zero, because a mask map that silently defaulted would provision
    a replica of all-zeros and then verify the decode against it.
    """
    # THE ESTATE'S CANONICAL KEY, NOT A LOCAL (min, max). Writing one
    # here produced `(-1, 0)` where every other module says `(0, -1)`:
    # BOUNDARY is always SECOND, a rule `logical_cut._key` carries with
    # the story of the run it once broke. Two keying conventions for one
    # edge is the same defect class the tag mechanism exists to catch,
    # arriving through the front door.
    from .logical_cut import _key

    out = []
    for i, (u, v, w) in enumerate(edges):
        key = _key(int(u), int(v))
        if key not in masks:
            raise ConfigurationFault(
                f"edge {key} has no observable mask. Defaulting it to 0 "
                f"would provision an all-zero replica and then verify the "
                f"decode against it, which is the P1 class certifying "
                f"itself")
        out.append(Edge(eid=i, weight=int(round(float(w) * scale)),
                        mask=int(masks[key]), endpoints=key))
    return out


def exposure_cycles(detect_latency_s: float, cycle_s: float = 1.1e-6) -> int:
    """How many QEC cycles run corrupted before a detector notices.

    The number the section is actually about. CRAM scrubbing at 10-100 ms
    leaves 9,091-90,909 exposed cycles at a 1.1 us cycle; this scheme
    leaves ONE. Rounded DOWN, because an exposure count that rounds up
    flatters the thing being replaced.
    """
    if cycle_s <= 0:
        raise ConfigurationFault("a non-positive cycle time has no exposure")
    return int(detect_latency_s // cycle_s)
