r"""Public randomness: a blind schedule nobody involved could have chosen.

THE WEAKNESS THIS CLOSES. Arm 3's injection schedule is seeded and
committed by hash before decoding, which is sound against tuning the
pipeline to the schedule -- but WE choose the seed. A reader has to take
on trust that we did not roll it repeatedly until we liked the shape.
`docs/HARDWARE_READINESS_PROGRAM.md` Phase 5 names the fix: a public
randomness beacon, "also closes the self-referential preregistration
weakness".

THE ORDERING IS THE WHOLE MECHANISM, and it only works one way round:

    1. the freeze declares a FUTURE beacon round N and is committed
    2. round N is published later, by drand, on its own schedule
    3. the schedule is DERIVED from round N's randomness, deterministically
    4. a reader fetches round N from any drand mirror and re-derives it

At step 1 the randomness does not exist yet, so it cannot have been
chosen; at step 4 it is public, so the derivation cannot be faked. Using
the LATEST round instead of a future one would destroy the property
entirely -- we would be choosing a value we can already see -- so
`declare_future_round` exists to make the correct move the easy one.

THIS MODULE IS PURE. No network, no clock, no filesystem: fetching is
`tools/fetch_beacon.py`'s job, and keeping them apart is what lets a
stranger re-derive a schedule offline from the pinned round alone.

CEILING, NAMED: this proves the schedule follows from a specific beacon
round. It does not prove drand itself is honest. That is why the round
number is declared in advance and the value is fetched by the READER
from a source we do not control -- our copy of it is a convenience, never
the authority.
"""

from __future__ import annotations

import hashlib
import hmac
from fractions import Fraction

#: drand's mainnet "quicknet" cadence, seconds per round. Used only to
#: turn a lead TIME into a lead in ROUNDS; never to compute a deadline.
DRAND_PERIOD_S = 3

_U64 = 1 << 64


def declare_future_round(current_round: int, *, lead_seconds: int = 3600,
                         period_s: int = DRAND_PERIOD_S) -> int:
    """The round to commit to: far enough ahead that it cannot be known.

    Refuses a non-positive lead outright. A "future" round that is
    already published is the one mistake that silently voids the entire
    construction, and it would look identical in every artifact.
    """
    if current_round < 1:
        raise ValueError(f"current_round must be >= 1, got {current_round}")
    if lead_seconds <= 0:
        raise ValueError(
            "lead_seconds must be positive: committing to the CURRENT or a "
            "PAST round means committing to a value we can already see, "
            "which is exactly the self-reference the beacon exists to close")
    if period_s <= 0:
        raise ValueError(f"period_s must be positive, got {period_s}")
    return current_round + max(1, -(-lead_seconds // period_s))


def _stream(randomness_hex: str, domain: str):
    """Deterministic uint64 stream keyed by the beacon value.

    HMAC in counter mode rather than a PRNG: it is specified in one line,
    reproducible in any language, and carries no library-version
    dependence that could make a reader's re-derivation differ from ours.
    """
    key = bytes.fromhex(randomness_hex)
    counter = 0
    while True:
        block = hmac.new(key, f"{domain}:{counter}".encode(),
                         hashlib.sha256).digest()
        for off in range(0, 32, 8):
            yield int.from_bytes(block[off:off + 8], "big")
        counter += 1


def _bound_domain(domain: str, shots: int, q: Fraction,
                  n_mechanisms: int) -> str:
    """Fold EVERY declared parameter into the stream's domain.

    Found by its own test: a 400-shot schedule verified against a claimed
    401 shots, because shot 400 simply had not been selected, so the two
    derivations coincided. The schedule therefore did not pin its own
    parameters -- a reader could restate the shot count and still see
    agreement. Binding them here makes any restatement change every draw,
    so the schedule is self-pinning rather than relying on the freeze to
    carry the numbers honestly.
    """
    return f"{domain}|shots={shots}|q={q.numerator}/{q.denominator}" \
           f"|mech={n_mechanisms}"


def _validate(randomness_hex: str, shots: int, q) -> Fraction:
    if not isinstance(randomness_hex, str) or len(randomness_hex) < 32:
        raise ValueError("beacon randomness must be a hex string of at "
                         "least 16 bytes")
    try:
        bytes.fromhex(randomness_hex)
    except ValueError as e:
        raise ValueError(f"beacon randomness is not hex: {e}") from e
    if shots < 0:
        raise ValueError(f"shots must be >= 0, got {shots}")
    # Fraction(0.1) is NOT 1/10. A freeze declaring q = 0.1 and a reader
    # re-deriving with Fraction(1, 10) would compute DIFFERENT schedules
    # and the reader would see a refusal they could not explain -- the
    # binary expansion of a decimal literal is not the decimal. So an
    # inexact float is refused at the door rather than silently adopted.
    if isinstance(q, float) and not q.is_integer():
        raise ValueError(
            f"q={q!r} is a float: Fraction({q!r}) is its binary expansion, "
            f"not the decimal you wrote, so two honest parties would derive "
            f"different schedules. Pass a Fraction or a string like '1/4'.")
    qf = Fraction(q)
    if not (0 <= qf <= 1):
        raise ValueError(f"q must lie in [0, 1], got {qf}")
    return qf


def derive_schedule(randomness_hex: str, *, shots: int, q,
                    n_mechanisms: int, domain: str = "oneq-phase6-inject/1"):
    """The injection schedule, as a pure function of public randomness.

    Returns a list of {"shot": i, "mech": k}. `q` is compared EXACTLY as
    a rational against a uint64 draw -- no float threshold, so two
    implementations cannot disagree at the boundary the way a `<` on
    doubles can.
    """
    qf = _validate(randomness_hex, shots, q)
    if n_mechanisms < 1:
        raise ValueError(f"n_mechanisms must be >= 1, got {n_mechanisms}")
    threshold = int(qf * _U64)          # draw < threshold  <=>  p = q
    gen = _stream(randomness_hex,
                  _bound_domain(domain, shots, qf, n_mechanisms))
    out = []
    for i in range(shots):
        pick = next(gen) < threshold
        mech = next(gen) % n_mechanisms   # drawn ALWAYS, so the stream
        if pick:                          # position never depends on q
            out.append({"shot": i, "mech": mech})
    return out


def verify_schedule(schedule, randomness_hex: str, *, shots: int, q,
                    n_mechanisms: int,
                    domain: str = "oneq-phase6-inject/1") -> tuple[bool, str]:
    """Re-derive and compare. Returns (ok, reason).

    This is the check a reader runs: it answers "does this schedule follow
    from that public beacon round", which is the only question that makes
    the blinding meaningful.
    """
    try:
        expect = derive_schedule(randomness_hex, shots=shots, q=q,
                                 n_mechanisms=n_mechanisms, domain=domain)
    except ValueError as e:
        return False, f"refused: {e}"
    got = [{"shot": int(e["shot"]), "mech": int(e["mech"])}
           for e in schedule]
    if got == expect:
        return True, f"schedule re-derives from the beacon ({len(expect)} injections)"
    if len(got) != len(expect):
        return False, (f"schedule has {len(got)} injections, the beacon "
                       f"derives {len(expect)}")
    first = next(i for i, (a, b) in enumerate(zip(got, expect)) if a != b)
    return False, (f"schedule diverges from the beacon at entry {first}: "
                   f"{got[first]} vs {expect[first]}")
