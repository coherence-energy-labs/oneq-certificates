r"""Seeds for an anchored preregistration (docs/PREREG_ANCHORING.md), from a drand round declared before it existed.

A separate module ON PURPOSE. These two functions extend `oneq.beacon` (the August 2026 blind-schedule beacon), but
three run records pin every module of `src/oneq` by sha256 as the code their runs used
(evidence/cnot_schedule/20261022/comparison.json and evidence/task_resource_frontier/20261021/{plan,batch_benchmark}
.json): editing beacon.py broke all three pins (tools/hash_pin_audit.py, 2026-11-07), so beacon.py is left byte for
byte as those runs had it and its HMAC stream is reused from here.

Pure: no network, clock or filesystem. `tools/prereg_anchor.py` does the fetching.
"""
from __future__ import annotations

from oneq.beacon import DRAND_PERIOD_S, _stream

SEED_DOMAIN = "oneq-prereg-seed/1"


def round_time(round_number: int, *, genesis_time: int, period_s: int = DRAND_PERIOD_S) -> int:
    """The Unix time at which a drand round is published: round 1 at genesis, one round per period (drand's
    specification). A reader needs no clock to know WHEN a round's value became public, only the chain's published
    genesis time and period."""
    if round_number < 1 or period_s <= 0:
        raise ValueError(f"round must be >= 1 and period > 0, got {round_number}, {period_s}")
    return genesis_time + (round_number - 1) * period_s


def derive_prereg_seeds(randomness_hex: str, *, prereg_sha256: str, n: int, domain: str = SEED_DOMAIN) -> list[int]:
    """The seeds of a preregistered run, as a pure function of a beacon round declared BEFORE the round was published
    and of the preregistration's exact bytes.

    The preregistration's sha256 and the count are folded into the stream's domain, for the reason
    `oneq.beacon._bound_domain` gives: a seed list that did not pin what it was derived for could be re-labelled.
    Changing one byte of the preregistration, or asking for one more seed, changes every seed.
    """
    if not isinstance(randomness_hex, str) or len(randomness_hex) < 32:
        raise ValueError("beacon randomness must be a hex string of at least 16 bytes")
    try:
        bytes.fromhex(randomness_hex)
    except ValueError as e:
        raise ValueError(f"beacon randomness is not hex: {e}") from e
    if not (isinstance(prereg_sha256, str) and len(prereg_sha256) == 64
            and all(c in "0123456789abcdef" for c in prereg_sha256)):
        raise ValueError("prereg_sha256 must be 64 lowercase hex characters")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    gen = _stream(randomness_hex, f"{domain}|prereg={prereg_sha256}|n={n}")
    return [next(gen) for _ in range(n)]
