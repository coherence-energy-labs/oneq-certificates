r"""Streaming certificate emission: ordered receipts with a bounded, HONEST lag.

Lane 3's first deliverable (docs/REALTIME_EMISSION_DESIGN.md). The decoder is
never stalled by certification; receipts flow beside the corrections with
three properties the batch path cannot give:

  ORDERED    receipts are emitted in shot order regardless of completion
             order, with a WATERMARK: `certified_through` = the highest shot
             id k such that every shot <= k has its final receipt.
  BOUNDED    at most `max_pending` shots are in flight; submission blocks
             (backpressure) rather than letting the queue grow into the
             OOM class the million-shot run already documented.
  HONEST     a shot whose certificate misses the lag budget is emitted as a
             DEFERRED receipt -- recorded, never skipped (a skip is a dark
             gate) -- and its certificate, when it lands, goes to a side
             channel named `late`, where it cannot be mistaken for having
             met the clock.

The producer pool recycles workers every `recycle_every` shots. Two measured
failure modes bound this knob from both sides: recycle at 1 and every shot
pays a full process spawn -- MEASURED at p50 2.5 s/receipt on a d3 config
whose batch cost is ~15 ms, spawn-bound not solve-bound; never recycle and
HiGHS's per-solve retention eventually kills a long-lived worker on ~2 MiB
allocations with 15 GB free. The default (100) bounds the leak to ~100
solves' retention while amortizing the spawn to noise.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from dataclasses import dataclass, field

from .certifying_decoder import DEFAULT_LADDER, CertifyingDecoder

_worker_decoder: CertifyingDecoder | None = None


def _init_worker(edges, ladder, max_family, max_candidates):
    global _worker_decoder
    _worker_decoder = CertifyingDecoder(edges, ladder=ladder,
                                        max_family=max_family,
                                        max_candidates=max_candidates)


def _certify_one(job):
    shot_id, syndrome, correction = job
    c = _worker_decoder.certify(syndrome, correction)
    return (shot_id, c.accepted, c.rung, c.primal, c.bound, c.gap,
            c.cert, c.reason)


#: THE STREAMING LANE'S VOCABULARY, DECLARED RATHER THAN COMMENTED.
#:
#: It is NOT the vocabulary `oneq.api.Receipt` speaks, and the overlap is
#: the dangerous part: "degraded" means the same thing in both, so it was
#: silently graded by `soundness_budget.leg_for`, while "certified" -- the
#: same acceptance, from the same checker -- was REFUSED as an unknown
#: producer. One word of three matching by coincidence is worse than none
#: matching, because it makes the lane look graded.
#:
#: A frozenset instead of a comment so `soundness_budget` can be checked
#: against it: a fourth word added here fails
#: `test_every_streaming_tier_is_accounted_for` rather than quietly
#: becoming ungradeable.
STREAM_TIERS = frozenset({"certified", "degraded", "deferred"})


@dataclass
class StreamReceipt:
    """One shot's receipt, exactly one of three named tiers."""

    shot_id: int
    tier: str                       # one of STREAM_TIERS
    rung: int = -1
    primal: float = 0.0
    bound: float = 0.0
    gap: float = 0.0
    cert: object = None             # MatchingCertificate when certified
    reason: str = ""


@dataclass
class StreamStats:
    emitted: int = 0
    certified: int = 0
    degraded: int = 0
    deferred: int = 0
    certified_through: int = -1     # the watermark
    late: list = field(default_factory=list)   # (shot_id, StreamReceipt)


class StreamingCertifier:
    """Certify a shot stream with ordered emission and explicit backpressure.

    `run(shots)` consumes an iterable of (syndrome, correction) pairs and
    YIELDS StreamReceipt in shot order as they resolve, so a consumer sees
    receipt k only after receipts 0..k-1 -- the property that makes the
    watermark meaningful. `lag_budget_s=None` disables deferral (pure
    throughput mode: every receipt waits as long as it takes).
    """

    def __init__(self, edges, *, ladder=DEFAULT_LADDER, workers: int = 2,
                 max_family: int = 600, max_candidates: int = 3000,
                 max_pending: int = 64, lag_budget_s: float | None = None,
                 recycle_every: int = 100):
        self.edges = [(int(u), int(v), float(w)) for u, v, w in edges]
        self.ladder = tuple(ladder)
        self.max_family = int(max_family)
        self.max_candidates = int(max_candidates)
        self.workers = max(1, int(workers))
        self.max_pending = max(1, int(max_pending))
        self.lag_budget_s = lag_budget_s
        self.recycle_every = max(1, int(recycle_every))
        self.stats = StreamStats()

    def _receipt(self, res) -> StreamReceipt:
        (shot_id, accepted, rung, primal, bound, gap, cert, reason) = res
        tier = "certified" if accepted else "degraded"
        return StreamReceipt(shot_id=shot_id, tier=tier, rung=rung,
                             primal=primal, bound=bound, gap=gap,
                             cert=cert, reason=reason)

    def run(self, shots):
        """Yield ordered receipts for (syndrome, correction) pairs."""
        st = self.stats = StreamStats()
        with mp.Pool(self.workers, initializer=_init_worker,
                     initargs=(self.edges, self.ladder, self.max_family,
                               self.max_candidates),
                     maxtasksperchild=self.recycle_every) as pool:
            pending: deque = deque()      # (shot_id, AsyncResult), shot order
            overdue: dict[int, object] = {}   # deferred but still running
            final: set[int] = set()       # shots whose FINAL receipt exists
            shot_iter = enumerate(shots)
            exhausted = False

            def advance_watermark():
                # `certified_through` = highest k with every shot <= k FINAL.
                # A deferred receipt is emitted but not final, so the
                # watermark stalls at the deferral until its late result
                # lands -- the whole point of having a watermark.
                w = st.certified_through
                while (w + 1) in final:
                    w += 1
                    final.discard(w)      # compacted into the watermark
                st.certified_through = w

            def submit_up_to_cap():
                nonlocal exhausted
                while not exhausted and len(pending) < self.max_pending:
                    try:
                        i, (syndrome, correction) = next(shot_iter)
                    except StopIteration:
                        exhausted = True
                        return
                    pending.append(
                        (i, pool.apply_async(_certify_one,
                                             ((i, syndrome, correction),))))

            def harvest_late():
                for sid in [s for s, ar in overdue.items() if ar.ready()]:
                    st.late.append((sid, self._receipt(overdue.pop(sid).get())))
                    final.add(sid)

            submit_up_to_cap()
            while pending:
                shot_id, ar = pending.popleft()
                try:
                    res = ar.get(self.lag_budget_s) \
                        if self.lag_budget_s is not None else ar.get()
                    r = self._receipt(res)
                except mp.TimeoutError:
                    overdue[shot_id] = ar
                    r = StreamReceipt(
                        shot_id=shot_id, tier="deferred",
                        reason=f"certificate exceeded the lag budget of "
                               f"{self.lag_budget_s}s; recorded as a "
                               f"watermark gap, resolution goes to `late`")
                if r.tier == "certified":
                    st.certified += 1
                    final.add(shot_id)
                elif r.tier == "degraded":
                    st.degraded += 1
                    final.add(shot_id)
                else:
                    st.deferred += 1
                st.emitted += 1
                harvest_late()
                advance_watermark()
                submit_up_to_cap()
                yield r
            # drain: deferred work is awaited to completion so nothing is
            # silently abandoned -- lateness is recorded, not lost
            for sid in sorted(overdue):
                st.late.append((sid, self._receipt(overdue[sid].get())))
                final.add(sid)
            advance_watermark()
