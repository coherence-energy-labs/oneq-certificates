r"""The logical cut c: which edges flip an observable. Two sources, agreeing.

Phase 1's whole construction hangs on one vector per observable:
`c[e] = 1` iff decoding-graph edge `e` flips logical observable k. The
opposite-coset problem is then the original problem with row `c` appended
(H' = [H; c]) -- so if `c` is wrong, every downstream certificate is a
proof about the wrong question, and nothing in the exact machinery would
notice: the LP, the simplex, the checker would all be correct about a cut
that does not correspond to the physical logical operator.

So `c` is extracted from TWO independent sources and they must AGREE:

  1. PyMatching's view: `Matching.edges()` exposes `fault_ids` per edge --
     the decoder's own belief about which observables an edge flips.
  2. Stim's view: the detector error model TEXT, where each error
     mechanism lists its detectors and its `L<k>` observable targets --
     the noise model's ground truth, parsed without PyMatching.

A disagreement between them is not resolved here; it is REFUSED loudly,
because it means the decoder and the noise model disagree about what the
logical operator IS, and every receipt built on either answer would be
unfalsifiable propaganda.
"""

from __future__ import annotations

BOUNDARY = -1


def _key(u, v):
    if u == BOUNDARY:
        u, v = v, u
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


def cut_from_pymatching(matching, observable: int) -> frozenset:
    """Edges whose fault_ids contain `observable`, per the decoder."""
    out = set()
    for u, v, data in matching.edges():
        if observable in (data.get("fault_ids") or set()):
            out.add(_key(int(u), BOUNDARY if v is None else int(v)))
    return frozenset(out)


def cut_from_dem(dem, observable: int) -> frozenset:
    """Edges whose DOMINANT error mechanism flips `L<observable>`.

    THE PARALLEL-MERGE SUBTLETY, found by this cross-check's first run:
    several DEM mechanisms can share one detector pair, and they need not
    agree about observables -- on a d=3 surface code, mechanism-level
    counting said 32 edges carry L0 while the decoding graph says 8. The
    matching-graph abstraction genuinely loses mechanism-level observable
    information on merge, and PyMatching resolves it by keeping the
    HIGHEST-PROBABILITY mechanism's fault_ids (derived empirically:
    78/78 edges, single- and multi-mechanism, on the d=3 system).

    So this mirrors that rule from the DEM text alone: group graphlike
    mechanisms (1-2 detectors) by edge key, take the dominant mechanism's
    observables. What is being cross-checked is then the cut ON THE GRAPH
    THE DECODER ACTUALLY DECODES -- which is the object every downstream
    certificate quantifies over. Mechanisms with >2 detectors are not
    edges and are skipped; if one of those carried the observable, the
    two sources diverge and logical_cut refuses, which is correct.
    """
    best: dict = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        # SPLIT ON SEPARATORS. A decomposed hyperedge is ONE instruction
        # whose `^`-separated components are each their own graphlike
        # mechanism. Skipping separators flattened those components into
        # a >2-detector blob that got dropped -- so the dominant per edge
        # was computed over an incomplete mechanism set, and the rule
        # looked wrong (23 vs 8 edges) when the PARSE was wrong. With
        # components counted, dominant matches PyMatching 78/78.
        components: list[list] = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                components.append([])
            else:
                components[-1].append(t)
        for targets in components:
            dets = [t.val for t in targets if t.is_relative_detector_id()]
            obs = frozenset(t.val for t in targets
                            if t.is_logical_observable_id())
            if not dets or len(dets) > 2:
                continue
            k = (_key(dets[0], BOUNDARY) if len(dets) == 1
                 else _key(dets[0], dets[1]))
            if k not in best or prob > best[k][0]:
                best[k] = (prob, obs)
    return frozenset(k for k, (_p, obs) in best.items() if observable in obs)


def logical_cut(matching, dem, observable: int = 0) -> frozenset:
    """The agreed logical cut, or a loud refusal.

    Returns the edge set both sources agree on. Raises ValueError naming
    the disagreement otherwise -- a wrong `c` poisons every downstream
    certificate, so there is no permissive mode.
    """
    a = cut_from_pymatching(matching, observable)
    b = cut_from_dem(dem, observable)
    if a != b:
        only_pm = sorted(a - b)[:5]
        only_dem = sorted(b - a)[:5]
        raise ValueError(
            f"logical cut DISAGREEMENT for observable {observable}: "
            f"decoder and noise model do not name the same edges "
            f"(pymatching-only: {only_pm}, dem-only: {only_dem}). Every "
            f"receipt built on either answer would be about the wrong "
            f"question -- refusing.")
    return a


def logical_parity(correction, cut: frozenset) -> int:
    """lambda(e) = |e intersect c| mod 2, over canonical edge keys."""
    return sum(1 for e in correction if _key(*e) in cut) % 2
