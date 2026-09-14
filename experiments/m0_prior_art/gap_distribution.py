r"""Milestone 0.3: the per-shot optimality gap, per decoder, per distance.

WHAT DOES NOT EXIST IN THE LITERATURE. Decoders are compared by logical error
rate: run many shots, count the failures, plot. That number is an average over
a distribution and it hides the thing an assurance argument needs. Two decoders
with the same logical error rate can fail completely differently -- one
consistently a little suboptimal, the other usually exact with a heavy tail --
and a benchmark cannot tell them apart because it never asks how far from
optimal any individual answer was.

It never asks because there has been no cheap way to know. Establishing that a
particular correction was minimum-weight means solving the instance exactly,
which is the decoder's whole job; comparing decoder A to decoder B tells you
they disagree, not which is right.

A CERTIFIED LOWER BOUND CHANGES THAT. The cut packing gives a bound that is
checkable edge by edge, on the SAME shot, without trusting any decoder. So for
every decoder, on every shot:

    gap = weight(that decoder's correction) - certified lower bound

is measurable, and its DISTRIBUTION can be published. A gap of zero means that
decoder achieved the certified optimum on that shot. It does not mean the
decoder is optimal in general, and the tail is the interesting part.

TWO ARTIFACTS, TWO ACCEPTANCE RULES. `evidence/gap_distribution.json` was
produced 2026-07-31 (cce1e91), when "proven optimal" meant a floating-point
gap of at most 1e-6 -- an epsilon-certificate; the certification semantics
audit shows every such certificate there proves slack below 1e-14 in exact
arithmetic. Run today, this script writes
`evidence/gap_distribution_exact_rule.json`: the "proven" subset is the one
the exact checker accepts, and the per-shot gaps are stored so a figure is
drawn from the measurement itself.

HONEST SCOPE, TWICE OVER.

  The bound is a bound, not the optimum. Where the ladder leaves a gap on
  PyMatching's own answer, part of the measured gap is our slack rather than
  the decoder's suboptimality. Shots where MWPM certified exactly are reported
  separately for that reason: on those, the bound IS the optimum and every
  other decoder's gap is exactly its excess weight, with nothing to argue about.

  Suboptimal is not the same as wrong. A heavier correction in the same
  homology class is harmless; a lighter-looking one in the wrong class is a
  logical error. Gap measures weight, not class, so the observable flip is
  recorded alongside it and neither number is allowed to stand in for the other.

DECODERS, AND WHAT THE NUMBER DOES NOT SAY ABOUT THEM. BP-OSD works in the
DEM's mechanism space, so its output is converted into the matching graph's
currency -- which detectors it makes odd, and what the correction weighs under
the SAME weights the bound uses -- before anything is compared. Scoring two
decoders under different weights and calling the difference a gap would be
meaningless in the way that looks like a result.

Even converted, **a large gap is not a defect**. BP-OSD is not trying to
minimise matching weight; it maximises likelihood over the full hypergraph,
where decomposed components are correlated and the min-weight T-join is not
the maximum-likelihood answer. So the honest reading of "BP-OSD, median gap 0,
p99 far from it" is: *usually it lands on the certified minimum-weight
correction, and when it departs it departs a long way* -- which is a
distributional fact about where it sits relative to a well-defined optimum,
not a scoreboard. Publishing it as "BP-OSD is worse than MWPM" would be
exactly the substitution this file exists to avoid.

BELIEF-MATCHING: A "FINDING" THAT WAS JUST ME STOPPING TOO EARLY. An earlier
version of this file recorded, as a result, that belief-matching returns only
predicted OBSERVABLES and therefore can be given no gap and no degraded
receipt. That was not a finding. `decode()` runs BP and, when BP does not
converge, builds a REWEIGHTED matching graph and decodes it -- a correction
exists in both branches, and reproducing the pipeline recovers it. Declaring a
limitation is a claim, an unverified negative is still a claim, and this one
cost nothing to check and was false.

Its gap does mean something different, though, and flattening that would be its
own error: belief-matching optimises under weights informed by BP posteriors,
so its correction is not attempting to be minimum-weight under the ORIGINAL
weights. A positive gap there is the price of a better-informed objective, not
evidence of a defect.

A SELF-CONSISTENCY CHECK THAT MUST HOLD. No decoder's syndrome-consistent
correction can weigh LESS than the certified lower bound -- a T-join cannot
undercut the minimum T-join. `frac_below_zero` is reported for every decoder
and any nonzero value means the harness or the bound is broken, not that a
decoder found something clever.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder     # noqa: E402
from oneq.matching_cert import BOUNDARY                   # noqa: E402


def dem_mechanisms(dem, W, key):
    """[(detectors, weight, p)] -- one entry per GRAPHLIKE COMPONENT.

    A DECOMPOSED ERROR IS NOT ONE MECHANISM. stim writes `error(p) D1 D5 ^ D4`
    for an error that decomposes into two graphlike pieces, and the `^` is a
    SEPARATOR target. Reading all the detectors of that instruction as a single
    mechanism produces a phantom object flipping three detectors: it is not an
    edge of the matching graph, its weight is wrong, and the odd-degree set it
    implies is wrong too. That bug made BP-OSD look like it was missing the
    optimum by a mean of 2.6 on shots where the optimum was PROVEN -- an
    implausible result that was measuring the harness, not the decoder.

    WEIGHTS COME FROM THE MATCHING GRAPH, not recomputed from p. The bound is
    stated in the matching graph's currency, so scoring a decoder in any other
    currency compares two different quantities and calls the difference a gap.
    PyMatching also MERGES parallel mechanisms when it builds the graph, which
    shifts a weight away from log((1-p)/p) for the individual error -- so
    recomputing would disagree with the bound for a reason that has nothing to
    do with the decoder.
    """
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        comps, cur = [], []
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(cur)
                cur = []
            elif t.is_relative_detector_id():
                cur.append(t.val)
        comps.append(cur)
        for dets in comps:
            if not dets or len(dets) > 2:
                # zero detectors: a pure observable flip, invisible to any
                # matching decoder. More than two after splitting: not
                # graphlike, so it has no edge and cannot be scored here.
                continue
            u = int(dets[0])
            v = int(dets[1]) if len(dets) == 2 else BOUNDARY
            k = key((u, v))
            if k not in W:
                continue
            out.append((k, W[k], min(max(p, 1e-15), 0.5 - 1e-15)))
    return out


def odd_and_weight(mechs, sel):
    """(detectors made odd, total weight) for a chosen set of mechanisms."""
    deg: dict[int, int] = {}
    w = 0.0
    for j in np.flatnonzero(sel):
        (u, v), wt, _p = mechs[j]
        w += wt
        for x in (u, v):
            if x != BOUNDARY:
                deg[x] = deg.get(x, 0) + 1
    return {d for d, c in deg.items() if c % 2 == 1}, w


class _BeliefMatchingCorrection:
    """Recovers the EDGE SET belief-matching actually applies.

    Reproduces `BeliefMatching.decode`'s own pipeline rather than calling it:
    BP first, and on non-convergence the reweighted matching it builds
    internally.

    RETURNS EDGE KEYS, NOT A MECHANISM SELECTION. beliefmatching enumerates its
    own graphlike edge columns, and those indices have nothing to do with our
    mechanism list -- reusing them as if they did would silently score the
    wrong edges and produce a plausible, meaningless gap. The detectors an edge
    touches are read out of its column in `edge_check_matrix`, which is the
    only thing that identifies it unambiguously.
    """

    returns_keys = True

    def __init__(self, bm, key):
        self.bm = bm
        self.mm = bm._matrices
        self.key = key
        H = self.mm.edge_check_matrix.tocsc()
        self.cols = []
        for j in range(H.shape[1]):
            dets = H.indices[H.indptr[j]:H.indptr[j + 1]]
            if len(dets) == 2:
                self.cols.append(key((int(dets[0]), int(dets[1]))))
            elif len(dets) == 1:
                self.cols.append(key((int(dets[0]), BOUNDARY)))
            else:
                self.cols.append(None)      # touches no detector: unweighable

    def decode_keys(self, syndrome):
        import numpy as np
        import pymatching
        s = np.asarray(syndrome).astype(np.uint8)
        corr = self.bm._bpd.decode(s)
        if self.bm._bpd.converge:
            sel = (self.mm.hyperedge_to_edge_matrix @ corr) % 2
            sel = np.asarray(sel).ravel().astype(bool)
        else:
            llrs = self.bm._bpd.log_prob_ratios
            ps_h = 1.0 / (1.0 + np.exp(llrs))
            ps_e = np.asarray(self.mm.hyperedge_to_edge_matrix @ ps_h).ravel()
            eps = 1e-14
            ps_e = np.clip(ps_e, eps, 1 - eps)
            M = pymatching.Matching.from_check_matrix(
                self.mm.edge_check_matrix, weights=-np.log(ps_e),
                faults_matrix=self.mm.edge_observables_matrix,
                use_virtual_boundary_node=True)
            # decode() RETURNS OBSERVABLE PREDICTIONS, NOT EDGES, because
            # faults_matrix is set -- reading it as an edge selection scored a
            # length-1 observable vector as if it indexed the 502 edge columns
            # and made every non-converged shot look syndrome-inconsistent.
            # 54 of 149 shots, which would have been written up as a property
            # of belief-matching rather than as my bug.
            return [self.key((int(u), int(v) if v >= 0 else BOUNDARY))
                    for u, v in M.decode_to_edges_array(s)]
        return [self.cols[j] for j in np.flatnonzero(sel) if self.cols[j] is not None]


def main() -> int:
    import pymatching
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=400)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--rounds", type=int, default=None, help="default = distance")
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    rows = []
    for d in a.distances:
        rounds = a.rounds or d
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds,
            after_clifford_depolarization=a.p,
            before_measure_flip_probability=a.p,
            after_reset_flip_probability=a.p,
            before_round_data_depolarization=a.p)
        dem = circ.detector_error_model(decompose_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
                  float(e[2].get("weight", 1.0))) for e in m.edges()]
        cd = CertifyingDecoder(edges)
        mechs = dem_mechanisms(dem, cd.W, cd.key)

        # H: detectors x mechanisms, over GF(2). The same matrix every
        # mechanism-space decoder consumes, so they are all scored alike.
        from scipy.sparse import csc_matrix
        rr, cc = [], []
        for j, ((u, v), _w, _p) in enumerate(mechs):
            for dd in (u, v):
                if dd != BOUNDARY:
                    rr.append(dd)
                    cc.append(j)
        H = csc_matrix((np.ones(len(rr), np.uint8), (rr, cc)),
                       shape=(dem.num_detectors, len(mechs)))
        probs = np.array([mm[2] for mm in mechs])

        decoders = {}
        try:
            from ldpc import BpOsdDecoder
            decoders["bp_osd"] = BpOsdDecoder(
                H, error_channel=list(probs), max_iter=30, bp_method="ms",
                schedule="parallel", osd_method="osd_cs", osd_order=5)
        except Exception as e:                       # noqa: BLE001
            print(f"  BP-OSD unavailable: {e}", file=sys.stderr)
        # BELIEF-MATCHING: I CALLED THIS UNCERTIFIABLE AND I WAS WRONG.
        #
        # An earlier version of this file recorded, as a FINDING, that
        # belief-matching "returns predicted observables, not a correction, so
        # it can be given no gap and no degraded receipt". That was not a
        # finding. It was stopping at the public method. `decode()` runs BP
        # and, when BP fails to converge, builds a REWEIGHTED pymatching graph
        # and decodes it -- so a correction exists in both branches and is
        # recoverable by reproducing the pipeline.
        #
        # Declaring a limitation is a claim, and an unverified negative claim
        # is still a claim. This one cost nothing to check and was false.
        #
        # THE GAP MEANS SOMETHING DIFFERENT HERE, and the difference must not
        # be flattened. Belief-matching deliberately optimises under REWEIGHTED
        # edge weights informed by BP posteriors. Its correction is therefore
        # not trying to be minimum-weight under the ORIGINAL weights, and a
        # positive gap is the expected consequence of a better-informed
        # objective rather than evidence of a defect.
        try:
            from beliefmatching import BeliefMatching
            bm = BeliefMatching(dem, max_bp_iters=20)
            decoders["belief_matching"] = _BeliefMatchingCorrection(bm, cd.key)
        except Exception as e:                       # noqa: BLE001
            print(f"  belief-matching unavailable: {e}", file=sys.stderr)

        det, obs = circ.compile_detector_sampler(seed=31337).sample(
            a.shots, separate_observables=True)

        stats: dict[str, dict] = {}
        n = 0
        n_exact = 0
        t0 = time.perf_counter()
        for s in range(a.shots):
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            n += 1
            mw = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                       for u, v in m.decode_to_edges_array(det[s]))
            r = cd.certify(fired, mw)
            bound = r.bound
            if r.accepted:
                n_exact += 1

            entries = {"mwpm": (r.primal, True)}
            for name, dec in decoders.items():
                try:
                    if getattr(dec, "returns_keys", False):
                        keys = dec.decode_keys(det[s])
                        deg: dict[int, int] = {}
                        w = 0.0
                        for k in keys:
                            if k not in cd.W:
                                continue
                            w += cd.W[k]
                            for x in k:
                                if x != BOUNDARY:
                                    deg[x] = deg.get(x, 0) + 1
                        odd = {n for n, c in deg.items() if c % 2 == 1}
                    else:
                        sel = np.asarray(dec.decode(
                            det[s].astype(np.uint8))).astype(bool).ravel()
                        odd, w = odd_and_weight(mechs, sel)
                except Exception:                     # noqa: BLE001
                    continue
                entries[name] = (w, odd == set(fired))

            for name, (w, consistent) in entries.items():
                st = stats.setdefault(name, {"n": 0, "gaps": [],
                                             "gaps_on_exact": [],
                                             "inconsistent": 0})
                st["n"] += 1
                if not consistent:
                    # A correction that does not explain the syndrome has no
                    # meaningful gap; counting it as one would flatter or
                    # slander the decoder depending on sign.
                    st["inconsistent"] += 1
                    continue
                st["gaps"].append(w - bound)
                if r.accepted:
                    st["gaps_on_exact"].append(w - r.primal)
        secs = time.perf_counter() - t0

        def summarise(v):
            if not v:
                return None
            arr = np.array(v)
            return {"n": int(arr.size), "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(arr.max()),
                    "frac_zero": float((np.abs(arr) < 1e-9).mean()),
                    "frac_below_zero": float((arr < -1e-9).mean())}

        row = {"distance": d, "rounds": rounds, "p": a.p, "shots": a.shots,
               "nontrivial": n, "mwpm_certified_exact": n_exact,
               "seconds": secs, "decoders": {}}
        for name, st in sorted(stats.items()):
            row["decoders"][name] = {
                "n": st["n"], "syndrome_inconsistent": st["inconsistent"],
                "gap_vs_certified_bound": summarise(st["gaps"]),
                "gap_vs_proven_optimum": summarise(st["gaps_on_exact"]),
                # The per-shot values the summary is computed from, so a
                # figure is drawn from the measurement rather than
                # reconstructed from five quantiles.
                "per_shot_gap_vs_proven_optimum": [float(x) for x in
                                                   st["gaps_on_exact"]]}
        rows.append(row)

        print(f"d={d} rounds={rounds}  {n} nontrivial, "
              f"{n_exact} with a PROVEN optimum ({100*n_exact/max(n,1):.1f}%)")
        for name, r2 in row["decoders"].items():
            g = r2["gap_vs_proven_optimum"]
            if not g:
                print(f"   {name:16s} no comparable shots")
                continue
            print(f"   {name:16s} on proven-optimal shots: mean {g['mean']:7.4f}  "
                  f"median {g['median']:7.4f}  p99 {g['p99']:7.4f}  "
                  f"max {g['max']:7.4f}  exact {100*g['frac_zero']:5.1f}%"
                  + (f"  INCONSISTENT {r2['syndrome_inconsistent']}"
                     if r2["syndrome_inconsistent"] else ""))

    out = {
        "schema": "oneq-gap-distribution/2", "rows": rows,
        "lower_bound_source": ("a certified cut packing on the SAME shot -- "
                               "checkable edge by edge, not another decoder's "
                               "opinion"),
        "two_columns_on_purpose": {
            "gap_vs_certified_bound": ("weight minus the bound. Includes OUR "
                                       "slack wherever the bound did not close, "
                                       "so it overstates the decoder's fault."),
            "gap_vs_proven_optimum": ("restricted to shots where MWPM was "
                                      "PROVEN optimal. There the bound IS the "
                                      "optimum, so this is exactly the "
                                      "decoder's excess weight -- the number "
                                      "worth quoting.")},
        "what_gap_is_not": ("weight, not homology class. A heavier correction "
                            "in the right class is harmless; a lighter one in "
                            "the wrong class is a logical error. Neither number "
                            "substitutes for the other."),
        "a_gap_is_not_a_defect": (
            "BP-OSD does not minimise matching weight -- it maximises "
            "likelihood over the full hypergraph, where decomposed components "
            "are correlated and the min-weight T-join is not the "
            "maximum-likelihood answer. A large gap says it departed from the "
            "certified minimum-weight correction, not that it was wrong. "
            "Reading this table as a decoder scoreboard is a misuse."),
        "belief_matching_is_certifiable": (
            "an earlier version of this artifact said belief-matching could "
            "not be measured because its API returns predicted observables. "
            "That was wrong: decode() runs BP and, on non-convergence, builds "
            "a reweighted matching graph and decodes it, so a correction "
            "exists in both branches and _BeliefMatchingCorrection recovers "
            "it. Its gap is different in kind -- it optimises under "
            "BP-informed weights, not the original ones."),
        "self_consistency": (
            "no syndrome-consistent correction can weigh LESS than the "
            "certified bound; frac_below_zero must be 0 for every decoder, and "
            "a nonzero value means the harness or the bound is broken"),
        "why_it_is_new": ("logical error rate averages this away. Two decoders "
                          "with the same error rate can fail completely "
                          "differently -- consistently a little suboptimal "
                          "versus usually exact with a heavy tail -- and no "
                          "benchmark distinguishes them, because none asks how "
                          "far from optimal an individual answer was."),
    }
    # NOT gap_distribution.json. That artifact is the 2026-07-31 measurement
    # under the pre-audit acceptance rule (float gap <= 1e-6), and it is what
    # paper 1 section 7 quotes; certification_semantics_audit.py re-runs the
    # pinned code and reproduces it exactly. Run from HEAD, this script
    # certifies under the EXACT rule, selects a different "proven optimal"
    # subset (d=7: 253 of 400 against 378), and is a second measurement --
    # so it gets its own artifact and can never silently replace the first.
    dest = (pathlib.Path(a.out) if a.out
            else ROOT / "evidence" / "gap_distribution_exact_rule.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nevidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
