r"""Paper 1 Sec. VIII: the per-shot optimality gap of other decoders against PyMatching's certified correction.

WHAT IS MEASURED. On the same shot, PyMatching's correction J is certified by a cut packing; another decoder's
correction K is projected onto the matching graph and weighed in the same currency. The candidate's excess over the
minimum-weight T-join is then bracketed by arithmetic, not estimated.

THE SCORER WAS WRONG BEFORE 2026-11-03, AND HOW (external audit 2, finding F01). A decomposed detector-error-model
instruction contributes one BP-OSD column per graphlike COMPONENT, and many components land on the same graph edge
(mean 7.1, up to 35 columns per edge at d=3). The released `odd_and_weight` reduced the syndrome mod 2 but summed the
edge weight once per selected component, so a decoder that selected nine copies of edge (7, 10) was charged nine
times for a correction that, as an edge set, contains it once. Re-running the published experiment bit for bit, 158
of the 1,000 Table V shots and 149 of the 863 Fig. 3 shots were scored that way, with an overcount of 7.6-146.3
weight units each; at d=3 and d=5 every one was a CONVERGED BP hard decision (unscaled min-sum flips every copy of a
group of identical columns, and an odd count satisfies the syndrome, so OSD never runs). A correction is an edge set:
the selection is now projected mod 2 (`project_selection`) and weighed in exact rationals of the binary64 weights,
the same numbers the certificate checker reads.

THE REFERENCE (also changed). The original artifacts restricted to shots whose PyMatching correction an exact or a
tolerance rule accepted, and called a gap "exact" when |gap| < 1e-9. Here:

  PRIMARY reference   J proven within eps(J) of the optimum by `check_epsilon` (eps_max = 1e-6). Then
                      OPT is in [w(J) - eps(J), w(J)], so for g = w(K) - w(J), computed exactly,
                      w(K) - OPT lies in [max(0, g), g + eps(J)] (`gap_interval`). A hit is PROVEN when the upper
                      end is 0, POSSIBLE when the lower end is 0, and K is PROVEN eps-optimal when the upper end is at
                      most eps_max. Nothing is rounded.
  SECONDARY reference J accepted by the exact checker (`CertifyingDecoder.certify`): OPT = w(J) exactly and g is
                      the excess. Reported because the original paper used it; it selects easier shots (at d=7 it
                      kept 253 of 400 against 399 for the epsilon reference, and the dropped shots fired more
                      detectors).

THE DECODER ARMS (fixed in docs/PREREG_gap_distribution_2026-11-03.md before any run of this version):

  bposd_merged        PRIMARY. One BP-OSD column per matching-graph edge, its probability the XOR-combination of
                      every graphlike component on that edge (reproduces PyMatching's merged weights to 5e-16
                      relative at d=3, 5, 7), so BP-OSD's objective is the matching graph's; scaled min-sum 0.625.
  bposd_component     SECONDARY, the input-model sensitivity arm: the released configuration (one column per
                      component, unscaled min-sum, the ldpc 2.4.1 default), now XOR-scored.
  belief_matching     as before; its edge keys are projected mod 2 too.
  pymatching          the reference decoder.

WHAT A GAP IS NOT. Weight, not homology class; and BP-OSD with finite iterations and bounded-order OSD is a
heuristic, not a maximum-likelihood decoder over any model. A positive gap says the decoder's answer is heavier than
the minimum-weight correction under the matching graph's weights, nothing more.

PER-SHOT RECORDS keep the shot index, the syndrome, each decoder's raw selection, its projected edge set and exact
cost, and the reference's verdict, so every statistic can be recomputed and a reader on another platform (Stim's
seeded sampler is not bit-identical across machines) can re-score the same syndromes.

    python experiments/m0_prior_art/gap_distribution.py --seed S --shots N --tag NAME
      -> evidence/gap_distribution_v4/NAME/summary.json and per_shot_d<d>.jsonl.gz

The 2026-07-31 artifact (`evidence/gap_distribution.json`, commit cce1e91) and the 2026-11-02 one
(`evidence/gap_distribution_exact_rule.json`) are left exactly as written; both used the defective scorer.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder     # noqa: E402
from oneq.matching_cert import BOUNDARY                   # noqa: E402

EPS_MAX = Fraction(1, 10**6)

#: The released (pre-2026-11-03) BP-OSD settings; ms_scaling_factor left at the ldpc default.
RELEASED_BPOSD = dict(max_iter=30, bp_method="ms", schedule="parallel", osd_method="osd_cs", osd_order=5)
#: The preregistered primary settings (docs/PREREG_gap_distribution_2026-11-03.md).
PRIMARY_BPOSD = dict(max_iter=30, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel",
                     osd_method="osd_cs", osd_order=5)


def dem_mechanisms(dem, W, key):
    """[(edge, weight, p)] -- one entry per GRAPHLIKE COMPONENT of a decomposed DEM instruction.

    stim writes `error(p) D1 D5 ^ D4` for an error that decomposes into two graphlike pieces; `^` separates them.
    Each piece becomes its own entry carrying the instruction's p, so the pieces are treated as INDEPENDENT -- the
    correlation between pieces of one instruction is discarded. Pieces with no detector (pure observable flips) or
    more than two (not graphlike) have no edge and are dropped. The weight is the matching graph's, the currency
    of the bound; many entries share one edge.
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
                continue
            u = int(dets[0])
            v = int(dets[1]) if len(dets) == 2 else BOUNDARY
            k = key((u, v))
            if k not in W:
                continue
            out.append((k, W[k], min(max(p, 1e-15), 0.5 - 1e-15)))
    return out


def _edge_of(col):
    """A column is either a mechanism entry (edge, weight, p) or an edge key (u, v)."""
    return col[0] if isinstance(col[0], tuple) else col


def project_selection(cols, sel, W):
    """A decoder's column selection as the EDGE SET it applies, and that set's exact cost.

    Two selected columns on the same edge cancel: the correction is the mod-2 sum of the selected columns'
    edges. The cost is the sum of Fraction(w_e) over that edge set -- exactly what the certificate checker reads.
    """
    idx = [int(j) for j in np.flatnonzero(np.asarray(sel).astype(bool).ravel())]
    count = Counter(_edge_of(cols[j]) for j in idx)
    edges = sorted(e for e, c in count.items() if c % 2)
    deg: Counter = Counter()
    for (u, v) in edges:
        for x in (u, v):
            if x != BOUNDARY:
                deg[x] += 1
    off = [e for e in edges if e not in W]
    cost = sum((Fraction(W[e]) for e in edges if e in W), Fraction(0))
    return {"selected": idx, "edges": edges, "odd": {x for x, c in deg.items() if c % 2},
            "cost_exact": cost, "cost_float": float(cost), "off_graph": len(off),
            "multiplicity_max": max(count.values()) if count else 0,
            "edges_with_cancelled_copies": sum(1 for c in count.values() if c > 1)}


def odd_and_weight(mechs, sel):
    """(detectors made odd, weight) of a mechanism selection, AFTER projection onto edges mod 2.

    Before 2026-11-03 this summed the weight once per selected component (audit 2, F01). The weight returned is the
    correctly rounded double of the projected edge set's exact cost.
    """
    W = {}
    for mm in mechs:
        W.setdefault(mm[0], mm[1])
    p = project_selection(mechs, sel, W)
    return p["odd"], p["cost_float"]


def merged_edge_columns(mechs):
    """(edge keys, probabilities): one column per graph edge, probability = P(an odd number of its components fire).

    The XOR-combination p <- p(1-q) + q(1-p) over independent components is how PyMatching merges parallel
    mechanisms, so -log(p/(1-p)) reproduces the matching graph's weight; BP-OSD on these columns optimises the
    matching graph's objective.
    """
    P: dict = {}
    for (k, _w, p) in mechs:
        q = P.get(k, 0.0)
        P[k] = q * (1.0 - p) + p * (1.0 - q)
    cols = sorted(P)
    return cols, [min(max(P[k], 1e-15), 0.5 - 1e-15) for k in cols]


def column_matrix(col_edges, num_detectors):
    """Detectors x columns over GF(2); a boundary edge touches one detector."""
    from scipy.sparse import csc_matrix
    rr, cc = [], []
    for j, (u, v) in enumerate(col_edges):
        for x in (u, v):
            if x != BOUNDARY:
                rr.append(x)
                cc.append(j)
    return csc_matrix((np.ones(len(rr), np.uint8), (rr, cc)), shape=(num_detectors, len(col_edges)))


def gap_interval(w_k: Fraction, w_j: Fraction, eps_j: Fraction):
    """[lo, hi] containing w(K) - OPT, given w(J) - eps(J) <= OPT <= w(J).

    A candidate lighter than the proven lower bound is impossible for a valid correction, so it means the harness
    or the bound is broken; that raises rather than being reported.
    """
    g = w_k - w_j
    if g + eps_j < 0:
        raise ValueError(f"candidate weight {w_k} is below the proven lower bound {w_j - eps_j}")
    return max(g, Fraction(0)), g + eps_j


class _BeliefMatchingCorrection:
    """Recovers the EDGE KEYS belief-matching actually applies (BP, then on non-convergence its reweighted matching).

    beliefmatching enumerates its own graphlike edge columns; the detectors an edge touches are read from its column
    in `edge_check_matrix`, the only thing that identifies it. The keys are projected mod 2 by the caller.
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
            ps_e = np.clip(ps_e, 1e-14, 1 - 1e-14)
            M = pymatching.Matching.from_check_matrix(
                self.mm.edge_check_matrix, weights=-np.log(ps_e),
                faults_matrix=self.mm.edge_observables_matrix, use_virtual_boundary_node=True)
            # decode() returns OBSERVABLE predictions when faults_matrix is set; the edges come from here.
            return [self.key((int(u), int(v) if v >= 0 else BOUNDARY)) for u, v in M.decode_to_edges_array(s)]
        return [self.cols[j] for j in np.flatnonzero(sel) if self.cols[j] is not None]


def _fr(x: Fraction | None) -> str | None:
    return None if x is None else str(x)


def _cp(k: int, n: int):
    """Exact two-sided 95% Clopper-Pearson interval on k/n, or None for n = 0."""
    if not n:
        return None
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _quantiles(vals):
    if not vals:
        return None
    arr = np.array([float(v) for v in vals])
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)), "max": float(arr.max())}


def summarise(recs, name):
    """Per-decoder statistics from the per-shot records, every comparison exact."""
    eps_ref = [r for r in recs if r["ref_eps"]["accepted"]]
    exact_ref = [r for r in recs if r["ref_exact_accepted"]]
    out = {"nontrivial": len(recs), "syndrome_inconsistent": sum(1 for r in recs if not r[name]["consistent"])}
    # Four counts, each a proof about K, not an estimate:
    #   proven_exact_hit      w(K) = OPT          (upper end 0; needs eps(J) = 0 and g = 0)
    #   possible_exact_hit    w(K) = OPT not excluded (lower end 0)
    #   proven_eps_optimal    w(K) - OPT <= eps_max (upper end <= eps_max)   -- the primary hit measure
    #   proven_miss_beyond_eps  w(K) - OPT > eps_max (lower end > eps_max)
    # proven_miss (lower end > 0) also counts K heavier than J by a rounding amount, so it overlaps eps-optimal.
    lo, hi, proven_hit, possible_hit, eps_hit, proven_miss, miss_beyond = [], [], 0, 0, 0, 0, 0
    for r in eps_ref:
        d = r[name]
        if not d["consistent"]:
            continue
        a, b = Fraction(d["gap_lo"]), Fraction(d["gap_hi"])
        lo.append(a)
        hi.append(b)
        proven_hit += b == 0
        possible_hit += a == 0
        eps_hit += b <= EPS_MAX
        proven_miss += a > 0
        miss_beyond += a > EPS_MAX
    n = len(lo)
    out["epsilon_reference"] = {
        "n": n, "reference_shots": len(eps_ref),
        "proven_exact_hit": proven_hit, "possible_exact_hit": possible_hit,
        "proven_eps_optimal": eps_hit, "proven_miss": proven_miss, "proven_miss_beyond_eps": miss_beyond,
        "undecided_by_eps": n - eps_hit - miss_beyond,
        "hit_rate_interval": [proven_hit / n, possible_hit / n] if n else None,
        "eps_optimal_rate": eps_hit / n if n else None,
        "clopper_pearson_95": {k: _cp(v, n) for k, v in (("proven_eps_optimal", eps_hit),
                                                         ("proven_exact_hit", proven_hit),
                                                         ("possible_exact_hit", possible_hit),
                                                         ("proven_miss_beyond_eps", miss_beyond))},
        "gap_lo": _quantiles(lo), "gap_hi": _quantiles(hi)}
    g, hit = [], 0
    for r in exact_ref:
        d = r[name]
        if not d["consistent"]:
            continue
        x = Fraction(d["cost_exact"]) - Fraction(r["pymatching"]["cost_exact"])
        if x < 0:
            raise ValueError(f"{name}: lighter than an exactly certified optimum on shot {r['shot']}")
        g.append(x)
        hit += x == 0
    out["exact_reference"] = {"n": len(g), "reference_shots": len(exact_ref), "exact_hit": hit,
                              "hit_rate": hit / len(g) if g else None,
                              "clopper_pearson_95": _cp(hit, len(g)), "gap": _quantiles(g)}
    out["shots_with_cancelled_copies"] = sum(1 for r in recs if r[name].get("edges_with_cancelled_copies", 0))
    return out


def _git(*args: str) -> str:
    """git's answer, or "" where there is no repository (a reader's extracted bundle): the record then names no
    commit, and its judged files are pinned by sha256 alone."""
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def main(argv=None) -> int:
    import importlib.metadata as md

    import pymatching
    import stim
    from ldpc import BpOsdDecoder
    from oneq.matching_cert import MatchingCertificate
    from oneq.matching_cert_epsilon import check_epsilon

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="sampler seed; distance d uses seed + d")
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--no-belief-matching", action="store_true")
    a = ap.parse_args(argv)
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else ROOT / "evidence" / "gap_distribution_v4" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # The commit and the judged bytes are taken at the START (2026-11-03, final referee): the preregistered run read
    # HEAD when it finished, after other lanes had committed, and so names 70b3c4d8 for a run started at 2008c81f. A
    # run whose judged bytes change while it runs is refused rather than attributed to either tree.
    judged = ["src/oneq/matching_cert.py", "src/oneq/matching_cert_epsilon.py", "src/oneq/certifying_decoder.py",
              "experiments/m0_prior_art/gap_distribution.py"]

    def judged_sha():
        return {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in judged}
    head = {"commit": _git("rev-parse", "HEAD"), "recorded": "at the start of the run",
            "uncommitted_changes": bool(_git("status", "--porcelain", "--", "src/oneq", "experiments/m0_prior_art")),
            "sha256": judged_sha()}

    rows = []
    for d in a.distances:
        rounds = d
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds,
            after_clifford_depolarization=a.p, before_measure_flip_probability=a.p,
            after_reset_flip_probability=a.p, before_round_data_depolarization=a.p)
        dem = circ.detector_error_model(decompose_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0)))
                 for e in m.edges()]
        cd_exact = CertifyingDecoder(edges)
        cd_eps = CertifyingDecoder(edges, epsilon_max=float(EPS_MAX))
        W = cd_eps.W
        mechs = dem_mechanisms(dem, W, cd_eps.key)
        comp_cols = [mm[0] for mm in mechs]
        merged_cols, merged_p = merged_edge_columns(mechs)
        merge_rel = max(abs(float(np.log((1 - p) / p)) - W[k]) / W[k] for k, p in zip(merged_cols, merged_p))
        arms = {
            "bposd_merged": (merged_cols, BpOsdDecoder(column_matrix(merged_cols, dem.num_detectors),
                                                       error_channel=merged_p, **PRIMARY_BPOSD)),
            "bposd_component": (comp_cols, BpOsdDecoder(column_matrix(comp_cols, dem.num_detectors),
                                                        error_channel=[mm[2] for mm in mechs], **RELEASED_BPOSD)),
        }
        # read back from the decoder objects, so the record states what ran rather than what was asked for
        settings = {name: {k: str(getattr(dec, k, "unavailable"))
                           for k in ("ms_scaling_factor", "bp_method", "schedule", "osd_method", "osd_order",
                                     "max_iter")} | {"columns": len(cols)}
                    for name, (cols, dec) in arms.items()}
        bm = None
        if not a.no_belief_matching:
            from beliefmatching import BeliefMatching
            bm = _BeliefMatchingCorrection(BeliefMatching(dem, max_bp_iters=20), cd_eps.key)

        seed = a.seed + d
        det, obs = circ.compile_detector_sampler(seed=seed).sample(a.shots, separate_observables=True)
        recs = []
        path = out_dir / f"per_shot_d{d}.jsonl.gz"
        t0 = time.perf_counter()
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for s in range(a.shots):
                fired = [int(i) for i in np.flatnonzero(det[s])]
                if not fired:
                    continue
                mw = tuple(sorted(cd_eps.key((u, v if v >= 0 else BOUNDARY))
                                  for u, v in m.decode_to_edges_array(det[s])))
                w_j = sum((Fraction(W[k]) for k in mw), Fraction(0))
                re_ = cd_eps.certify(fired, mw)
                duals = dict(re_.cert.blossom_duals) if re_.cert is not None else dict(re_.best_dual or {})
                v = check_epsilon(edges=edges, syndrome=fired, eps_max=EPS_MAX,
                                  cert=MatchingCertificate(matched_edges=mw, node_potentials={}, blossom_duals=duals))
                rx = cd_exact.certify(fired, mw)
                rec = {"shot": s, "fired": fired, "observable_flip": bool(obs[s][0]),
                       "pymatching": {"edges": [list(e) for e in mw], "cost_exact": str(w_j)},
                       "ref_eps": {"status": v.status, "accepted": v.accepted, "epsilon": _fr(v.epsilon),
                                   "lower_bound": _fr(v.lower_bound)},
                       "ref_exact_accepted": bool(rx.accepted)}
                cands = {}
                for name, (cols, dec) in arms.items():
                    sel = np.asarray(dec.decode(det[s].astype(np.uint8))).astype(bool).ravel()
                    cands[name] = project_selection(cols, sel, W)
                    cands[name]["bp_converged"] = bool(dec.converge)
                if bm is not None:
                    keys = bm.decode_keys(det[s])
                    cnt = Counter(keys)
                    cands["belief_matching"] = {
                        "selected": None, "edges": sorted(k for k, c in cnt.items() if c % 2),
                        "multiplicity_max": max(cnt.values()) if cnt else 0,
                        "edges_with_cancelled_copies": sum(1 for c in cnt.values() if c > 1)}
                    e_bm = cands["belief_matching"]["edges"]
                    deg: Counter = Counter()
                    for (x, y) in e_bm:
                        for z in (x, y):
                            if z != BOUNDARY:
                                deg[z] += 1
                    cost = sum((Fraction(W[e]) for e in e_bm if e in W), Fraction(0))
                    cands["belief_matching"].update(
                        {"odd": {z for z, c in deg.items() if c % 2}, "cost_exact": cost,
                         "off_graph": sum(1 for e in e_bm if e not in W)})
                for name, c in cands.items():
                    consistent = c["odd"] == set(fired) and c["off_graph"] == 0
                    entry = {"edges": [list(e) for e in c["edges"]], "cost_exact": str(c["cost_exact"]),
                             "consistent": consistent, "multiplicity_max": c["multiplicity_max"],
                             "edges_with_cancelled_copies": c["edges_with_cancelled_copies"]}
                    if c.get("selected") is not None:
                        entry["selected_columns"] = c["selected"]
                        entry["bp_converged"] = c["bp_converged"]
                    if consistent and v.accepted:
                        g_lo, g_hi = gap_interval(c["cost_exact"], w_j, v.epsilon)
                        entry["gap_lo"], entry["gap_hi"] = str(g_lo), str(g_hi)
                    rec[name] = entry
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                recs.append(rec)
        secs = time.perf_counter() - t0
        row = {"distance": d, "rounds": rounds, "p": a.p, "seed": seed, "shots": a.shots, "nontrivial": len(recs),
               "graph_edges": len(W), "component_columns": len(comp_cols),
               "merged_weight_max_relative_difference": merge_rel,
               "reference": {"epsilon_accepted": sum(1 for r in recs if r["ref_eps"]["accepted"]),
                             "epsilon_status": dict(Counter(r["ref_eps"]["status"] for r in recs)),
                             "exact_checker_accepted": sum(1 for r in recs if r["ref_exact_accepted"])},
               "bposd_settings": settings, "seconds": secs,
               "per_shot": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
               "per_shot_sha256": hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest(),
               "decoders": {name: summarise(recs, name) for name in [k for k in recs[0] if k in
                                                                    ("bposd_merged", "bposd_component",
                                                                     "belief_matching")]} if recs else {}}
        rows.append(row)
        print(f"d={d}: {len(recs)} nontrivial, eps reference {row['reference']['epsilon_accepted']}, "
              f"exact reference {row['reference']['exact_checker_accepted']}, {secs:.0f} s", flush=True)

    if judged_sha() != head["sha256"]:
        raise SystemExit("a judged file changed while the run was running; it is attributed to neither tree")
    head["commit_at_end"] = _git("rev-parse", "HEAD")
    doc = {"schema": "oneq-gap-distribution/4", "head": head, "argv": sys.argv[1:],
           "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                           **{k: md.version(k) for k in ("stim", "pymatching", "ldpc", "numpy", "scipy",
                                                         "beliefmatching")}},
           "eps_max": str(EPS_MAX), "rows": rows,
           "scoring": "candidate selections projected onto graph edges mod 2; costs in exact rationals of the "
                      "binary64 weights; gap interval [max(0, g), g + eps(J)] against the epsilon reference",
           "preregistration": "docs/PREREG_gap_distribution_2026-11-03.md"}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"evidence -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
