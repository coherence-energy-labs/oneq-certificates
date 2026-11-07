r"""POST HOC: the missing two cells of paper 1 Sec. VIII's BP-OSD comparison (external audit 3, 2026-11-06).

    python -B experiments/m0_prior_art/gap_distribution_2x2_posthoc.py
      -> evidence/gap_distribution_v4/posthoc_2x2_2026-11-06/summary.json and per_shot_d<d>.jsonl.gz

WHY. The preregistered gap run (docs/PREREG_gap_distribution_2026-11-03.md) compared two BP-OSD arms that differ in
TWO factors at once: `bposd_merged` (one column per matching-graph edge, min-sum scaling 0.625) and `bposd_component`
(one column per graphlike component, the library's default unscaled min-sum, 1.0). Its difference therefore cannot be
attributed to the representation, and the mechanism the paper named for it -- on independent components BP-OSD
selects several parallel columns of one edge -- was traced in ledger R235 to UNSCALED min-sum flipping every copy of a
group of identical columns. This runs the two missing cells of the 2x2,

                       min-sum 0.625                 min-sum 1.0 (unscaled)
    merged edges       bposd_merged (preregistered)  NEW, post hoc
    components         NEW, post hoc                 bposd_component (preregistered)

on exactly the preregistered shots (the syndromes the per-shot records keep), scored by exactly the preregistered
scoring (`gap_distribution.project_selection`, `gap_interval`, `summarise`) against exactly the preregistered
reference (PyMatching's correction and its recorded epsilon verdict, read from the record, never recomputed).

WHAT IT REFUSES BEFORE WRITING ANYTHING.
  * the per-shot records must hash to the summary's per_shot_sha256;
  * the regenerated graph must be the record's: the edge and column counts equal, and on EVERY shot the exact cost of
    every stored edge set (PyMatching's and each stored decoder's) re-summed over this graph's binary64 weights must
    equal the recorded cost;
  * the two PREREGISTERED cells, re-decoded here from the stored syndromes, must reproduce the record on every shot:
    the selected columns, the projected edge set and BP's convergence flag, and their per-distance summaries must equal
    the stored ones. A harness that cannot reproduce the two known cells has no standing to report the two new ones.

Nothing here is written into the preregistered record (`prereg_2026-11-03/`); the result is POST HOC and labelled so.
Weight, not homology class, as in the preregistered run.
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
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

import gap_distribution as gd          # noqa: E402  the preregistered runner's own functions

RUN = ROOT / "evidence" / "gap_distribution_v4" / "prereg_2026-11-03"
OUT_DIR = ROOT / "evidence" / "gap_distribution_v4" / "posthoc_2x2_2026-11-06"
PREREG = "docs/PREREG_gap_distribution_2026-11-03.md"

#: The four cells: (input model, min-sum scaling) -> the name in the record. The two preregistered arms keep their
#: preregistered names and are REPRODUCED; the two new ones are named for what they are.
CELLS = {
    ("merged", 0.625): "bposd_merged",
    ("component", 1.0): "bposd_component",
    ("merged", 1.0): "bposd_merged_unscaled",
    ("component", 0.625): "bposd_component_scaled",
}
PREREGISTERED = ("bposd_merged", "bposd_component")
NEW = ("bposd_merged_unscaled", "bposd_component_scaled")
#: Every other setting is the preregistered one (gap_distribution.PRIMARY_BPOSD less its scaling).
BASE = {k: v for k, v in gd.PRIMARY_BPOSD.items() if k != "ms_scaling_factor"}
#: The four contrasts that each change ONE factor, as (A, B): A minus B.
CONTRASTS = {
    "representation_at_0.625": ("bposd_merged", "bposd_component_scaled"),
    "representation_at_1.0": ("bposd_merged_unscaled", "bposd_component"),
    "scaling_on_merged": ("bposd_merged", "bposd_merged_unscaled"),
    "scaling_on_components": ("bposd_component_scaled", "bposd_component"),
}
JUDGED = ["experiments/m0_prior_art/gap_distribution.py", "experiments/m0_prior_art/gap_distribution_2x2_posthoc.py"]


def settings_for(scaling: float) -> dict:
    """The preregistered BP-OSD settings at the given min-sum scaling."""
    return {**BASE, "ms_scaling_factor": float(scaling)}


def edge_set_cost(edges, W) -> Fraction:
    """Exact cost of an edge set over the binary64 weights (what the checker and the scorer read)."""
    return sum((Fraction(W[tuple(e)]) for e in edges), Fraction(0))


def read_records(row: dict) -> list[dict]:
    """The preregistered per-shot records of one distance, refused unless they hash to the summary's digest."""
    body = gzip.decompress((ROOT / row["per_shot"]).read_bytes())
    if hashlib.sha256(body).hexdigest() != row["per_shot_sha256"]:
        raise SystemExit(f"{row['per_shot']}: does not hash to the summary's per_shot_sha256")
    return [json.loads(x) for x in body.decode().splitlines() if x.strip()]


def score(c: dict, fired, rec: dict) -> dict:
    """One candidate as the preregistered runner records it (gap_distribution.main), against the RECORDED reference."""
    consistent = c["odd"] == set(fired) and c["off_graph"] == 0
    entry = {"edges": [list(e) for e in c["edges"]], "cost_exact": str(c["cost_exact"]), "consistent": consistent,
             "multiplicity_max": c["multiplicity_max"], "edges_with_cancelled_copies": c["edges_with_cancelled_copies"],
             "selected_columns": c["selected"], "bp_converged": c["bp_converged"]}
    if consistent and rec["ref_eps"]["accepted"]:
        g_lo, g_hi = gd.gap_interval(c["cost_exact"], Fraction(rec["pymatching"]["cost_exact"]),
                                     Fraction(rec["ref_eps"]["epsilon"]))
        entry["gap_lo"], entry["gap_hi"] = str(g_lo), str(g_hi)
    return entry


def proven_within(entry: dict) -> bool | None:
    """Proven within eps_max of optimal (the upper end of the exact interval at most 1e-6); None off the reference."""
    if "gap_hi" not in entry:
        return None
    return Fraction(entry["gap_hi"]) <= gd.EPS_MAX


def paired(recs: list[dict], a: str, b: str) -> dict:
    """Paired outcomes of two cells on the epsilon reference: discordant counts and the exact two-sided sign test
    (McNemar's exact test) on them. Descriptive and post hoc: no multiplicity correction is claimed."""
    from scipy.stats import binomtest
    only_a = only_b = both = neither = 0
    for r in recs:
        x, y = proven_within(r[a]), proven_within(r[b])
        if x is None or y is None:
            continue
        both += x and y
        neither += (not x) and (not y)
        only_a += x and not y
        only_b += y and not x
    n = only_a + only_b
    return {"a": a, "b": b, "both_proven_within": both, "neither": neither, "only_a": only_a, "only_b": only_b,
            "exact_mcnemar_p_two_sided": float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0}


def _rel(p: pathlib.Path) -> str:
    """A path as the record names it: repository-relative inside the tree, absolute outside it (a test's run)."""
    return p.relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT.resolve()) else str(p)


def _git(*args: str) -> str:
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
    from oneq.certifying_decoder import CertifyingDecoder
    from oneq.matching_cert import BOUNDARY

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="decode only the first N records per distance (a cost probe; refused under evidence/)")
    ap.add_argument("--distances", type=int, nargs="+", default=None)
    a = ap.parse_args(argv)
    if a.limit is not None and a.out_dir.resolve().is_relative_to((ROOT / "evidence").resolve()):
        raise SystemExit("--limit is a cost probe and never writes under evidence/")
    summ = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    if summ.get("preregistration") != PREREG:
        raise SystemExit("not the preregistered gap run")
    env = {k: md.version(k) for k in ("stim", "pymatching", "ldpc", "numpy", "scipy")}
    if any(env[k] != summ["environment"][k] for k in ("stim", "pymatching", "ldpc")):
        raise SystemExit(f"the decoders' versions differ from the preregistered run's: {env} vs {summ['environment']}")

    def judged_sha():
        return {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in JUDGED}
    head = {"commit": _git("rev-parse", "HEAD"), "recorded": "at the start of the run",
            "uncommitted_changes_in_judged_files": bool(_git("status", "--porcelain", "--", *JUDGED)),
            "sha256": judged_sha()}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in summ["rows"]:
        d = row["distance"]
        if a.distances and d not in a.distances:
            continue
        recs = read_records(row)
        if a.limit is not None:
            recs = recs[:a.limit]
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=row["rounds"],
            after_clifford_depolarization=row["p"], before_measure_flip_probability=row["p"],
            after_reset_flip_probability=row["p"], before_round_data_depolarization=row["p"])
        dem = circ.detector_error_model(decompose_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0)))
                 for e in m.edges()]
        cd = CertifyingDecoder(edges, epsilon_max=float(gd.EPS_MAX))
        W = cd.W
        mechs = gd.dem_mechanisms(dem, W, cd.key)
        comp_cols = [mm[0] for mm in mechs]
        merged_cols, merged_p = gd.merged_edge_columns(mechs)
        st = row["bposd_settings"]
        if (len(W), len(comp_cols), len(merged_cols)) != (row["graph_edges"], row["component_columns"],
                                                          st["bposd_merged"]["columns"]) \
                or st["bposd_component"]["columns"] != len(comp_cols):
            raise SystemExit(f"d={d}: the regenerated graph's sizes are not the record's")
        model = {"merged": (merged_cols, gd.column_matrix(merged_cols, dem.num_detectors), merged_p),
                 "component": (comp_cols, gd.column_matrix(comp_cols, dem.num_detectors), [mm[2] for mm in mechs])}
        decs = {name: (model[rep][0], BpOsdDecoder(model[rep][1], error_channel=model[rep][2], **settings_for(s)))
                for (rep, s), name in CELLS.items()}
        read_back = {name: {k: str(getattr(dec, k, "unavailable"))
                            for k in ("ms_scaling_factor", "bp_method", "schedule", "osd_method", "osd_order",
                                      "max_iter")} | {"columns": len(cols)}
                     for name, (cols, dec) in decs.items()}
        for name in PREREGISTERED:      # the reproduced cells must be configured as the record says they ran
            if read_back[name] != st[name]:
                raise SystemExit(f"d={d}: {name} configured {read_back[name]}, the record ran {st[name]}")
        out, secs = [], {name: 0.0 for name in CELLS.values()}
        for rec in recs:
            fired = rec["fired"]
            # THE GRAPH IS THE RECORD'S: every stored edge set re-costed here equals its recorded exact cost
            for who in ("pymatching", "bposd_merged", "bposd_component", "belief_matching"):
                if edge_set_cost(rec[who]["edges"], W) != Fraction(rec[who]["cost_exact"]):
                    raise SystemExit(f"d={d} shot {rec['shot']}: {who}'s edge set costs differently on this graph")
            syn = np.zeros(dem.num_detectors, dtype=np.uint8)
            syn[fired] = 1
            new = {"shot": rec["shot"], "fired": fired, "pymatching": rec["pymatching"], "ref_eps": rec["ref_eps"],
                   "ref_exact_accepted": rec["ref_exact_accepted"]}
            for name, (cols, dec) in decs.items():
                t0 = time.perf_counter()
                sel = np.asarray(dec.decode(syn)).astype(bool).ravel()
                secs[name] += time.perf_counter() - t0
                c = gd.project_selection(cols, sel, W)
                c["bp_converged"] = bool(dec.converge)
                new[name] = score(c, fired, rec)
            for name in PREREGISTERED:  # THE REPRODUCTION GATE
                got, want = new[name], rec[name]
                for k in ("selected_columns", "edges", "cost_exact", "bp_converged", "consistent",
                          "edges_with_cancelled_copies", "gap_lo", "gap_hi"):
                    if got.get(k) != want.get(k):
                        raise SystemExit(f"d={d} shot {rec['shot']}: {name} does not reproduce the record ({k})")
            out.append(new)
        summaries = {name: gd.summarise(out, name) for name in CELLS.values()}
        if a.limit is None:
            for name in PREREGISTERED:
                if summaries[name] != row["decoders"][name]:
                    raise SystemExit(f"d={d}: {name}'s summary does not reproduce the record's")
        path = a.out_dir / f"per_shot_d{d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            for new in out:
                fh.write(json.dumps({"shot": new["shot"], **{nm: new[nm] for nm in NEW}}, separators=(",", ":"))
                         + "\n")
        rows.append({"distance": d, "rounds": row["rounds"], "nontrivial": len(out),
                     "epsilon_reference_shots": sum(1 for r in out if r["ref_eps"]["accepted"]),
                     "settings_read_back": read_back,
                     "reproduced_preregistered_cells_on_every_shot": True,
                     "reproduced_preregistered_summaries": a.limit is None,
                     "decode_seconds": secs,
                     "per_shot": _rel(path),
                     "per_shot_sha256": hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest(),
                     "cells": summaries,
                     "contrasts": {k: paired(out, x, y) for k, (x, y) in CONTRASTS.items()}})
        print(f"d={d}: {len(out)} shots; proven within eps: " + ", ".join(
            f"{nm} {summaries[nm]['epsilon_reference']['proven_eps_optimal']}" for nm in CELLS.values())
            + f"; decode s {json.dumps({k: round(v, 1) for k, v in secs.items()})}", flush=True)
    if judged_sha() != head["sha256"]:
        raise SystemExit("a judged file changed while the run was running; it is attributed to neither tree")
    doc = {"schema": "oneq-gap-distribution-2x2-posthoc/1", "post_hoc": True,
           "label": "POST HOC (external audit 3, 2026-11-06): not preregistered, never merged into the preregistered "
                    "record; the two preregistered cells are reproduced here, the two others are new",
           "preregistered_run": {"summary": _rel(RUN / "summary.json"), "head": summ["head"], "argv": summ["argv"]},
           "preregistration": PREREG, "head": head, "argv": sys.argv[1:] if argv is None else list(argv),
           "limit": a.limit,
           "cells": {name: {"input_model": rep, "ms_scaling_factor": s, "preregistered": name in PREREGISTERED}
                     for (rep, s), name in CELLS.items()},
           "settings_common": BASE, "contrasts": {k: {"a": x, "b": y} for k, (x, y) in CONTRASTS.items()},
           "scoring": "the preregistered scoring: selections projected onto graph edges mod 2, exact rational costs, "
                      "gap interval [max(0, g), g + eps(J)] against the RECORDED epsilon reference",
           "environment": {"python": sys.version.split()[0], "platform": platform.platform(), **env},
           "rows": rows}
    (a.out_dir / "summary.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
