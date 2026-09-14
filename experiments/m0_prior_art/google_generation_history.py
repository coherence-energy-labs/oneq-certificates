r"""The Willow certification rate, generation by generation, as one artifact.

WHY THIS EXISTS. Paper 1's hardware table reports six producer
generations. Generations 1 and 2 were written to
`evidence/google_corpus_gate.json` and then superseded IN PLACE, so their
per-configuration rates survived only as git blobs -- and the claims
provenance gate, which matches numbers against the evidence tree as a bag,
"traced" the generation-1 aggregate 78.44 to two unrelated artifacts that
happen to contain 0.78443. A number that passes a gate by coincidence is
not backed.

This module reads every generation from the exact object it was produced
as -- a pinned commit's blob for generations 1 and 2, the in-tree
artifacts for 3 through 6 -- and records, per configuration and basis,
the nontrivial-shot count, the certified count and the rate, with the
commit and blob id of each source. Figures and the LaTeX table are
generated from this file, never retyped.

It also records COVERAGE, which the prose table did not: generation 1 ran
50,000 shots per configuration; generations 2 and 3 ran partial coverage
(generation 2 has 800 nontrivial shots at d=7, 30 rounds, X), so their aggregates
weight configurations unequally and are not comparable to generation 4's
full 10,000-per-configuration aggregate as a like-for-like rate.

Reproducing it needs a clone WITH HISTORY (generations 1-2 are blobs).
The artifact carries the blob ids so a shallow reader can still verify
any value against `git cat-file -p <blob>` elsewhere.

    python experiments/m0_prior_art/google_generation_history.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "google_generation_history.json"

#: Generation -> (commit, path). Commits are the ones whose messages
#: announce that generation's aggregate; each is pinned by full hash.
HISTORICAL = {
    1: ("9ca8bddca662db1dd3dd16daa8a5f7edc6e2721f",
        "evidence/google_corpus_gate.json"),
    2: ("3760cc72f39a5785b50900b68101843a1d8f56d7",
        "evidence/google_corpus_gate.json"),
}
IN_TREE = {
    3: "evidence/google_corpus_gate_gen3.json",
    4: "evidence/google_corpus_gate_gen4.json",
}
#: Generations 5 and 6 re-ran only the hardest configuration, per basis.
HARDEST = {
    5: {"X": "evidence/corpus_escalated_d7X30.json",
        "Z": "evidence/corpus_escalated_d7Z30.json"},
    6: {"X": "evidence/corpus_gen6_d7X30.json",
        "Z": "evidence/corpus_gen6_d7Z30.json"},
}
HARDEST_CONFIG = "d7_at_q6_7_{basis}_r30"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True,
                          encoding="utf-8").stdout


def _per_config(doc: dict) -> dict[str, dict]:
    """Normalise the two per_config shapes the corpus gate has written."""
    pc = doc["per_config"]
    out: dict[str, dict] = {}
    if isinstance(pc, dict):
        rows = pc.items()
    else:
        rows = ((r["config"], r.get("per_config_detail", r)) for r in pc)
    for name, r in rows:
        n, c = r["nontrivial"], r["certified"]
        out[name] = {"nontrivial": n, "certified": c, "rate": c / n}
    return out


def _generation(doc: dict, source: dict) -> dict:
    per = _per_config(doc)
    n = sum(r["nontrivial"] for r in per.values())
    c = sum(r["certified"] for r in per.values())
    t = doc["totals"]
    if (n, c) != (t["nontrivial"], t["certified"]):
        raise ValueError(f"per-config sums {n}/{c} disagree with the "
                         f"artifact's own totals {t} at {source}")
    return {"source": source, "per_config": per,
            "nontrivial": n, "certified": c, "rate": c / n,
            "forgeries_attempted": t["forgeries_attempted"],
            "forgeries_accepted": t["forgeries_accepted"],
            "min_config_nontrivial": min(r["nontrivial"] for r in per.values()),
            "max_config_nontrivial": max(r["nontrivial"] for r in per.values())}


def build() -> dict:
    gens: dict[int, dict] = {}
    for g, (commit, path) in HISTORICAL.items():
        blob = _git("rev-parse", f"{commit}:{path}").strip()
        doc = json.loads(_git("cat-file", "-p", blob))
        gens[g] = _generation(doc, {"commit": commit, "path": path,
                                    "blob": blob})
    for g, path in IN_TREE.items():
        doc = json.loads((ROOT / path).read_text(encoding="utf-8"))
        gens[g] = _generation(doc, {"path": path})
    for g, per_basis in HARDEST.items():
        per = {}
        for basis, path in per_basis.items():
            t = json.loads((ROOT / path).read_text(encoding="utf-8"))["totals"]
            per[HARDEST_CONFIG.format(basis=basis)] = {
                "nontrivial": t["nontrivial"], "certified": t["certified"],
                "rate": t["certified"] / t["nontrivial"],
                "forgeries_attempted": t["forgeries_attempted"],
                "forgeries_accepted": t["forgeries_accepted"],
                "source": path}
        gens[g] = {"per_config": per, "scope": "d=7, 30 rounds only"}

    # FINAL GENERATION: each configuration at the last generation that ran it.
    final = dict(gens[4]["per_config"])
    final.update({k: {kk: v[kk] for kk in ("nontrivial", "certified", "rate")}
                  for k, v in gens[6]["per_config"].items()})
    fn = sum(r["nontrivial"] for r in final.values())
    fc = sum(r["certified"] for r in final.values())
    forg = (gens[4]["forgeries_attempted"]
            + sum(v["forgeries_attempted"] for v in gens[5]["per_config"].values())
            + sum(v["forgeries_attempted"] for v in gens[6]["per_config"].values()))
    acc = (gens[4]["forgeries_accepted"]
           + sum(v["forgeries_accepted"] for v in gens[5]["per_config"].values())
           + sum(v["forgeries_accepted"] for v in gens[6]["per_config"].values()))
    return {
        "schema": "oneq-google-generation-history/1",
        "how_produced": "experiments/m0_prior_art/google_generation_history.py "
                        "(generations 1-2 read from pinned git blobs)",
        "generations": {str(g): v for g, v in sorted(gens.items())},
        "final_generation": {
            "rule": "each configuration counted at the last generation that "
                    "ran it: generation 4 for eleven configurations, "
                    "generation 6 for d=7, 30 rounds (both bases)",
            "per_config": final, "nontrivial": fn, "certified": fc,
            "rate": fc / fn,
            "forgeries_attempted_gen4_gen5_gen6": forg,
            "forgeries_accepted": acc},
        "coverage_note": "generation 1 ran 50,000 shots per configuration; "
                         "generations 2 and 3 ran partial, unequal coverage, "
                         "so their aggregates weight configurations "
                         "unequally; generation 4 ran every configuration "
                         "at 10,000 shots.",
    }


def main() -> int:
    doc = build()
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for g, v in doc["generations"].items():
        if "rate" in v:
            print(f"gen {g}: {v['certified']:,}/{v['nontrivial']:,} = "
                  f"{100 * v['rate']:.2f}%  (per-config nontrivial "
                  f"{v['min_config_nontrivial']:,}-{v['max_config_nontrivial']:,})")
    f = doc["final_generation"]
    print(f"final: {f['certified']:,}/{f['nontrivial']:,} = "
          f"{100 * f['rate']:.2f}%, forgeries "
          f"{f['forgeries_attempted_gen4_gen5_gen6']:,} attempted, "
          f"{f['forgeries_accepted']} accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
