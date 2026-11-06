r"""Paper 1's held-out distance-7 Willow sample for the full ladder (docs/PREREG_heldout_d7_2026-11-03.md).

    python experiments/m0_prior_art/heldout_willow_d7.py --workers W      # certify every arm, every receipt kept
    python experiments/m0_prior_art/heldout_willow_d7.py --merge          # summaries + the preregistered analysis

Three arms, each in both bases, all run with the producer frozen at the commit that carries the preregistration
(`CertifyingDecoder(edges, epsilon_max=1e-6).certify_hard`, the call epsilon_rerun.py makes for the development
d=7 30-round configurations) through the epsilon rerun's own machinery (chunking, checker, receipt format, forgery
battery) under names that nothing reading the rerun's `willow_*` or `heldout_*` results will see:

  heldout_d7_*_r50   PRIMARY   first N shots of d7_at_q6_7 at 50 rounds (never read in development)
  heldout_d7_*_r10   BRACKET   first N shots at 10 rounds (never read in development)
  samedev_d7_*_r30   SAME-PRODUCER DEVELOPMENT   first N shots at 30 rounds (the development shots), so the
                     held-out comparison is made by one producer; the 2026-09-17 development rate came from an
                     earlier producer (before the 2026-10-21 optimisations)

The held-out configurations must be fetched (`tools/fetch_google_corpus.py --rounds 50 10 --distances 7`) and their
bytes pinned before this runs; `--merge` refuses if any chunk is missing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
import epsilon_rerun as er  # noqa: E402

PREREG = "docs/PREREG_heldout_d7_2026-11-03.md"
#: Fixed by the preregistration from the measured development cost (evidence/heldout_d7_cost_calibration.json).
SHOTS = {"r50": 2000, "r10": 2000, "r30": 2000}
CHUNK = 50
OUT = ROOT / "evidence" / "epsilon_rerun" / "heldout_d7.json"
#: The development rate the abstract quotes, recorded 2026-09-17 by an earlier producer: 2 not proven of 100,000.
RECORDED_DEV = {"nontrivial": 100000, "not_proven": 2}

ARMS = {}
for _b in ("X", "Z"):
    for _r, _prefix in (("r50", "heldout_d7"), ("r10", "heldout_d7"), ("r30", "samedev_d7")):
        _name = f"{_prefix}_{_b}_{_r}"
        ARMS[_name] = _r
        er.CONFIGS[_name] = {"kind": "willow", "corpus_config": f"d7_at_q6_7_{_b}_{_r}", "chunk": CHUNK,
                             "seeds": (1, SHOTS[_r] // CHUNK), "method": "certify_hard",
                             "reproduces": f"preregistered in {PREREG}"}


def cp_upper_one_sided(k: int, n: int, level: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound on a binomial rate."""
    from scipy.stats import beta
    return 1.0 if k == n else float(beta.ppf(level, k + 1, n - k))


def conditional_test(k_h: int, n_h: int, k_d: int, n_d: int) -> float:
    """One-sided exact conditional binomial p-value for H0: rate_heldout <= rate_dev (large k_h is evidence against).

    Given k_h + k_d not-proven shots in total, under equal rates k_h ~ Binomial(k_h + k_d, n_h / (n_h + n_d)).
    """
    from scipy.stats import binom
    k = k_h + k_d
    if k == 0:
        return 1.0
    return float(binom.sf(k_h - 1, k, n_h / (n_h + n_d)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="comma-separated arm names (resume a subset)")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(ARMS)
    if not a.merge:
        for nm in names:
            er.run(nm, a.workers)
        return 0
    rows = {}
    for nm in ARMS:
        er.merge(nm)
        s = json.loads((er.INDEX_DIR / nm / "summary.json").read_text(encoding="utf-8"))
        st = s["status"]
        rows[nm] = {"rounds": ARMS[nm], "nontrivial": s["nontrivial"], "accepted": s["accepted"],
                    "exact_optimal": st.get("EXACT_OPTIMAL", 0), "not_proven": st.get("NOT_PROVEN", 0),
                    "largest_accepted_epsilon": s["largest_accepted_epsilon"],
                    "forgeries_attempted": s["forgeries_attempted"], "forgeries_accepted": s["forgeries_accepted"],
                    "battery_attacks_not_false": s["battery_attacks_not_false"]}
    pooled = {}
    for r in ("r50", "r10", "r30"):
        n = sum(v["nontrivial"] for v in rows.values() if v["rounds"] == r)
        k = sum(v["not_proven"] for v in rows.values() if v["rounds"] == r)
        pooled[r] = {"nontrivial": n, "not_proven": k, "not_proven_rate": k / n if n else None,
                     "not_proven_upper_95_one_sided": cp_upper_one_sided(k, n) if n else None}
    out = {"schema": "oneq-heldout-willow-d7/1", "preregistration": PREREG, "arms": rows, "pooled": pooled,
           "recorded_development_2026_09_17": {**RECORDED_DEV,
                                               "not_proven_upper_95_one_sided": cp_upper_one_sided(
                                                   RECORDED_DEV["not_proven"], RECORDED_DEV["nontrivial"]),
                                               "note": "an earlier producer; reported beside, never pooled"},
           "test_heldout_r50_vs_same_producer_dev_r30": {
               "p_one_sided": conditional_test(pooled["r50"]["not_proven"], pooled["r50"]["nontrivial"],
                                               pooled["r30"]["not_proven"], pooled["r30"]["nontrivial"]),
               "h0": "held-out 50-round not-proven rate <= same-producer development 30-round rate"},
           "test_heldout_r10_vs_same_producer_dev_r30": {
               "p_one_sided": conditional_test(pooled["r10"]["not_proven"], pooled["r10"]["nontrivial"],
                                               pooled["r30"]["not_proven"], pooled["r30"]["nontrivial"])},
           "forgeries_accepted_total": sum(v["forgeries_accepted"] for v in rows.values())}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["pooled"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
