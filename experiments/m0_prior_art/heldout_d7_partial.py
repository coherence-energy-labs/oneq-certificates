r"""The held-out distance-7 sample as its stopping rule leaves it (docs/PREREG_heldout_d7_2026-11-03.md).

    python -B experiments/m0_prior_art/heldout_d7_partial.py --run-root DIR --stop-record DIR/d7_stop_record.json

The preregistration's stopping rule: "If after 40 core-hours the run has not completed, it stops at a chunk boundary,
every completed chunk is reported as a partial sample with the reason, and no chunk is re-selected or replaced." The
runner's `--merge` refuses a sample with a missing chunk, so this reports the partial one. It runs only on a stopped
run (the stop record written by the watchdog that enforced the rule) and never on a live one: reading a running
sample's verdicts would let them steer a decision about the rest of it.

Three views of each arm, side by side, because which chunks completed is not independent of what they contain: a
chunk finishes late when its shots need the expensive tiers of the ladder, which is where refusals come from.

  completed      every completed chunk (what the rule names), with the preregistered exact one-sided 95%
                 Clopper-Pearson upper bound on the not-proven rate
  dispatched     every chunk that had been handed to a worker when the run was stopped (in the amended order of
                 DEVIATIONS.md, read per arm from the driver's dispatch markers by heldout_d7_watchdog.py), with
                 every nontrivial shot of an unfinished chunk counted as NOT PROVEN: a bound that no completion order
                 can flatter. The nontrivial count of an unfinished chunk is read from the corpus (a shot is trivial
                 when no detector fired), which says nothing about its verdict
  prefix         the longest run of completed chunks from seed 1, for reference

Every view is per arm, with its own sample size, and pooled per round count as the preregistration pools them.

Receipts are verified against their index's sha256 before a chunk is counted, as the runner's own resume does. The
preregistered conditional test is computed only between arms that both have data, and labelled partial.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import sys
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

PREREG = "docs/PREREG_heldout_d7_2026-11-03.md"
OUT = ROOT / "evidence" / "epsilon_rerun" / "heldout_d7_partial.json"
CHUNK = 50
SEEDS = tuple(range(1, 41))
ARMS = {f"{p}_{b}_{r}": r for b in ("X", "Z") for r, p in (("r50", "heldout_d7"), ("r10", "heldout_d7"),
                                                             ("r30", "samedev_d7"))}
CORPUS = {name: f"d7_at_q6_7_{name.split('_')[2]}_{r}" for name, r in ARMS.items()}


def cp_upper_one_sided(k: int, n: int, level: float = 0.95) -> float | None:
    from scipy.stats import beta
    if n == 0:
        return None
    return 1.0 if k == n else float(beta.ppf(level, k + 1, n - k))


def conditional_test(k_h: int, n_h: int, k_d: int, n_d: int) -> float | None:
    """The preregistration's one-sided exact conditional binomial p-value (heldout_willow_d7.conditional_test)."""
    from scipy.stats import binom
    if not n_h or not n_d:
        return None
    k = k_h + k_d
    return 1.0 if k == 0 else float(binom.sf(k_h - 1, k, n_h / (n_h + n_d)))


def completed_chunks(run_root: pathlib.Path, arm: str) -> dict[int, dict]:
    """Seed -> index record, for chunks whose receipts file exists and hashes to the index's sha256."""
    p = run_root / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rp = run_root / r["receipts_file"]
        if rp.exists() and hashlib.sha256(gzip.decompress(rp.read_bytes())).hexdigest() == r["receipts_sha256"]:
            out[r["seed"]] = r
    return out


def aggregate(recs: list[dict]) -> dict:
    """The runner's own merge arithmetic (epsilon_rerun.merge) over the given chunk records."""
    tot, forg, forg_acc = Counter(), Counter(), Counter()
    eps, bogus = Fraction(0), 0
    for r in recs:
        tot.update(r["status"])
        forg.update(r["forgeries"])
        forg_acc.update(r["forgeries_accepted"])
        eps = max(eps, Fraction(r["epsilon_max_accepted"]))
        bogus += r["battery_not_false"]
    n = sum(v for k, v in tot.items() if k != "TRIVIAL")
    acc = tot["EXACT_OPTIMAL"] + tot["EPSILON_OPTIMAL"]
    # the preregistered analysis counts NOT_PROVEN (heldout_willow_d7.main); any other refusal is reported beside it
    return {"chunks": len(recs), "shots": CHUNK * len(recs), "nontrivial": n, "accepted": acc,
            "exact_optimal": tot["EXACT_OPTIMAL"], "not_proven": tot["NOT_PROVEN"],
            "refused_other": n - acc - tot["NOT_PROVEN"],
            "largest_accepted_epsilon": str(eps), "forgeries_attempted": sum(forg.values()),
            "forgeries_accepted": sum(forg_acc.values()), "battery_attacks_not_false": bogus,
            "not_proven_upper_95_one_sided": cp_upper_one_sided(tot["NOT_PROVEN"], n)}


def corpus_nontrivial(arm: str, seeds: list[int]) -> int:
    """Nontrivial shots in the given chunks, read from the corpus: a shot is trivial when no detector fired."""
    if not seeds:
        return 0
    import numpy as np
    from google_corpus_gate import CORPUS as CORPUS_DIR, load_config
    _m, _edges, shots, _meta = load_config(CORPUS_DIR / CORPUS[arm])
    total = 0
    for s in seeds:
        det = shots.window((s - 1) * CHUNK, s * CHUNK)
        if len(det) != CHUNK:
            raise SystemExit(f"{arm} chunk {s}: the corpus window holds {len(det)} shots, not {CHUNK}")
        total += int(np.count_nonzero(np.asarray(det).any(axis=1)))
    return total


def unfinished_of(stop: dict) -> dict[str, list[int]]:
    """Arm -> seeds being computed at the stop. Schema /2 (heldout_d7_watchdog.py, the amended order) lists them per
    arm from the driver's dispatch markers; schema /1 (one arm at a time, the committed order) names one arm."""
    if stop.get("schema") == "oneq-heldout-d7-stop/2":
        return {a: list(v) for a, v in stop["dispatched_unfinished"].items()}
    if stop.get("schema") == "oneq-heldout-d7-stop/1":
        return {stop["current_arm"]: list(stop["dispatched_unfinished_seeds"])} if stop.get("current_arm") else {}
    raise SystemExit(f"not a held-out d7 stop record: schema {stop.get('schema')}")


def analyse(run_root: pathlib.Path, stop: dict, nontrivial_of=corpus_nontrivial) -> dict:
    unfinished = unfinished_of(stop)
    arms = {}
    for arm in ARMS:
        done = completed_chunks(run_root, arm)
        comp = aggregate([done[s] for s in sorted(done)])
        k = 0
        while k < len(SEEDS) and SEEDS[k] in done:
            k += 1
        unf = sorted(set(unfinished.get(arm, [])) - set(done))
        u = nontrivial_of(arm, unf)
        worst_n = comp["nontrivial"] + u
        worst_k = comp["nontrivial"] - comp["accepted"] + u
        arms[arm] = {"rounds": ARMS[arm], "completed_seeds": sorted(done), "completed": comp,
                     "prefix": {"chunks": k, **{x: aggregate([done[s] for s in SEEDS[:k]])[x]
                                                for x in ("nontrivial", "accepted", "not_proven",
                                                          "not_proven_upper_95_one_sided")}},
                     "dispatched_unfinished_seeds": unf, "dispatched_unfinished_nontrivial": u,
                     "dispatched_worst_case": {"chunks": comp["chunks"] + len(unf), "nontrivial": worst_n,
                                               "not_proven_or_unfinished": worst_k,
                                               "upper_95_one_sided": cp_upper_one_sided(worst_k, worst_n)}}
    pooled = {}
    for r in ("r50", "r10", "r30"):
        sel = [a for a in arms.values() if a["rounds"] == r]
        n = sum(a["completed"]["nontrivial"] for a in sel)
        kk = sum(a["completed"]["not_proven"] for a in sel)
        wn = sum(a["dispatched_worst_case"]["nontrivial"] for a in sel)
        wk = sum(a["dispatched_worst_case"]["not_proven_or_unfinished"] for a in sel)
        pooled[r] = {"arms_with_data": [nm for nm, a in arms.items() if a["rounds"] == r and a["completed"]["chunks"]],
                     "nontrivial": n, "not_proven": kk, "not_proven_rate": kk / n if n else None,
                     "not_proven_upper_95_one_sided": cp_upper_one_sided(kk, n),
                     "worst_case_nontrivial": wn, "worst_case_not_proven": wk,
                     "worst_case_upper_95_one_sided": cp_upper_one_sided(wk, wn)}
    tests = {}
    for r in ("r50", "r10"):
        p = conditional_test(pooled[r]["not_proven"], pooled[r]["nontrivial"],
                             pooled["r30"]["not_proven"], pooled["r30"]["nontrivial"])
        tests[f"heldout_{r}_vs_same_producer_dev_r30"] = {
            "p_one_sided": p, "computed": p is not None,
            "note": "partial sample; bases pooled as the preregistration pools them" if p is not None
                    else "not computable: one of the two arms has no completed chunk"}
    return {"schema": "oneq-heldout-willow-d7-partial/1", "preregistration": PREREG,
            "status": "STOPPED_BY_RULE", "stop_record": stop, "arms": arms, "pooled": pooled, "tests": tests,
            "forgeries_accepted_total": sum(a["completed"]["forgeries_accepted"] for a in arms.values()),
            "views": {"completed": "every completed chunk, as the stopping rule names",
                      "dispatched_worst_case": "every dispatched chunk, each unfinished nontrivial shot counted as "
                                               "not proven; completion order cannot flatter it",
                      "prefix": "the longest run of completed chunks from seed 1"}}


#: The producer the preregistration froze: every module the chunk runner imports, read from the run's own tree and
#: held by the generator to the bytes of PRODUCER_COMMIT (paper/tools/make_numbers.py, judged_at_its_commit).
PRODUCER_COMMIT = "2008c81f6de5775ebdd0278bef7e2dc3619906fb"
PRODUCER_FILES = ["src/oneq/certifying_decoder.py", "src/oneq/matching_cert.py", "src/oneq/matching_cert_epsilon.py",
                  "src/oneq/forgeries.py", "src/oneq/moat_dual.py", "src/oneq/moat_growth.py", "src/oneq/exact_lp.py",
                  "src/oneq/repair_impossible.py", "experiments/m0_prior_art/epsilon_rerun.py",
                  "experiments/m0_prior_art/heldout_willow_d7.py", "experiments/m0_prior_art/google_corpus_gate.py"]
#: Deviation 1's driver and watchdog, committed in 4030f88d before the restart (DEVIATIONS.md, Bytes).
DEVIATION_COMMIT = "4030f88d"
DEVIATION_FILES = ["experiments/m0_prior_art/heldout_willow_d7_interleaved.py",
                   "experiments/m0_prior_art/heldout_d7_watchdog.py"]
REPORT_FILES = ["experiments/m0_prior_art/heldout_d7_partial.py"]


def _sha_lf(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(*args: str) -> str | None:
    """git's answer, or None where there is no repository (a reader's extracted bundle): the report's bytes are then
    pinned by sha256 alone and its head says the commit comparison was not possible."""
    import subprocess
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def head(run_root: pathlib.Path, copied_from: pathlib.Path | None, records: pathlib.Path, stop_path: pathlib.Path) -> dict:
    """Who produced the sample and who reports it, as bytes: the producer's modules from the run's tree, the deviation's
    driver and watchdog, this report, the run records, and -- when the sample was copied out of the run's tree -- the
    proof that every index line and receipt file here is byte for byte the run's."""
    src = copied_from or run_root
    h = {"producer": {"commit": PRODUCER_COMMIT, "tree": str(src),
                      "sha256": {p: _sha_lf(src / p) for p in PRODUCER_FILES}},
         "deviation": {"commit": DEVIATION_COMMIT, "sha256": {p: _sha_lf(src / p) for p in DEVIATION_FILES}},
         "report": {"commit": _git("rev-parse", "HEAD"),
                    "repository": "present" if _git("rev-parse", "HEAD") else
                                  "absent: commit comparison not possible, bytes pinned by sha256",
                    "uncommitted_changes_in_judged_files": (bool(_git("status", "--porcelain", "--", *REPORT_FILES))
                                                            if _git("rev-parse", "HEAD") else None),
                    "sha256": {p: _sha_lf(ROOT / p) for p in REPORT_FILES}},
         "records": {"dir": (records.resolve().relative_to(ROOT.resolve()).as_posix()
                             if records.resolve().is_relative_to(ROOT.resolve()) else records.resolve().as_posix()),
                     "stop_record": stop_path.name, "sha256": {}}}
    for name in ("d7_stop_record.json", "d7_amendment_stop_record.json", "d7_watchdog.log", "d7_babysitter.log",
                 "d7_run.log", "d7_run_interleaved.log"):
        p = records / name
        if not p.is_file():
            raise SystemExit(f"run record {name} is missing from {records}")
        h["records"]["sha256"][name] = _sha_lf(p)
    markers = sorted((records / "dispatch").glob("*.json")) if (records / "dispatch").is_dir() else []
    h["records"]["dispatch_markers"] = len(markers)
    amend = json.loads((records / "d7_amendment_stop_record.json").read_text(encoding="utf-8"))
    h["records"]["amendment_stop"] = {k: amend[k] for k in ("stopped", "core_hours_spent", "completed_seeds",
                                                              "in_flight_seeds_lost")}
    if copied_from is not None:
        same = {}
        for arm in ARMS:
            a_idx, b_idx = copied_from / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl", \
                run_root / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
            same[arm] = a_idx.read_bytes() == b_idx.read_bytes() and all(
                (copied_from / r["receipts_file"]).read_bytes() == (run_root / r["receipts_file"]).read_bytes()
                for r in (json.loads(x) for x in b_idx.read_text(encoding="utf-8").splitlines() if x.strip()))
            same[arm] = same[arm] and (copied_from / "evidence" / "instances" / f"matching_{arm}.json").read_bytes() \
                == (run_root / "evidence" / "instances" / f"matching_{arm}.json").read_bytes()
        if not all(same.values()):
            raise SystemExit(f"the copy differs from the run's tree: {[a for a, ok in same.items() if not ok]}")
        h["copied_from"] = {"tree": str(copied_from), "index_receipts_and_instances_identical": True}
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--stop-record", type=pathlib.Path, required=True)
    ap.add_argument("--records", type=pathlib.Path, default=None,
                    help="directory holding both stop records, the logs and the dispatch markers (default: the "
                         "stop record's directory)")
    ap.add_argument("--copied-from", type=pathlib.Path, default=None,
                    help="the run's own tree, when --run-root is a copy of it: the copy is verified byte for byte")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args(argv)
    if not a.stop_record.is_file():
        raise SystemExit("no stop record: this reads a STOPPED run only (a complete one is merged by "
                         "heldout_willow_d7.py --merge, a running one is not read)")
    stop = json.loads(a.stop_record.read_text(encoding="utf-8"))
    if stop.get("schema") not in ("oneq-heldout-d7-stop/1", "oneq-heldout-d7-stop/2"):
        raise SystemExit(f"{a.stop_record} is not a held-out d7 stop record")
    doc = analyse(a.run_root, stop)
    doc["head"] = head(a.run_root, a.copied_from, a.records or a.stop_record.parent, a.stop_record)
    doc["deviations"] = "docs/PREREG_heldout_d7_2026-11-03_DEVIATIONS.md"
    # the recorded 2026-09-17 development rate, reported beside and never pooled, as the complete merge reports it
    import heldout_willow_d7 as hw
    doc["recorded_development_2026_09_17"] = {
        **hw.RECORDED_DEV, "not_proven_upper_95_one_sided": hw.cp_upper_one_sided(hw.RECORDED_DEV["not_proven"],
                                                                                  hw.RECORDED_DEV["nontrivial"]),
        "note": "an earlier producer; reported beside, never pooled"}
    a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(doc["pooled"]))
    return 0 if doc["forgeries_accepted_total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
