r"""BP-OSD's own observable outcome on the held-out circuit-level shots (audit F-18).

The second external audit (docs/EXTERNAL_AUDIT_2026-09-15.md, F-18) noted that paper 2 reports
the CERTIFIED answer's observable agreement on the 149 nontrivial circuit-level shots (149 of 149)
but never says how BP-OSD's own answer would have done on the same shots. That number is derivable
from the records without re-sampling anything:

  * each per-shot record stores BP-OSD's candidate C and the certified answer A;
  * both satisfy the same syndrome, so C xor A lies in the kernel of H, and the logical observable
    of C differs from that of A exactly when the observable row has odd overlap with C xor A;
  * the record stores whether A matched the sampled observable, so BP-OSD's match follows.

Everything the derivation touches is pinned: the records by evidence/per_shot/MANIFEST.json, the
weights by their own digest, and the detector error model by the circuit/DEM/mechanism-order
digests the frozen runner wrote into evidence/heldout_qldpc_bb_refsched_gate.json. The DEM is
rebuilt here by the same builder (unchanged since the freeze commit) and must hash to those
digests, or the tool refuses. Three further checks can each refuse: C and A must have the same
syndrome, BP-OSD's recorded objective must equal the pinned weights summed over C, a tier-1
receipt must carry C == A, a tier-2 receipt a strictly lighter A, and a branch-dual receipt an A
no heavier than C (the two branch-dual shots are optimum hits the flat certificate could not prove,
so there C == A).

Writes evidence/bposd_circuit_outcome.json. Run: python tools/bposd_circuit_outcome.py
With --check it writes nothing and instead re-derives everything and compares it with the
shipped file (the release rehearsal runs it this way).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

SUMMARY = ROOT / "evidence" / "heldout_qldpc_bb_refsched_gate.json"
RECORDS = ROOT / "evidence" / "per_shot" / "heldout_bb_refsched.jsonl"
WEIGHTS = ROOT / "evidence" / "weights" / "heldout_bb_refsched_weights.json"
MANIFEST = ROOT / "evidence" / "per_shot" / "MANIFEST.json"
OUT = ROOT / "evidence" / "bposd_circuit_outcome.json"


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parity(support, row: set) -> int:
    return sum(1 for i in support if i in row) % 2


def syndrome(support, checks) -> tuple:
    s = set(support)
    return tuple(sum(1 for i in cc if i in s) % 2 for cc in checks)


def bposd_match_from_record(rec: dict, obs_row: set) -> tuple[bool, bool]:
    """(same_observable_as_certified, bposd_matched_the_sampled_observable).

    obs_bit = parity(A) xor (not match_A); BP-OSD matches iff parity(C) == obs_bit.
    """
    cand = rec["bposd"]["candidate"]
    ans = rec["receipt"]["answer"]
    match_a = bool(rec["retrospective_simulation"]["observable_match"])
    p_a, p_c = parity(ans, obs_row), parity(cand, obs_row)
    obs_bit = p_a ^ (0 if match_a else 1)
    return (p_a == p_c), (p_c == obs_bit)


COMPARED = ("records", "summary", "weights", "observable_row_sha256", "observable_row_size",
            "counts", "shots", "binding_rebuilt_and_matched")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare with the shipped evidence file; write nothing")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    binding = summary["binding"]

    # ---- the records are the ones the manifest pins ----
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rung = None
    for block in manifest.values():
        if isinstance(block, dict):
            for name, r in (block.get("rungs") or {}).items():
                if isinstance(r, dict) and r.get("records_file", "").endswith("heldout_bb_refsched.jsonl"):
                    rung = r
    if rung is None:
        print("REFUSING: the per-shot manifest does not pin heldout_bb_refsched.jsonl", file=sys.stderr)
        return 2
    if sha256_file(RECORDS) != rung["records_sha256"]:
        print("REFUSING: heldout_bb_refsched.jsonl does not match the manifest's digest", file=sys.stderr)
        return 2
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    wvec = [int(x) for x in weights["vector"]]
    wblob_sha = hashlib.sha256(json.dumps(wvec).encode()).hexdigest()
    if wblob_sha != binding["weights_sha256"] or wblob_sha != weights["spec"]["weights_sha256"]:
        print("REFUSING: the weights file does not match the pinned weights digest", file=sys.stderr)
        return 2

    # ---- rebuild the DEM with the frozen builder and bind it ----
    from qldpc_bb_refsched_gate import build_refsched_circuit  # noqa: E402
    from qldpc_circuit_gate import dem_to_matrices  # noqa: E402
    rounds = int(summary["circuit"].split("r=")[1].split(",")[0])
    p = float(summary["circuit"].split("p=")[1])
    circ = build_refsched_circuit(rounds, p)
    dem = circ.detector_error_model(decompose_errors=False)
    got = {"circuit_sha256": hashlib.sha256(str(circ).encode()).hexdigest(),
           "dem_sha256": hashlib.sha256(str(dem).encode()).hexdigest()}
    checks, wllr, obs_rows, n_mech = dem_to_matrices(dem)
    got["mechanism_order_sha256"] = hashlib.sha256(json.dumps(
        [sorted(int(i) for i in cc) for cc in checks]).encode()).hexdigest()
    wint = [int(math.floor(1000 * float(wllr[i]) + 0.5)) for i in range(n_mech)]
    got["weights_sha256"] = hashlib.sha256(json.dumps(wint).encode()).hexdigest()
    for k, v in got.items():
        if binding[k] != v:
            print(f"REFUSING: rebuilt {k} {v[:16]} != pinned {binding[k][:16]}", file=sys.stderr)
            return 2
    if wint != wvec:
        print("REFUSING: rebuilt integer weights differ from the shipped weights file", file=sys.stderr)
        return 2
    if set(obs_rows) != {0}:
        print(f"REFUSING: expected one logical observable, found rows {sorted(obs_rows)}", file=sys.stderr)
        return 2
    obs_row = set(int(i) for i in obs_rows[0])

    # ---- per shot ----
    rows, refusals = [], []
    counts = {"shots": 0, "identical": 0, "differing": 0, "differing_same_observable": 0,
              "differing_other_observable": 0, "bposd_match": 0, "certified_match": 0,
              "by_tier": {}}
    with open(RECORDS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cand = [int(i) for i in rec["bposd"]["candidate"]]
            ans = [int(i) for i in rec["receipt"]["answer"]]
            tier = rec["receipt"]["tier"]
            idx = rec["shot_index"]
            if syndrome(cand, checks) != syndrome(ans, checks):
                refusals.append(f"shot {idx}: candidate and answer have different syndromes")
                continue
            w_c, w_a = sum(wint[i] for i in cand), sum(wint[i] for i in ans)
            if w_c != int(rec["bposd"]["objective"]):
                refusals.append(f"shot {idx}: recorded objective {rec['bposd']['objective']} != {w_c}")
                continue
            if tier == "tier1" and sorted(cand) != sorted(ans):
                refusals.append(f"shot {idx}: tier-1 receipt whose answer is not the candidate")
                continue
            if tier == "tier2" and not w_a < w_c:
                refusals.append(f"shot {idx}: tier-2 receipt whose answer is not strictly lighter")
                continue
            if tier == "tierBD" and not w_a <= w_c:
                # a branch-dual tree proves the exact optimum; BP-OSD may already sit on it
                # (an optimum hit the flat certificate could not prove), so equality is allowed
                refusals.append(f"shot {idx}: branch-dual receipt whose answer is heavier than the candidate")
                continue
            same_obs, b_match = bposd_match_from_record(rec, obs_row)
            c_match = bool(rec["retrospective_simulation"]["observable_match"])
            counts["shots"] += 1
            counts["by_tier"][tier] = counts["by_tier"].get(tier, 0) + 1
            counts["certified_match"] += int(c_match)
            counts["bposd_match"] += int(b_match)
            if sorted(cand) == sorted(ans):
                counts["identical"] += 1
            else:
                counts["differing"] += 1
                counts["differing_same_observable" if same_obs else "differing_other_observable"] += 1
            rows.append({"shot_index": idx, "tier": tier, "weight_candidate": w_c,
                         "weight_certified": w_a, "identical": sorted(cand) == sorted(ans),
                         "same_observable": same_obs, "certified_match": c_match,
                         "bposd_match": b_match})
    if refusals:
        print("REFUSING: " + "; ".join(refusals[:5]) + (" ..." if len(refusals) > 5 else ""),
              file=sys.stderr)
        return 2
    if counts["shots"] != int(summary["nontrivial"]):
        print(f"REFUSING: {counts['shots']} records vs {summary['nontrivial']} nontrivial in the summary",
              file=sys.stderr)
        return 2
    if [counts["certified_match"], counts["shots"]] != list(summary["observable_agreement"]):
        print("REFUSING: the certified agreement recomputed from records differs from the summary",
              file=sys.stderr)
        return 2

    out = {"schema": "oneq-bposd-circuit-outcome/1",
           "what": ("BP-OSD's own answer scored against the sampled logical observable on the held-out "
                    "circuit-level shots, derived from the records (candidate, certified answer, the "
                    "certified answer's match) and the pinned DEM's observable row; nothing re-sampled"),
           "records": {"file": (RECORDS.relative_to(ROOT).as_posix() if RECORDS.is_relative_to(ROOT) else str(RECORDS)), "sha256": rung["records_sha256"]},
           "summary": {"file": (SUMMARY.relative_to(ROOT).as_posix() if SUMMARY.is_relative_to(ROOT) else str(SUMMARY)), "sha256": sha256_file(SUMMARY)},
           "weights": {"file": (WEIGHTS.relative_to(ROOT).as_posix() if WEIGHTS.is_relative_to(ROOT) else str(WEIGHTS)), "sha256": sha256_file(WEIGHTS)},
           "binding_rebuilt_and_matched": sorted(got),
           "observable_row_size": len(obs_row),
           "observable_row_sha256": hashlib.sha256(json.dumps(sorted(obs_row)).encode()).hexdigest(),
           "counts": counts,
           "shots": rows,
           "seconds": round(time.perf_counter() - t0, 1)}
    c = counts
    if a.check:
        if not OUT.exists():
            print(f"REFUSING: --check needs the shipped {OUT.name}, which is absent", file=sys.stderr)
            return 2
        shipped = json.loads(OUT.read_text(encoding="utf-8"))
        differing = [k for k in COMPARED if shipped.get(k) != out[k]]
        if differing:
            print(f"REFUSING: the shipped {OUT.name} differs from today's derivation in {differing}",
                  file=sys.stderr)
            return 2
        print(f"BP-OSD CIRCUIT OUTCOME PASSED (re-derived and identical to the shipped file): "
              f"{c['bposd_match']}/{c['shots']} BP-OSD vs {c['certified_match']}/{c['shots']} certified, "
              f"{c['differing']} replaced, {c['differing_other_observable']} in another class; "
              f"{out['seconds']} s")
        return 0
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"BP-OSD CIRCUIT OUTCOME PASSED: {c['bposd_match']}/{c['shots']} BP-OSD answers match the sampled "
          f"observable vs {c['certified_match']}/{c['shots']} certified; the certificate replaced "
          f"{c['differing']} answers ({c['differing_same_observable']} same observable, "
          f"{c['differing_other_observable']} different); tiers {c['by_tier']}; "
          f"{out['seconds']} s")
    shown = OUT.relative_to(ROOT).as_posix() if OUT.is_relative_to(ROOT) else str(OUT)
    print(f"-> {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
