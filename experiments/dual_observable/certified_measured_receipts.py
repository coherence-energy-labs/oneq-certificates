r"""Paper 1 Sec. VI E, re-run with every receipt KEPT, replayed by both verifiers, and attacked.

External review 1 (2026-11-03): the section's certificates "were not kept, not replayed and not attacked", against
the paper's own standard. This re-runs the measured-model arm of `certified_measured_lane.py` -- the same six
preregistered records, the same estimator fitted on the first half of each record, the same held-out second half,
the same `certify_hard` -- and this time keeps, per held-out nontrivial shot, the correction and the packing, so that
a stranger can replay them:

  * the measured graph of each record is written as an instance file (`oneq-matching-instance/1`, edges as
    binary64 hex, `edges_sha256` over the compact JSON of the edge list), the format the replay tool reads;
  * every receipt is decided by `check_epsilon` (eps_max = 1e-6) AND by the scaled-integer second verifier
    (`tools/replay_epsilon_receipts.verify_b`) through the replay's own loader; the exact checker's verdict
    (the one Sec. VI E reports) is recorded beside them;
  * the forgery battery (`oneq.forgeries`, 2026-11-03 version: heavier corrections are genuine edge sets) runs on
    every shot with a packing, each attack's falsity re-established by `assert_all_false`.

It must reproduce the lane's counts (`evidence/certified_measured_lane.json`, measured model): nontrivial and
exact-certified per record. It refuses to write if they differ.

    python experiments/dual_observable/certified_measured_receipts.py
      -> evidence/certified_measured_receipts/summary.json, <record>.receipts.jsonl.gz, instances/<record>.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ("src", "experiments/ibm_live", "experiments/dual_observable", "tools"):
    sys.path.insert(0, str(ROOT / p))

from oneq.certifying_decoder import CertifyingDecoder                # noqa: E402
from oneq.dem_measured import MAX_SPAN, measured_graph                # noqa: E402
from oneq.forgeries import assert_all_false, forgeries                # noqa: E402
from oneq.matching_cert import BOUNDARY, MatchingCertificate          # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon                  # noqa: E402
from certified_measured_lane import RECORDS                           # noqa: E402
from measured_model_lane import load                                  # noqa: E402
import replay_epsilon_receipts as rep                                 # noqa: E402

OUT = ROOT / "evidence" / "certified_measured_receipts"
LANE = ROOT / "evidence" / "certified_measured_lane.json"
EPS_MAX = Fraction(1, 10**6)


def _instance(name, edges):
    body = [[int(u), int(v), float(w).hex()] for u, v, w in edges]
    return {"schema": "oneq-matching-instance/1", "name": name, "edges": body,
            "edges_sha256": hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()}


def one_record(path: str, max_span: int):
    doc, dem_edges, det = load(path)
    d, r = doc["circuit"]["distance"], doc["circuit"]["rounds"]
    positions, n_rounds = d - 1, r + 1
    half = det.shape[0] // 2
    g = measured_graph(dem_edges, det[:half], positions, n_rounds, det.shape[1], max_span=max_span)
    edges = list(g.edges)
    det3 = det[half:].reshape(-1, n_rounds, positions)
    cd = CertifyingDecoder(edges)
    import pymatching
    m = pymatching.Matching()
    for (u, v, w) in edges:
        if v == BOUNDARY:
            m.add_boundary_edge(int(u), weight=w)
        else:
            m.add_edge(int(u), int(v), weight=w)
    n_nodes = (r + 1) * (d - 1)
    lines, status = [], Counter()
    nontrivial = exact_accepted = verifiers_agree = 0
    forg, forg_acc, bogus = Counter(), Counter(), 0
    t0 = time.perf_counter()
    for s in range(det3.shape[0]):
        fired = [int(i) for i in np.flatnonzero(det3[s].reshape(-1))]
        if not fired:
            continue
        nontrivial += 1
        syn = np.zeros(n_nodes, dtype=np.uint8)
        syn[fired] = 1
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY)) for a, b in m.decode_to_edges_array(syn))
        res = cd.certify_hard(fired, corr)
        exact_accepted += res.accepted
        duals = dict(res.cert.blossom_duals) if res.cert is not None else dict(res.best_dual or {})
        rec = {"shot": s, "fired": fired, "correction": [list(e) for e in corr],
               "duals": [[sorted(int(x) for x in S), float(z).hex()] for S, z in duals.items() if z > 0]}
        a, b = rep._decide_both(edges, rec, rep._receipt_duals(rec), EPS_MAX)
        same = (a.status == b[0] and a.epsilon == b[1] and a.lower_bound == b[2])
        verifiers_agree += same
        status[a.status] += 1
        rec.update({"exact_checker_accepted": bool(res.accepted), "status": a.status,
                    "epsilon": None if a.epsilon is None else str(a.epsilon),
                    "lower_bound": None if a.lower_bound is None else str(a.lower_bound),
                    "verifiers_agree": bool(same)})
        lines.append(json.dumps(rec, separators=(",", ":")))
        if duals:
            battery = forgeries(cd.W, fired, corr, duals, res.primal)
            some = next(iter(duals))
            battery += [("N_nan_dual", MatchingCertificate(matched_edges=corr, node_potentials={},
                                                           blossom_duals={**duals, some: math.nan}), "non-finite"),
                        ("N_inf_dual", MatchingCertificate(matched_edges=corr, node_potentials={},
                                                           blossom_duals={**duals, some: math.inf}), "non-finite")]
            bogus += len(assert_all_false(cd.W, fired, res.primal, [x for x in battery if x[2] != "non-finite"]))
            for nm, fc, _why in battery:
                forg[nm] += 1
                if check_epsilon(edges=edges, syndrome=fired, cert=fc, eps_max=EPS_MAX).accepted:
                    forg_acc[nm] += 1
    stem = pathlib.Path(path).stem
    body = ("\n".join(lines) + "\n").encode()
    row = {"record": path, "backend": doc.get("backend"), "distance": d, "rounds": r,
           "held_out_shots": int(det3.shape[0]), "fitted_on_shots": int(half), "edges": len(edges),
           "nontrivial": nontrivial, "exact_checker_accepted": exact_accepted,
           "epsilon_status": dict(status), "verifiers_agree_on": verifiers_agree,
           "forgeries_attempted": sum(forg.values()), "forgeries_by_class": dict(forg),
           "forgeries_accepted": sum(forg_acc.values()), "forgeries_accepted_by_class": dict(forg_acc),
           "battery_attacks_not_false": bogus, "seconds": round(time.perf_counter() - t0, 1),
           "receipts": f"evidence/certified_measured_receipts/{stem}.receipts.jsonl.gz",
           "receipts_sha256": hashlib.sha256(body).hexdigest(),
           "instance": f"evidence/certified_measured_receipts/instances/{stem}.json"}
    return row, body, _instance(f"measured_{stem}", edges)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default=None)
    ap.add_argument("--max-span", type=int, default=MAX_SPAN)
    a = ap.parse_args(argv)
    runs = a.runs.split(",") if a.runs else list(RECORDS)
    lane = {r["record"]: r["models"]["measured"] for r in json.loads(LANE.read_text(encoding="utf-8"))["runs"]}
    rows, files = [], []
    for path in runs:
        row, body, inst = one_record(path, a.max_span)
        ref = lane[path]
        row["reproduces_lane"] = (row["nontrivial"], row["exact_checker_accepted"]) == (ref["nontrivial"],
                                                                                       ref["certified"])
        if not row["reproduces_lane"]:
            raise SystemExit(f"{path}: {row['nontrivial']}/{row['exact_checker_accepted']} does not reproduce the "
                             f"lane's {ref['nontrivial']}/{ref['certified']}; not writing")
        if row["forgeries_accepted"] or row["battery_attacks_not_false"]:
            raise SystemExit(f"{path}: a forgery was accepted or an attack was not false; not writing")
        rows.append(row)
        files.append((row, body, inst))
        print(f"{path}: {row['nontrivial']} nontrivial, exact {row['exact_checker_accepted']}, eps "
              f"{row['epsilon_status']}, verifiers agree {row['verifiers_agree_on']}, forgeries "
              f"{row['forgeries_attempted']} (0 accepted), {row['seconds']} s", flush=True)
    (OUT / "instances").mkdir(parents=True, exist_ok=True)
    for row, body, inst in files:
        (ROOT / row["receipts"]).write_bytes(gzip.compress(body, 9, mtime=0))
        (ROOT / row["instance"]).write_text(json.dumps(inst, separators=(",", ":")) + "\n", encoding="utf-8")

    def _git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    judged = ["src/oneq/matching_cert.py", "src/oneq/matching_cert_epsilon.py", "src/oneq/certifying_decoder.py",
              "src/oneq/forgeries.py", "tools/replay_epsilon_receipts.py",
              "experiments/dual_observable/certified_measured_receipts.py"]
    doc = {"schema": "oneq/certified-measured-receipts/v1", "max_span": a.max_span, "eps_max": str(EPS_MAX),
           "runs": rows,
           "totals": {"records": len(rows), "nontrivial": sum(r["nontrivial"] for r in rows),
                      "exact_checker_accepted": sum(r["exact_checker_accepted"] for r in rows),
                      "epsilon_accepted": sum(r["epsilon_status"].get("EXACT_OPTIMAL", 0)
                                              + r["epsilon_status"].get("EPSILON_OPTIMAL", 0) for r in rows),
                      "verifiers_agree_on": sum(r["verifiers_agree_on"] for r in rows),
                      "forgeries_attempted": sum(r["forgeries_attempted"] for r in rows),
                      "forgeries_accepted": 0},
           "head": {"commit": _git("rev-parse", "HEAD"),
                    "uncommitted_changes": bool(_git("status", "--porcelain", "--", *judged)),
                    "sha256": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                               for p in judged}},
           "inputs": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in runs}}
    (OUT / "summary.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"-> {OUT.relative_to(ROOT).as_posix()}/summary.json  totals {doc['totals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
