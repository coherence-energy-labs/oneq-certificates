r"""Re-judge the preregistered gap run's REFERENCE verdicts with this tree's checker.

The preregistered run (`evidence/gap_distribution_v4/prereg_2026-11-03/`) was judged by `check_epsilon` as committed
at 2008c81f. Later the same day the checker's input domain was tightened (no coercion of vertex ids, simple graphs
only, packing members must be vertices; 2026-11-03, after the JavaScript differential fuzzer). That change is meant to
alter no verdict on well-formed input. This re-derives, from the same seeds, every shot's PyMatching correction and
packing with the same producer and decides it with the CURRENT checker, and requires the status, the exact epsilon
and the exact lower bound to equal the preregistered record on every shot. It changes nothing in the run's record.

    python experiments/m0_prior_art/gap_distribution_rejudge.py
      -> evidence/gap_distribution_v4/prereg_2026-11-03/rejudged.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder     # noqa: E402
from oneq.matching_cert import BOUNDARY, MatchingCertificate   # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon      # noqa: E402

RUN = ROOT / "evidence" / "gap_distribution_v4" / "prereg_2026-11-03"
JUDGED = ["src/oneq/matching_cert.py", "src/oneq/matching_cert_epsilon.py", "src/oneq/certifying_decoder.py",
          "experiments/m0_prior_art/gap_distribution_rejudge.py"]


def main() -> int:
    import pymatching
    import stim
    summ = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for row in summ["rows"]:
        d, p, seed, shots = row["distance"], row["p"], row["seed"], row["shots"]
        circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=row["rounds"],
                                      after_clifford_depolarization=p, before_measure_flip_probability=p,
                                      after_reset_flip_probability=p, before_round_data_depolarization=p)
        m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0)))
                 for e in m.edges()]
        cd = CertifyingDecoder(edges, epsilon_max=1e-6)
        det, _ = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        recs = [json.loads(x) for x in gzip.decompress((ROOT / row["per_shot"]).read_bytes()).decode().splitlines()]
        if hashlib.sha256(gzip.decompress((ROOT / row["per_shot"]).read_bytes())).hexdigest() != \
                row["per_shot_sha256"]:
            raise SystemExit(f"d={d}: the per-shot record does not hash to its summary")
        by_shot = {r["shot"]: r for r in recs}
        same = differ = 0
        first = None
        for s in range(shots):
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            rec = by_shot[s]
            if rec["fired"] != fired:
                raise SystemExit(f"d={d} shot {s}: the sampler did not reproduce the recorded syndrome")
            mw = tuple(sorted(cd.key((u, v if v >= 0 else BOUNDARY)) for u, v in m.decode_to_edges_array(det[s])))
            r = cd.certify(fired, mw)
            duals = dict(r.cert.blossom_duals) if r.cert is not None else dict(r.best_dual or {})
            v = check_epsilon(edges=edges, syndrome=fired, eps_max=1e-6,
                              cert=MatchingCertificate(matched_edges=mw, node_potentials={}, blossom_duals=duals))
            got = (v.status, None if v.epsilon is None else str(v.epsilon),
                   None if v.lower_bound is None else str(v.lower_bound), [list(e) for e in mw])
            want = (rec["ref_eps"]["status"], rec["ref_eps"]["epsilon"], rec["ref_eps"]["lower_bound"],
                    rec["pymatching"]["edges"])
            if got == want:
                same += 1
            else:
                differ += 1
                first = first or {"shot": s, "now": got[:3], "recorded": want[:3]}
        rows.append({"distance": d, "nontrivial": len(recs), "identical": same, "different": differ,
                     "first_difference": first})
        print(f"d={d}: {same} identical, {differ} different", flush=True)

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    out = {"schema": "oneq-gap-distribution-rejudge/1", "run": summ["head"], "rows": rows,
           "all_identical": all(r["different"] == 0 for r in rows),
           "head": {"commit": git("rev-parse", "HEAD"),
                    "uncommitted_changes_in_judged_files": bool(git("status", "--porcelain", "--", *JUDGED)),
                    "sha256": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                               for p in JUDGED}}}
    (RUN / "rejudged.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0 if out["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
