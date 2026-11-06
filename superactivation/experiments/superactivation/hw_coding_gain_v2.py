"""Driver for docs/PREREG_superactivation_coding_gain_v2_2026-10-12.md: hw_coding_gain's circuits
and analysis, unchanged, on the chain (148, 149, 150).

  python hw_coding_gain_v2.py validate | submit --i-understand-this-spends-credits | fetch
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("hw_coding_gain", HERE.parent / "hw_coding_gain.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.LAYOUT = [148, 149, 150]
base.PREREG = base.ROOT / "docs" / "PREREG_superactivation_coding_gain_v2_2026-10-12.md"
base.PREREG_SHA256 = "219ba8e1508cf0b41f0baddd1b526866e2317d298d2d62bd2fd2cf312671075c"


def _records_glob(pattern):
    return sorted(base.RECORDS.glob(pattern))


# job records of v2 are kept apart from v1's: same driver code, different prefix
_orig_main = base.main


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "fetch":
        import json
        rec = json.loads(_records_glob("coding_gain_v2_*.job.json")[-1].read_text(encoding="utf-8"))
        from qiskit_ibm_runtime import QiskitRuntimeService
        job = QiskitRuntimeService().job(rec["job_id"])
        if "DONE" not in str(job.status()):
            print(f"job {rec['job_id']}: {job.status()}")
            return 4
        result = job.result()
        counts = {nm: result[i].data.c.get_counts() for i, nm in enumerate(rec["circuit_names"])}
        spans = result.metadata.get("execution", {}).get("execution_spans")
        res = base.analyse(counts)
        res["predictions"] = {
            "H1_gain_positive_3se": bool(res["gain"] > 3 * res["gain_se_approx"]),
            "H2_mirror_branches_within_0.02": bool(abs(res["n2"]["F10"] - res["n2"]["F01"]) < 0.02),
            "H3_gain_in_[0.003,0.035]": bool(0.003 <= res["gain"] <= 0.035),
        }
        out = {"job": rec, "counts": counts, "analysis": res,
               "execution_seconds": (spans.stop - spans.start).total_seconds() if spans else None}
        dest = base.RECORDS / f"coding_gain_v2_{rec['job_id']}.result.json"
        dest.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(res, indent=2))
        print(f"-> {dest}")
        return 0
    rc = _orig_main()
    if mode == "submit" and rc == 0:
        # rename the record the base driver just wrote so v1 and v2 never share a prefix
        recs = _records_glob("coding_gain_*.job.json")
        newest = max(recs, key=lambda p: p.stat().st_mtime)
        if not newest.name.startswith("coding_gain_v2_"):
            newest.rename(newest.with_name(newest.name.replace("coding_gain_", "coding_gain_v2_", 1)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
