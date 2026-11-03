#!/usr/bin/env python3
"""Recheck retained graph-gap evidence with explicitly identified current code.

The original replay remains source-pinned and deliberately refuses changed
checker bytes. This separate command checks compatibility after a checker
contract update. It preserves the measured artifact and reports every source
hash difference. It does not regenerate measurements or qualify old large-run
mutation/hardware campaigns under a new checker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replay_corrected_gaps import ROOT, verify


def recheck(path):
    raw = path.read_bytes()
    artifact = json.loads(raw)
    expected = artifact["head"]["sha256"]
    current = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
               for name in expected}
    changes = {name: {"measured_sha256": expected[name], "current_sha256": current[name]}
               for name in expected if current[name] != expected[name]}
    # Source identity is explicitly NOT asserted by this compatibility mode.
    # The unchanged replay still checks trace/graph hashes, receipts, denominators
    # and aggregate arithmetic in the evidence itself.
    result = verify(path, verify_sources=False)
    return {**result,
            "qualification": "retained-evidence replay under explicitly identified current sources",
            "summary_sha256": hashlib.sha256(raw).hexdigest(),
            "trace_sha256": artifact["trace"]["sha256"],
            "measured_source_commit": artifact["head"]["commit"],
            "measured_sources_identical": not changes,
            "source_changes": changes,
            "current_source_sha256": current,
            "new_measurement": False,
            "historical_scale_campaigns_requalified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        result = recheck(args.summary)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(json.dumps({"accepted": False, "reason": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
