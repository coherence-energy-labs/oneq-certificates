#!/usr/bin/env python3
r"""Reproduce paper 2's frozen held-out campaign from the PUBLIC release alone -- no git history.

External audit 2026-09-15, finding 5: the source snapshots make the frozen revision available, but
the frozen runner still asserts `git rev-parse held-out-freeze`, so a reader with the release and no
repository history could not run it. This is the one tested route:

  1. verify the source snapshots against evidence/revision_snapshots.json with the shipped tool;
  2. extract oneq-source-aba2aec.tar into a SEPARATE reproduction directory;
  3. supply what the frozen tree needs and the snapshot does not carry: the pinned reference
     checker (tools/external/exact_lp_certificate_reference.py, whose sha256 the frozen runner
     asserts) and the freeze identity (evidence/freeze_identity.json), the latter by answering the
     runner's single `git rev-parse held-out-freeze` call in-process -- not one byte of the frozen
     code is edited, and every seed derives from the declared commit exactly as it did in the run
     of record;
  4. run the frozen campaign there, writing into the reproduction directory's own evidence/;
  5. compare every non-timing number of the result with the shipped evidence/qldpc_heldout.json,
     rung by rung, and every regenerated per-shot record file with the shipped one by sha256.

    python tools/reproduce_heldout_public.py --work <dir>          # a fresh directory; hours of solver time
    python tools/reproduce_heldout_public.py --work <dir> --keep   # keep the reproduction tree afterwards

Writes evidence/heldout_public_reproduction.json with the comparison. Dependencies are the
release's REQUIREMENTS (numpy, scipy, ldpc, stim, pymatching).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "evidence" / "freeze_identity.json"
SNAPSHOTS = ROOT / "data" / "revision_snapshots"
REFERENCE = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
SHIPPED = ROOT / "evidence" / "qldpc_heldout.json"
OUT = ROOT / "evidence" / "heldout_public_reproduction.json"
NON_TIMING = ("nontrivial", "trivial_tier0", "sampled", "tier1", "tier2", "exact_certified",
              "float_only_disagreements", "ref_checked", "ref_agreed", "tier3A", "tier3B",
              "residue_unsolved", "weights_spec", "seed")


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="a fresh reproduction directory")
    ap.add_argument("--snapshots", default=str(SNAPSHOTS))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    ident = json.loads(IDENTITY.read_text(encoding="utf-8"))
    work = pathlib.Path(a.work).resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"REFUSED: {work} is not empty; a reproduction must start from nothing")
    work.mkdir(parents=True, exist_ok=True)

    # 1. the snapshot verifies against its manifest, with the shipped tool
    v = subprocess.run([sys.executable, str(ROOT / "tools" / "revision_snapshots.py"), "--verify", "--no-git",
                        "--out", a.snapshots], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if v.returncode != 0:
        raise SystemExit(f"REFUSED: source snapshots do not verify:\n{v.stdout[-800:]}{v.stderr[-800:]}")
    archive = pathlib.Path(a.snapshots) / ident["source_snapshot"]

    # 2. extract into the reproduction directory
    with tarfile.open(archive) as tf:
        top = {m.name.split("/")[0] for m in tf.getmembers()}
        if len(top) != 1:
            raise SystemExit(f"REFUSED: the snapshot has {len(top)} top-level entries")
        tf.extractall(work, filter="data")
    tree = work / next(iter(top))

    # 3. supply the reference checker (pinned by the frozen runner) and the freeze identity
    (tree / "tools" / "external").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFERENCE, tree / "tools" / "external" / REFERENCE.name)
    for d in ("evidence", "evidence/per_shot", "evidence/weights"):
        (tree / d).mkdir(parents=True, exist_ok=True)
    (tree / "evidence" / "freeze_identity.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")

    runner = tree / "experiments" / "m0_prior_art" / "qldpc_heldout_gate.py"
    for p in (str(tree / "src"), str(tree / "experiments" / "m0_prior_art")):
        sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("qldpc_heldout_gate_frozen", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    real_run = mod.subprocess.run

    def answering_run(cmd, *args, **kw):
        if list(cmd[:3]) == ["git", "rev-parse", ident["tag"]]:
            return subprocess.CompletedProcess(cmd, 0, stdout=ident["commit"] + "\n", stderr="")
        return real_run(cmd, *args, **kw)

    mod.subprocess.run = answering_run          # the ONLY thing the frozen code cannot do without git
    print(f"reproducing in {tree} with freeze identity {ident['commit'][:12]} ...", flush=True)
    t0 = time.time()
    cwd = pathlib.Path.cwd()
    try:
        import os
        os.chdir(tree)
        rc = mod.main()
    finally:
        os.chdir(cwd)
    seconds = round(time.time() - t0, 1)

    # 5. compare with the shipped result
    got = json.loads((tree / "evidence" / "qldpc_heldout.json").read_text(encoding="utf-8"))
    ship = json.loads(SHIPPED.read_text(encoding="utf-8"))
    diffs = []
    if got.get("freeze_commit") != ship.get("freeze_commit"):
        diffs.append(f"freeze_commit {got.get('freeze_commit')} vs shipped {ship.get('freeze_commit')}")
    for rung, srow in ship["rungs"].items():
        grow = got["rungs"].get(rung)
        if grow is None:
            diffs.append(f"{rung}: missing in the reproduction")
            continue
        for k in NON_TIMING:
            if k in srow and srow[k] != grow.get(k):
                diffs.append(f"{rung}.{k}: shipped {srow[k]} vs reproduced {grow.get(k)}")
    records = {}
    for p in sorted((tree / "evidence" / "per_shot").glob("heldout_*.jsonl")):
        shipped_p = ROOT / "evidence" / "per_shot" / p.name
        same = shipped_p.exists() and sha(shipped_p) == sha(p)
        # informational, not a difference: verdicts are pinned, certificate BYTES are not (ledger R56b:
        # 58 of 2,002 held-out flats regenerate as a different optimal dual of the same degenerate LP)
        records[p.name] = {"reproduced_sha256": sha(p), "identical_to_shipped": same}
    verdict = {"schema": "oneq-heldout-public-reproduction/1", "passed": rc == 0 and not diffs,
               "runner_exit": rc, "seconds": seconds, "freeze_identity": ident["commit"],
               "snapshot": ident["source_snapshot"], "snapshot_sha256": sha(archive),
               "reproduction_dir": str(tree), "compared_fields": list(NON_TIMING),
               "rungs_compared": len(ship["rungs"]), "per_shot_records": records, "differences": diffs,
               "python": sys.version.split()[0], "command": "python tools/reproduce_heldout_public.py --work <dir>"}
    OUT.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8", newline="\n")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(("PUBLIC REPRODUCTION PASSED" if verdict["passed"] else "PUBLIC REPRODUCTION FAILED")
          + f": {len(ship['rungs'])} rungs compared on {len(NON_TIMING)} non-timing fields, "
          f"{sum(1 for r in records.values() if r['identical_to_shipped'])}/{len(records)} record files byte-identical, "
          f"{seconds} s" + (f"; differences: {diffs[:5]}" if diffs else ""))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
