#!/usr/bin/env python3
"""Inventory and replay all 304 pinned circuit-distance receipts.

Missing proof blobs are not verified. Prefix proofs exclude only irreducible
errors in their stated prefix; this command does not replay a window theorem.
Only Python's standard library and a C11 compiler are required. Exit 0 means
all 304 mathematical replays passed, 2 means missing evidence, 3 means an
inventory without replay, and 1 means a failed check or malformed input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
CHECKER = "experiments/circuit_distance_exact/ccheck.c"
CHECKER_SHA256_LF = "263a54bfa2545cdfcf6ebff4973d6cb7c4ac10d6f001eb3794157ba98e877504"
# These are manifest-content pins, not claims that the historical verifier ran.
# Text pins use LF; model and proof hashes always bind their exact binary bytes.
MANIFESTS = (
    ("ibm_recert_20261026", "11570b0437b458987dc43ab2765a9d9fa33e69e3103026f4bd213108bc80bace", "observations", 22),
    ("recert_20261102", "8108de5d5955224eeadba40b84366dbb5ba423a24314c765375d501d76fff7f0", "cells", 282),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path, destination=None):
    """Optionally freeze exactly the bytes being hashed for subsequent replay."""
    h, size = hashlib.sha256(), 0
    target = destination.open("wb") if destination is not None else None
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                h.update(block)
                size += len(block)
                if target is not None:
                    target.write(block)
    finally:
        if target is not None:
            target.close()
    return h.hexdigest(), size


def contained_path(directory, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("manifest path must be a nonempty relative path")
    path = (directory/relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("manifest path escapes its evidence directory")
    return path


def load_catalog(root=ROOT):
    """A resealed or shortened manifest cannot silently redefine completeness."""
    catalog, seen = [], set()
    for suite, expected_hash, field, count in MANIFESTS:
        directory = Path(root)/"evidence/circuit_distance_exact"/suite
        raw = (directory/"manifest.json").read_bytes().replace(b"\r\n", b"\n")
        if digest(raw) != expected_hash:
            raise ValueError("manifest pin mismatch: "+suite)
        manifest = json.loads(raw)
        if len(manifest[field]) != count:
            raise ValueError("manifest receipt count mismatch: "+suite)
        for row in manifest[field]:
            ibm = suite == "ibm_recert_20261026"
            family, circuit = ("ibm", "ibm") if ibm else (row["family"], row["circuit"])
            kind, maxround = ("shift_cert", 1) if ibm else (row["kind"], row["maxround"])
            identity = ":".join(map(str, (suite, family, kind, circuit, row["basis"], row["rounds"], row["w"], maxround)))
            if identity in seen:
                raise ValueError("duplicate manifest receipt identity")
            seen.add(identity)
            catalog.append({"id": identity, "suite": suite, "family": family, "circuit": circuit,
                            "basis": row["basis"], "rounds": row["rounds"], "kind": kind,
                            "excluded_weight": row["w"], "maxround": maxround,
                            "group": "full" if ibm else "c3" if family in ("3c", "3c_t2") else "ex",
                            "model": contained_path(directory, row["model"]),
                            "model_sha256": row["model_sha256"],
                            "proof": contained_path(directory, row["proof"]) if row["proof"] else None,
                            "proof_sha256": row["proof_sha256"], "proof_bytes": row["proof_bytes"],
                            "proof_magic": row["proof_magic"],
                            "manifest_sha256_lf": expected_hash})
    return catalog


def index_received(directories):
    """Find supplied raw .bin files by content, never by an asserted filename."""
    index, files = {}, []
    for directory in directories:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError("received-proof directory does not exist: "+str(directory))
        for path in sorted(directory.rglob("*.bin")):
            if not path.is_file() or not path.resolve().is_relative_to(directory):
                continue
            sha, size = file_digest(path)
            index.setdefault(sha, path)
            files.append({"path": str(path), "sha256": sha, "bytes": size})
    return index, files


def build_checker(root, directory):
    raw = (Path(root)/CHECKER).read_bytes().replace(b"\r\n", b"\n")
    if digest(raw) != CHECKER_SHA256_LF:
        raise ValueError("reviewed checker source pin mismatch")
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise ValueError("gcc or clang is required for mathematical replay")
    source = directory/"reviewed_checker.c"
    executable = directory/("reviewed_checker.exe" if os.name == "nt" else "reviewed_checker")
    source.write_bytes(raw)
    started = perf_counter_ns()
    command = [compiler, "-O3", "-std=c11", str(source), "-o", str(executable)]
    completed = subprocess.run(command[:1]+["-fopenmp"]+command[1:],
                               capture_output=True, text=True, timeout=60)
    openmp = completed.returncode == 0
    if not openmp:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode:
        raise ValueError("C checker compilation failed: "+completed.stderr[-4000:])
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=10)
    return executable, {"source": CHECKER, "source_sha256_lf": CHECKER_SHA256_LF,
                        "binary_sha256": file_digest(executable)[0], "compiler": compiler,
                        "compiler_version": version.stdout.splitlines()[0] if version.stdout else "unavailable",
                        "flags": ["-O3", "-std=c11"]+(["-fopenmp"] if openmp else []),
                        "openmp": openmp,
                        "compile_ns": perf_counter_ns()-started}


def claim_scope(raw, receipt):
    if len(raw) < 20:
        raise ValueError("truncated proof header")
    magic, weight, orbits, maxround, anchors = struct.unpack("<4s4i", raw[:20])
    if (magic.decode("ascii") != receipt["proof_magic"] or weight != receipt["excluded_weight"]
            or maxround != receipt["maxround"] or not 0 <= weight <= 16
            or not 0 < anchors <= orbits or maxround < -1):
        raise ValueError("proof header differs from the pinned receipt scope")
    prefix = maxround >= 0
    return {"kind": "PREFIX_IRREDUCIBLE_EXCLUSION" if prefix else "GLOBAL_DISTANCE_LOWER_BOUND",
            "excluded_weight": weight, "max_earliest_detector_round": maxround if prefix else None,
            "irreducible_only": prefix, "orbits": orbits, "anchors": anchors}


def inspect_receipt(receipt, received, directory, *, executable=None, timeout_seconds=600, threads=4):
    """Inspect one pinned catalog row; only a fresh C replay can set VERIFIED."""
    result = {k: v for k, v in receipt.items() if k not in ("model", "proof")}
    result.update(model=str(receipt["model"]), declared_proof=str(receipt["proof"]) if receipt["proof"] else None,
                  status="NOT_CHECKED", verified_scope=None, global_distance_lower_bound=None,
                  all_round_counts_proven=False, mathematical_certificate_valid=False)
    proof = receipt["proof"]
    # A damaged shipped file is reported, not silently replaced by a good copy.
    if proof is None or not proof.is_file():
        proof = received.get(receipt["proof_sha256"])
    result["proof_source"] = str(proof) if proof is not None else None
    result["proof_available"] = proof is not None
    try:
        if not receipt["model"].is_file():
            result["status"] = "MISSING_MODEL"
            return result
        model_snapshot = directory/"model.txt" if executable else None
        model_hash, _ = file_digest(receipt["model"], model_snapshot)
        if model_hash != receipt["model_sha256"]:
            result["status"] = "MODEL_HASH_MISMATCH"
            return result
        if proof is None:
            result["status"] = "MISSING_PROOF"
            return result
        proof_snapshot = directory/"proof.bin" if executable else None
        proof_hash, size = file_digest(proof, proof_snapshot)
        if proof_hash != receipt["proof_sha256"] or size != receipt["proof_bytes"]:
            result["status"] = "PROOF_HASH_OR_SIZE_MISMATCH"
            return result
        with (proof_snapshot or proof).open("rb") as source:
            scope = claim_scope(source.read(20), receipt)
        result["declared_scope"] = scope
        if executable is None:
            result["status"] = "PRESENT_NOT_REPLAYED"
            return result
        started = perf_counter_ns()
        completed = subprocess.run([str(executable), str(model_snapshot), str(proof_snapshot), receipt["group"]],
                                   capture_output=True, text=True, timeout=timeout_seconds,
                                   env=dict(os.environ, OMP_NUM_THREADS=str(threads)))
        result.update(checker_ns=perf_counter_ns()-started, checker_exit_code=completed.returncode,
                      checker_stdout=completed.stdout.strip(), checker_stderr=completed.stderr.strip())
        statement = ("VERIFIED: no irreducible undetectable logical error of weight <= " if scope["irreducible_only"]
                     else "VERIFIED: no undetectable logical error of weight <= ")
        if completed.returncode != 0 or not completed.stdout.startswith(statement+str(receipt["excluded_weight"])+" "):
            result["status"] = "CHECKER_REFUSED"
            return result
        result.update(status="VERIFIED", mathematical_certificate_valid=True, verified_scope=scope)
        if not scope["irreducible_only"]:
            result["global_distance_lower_bound"] = receipt["excluded_weight"]+1
    except subprocess.TimeoutExpired:
        result["status"] = "CHECKER_TIMEOUT"
    except (OSError, ValueError, struct.error) as error:
        result.update(status="INPUT_ERROR", error=str(error))
    return result


def summarize(records, *, replay):
    counts = Counter(record["status"] for record in records)
    missing = [r for r in records if not r["proof_available"]]
    failures = sum(v for k, v in counts.items() if k not in {"VERIFIED", "MISSING_PROOF", "PRESENT_NOT_REPLAYED"})
    complete = bool(records) and counts["VERIFIED"] == len(records)
    verdict = ("REPLAY_FAILED" if failures else "ALL_MANIFESTED_RECEIPTS_VERIFIED" if complete
               else "INCOMPLETE_PROOF_SET" if missing else "INVENTORY_ONLY_NOT_VERIFIED")
    return {"verdict": verdict, "manifested_receipts": len(records), "verified_receipts": counts["VERIFIED"],
            "available_proofs": len(records)-len(missing), "missing_proofs": len(missing),
            "missing_proof_bytes": sum(r["proof_bytes"] for r in missing),
            "failed_receipts": failures, "all_manifested_verified": complete,
            "replay_requested": replay, "status_counts": dict(sorted(counts.items())),
            "verified_global_bounds": sum(r["global_distance_lower_bound"] is not None for r in records),
            "verified_prefix_exclusions": sum(r["status"] == "VERIFIED" and r["verified_scope"]["irreducible_only"] for r in records)}


def run(root=ROOT, *, received=(), replay=True, timeout_seconds=600, threads=4, progress=None):
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be a finite positive number of seconds")
    if type(threads) is not int or not 1 <= threads <= 256:
        raise ValueError("threads must be an integer in [1,256]")
    catalog = load_catalog(root)
    supplied, supplied_files = index_received(received)
    records, checker = [], None
    with tempfile.TemporaryDirectory(prefix="oneq-distance-replay-") as temporary:
        directory = Path(temporary)
        executable = None
        if replay:
            executable, checker = build_checker(root, directory)
            checker["requested_threads"] = threads
        for index, receipt in enumerate(catalog, 1):
            record = inspect_receipt(receipt, supplied, directory, executable=executable,
                                     timeout_seconds=timeout_seconds, threads=threads)
            records.append(record)
            if progress is not None:
                progress(index, len(catalog), record)
    expected = {r["proof_sha256"] for r in catalog}
    report = {"schema": "oneq.distance-receipt-inventory/1", "summary": summarize(records, replay=replay),
              "checker": checker, "receipts": records,
              "environment": {"python": sys.version, "platform": sys.platform},
              "received_files_not_in_catalog": [f for f in supplied_files if f["sha256"] not in expected],
              "limits": {"historical_verdicts_counted_as_fresh_replay": False,
                         "prefix_to_global_window_theorems_replayed": False,
                         "every_round_count_proven": False, "hardware_adequacy_proven": False,
                         "external_human_reproduction_established": False}}
    report["report_sha256"] = digest(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return report


def exit_code(report):
    summary = report["summary"]
    if summary["failed_receipts"]:
        return 1
    if summary["all_manifested_verified"]:
        return 0
    if summary["missing_proofs"]:
        return 2
    return 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received", type=Path, action="append", default=[], help="directory of raw .bin proof files; repeatable")
    parser.add_argument("--inventory-only", action="store_true", help="hash inventory only; no C compilation or proof verification")
    parser.add_argument("--timeout-seconds", type=float, default=600, help="timeout per mathematical check (default 600)")
    parser.add_argument("--threads", type=int, default=4, help="OpenMP threads when available (default 4)")
    parser.add_argument("--output", type=Path, help="save full JSON report here and print its summary")
    args = parser.parse_args(argv)
    def progress(index, total, receipt):
        if index <= 22 or index % 25 == 0 or index == total:
            print(f"[{index}/{total}] {receipt['id']} {receipt['status']}", file=sys.stderr, flush=True)
    try:
        report = run(received=args.received, replay=not args.inventory_only,
                     timeout_seconds=args.timeout_seconds, threads=args.threads, progress=progress)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
        print(json.dumps(report["summary"] if args.output else report, indent=2))
        return exit_code(report)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"verdict": "REPLAY_FAILED", "error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
