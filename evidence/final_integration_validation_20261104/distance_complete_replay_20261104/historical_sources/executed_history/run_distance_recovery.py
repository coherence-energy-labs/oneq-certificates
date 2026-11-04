"""Recover exact omitted proofs, then invoke the unchanged complete-catalog CLI.

Preparation and replay are separate explicit actions. This helper never invokes a
prover, edits tracked sources, or promotes a historical replay record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PLAN_SHA256 = "7cae8f6b95f7a914ca5320320a8a8b6c947d44bb80ff8abe174a700c33112c4b"
HEAD = "15c34c552d427e47bec8aad5b847140bc01b88f6"
ROOT_NAME = "p1_final_qualification_20261104"
RECEIVED = "data/received_distance_proofs_20261104"
HISTORY = "evidence/continuation_20261104/distance_recovery_20261104_history"
REPORT = "evidence/distance_receipt_replay_20261104_recovered.json"
BINDING = "evidence/distance_receipt_replay_20261104_recovered.binding.json"
PRIOR = ("evidence/distance_receipt_replay_20261103.json",
         "evidence/circuit_distance_exact/missing_proofs_20261103.json")
SOURCES = {
    "tools/distance_receipt_inventory.py": "3dc0f4690ec497a1676efd27af18cca81a034deec99cf7f64cf2ff72c21576b7",
    "experiments/circuit_distance_exact/ccheck.c": "263a54bfa2545cdfcf6ebff4973d6cb7c4ac10d6f001eb3794157ba98e877504",
    "experiments/circuit_distance_exact/cdist_prove.c": "2339dad4478e32dd5350424bb9bfaed7ce5bf9839030b647dae43ffb9b2fceab",
}
MANIFESTS = {
    "evidence/circuit_distance_exact/ibm_recert_20261026/manifest.json": "11570b0437b458987dc43ab2765a9d9fa33e69e3103026f4bd213108bc80bace",
    "evidence/circuit_distance_exact/recert_20261102/manifest.json": "8108de5d5955224eeadba40b84366dbb5ba423a24314c765375d501d76fff7f0",
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as f:
        f.write(raw)


def safe_rel(root, relative):
    need(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "unsafe relative path")
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), "path escapes execution root")
    return path


def git(root, *args):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True, check=True,
    ).stdout


def git_state(root):
    return {"head": git(root, "rev-parse", "HEAD").decode().strip(),
            "index_sha256": bytes_sha(git(root, "ls-files", "--stage", "-z"))}


def current_sources(root):
    result = {}
    for rel, expected in {**SOURCES, **MANIFESTS}.items():
        raw = safe_rel(root, rel).read_bytes()
        need(bytes_sha(raw.replace(b"\r\n", b"\n")) == expected, "source/manifest LF pin mismatch: " + rel)
        result[rel] = bytes_sha(raw)
    return result


def validate_plan(plan, catalog):
    need(plan.get("schema") == "oneq-distance-proof-recovery-plan/1", "unexpected plan schema")
    rows = plan.get("files", [])
    need(len(rows) == 30 and plan.get("total_files") == 30, "expected exactly30 recovered proofs")
    need(sum(r["bytes"] for r in rows) == plan.get("total_bytes") == 732193669, "recovery byte total differs")
    missing = {r["id"]: r for r in catalog if r["proof"] is None}
    need(len(catalog) == 304 and len(missing) == 30, "full immutable304 catalog required")
    need(len({r["id"] for r in rows}) == 30 and {r["id"] for r in rows} == set(missing), "recovery identities differ")
    for row in rows:
        expected = missing[row["id"]]
        need(row["sha256"] == expected["proof_sha256"] and row["bytes"] == expected["proof_bytes"],
             "plan proof pin differs from immutable catalog")
        need(row["model_sha256"] == expected["model_sha256"], "plan model pin differs")
        need(row["destination"] == RECEIVED + "/" + row["sha256"] + ".bin", "unexpected recovery destination")


def context(root, plan_path):
    root = root.resolve()
    need(os.name == "nt" and root.name == ROOT_NAME, "only the named private final qualification tree is permitted")
    need(git_state(root)["head"] == HEAD, "unexpected execution HEAD")
    need(not git(root, "ls-files", "--unmerged").strip(), "unmerged index refused")
    need(sha(plan_path) == PLAN_SHA256, "reviewed recovery plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    sources = current_sources(root)
    spec = importlib.util.spec_from_file_location("oneq_distance_inventory", root / "tools/distance_receipt_inventory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = module.load_catalog(root)
    validate_plan(plan, catalog)
    return root, plan, catalog, sources


def checked_file(path, expected, size=None):
    need(path.is_file() and not path.is_symlink(), "missing or symlink input: " + str(path))
    if size is not None:
        need(path.stat().st_size == size, "input size differs: " + str(path))
    need(sha(path) == expected, "input SHA differs: " + str(path))


def input_pins(root, catalog, *, external=None):
    result = {}
    for row in catalog:
        model = row["model"]
        checked_file(model, row["model_sha256"])
        result[model.relative_to(root).as_posix()] = row["model_sha256"]
        if row["proof"] is not None:
            proof = row["proof"]
        elif external is not None:
            proof = Path(external[row["id"]]["source"])
        else:
            proof = root / RECEIVED / (row["proof_sha256"] + ".bin")
        checked_file(proof, row["proof_sha256"], row["proof_bytes"])
        key = proof.relative_to(root).as_posix() if proof.is_relative_to(root) else str(proof)
        result[key] = row["proof_sha256"]
    for rel in PRIOR:
        result[rel] = sha(root / rel)
    return result


def copy_exact(source, destination, expected, size):
    checked_file(source, expected, size)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    checked_file(destination, expected, size)


def preparation(root, plan_path):
    root, plan, catalog, sources = context(root, plan_path)
    data, history = root / RECEIVED, root / HISTORY
    need(not data.exists() and not history.exists(), "new recovery directories required; no overwrite/resume")
    need(not (root / REPORT).exists() and not (root / BINDING).exists(), "dated outputs already exist")
    state = git_state(root)
    external = {r["id"]: r for r in plan["files"]}
    for row in external.values():
        p = Path(row["source"])
        need(p.is_absolute() and p.parent.parent.parent in
             [Path("C:/lq") / n for n in ("recert", "recert_x", "recert_z", "recert_t2x", "recert_t2z")],
             "unreviewed recovery source location")
    before = input_pins(root, catalog, external=external)
    history.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    # This new directory is not covered by existing ignore rules. Its own rule
    # ignores itself and every payload, without touching shared/tracked Git files.
    with (data / ".gitignore").open("xb") as f:
        f.write(b"*\n")
    need(subprocess.run(["git", "--no-optional-locks", "-C", str(root), "check-ignore",
                         RECEIVED + "/probe.bin"], capture_output=True).returncode == 0,
         "new receipt data are not ignored")
    started = {"schema": "oneq-distance-proof-copy/1", "status": "RUNNING", "started_utc": now(),
               "execution_root": str(root), **state, "source_sha256": sources,
               "plan_sha256": PLAN_SHA256, "helper_sha256": sha(Path(__file__)),
               "original_input_sha256": before}
    write_new(history / "copy_started.json", started)
    snapshots = {}
    for rel in [*SOURCES, *MANIFESTS]:
        dst = history / "sources" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy_exact(root / rel, dst, sources[rel], (root / rel).stat().st_size)
        snapshots[dst.relative_to(root).as_posix()] = sha(dst)
    for name, path in (("run_distance_recovery.py", Path(__file__)), ("recovery_plan.json", plan_path)):
        dst = history / name
        copy_exact(path, dst, sha(path), path.stat().st_size)
        snapshots[dst.relative_to(root).as_posix()] = sha(dst)
    for index, row in enumerate(plan["files"], 1):
        copy_exact(Path(row["source"]), safe_rel(root, row["destination"]), row["sha256"], row["bytes"])
        print(f"Recovered {index}/30 {row['id']}", flush=True)
    need(input_pins(root, catalog, external=external) == before, "original inputs changed while copying")
    inputs = input_pins(root, catalog)
    need(current_sources(root) == sources and git_state(root) == state, "source/HEAD/index changed while copying")
    complete = dict(started, status="PASSED", completed_utc=now(), recovered_files=30,
                    recovered_bytes=732193669, source_and_original_input_stable=True,
                    input_sha256=inputs, archived_sha256=snapshots,
                    scope="Hash-qualified availability only; no mathematical replay was executed by preparation.")
    write_new(history / "copy_manifest.json", complete)
    print(json.dumps({"status": "PASSED", "files": 30, "bytes": 732193669,
                      "next": "run_distance_recovery.py --root <same-root> --replay"}, indent=2))


def prepared_context(root, plan_path):
    root, plan, catalog, sources = context(root, plan_path)
    copy_path = root / HISTORY / "copy_manifest.json"
    record = json.loads(copy_path.read_text(encoding="utf-8"))
    need(record.get("status") == "PASSED" and record.get("recovered_files") == 30
         and record.get("recovered_bytes") == 732193669, "successful exact copy required")
    need(record["plan_sha256"] == PLAN_SHA256 and record["helper_sha256"] == sha(Path(__file__)),
         "preparation helper/plan differs")
    need(record["source_sha256"] == sources, "raw source/manifest bytes differ from preparation")
    need(git_state(root) == {k: record[k] for k in ("head", "index_sha256")}, "execution HEAD/index differs")
    for rel, expected in record["archived_sha256"].items():
        checked_file(safe_rel(root, rel), expected)
    actual_bins = {p.name for p in (root / RECEIVED).rglob("*.bin")}
    need(actual_bins == {r["sha256"] + ".bin" for r in plan["files"]}, "received proof inventory differs")
    inputs = input_pins(root, catalog)
    need(inputs == record["input_sha256"], "prepared catalog inputs changed")
    return root, plan, catalog, sources, inputs, record


def validate_report(report, catalog):
    need(report.get("schema") == "oneq.distance-receipt-inventory/1", "wrong native report schema")
    unhashed = {k: v for k, v in report.items() if k != "report_sha256"}
    digest = bytes_sha(json.dumps(unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    need(digest == report.get("report_sha256"), "native report self-hash differs")
    summary = report["summary"]
    need(summary.get("verdict") == "ALL_MANIFESTED_RECEIPTS_VERIFIED"
         and summary.get("status_counts") == {"VERIFIED": 304}, "native verdict/counts differ")
    required = {"manifested_receipts": 304, "verified_receipts": 304, "available_proofs": 304,
                "missing_proofs": 0, "missing_proof_bytes": 0, "failed_receipts": 0,
                "all_manifested_verified": True, "replay_requested": True,
                "verified_global_bounds": 62, "verified_prefix_exclusions": 242}
    for key, expected in required.items():
        need(summary.get(key) == expected, "full replay summary differs: " + key)
    rows = report.get("receipts", [])
    want = {r["id"]: r for r in catalog}
    need(len(rows) == 304 and len({r["id"] for r in rows}) == 304
         and {r["id"] for r in rows} == set(want), "shortened/duplicate/changed receipt catalog")
    for row in rows:
        need(row.get("status") == "VERIFIED" and row.get("mathematical_certificate_valid") is True
             and row.get("checker_exit_code") == 0, "receipt lacks successful mathematical replay")
        for key in ("model_sha256", "proof_sha256", "proof_bytes", "manifest_sha256_lf",
                    "excluded_weight", "maxround", "group"):
            need(row.get(key) == want[row["id"]][key], "receipt input/scope differs: " + key)
    need(report.get("received_files_not_in_catalog") == [], "unexpected supplied receipt")
    checker = report.get("checker", {})
    need(checker.get("source_sha256_lf") == SOURCES["experiments/circuit_distance_exact/ccheck.c"]
         and checker.get("requested_threads") == 8, "checker identity/threads differ")
    for key in ("historical_verdicts_counted_as_fresh_replay", "prefix_to_global_window_theorems_replayed",
                "every_round_count_proven", "hardware_adequacy_proven", "external_human_reproduction_established"):
        need(report.get("limits", {}).get(key) is False, "unsupported enlarged claim: " + key)


def replay(root, plan_path):
    root, plan, catalog, sources, inputs, prepared = prepared_context(root, plan_path)
    history = root / HISTORY
    output, binding = root / REPORT, root / BINDING
    need(not output.exists() and not binding.exists() and not (history / "replay_started.json").exists(),
         "fresh exclusive replay outputs required")
    command = [sys.executable, "-B", "tools/distance_receipt_inventory.py",
               "--received", RECEIVED, "--timeout-seconds", "14400", "--threads", "8", "--output", REPORT]
    started = {"schema": "oneq-distance-replay-qualification/1", "status": "RUNNING",
               "started_utc": now(), "execution_root": str(root), "execution_head": HEAD,
               "index_sha256": prepared["index_sha256"], "command": command,
               "source_sha256": sources, "source_sha256_lf": {**SOURCES, **MANIFESTS},
               "input_sha256": inputs, "plan_sha256": PLAN_SHA256,
               "helper_sha256": sha(Path(__file__)), "copy_manifest_sha256": sha(history / "copy_manifest.json"),
               "archived_sha256": prepared["archived_sha256"],
               "historical_runtime_estimate": plan["historical_timing"],
               "limits": {"historical_verdicts_counted_as_fresh": False, "outside_human_reproduction": False,
                          "hardware_adequacy_proven": False, "prefix_to_global_theorems_replayed": False}}
    write_new(history / "replay_started.json", started)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8")
    stdout, stderr = history / "replay.stdout.log", history / "replay.stderr.log"
    print("Starting complete304 catalog replay: one process,8 OpenMP threads,14400s timeout per receipt.", flush=True)
    t0 = time.perf_counter()
    error, returncode = None, None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            returncode = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err).returncode
        need(returncode == 0, "native replay exited " + str(returncode))
        report = json.loads(output.read_text(encoding="utf-8"))
        validate_report(report, catalog)
    except Exception as exc:
        error = repr(exc)
    source_stable = input_stable = state_stable = helper_stable = False
    try:
        source_stable = current_sources(root) == sources
        input_stable = input_pins(root, catalog) == inputs
        state_stable = git_state(root) == {"head": HEAD, "index_sha256": prepared["index_sha256"]}
        helper_stable = sha(Path(__file__)) == started["helper_sha256"] and sha(plan_path) == PLAN_SHA256
    except Exception as exc:
        error = (error + "; " if error else "") + "post-run pin check: " + repr(exc)
    passed = error is None and source_stable and input_stable and state_stable and helper_stable
    final = dict(started, status="PASSED" if passed else "FAILED", completed_utc=now(),
                 elapsed_seconds=time.perf_counter() - t0, native_exit_code=returncode, error=error,
                 source_stable_during_run=source_stable, input_stable_during_run=input_stable,
                 execution_head_index_stable=state_stable, helper_plan_stable=helper_stable,
                 report={"path": REPORT, "sha256": sha(output), "bytes": output.stat().st_size} if output.is_file() else None,
                 logs={p.relative_to(root).as_posix(): sha(p) for p in (stdout, stderr) if p.is_file()})
    write_new(binding, final)
    print(json.dumps({"status": final["status"], "binding": BINDING, "error": error}, indent=2), flush=True)
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=Path(__file__).with_name("recovery_plan.json"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    try:
        if args.prepare:
            preparation(args.root, args.plan.resolve())
            return 0
        return replay(args.root, args.plan.resolve())
    except Exception as exc:
        print(json.dumps({"status": "REFUSED_OR_FAILED", "error": repr(exc)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

