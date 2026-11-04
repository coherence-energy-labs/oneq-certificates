"""Copy the reviewed completed-distance records into the public review checkout, without publishing."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, re, subprocess, sys

PRIVATE = Path(r"C:\Users\Josh\Projects\ONE_Q_wt\certified_model_runtime_20261103")
PUBLIC = Path(r"C:\Users\Josh\Projects\ONE_Q_wt\public_certificates_review_20261103")
SOURCE = PRIVATE / ".build/continuation_20261104/paper_c_build_public_plan_20261104"
SETUP = Path(__file__).absolute().parent
PLAN_SHA = "f1de987a743e3c250a919352d50e85922ec19db5a717cf5a0ed0bc47991e50f0"
HISTORY = PUBLIC / "evidence/final_integration_validation_20261104/distance_complete_replay_20261104"
WALL = PRIVATE / ".build/continuation_20261104/wall_history/final_20261104"

def need(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(body):
    return hashlib.sha256(body).hexdigest()

def raw(path, digest, size=None):
    need(path.is_file() and not path.is_symlink(), "Missing/nonregular file: " + str(path))
    body = path.read_bytes()
    need(sha(body) == digest and (size is None or len(body) == size), "Pin differs: " + str(path))
    return body

def safe(value):
    p = PurePosixPath(value)
    need(not p.is_absolute() and not any(x in ("", ".", "..") for x in value.split("/"))
        and ":" not in value and "\\" not in value, "Unsafe relative path.")
    return p

def identity(root):
    p = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD", "--absolute-git-dir"],
        cwd=root, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=60)
    need(p.returncode == 0, "Cannot read checkout identity.")
    lines = p.stdout.splitlines()
    need(len(lines) == 2, "Unexpected Git identity output.")
    meta = Path(lines[1])
    return {"commit": lines[0], "git_dir": str(meta),
        "HEAD_sha256": sha((meta / "HEAD").read_bytes()),
        "index_sha256": sha((meta / "index").read_bytes())}

def put(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    need(path.read_bytes() == body, "Exclusive copy differs: " + str(path))

def jbytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

def public_context(plan):
    state = {path: sha(raw(PUBLIC / safe(path), pin))
        for path, pin in plan["unchanged_public_files"].items()}
    for path in plan["unchanged_public_absent_paths"]:
        need(not (PUBLIC / safe(path)).exists(), "Historical absence changed: " + path)
    return state

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall-terminal", choices=("execution_finished.json", "failure.json"),
        default="execution_finished.json")
    parser.add_argument("--wall-terminal-sha256", required=True)
    parser.add_argument("--expected-private-head", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    need(re.fullmatch("[0-9a-f]{64}", args.wall_terminal_sha256)
        and re.fullmatch("[0-9a-f]{40}", args.expected_private_head), "Exact identities required.")
    plan_body = raw(SOURCE / "public_propagation_plan.json", PLAN_SHA)
    plan = json.loads(plan_body)
    need(plan["schema"] == "oneq-public-distance-completion-propagation-proposal/1"
        and plan["public_root"] == str(PUBLIC)
        and plan["new_payload_count"] == 15 and plan["new_payload_bytes"] == 1617201,
        "Unreviewed public plan.")
    need(plan["replace_public_manuscript"] is False and plan["copy_private_review_packet"] is False
        and plan["publish_signed_release"] is False, "Scope changed.")
    wall_body = raw(WALL / args.wall_terminal, args.wall_terminal_sha256)
    wall = json.loads(wall_body)
    need(wall.get("source_commit") == "40fd68350368666ca7428033031add0b60928a16"
        and wall.get("status") in ("COMPLETED_GREEN", "COMPLETED_RED",
            "REFUSED_INCOMPLETE_OR_MOVED_EXECUTION", "REFUSED"), "Original wall is not terminal.")
    private_before, public_before = identity(PRIVATE), identity(PUBLIC)
    need(private_before["commit"] == args.expected_private_head
        and public_before["commit"] == plan["expected_public_head"], "Unexpected checkout HEAD.")
    before_context = public_context(plan)
    mutation = plan["document_mutation"]
    need(mutation["path"] == "DISTANCE_PROOF_REPLAY.md", "Unexpected document mutation.")
    before_guide = raw(PUBLIC / mutation["path"], mutation["before_raw_sha256"])
    after_guide = raw(SOURCE / "proposed" / mutation["path"], mutation["after_raw_sha256"])
    need(not HISTORY.exists(), "New public history already exists; never overwrite it.")
    payloads = []
    for item in plan["new_exact_original_payloads"]:
        destination = PUBLIC / safe(item["public_destination"])
        need(destination == HISTORY / safe(item["destination"]), "Payload destination escaped history.")
        body = raw(SOURCE / "payload" / safe(item["public_destination"]), item["sha256"], item["bytes"])
        # The frozen original and its staged copy must still agree.
        raw(Path(item["source_absolute"]), item["sha256"], item["bytes"])
        need(not destination.exists(), "New public payload already exists.")
        payloads.append((item, body))
    need(len(payloads) == 15 and sum(len(body) for _, body in payloads) == 1617201,
        "Original payload inventory differs.")
    helper_body = Path(__file__).read_bytes()
    if not args.apply:
        print(json.dumps({"status": "READY", "new_original_payloads": 15,
            "original_bytes": 1617201, "document": mutation, "wall_original_status": wall["status"],
            "public_head": public_before["commit"], "original_execution_head": plan["execution_head_preserved"],
            "source_or_evidence_writes": 0, "publication": False}))
        return 0
    HISTORY.mkdir(parents=True, exist_ok=False)
    record = {"schema": "oneq-public-distance-completion-adoption/1", "status": "STARTED",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "private_identity_before": private_before, "public_identity_before": public_before,
        "plan_sha256": PLAN_SHA, "helper_sha256": sha(helper_body),
        "original_execution_head_preserved": plan["execution_head_preserved"],
        "wall_original_status": wall["status"], "wall_terminal_sha256": args.wall_terminal_sha256,
        "publication": False, "manuscript_replaced": False, "private_review_packet_copied": False}
    put(HISTORY / "adoption_started.json", jbytes(record))
    try:
        for name, body in (
            ("historical_sources/before/DISTANCE_PROOF_REPLAY.md", before_guide),
            ("historical_sources/reviewed_application/apply_public_distance_completion.py", helper_body),
            ("historical_sources/reviewed_application/public_propagation_plan.json", plan_body),
            ("historical_sources/reviewed_application/reviewed_DISTANCE_PROOF_REPLAY.md", after_guide),
            ("historical_sources/reviewed_application/wall_terminal.json", wall_body),
        ):
            put(HISTORY / name, body)
        for item, body in payloads:
            put(PUBLIC / item["public_destination"], body)
        need(identity(PRIVATE) == private_before and identity(PUBLIC) == public_before,
            "Checkout HEAD/raw index changed during preservation.")
        need(public_context(plan) == before_context, "Unchanged public source context moved.")
        raw(PUBLIC / mutation["path"], mutation["before_raw_sha256"])
        raw(SOURCE / "public_propagation_plan.json", PLAN_SHA)
        need(Path(__file__).read_bytes() == helper_body, "Copier source changed.")
        temporary = PUBLIC / (mutation["path"] + ".distance_completion.tmp")
        need(not temporary.exists(), "Temporary guide destination already exists.")
        put(temporary, after_guide)
        raw(PUBLIC / mutation["path"], mutation["before_raw_sha256"])
        os.replace(temporary, PUBLIC / mutation["path"])
        raw(PUBLIC / mutation["path"], mutation["after_raw_sha256"])
        for item, _ in payloads:
            raw(PUBLIC / item["public_destination"], item["sha256"], item["bytes"])
            raw(Path(item["source_absolute"]), item["sha256"], item["bytes"])
        need(public_context(plan) == before_context, "Unchanged public files moved.")
        need(identity(PRIVATE) == private_before and identity(PUBLIC) == public_before,
            "HEAD/raw index changed after adoption.")
        need(Path(__file__).read_bytes() == helper_body, "Copier source changed after adoption.")
        record["status"] = "APPLIED"
        record["error"] = None
        record["new_original_payloads"] = {item["public_destination"]: item["sha256"] for item, _ in payloads}
        record["document_after_sha256"] = mutation["after_raw_sha256"]
    except BaseException as exc:
        record["status"] = "FAILED_PRESERVED"
        record["error"] = repr(exc)
    record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    put(HISTORY / "adoption.json", jbytes(record))
    print(json.dumps({"status": record["status"], "error": record.get("error"),
        "record": str(HISTORY / "adoption.json"), "record_sha256": sha((HISTORY / "adoption.json").read_bytes())}))
    return 0 if record["status"] == "APPLIED" else 1

if __name__ == "__main__":
    raise SystemExit(main())
