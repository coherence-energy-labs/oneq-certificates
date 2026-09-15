r"""Build the pinned manifest of every paper-2 receipt a complete replay must check.

External audit 2026-09-14 (R-01): `recheck_all.py` exited 0 with the flat
certificate file absent and never enforced corpus counts, so "everything the
command iterated over passed" stood in for "every receipt behind the published
result was present and passed". The manifest is the missing contract: it names
every expected receipt, the instance and weight vector it is a proof about,
hashes of its syndrome, candidate and proof payload, and the per-shot record
it must agree with. `tools/replay_receipts.py` verifies the shipped files
against it and fails on anything missing, duplicated, extra or mismatched.

Building the manifest also cross-binds the artifacts and refuses to write if
they disagree: a receipt's syndrome and candidate must equal its per-shot
record's; the record's weight hash must equal the stored vector's; the
record's H_hash must equal the stored instance's.

    python tools/receipt_manifest.py      # writes evidence/receipt_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
OUT = EV / "receipt_manifest.json"

HELDOUT_INSTANCE = {
    "uniform_p01": "bb144_z_code_capacity", "uniform_p02": "bb144_z_code_capacity",
    "stress_p05": "bb144_z_code_capacity", "hetero_p02": "bb144_z_code_capacity",
    "xsector_p02": "bb144_x_code_capacity", "phenom_p01": "bb144_phenom_r4",
    "phenom_p02": "bb144_phenom_r4", "bb_refsched": "bb144_refsched_r3_p002",
}
EXPECTED = {"heldout_flat": 2002, "heldout_tree": 17, "dev_tree": 19, "dev_refsched_tree": 4}


def sha_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def rel(p: pathlib.Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _records(path: pathlib.Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["shot_index"] in out:
            raise SystemExit(f"duplicate shot {r['shot_index']} in {rel(path)}")
        out[r["shot_index"]] = r
    return out


def _record_answer(r: dict):
    rc = r.get("receipt") or {}
    if rc.get("answer") is not None:
        return [int(i) for i in rc["answer"]]
    fb = r.get("fallback") or {}
    return None if fb.get("answer") is None else [int(i) for i in fb["answer"]]


def build() -> dict:
    files: dict[str, str] = {}
    instances = {}
    for name in sorted(set(HELDOUT_INSTANCE.values())):
        p = EV / "instances" / f"{name}.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        files[rel(p)] = sha_file(p)
        instances[name] = {"file": rel(p), "checks_sha256": doc["checks_sha256"],
                           "legacy_H_hash": doc["legacy_H_hash"], "n_variables": doc["n_variables"]}

    def weights(fname):
        p = EV / "weights" / fname
        doc = json.loads(p.read_text(encoding="utf-8"))
        vh = hashlib.sha256(json.dumps([int(x) for x in doc["vector"]]).encode()).hexdigest()
        if vh != doc["spec"]["weights_sha256"]:
            raise SystemExit(f"{fname}: stored vector does not match its recorded hash")
        files[rel(p)] = sha_file(p)
        return rel(p), vh

    receipts: list[dict] = []
    errors: list[str] = []

    def add(cohort, rung, shot, instance, wfile, record_file, syn, cand, proof, bind_syndrome=True):
        wrel, wh = weights(wfile)
        rp = EV / "per_shot" / record_file
        files[rel(rp)] = sha_file(rp)
        rec = _records(rp).get(shot)
        rid = f"{cohort}/{rung}/{shot}"
        if rec is None:
            errors.append(f"{rid}: no per-shot record")
            return
        inst_block = rec.get("instance") or {}
        if bind_syndrome:
            if sorted(inst_block.get("syndrome_support", [])) != sorted(syn):
                errors.append(f"{rid}: receipt syndrome != record syndrome")
            if inst_block.get("weights_sha256") != wh:
                errors.append(f"{rid}: record weights hash != stored vector")
            if inst_block.get("H_hash") != instances[instance]["legacy_H_hash"]:
                errors.append(f"{rid}: record H_hash != stored instance")
        if _record_answer(rec) != sorted(int(i) for i in cand) and _record_answer(rec) != [int(i) for i in cand]:
            errors.append(f"{rid}: receipt candidate != record answer")
        receipts.append({
            "id": rid, "cohort": cohort, "rung": rung, "shot_index": shot,
            "instance": instance, "weights_file": wrel, "weights_vector_sha256": wh,
            "record_file": rel(rp),
            "record_binds": ["syndrome", "candidate", "weights", "instance"] if bind_syndrome
            else ["candidate"],
            "syndrome_sha256": sha_json(sorted(int(i) for i in syn)),
            "candidate_sha256": sha_json(sorted(int(i) for i in cand)),
            "proof_sha256": sha_json(proof)})

    flat_p = EV / "qldpc_heldout_flat_certs.json"
    files[rel(flat_p)] = sha_file(flat_p)
    for c in json.loads(flat_p.read_text(encoding="utf-8"))["certificates"]:
        add("heldout_flat", c["rung"], c["shot_index"], HELDOUT_INSTANCE[c["rung"]],
            f"heldout_{c['rung']}_weights.json", f"heldout_{c['rung']}.jsonl",
            c["syndrome_support"], c["candidate"], c["facet_duals"],
            bind_syndrome=(c["rung"] != "bb_refsched"))

    add_p = EV / "qldpc_heldout_addendum.json"
    files[rel(add_p)] = sha_file(add_p)
    add_doc = json.loads(add_p.read_text(encoding="utf-8"))
    for t in add_doc["bd_trees"]:
        add("heldout_tree", t["rung"], t["shot_index"], HELDOUT_INSTANCE[t["rung"]],
            f"heldout_{t['rung']}_weights.json", f"heldout_{t['rung']}.jsonl",
            t["syndrome_support"], t["candidate"], t["tree"],
            bind_syndrome=(t["rung"] != "bb_refsched"))
    for t in add_doc["dev_bd_trees"]:
        add("dev_tree", t["rung"], t["shot_index"], "bb144_z_code_capacity",
            f"{t['rung']}_weights.json", f"{t['rung']}.jsonl",
            t["syndrome_support"], t["candidate"], t["tree"])
    dr_p = EV / "qldpc_dev_refsched_trees.json"
    files[rel(dr_p)] = sha_file(dr_p)
    for t in json.loads(dr_p.read_text(encoding="utf-8"))["trees"]:
        add("dev_refsched_tree", t["rung"], t["shot_index"], "bb144_refsched_r3_p002",
            "bb_refsched_weights.json", "bb_refsched.jsonl",
            t["syndrome_support"], t["candidate"], t["tree"], bind_syndrome=False)

    ids = [r["id"] for r in receipts]
    if len(ids) != len(set(ids)):
        errors.append("duplicate receipt ids")
    counts = {k: sum(r["cohort"] == k for r in receipts) for k in EXPECTED}
    for k, v in EXPECTED.items():
        if counts[k] != v:
            errors.append(f"cohort {k}: {counts[k]} receipts, expected {v}")
    if errors:
        raise SystemExit("REFUSING TO WRITE A MANIFEST OVER DISAGREEING ARTIFACTS:\n  "
                         + "\n  ".join(errors[:30]))
    return {"schema": "oneq-receipt-manifest/1",
            "what": "every receipt behind paper 2's receipt claims; a complete replay must "
                    "check exactly these, against exactly these files",
            "expected_counts": EXPECTED, "instances": instances,
            "files": dict(sorted(files.items())), "receipts": receipts}


def main() -> int:
    doc = build()
    body = json.dumps(doc, indent=1) + "\n"
    OUT.write_text(body, encoding="utf-8", newline="\n")
    print(f"manifest: {len(doc['receipts'])} receipts, {len(doc['files'])} bound files, "
          f"sha256 {hashlib.sha256(body.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
