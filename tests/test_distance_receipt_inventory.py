"""Receipt availability, scope and fresh replay cannot be substituted for each other."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("distance_receipt_inventory", ROOT/"tools/distance_receipt_inventory.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


@pytest.fixture(scope="module")
def catalog():
    return inventory.load_catalog(ROOT)


@pytest.fixture(scope="module")
def checker(tmp_path_factory):
    if not (shutil.which("gcc") or shutil.which("clang")):
        pytest.skip("fresh mathematical replay requires a C11 compiler")
    directory = tmp_path_factory.mktemp("distance_inventory_checker")
    executable, build = inventory.build_checker(ROOT, directory)
    assert build["source_sha256_lf"] == inventory.CHECKER_SHA256_LF
    return executable


def global_receipt(catalog):
    return next(r for r in catalog if r["circuit"] == "c42a" and r["kind"] == "cert"
                and r["basis"] == "X" and r["rounds"] == 1)


def test_pinned_inventory_keeps_every_missing_receipt(catalog):
    assert len(catalog) == 304
    missing = [r for r in catalog if r["proof"] is None]
    assert len(missing) == 30
    assert sum(r["proof_bytes"] for r in missing) == 732193669
    assert {r["family"] for r in missing} == {"3c", "3c_t2", "morph144"}


def test_missing_proof_snapshot_matches_all_thirty_pinned_receipts(catalog):
    snapshot = json.loads((ROOT/"evidence/circuit_distance_exact/missing_proofs_20261103.json").read_text())
    expected = {r["id"]: r for r in catalog if r["proof"] is None}
    assert len(snapshot["receipts"]) == snapshot["missing_proof_receipts"] == len(expected) == 30
    assert snapshot["missing_proof_bytes"] == sum(r["proof_bytes"] for r in expected.values())
    for recorded in snapshot["receipts"]:
        actual = expected.pop(recorded["id"])
        for key in ("proof_sha256", "model_sha256", "proof_bytes", "excluded_weight", "maxround"):
            assert recorded[key] == actual[key]
        assert recorded["status"] == "MISSING_PROOF"
        assert recorded["mathematical_certificate_valid"] is False
        assert recorded["global_distance_lower_bound"] is None
    assert not expected


def test_deleting_manifest_rows_cannot_redefine_completeness(tmp_path):
    suite = inventory.MANIFESTS[0][0]
    destination = tmp_path/"evidence/circuit_distance_exact"/suite/"manifest.json"
    destination.parent.mkdir(parents=True)
    manifest = json.loads((ROOT/"evidence/circuit_distance_exact"/suite/"manifest.json").read_text())
    manifest["observations"].pop()
    destination.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest pin"):
        inventory.load_catalog(tmp_path)


def test_manifest_text_pins_are_portable_across_crlf_checkouts(tmp_path):
    for suite, _, _, _ in inventory.MANIFESTS:
        relative = Path("evidence/circuit_distance_exact")/suite/"manifest.json"
        destination = tmp_path/relative
        destination.parent.mkdir(parents=True)
        destination.write_bytes((ROOT/relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert len(inventory.load_catalog(tmp_path)) == 304


def test_inventory_only_never_claims_fresh_verification(catalog, tmp_path):
    result = inventory.inspect_receipt(global_receipt(catalog), {}, tmp_path)
    assert result["status"] == "PRESENT_NOT_REPLAYED"
    assert result["mathematical_certificate_valid"] is False
    assert result["global_distance_lower_bound"] is None
    summary = inventory.summarize([result], replay=False)
    assert summary["verified_receipts"] == 0
    assert inventory.exit_code({"summary": summary}) == 3


def test_missing_proof_cannot_borrow_the_historical_verified_flag(catalog, tmp_path):
    receipt = next(r for r in catalog if r["proof"] is None)
    result = inventory.inspect_receipt(receipt, {}, tmp_path)
    assert result["status"] == "MISSING_PROOF"
    assert result["mathematical_certificate_valid"] is False
    assert result["verified_scope"] is None
    assert result["global_distance_lower_bound"] is None
    summary = inventory.summarize([result], replay=True)
    assert summary["all_manifested_verified"] is False
    assert summary["missing_proof_bytes"] == receipt["proof_bytes"]
    assert inventory.exit_code({"summary": summary}) == 2


@pytest.mark.parametrize("prefix", [False, True])
def test_real_replay_preserves_exact_global_or_irreducible_scope(catalog, checker, tmp_path, prefix):
    receipt = catalog[0] if prefix else global_receipt(catalog)
    result = inventory.inspect_receipt(receipt, {}, tmp_path, executable=checker)
    assert result["status"] == "VERIFIED", result
    assert result["mathematical_certificate_valid"] is True
    assert result["verified_scope"]["irreducible_only"] is prefix
    assert result["all_round_counts_proven"] is False
    assert result["global_distance_lower_bound"] == (None if prefix else 6)
    assert result["verified_scope"]["kind"] == ("PREFIX_IRREDUCIBLE_EXCLUSION" if prefix else "GLOBAL_DISTANCE_LOWER_BOUND")


def test_received_proof_is_identified_by_hash_not_filename(catalog, checker, tmp_path):
    receipt = deepcopy(global_receipt(catalog))
    folder = tmp_path/"received"
    folder.mkdir()
    renamed = folder/"arbitrary_name.bin"
    renamed.write_bytes(receipt["proof"].read_bytes())
    receipt["proof"] = None
    received, files = inventory.index_received([folder])
    assert files[0]["sha256"] == receipt["proof_sha256"]
    result = inventory.inspect_receipt(receipt, received, tmp_path, executable=checker)
    assert result["status"] == "VERIFIED"
    assert result["proof_source"] == str(renamed)


def test_wrong_received_bytes_cannot_satisfy_a_missing_proof(catalog, tmp_path):
    receipt = deepcopy(global_receipt(catalog))
    named = tmp_path/receipt["proof"].name
    named.write_bytes(b"correct filename, wrong mathematical evidence")
    receipt["proof"] = None
    received, _ = inventory.index_received([tmp_path])
    result = inventory.inspect_receipt(receipt, received, tmp_path)
    assert result["status"] == "MISSING_PROOF"


def test_corrupted_shipped_proof_is_not_hidden_by_a_good_received_copy(catalog, tmp_path):
    receipt = deepcopy(global_receipt(catalog))
    good = receipt["proof"]
    broken = tmp_path/"broken.bin"
    broken.write_bytes(good.read_bytes()+b"changed")
    receipt["proof"] = broken
    result = inventory.inspect_receipt(receipt, {receipt["proof_sha256"]: good}, tmp_path)
    assert result["status"] == "PROOF_HASH_OR_SIZE_MISMATCH"


def test_resealed_trailing_proof_data_is_rejected_by_actual_checker(catalog, checker, tmp_path):
    receipt = deepcopy(global_receipt(catalog))
    raw = receipt["proof"].read_bytes()+b"trailing bytes are not a proof"
    received = tmp_path/"resealed.bin"
    received.write_bytes(raw)
    receipt.update(proof=received, proof_sha256=hashlib.sha256(raw).hexdigest(), proof_bytes=len(raw))
    result = inventory.inspect_receipt(receipt, {}, tmp_path, executable=checker)
    assert result["status"] == "CHECKER_REFUSED"
    assert result["global_distance_lower_bound"] is None


def test_model_hash_binding_is_enforced(catalog, tmp_path):
    receipt = deepcopy(global_receipt(catalog))
    altered = tmp_path/"altered.txt"
    altered.write_bytes(receipt["model"].read_bytes()+b"\n")
    receipt["model"] = altered
    result = inventory.inspect_receipt(receipt, {}, tmp_path)
    assert result["status"] == "MODEL_HASH_MISMATCH"


def test_scope_cannot_be_promoted_by_changing_only_manifest_metadata(catalog):
    receipt = deepcopy(catalog[0])
    receipt["maxround"] = -1
    with pytest.raises(ValueError, match="scope"):
        inventory.claim_scope(receipt["proof"].read_bytes()[:20], receipt)


def test_unreviewed_checker_source_is_refused(tmp_path):
    source = tmp_path/inventory.CHECKER
    source.parent.mkdir(parents=True)
    source.write_text('int main(void){return 0;}', encoding="ascii")
    with pytest.raises(ValueError, match="source pin"):
        inventory.build_checker(tmp_path, tmp_path)


def test_timeout_is_an_unverified_failure(catalog, checker, tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(inventory.subprocess, "run", timeout)
    result = inventory.inspect_receipt(global_receipt(catalog), {}, tmp_path, executable=checker)
    assert result["status"] == "CHECKER_TIMEOUT"
    assert result["mathematical_certificate_valid"] is False
    assert inventory.exit_code({"summary": inventory.summarize([result], replay=True)}) == 1
