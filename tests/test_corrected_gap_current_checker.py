"""New checker qualification must not rewrite a measured source identity."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "evidence/corrected_gap_20261103/summary.json"


def test_retained_source_pins_refuse_changed_checker_but_explicit_recheck_qualifies_it():
    original = subprocess.run(
        [sys.executable, str(ROOT / "tools/replay_corrected_gaps.py"), str(SUMMARY)],
        cwd=ROOT, stdin=subprocess.DEVNULL, text=True, capture_output=True,
    )
    assert original.returncode == 1
    assert json.loads(original.stdout)["reason"] == "source hash mismatch: src/oneq/matching_cert_epsilon.py"
    current = subprocess.run(
        [sys.executable, str(ROOT / "tools/recheck_corrected_gaps_current.py"), str(SUMMARY)],
        cwd=ROOT, stdin=subprocess.DEVNULL, text=True, capture_output=True, check=True,
    )
    result = json.loads(current.stdout)
    assert result["accepted"]
    assert result["reference_witnesses_rechecked"] == 1047
    assert set(result["source_changes"]) == {"src/oneq/matching_cert_epsilon.py"}
    assert result["measured_source_commit"] == "282ad3b38f724428582d1102fcfcaf4d2c0966d4"
    assert not result["measured_sources_identical"]
    assert not result["new_measurement"]
    assert not result["historical_scale_campaigns_requalified"]
