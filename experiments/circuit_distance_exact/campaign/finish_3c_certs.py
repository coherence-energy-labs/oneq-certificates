r"""R198: finish the 3-colour every-T claim once the 24 time-shift certificates are in.

    python finish_3c_certs.py            # status: which (basis, T) cells are VERIFIED on their swept models
    python finish_3c_certs.py --import   # at 24/24 only: write the record and regenerate results.json

Reads out/cert_shift.jsonl (written by certify_shift.sh), keeps the VERIFIED row per (basis, T), refuses any
certificate whose model hash is not the swept model's, and -- only with all 24 -- writes
evidence/circuit_distance_exact/records/3c_shift_certificates_w10.jsonl with workstation paths redacted, then
runs collect_results.py. Committing stays a session's job (lane tests, precheck, then commit only these files).
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
LANE = HERE.parent
ROOT = LANE.parents[1]
OUT = HERE / "out"
REC = ROOT / "evidence" / "circuit_distance_exact" / "records"
PAT = re.compile(r"[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])*?(?:work|scratchpad|r198_campaign|out)[\\/]")
sys.path.insert(0, str(LANE))
from collect_results import _shift_verified  # noqa: E402  (the same verdict test the claim uses)


def verified(cert_lines: list[str], sweep: dict) -> dict:
    """(basis, T) -> the VERIFIED certificate row on the swept model; SystemExit on a certificate for another model."""
    best = {}
    for line in cert_lines:
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["basis"], r["rounds"])
        if r.get("checker_exit") == 0 and _shift_verified(r.get("checker", ""), 10):
            if r["model_sha256"] != sweep[k]:
                raise SystemExit(f"STOP: certificate {k} is for model {r['model_sha256'][:12]}, the sweep searched {sweep[k][:12]}")
            best[k] = r
    return best


def main():
    sweep = {(r["basis"], r["rounds"]): r["model_sha256"]
             for r in map(json.loads, (REC / "3c_sweep_shift_w10.jsonl").read_text().splitlines()) if r}
    src = OUT / "cert_shift.jsonl"
    best = verified(src.read_text().splitlines() if src.exists() else [], sweep)
    need = sorted(sweep)
    missing = [k for k in need if k not in best]
    hours = sum(r["prover_s"] + r["checker_s"] for r in best.values()) / 3600
    print(f"verified {len(best)}/24 ({hours:.1f} h of prove+check); missing {missing}")
    if "--import" not in sys.argv:
        return
    if missing:
        sys.exit("not importing: the claim needs all 24")
    text = "".join(json.dumps({k: (PAT.sub("<work>/", v) if isinstance(v, str) else v) for k, v in best[c].items()})
                   + "\n" for c in need)
    assert "Users" not in text
    # UTF-8, CRLF on every platform: the archived record's bytes, which a release statement pins (serialization.py)
    import serialization as ser  # noqa: E402  (LANE is on sys.path)
    ser.write_text_crlf(REC / "3c_shift_certificates_w10.jsonl", text)
    subprocess.run([sys.executable, str(LANE / "collect_results.py")], check=True)
    claim = json.loads((ROOT / "evidence" / "circuit_distance_exact" / "results.json").read_text())["three_colour_every_T"]
    print("evidence_class:", claim["evidence_class"])


if __name__ == "__main__":
    main()
