r"""The byte format of every file this lane writes (external audit 2, finding F04 / P3-07, 2026-11-03).

A file whose sha256 is recorded somewhere -- a model in a run record or manifest, a proof, a record a release
statement pins -- is only reproducible if its BYTES are a function of its content alone. Python's text mode writes
"\n" as os.linesep, so the same exporter wrote CRLF on the Windows box the archive was made on and LF on Linux, and a
clean Linux checkout failed seven hash bindings while every mathematical object was identical (the audit's
reproduction; ours, by emulating LF in-process, in tests/test_r198_p3_audit_fixes.py). No lane writer may now leave
the newline to the platform; each output has ONE declared format, the one its archived bytes already have:

  output                                             writer                                format
  -------------------------------------------------  ------------------------------------  ------------------------
  exported models (the engine/checker text format)   export_model.export,                  ASCII, CRLF  (open_model)
    pinned by sha256 in records/ and manifests        morph2026.export_circuit
  records/3c_shift_certificates_w10.jsonl            campaign/finish_3c_certs.py           UTF-8, CRLF  (write_text_crlf)
    (CRLF in the archive; a release statement pins it)
  evidence/circuit_distance_exact/results.json       collect_results.py                    UTF-8, LF    (write_json_lf)
  evidence/circuit_distance_exact/lean_r198_receipt  lean_receipt.py                       UTF-8, LF    (write_json_lf)
  records/recert_*.jsonl, records/pan_crosscheck     recert_*.py, pan_crosscheck.py         UTF-8, LF    (explicit)
  stub modules written for the authors' builders     morph_vs_authors.py, morph2026_...    UTF-8, LF    (temporary)

CRLF for the models is not a preference: 332 archived models and every hash that binds them are CRLF, so CRLF is the
only choice that reproduces them on every platform. A future format may change it under a new format version and
re-pinned hashes, never by a silent platform default. The engine and checker read models with fopen(.., "r") and
scanf-style parsing, to which the CR is whitespace on Linux and is translated away on Windows.

Records that a lane script prints to stdout and the shell redirects into records/ are compared by CONTENT (every test
and generator parses them); their bytes follow the shell and are not hash-reproducible by design.

tests/test_r198_p3_audit_fixes.py holds this: every model writer reproduces the archived bytes under both platform
defaults, an AST scan refuses any text-mode write in the lane that does not name its newline, and the committed
files have the format this table declares.
"""
from __future__ import annotations

import json
import pathlib

MODEL_NEWLINE = "\r\n"
MODEL_ENCODING = "ascii"


def open_model(path):
    """A model file for writing: ASCII with CRLF line ends on every platform (the archived bytes)."""
    return open(path, "w", encoding=MODEL_ENCODING, newline=MODEL_NEWLINE)


def json_lf_bytes(obj, indent: int = 1) -> bytes:
    return (json.dumps(obj, indent=indent) + "\n").encode("utf-8")


def write_json_lf(path: pathlib.Path, obj, indent: int = 1) -> None:
    """JSON + one trailing newline, UTF-8, LF line ends on every platform."""
    pathlib.Path(path).write_bytes(json_lf_bytes(obj, indent))


def write_text_crlf(path: pathlib.Path, text: str) -> None:
    """Text, UTF-8, CRLF line ends on every platform (for archived CRLF records)."""
    pathlib.Path(path).write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
