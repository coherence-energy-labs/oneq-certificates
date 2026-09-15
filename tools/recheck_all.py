r"""Re-check every stored certificate a stranger was handed -- COMPLETELY.

This is the command the release README and AUDIT.md advertise. Until
2026-09-14 it rebuilt instances (including integer weights through float
logarithms), read receipt files only if they existed, and exited 0 when the
flat-certificate file was absent -- so a successful run meant "what was
iterated over passed", not "every receipt behind the result was present and
passed" (external audit, findings R-01 and R-03).

It is now the manifest-pinned complete replay in tools/replay_receipts.py:
every one of the receipts named by evidence/receipt_manifest.json, checked by
two independently written checkers against the persisted instances and
integer weight vectors, failing on any missing, duplicated, unlisted or
altered record. No LP, no MIP, no producer, no rebuilt input.

    python tools/recheck_all.py                    # complete
    python tools/replay_receipts.py --subset dev_tree   # an explicitly named subset
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from replay_receipts import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
