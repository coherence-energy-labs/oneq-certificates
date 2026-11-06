r"""One way to digest a source file for a receipt, because two ways disagreed silently.

`evidence/wall_census.json` carries `roster_source_sha256`: a digest of `tools/gate_all.py`
that says "this census describes THAT wall". `tools/gate_all.py` writes it and
`tools/doc_claims_gate.py` re-computes it to decide whether the census may be believed. Two
call sites, one meaning -- which is how they came to disagree with every reader but one.

*** WHAT WENT WRONG. *** Both hashed `path.read_bytes()` raw. `.gitattributes` says
`*.py text eol=lf` and the index agrees, but a Windows checkout materialises CRLF, so the
digest was a property of THE CHECKOUT rather than of the source. Measured on 2026-09-20:

  * the committed census matched `gate_all.py` at 773bbd63 **in its CRLF form**;
  * a detached worktree at `origin/main` -- LF, nothing modified -- failed
    `doc_claims_gate` with five UNCHECKABLE findings.

Because the pre-push hook runs that gate, **a clean clone of this repository could not push
at all**: not on Linux, not in CI, not for an outside contributor. The front page's gate and
skip counts were undecidable for everyone except the machine that happened to run the wall,
and it surfaced as a friendly "if you just edited the wall" message that reads like a local
nuisance rather than a claim nobody else can check.

Normalising the newline makes the digest a property of the SOURCE. That is the same defect
class, and the same words, as `test_published_bytes`: *a clean clone receives different bytes
than the ones our hashes cover.*

Use this function. Do not inline `sha256(read_bytes())` beside it -- a receipt computed two
ways is a receipt that verifies in one place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def normalise(data: bytes) -> bytes:
    """CRLF -> LF. The canonical form `.gitattributes` already declares for `*.py`."""
    return data.replace(b"\r\n", b"\n")


def roster_digest(path: str | Path) -> str:
    """sha256 over the newline-normalised bytes of `path`.

    Invariant, and the whole point: a CRLF checkout and an LF checkout of the same committed
    source return the same digest.
    """
    return hashlib.sha256(normalise(Path(path).read_bytes())).hexdigest()
