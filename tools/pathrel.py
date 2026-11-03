r"""One safe way to print a path, because there was none and it cost a run.

`pathlib.Path.relative_to` RAISES when the path is not under the base.
Every tool here ends by printing where it wrote its artifact:

    print(f"evidence -> {(dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst)}")

and every one of those takes `dst` from a `--json` / `--out` argument.
Point any of them outside the repository and the tool **completes all of
its work and then dies formatting the success message**.

MEASURED, twice. The GT-A0 sweep hit it first and was repaired in place
-- one instance fixed, the class left standing. It came back on
2026-08-20 in `verify_d8_closure.py`, which independently re-decided both
d=8 closures, printed "2/2 claimed closure(s) independently confirmed",
and then raised `ValueError` on the next line because the wall had
handed it a staging path under %TEMP%. 122 seconds of correct work,
reported as a failed gate.

So: one implementation, and the next tool that needs it finds this
rather than writing a third copy. The four-class decomposition in this
repository was duplicated across four scripts and had DRIFTED before
anybody noticed; a one-line convenience is exactly the kind of thing
that gets re-typed.

A path outside the base is not an error to this function. It is a path
outside the base, and the honest thing to show a reader is the absolute
path they can actually go and look at.
"""

from __future__ import annotations

import pathlib


def rel(path, base) -> str:
    """`path` relative to `base` when it is under it, else absolute.

    Never raises. Printing where a file went must not be able to fail
    after the file has already been written.
    """
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(pathlib.Path(base).resolve()))
    except (ValueError, OSError):
        return str(p)
