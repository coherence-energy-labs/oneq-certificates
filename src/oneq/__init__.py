"""ONE-Q: a proof-gated adaptive operating system for quantum computation."""
from importlib import import_module as _import_module
from typing import TYPE_CHECKING

__version__ = "0.0.1"

# Keep the declared public types visible to static tools. Importing a leaf
# checker or decoder need not initialize the entire product API and passport.
if TYPE_CHECKING:
    from .passport import L, Provenance, Verdict, WeakKeyError, build_core, mint
    from .passport import verify as verify_passport
    from .api import BOUNDARY, MatchingCertifier, QldpcCertifier, Receipt, verify

__all__ = ["L", "Provenance", "build_core", "mint", "verify_passport",
           "Verdict", "WeakKeyError",
           "BOUNDARY", "MatchingCertifier", "QldpcCertifier", "Receipt",
           "verify"]

_EXPORTS = {name: (".passport", name) for name in
            ("L", "Provenance", "Verdict", "WeakKeyError", "build_core", "mint")}
_EXPORTS["verify_passport"] = (".passport", "verify")
_EXPORTS.update({name: (".api", name) for name in
                 ("BOUNDARY", "MatchingCertifier", "QldpcCertifier", "Receipt", "verify")})


def __getattr__(name):
    """Load a public export on first use, preserving its original identity."""
    try:
        module, target = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module, __name__), target)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
