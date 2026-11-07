"""The superactivation bundle's requirements.txt lists every third-party module its shipped tests import.

External audit 4, R5 (2026-11-07): in a fresh clone of the public repository, `pip install -r requirements.txt` and
`pytest tests/` from `superactivation/` failed 2 tests (tests/test_superactivation_hw_two_use.py), which import
`qiskit_ibm_runtime.fake_provider` INSIDE the test functions, while requirements.txt listed qiskit-ibm-runtime among
the packages not needed. The file had been "checked against every module the bundle's code imports", by hand, and a
function-level import is what such a reading misses. This check walks EVERY import node of every shipped test file, at
any depth, and requires its top-level module to be the standard library, a module of the tree, or a pinned requirement.

What it does not claim: an import reached only THROUGH a lane module (a test imports a module of the tree that imports
a package) is not followed, and `pytest.importorskip` is a call, not an import, so a test that skips on a missing
package is not counted (it skips by design). Both test files and requirements are found the same way in the
repository (packaging/superactivation/requirements.txt) and in the bundle (its top-level requirements.txt).
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: the shipped tests (tools/build_certificate_bundle.py, _P4_SOURCES)
TEST_GLOBS = ("test_superactivation*.py", "test_r142_*.py", "test_nuse_certificate.py")
#: import name -> distribution name, where they differ
DIST = {"flint": "python-flint", "qiskit_ibm_runtime": "qiskit-ibm-runtime", "qiskit_aer": "qiskit-aer",
        "yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn"}
#: where the tree's own modules live (what a test's sys.path insertions and conftest.py reach)
LOCAL_DIRS = ("src", "experiments", "tools", "paper/tools", "tests")


def requirements_file() -> pathlib.Path:
    repo = ROOT / "packaging" / "superactivation" / "requirements.txt"
    return repo if repo.is_file() else ROOT / "requirements.txt"


def pinned(text: str) -> set[str]:
    """distribution names of the non-comment lines, normalized (case, '-' and '_' alike)"""
    out = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return out


def local_modules() -> set[str]:
    names = {"conftest"}
    for d in LOCAL_DIRS:
        base = ROOT / d
        if base.is_dir():
            for p in base.rglob("*.py"):
                if "__pycache__" not in p.parts:
                    names.add(p.stem)
                    names.update(part for part in p.relative_to(base).parts[:-1])
    return names


def third_party_imports(source: str, local: set[str]) -> set[str]:
    """top-level modules of every import node in `source`, at any depth, that are neither stdlib nor local"""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module]
        else:
            continue
        for m in mods:
            top = m.split(".")[0]
            if top != "__future__" and top not in sys.stdlib_module_names and top not in local:
                found.add(top)
    return found


def missing_requirements(sources: dict[str, str], req_text: str, local: set[str]) -> dict[str, list[str]]:
    """{distribution: [files importing it]} for every third-party import that the requirements do not pin"""
    have = pinned(req_text)
    miss: dict[str, list[str]] = {}
    for name, src in sources.items():
        for top in third_party_imports(src, local):
            dist = DIST.get(top, top).lower().replace("_", "-")
            if dist not in have:
                miss.setdefault(dist, []).append(name)
    return miss


def shipped_tests() -> dict[str, str]:
    files = sorted({p for g in TEST_GLOBS for p in (ROOT / "tests").glob(g)})
    return {p.name: p.read_text(encoding="utf-8") for p in files}


def test_every_third_party_module_a_shipped_test_imports_is_pinned():
    tests = shipped_tests()
    assert len(tests) >= 20, f"only {len(tests)} shipped tests found under {ROOT / 'tests'}"
    req = requirements_file()
    assert req.is_file(), req
    miss = missing_requirements(tests, req.read_text(encoding="utf-8"), local_modules())
    assert not miss, f"{req.name} does not pin what these shipped tests import: {miss}"


def test_the_check_sees_an_import_inside_a_function_and_a_missing_pin():
    """the defect R5 found, planted: a package imported only inside a test function, and not pinned"""
    planted = "import numpy\n\ndef test_x():\n    from qiskit_ibm_runtime.fake_provider import FakeKingston\n"
    local = local_modules()
    assert third_party_imports(planted, local) == {"numpy", "qiskit_ibm_runtime"}
    assert missing_requirements({"t.py": planted}, "numpy==2.3.5\n", local) == {"qiskit-ibm-runtime": ["t.py"]}
    assert missing_requirements({"t.py": planted}, "numpy==2.3.5\nqiskit-ibm-runtime==0.47.0  # x\n", local) == {}
    # a commented-out pin is not a pin, and the package named only in a comment block stays missing
    assert "qiskit-ibm-runtime" in missing_requirements({"t.py": planted}, "numpy\n#   qiskit-ibm-runtime==0.47.0\n",
                                                        local)


def test_the_requirements_name_the_two_hardware_tests_package():
    """the root fix itself: the pin is present and is the version the hardware submissions were made with"""
    text = requirements_file().read_text(encoding="utf-8")
    assert "qiskit-ibm-runtime==0.47.0" in [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
