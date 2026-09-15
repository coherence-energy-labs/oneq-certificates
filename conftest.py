"""Make `pytest` work from the repository root without PYTHONPATH=src.

An external audit found that bare `pytest -q` fails collection here, and that
CI hid it by running `pip install -e .` first. A test suite that only runs
under a flag the README does not mention is a suite most contributors will
believe is broken.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
