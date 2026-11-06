"""Machine check required by the global rules: no source/report file may contain a 0x00 byte.

Usage: python check_nul.py
"""
import pathlib

bad = [p.name for p in pathlib.Path(__file__).parent.iterdir()
       if p.suffix in (".py", ".md", ".sh") and bytes([0]) in p.read_bytes()]
print("files with NUL bytes:", bad if bad else "none")
raise SystemExit(1 if bad else 0)
