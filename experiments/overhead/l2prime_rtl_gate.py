r"""L2' IN RTL: bit-exact against the Python detector, across languages.

C2's deliverable is a THREE-LAYER DETECTOR IN HARDWARE, not a simulation
of one. This closes the functional half of that for L2' -- the layer the
measurements showed C2 was missing -- by implementing it as synthesizable
Verilog and requiring it to agree with the Python implementation on every
vector.

WHY A CROSS-LANGUAGE CHECK IS WORTH ANYTHING. The RTL and the Python
compute the same predicate, and it would be trivial to write a testbench
that recomputed the expected value with the same expression the DUT uses
-- which would agree with itself and prove nothing. So the expected
`violation` bit is produced by the PYTHON detector and frozen into the
vector files; Verilog only ever reads it. Any disagreement is a real
divergence between two independent expressions of the same rule.

VECTORS ARE ADVERSARIAL, not uniform. Random syndromes alone would leave
the interesting cases to chance, so the generator forces:
  * shots where the two representatives agree and f-parity is 0
  * shots where they disagree and f-parity is 1 (the cancelling case,
    which a sign error gets exactly backwards)
  * all-zero and all-ones syndromes
  * masks with popcount 0, 1, and many
and the bench refuses a vector set in which the detector never fires, or
always fires, because either agrees trivially.

WHAT THIS DOES NOT CLAIM. C2 quotes ~5,400 LUTs, 0.26% of IBM's
2,106,738-LUT gross-code decoder. That is a SYNTHESIS claim and it is not
made here: yosys is not installed on this machine. What is supported is
the FUNCTION plus an analytic gate count -- an AND-reduce and an XOR tree
over popcount(F_MASK) bits, which is (d+1)^2/2 = 8/18/32/50 at
d=3/5/7/9, depth ceil(log2 popcount)+2, no state, no memory, no
multiplier. Anyone with a toolchain can close the gap; nobody should
quote a LUT number no synthesizer produced.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
RTL = ROOT / "rtl"
N_DET = 32


def py_violation(syn: int, a: int, b: int, fmask: int) -> int:
    """The Python detector -- the reference the RTL must match."""
    return (a ^ b ^ (bin(syn & fmask).count("1") & 1)) & 1


def gen_vectors(fmask: int, n: int, rng: random.Random):
    """Adversarial, not uniform: force the cancelling cases."""
    vecs = []
    forced = [
        (0, 0, 0), ((1 << N_DET) - 1, 0, 0),
        (0, 1, 0), (0, 1, 1), ((1 << N_DET) - 1, 1, 0),
    ]
    for (s, a, b) in forced:
        vecs.append((s, a, b))
    while len(vecs) < n:
        s = rng.getrandbits(N_DET)
        a = rng.getrandbits(1)
        # half the time, pick b so the three terms CANCEL -- the case a
        # sign error gets exactly backwards
        if rng.random() < 0.5:
            b = a ^ (bin(s & fmask).count("1") & 1)
        else:
            b = rng.getrandbits(1)
        vecs.append((s, a, b))
    return vecs


def run_case(fmask: int, nvec: int, seed: int, keep: pathlib.Path | None):
    rng = random.Random(seed)
    vecs = gen_vectors(fmask, nvec, rng)
    exp = [py_violation(s, a, b, fmask) for (s, a, b) in vecs]

    work = pathlib.Path(tempfile.mkdtemp(prefix="l2p_rtl_"))
    try:
        (work / "vec_syn.hex").write_text(
            "\n".join(f"{s:08x}" for (s, _a, _b) in vecs) + "\n")
        (work / "vec_a.bin").write_text(
            "\n".join(str(a) for (_s, a, _b) in vecs) + "\n")
        (work / "vec_b.bin").write_text(
            "\n".join(str(b) for (_s, _a, b) in vecs) + "\n")
        (work / "vec_exp.bin").write_text(
            "\n".join(str(e) for e in exp) + "\n")

        dut = (RTL / "l2prime_detector.v").read_text(encoding="utf-8")
        tb = (RTL / "tb_l2prime.v").read_text(encoding="utf-8")
        # the mask is a synthesis-time parameter, so the bench is
        # regenerated per case rather than made runtime-configurable
        tb = tb.replace(".F_MASK(32'hDEADBEEF)", f".F_MASK(32'h{fmask:08X})")
        (work / "dut.v").write_text(dut, encoding="utf-8")
        (work / "tb.v").write_text(tb, encoding="utf-8")

        c = subprocess.run(["iverilog", "-g2012", "-o", "sim",
                            "dut.v", "tb.v"], cwd=work,
                           capture_output=True, text=True, timeout=300)
        if c.returncode != 0:
            return {"fmask": fmask, "compiled": False,
                    "error": (c.stderr or c.stdout)[:400]}
        r = subprocess.run(["vvp", "sim"], cwd=work,
                           capture_output=True, text=True, timeout=600)
        out = r.stdout or ""
        m = re.search(r"VECTORS (\d+)\s+FIRED (\d+)\s+ERRORS (\d+)", out)
        if not m:
            return {"fmask": fmask, "compiled": True, "parsed": False,
                    "raw": out[:400]}
        n, fired, errors = (int(m.group(1)), int(m.group(2)),
                            int(m.group(3)))
        return {"fmask": f"0x{fmask:08X}",
                "mask_popcount": bin(fmask).count("1"),
                "compiled": True, "vectors": n, "fired": fired,
                "errors": errors,
                "vacuous": (fired == 0 or fired == n),
                "mismatch_lines": [ln for ln in out.splitlines()
                                   if "!=" in ln][:5]}
    finally:
        if keep:
            shutil.copytree(work, keep, dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=int, default=4000)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if not shutil.which("iverilog"):
        print("iverilog not installed -- REFUSING to report an RTL result")
        return 1

    # masks chosen to span the degenerate ends as well as realistic
    # popcounts: 0 (the functional is empty), 1, and the (d+1)^2/2
    # weights the solver actually produces at d=3,5,7
    masks = [0x00000000, 0x00000001, 0x000000FF, 0x0003FFFF,
             0xFFFFFFFF, 0xA5A5A5A5, 0xDEADBEEF]
    rows, bad = [], []
    print("L2' RTL vs the Python detector, bit-exact, per vector")
    for i, fm in enumerate(masks):
        r = run_case(fm, a.vectors, 1000 + i, None)
        rows.append(r)
        if not r.get("compiled"):
            print(f"  mask 0x{fm:08X}: COMPILE FAILED")
            bad.append(r)
            continue
        print(f"  mask 0x{fm:08X} pop {r['mask_popcount']:2d}  "
              f"vectors {r['vectors']:5d}  fired {r['fired']:5d}  "
              f"errors {r['errors']}"
              + ("   VACUOUS" if r["vacuous"] else ""))
        if r["errors"] or (r["vacuous"] and r["mask_popcount"] not in (0,)):
            bad.append(r)

    # popcount 0 is legitimately degenerate: with an empty functional the
    # predicate is pred_a ^ pred_b and half the vectors fire, so it is
    # not vacuous -- but a mask of all ones with nothing firing would be.
    total_err = sum(r.get("errors", 1) for r in rows)
    out = {"schema": "oneq-l2prime-rtl/1",
           "claim": ("the L2' detector in synthesizable Verilog agrees "
                     "bit-exactly with the Python implementation on every "
                     "vector, across a language boundary"),
           "simulator": "Icarus Verilog 13.0",
           "vectors_per_mask": a.vectors,
           "masks": rows,
           "total_errors": total_err,
           "cost_analytic": {
               "logic": "AND-reduce over N_DET plus an XOR tree over "
                        "popcount(F_MASK), then two XORs",
               "xor_tree_width_by_d": {"d=3": 8, "d=5": 18, "d=7": 32,
                                       "d=9": 50},
               "formula": "popcount(F_MASK) = (d+1)^2/2",
               "depth": "ceil(log2(popcount)) + 2",
               "state": "none beyond the output register",
               "NOT_CLAIMED": ("C2 quotes ~5,400 LUTs and 0.26% of IBM's "
                               "2,106,738-LUT decoder. That is a SYNTHESIS "
                               "claim; yosys is not installed here, so no "
                               "LUT count is reported. The function is "
                               "verified; the area is not.")},
           "passed": total_err == 0 and not bad}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "l2prime_rtl.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    if total_err or bad:
        print(f"\nRTL GATE FAILED: {total_err} vector mismatches")
        for r in bad[:3]:
            for ln in r.get("mismatch_lines", [])[:3]:
                print(f"  {ln}")
        return 1
    print(f"\nRTL AGREES: {len(masks)} masks x {a.vectors} vectors, "
          f"0 mismatches against the Python detector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
