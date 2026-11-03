"""Write a hand-crafted CDP3 proof (the format of cdist_prove.c / ccheck.c lines 197-236).

  python craft.py out.bin W NORB NANCH CODE
    CODE 1: per anchor, root code 1 with a claimed child count of 0 -- what a lying or buggy prover
            can emit for an anchor whose syndrome is zero (ccheck.c HEAD lines 221-225: pick_c
            returns -1, so nch = 0 and nothing is checked).               [finding F1]
    CODE 2: per anchor (needs W <= 3), root code 2 with an EMPTY bitstream: every node at depth 1
            has remaining budget W - 1 <= 2, so the checker decides it as a leaf.   [finding F5]
maxround is -1 (full scope). Bytes are built from numeric values only."""
import struct
import sys

out, w, norb, nanch, code = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
data = bytes([67, 68, 80, 51])                       # "CDP3"
data += struct.pack("<iiii", w, norb, -1, nanch)     # w, NORB, maxround = -1, nanch
for _ in range(nanch):
    if code == 1:
        data += bytes([1]) + struct.pack("<i", 0)    # root code 1, claimed 0 children
    elif code == 2:
        data += bytes([2]) + struct.pack("<q", 0)    # root code 2, one stream of 0 bits
    else:
        raise SystemExit("CODE must be 1 or 2")
with open(out, "wb") as f:
    f.write(data)
print(out, len(data), "bytes")
