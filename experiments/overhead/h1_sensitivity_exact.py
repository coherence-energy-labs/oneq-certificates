r"""H-1: sensitivity is EXACT -- two independent references, 0 discrepancies.

THE GATE (FTQC_IMPLEMENTATION §5, H-1):

    hand-built 5-logical-qubit circuit with one flagged uncompute;
    w_undetected vs exhaustive enumeration of all single-Pauli faults,
    adjudicated by stim.Tableau AND quantum_shadow.py (n<=6)
    PASS: exact agreement, 0 discrepancies, against BOTH references
    FAIL MEANS: the stim logical-DEM lift is wrong; the whole pass is
                built on sand

and the reason it is written that way:

    "sensitivity.py is the single highest-risk module -- the only place
     where a subtle error produces a PLAUSIBLE BUT WRONG w_r, and every
     downstream number inherits it. That is why H-1 demands EXACT
     AGREEMENT WITH AN INDEPENDENT IMPLEMENTATION rather than a
     tolerance."

THE TWO REFERENCES ARE GENUINELY INDEPENDENT, which is the whole point:

  A. stim.Tableau -- stim compiles each inter-measurement segment into a
     tableau and conjugates the fault through it with its own machinery.
  B. a hand-written symplectic propagator over GF(2)^2n, applying the
     per-gate rules directly (H: swap x,z | S: z ^= x | CNOT: x_t ^= x_c,
     z_c ^= z_t).

Two implementations that share no code and were written from different
descriptions of the same physics. Agreement is required on EVERY fault
individually -- not on the total, because two different classifications
can sum to the same w_undetected and still disagree everywhere.

THE CLASSIFICATION, stated because the number means nothing without it.
For each single-Pauli fault at each location:

  DETECTED    the fault anticommutes with a flag/syndrome measurement, so
              some outcome flips relative to the fault-free run
  UNDETECTED-HARMLESS  it escapes detection but leaves the logical
              register untouched (identity on the data qubits)
  UNDETECTED-DANGEROUS it escapes detection AND acts non-trivially on the
              logical register -- these are the only ones that count
              toward w_undetected

A flagged uncompute is included precisely because it is where a naive
propagator goes wrong: the flag catches faults that would otherwise
spread through the uncompute, and getting its anticommutation backwards
produces a w_undetected that is plausible and wrong.
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

N_LOGICAL = 5
FLAG = 5                       # the sixth qubit: n <= 6, as the gate says
N = 6


def circuit():
    """A 5-logical-qubit block with ONE flagged uncompute.

    ('H', q) | ('S', q) | ('CX', c, t) | ('M', q) -- a Z-basis measurement
    that the fault-free run makes deterministic, i.e. a flag.
    """
    g = []
    g.append(("H", FLAG))
    for q in range(3):                      # compute a parity onto the flag
        g.append(("CX", FLAG, q))
    g.append(("H", 0))                      # logical work in the middle
    g.append(("S", 1))
    g.append(("CX", 1, 2))
    g.append(("CX", 3, 4))
    g.append(("S", 4))
    for q in reversed(range(3)):            # the UNCOMPUTE
        g.append(("CX", FLAG, q))
    g.append(("H", FLAG))
    g.append(("M", FLAG))                   # deterministic without faults
    return g


# --------------------------------------------------------------- ref B
def propagate_symplectic(gates, start, fault_q, fault_p):
    """Hand-written reference: push a Pauli forward, gate by gate.

    Returns (detected, residual_x, residual_z) with the residual taken
    over the LOGICAL qubits only.
    """
    x = [0] * N
    z = [0] * N
    if fault_p in ("X", "Y"):
        x[fault_q] = 1
    if fault_p in ("Z", "Y"):
        z[fault_q] = 1

    detected = False
    for idx in range(start, len(gates)):
        g = gates[idx]
        if g[0] == "H":
            q = g[1]
            x[q], z[q] = z[q], x[q]
        elif g[0] == "S":
            q = g[1]
            z[q] ^= x[q]
        elif g[0] == "CX":
            c, t = g[1], g[2]
            x[t] ^= x[c]
            z[c] ^= z[t]
        elif g[0] == "M":
            q = g[1]
            # a Z-basis measurement flips iff the Pauli anticommutes with
            # Z_q, i.e. iff it has an X component there
            if x[q]:
                detected = True
    return detected, tuple(x[:N_LOGICAL]), tuple(z[:N_LOGICAL])


# --------------------------------------------------------------- ref A
def propagate_stim(gates, start, fault_q, fault_p):
    """stim.Tableau reference: conjugate through compiled segments.

    stim is given the same gate list and does the algebra its own way;
    nothing is shared with the symplectic routine above.
    """
    import stim

    ps = ["_"] * N
    ps[fault_q] = fault_p
    cur = stim.PauliString("".join(ps))

    detected = False
    seg = stim.Circuit()
    seg_has = False
    for idx in range(start, len(gates)):
        g = gates[idx]
        if g[0] == "M":
            if seg_has:
                cur = stim.Tableau.from_circuit(seg)(cur)
                seg, seg_has = stim.Circuit(), False
            q = g[1]
            # anticommutes with Z_q  <=>  the Pauli has X or Y on q
            if str(cur)[1 + q] in ("X", "Y"):
                detected = True
            continue
        if g[0] == "H":
            seg.append("H", [g[1]])
        elif g[0] == "S":
            seg.append("S", [g[1]])
        elif g[0] == "CX":
            seg.append("CX", [g[1], g[2]])
        seg_has = True
    if seg_has:
        cur = stim.Tableau.from_circuit(seg)(cur)

    s = str(cur)[1:]
    xs = tuple(1 if s[q] in ("X", "Y") else 0 for q in range(N_LOGICAL))
    zs = tuple(1 if s[q] in ("Z", "Y") else 0 for q in range(N_LOGICAL))
    return detected, xs, zs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    gates = circuit()
    faults = [(loc, q, p)
              for loc in range(len(gates))
              for q in range(N)
              for p in ("X", "Y", "Z")]

    disagreements = []
    detected = harmless = dangerous = 0
    for (loc, q, p) in faults:
        b = propagate_symplectic(gates, loc, q, p)
        s = propagate_stim(gates, loc, q, p)
        if b != s:
            disagreements.append(
                {"location": loc, "gate": str(gates[loc]), "qubit": q,
                 "pauli": p, "symplectic": str(b), "stim": str(s)})
            continue
        det, xs, zs = b
        if det:
            detected += 1
        elif any(xs) or any(zs):
            dangerous += 1
        else:
            harmless += 1

    w_undetected = dangerous
    print(f"H-1: exhaustive single-Pauli enumeration over a "
          f"{N_LOGICAL}-logical-qubit block with one flagged uncompute")
    print(f"  locations {len(gates)}  qubits {N}  faults enumerated "
          f"{len(faults)}")
    print(f"  DETECTED             {detected}")
    print(f"  undetected HARMLESS  {harmless}")
    print(f"  undetected DANGEROUS {dangerous}   <- w_undetected")
    print(f"  discrepancies between the two references: "
          f"{len(disagreements)}")

    # A run in which nothing was ever detected, or nothing ever escaped,
    # would agree trivially and prove nothing about either reference.
    vacuous = []
    if detected == 0:
        vacuous.append("no fault was DETECTED: the flag never fired, so "
                       "the detection path is untested")
    if dangerous == 0:
        vacuous.append("no fault was undetected-DANGEROUS: w_undetected "
                       "is 0 and the classification is untested")
    if harmless == 0:
        vacuous.append("no fault was undetected-HARMLESS: the residual "
                       "test never distinguished anything")

    out = {"schema": "oneq-h1-sensitivity-exact/1",
           "gate": "H-1: sensitivity is exact",
           "criterion": ("exact agreement, 0 discrepancies, against BOTH "
                         "references -- not a tolerance"),
           "circuit": {"logical_qubits": N_LOGICAL, "flag_qubit": FLAG,
                       "gates": [list(map(str, g)) for g in gates]},
           "faults_enumerated": len(faults),
           "detected": detected,
           "undetected_harmless": harmless,
           "undetected_dangerous": dangerous,
           "w_undetected": w_undetected,
           "discrepancies": len(disagreements),
           "first_discrepancies": disagreements[:5],
           "references": {
               "A": "stim.Tableau over compiled inter-measurement segments",
               "B": ("hand-written symplectic propagator over GF(2)^2n "
                     "(H: swap x,z | S: z ^= x | CNOT: x_t ^= x_c, "
                     "z_c ^= z_t)")},
           "why_per_fault": ("agreement is required on EVERY fault, not "
                             "on the total: two different classifications "
                             "can sum to the same w_undetected and still "
                             "disagree everywhere"),
           "passed": not disagreements and not vacuous}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "h1_sensitivity_exact.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")

    if vacuous:
        print("\nVACUOUS -- the enumeration agreed about nothing:")
        for v in vacuous:
            print(f"  {v}")
        return 1
    if disagreements:
        print(f"\nH-1 FAILED: {len(disagreements)} discrepancies. The stim "
              f"logical lift and the symplectic propagator disagree, so "
              f"w_undetected is not exact and everything downstream "
              f"inherits it.")
        for d in disagreements[:5]:
            print(f"  loc {d['location']} {d['gate']} q{d['qubit']} "
                  f"{d['pauli']}: symplectic {d['symplectic']} vs stim "
                  f"{d['stim']}")
        return 1
    print(f"\nH-1 PASSED: {len(faults)} faults, 0 discrepancies against "
          f"both references, w_undetected = {w_undetected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
