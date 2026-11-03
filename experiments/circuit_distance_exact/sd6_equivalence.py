r"""Is the lane's fault placement equivalent to SD6 on the gross-code circuits? Compare the mechanism sets.

    python sd6_equivalence.py > ../../evidence/circuit_distance_exact/records/sd6_equivalence.jsonl   # the committed record

WHY. Paper 3's distances are properties of the lane's fault placement (Sec. II C of the paper), which differs from
the SD6 model of Gidney, Newman, Fowler and Broughton (the standard depolarizing model used by the concurrent results
the paper compares with): IBM's cycle here has noiseless data initialization and idle noise only in CNOT layers; the
three-colour cycle faults no idle ancilla. If the two placements give the SAME set of distinct (detectors, observables)
symptoms, the distance is the same object under both, and the comparison is exact; if not, the differing mechanisms
are listed and the comparison is only numerical (paper 3 referee, 2026-11-02).

SD6, as built here from the lane's own instruction stream with every noise line removed, moment by moment (moments are
the TICK-separated blocks): DEPOLARIZE2 after every two-qubit gate; DEPOLARIZE1 after every single-qubit gate; a flip
after every reset (X_ERROR after R, Z_ERROR after RX); a flip before every measurement (X_ERROR before M, Z_ERROR
before MX); DEPOLARIZE1 on every qubit the moment does not touch. Annotation instructions are kept; any other
instruction is refused by name.

Each comparison is recorded for the full DEM and for the same-type projection the search runs on. Exit 1 if any
compared pair differs (a difference is a finding to report, not a crash).
"""
from __future__ import annotations

import argparse
import json
import sys

import stim

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402

NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"}
ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}
ONE_Q = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "X", "Y", "Z", "I"}
TWO_Q = {"CX", "CZ", "CY", "SWAP"}


def sd6(noisy: stim.Circuit, p: float = 1e-3) -> stim.Circuit:
    """The same gates, measurements and annotations, with SD6 noise in place of the lane's."""
    flat = noisy.flattened()
    nq = noisy.num_qubits
    out = stim.Circuit()
    moment: list[stim.CircuitInstruction] = []

    def flush(end_with_tick: bool):
        touched: set[int] = set()
        for inst in moment:
            name, tg = inst.name, [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if name in ANNOTATIONS:
                out.append(inst)
                continue
            if name in ("M", "MX"):
                out.append("X_ERROR" if name == "M" else "Z_ERROR", tg, p)
                out.append(inst)
            elif name in ("R", "RX"):
                out.append(inst)
                out.append("X_ERROR" if name == "R" else "Z_ERROR", tg, p)
            elif name in ONE_Q:
                out.append(inst)
                out.append("DEPOLARIZE1", tg, p)
            elif name in TWO_Q:
                out.append(inst)
                out.append("DEPOLARIZE2", tg, p)
            else:
                raise ValueError(f"instruction {name} is not handled by the SD6 transform")
            touched.update(tg)
        gates = [i for i in moment if i.name not in ANNOTATIONS]
        if gates:
            idle = [q for q in range(nq) if q not in touched]
            if idle:
                out.append("DEPOLARIZE1", idle, p)
        if end_with_tick:
            out.append("TICK")

    for inst in flat:
        if inst.name in NOISE:
            continue
        if inst.name == "TICK":
            flush(True)
            moment = []
            continue
        moment.append(inst)
    flush(False)
    return out


def mech_sets(circuit: stim.Circuit, basis: str):
    _dem, mech, coords = em.mechanisms(circuit)
    proj, _ = em.project_same_type(mech, coords, basis)
    return set(mech), set(proj)


def drop_noise_line(c: stim.Circuit, name: str, k: int) -> stim.Circuit:
    """A copy of `c` with its k-th `name` line removed (the control: the comparison must see it)."""
    lines = str(c).splitlines()
    at = [i for i, s in enumerate(lines) if s.startswith(name)][k]
    return stim.Circuit("\n".join(lines[:at] + lines[at + 1:]))


def noise_targets(c: stim.Circuit) -> int:
    return sum(len(i.targets_copy()) for i in c.flattened() if i.name in NOISE)


def compare(kind: str, basis: str, rounds: int) -> dict:
    ours = bc.build(kind, basis, rounds)
    s6 = sd6(ours)
    full_a, proj_a = mech_sets(ours, basis)
    full_b, proj_b = mech_sets(s6, basis)
    # CONTROL: SD6 with one two-qubit noise line removed, at the first position that removes a mechanism, must differ
    # from ours -- an equality the comparison could not break would be dark.
    ctl = None
    for k in range(sum(1 for i in s6.flattened() if i.name == "DEPOLARIZE2")):
        full_c, _ = mech_sets(drop_noise_line(s6, "DEPOLARIZE2", k), basis)
        if full_c != full_a:
            ctl = k
            break
    if ctl is None:
        # at T = 1 no single two-qubit noise line is load-bearing (the fault set is redundant there), so the control
        # removes every two-qubit noise line; that must still change the set
        stripped = stim.Circuit("\n".join(s for s in str(s6).splitlines() if not s.startswith("DEPOLARIZE2")))
        if mech_sets(stripped, basis)[0] != full_a:
            ctl = "all"
    return dict(circuit=kind, basis=basis, rounds=rounds,
                noise_targets_ours=noise_targets(ours), noise_targets_sd6=noise_targets(s6),
                control_dropped_depolarize2_line=ctl, control_differs=ctl is not None,
                full_ours=len(full_a), full_sd6=len(full_b), full_equal=full_a == full_b,
                full_only_ours=len(full_a - full_b), full_only_sd6=len(full_b - full_a),
                proj_ours=len(proj_a), proj_sd6=len(proj_b), proj_equal=proj_a == proj_b,
                proj_only_ours=len(proj_a - proj_b), proj_only_sd6=len(proj_b - proj_a))


def main() -> int:
    ap = argparse.ArgumentParser()
    # the defaults regenerate the committed record, records/sd6_equivalence.jsonl (second referee of 2026-11-02)
    ap.add_argument("--circuits", nargs="+", default=["ibm", "3c", "morph72", "morph144"])
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()
    differ = 0
    for kind in a.circuits:
        for basis in "ZX":
            for T in a.rounds:
                row = compare(kind, basis, T)
                differ += not (row["full_equal"] and row["proj_equal"] and row["control_differs"])
                print(json.dumps(row), flush=True)
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
