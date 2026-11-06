"""Frame correction: cancel each qubit's COHERENT fixed detuning with virtual rz -- free on IBM hardware.

A qubit whose frequency sits Delta off its drive frame turns by rz(Delta t) over every t it spends in superposition
(DeviceSnapshot.detuning; R161 measured q149 on ibm_kingston at +49.7 kHz with contrast 0.984). The rotation commutes
with rz, cz and idles and is inverted by an x, so over a circuit it collects into one rz(Delta T) per PHASE WINDOW,
closing before the sx that ends the window (oneq.hw.noise.phase_windows -- the same function the engine charges it
with). correct() inserts the inverse, rz(-Delta T), immediately before that sx. rz is virtual on IBM hardware (a frame
change in the control software), so the correction costs no time and no error.

The correction is computed on a SCHEDULE, and only holds for that timing; correct() therefore returns the scheduled
circuit itself, with every wait an explicit delay, so the timing it was computed for is the timing that runs. It
refuses a circuit that is not in the ISA basis or that measures mid-circuit (the engine's own refusals).

  correct(qc, snap, detuning=None, schedule="alap") -> (corrected circuit, report)
"""
from __future__ import annotations

import math

from .noise import ISA, ScheduleError, durations, phase_windows
from .snapshot import DeviceSnapshot, SnapshotError


class FrameRefused(ValueError):
    """the circuit or the detuning map cannot be corrected faithfully"""


def schedule(qc, snap: DeviceSnapshot, how: str = "alap"):
    """the circuit padded with explicit delays from the snapshot's durations (ALAP, as ibm_kingston runs: R146)"""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, ASAPScheduleAnalysis, PadDelay
    if how not in ("alap", "asap"):
        raise FrameRefused(f"unknown schedule {how!r}")
    analysis = ALAPScheduleAnalysis if how == "alap" else ASAPScheduleAnalysis
    return PassManager([analysis(durations(snap)), PadDelay(durations=durations(snap))]).run(qc)


def correct(qc, snap: DeviceSnapshot, detuning: dict | None = None, how: str = "alap") -> tuple:
    """(the scheduled circuit with virtual rz(-Delta T) before every window-closing sx, report). detuning defaults to
    the snapshot's own map; a qubit the map lacks is left alone."""
    from qiskit.circuit.library import RZGate
    det = dict(snap.detuning if detuning is None else detuning)
    for q, v in det.items():
        if not math.isfinite(v):
            raise FrameRefused(f"detuning of qubit {q} is not finite: {v}")
    seen_measure: set = set()
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(x).index for x in inst.qubits]
        if name not in ISA:
            raise FrameRefused(f"gate {name} is not in the ISA basis (transpile first)")
        if name == "cz":                               # an uncalibrated edge has no duration to schedule it by
            try:
                snap.edge(*qs)
            except SnapshotError as e:
                raise FrameRefused(f"cz on {qs}: {e}") from e
        if name == "measure":
            seen_measure.update(qs)
        elif name not in ("barrier", "delay") and seen_measure & set(qs):
            raise FrameRefused(f"{name} after a measurement on {qs}: a mid-circuit read is not corrected here")
    from qiskit.transpiler.exceptions import TranspilerError
    try:
        sched = schedule(qc, snap, how)
    except (ScheduleError, TranspilerError) as e:   # qiskit's scheduler raises its own error, not ours
        raise FrameRefused(str(e)) from e
    used = {sched.find_bit(x).index for inst in sched.data for x in inst.qubits}
    at = phase_windows(sched, snap, set(det) & used, close_at_cz=False)
    out = sched.copy_empty_like()
    inserted = []
    for k, inst in enumerate(sched.data):
        for q, T in at.get(k, ()):
            theta = -det[q] * T
            if theta:
                out.append(RZGate(theta), [out.qubits[q]])
                inserted.append(dict(qubit=q, before_instruction=k, window_s=T, rz=theta))
        out.append(inst.operation, inst.qubits, inst.clbits)
    total: dict = {}
    for row in inserted:
        total[row["qubit"]] = total.get(row["qubit"], 0.0) + abs(row["rz"])
    return out, dict(schedule=how, corrections=len(inserted), inserted=inserted,
                     total_rotation_cancelled_rad={str(q): v for q, v in sorted(total.items())})
