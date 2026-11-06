"""The server's own schedule of each circuit AS IT EXECUTED: SamplerV2's experimental `scheduler_timing`.

Why it exists (R165, 2026-10-17). Under the runtime's default twirling, an idle qubit beside a CZ ladder flips at
6.6e-4 per layer, 2.65x the one-pulse estimate. What the server actually puts on that qubit (an x for each X or Y
Pauli, or a fixed rz.sx.rz.sx.rz dressing in every layer; how long each layer becomes) is not documented, and two
physical readings fit the shots about equally. The compiled schedule names every pulse and its time, so it decides
the question. It is free (metadata), so every job of this pipeline requests it (oneq.hw.submit.Plan.scheduler_timing).

The format (qiskit_ibm_runtime.utils.circuit_schedule, which this reader matches): one string, one instruction per
line, six comma-separated fields

    Branch, Instruction, Channel, T0, Duration, Pulse        e.g.  main,sx_3,Qubit 3,120,8,play

with T0 and Duration in scheduling cycles (dt). The gate name is the Instruction up to its first underscore. A
physical pulse is Pulse == "play"; "shift_phase" is a virtual rz; "capture" is a readout acquisition. Lines whose
first field contains shift_phase are skipped, as IBM's own parser skips them.

Every reader here is pure and refuses malformed input by name (TimingRefused): a schedule that parses wrongly would
silently miscount pulses, which is the one number this module exists to get right."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

HEADER = ("branch", "instruction", "channel", "t0", "duration", "pulse")
QUBIT_CHANNEL = re.compile(r"^Qubit (\d+)$")
PHYSICAL = "play"
COMPILED_GATES = ("sx", "x", "cz", "measure")       # the ISA gates that cost a physical pulse or a read


class TimingRefused(ValueError):
    """the schedule cannot be read faithfully"""


@dataclass(frozen=True)
class Row:
    branch: str
    instruction: str
    gate: str
    channel: str
    qubit: int | None          # from a "Qubit q" channel; None for readout (AWGR...), Hub and Receive channels
    t0: int                    # dt
    duration: int              # dt
    pulse: str

    @property
    def end(self) -> int:
        return self.t0 + self.duration


def _lines(timing) -> list:
    if isinstance(timing, str):
        return timing.split("\n")
    if isinstance(timing, (list, tuple)):
        out = []
        for i, part in enumerate(timing):
            if not isinstance(part, str):
                raise TimingRefused(f"timing entry {i} is {type(part).__name__}, not str")
            out.extend(part.split("\n"))
        return out
    raise TimingRefused(f"timing must be a str or a list of str, got {type(timing).__name__}")


def parse(timing) -> list:
    """the schedule as Rows, in the order given. A header line is accepted only as the first non-empty line"""
    rows: list = []
    seen_content = False
    for n, raw in enumerate(_lines(timing)):
        line = raw.strip()
        if not line:
            continue
        words = [w.strip() for w in line.split(",")]
        if "shift_phase" in words[0]:
            continue
        if len(words) != 6:
            raise TimingRefused(f"line {n}: {len(words)} fields, expected 6 "
                                f"(Branch, Instruction, Channel, T0, Duration, Pulse): {line[:80]!r}")
        if tuple(w.lower() for w in words) == HEADER:
            if seen_content:
                raise TimingRefused(f"line {n}: a header after the first instruction")
            seen_content = True
            continue
        seen_content = True
        branch, instruction, channel, t0s, durs, pulse = words
        try:
            t0, dur = int(t0s), int(durs)
        except ValueError:
            raise TimingRefused(f"line {n}: T0 {t0s!r} / Duration {durs!r} are not integers") from None
        if t0 < 0 or dur < 0:
            raise TimingRefused(f"line {n}: negative time (T0 {t0}, Duration {dur})")
        if not instruction:
            raise TimingRefused(f"line {n}: empty instruction")
        m = QUBIT_CHANNEL.match(channel)
        rows.append(Row(branch, instruction, instruction.split("_")[0], channel, int(m.group(1)) if m else None,
                        t0, dur, pulse))
    if not rows:
        raise TimingRefused("the schedule holds no instructions")
    return rows


def from_metadata(pub_metadata) -> list | None:
    """the Rows from one PUB's result metadata (metadata["compilation"]["scheduler_timing"]["timing"]); None when
    the job did not ask for it (the runtime then leaves scheduler_timing null), refused when it is there but
    malformed"""
    comp = (pub_metadata or {}).get("compilation") or {}
    st = comp.get("scheduler_timing")
    if st is None:
        return None
    if not isinstance(st, dict) or "timing" not in st:
        raise TimingRefused(f"scheduler_timing has no 'timing' field: {str(st)[:80]}")
    return parse(st["timing"])


def per_qubit(rows: list, branch: str = "main") -> dict:
    """{qubit: its drive-channel Rows in time order}, for one branch"""
    out: dict = {}
    for r in rows:
        if r.qubit is not None and r.branch == branch:
            out.setdefault(r.qubit, []).append(r)
    for q in out:
        out[q].sort(key=lambda r: (r.t0, r.instruction))
    return out


def pulse_counts(rows: list, branch: str = "main") -> dict:
    """{qubit: Counter(gate -> physical pulses played)}"""
    return {q: Counter(r.gate for r in rs if r.pulse == PHYSICAL) for q, rs in per_qubit(rows, branch).items()}


def duration_dt(rows: list) -> int:
    """the schedule's end, in dt"""
    return max(r.end for r in rows)


def compiled_counts(qc) -> dict:
    """{physical qubit: Counter(gate -> count)} of the ISA circuit as submitted, for the gates that cost a pulse
    or a read (rz is virtual, barriers and delays are not instructions)"""
    out: dict = {}
    for inst in qc.data:
        name = inst.operation.name
        if name not in COMPILED_GATES:
            continue
        for b in inst.qubits:
            out.setdefault(qc.find_bit(b).index, Counter())[name] += 1
    return out


def compare(rows: list, qc, branch: str = "main") -> dict:
    """per qubit, what was submitted against what was played: {q: dict(compiled, executed, extra)} where extra is
    executed minus compiled per single-qubit gate (sx, x): pulses the server added, e.g. twirl dressing on an idle
    qubit. Qubits the submitted circuit never uses but the server drove are included (compiled empty)"""
    exe = pulse_counts(rows, branch)
    com = compiled_counts(qc)
    out = {}
    for q in sorted(set(exe) | set(com)):
        e, c = exe.get(q, Counter()), com.get(q, Counter())
        out[q] = dict(compiled=dict(c), executed=dict(e), extra={g: e.get(g, 0) - c.get(g, 0) for g in ("sx", "x")
                                                                 if e.get(g, 0) != c.get(g, 0)})
    return out


def summary(rows: list, qc=None, dt: float | None = None, branch: str = "main") -> dict:
    """a JSON-able account of one PUB's schedule: its duration (dt, and seconds when dt is given), the pulses played
    per qubit, and (with the submitted circuit) the pulses the server added"""
    total_dt = duration_dt(rows)
    out: dict[str, object] = dict(duration_dt=total_dt, n_rows=len(rows),
                                  pulses={str(q): dict(c) for q, c in pulse_counts(rows, branch).items()})
    if dt is not None:
        out["duration_s"] = total_dt * dt
    if qc is not None:
        out["vs_compiled"] = {str(q): v for q, v in compare(rows, qc, branch).items()}
    return out
